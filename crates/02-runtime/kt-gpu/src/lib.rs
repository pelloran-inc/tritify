//! kt-gpu — the GPU compute backend: portable WGSL compute shaders run through wgpu (Metal, Vulkan, DX12). The
//! kernels:
//! - `gemm_i64` is exact for any magnitude. It emulates 64-bit integers as two 32-bit words, so it runs on any shader
//!   device. It has two forms: one thread per output, and split-K for products with few outputs (decode), where one
//!   workgroup shares each dot product. Both give the same `i64`, because integer addition is associative.
//! - `gemm_i32_tiled` is the fast 16×16 shared-memory kernel. It is exact only while every sum fits `i32`, so the
//!   engine never uses it; the demo shows it.
//!
//! [`GpuLinear`] is the engine's linear accumulate on the GPU. By default it keeps every weight it has seen resident on
//! the device: uploaded once, under a byte budget, keyed by the engine's `WeightId`. No float anywhere: operands,
//! accumulators and results are integers. A machine without a GPU adapter gets `None` from [`Gpu::open`] and runs on
//! the CPU.
//!
//! # Example
//!
//! ```
//! // Y = X · Wᵀ on the GPU when there is one; the CPU answer is the same integers either way.
//! let (x, w) = (vec![1, 2, 3, 4, 5, 6], vec![1, 0, -1, 2, 2, 2]);
//! if let Some(gpu) = kt_gpu::Gpu::open(None) {
//!     assert_eq!(gpu.gemm_i64(&x, &w, 2, 2, 3), vec![-2, 12, -2, 30]);
//!     assert_eq!(gpu.gemm_i64_with(&x, &w, 2, 2, 3, kt_gpu::Kernel::SplitK), vec![-2, 12, -2, 30]);
//! }
//! ```

mod dispatch;
mod resident;

#[cfg(feature = "plants")]
pub use resident::Plant;
pub use resident::{GpuLinear, KernelChoice};

use pollster::block_on;
use std::collections::HashMap;
use std::sync::Mutex;

const INT64: &str = include_str!("../shaders/int64.wgsl");
const WGSL_TILED: &str = include_str!("../shaders/trit_matmul_tiled.wgsl");
const WGSL_I64: &str = include_str!("../shaders/trit_matmul_i64.wgsl");
const WGSL_SPLITK: &str = include_str!("../shaders/trit_matmul_i64_splitk.wgsl");

/// Which exact `i64` kernel computes a product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// One thread per output runs the whole dot product: best for many rows.
    OneThread,
    /// One workgroup per output shares the dot product: best when there are few outputs in all.
    SplitK,
}

/// One GPU adapter as the host reports it.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    /// The graphics API: `Metal`, `Vulkan`, `Dx12`, `Gl`.
    pub backend: String,
    /// `DiscreteGpu`, `IntegratedGpu`, `VirtualGpu`, `Cpu` (a software rasterizer), `Other`.
    pub device_type: String,
}

/// Every adapter the native graphics APIs expose on this host.
pub fn adapters() -> Vec<GpuInfo> {
    let inst = wgpu::Instance::default();
    inst.enumerate_adapters(wgpu::Backends::all()).iter().map(|a| info_of(&a.get_info())).collect()
}

fn info_of(i: &wgpu::AdapterInfo) -> GpuInfo {
    GpuInfo { name: i.name.clone(), backend: format!("{:?}", i.backend), device_type: format!("{:?}", i.device_type) }
}

/// An open GPU device with its kernels compiled.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: GpuInfo,
    max_binding: u64,
    layout: wgpu::BindGroupLayout,
    i64_pipe: wgpu::ComputePipeline,
    splitk_pipe: wgpu::ComputePipeline,
    tiled_pipe: wgpu::ComputePipeline,
    /// Output + readback buffers by byte size, reused across calls.
    pool: Mutex<HashMap<u64, Vec<(wgpu::Buffer, wgpu::Buffer)>>>,
}

impl Gpu {
    /// Open adapter `index` from [`adapters`], or (`None`) the high-performance default. `None` if there is none.
    pub fn open(index: Option<usize>) -> Option<Gpu> {
        let inst = wgpu::Instance::default();
        let adapter = match index {
            Some(i) => inst.enumerate_adapters(wgpu::Backends::all()).into_iter().nth(i)?,
            None => block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))?,
        };
        let info = info_of(&adapter.get_info());
        let limits = adapter.limits();
        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("kt-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;
        let layout = dispatch::bind_layout(&device);
        let i64_pipe = dispatch::pipeline(&device, &layout, &format!("{INT64}\n{WGSL_I64}"));
        let splitk_pipe = dispatch::pipeline(&device, &layout, &format!("{INT64}\n{WGSL_SPLITK}"));
        let tiled_pipe = dispatch::pipeline(&device, &layout, WGSL_TILED);
        let max_binding = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        let pool = Mutex::new(HashMap::new());
        Some(Gpu { device, queue, info, max_binding, layout, i64_pipe, splitk_pipe, tiled_pipe, pool })
    }

    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// Exact `Y[r,o] = Σ_k x[r,k]·w[o,k]` (`x` is `m × k`, `w` is `n × k`), the weights uploaded for this call, one
    /// thread per output. Large problems are split into blocks of outputs and rows so that no binding or dispatch
    /// exceeds the device limits; each output is still one full sum, so the split changes nothing.
    pub fn gemm_i64(&self, x: &[i32], w: &[i32], m: usize, n: usize, k: usize) -> Vec<i64> {
        self.gemm_i64_with(x, w, m, n, k, Kernel::OneThread)
    }

    /// [`Gpu::gemm_i64`] with a chosen kernel; the result is the same either way.
    pub fn gemm_i64_with(&self, x: &[i32], w: &[i32], m: usize, n: usize, k: usize, kernel: Kernel) -> Vec<i64> {
        check_shapes(x, w, m, n, k);
        let blocks = self.blocks_of_outputs(n, k);
        let srcs: Vec<dispatch::WSrc> =
            blocks.iter().map(|&(n0, nc)| dispatch::WSrc::Data(&w[n0 * k..(n0 + nc) * k])).collect();
        self.gemm_blocks(self.pipe_for(kernel), kernel, x, &blocks, &srcs, m, n, k)
    }

    /// The tiled `i32` kernel: `Y = X · Wᵀ` with an `i32` accumulator. Exact only while every partial sum fits `i32`;
    /// the caller must check that (the demo does). Not split into blocks: it serves demo-sized problems.
    pub fn gemm_i32_tiled(&self, x: &[i32], w: &[i32], m: usize, n: usize, k: usize) -> Vec<i32> {
        check_shapes(x, w, m, n, k);
        self.run_tiled(x, w, m, n, k)
    }

    fn pipe_for(&self, kernel: Kernel) -> &wgpu::ComputePipeline {
        match kernel {
            Kernel::OneThread => &self.i64_pipe,
            Kernel::SplitK => &self.splitk_pipe,
        }
    }

    /// Output blocks `(first, count)` such that a block of weights fits one storage binding.
    pub(crate) fn blocks_of_outputs(&self, n: usize, k: usize) -> Vec<(usize, usize)> {
        let per = (self.max_rows(k)).min(n).max(1);
        (0..n).step_by(per).map(|n0| (n0, per.min(n - n0))).collect()
    }

    /// How many `k`-long `i32` rows fit one storage binding.
    pub(crate) fn max_rows(&self, k: usize) -> usize {
        ((self.max_binding / (4 * k.max(1) as u64)) as usize).max(1)
    }
}

pub(crate) fn check_shapes(x: &[i32], w: &[i32], m: usize, n: usize, k: usize) {
    assert_eq!(x.len(), m * k, "kt-gpu: x is not m·k");
    assert_eq!(w.len(), n * k, "kt-gpu: w is not n·k");
}

#[cfg(test)]
mod tests;
