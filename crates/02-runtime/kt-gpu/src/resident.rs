//! [`GpuLinear`], the engine's linear accumulate on the GPU. A weight matrix the engine names by its `WeightId` is
//! uploaded once and kept on the device. When the resident bytes would exceed the budget, the least recently used
//! weights are dropped first. The id is only a cache key: it never reaches a result, which is the same integers whether
//! a weight was resident, re-uploaded or evicted in between.

use crate::dispatch::WSrc;
use crate::{Gpu, Kernel};
use kt_core::backend::{LinearBackend, WeightId, Weights};
use std::collections::HashMap;
use std::sync::Mutex;

/// Split-K serves products with few outputs in all (`rows · out ≤ SPLITK_MAX_OUTPUTS`) and at least `SPLITK_MIN_K`
/// terms per dot product. There, one thread per output would leave the GPU idle. With many outputs the one-thread
/// kernel already fills the GPU and wins. Measured on the M3 Max with the `gpu_bench` example:
/// - split-K is ahead up to 8 × 2048 = 16,384 outputs
/// - one-thread is ahead at 16 × 2048, and at Qwen's LM head (151,936 outputs)
const SPLITK_MAX_OUTPUTS: usize = 16_384;
const SPLITK_MIN_K: usize = 128;

/// Which kernel a [`GpuLinear`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelChoice {
    /// Split-K for few outputs, one thread per output otherwise (the measured rule above).
    Auto,
    Fixed(Kernel),
}

/// A planted defect (the `plants` feature only, for the firing controls; never in a product build).
#[cfg(feature = "plants")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plant {
    /// The weight buffer is allocated but never written.
    DropUpload,
    /// The cache is keyed by shape, so two layers of one shape share the first one's weights.
    CacheByShape,
    /// The split-K kernel drops the last thread's slice of every dot product.
    SplitKSkipLast,
}

/// A cache key: the weights' id and shape. A planted cache drops the id.
type Key = (Option<WeightId>, usize, usize);

struct Resident {
    blocks: Vec<(usize, usize)>,
    bufs: Vec<wgpu::Buffer>,
    bytes: u64,
    last_use: u64,
}

#[derive(Default)]
struct Cache {
    map: HashMap<Key, Resident>,
    bytes: u64,
    tick: u64,
}

/// The engine's linear accumulate on one GPU.
pub struct GpuLinear {
    gpu: Gpu,
    resident: bool,
    kernel: KernelChoice,
    budget: u64,
    cache: Mutex<Cache>,
    #[cfg(feature = "plants")]
    plant: Option<(Plant, Option<wgpu::ComputePipeline>)>,
}

impl GpuLinear {
    /// The default budget for resident weights: 8 GiB.
    pub const DEFAULT_BUDGET: u64 = 8 << 30;

    /// Weights resident under the default budget, the kernel chosen by shape.
    pub fn new(gpu: Gpu) -> GpuLinear {
        GpuLinear {
            gpu,
            resident: true,
            kernel: KernelChoice::Auto,
            budget: GpuLinear::DEFAULT_BUDGET,
            cache: Mutex::new(Cache::default()),
            #[cfg(feature = "plants")]
            plant: None,
        }
    }

    /// The original behaviour, kept for comparison: the weights are uploaded on every call, one thread per output.
    pub fn per_call(gpu: Gpu) -> GpuLinear {
        GpuLinear { resident: false, kernel: KernelChoice::Fixed(Kernel::OneThread), ..GpuLinear::new(gpu) }
    }

    /// Cap the resident bytes; least recently used weights are dropped first. A single matrix larger than the budget is
    /// still uploaded (alone).
    pub fn with_budget(mut self, bytes: u64) -> GpuLinear {
        self.budget = bytes;
        self
    }

    pub fn with_kernel(mut self, kernel: KernelChoice) -> GpuLinear {
        self.kernel = kernel;
        self
    }

    /// A backend with a planted defect.
    #[cfg(feature = "plants")]
    pub fn planted(gpu: Gpu, plant: Plant) -> GpuLinear {
        let pipe = (plant == Plant::SplitKSkipLast).then(|| {
            let src = include_str!("../shaders/trit_matmul_i64_splitk.wgsl")
                .replace("// PLANT-HOOK", "if (lid.x == 63u) { acc = vec2<u32>(0u, 0u); }");
            crate::dispatch::pipeline(&gpu.device, &gpu.layout, &format!("{}\n{src}", crate::INT64))
        });
        let kernel =
            if plant == Plant::SplitKSkipLast { KernelChoice::Fixed(Kernel::SplitK) } else { KernelChoice::Auto };
        GpuLinear { kernel, plant: Some((plant, pipe)), ..GpuLinear::new(gpu) }
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The bytes of weights resident right now.
    pub fn resident_bytes(&self) -> u64 {
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).bytes
    }

    fn kernel_for(&self, rows: usize, out: usize, k: usize) -> Kernel {
        match self.kernel {
            KernelChoice::Fixed(kernel) => kernel,
            KernelChoice::Auto if rows * out <= SPLITK_MAX_OUTPUTS && k >= SPLITK_MIN_K => Kernel::SplitK,
            KernelChoice::Auto => Kernel::OneThread,
        }
    }

    fn pipe(&self, kernel: Kernel) -> &wgpu::ComputePipeline {
        #[cfg(feature = "plants")]
        if let Some((_, Some(p))) = &self.plant {
            return p;
        }
        self.gpu.pipe_for(kernel)
    }

    fn key(&self, w: &Weights<'_>, out: usize, k: usize) -> Key {
        #[cfg(feature = "plants")]
        if matches!(self.plant, Some((Plant::CacheByShape, _))) {
            return (None, out, k);
        }
        (Some(w.id), out, k)
    }

    fn upload(&self, codes: &[i32]) -> wgpu::Buffer {
        #[cfg(feature = "plants")]
        if matches!(self.plant, Some((Plant::DropUpload, _))) {
            return self.gpu.upload_blank(codes.len());
        }
        self.gpu.upload(codes)
    }
}

impl LinearBackend for GpuLinear {
    fn name(&self) -> String {
        let mut n = format!("gpu:{}", self.gpu.info.name);
        if !self.resident {
            n.push_str(" [per-call]");
        }
        if self.budget != GpuLinear::DEFAULT_BUDGET {
            n.push_str(&format!(" [budget {} B]", self.budget));
        }
        match self.kernel {
            KernelChoice::Fixed(Kernel::SplitK) => n.push_str(" [split-k]"),
            KernelChoice::Fixed(Kernel::OneThread) if self.resident => n.push_str(" [one-thread]"),
            _ => {}
        }
        #[cfg(feature = "plants")]
        if let Some((p, _)) = &self.plant {
            n = format!("planted-gpu-{p:?}");
        }
        n
    }

    /// No id: the weights are uploaded for this call.
    fn linear(&self, x: &[i32], w: &[i32], rows: usize, out: usize, k: usize) -> Vec<i64> {
        let kernel = self.kernel_for(rows, out, k);
        crate::check_shapes(x, w, rows, out, k);
        let blocks = self.gpu.blocks_of_outputs(out, k);
        let srcs: Vec<WSrc> = blocks.iter().map(|&(n0, nc)| WSrc::Data(&w[n0 * k..(n0 + nc) * k])).collect();
        self.gpu.gemm_blocks(self.pipe(kernel), kernel, x, &blocks, &srcs, rows, out, k)
    }

    fn linear_weights(&self, x: &[i32], w: Weights<'_>, rows: usize, out: usize, k: usize) -> Vec<i64> {
        if !self.resident {
            return self.linear(x, w.codes, rows, out, k);
        }
        crate::check_shapes(x, w.codes, rows, out, k);
        let kernel = self.kernel_for(rows, out, k);
        let key = self.key(&w, out, k);
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        c.tick += 1;
        let tick = c.tick;
        if !c.map.contains_key(&key) {
            let bytes = (w.codes.len() * 4) as u64;
            while c.bytes + bytes > self.budget && !c.map.is_empty() {
                let lru = *c.map.iter().min_by_key(|(_, r)| r.last_use).map(|(k, _)| k).expect("non-empty");
                let gone = c.map.remove(&lru).expect("present");
                c.bytes -= gone.bytes;
            }
            let blocks = self.gpu.blocks_of_outputs(out, k);
            let bufs = blocks.iter().map(|&(n0, nc)| self.upload(&w.codes[n0 * k..(n0 + nc) * k])).collect();
            c.bytes += bytes;
            c.map.insert(key, Resident { blocks, bufs, bytes, last_use: tick });
        }
        let r = c.map.get_mut(&key).expect("just inserted");
        r.last_use = tick;
        let srcs: Vec<WSrc> = r.bufs.iter().map(WSrc::Buf).collect();
        self.gpu.gemm_blocks(self.pipe(kernel), kernel, x, &r.blocks, &srcs, rows, out, k)
    }
}
