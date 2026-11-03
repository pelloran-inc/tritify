//! Dispatch: one product split into blocks of outputs (each block's weights fit one binding) and rows (each dispatch
//! fits the device's per-dimension limit). It runs the chosen kernel and reads the `i64` results back. The weights of
//! a block come either uploaded for the call or from a resident buffer. Output and readback buffers are reused by
//! size.

use crate::{Gpu, Kernel};
use wgpu::util::DeviceExt;

/// Workgroup size of the one-thread kernel, and the dispatch limit per dimension.
const I64_GROUP: usize = 64;
const MAX_GROUPS: usize = 65_535;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Meta {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

/// Where a block's weights come from.
pub(crate) enum WSrc<'a> {
    /// Upload these codes for this call.
    Data(&'a [i32]),
    /// A buffer already on the device.
    Buf(&'a wgpu::Buffer),
}

impl Gpu {
    /// Exact accumulators for `x` (`m × k`) against weight blocks `(n0, nc)` with sources `srcs`, run by `kernel`
    /// through `pipe`; the result is `m × n` in `i64`.
    #[allow(clippy::too_many_arguments)] // one product: the kernel, the operands, the block layout and the shape
    pub(crate) fn gemm_blocks(
        &self,
        pipe: &wgpu::ComputePipeline,
        kernel: Kernel,
        x: &[i32],
        blocks: &[(usize, usize)],
        srcs: &[WSrc],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<i64> {
        let mut y = vec![0i64; m * n];
        for (&(n0, nc), src) in blocks.iter().zip(srcs) {
            // rows per dispatch: the binding limit on x, and for the one-thread kernel the group limit on m·nc
            let by_groups = match kernel {
                Kernel::OneThread => MAX_GROUPS * I64_GROUP / nc.max(1),
                Kernel::SplitK => MAX_GROUPS * MAX_GROUPS / nc.max(1),
            };
            let rows = by_groups.min(self.max_rows(k)).max(1);
            for m0 in (0..m).step_by(rows) {
                let mc = rows.min(m - m0);
                let words = self.dispatch_i64(pipe, kernel, &x[m0 * k..(m0 + mc) * k], src, mc, nc, k);
                for r in 0..mc {
                    for c in 0..nc {
                        let i = r * nc + c;
                        let (lo, hi) = (words[2 * i] as u32 as u64, words[2 * i + 1] as u32 as u64);
                        y[(m0 + r) * n + n0 + c] = ((hi << 32) | lo) as i64;
                    }
                }
            }
        }
        y
    }

    /// Upload `codes` as a storage buffer that stays on the device.
    pub(crate) fn upload(&self, codes: &[i32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kt-gpu weights"),
            contents: bytemuck::cast_slice(codes),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    /// PLANTED (the `plants` feature only): a weight buffer allocated for `len` codes and never written.
    #[cfg(feature = "plants")]
    pub(crate) fn upload_blank(&self, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kt-gpu planted blank weights"),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    #[allow(clippy::too_many_arguments)] // one dispatch: the pipeline, the kernel, both operands and the shape
    fn dispatch_i64(
        &self,
        pipe: &wgpu::ComputePipeline,
        kernel: Kernel,
        x: &[i32],
        w: &WSrc,
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<i32> {
        let uploaded;
        let wb = match w {
            WSrc::Buf(b) => *b,
            WSrc::Data(d) => {
                uploaded = self.storage(d);
                &uploaded
            }
        };
        let total = (m * n) as u32;
        let (gx, gy) = match kernel {
            Kernel::OneThread => (total.div_ceil(I64_GROUP as u32), 1),
            Kernel::SplitK => {
                let gx = total.min(MAX_GROUPS as u32);
                (gx, total.div_ceil(gx))
            }
        };
        self.run(pipe, x, wb, m, n, k, 2, (gx, gy))
    }

    /// The tiled `i32` kernel, uploading both operands (the demo's path).
    pub(crate) fn run_tiled(&self, x: &[i32], w: &[i32], m: usize, n: usize, k: usize) -> Vec<i32> {
        let wb = self.storage(w);
        let groups = ((m as u32).div_ceil(16), (n as u32).div_ceil(16));
        self.run(&self.tiled_pipe, x, &wb, m, n, k, 1, groups)
    }

    fn storage(&self, data: &[i32]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }

    #[allow(clippy::too_many_arguments)] // one dispatch: the pipeline, both operands, the shape and the grid
    fn run(
        &self,
        pipe: &wgpu::ComputePipeline,
        x: &[i32],
        wb: &wgpu::Buffer,
        m: usize,
        n: usize,
        k: usize,
        words_per_out: usize,
        (gx, gy): (u32, u32),
    ) -> Vec<i32> {
        let dev = &self.device;
        let xb = self.storage(x);
        let out_bytes = (m * n * words_per_out * 4) as u64;
        let (ob, rb) = self.take_buffers(out_bytes);
        let meta = Meta { m: m as u32, n: n as u32, k: k as u32, _pad: 0 };
        let mb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&meta),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: xb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: ob.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mb.as_entire_binding() },
            ],
        });
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(pipe);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(&ob, 0, &rb, 0, out_bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = dev.poll(wgpu::Maintain::Wait);
        rx.recv().expect("kt-gpu: map callback dropped").expect("kt-gpu: reading the result back failed");
        let out: Vec<i32> = bytemuck::cast_slice(&slice.get_mapped_range()[..]).to_vec();
        rb.unmap();
        self.give_buffers(out_bytes, ob, rb);
        out
    }

    /// An output buffer and its readback twin of `bytes`, from the pool or new.
    fn take_buffers(&self, bytes: u64) -> (wgpu::Buffer, wgpu::Buffer) {
        if let Some(pair) = self.pool.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&bytes).and_then(Vec::pop) {
            return pair;
        }
        let buf = |usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        (
            buf(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            buf(wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ),
        )
    }

    fn give_buffers(&self, bytes: u64, ob: wgpu::Buffer, rb: wgpu::Buffer) {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        let free = pool.entry(bytes).or_default();
        if free.len() < 4 {
            free.push((ob, rb));
        }
    }
}

pub(crate) fn bind_layout(dev: &wgpu::Device) -> wgpu::BindGroupLayout {
    let storage = |read_only| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let uniform = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let entry =
        |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
    dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("kt-gpu gemm"),
        entries: &[entry(0, storage(true)), entry(1, storage(true)), entry(2, storage(false)), entry(3, uniform)],
    })
}

pub(crate) fn pipeline(dev: &wgpu::Device, layout: &wgpu::BindGroupLayout, wgsl: &str) -> wgpu::ComputePipeline {
    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}
