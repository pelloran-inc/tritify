//! The reproducing anchors under every compute backend this host can run. Builds a real OLMoE once (2
//! layers or all of them), then runs the engine's self-check with each backend installed in turn and prints the
//! reproducing hash per backend. Every backend must print the same hash: `505cb108…` for 2 layers, `74005166…` for
//! all 16. With `--plant`, deliberately wrong kernels run too: on the CPU an `i32` accumulator that wraps
//! and an `f32` accumulator, and on a GPU the demo's tiled `i32` kernel. Their hashes must differ, which shows the
//! comparison can fail.
//!
//! `--cpu-only` skips GPU adapters (a headless server may expose only a slow software rasterizer). `--gpu-variants` adds
//! the per-call GPU and a GPU whose budget evicts on every call. With `--plant` on a GPU host, the
//! three planted residency defects run too: a dropped upload, a cache keyed by shape, and split-K skipping a slice.
//!
//! A CUDA device joins through `kt-runtime` when the build has the `cuda` feature. `--cuda-emu` adds the digit
//! backend's host executor. `--gpu-variants` adds a digit backend that evicts on every call. With `--plant`, the four
//! planted digit defects run on each digit executor: a dropped `(2,2)` plane, a wrong carry, no `int32` flush (the
//! models' `k` never reaches the bound, so this one gives the anchor) and a cache keyed by shape.
//!
//! With `--plant`, three planted partitions of the CPU linear layer run too (wrong run offsets, a dropped last
//! run, rows and outputs swapped), at the engine's own thread plan.
//!
//! `--rms-norm-eps <v>` answers the config's `rms_norm_eps` with `v`, an overlay in this example only (the engine has
//! no override). OLMoE's config omits the key, so without it the build takes transformers' `1e-5`. The earlier
//! anchors, `6ae7e544…` (2 layers) and `3dadb4f1…` (16), are the `1e-6` model's, the engine's earlier fallback.
//! With `--rms-norm-eps 1e-6` they must reproduce.
//!
//!   cargo run --release -p kt-convert --example anchor_backends [--features kt-runtime/cuda] -- <olmoe_dir>
//!     [layers=2|all] [--plant] [--cpu-only] [--gpu-variants] [--cuda-emu] [--rms-norm-eps <v>]

use kt_convert::{HfModelSource, MoeArch};
use kt_core::backend::{install_backend, linear_cpu, reset_backend, DotI32, LinearBackend, Portable};
use kt_core::plugin::ModelSource;
use kt_core::VerifiableModel;
use std::sync::Arc;
use std::time::Instant;

/// Override `num_hidden_layers` to build a bounded prefix of the real checkpoint (as `olmoe_build` does), and with
/// `--rms-norm-eps`, `rms_norm_eps`.
struct CappedLayers<'a> {
    inner: &'a HfModelSource,
    cap: usize,
    eps: Option<f64>,
}
impl ModelSource for CappedLayers<'_> {
    fn hf_arch_ids(&self) -> Vec<String> {
        self.inner.hf_arch_ids()
    }
    fn config_usize(&self, key: &str) -> Option<usize> {
        if key == "num_hidden_layers" {
            return Some(self.cap);
        }
        self.inner.config_usize(key)
    }
    fn config_f64(&self, key: &str) -> Option<f64> {
        if key == "rms_norm_eps" && self.eps.is_some() {
            return self.eps;
        }
        self.inner.config_f64(key)
    }
    fn config_bool(&self, key: &str) -> Option<bool> {
        self.inner.config_bool(key)
    }
    fn tensor(&self, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        self.inner.tensor(name)
    }
    fn tensor_names(&self) -> Vec<String> {
        self.inner.tensor_names()
    }
}

/// PLANTED: `i32` lanes that wrap on overflow.
fn dot_wrap_i32(a: &[i32], b: &[i32]) -> i64 {
    a.iter().zip(b).fold(0i32, |s, (&x, &y)| s.wrapping_add(x.wrapping_mul(y))) as i64
}

/// PLANTED: an `f32` accumulator.
fn dot_f32(a: &[i32], b: &[i32]) -> i64 {
    a.iter().zip(b).map(|(&x, &y)| x as f32 * y as f32).sum::<f32>() as i64
}

/// PLANTED on the GPU: the demo's tiled `i32` kernel, exact only while a sum fits `i32`.
struct GpuI32Tiled(kt_runtime::kt_gpu::Gpu);
impl LinearBackend for GpuI32Tiled {
    fn name(&self) -> String {
        "planted-gpu-i32-tiled".into()
    }
    fn linear(&self, x: &[i32], w: &[i32], rows: usize, out: usize, k: usize) -> Vec<i64> {
        self.0.gemm_i32_tiled(x, w, rows, out, k).into_iter().map(i64::from).collect()
    }
}

struct Planted(&'static str, DotI32);
impl LinearBackend for Planted {
    fn name(&self) -> String {
        self.0.into()
    }
    fn linear(&self, x: &[i32], w: &[i32], rows: usize, out: usize, k: usize) -> Vec<i64> {
        linear_cpu(x, w, rows, out, k, self.1)
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!(
            "usage: anchor_backends <olmoe_dir> [layers=2|all] [--plant] [--cpu-only] [--gpu-variants] [--cuda-emu] \
             [--rms-norm-eps <v>]"
        );
        std::process::exit(2);
    }
    let plant = a.iter().any(|s| s == "--plant");
    let cpu_only = a.iter().any(|s| s == "--cpu-only");
    let gpu_variants = a.iter().any(|s| s == "--gpu-variants");
    let cuda_emu = a.iter().any(|s| s == "--cuda-emu");
    let eps = a.iter().position(|s| s == "--rms-norm-eps").map(|i| {
        a.get(i + 1).and_then(|v| v.parse::<f64>().ok()).unwrap_or_else(|| {
            eprintln!("--rms-norm-eps needs a number");
            std::process::exit(2);
        })
    });
    let src = HfModelSource::from_dir(&a[1]).unwrap_or_else(|e| {
        eprintln!("load: {e}");
        std::process::exit(1);
    });
    let total = src.config_usize("num_hidden_layers").expect("num_hidden_layers");
    let cap = match a.get(2).map(String::as_str) {
        Some("all") => total,
        Some(n) if !n.starts_with("--") => n.parse().unwrap_or(2).min(total),
        _ => 2.min(total),
    };
    let model = MoeArch::build_decoder_moe(&CappedLayers { inner: &src, cap, eps }).unwrap_or_else(|e| {
        eprintln!("build: {e}");
        std::process::exit(1);
    });

    let host = kt_runtime::detect();
    let names: Vec<&str> = host.cpu_variants.iter().map(|v| v.name()).collect();
    let gpus: Vec<String> = host.gpus.iter().map(|g| format!("{} / {}", g.api, g.name)).collect();
    println!(
        "anchor_backends · OLMoE {cap}/{total} layers · host {} / {} · cpu {names:?} · gpu {gpus:?} · cuda {:?}",
        host.arch, host.os, host.cudas
    );
    if let Some(eps) = eps {
        println!("  rms_norm_eps overlay {eps:e}");
    }

    let mut backends: Vec<Arc<dyn LinearBackend>> = vec![Arc::new(Portable)];
    backends.extend(
        kt_runtime::available_backends(&host).into_iter().filter(|b| !cpu_only || !b.name().starts_with("gpu:")),
    );
    let gpu = || kt_gpu::Gpu::open(None).filter(|_| !cpu_only);
    if gpu_variants {
        backends.extend(gpu().map(|g| Arc::new(kt_gpu::GpuLinear::per_call(g)) as Arc<dyn LinearBackend>));
        backends.extend(gpu().map(|g| Arc::new(kt_gpu::GpuLinear::new(g).with_budget(1)) as Arc<dyn LinearBackend>));
        backends.extend(kt_cuda::open(None, Some(1)));
    }
    if cuda_emu {
        backends.push(Arc::new(kt_cuda::emulator()));
        if gpu_variants {
            backends.push(Arc::new(kt_cuda::emulator().with_budget(1)));
        }
    }
    if plant {
        backends.push(Arc::new(Planted("planted-wrap-i32", dot_wrap_i32)));
        backends.push(Arc::new(Planted("planted-f32", dot_f32)));
        for bad in [
            kt_runtime::planted::BadPartition::FloorOffsets,
            kt_runtime::planted::BadPartition::DropLastRun,
            kt_runtime::planted::BadPartition::SwappedIndex,
        ] {
            backends.push(Arc::new(kt_runtime::planted::PlantedPartition { bad, threads: None }));
        }
        if let Some(g) = gpu() {
            backends.push(Arc::new(GpuI32Tiled(g)));
        }
        for p in [kt_gpu::Plant::DropUpload, kt_gpu::Plant::CacheByShape, kt_gpu::Plant::SplitKSkipLast] {
            backends.extend(gpu().map(|g| Arc::new(kt_gpu::GpuLinear::planted(g, p)) as Arc<dyn LinearBackend>));
        }
        for p in [
            kt_cuda::Plant::DropPlane22,
            kt_cuda::Plant::WrongCarry,
            kt_cuda::Plant::NoFlush,
            kt_cuda::Plant::CacheByShape,
        ] {
            if cuda_emu {
                backends.push(Arc::new(kt_cuda::TcLinear::planted(kt_cuda::HostExec, p)));
            }
            backends.extend(kt_cuda::open_planted(None, p));
        }
    }
    for b in backends {
        let name = b.name();
        install_backend(b);
        let t = Instant::now();
        let rep = model.self_check();
        let ms = t.elapsed().as_millis();
        println!("  {name:<18} hash = {}  det={}  {ms} ms", rep.hash, rep.deterministic);
    }
    reset_backend();
    println!("  digit-backend fallbacks to portable: {}", kt_cuda::fallbacks());
}
