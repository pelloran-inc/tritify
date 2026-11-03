//! The backend firing controls. Every variant a host offers must match `portable` byte for byte, and the gate must
//! catch the two realistic ways a fast kernel goes wrong: an `i32` lane accumulator that wraps (the
//! `mullo_epi32` + `add_epi32` mistake), and a float accumulator (the "GPU shortcut"). A reordered integer sum is no
//! control, because it is exact and passes.

use super::*;
use kt_core::backend::{dot_portable, linear_cpu, Portable};

/// PLANTED: products and the running sum in `i32` lanes, wrapping on overflow.
fn dot_wrap_i32(a: &[i32], b: &[i32]) -> i64 {
    a.iter().zip(b).fold(0i32, |s, (&x, &y)| s.wrapping_add(x.wrapping_mul(y))) as i64
}

/// PLANTED: accumulate in `f32` and round back.
fn dot_f32(a: &[i32], b: &[i32]) -> i64 {
    a.iter().zip(b).map(|(&x, &y)| x as f32 * y as f32).sum::<f32>() as i64
}

struct Planted(&'static str, kt_core::backend::DotI32);
impl LinearBackend for Planted {
    fn name(&self) -> String {
        self.0.into()
    }
    fn linear(&self, x: &[i32], w: &[i32], rows: usize, out: usize, k: usize) -> Vec<i64> {
        linear_cpu(x, w, rows, out, k, self.1)
    }
}

fn reference() -> Arc<dyn LinearBackend> {
    Arc::new(Portable)
}

#[test]
fn every_available_backend_is_byte_identical() {
    let host = detect();
    let mut backends = vec![reference()];
    backends.extend(available_backends(&host));
    let reports = match equality_gate(&backends, &corpus()) {
        Ok(r) => r,
        Err(r) => panic!("a variant diverged: {:?}", r.iter().map(|g| (&g.backend, &g.diverged)).collect::<Vec<_>>()),
    };
    let ran: Vec<&str> = reports.iter().map(|r| r.backend.as_str()).collect();
    println!("host {} · backends executed {ran:?} · corpus hash {}", host.arch, reports[0].hash);
    assert!(ran.contains(&"portable"));
}

#[test]
fn the_gate_catches_a_wrapping_i32_accumulator() {
    let backends = [reference(), Arc::new(Planted("planted-wrap-i32", dot_wrap_i32)) as Arc<dyn LinearBackend>];
    let reports = equality_gate(&backends, &corpus()).err().expect("the wrapping variant must be caught");
    assert!(reports[0].diverged.is_empty(), "the reference never diverges from itself");
    assert!(reports[1].diverged.contains(&"extremes ±WLEVELS"), "got {:?}", reports[1].diverged);
}

#[test]
fn the_gate_catches_a_float_accumulator() {
    let backends = [reference(), Arc::new(Planted("planted-f32", dot_f32)) as Arc<dyn LinearBackend>];
    let reports = equality_gate(&backends, &corpus()).err().expect("the f32 variant must be caught");
    assert!(!reports[1].diverged.is_empty());
}

/// PLANTED on the GPU: the demo's tiled `i32` kernel used as an engine backend. It is exact only while a sum fits
/// `i32`, which engine-sized codes never do.
#[cfg(feature = "gpu")]
struct GpuI32Tiled(kt_gpu::Gpu);
#[cfg(feature = "gpu")]
impl LinearBackend for GpuI32Tiled {
    fn name(&self) -> String {
        "planted-gpu-i32-tiled".into()
    }
    fn linear(&self, x: &[i32], w: &[i32], rows: usize, out: usize, k: usize) -> Vec<i64> {
        self.0.gemm_i32_tiled(x, w, rows, out, k).into_iter().map(i64::from).collect()
    }
}

#[cfg(feature = "gpu")]
#[test]
fn the_gate_catches_the_i32_gpu_kernel_as_an_engine_backend() {
    let Some(gpu) = kt_gpu::Gpu::open(None) else {
        println!("no GPU adapter on this host — the GPU plant is not executed");
        return;
    };
    let backends = [reference(), Arc::new(GpuI32Tiled(gpu)) as Arc<dyn LinearBackend>];
    let reports = equality_gate(&backends, &corpus()).err().expect("the i32 GPU kernel must be caught");
    assert!(reports[1].diverged.contains(&"extremes ±WLEVELS"), "got {:?}", reports[1].diverged);
}

#[test]
fn a_gpu_the_host_lacks_is_refused() {
    let cfg = RuntimeConfig { backend: BackendChoice::Gpu(Some(9_999)), ..Default::default() };
    assert!(build(&cfg, &detect()).is_err());
}

#[test]
fn a_correct_kernel_in_another_order_passes() {
    // The control that cannot fail, shown once: summing backwards is still exact, so the gate rightly passes it.
    fn dot_reversed(a: &[i32], b: &[i32]) -> i64 {
        a.iter().zip(b).rev().map(|(&x, &y)| x as i64 * y as i64).sum()
    }
    let backends = [reference(), Arc::new(Planted("reversed", dot_reversed)) as Arc<dyn LinearBackend>];
    assert!(equality_gate(&backends, &corpus()).is_ok());
}

#[test]
fn portable_is_the_engine_kernel() {
    let c = &corpus()[1];
    let direct: Vec<i64> = (0..c.rows * c.out)
        .map(|i| dot_portable(&c.x[(i / c.out) * c.k..][..c.k], &c.w[(i % c.out) * c.k..][..c.k]))
        .collect();
    assert_eq!(CpuBackend::new(CpuVariant::Portable).unwrap().linear(&c.x, &c.w, c.rows, c.out, c.k), direct);
}

#[test]
fn auto_picks_the_best_ranked_variant_and_refuses_what_the_host_lacks() {
    let host = detect();
    assert!(host.cpu_variants.contains(&CpuVariant::Portable));
    let auto = auto_variant(&host);
    assert!(host.cpu_variants.contains(&auto));
    if cfg!(target_arch = "aarch64") {
        assert_eq!(auto, CpuVariant::Portable, "on aarch64 the compiler's loop is the measured-faster kernel");
    }
    if cfg!(target_arch = "x86_64") && CpuVariant::Avx2.available() {
        assert_ne!(auto, CpuVariant::Portable);
    }
    for v in CpuVariant::ALL {
        let cfg = RuntimeConfig { backend: BackendChoice::Cpu(Some(v)), ..Default::default() };
        let built = build(&cfg, &host);
        assert_eq!(built.is_ok(), v.available(), "{}: forced build must succeed iff available", v.name());
        if let Ok((b, sel)) = built {
            assert_eq!(b.name(), v.name());
            assert_eq!(sel.backend, v.name());
        }
    }
}

#[test]
fn install_reaches_the_engine_and_the_thread_cap_is_applied() {
    let cfg = RuntimeConfig {
        backend: BackendChoice::Cpu(Some(CpuVariant::Portable)),
        max_threads: Some(3),
        ..Default::default()
    };
    let sel = install(&cfg).unwrap();
    assert_eq!(sel.backend, "portable");
    assert_eq!(kt_core::backend::backend().name(), "portable");
    assert_eq!(kt_core::max_threads(), 3);
    kt_core::set_max_threads(0);
    kt_core::backend::reset_backend();
}

/// The resident GPU backend, its eviction under a tiny budget, and both kernels pass the residency gate (the
/// corpus twice, then same-shape twins); each planted residency defect is caught by the same gate.
#[cfg(feature = "gpu")]
#[test]
fn residency_holds_and_every_planted_residency_defect_is_caught() {
    use kt_gpu::{Gpu, GpuLinear, Kernel, KernelChoice, Plant};
    if Gpu::open(None).is_none() {
        println!("no GPU adapter on this host — the residency gate is not executed");
        return;
    }
    let open = || Gpu::open(None).unwrap();
    let cases = corpus();
    let good: [Arc<dyn LinearBackend>; 4] = [
        Arc::new(GpuLinear::new(open())),
        Arc::new(GpuLinear::new(open()).with_budget(1)),
        Arc::new(GpuLinear::new(open()).with_kernel(KernelChoice::Fixed(Kernel::OneThread))),
        Arc::new(GpuLinear::new(open()).with_kernel(KernelChoice::Fixed(Kernel::SplitK))),
    ];
    for b in &good {
        residency_gate(b, &cases).unwrap_or_else(|e| panic!("{e}"));
        println!("residency gate: {} passes", b.name());
    }
    for p in [Plant::DropUpload, Plant::CacheByShape, Plant::SplitKSkipLast] {
        let b: Arc<dyn LinearBackend> = Arc::new(GpuLinear::planted(open(), p));
        let err = residency_gate(&b, &cases).expect_err("a planted residency defect must be caught");
        println!("residency gate: {err}");
    }
}

/// Each planted partition of the CPU linear layer gives another answer than the serial loop on some shape of
/// the property grid, at a forced worker count; the equality gate's corpus, at the engine's own plan, is reported.
#[test]
fn every_planted_partition_is_caught() {
    use crate::planted::{BadPartition, PlantedPartition};
    let grid = |bad| {
        let mut wrong = Vec::new();
        for (rows, out, k) in [(1usize, 17usize, 5usize), (3, 17, 5), (7, 64, 33), (16, 2, 33), (31, 17, 1)] {
            let x: Vec<i32> = (0..rows * k).map(|i| (i as i32 * 7919) % 1001 - 500).collect();
            let w: Vec<i32> = (0..out * k).map(|i| (i as i32 * 104729) % 997 - 498).collect();
            for threads in [2usize, 3, 16, 64] {
                let b = PlantedPartition { bad, threads: Some(threads) };
                if b.linear(&x, &w, rows, out, k) != Portable.linear(&x, &w, rows, out, k) {
                    wrong.push(format!("{rows}×{out}×{k}/{threads}"));
                }
            }
        }
        wrong
    };
    for bad in [BadPartition::FloorOffsets, BadPartition::DropLastRun, BadPartition::SwappedIndex] {
        let wrong = grid(bad);
        assert!(!wrong.is_empty(), "{bad:?} must give another answer on some shape");
        let planned: Arc<dyn LinearBackend> = Arc::new(PlantedPartition { bad, threads: None });
        let corpus_verdict = match equality_gate(&[reference(), planned], &corpus()) {
            Ok(_) => "passes".to_string(),
            Err(r) => format!("diverges on {:?}", r[1].diverged),
        };
        println!("planted partition {bad:?}: wrong on {} of 20 grid cells ({}…); corpus at the engine's plan {corpus_verdict}", wrong.len(), wrong[0]);
    }
}
