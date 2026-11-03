//! Speed: the linear accumulate on the CPU (`portable`, and the host's auto variant when it differs), on the
//! resident wgpu backend and on the CUDA digit backend. The shapes are those of `gpu_bench`: an OLMoE-sized prefill
//! (8 × 2048 × 2048), a Qwen-sized decode projection (1 × 4864 × 896), Qwen's LM head (1 × 151936 × 896) and a row
//! sweep. Each answer is checked against the CPU's before it is timed.
//! Backends the host lacks are skipped; `--cuda-emu` adds the host executor of the digit algorithm.
//!
//!   cargo run --release -p kt-runtime --features cuda --example cuda_bench [-- --cuda-emu]

use kt_core::backend::{LinearBackend, Portable, WeightId, Weights};
use std::sync::Arc;
use std::time::Instant;

fn codes(n: usize, mut s: u64) -> Vec<i32> {
    let lv = kt_core::layers::WLEVELS;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % (2 * lv as u64 + 1)) as i64 as i32 - lv as i32
        })
        .collect()
}

/// Microseconds per call: one untimed call first (resident backends take the weights in), then at least two calls and
/// about half a second.
fn time(b: &dyn LinearBackend, x: &[i32], w: Weights<'_>, (m, n, k): (usize, usize, usize)) -> u128 {
    b.linear_weights(x, w, m, n, k);
    let (t, mut reps) = (Instant::now(), 0u128);
    while reps < 2 || (t.elapsed().as_millis() < 500 && reps < 200) {
        std::hint::black_box(b.linear_weights(x, w, m, n, k));
        reps += 1;
    }
    t.elapsed().as_micros() / reps
}

fn main() {
    let host = kt_runtime::detect();
    let mut backends: Vec<(String, Arc<dyn LinearBackend>)> =
        vec![("cpu portable (all cores)".into(), Arc::new(Portable))];
    let auto = kt_runtime::auto_variant(&host);
    if auto != kt_runtime::CpuVariant::Portable {
        let b = kt_runtime::CpuBackend::new(auto).expect("the auto variant runs on this host");
        backends.push((format!("cpu {} (auto, all cores)", auto.name()), Arc::new(b)));
    }
    #[cfg(feature = "gpu")]
    if let Some(g) = kt_runtime::kt_gpu::Gpu::open(None) {
        let b = kt_runtime::kt_gpu::GpuLinear::new(g);
        backends.push((format!("wgpu resident ({})", b.name()), Arc::new(b)));
    }
    if let Some(b) = kt_runtime::kt_cuda::open(None, None) {
        backends.push((format!("cuda ({})", b.name()), b));
    }
    if std::env::args().any(|a| a == "--cuda-emu") {
        backends.push(("cuda-emu (host executor)".into(), Arc::new(kt_runtime::kt_cuda::emulator())));
    }
    println!(
        "cuda_bench · {} / {} · {} cores · cuda {:?} · µs per call, MMAC/s",
        host.arch, host.os, host.cores, host.cudas
    );
    let mut shapes = vec![
        (8, 2048, 2048, "prefill 8×2048×2048"),
        (1, 4864, 896, "decode 1×4864×896 (Qwen MLP)"),
        (1, 151_936, 896, "decode 1×151936×896 (Qwen LM head)"),
    ];
    for rows in [1usize, 2, 4, 8, 16, 32, 64] {
        shapes.push((rows, 2048, 2048, "sweep"));
    }
    for (m, n, k, label) in shapes {
        let (x, w) = (codes(m * k, 1 + m as u64), codes(n * k, 2 + n as u64));
        let want = Portable.linear(&x, &w, m, n, k);
        let id = WeightId::fresh();
        print!("  {label:<36} m={m:<3}");
        for (bname, b) in &backends {
            let wts = Weights { id, codes: &w };
            assert_eq!(b.linear_weights(&x, wts, m, n, k), want, "{bname} differs on {label}");
            let us = time(b.as_ref(), &x, wts, (m, n, k));
            print!("  | {bname}: {us:>8} µs {:>7}", (m * n * k) as u128 / us.max(1));
        }
        println!();
    }
    println!("  digit-backend fallbacks to portable: {}", kt_runtime::kt_cuda::fallbacks());
}
