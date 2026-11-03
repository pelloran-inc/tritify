//! Speed: the linear accumulate on the CPU and on the GPU in three forms. The GPU forms are:
//! - per-call: the original behaviour, which uploads the weights every call
//! - resident with the one-thread kernel
//! - resident with the split-K kernel
//!
//! Shapes: an OLMoE-sized prefill (8 × 2048 × 2048), a Qwen-sized decode projection (1 × 4864 × 896), Qwen's LM head
//! (1 × 151936 × 896), and a row sweep that sets the split-K threshold. Each answer is checked against the CPU's
//! before it is timed.
//!
//!   cargo run --release -p kt-runtime --example gpu_bench

use kt_core::backend::{LinearBackend, Portable, WeightId, Weights};
use kt_gpu::{Gpu, GpuLinear, Kernel, KernelChoice};
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

/// Microseconds per call: at least two calls and about half a second; resident backends get one untimed call first.
fn time(b: &dyn LinearBackend, x: &[i32], w: Weights<'_>, (m, n, k): (usize, usize, usize), warm: bool) -> u128 {
    if warm {
        b.linear_weights(x, w, m, n, k);
    }
    let (t, mut reps) = (Instant::now(), 0u128);
    while reps < 2 || (t.elapsed().as_millis() < 500 && reps < 50) {
        std::hint::black_box(b.linear_weights(x, w, m, n, k));
        reps += 1;
    }
    t.elapsed().as_micros() / reps
}

fn main() {
    let Some(name) = Gpu::open(None).map(|g| g.info().name.clone()) else {
        println!("gpu_bench: no GPU adapter on this host");
        return;
    };
    let open = || Gpu::open(None).expect("adapter");
    let backends: Vec<(&str, Box<dyn LinearBackend>, bool)> = vec![
        ("cpu portable (all cores)", Box::new(Portable), false),
        ("gpu per-call (upload every call)", Box::new(GpuLinear::per_call(open())), false),
        (
            "gpu resident one-thread",
            Box::new(GpuLinear::new(open()).with_kernel(KernelChoice::Fixed(Kernel::OneThread))),
            true,
        ),
        (
            "gpu resident split-k",
            Box::new(GpuLinear::new(open()).with_kernel(KernelChoice::Fixed(Kernel::SplitK))),
            true,
        ),
    ];
    println!("gpu_bench · {name} · {} · µs per call, MMAC/s", std::env::consts::ARCH);
    let mut shapes = vec![
        (8, 2048, 2048, "prefill 8×2048×2048"),
        (1, 4864, 896, "decode 1×4864×896 (Qwen MLP)"),
        (1, 151_936, 896, "decode 1×151936×896 (Qwen LM head)"),
    ];
    for rows in [1usize, 2, 4, 8, 16, 32] {
        shapes.push((rows, 2048, 2048, "sweep"));
    }
    for (m, n, k, label) in shapes {
        let (x, w) = (codes(m * k, 1 + m as u64), codes(n * k, 2 + n as u64));
        let want = Portable.linear(&x, &w, m, n, k);
        let id = WeightId::fresh();
        print!("  {label:<36} m={m:<3}");
        for (bname, b, warm) in &backends {
            let wts = Weights { id, codes: &w };
            assert_eq!(b.linear_weights(&x, wts, m, n, k), want, "{bname} differs on {label}");
            let us = time(b.as_ref(), &x, wts, (m, n, k), *warm);
            print!("  | {bname}: {us:>8} µs {:>6}", (m * n * k) as u128 / us.max(1));
        }
        println!();
    }
}
