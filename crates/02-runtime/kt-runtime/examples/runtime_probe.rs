//! Probe a host: what it offers, what auto picks, whether every backend it can run returns the same bytes on the
//! fixed corpus, and how fast each one is. Exits 1 if any backend diverges. CI runs this on its x86 runner, so the
//! log records which variants that silicon actually executed.
//!
//! `--cuda-emu` adds the digit backend's host executor. With `--features cuda` the CUDA devices are probed
//! too.
//!
//!   cargo run --release -p kt-runtime --example runtime_probe [--features cuda] [-- --cuda-emu]

use kt_core::backend::{LinearBackend, Portable, Weights};
use kt_runtime::{auto_variant, available_backends, corpus, detect, equality_gate, flush_case};
use std::sync::Arc;
use std::time::Instant;

fn kt_cuda_built() -> bool {
    kt_runtime::kt_cuda::CUDA
}

fn main() {
    let host = detect();
    let names: Vec<&str> = host.cpu_variants.iter().map(|v| v.name()).collect();
    println!("kt-runtime probe");
    println!("  host        : {} / {} · {} cores", host.arch, host.os, host.cores);
    println!("  cpu variants: {names:?}");
    for (i, g) in host.gpus.iter().enumerate() {
        println!("  gpu {i}       : {} / {} ({})", g.api, g.name, g.kind);
    }
    if host.gpus.is_empty() {
        println!("  gpu         : none");
    }
    for (i, c) in host.cudas.iter().enumerate() {
        println!("  cuda {i}      : {c}");
    }
    if host.cudas.is_empty() {
        println!("  cuda        : none{}", if kt_cuda_built() { "" } else { " (built without the cuda feature)" });
    }
    println!("  auto        : {}", auto_variant(&host).name());

    let mut backends: Vec<Arc<dyn LinearBackend>> = vec![Arc::new(Portable)];
    backends.extend(available_backends(&host));
    if std::env::args().any(|a| a == "--cuda-emu") {
        backends.push(Arc::new(kt_runtime::kt_cuda::emulator()));
    }
    let (mut ok, reports) = match equality_gate(&backends, &corpus()) {
        Ok(r) => (true, r),
        Err(r) => (false, r),
    };
    for r in &reports {
        let verdict =
            if r.diverged.is_empty() { "identical".to_string() } else { format!("DIVERGED {:?}", r.diverged) };
        println!("  gate        : {:<10} {}  {verdict}", r.backend, r.hash);
    }
    // the digit backend's bound case (the smallest k beyond one int32 flush), on every backend
    let bound = flush_case();
    let want = Portable.linear(&bound.x, &bound.w, bound.rows, bound.out, bound.k);
    for b in &backends[1..] {
        let got = b.linear_weights(&bound.x, Weights { id: bound.id, codes: &bound.w }, bound.rows, bound.out, bound.k);
        let verdict = if got == want { "identical" } else { "DIVERGED" };
        ok &= got == want;
        println!("  bound       : {:<10} k = {}  {verdict}", b.name(), bound.k);
    }
    println!("  fallbacks   : {} (calls a digit backend handed to portable)", kt_runtime::kt_cuda::fallbacks());

    // speed: an OLMoE-sized projection (8 rows × 2048 out × 2048 in), serial and with all cores
    let (rows, out, k) = (8usize, 2048usize, 2048usize);
    let c = &corpus()[5];
    let x: Vec<i32> = c.x.iter().cycle().take(rows * k).copied().collect();
    let w: Vec<i32> = c.w.iter().cycle().take(out * k).copied().collect();
    for threads in [1usize, 0] {
        kt_core::set_max_threads(threads);
        for b in &backends[1..] {
            let reps = 10u128;
            let t = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(b.linear(&x, &w, rows, out, k));
            }
            let us = t.elapsed().as_micros() / reps;
            let mmacs = (rows * out * k) as u128 / us.max(1);
            let label = if threads == 1 { "1 thread" } else { "all cores" };
            println!("  speed       : {:<10} {label:<9} {us:>8} µs/call  {mmacs:>6} MMAC/s", b.name());
        }
    }
    kt_core::set_max_threads(0);

    println!("  verdict     : {}", if ok { "every backend byte-identical" } else { "A BACKEND DIVERGED" });
    if !ok {
        std::process::exit(1);
    }
}
