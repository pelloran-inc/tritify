//! The CPU work partition, measured. It prints:
//! - the cost of a scoped spawn with `t` threads (`t − 1` spawned, the caller works too), per call
//! - for each CPU variant, the decode shapes and two many-row shapes, the time of the flat output partition at
//!   `t = 1, 2, 4, …` up to the host's threads, each answer checked against the serial loop first
//! - the engine's own `linear_cpu` on the same shapes, with the partition the engine at this commit chooses
//! - an interleaved A/B on every shape: the earlier row partition of `linear_cpu` (`par_blocks` over rows, as
//!   it was) against the engine, alternating in one process, the medians of 15 rounds. Separate runs on a host with
//!   efficiency cores differ by more than 10% on their own, so the comparison is made here.
//!
//! The sweep sets the per-thread work constant `W`. The flat partition here is the engine's design, written out
//! with an explicit `t`, so it can be measured against the earlier row partition.
//!
//!   cargo run --release -p kt-runtime --example cpu_partition

use kt_core::backend::{linear_cpu, DotI32};
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

/// The flat partition with an explicit `t`: `rows × out` cut into contiguous runs; element `i` is
/// `dot(x[i / out], w[i % out])`. The calling thread computes the first run.
fn flat(x: &[i32], w: &[i32], (rows, out, k): (usize, usize, usize), dot: DotI32, t: usize) -> Vec<i64> {
    let mut y = vec![0i64; rows * out];
    if y.is_empty() {
        return y;
    }
    let chunk = y.len().div_ceil(t.clamp(1, y.len()));
    let fill = |start: usize, ys: &mut [i64]| {
        for (e, slot) in ys.iter_mut().enumerate() {
            let i = start + e;
            *slot = dot(&x[i / out * k..][..k], &w[i % out * k..][..k]);
        }
    };
    std::thread::scope(|s| {
        let mut runs = y.chunks_mut(chunk).enumerate();
        let first = runs.next();
        for (ci, ys) in runs {
            let fill = &fill;
            s.spawn(move || fill(ci * chunk, ys));
        }
        if let Some((_, ys)) = first {
            fill(0, ys);
        }
    });
    y
}

/// The earlier row partition of `linear_cpu`: blocks of whole rows on `min(cap, rows)` spawned workers, the caller
/// waiting, serial below 2^20 multiply-accumulates.
fn rows_partition(x: &[i32], w: &[i32], (rows, out, k): (usize, usize, usize), dot: DotI32, cap: usize) -> Vec<i64> {
    let mut y = vec![0i64; rows * out];
    let threads = cap.min(rows).max(1);
    let fill = |r: usize, yr: &mut [i64]| {
        for (o, slot) in yr.iter_mut().enumerate() {
            *slot = dot(&x[r * k..(r + 1) * k], &w[o * k..(o + 1) * k]);
        }
    };
    if out == 0 || threads < 2 || rows * out * k < (1 << 20) {
        y.chunks_mut(out.max(1)).enumerate().for_each(|(r, yr)| fill(r, yr));
        return y;
    }
    let chunk = rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (ci, group) in y.chunks_mut(chunk * out).enumerate() {
            let fill = &fill;
            s.spawn(move || group.chunks_mut(out).enumerate().for_each(|(bi, yr)| fill(ci * chunk + bi, yr)));
        }
    });
    y
}

/// The median of `xs`.
fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs[xs.len() / 2]
}

/// Microseconds per call of `f`: at least three calls and about 0.3 s.
fn time(mut f: impl FnMut()) -> f64 {
    f();
    let (t, mut reps) = (Instant::now(), 0u32);
    while reps < 3 || (t.elapsed().as_millis() < 300 && reps < 2000) {
        f();
        reps += 1;
    }
    t.elapsed().as_nanos() as f64 / reps as f64 / 1000.0
}

fn main() {
    let host = kt_runtime::detect();
    let cores = host.cores;
    let ts: Vec<usize> = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128].into_iter().filter(|&t| t <= cores).collect();
    println!("cpu_partition · {} / {} · {cores} threads · µs per call", host.arch, host.os);

    print!("  spawn cost (scope with t threads)  ");
    for &t in &ts[1..] {
        let us = time(|| {
            std::thread::scope(|s| {
                for _ in 1..t {
                    s.spawn(|| std::hint::black_box(0));
                }
            })
        });
        print!("  t={t}: {us:.1}");
    }
    println!();

    let shapes = [
        (1, 896, 896, "Qwen q/o 1×896×896"),
        (1, 4864, 896, "Qwen gate/up 1×4864×896"),
        (1, 896, 4864, "Qwen down 1×896×4864"),
        (1, 1024, 2048, "OLMoE expert 1×1024×2048"),
        (1, 2048, 2048, "OLMoE attn 1×2048×2048"),
        (1, 151_936, 896, "Qwen LM head 1×151936×896"),
        (8, 2048, 2048, "prefill 8×2048×2048"),
        (64, 2048, 2048, "prefill 64×2048×2048"),
    ];
    for v in host.cpu_variants.iter().copied() {
        let dot = v.dot().expect("an available variant has a dot kernel");
        for &(m, n, k, label) in &shapes {
            let (x, w) = (codes(m * k, 1 + m as u64), codes(n * k, 2 + n as u64));
            let want = flat(&x, &w, (m, n, k), dot, 1);
            print!("  {:<8} {label:<28} work {:>11}", v.name(), m * n * k);
            for &t in &ts {
                assert_eq!(flat(&x, &w, (m, n, k), dot, t), want, "{} {label}: t = {t} differs", v.name());
                let us = time(|| {
                    std::hint::black_box(flat(&x, &w, (m, n, k), dot, t));
                });
                print!("  t={t}: {us:.0}");
            }
            assert_eq!(linear_cpu(&x, &w, m, n, k, dot), want, "{} {label}: the engine differs", v.name());
            let us = time(|| {
                std::hint::black_box(linear_cpu(&x, &w, m, n, k, dot));
            });
            println!("  | engine: {us:.0}");
        }
    }

    println!("  A/B, interleaved, median of 15 rounds: the earlier row partition | the engine | ratio");
    for v in host.cpu_variants.iter().copied() {
        let dot = v.dot().expect("an available variant has a dot kernel");
        for &(m, n, k, label) in &shapes {
            let (x, w) = (codes(m * k, 1 + m as u64), codes(n * k, 2 + n as u64));
            assert_eq!(rows_partition(&x, &w, (m, n, k), dot, cores), linear_cpu(&x, &w, m, n, k, dot));
            let (mut old, mut new) = (Vec::new(), Vec::new());
            for _ in 0..15 {
                old.push(time(|| {
                    std::hint::black_box(rows_partition(&x, &w, (m, n, k), dot, cores));
                }));
                new.push(time(|| {
                    std::hint::black_box(linear_cpu(&x, &w, m, n, k, dot));
                }));
            }
            let (o, e) = (median(old), median(new));
            println!("  {:<8} {label:<28} rows {o:>9.0}  engine {e:>9.0}  ratio {:.2}", v.name(), o / e);
        }
    }
}
