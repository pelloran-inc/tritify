//! Exactness of the GPU kernels against the CPU's plain integer sum. On a host without a GPU adapter each test says
//! so and checks nothing, so no GPU claim can come from such a host.

use super::*;

const LV: i32 = 265_720; // WLEVELS, the largest code the engine produces

fn cpu(x: &[i32], w: &[i32], m: usize, n: usize, k: usize) -> Vec<i64> {
    (0..m * n).map(|i| (0..k).map(|t| x[(i / n) * k + t] as i64 * w[(i % n) * k + t] as i64).sum()).collect()
}

fn codes(n: usize, mut s: u64) -> Vec<i32> {
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % (2 * LV as u64 + 1)) as i32 - LV
        })
        .collect()
}

fn gpu_or_skip(test: &str) -> Option<Gpu> {
    let g = Gpu::open(None);
    match &g {
        Some(g) => println!("{test}: adapter {} / {} ({})", g.info.backend, g.info.name, g.info.device_type),
        None => println!("{test}: no GPU adapter on this host — not executed"),
    }
    g
}

#[test]
fn gemm_i64_is_exact_on_full_range_codes() {
    let Some(g) = gpu_or_skip("gemm_i64_is_exact_on_full_range_codes") else { return };
    for (m, n, k, seed) in [(1, 1, 1, 1), (3, 5, 7, 2), (4, 64, 1023, 3), (2, 9, 4099, 4)] {
        let (x, w) = (codes(m * k, seed), codes(n * k, seed + 100));
        assert_eq!(g.gemm_i64(&x, &w, m, n, k), cpu(&x, &w, m, n, k), "m={m} n={n} k={k}");
    }
    // every product ±WLEVELS²: the largest magnitudes, carries through both 32-bit words, both signs
    let k = 2048;
    let x: Vec<i32> = (0..2 * k).map(|i| if i % 3 == 0 { -LV } else { LV }).collect();
    let w = vec![LV; 3 * k];
    assert_eq!(g.gemm_i64(&x, &w, 2, 3, k), cpu(&x, &w, 2, 3, k));
}

#[test]
fn gemm_i64_splits_a_problem_larger_than_one_dispatch() {
    let Some(g) = gpu_or_skip("gemm_i64_splits_a_problem_larger_than_one_dispatch") else { return };
    // 8 × 600,000 outputs is more than one dispatch (65,535 groups × 64), so the rows are split.
    let (m, n, k) = (8, 600_000, 1);
    let (x, w) = (codes(m * k, 7), codes(n * k, 8));
    assert_eq!(g.gemm_i64(&x, &w, m, n, k), cpu(&x, &w, m, n, k));
}

#[test]
fn gemm_i32_tiled_is_exact_while_sums_fit_i32_and_wrong_beyond() {
    let Some(g) = gpu_or_skip("gemm_i32_tiled_is_exact_while_sums_fit_i32_and_wrong_beyond") else { return };
    let (m, n, k) = (17, 33, 65);
    let small: Vec<i32> = codes(m * k, 9).iter().map(|v| v % 128).collect();
    let tern: Vec<i32> = codes(n * k, 10).iter().map(|v| v % 2).collect();
    let got: Vec<i64> = g.gemm_i32_tiled(&small, &tern, m, n, k).into_iter().map(i64::from).collect();
    assert_eq!(got, cpu(&small, &tern, m, n, k), "fits i32 → exact");
    let (x, w) = (codes(m * k, 11), codes(n * k, 12));
    let got: Vec<i64> = g.gemm_i32_tiled(&x, &w, m, n, k).into_iter().map(i64::from).collect();
    assert_ne!(got, cpu(&x, &w, m, n, k), "engine-sized codes overflow i32 → this kernel must not serve the engine");
}

#[test]
fn split_k_equals_one_thread_on_full_range_codes() {
    let Some(g) = gpu_or_skip("split_k_equals_one_thread_on_full_range_codes") else { return };
    for (m, n, k, seed) in
        [(1, 1, 1, 21), (1, 7, 63, 22), (1, 9, 64, 23), (1, 300, 897, 24), (3, 17, 4099, 25), (8, 64, 2048, 26)]
    {
        let (x, w) = (codes(m * k, seed), codes(n * k, seed + 100));
        let want = cpu(&x, &w, m, n, k);
        assert_eq!(g.gemm_i64_with(&x, &w, m, n, k, Kernel::SplitK), want, "split-k m={m} n={n} k={k}");
        assert_eq!(g.gemm_i64_with(&x, &w, m, n, k, Kernel::OneThread), want, "one-thread m={m} n={n} k={k}");
    }
}

#[test]
fn resident_weights_give_the_per_call_answer_through_repeats_twins_and_eviction() {
    use kt_core::backend::{LinearBackend, WeightId, Weights};
    let Some(gpu) = gpu_or_skip("resident_weights_give_the_per_call_answer") else { return };
    let (m, n, k) = (2, 33, 257);
    // two weight matrices of one shape (a shape-keyed cache would confuse them), one of another shape
    let ws = [
        (WeightId::fresh(), codes(n * k, 31)),
        (WeightId::fresh(), codes(n * k, 32)),
        (WeightId::fresh(), codes(9 * k, 33)),
    ];
    let x = codes(m * k, 34);
    let want: Vec<Vec<i64>> = ws.iter().map(|(_, w)| cpu(&x, w, m, w.len() / k, k)).collect();
    let check = |b: &GpuLinear, label: &str| {
        for round in 0..3 {
            for (i, (id, w)) in ws.iter().enumerate() {
                let got = b.linear_weights(&x, Weights { id: *id, codes: w }, m, w.len() / k, k);
                assert_eq!(got, want[i], "{label}: round {round}, weights {i}");
            }
        }
    };
    let resident = GpuLinear::new(gpu);
    check(&resident, "resident");
    assert_eq!(resident.resident_bytes(), (ws.iter().map(|(_, w)| w.len()).sum::<usize>() * 4) as u64);
    let tiny = GpuLinear::new(Gpu::open(None).unwrap()).with_budget(1);
    check(&tiny, "budget 1 B (every call evicts)");
    assert!(tiny.resident_bytes() <= (n * k * 4) as u64, "at most the one matrix in use stays");
}
