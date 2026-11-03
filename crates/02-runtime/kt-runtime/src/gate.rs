//! The cross-backend equality gate: run the linear accumulate through every backend on one fixed corpus and require
//! identical bytes. The corpus is deterministic and covers:
//! - tails no SIMD width divides (`k` = 1, 7, 33, 1023, 4099)
//! - the extreme codes ±`WLEVELS`
//! - a long dot product
//! - many rows and outputs, so the threaded fill is exercised

use kt_core::backend::{LinearBackend, WeightId, Weights};
use kt_core::layers::WLEVELS;
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// One linear-accumulate case: `x` is `rows × k`, `w` is `out × k`. `id` names the weights, so a backend that keeps
/// weights resident sees the same matrix again on a repeat.
pub struct Case {
    pub label: &'static str,
    pub id: WeightId,
    pub rows: usize,
    pub out: usize,
    pub k: usize,
    pub x: Vec<i32>,
    pub w: Vec<i32>,
}

/// The fixed corpus. Same bytes on every machine (a seeded LCG, no float).
pub fn corpus() -> Vec<Case> {
    let lv = WLEVELS as i32;
    let mut s = 0x5032_0b00_u64;
    let mut uniform = |n: usize| -> Vec<i32> {
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) % (2 * WLEVELS as u64 + 1)) as i32 - lv
            })
            .collect()
    };
    let mut cases = Vec::new();
    for (label, rows, out, k) in [
        ("k=1", 1, 1, 1),
        ("odd tail", 3, 5, 7),
        ("k=33", 2, 17, 33),
        ("k=1023", 4, 64, 1023),
        ("k=4099 (long)", 1, 9, 4099),
        ("threaded 8×512×2048", 8, 512, 2048),
    ] {
        cases.push(Case { label, id: WeightId::fresh(), rows, out, k, x: uniform(rows * k), w: uniform(out * k) });
    }
    // extremes: every product is +WLEVELS² or −WLEVELS², the largest magnitudes the engine produces
    let k = 1027;
    let x: Vec<i32> = (0..2 * k).map(|i| if i % 3 == 0 { -lv } else { lv }).collect();
    let w: Vec<i32> = (0..3 * k).map(|i| if i % 2 == 0 { lv } else { -lv }).collect();
    cases.push(Case { label: "extremes ±WLEVELS", id: WeightId::fresh(), rows: 2, out: 3, k, x, w });
    let x = vec![lv; 4 * 2048];
    cases.push(Case { label: "all +WLEVELS", id: WeightId::fresh(), rows: 4, out: 4, k: 2048, x: x.clone(), w: x });
    cases
}

/// The bound case of the digit backend: the smallest `k` beyond one `int32` flush, every code +`WLEVELS`,
/// against one weight row of +`WLEVELS` and one of −`WLEVELS`. An accumulator that is not flushed at its bound wraps
/// here; the answer is `±k · WLEVELS²`. It is not part of the corpus, so it leaves the corpus hash unchanged.
pub fn flush_case() -> Case {
    let k = (kt_cuda::digits::KT_FLUSH + 1) * kt_cuda::digits::TILE;
    let lv = WLEVELS as i32;
    let w = [vec![lv; k], vec![-lv; k]].concat();
    Case { label: "k beyond one int32 flush", id: WeightId::fresh(), rows: 1, out: 2, k, x: vec![lv; k], w }
}

/// One backend's result on the corpus.
pub struct GateReport {
    pub backend: String,
    /// SHA-256 over every accumulator of every case, in order (little-endian `i64`).
    pub hash: String,
    /// Labels of the cases whose accumulators differ from the reference (the first backend).
    pub diverged: Vec<&'static str>,
}

/// Run every backend over the corpus. The first backend is the reference. `Ok` if all are byte-identical; `Err`
/// carries every report, so a caller can name the divergent backend and the cases it got wrong.
pub fn equality_gate(backends: &[Arc<dyn LinearBackend>], cases: &[Case]) -> Result<Vec<GateReport>, Vec<GateReport>> {
    assert!(!backends.is_empty(), "equality_gate: no backend");
    let run = |b: &Arc<dyn LinearBackend>| -> Vec<Vec<i64>> {
        cases.iter().map(|c| b.linear_weights(&c.x, Weights { id: c.id, codes: &c.w }, c.rows, c.out, c.k)).collect()
    };
    let reference = run(&backends[0]);
    let mut reports = Vec::new();
    for b in backends {
        let got = run(b);
        let mut h = Sha256::new();
        for v in got.iter().flatten() {
            h.update(v.to_le_bytes());
        }
        let diverged = cases.iter().zip(got.iter().zip(&reference)).filter(|(_, (g, r))| g != r).map(|(c, _)| c.label);
        reports.push(GateReport {
            backend: b.name(),
            hash: h.finalize().iter().map(|x| format!("{x:02x}")).collect(),
            diverged: diverged.collect(),
        });
    }
    if reports.iter().all(|r| r.diverged.is_empty()) {
        Ok(reports)
    } else {
        Err(reports)
    }
}

/// The residency check for one backend: the corpus twice over (the second pass reuses every resident weight),
/// then two weight matrices of one shape interleaved four times. Every answer must equal the engine's own kernel.
/// `Err` names the first case that differs.
pub fn residency_gate(b: &Arc<dyn LinearBackend>, cases: &[Case]) -> Result<(), String> {
    let reference = kt_core::backend::Portable;
    for pass in 1..=2 {
        for c in cases {
            let w = Weights { id: c.id, codes: &c.w };
            if b.linear_weights(&c.x, w, c.rows, c.out, c.k) != reference.linear(&c.x, &c.w, c.rows, c.out, c.k) {
                return Err(format!("{}: pass {pass}, case '{}' differs", b.name(), c.label));
            }
        }
    }
    let twin = |seed: u64| -> Vec<i32> {
        let mut s = seed;
        (0..64 * 1023)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) % (2 * WLEVELS as u64 + 1)) as i32 - WLEVELS as i32
            })
            .collect()
    };
    let x: Vec<i32> =
        cases.iter().find(|c| c.label == "k=1023").map_or_else(|| twin(7)[..4 * 1023].to_vec(), |c| c.x.clone());
    let pair = [(WeightId::fresh(), twin(11)), (WeightId::fresh(), twin(12))];
    for round in 0..4 {
        for (i, (id, w)) in pair.iter().enumerate() {
            if b.linear_weights(&x, Weights { id: *id, codes: w }, 4, 64, 1023) != reference.linear(&x, w, 4, 64, 1023)
            {
                return Err(format!("{}: twin {i} of one shape differs in round {round}", b.name()));
            }
        }
    }
    Ok(())
}
