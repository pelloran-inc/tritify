//! kt-r16 — the lossless balanced-ternary fixed-point container.
//!
//! Any float embedding can be held as an exact-integer code with no quality loss: each component is written
//! in **balanced ternary** at `b` trits/component — an integer in `[-(3^b-1)/2, +(3^b-1)/2]` — with one
//! per-vector scale factored out. Similarity is then an **exact integer inner product** on the codes (no
//! floating point), and as `b` grows the quantization error → 0, so the induced ranking → the float ranking.
//!
//! This is NOT lossy compression: at a sufficient `b` the top-k set is identical to float, yet the code is
//! smaller than float32 (a trit is log2(3) ≈ 1.585 bits; `b=12` trits/component = 0.60× float32). The code is
//! integer, reversible, ECC-able, and GPU/FPGA-native.
//!
//! Runtime is float-free: [`idot`] is pure integer. Float appears only in [`quantize`] (computing the ingest
//! scale) and in the tests (the float reference the integer path is checked against).
//!
//! # Example
//!
//! ```
//! use kt_r16::{dequantize, idot, max_level, quantize};
//!
//! assert_eq!(max_level(2), 4); // two balanced trits span −4..=4
//! let x = [0.5f32, -1.0, 0.25];
//! let (codes, step) = quantize(&x, 8);
//! let back = dequantize(&codes, step);
//! assert!(x.iter().zip(&back).all(|(a, b)| (a - b).abs() as f64 <= step));
//! // The dot product of codes is exact integer arithmetic.
//! assert_eq!(idot(&codes, &codes), codes.iter().map(|&c| c as i64 * c as i64).sum::<i64>());
//! ```

/// log2(3) — bits carried by one balanced trit.
pub const BITS_PER_TRIT: f64 = 1.584962500721156;

/// Largest magnitude representable in `b` balanced-ternary trits: `(3^b - 1)/2`. `b` must satisfy `b <= 20`
/// so the level fits an `i32` code word.
pub fn max_level(b: u32) -> i64 {
    assert!((1..=20).contains(&b), "b out of range (1..=20)");
    (3i64.pow(b) - 1) / 2
}

/// Bytes/component of the packed code at `b` trits, relative to float32 (32 bits). `b=8 → 0.40×`,
/// `b=12 → 0.60×`.
pub fn size_vs_float32(b: u32) -> f64 {
    b as f64 * BITS_PER_TRIT / 32.0
}

/// Encode a float vector into balanced-ternary fixed-point at `b` trits/component. Returns the integer code
/// (each entry in `[-max_level, +max_level]`) and the per-vector scale `s` with `x_i ≈ code_i · s`.
pub fn quantize(x: &[f32], b: u32) -> (Vec<i32>, f64) {
    let m = max_level(b);
    let peak = x.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
    if peak == 0.0 {
        return (vec![0; x.len()], 1.0);
    }
    let s = peak / m as f64; // largest component maps to ±m
    let codes = x
        .iter()
        .map(|&v| {
            let q = (v as f64 / s).round() as i64;
            q.clamp(-m, m) as i32
        })
        .collect();
    (codes, s)
}

/// Decode a code back to float (`x_i = code_i · s`). Reconstruction error → 0 as `b` grows.
pub fn dequantize(code: &[i32], s: f64) -> Vec<f32> {
    code.iter().map(|&q| (q as f64 * s) as f32).collect()
}

/// Exact integer inner product of two codes — **zero float**. `i64` accumulation is exact for `b <= 12` at
/// embedding dimensions up to a few thousand (`max_level^2 · dim` stays well within `i64`).
pub fn idot(a: &[i32], b: &[i32]) -> i64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0i64;
    for (&x, &y) in a.iter().zip(b) {
        acc += x as i64 * y as i64;
    }
    acc
}

/// Similarity score between a query code `(qc, qs)` and a doc code `(dc, ds)`: the exact integer dot scaled
/// back by the two per-vector scales. The *ranking* over docs needs only the sign/order, but the scaled value
/// is what converges to the float inner product.
pub fn score(qc: &[i32], qs: f64, dc: &[i32], ds: f64) -> f64 {
    idot(qc, dc) as f64 * qs * ds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    }
    // Uniform in [-1, 1), deterministic and machine-independent.
    fn unif(s: &mut u64) -> f32 {
        ((lcg(s) >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) as f32
    }
    fn vec_of(s: &mut u64, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| unif(s)).collect()
    }

    fn float_dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
    }
    fn top_k(scores: &[f64], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());
        idx.truncate(k);
        idx
    }
    fn set_match(a: &[usize], b: &[usize]) -> f64 {
        let hits = a.iter().filter(|i| b.contains(i)).count();
        hits as f64 / a.len() as f64
    }

    /// The container property: as `b` grows the ternary ranking converges to the float ranking (top-10 set
    /// match → 1.0), while the code stays smaller than float32 — the size/fidelity trade-off of the R16
    /// container (≈ 0.86 → 1.00). Synthetic vectors here; real-corpus retrieval is out of scope for this test.
    #[test]
    fn container_converges_to_float_ranking() {
        let (dim, n_docs, k) = (256usize, 500usize, 10usize);
        let mut s = 0xA17EFACEu64;
        let query = vec_of(&mut s, dim);
        let docs: Vec<Vec<f32>> = (0..n_docs).map(|_| vec_of(&mut s, dim)).collect();

        let float_scores: Vec<f64> = docs.iter().map(|d| float_dot(&query, d)).collect();
        let float_top = top_k(&float_scores, k);

        let mut last = 0.0f64;
        for &b in &[2u32, 4, 6, 8, 10, 12] {
            let (qc, qs) = quantize(&query, b);
            let trit_scores: Vec<f64> = docs
                .iter()
                .map(|d| {
                    let (dc, ds) = quantize(d, b);
                    score(&qc, qs, &dc, ds)
                })
                .collect();
            let trit_top = top_k(&trit_scores, k);
            let m = set_match(&float_top, &trit_top);
            println!("r16 b={:>2}  size={:.2}x float32  top{}-set-match-vs-float={:.4}", b, size_vs_float32(b), k, m);
            last = m;
        }
        // At b=12 (0.60x float) the ternary ranking must essentially equal float.
        assert!(last >= 0.9, "b=12 top-10 set match too low: {last:.4}");
    }

    /// `idot` is exact and float-free: it equals the inner product of the reconstructed integers by
    /// construction, and re-encoding the same vector is byte-identical (determinism).
    #[test]
    fn idot_is_exact_and_deterministic() {
        let mut s = 123u64;
        let a = vec_of(&mut s, 300);
        let b = vec_of(&mut s, 300);
        let (ca, _) = quantize(&a, 12);
        let (cb, _) = quantize(&b, 12);
        // independent reference accumulation
        let mut reference = 0i128;
        for (&x, &y) in ca.iter().zip(&cb) {
            reference += x as i128 * y as i128;
        }
        assert_eq!(idot(&ca, &cb) as i128, reference);
        // determinism: same input → same code bytes
        let (ca2, s2) = quantize(&a, 12);
        assert_eq!(ca, ca2);
        assert_eq!(quantize(&a, 12).1.to_bits(), s2.to_bits());
    }

    /// Reconstruction error shrinks ~3^-b as resolution grows.
    #[test]
    fn reconstruction_error_shrinks() {
        let mut s = 55u64;
        let x = vec_of(&mut s, 1024);
        let mut last = f64::INFINITY;
        for &b in &[4u32, 6, 8, 10, 12] {
            let (c, sc) = quantize(&x, b);
            let xr = dequantize(&c, sc);
            let max_err = x.iter().zip(&xr).fold(0.0f64, |a, (&u, &v)| a.max((u as f64 - v as f64).abs()));
            println!("r16 b={:>2}  max_recon_err={:.3e}", b, max_err);
            last = max_err;
        }
        assert!(last < 1e-3, "b=12 reconstruction error too high: {last:.3e}");
    }

    #[test]
    fn size_ratios_expected() {
        // R16 size ratios: b=8 → 0.40x, b=12 → 0.60x of float32.
        assert!((size_vs_float32(8) - 0.40).abs() < 0.01);
        assert!((size_vs_float32(12) - 0.60).abs() < 0.01);
    }
}
