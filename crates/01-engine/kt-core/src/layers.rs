//! The generic op library.
//!
//! Matmul accumulation is exact integer; its operands are the b=12 weights and the activation dynamically
//! quantized per-tensor by its own max (`round(x·M/max|x|)`, scale-invariant, no calibration). Scales are the
//! integer [`Scale`] (`mant·2^(-shift)`), composed with integer arithmetic — no float on the matmul/requant
//! data path. The whole `forward_int` path is integer: normalization statistics and the inverse-RMS/std
//! ([`kt_int::Scale::rsqrt`]), the nonlinearity lookup tables, and scale compose/compare — float appears
//! only in the `forward_ref` twins and in offline (weight / eps) quantization, as in a pure-integer core.

use crate::module::Module;
use crate::router::{RouterScore, SigmoidScore};
use crate::tensor::IntTensor;
use kt_int::{CodeLut, Scale};

/// Balanced-ternary b=12 level — the matmul operand grid (weights and dynamically-quantized activations).
pub const WLEVELS: i64 = (3i64.pow(12) - 1) / 2;
/// Activation interchange level (~28-bit) — the precision activations are stored at between ops.
pub const ACT_LEVELS: i64 = 1 << 28;

/// `round(data · WLEVELS / xmax)` elementwise — scale-invariant dynamic quant of the matmul input. Integer.
/// Output codes are in `[-WLEVELS, WLEVELS]` (b=12), which fits `i32` — kept as `i32` so the matmul dot is an
/// `i32 × i32 → i64` widening multiply-accumulate that auto-vectorizes (NEON `smlal` / AVX-512), bit-identically.
pub(crate) fn quant_to(data: &[i32], xmax: i64) -> Vec<i32> {
    let half = xmax / 2;
    data.iter()
        .map(|&d| {
            let num = d as i64 * WLEVELS;
            (if num >= 0 { (num + half) / xmax } else { -((-num + half) / xmax) }) as i32
        })
        .collect()
}

/// `round(v · num_level / den)` — dynamic requant of an integer to a target level. Integer, round-half-away.
pub fn requant_dyn(v: i64, num_level: i64, den: i64) -> i32 {
    let prod = v as i128 * num_level as i128;
    let half = den as i128 / 2;
    let r = if prod >= 0 { (prod + half) / den as i128 } else { -(((-prod) + half) / den as i128) };
    r as i32
}

/// Process-global cap on worker threads for the deterministic parallel fills (the linear layer's partition,
/// [`par_blocks`] and the attention cores). `0` = auto (all cores reported by [`std::thread::available_parallelism`]); `1` = single-threaded
/// (serial); `n` = at most `n` workers. The output is **bit-identical for any value** — the thread count changes
/// only speed, never a single output bit — so this is purely a performance / scheduling knob (benchmarking,
/// pinning, sharing a host, edge devices with few cores). Set it from code with [`set_max_threads`]; if never set
/// explicitly it is seeded once from the `TRITIFY_THREADS` environment variable (absent ⇒ auto). `usize::MAX` is
/// the sentinel for "not yet initialized".
static MAX_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Set the worker-thread cap for the deterministic parallel paths (`0` = auto, `1` = serial, `n` = at most `n`).
/// Takes effect on the next op. Does not affect output bits — only speed.
pub fn set_max_threads(n: usize) {
    MAX_THREADS.store(n, std::sync::atomic::Ordering::Relaxed);
}

/// The configured worker-thread cap (`0` = auto). On first use, if never set via [`set_max_threads`], it is
/// seeded from the `TRITIFY_THREADS` environment variable (unparseable or absent ⇒ auto).
pub fn max_threads() -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    let v = MAX_THREADS.load(Relaxed);
    if v != usize::MAX {
        return v;
    }
    let seeded = std::env::var("TRITIFY_THREADS").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0);
    // Benign race: concurrent first-callers all compute the same env-derived value.
    let _ = MAX_THREADS.compare_exchange(usize::MAX, seeded, Relaxed, Relaxed);
    MAX_THREADS.load(Relaxed)
}

/// Worker-thread count to use for `nblocks` independent blocks, honoring the configured cap ([`max_threads`]):
/// auto uses every available core, an explicit cap limits it, and it never exceeds `nblocks` (more workers than
/// blocks is pointless). Always ≥ 1.
pub(crate) fn resolve_threads(nblocks: usize) -> usize {
    let cap = max_threads();
    let want = if cap == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { cap };
    want.max(1).min(nblocks.max(1))
}

/// Deterministic parallel fill of a flat integer buffer by independent contiguous blocks. `buf` holds `nblocks`
/// contiguous blocks of `block` elements (`buf.len() == block·nblocks`); `fill(k, block_k)` writes block `k`,
/// reading only shared state. Because the blocks are disjoint and each one's contents depend solely on `k` and
/// shared inputs (never on another block), the result is **order-independent and bit-identical** to the serial
/// loop — this holds for integer arithmetic, not for float. Threads are used only when `work` (an estimate of
/// the total scalar multiply-accumulates) is large enough to amortize spawning; below that it is the plain
/// serial loop. `std`-only ([`std::thread::scope`]), no third-party dependency. Used by the convolutions
/// ([`crate::conv`]); the attention cores ([`crate::attention`], [`crate::sdpa`]) split their heads the same way, and
/// the linear layer has its own partition by element (`backend::linear_cpu`).
pub(crate) fn par_blocks<F>(buf: &mut [i64], block: usize, nblocks: usize, work: usize, fill: F)
where
    F: Fn(usize, &mut [i64]) + Sync,
{
    debug_assert_eq!(buf.len(), block.saturating_mul(nblocks), "par_blocks: buffer is not block·nblocks");
    if block == 0 {
        return; // degenerate (empty output); chunks_mut(0) would panic
    }
    let nthreads = resolve_threads(nblocks);
    // Output blocks are independent, so they parallelize deterministically (disjoint writes → bit-identical,
    // order-independent). Spread across cores only when the work is large enough to amortize spawning.
    if nthreads > 1 && work >= (1 << 20) {
        let chunk = nblocks.div_ceil(nthreads);
        std::thread::scope(|s| {
            for (ci, group) in buf.chunks_mut(chunk * block).enumerate() {
                let (base, fill) = (ci * chunk, &fill);
                s.spawn(move || {
                    for (bi, b) in group.chunks_mut(block).enumerate() {
                        fill(base + bi, b);
                    }
                });
            }
        });
    } else {
        for (bi, b) in buf.chunks_mut(block).enumerate() {
            fill(bi, b);
        }
    }
}

/// Elementwise add of two activations at (possibly) different scales: bring both to a common `Scale`, add in
/// integer, dynamic-quant the result. This is the residual-stream add and the positional-embedding add — no
/// float on the data path. (Shapes must match elementwise.)
pub fn add(a: &IntTensor, b: &IntTensor) -> IntTensor {
    assert_eq!(a.len(), b.len(), "add: length mismatch");
    let common = if a.scale.ge(b.scale) { a.scale } else { b.scale };
    if common.mant == 0 {
        return IntTensor::new(vec![0; a.len()], a.shape.clone(), Scale::ZERO);
    }
    let cinv = common.recip();
    let (fa, fb) = (a.scale.mul(cinv), b.scale.mul(cinv));
    let sum: Vec<i64> = (0..a.len()).map(|i| fa.apply(a.data[i] as i64) + fb.apply(b.data[i] as i64)).collect();
    let gmax = sum.iter().fold(0i64, |m, &z| m.max(z.abs()));
    if gmax == 0 {
        return IntTensor::new(vec![0; a.len()], a.shape.clone(), Scale::ZERO);
    }
    let out: Vec<i32> = sum.iter().map(|&z| requant_dyn(z, ACT_LEVELS, gmax)).collect();
    IntTensor::new(out, a.shape.clone(), Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(common))
}

/// Integer matmul of two **activation** tensors `a[m,k] · b[k,n] → [m,n]` (both dynamically quantized, unlike
/// [`QuantLinear`] which has a fixed weight). Exact integer accumulation (`i128`, so a large shared dim can't
/// overflow), output real = `accum · a.scale · b.scale`, dynamic-requantized to the activation grid. SAM's mask
/// head (`hyper_in @ upscaled_embedding`) is one of these.
pub fn matmul(a: &IntTensor, b: &IntTensor) -> IntTensor {
    assert_eq!(a.shape.len(), 2, "matmul: a must be [m,k]");
    assert_eq!(b.shape.len(), 2, "matmul: b must be [k,n]");
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    assert_eq!(k, k2, "matmul: inner dim mismatch");
    let mut acc = vec![0i128; m * n];
    for i in 0..m {
        for l in 0..k {
            let av = a.data[i * k + l] as i128;
            if av == 0 {
                continue;
            }
            let brow = &b.data[l * n..(l + 1) * n];
            let arow = &mut acc[i * n..(i + 1) * n];
            for (o, &bv) in arow.iter_mut().zip(brow.iter()) {
                *o += av * bv as i128;
            }
        }
    }
    let gmax = acc.iter().fold(0i128, |mx, &z| mx.max(z.abs()));
    let c = a.scale.mul(b.scale);
    if gmax == 0 || c.mant == 0 {
        return IntTensor::new(vec![0; m * n], vec![m, n], Scale::ZERO);
    }
    let num = ACT_LEVELS as i128;
    let half = gmax / 2;
    let out: Vec<i32> = acc
        .iter()
        .map(|&z| {
            let prod = z * num;
            (if prod >= 0 { (prod + half) / gmax } else { -(((-prod) + half) / gmax) }) as i32
        })
        .collect();
    IntTensor::new(out, vec![m, n], Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(c))
}

/// Concatenate row-blocks `[r_i, cols]` (all sharing `cols`) into one `[Σ r_i, cols]` tensor, bringing them to a
/// common `Scale` and dynamic-requantizing — the integer "stack these independently-scaled pieces into one
/// activation" op (e.g. SAM's per-window attention outputs → one feature map before the output projection).
pub fn concat_rows(parts: &[IntTensor]) -> IntTensor {
    assert!(!parts.is_empty(), "concat_rows: no parts");
    let cols = *parts[0].shape.last().expect("concat_rows: part has no shape");
    let total: usize = parts.iter().map(|p| p.len()).sum();
    let rows = total / cols;
    let common = parts.iter().map(|p| p.scale).fold(Scale::ZERO, |a, b| if b.ge(a) { b } else { a });
    if common.mant == 0 {
        return IntTensor::new(vec![0; total], vec![rows, cols], Scale::ZERO);
    }
    let cinv = common.recip();
    let mut buf = vec![0i64; total];
    let mut off = 0usize;
    for p in parts {
        let f = p.scale.mul(cinv);
        for &d in &p.data {
            buf[off] = f.apply(d as i64);
            off += 1;
        }
    }
    let gmax = buf.iter().fold(0i64, |m, &z| m.max(z.abs()));
    if gmax == 0 {
        return IntTensor::new(vec![0; total], vec![rows, cols], Scale::ZERO);
    }
    let out: Vec<i32> = buf.iter().map(|&z| requant_dyn(z, ACT_LEVELS, gmax)).collect();
    IntTensor::new(out, vec![rows, cols], Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(common))
}

/// A quantized linear layer `y = W·x (+ bias)` — exact integer accumulation, dynamic per-tensor quant, integer
/// `Scale` throughout the data path.
#[derive(Clone)]
pub struct QuantLinear {
    pub name: String,
    pub out_features: usize,
    pub in_features: usize,
    /// Fixed at construction: [`QuantLinear::w_codes`] reads them, nothing writes them, so `wid` always names them.
    w_codes: Vec<i32>,
    /// The weights' cache key for a backend that keeps them resident (never part of a hash).
    wid: crate::backend::WeightId,
    pub w_scale: Scale,
    bias: Option<(Vec<i64>, Scale)>, // codes + scale (real bias = code · scale)
}

impl QuantLinear {
    pub fn from_weights(name: impl Into<String>, w: &[f32], out_features: usize, in_features: usize) -> Self {
        assert_eq!(w.len(), out_features * in_features);
        let peak = w.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
        let (w_scale, w_codes) = if peak == 0.0 {
            (Scale::ZERO, vec![0i32; w.len()])
        } else {
            let step = peak / WLEVELS as f64;
            let codes = w.iter().map(|&v| ((v as f64 / step).round() as i64).clamp(-WLEVELS, WLEVELS) as i32).collect();
            (Scale::from_f64(step), codes)
        };
        QuantLinear {
            name: name.into(),
            out_features,
            in_features,
            w_codes,
            wid: crate::backend::WeightId::fresh(),
            w_scale,
            bias: None,
        }
    }

    pub fn with_bias(mut self, bias: Option<&[f32]>) -> Self {
        if let Some(b) = bias {
            assert_eq!(b.len(), self.out_features, "{}: bias len mismatch", self.name);
            let bmax = b.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
            if bmax > 0.0 {
                let step = bmax / ACT_LEVELS as f64;
                let codes = b.iter().map(|&v| (v as f64 / step).round() as i64).collect();
                self.bias = Some((codes, Scale::from_f64(step)));
            }
        }
        self
    }

    /// The weight codes, `out_features × in_features`, row-major.
    pub fn w_codes(&self) -> &[i32] {
        &self.w_codes
    }

    #[inline]
    fn row(&self, o: usize) -> &[i32] {
        &self.w_codes[o * self.in_features..(o + 1) * self.in_features]
    }
}

impl Module for QuantLinear {
    fn name(&self) -> &str {
        &self.name
    }

    fn forward_int(&self, x: &IntTensor) -> IntTensor {
        assert_eq!(x.len() % self.in_features, 0, "{}: input not a multiple of in_features", self.name);
        let rows = x.len() / self.in_features;
        let shape = if rows == 1 { vec![self.out_features] } else { vec![rows, self.out_features] };
        let xmax = x.max_abs_data();

        // input all zero → output is the (broadcast) bias, or zero.
        if xmax == 0 {
            return match &self.bias {
                Some((codes, bscale)) => {
                    let mut data = Vec::with_capacity(rows * self.out_features);
                    for _ in 0..rows {
                        for &c in codes {
                            data.push(c as i32);
                        }
                    }
                    IntTensor::new(data, shape, *bscale)
                }
                None => IntTensor::new(vec![0; rows * self.out_features], shape, Scale::ZERO),
            };
        }

        let qx = quant_to(&x.data, xmax); // b=12 operands, whole tensor
        let s_qx = Scale::from_ratio(xmax as u128, WLEVELS as u128).mul(x.scale);
        let c = s_qx.mul(self.w_scale); // output real = accum · c (+ bias)
        if c.mant == 0 {
            return IntTensor::new(vec![0; rows * self.out_features], shape, Scale::ZERO);
        }
        // bias expressed in units of c (integer): bias_real / c = (bias_scale · c⁻¹) · bias_code
        let bias_in_c: Option<Vec<i64>> = self.bias.as_ref().map(|(codes, bscale)| {
            let f = bscale.mul(c.recip());
            codes.iter().map(|&bc| f.apply(bc)).collect()
        });

        // The exact i32 × i32 → i64 accumulate runs on the installed compute backend (the engine's portable kernel
        // unless the runtime layer installed another); every backend returns the same integers.
        let out_f = self.out_features;
        let w = crate::backend::Weights { id: self.wid, codes: &self.w_codes };
        let mut combined = crate::backend::backend().linear_weights(&qx, w, rows, out_f, self.in_features);
        if let Some(b) = &bias_in_c {
            for row in combined.chunks_mut(out_f) {
                for (v, &bc) in row.iter_mut().zip(b) {
                    *v += bc;
                }
            }
        }
        let ymax = combined.iter().fold(0i64, |a, &v| a.max(v.abs()));
        if ymax == 0 {
            return IntTensor::new(vec![0; rows * self.out_features], shape, Scale::ZERO);
        }
        let out_data: Vec<i32> = combined.iter().map(|&v| requant_dyn(v, ACT_LEVELS, ymax)).collect();
        let out_scale = Scale::from_ratio(ymax as u128, ACT_LEVELS as u128).mul(c);
        IntTensor::new(out_data, shape, out_scale)
    }

    fn forward_ref(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len() % self.in_features, 0);
        let rows = x.len() / self.in_features;
        let ws = self.w_scale.to_f64();
        let bias_real: Option<Vec<f64>> = self.bias.as_ref().map(|(codes, bs)| {
            let s = bs.to_f64();
            codes.iter().map(|&c| c as f64 * s).collect()
        });
        let mut out = vec![0f32; rows * self.out_features];
        for r in 0..rows {
            let xr = &x[r * self.in_features..(r + 1) * self.in_features];
            for o in 0..self.out_features {
                let wr = self.row(o);
                let acc: f64 = (0..self.in_features).map(|i| xr[i] as f64 * (wr[i] as f64 * ws)).sum();
                let b = bias_real.as_ref().map_or(0.0, |bb| bb[o]);
                out[r * self.out_features + o] = (acc + b) as f32;
            }
        }
        out
    }
}

/// RMSNorm — `y = x / sqrt(mean(x²)+eps) · weight`, row-wise, integer data path. The per-element product
/// `data_i·wcode_i` is integer; the per-row normalization factor `C = x.scale · (1/rms) · w_scale` is an
/// integer [`Scale`] (its `1/rms` is an integer inverse square root, [`kt_int::Scale::rsqrt`], of the
/// per-row mean-square — a reduction result, not a data-path op). Rows are brought to a common `Scale` and
/// dynamically quantized.
pub struct RmsNorm {
    pub name: String,
    pub dim: usize,
    wcodes: Vec<i32>,
    w_scale: Scale,
    weight_f64: Vec<f64>, // for the float reference path
    pub eps: f64,
    eps_scale: Scale, // eps as a fixed-point scale (converted once; the only float is this offline boundary)
}

impl RmsNorm {
    pub fn new(name: impl Into<String>, weight: &[f32], eps: f64) -> Self {
        assert!(!weight.is_empty());
        let peak = weight.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
        let (w_scale, wcodes) = if peak == 0.0 {
            (Scale::ZERO, vec![0i32; weight.len()])
        } else {
            let step = peak / WLEVELS as f64;
            (
                Scale::from_f64(step),
                weight.iter().map(|&v| ((v as f64 / step).round() as i64).clamp(-WLEVELS, WLEVELS) as i32).collect(),
            )
        };
        RmsNorm {
            name: name.into(),
            dim: weight.len(),
            wcodes,
            w_scale,
            weight_f64: weight.iter().map(|&w| w as f64).collect(),
            eps,
            eps_scale: Scale::from_f64(eps),
        }
    }
}

impl Module for RmsNorm {
    fn name(&self) -> &str {
        &self.name
    }

    fn forward_int(&self, x: &IntTensor) -> IntTensor {
        assert_eq!(x.len() % self.dim, 0, "{}: input not a multiple of dim", self.name);
        let rows = x.len() / self.dim;
        let shape = if rows == 1 { vec![self.dim] } else { vec![rows, self.dim] };
        let mut p = vec![0i64; x.len()]; // data_i · wcode_i (integer)
        let mut c_rows: Vec<Scale> = Vec::with_capacity(rows);
        for r in 0..rows {
            let xr = &x.data[r * self.dim..(r + 1) * self.dim];
            let ss: i128 = xr.iter().map(|&v| v as i128 * v as i128).sum(); // exact integer Σ code²
                                                                            // mean-square (real) = (ss / dim) · x.scale² — assembled as a Scale; inverse-RMS = its rsqrt. No float.
            let mean_sq = Scale::from_ratio(ss as u128, self.dim as u128).mul(x.scale).mul(x.scale);
            let denom = mean_sq.add(self.eps_scale);
            let inv = if denom.mant == 0 { Scale::ZERO } else { denom.rsqrt() };
            c_rows.push(x.scale.mul(inv).mul(self.w_scale));
            for i in 0..self.dim {
                p[r * self.dim + i] = xr[i] as i64 * self.wcodes[i] as i64;
            }
        }
        // bring all rows to the common (largest-magnitude) scale, then dynamic-quant the whole tensor.
        let c_common = c_rows.iter().copied().fold(Scale::ZERO, |a, b| if b.ge(a) { b } else { a });
        if c_common.mant == 0 {
            return IntTensor::new(vec![0; x.len()], shape, Scale::ZERO);
        }
        let cinv = c_common.recip();
        let mut v = vec![0i64; x.len()];
        for r in 0..rows {
            let f = c_rows[r].mul(cinv);
            for i in 0..self.dim {
                v[r * self.dim + i] = f.apply(p[r * self.dim + i]);
            }
        }
        let gmax = v.iter().fold(0i64, |a, &z| a.max(z.abs()));
        if gmax == 0 {
            return IntTensor::new(vec![0; x.len()], shape, Scale::ZERO);
        }
        let out_data: Vec<i32> = v.iter().map(|&z| requant_dyn(z, ACT_LEVELS, gmax)).collect();
        IntTensor::new(out_data, shape, Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(c_common))
    }

    fn forward_ref(&self, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / self.dim;
        let mut out = vec![0f32; x.len()];
        for r in 0..rows {
            let xr = &x[r * self.dim..(r + 1) * self.dim];
            let ms: f64 = xr.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / self.dim as f64;
            let inv = 1.0 / (ms + self.eps).sqrt();
            for i in 0..self.dim {
                out[r * self.dim + i] = (xr[i] as f64 * inv * self.weight_f64[i]) as f32;
            }
        }
        out
    }
}

/// LayerNorm **with mean** — `y = (x − mean) / sqrt(var + eps) · weight + bias`, row-wise over the last `dim`.
/// SAM 2.1 / ViT use this (not RMSNorm). Same discipline as [`RmsNorm`]: the per-row reduction scalars (the mean
/// `mu` and the inverse std `inv`) are computed in integer — an O(dim) reduction, not the O(dim·rows) data path — while every
/// per-element step stays integer. Each output code (in units of the per-row scale `C = x.scale·inv·w_scale`) is
/// `d_i·wcode_i − mu_d·wcode_i (+ bias/C)`, an integer value; rows are then brought to a common `Scale` and
/// dynamically requantized, exactly as [`RmsNorm`]. The sum of squares is accumulated in `i128` (SAM's channel
/// counts × the ~28-bit activation grid overflow an `i64`).
pub struct LayerNorm {
    pub name: String,
    pub dim: usize,
    wcodes: Vec<i32>,
    w_scale: Scale,
    weight_f64: Vec<f64>,
    bias_codes: Vec<i64>,
    bias_scale: Scale,
    bias_f64: Vec<f64>,
    pub eps: f64,
    eps_scale: Scale,
}

impl LayerNorm {
    pub fn new(name: impl Into<String>, weight: &[f32], bias: &[f32], eps: f64) -> Self {
        assert!(!weight.is_empty());
        assert_eq!(weight.len(), bias.len(), "LayerNorm weight/bias dim mismatch");
        let wpeak = weight.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
        let (w_scale, wcodes) = if wpeak == 0.0 {
            (Scale::ZERO, vec![0i32; weight.len()])
        } else {
            let step = wpeak / WLEVELS as f64;
            (
                Scale::from_f64(step),
                weight.iter().map(|&v| ((v as f64 / step).round() as i64).clamp(-WLEVELS, WLEVELS) as i32).collect(),
            )
        };
        let bpeak = bias.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
        let (bias_scale, bias_codes) = if bpeak == 0.0 {
            (Scale::ZERO, vec![0i64; bias.len()])
        } else {
            let step = bpeak / ACT_LEVELS as f64;
            (Scale::from_f64(step), bias.iter().map(|&v| (v as f64 / step).round() as i64).collect())
        };
        LayerNorm {
            name: name.into(),
            dim: weight.len(),
            wcodes,
            w_scale,
            weight_f64: weight.iter().map(|&w| w as f64).collect(),
            bias_codes,
            bias_scale,
            bias_f64: bias.iter().map(|&b| b as f64).collect(),
            eps,
            eps_scale: Scale::from_f64(eps),
        }
    }
}

impl Module for LayerNorm {
    fn name(&self) -> &str {
        &self.name
    }

    fn forward_int(&self, x: &IntTensor) -> IntTensor {
        assert_eq!(x.len() % self.dim, 0, "{}: input not a multiple of dim", self.name);
        let rows = x.len() / self.dim;
        let shape = if rows == 1 { vec![self.dim] } else { vec![rows, self.dim] };
        let mut combined = vec![0i64; x.len()];
        let mut c_rows: Vec<Scale> = Vec::with_capacity(rows);
        let dim2 = (self.dim as u128) * (self.dim as u128);
        for r in 0..rows {
            let xr = &x.data[r * self.dim..(r + 1) * self.dim];
            let sum: i64 = xr.iter().map(|&v| v as i64).sum();
            let ss: i128 = xr.iter().map(|&v| v as i128 * v as i128).sum(); // exact integer Σ code²
                                                                            // variance (real) = ((dim·Σx² − (Σx)²) / dim²) · x.scale² — exact integer numerator (Cauchy–Schwarz ≥ 0)
            let var_num = (ss * self.dim as i128 - (sum as i128) * (sum as i128)).max(0) as u128;
            let var = Scale::from_ratio(var_num, dim2).mul(x.scale).mul(x.scale);
            let denom = var.add(self.eps_scale);
            let inv = if denom.mant == 0 { Scale::ZERO } else { denom.rsqrt() }; // 1/sqrt(var+eps), integer
            let c_row = self.w_scale.mul(x.scale).mul(inv);
            c_rows.push(c_row);
            // mean term (|mu| = |Σx|/dim as a Scale, carrying its sign) and the bias term, both in c_row units.
            let mu_mag = Scale::from_ratio(sum.unsigned_abs() as u128, self.dim as u128);
            let mu_sign: i64 = if sum < 0 { -1 } else { 1 };
            let bfac = self.bias_scale.mul(c_row.recip());
            for i in 0..self.dim {
                let w = self.wcodes[i] as i64;
                let main = xr[i] as i64 * w; // d_i · wcode_i
                let mean = mu_sign * mu_mag.apply(w); // mu_d · wcode_i
                let bias = bfac.apply(self.bias_codes[i]);
                combined[r * self.dim + i] = main - mean + bias;
            }
        }
        // bring all rows to the common (largest-magnitude) scale, then dynamic-quant the whole tensor.
        let c_common = c_rows.iter().copied().fold(Scale::ZERO, |a, b| if b.ge(a) { b } else { a });
        if c_common.mant == 0 {
            return IntTensor::new(vec![0; x.len()], shape, Scale::ZERO);
        }
        let cinv = c_common.recip();
        let mut v = vec![0i64; x.len()];
        for r in 0..rows {
            let f = c_rows[r].mul(cinv);
            for i in 0..self.dim {
                v[r * self.dim + i] = f.apply(combined[r * self.dim + i]);
            }
        }
        let gmax = v.iter().fold(0i64, |a, &z| a.max(z.abs()));
        if gmax == 0 {
            return IntTensor::new(vec![0; x.len()], shape, Scale::ZERO);
        }
        let out_data: Vec<i32> = v.iter().map(|&z| requant_dyn(z, ACT_LEVELS, gmax)).collect();
        IntTensor::new(out_data, shape, Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(c_common))
    }

    fn forward_ref(&self, x: &[f32]) -> Vec<f32> {
        let rows = x.len() / self.dim;
        let mut out = vec![0f32; x.len()];
        for r in 0..rows {
            let xr = &x[r * self.dim..(r + 1) * self.dim];
            let mu: f64 = xr.iter().map(|&v| v as f64).sum::<f64>() / self.dim as f64;
            let var: f64 = xr.iter().map(|&v| (v as f64 - mu) * (v as f64 - mu)).sum::<f64>() / self.dim as f64;
            let inv = 1.0 / (var + self.eps).sqrt();
            for i in 0..self.dim {
                out[r * self.dim + i] = ((xr[i] as f64 - mu) * inv * self.weight_f64[i] + self.bias_f64[i]) as f32;
            }
        }
        out
    }
}

/// SwiGLU MLP — `down( SiLU(gate(x)) ⊙ up(x) )`. Matmuls exact-integer; SiLU an **integer-code** LUT, so the
/// `silu⊙up` product is an integer multiply (`silu_code · u_data`) — zero float on the data path.
pub struct SwiGluMlp {
    pub name: String,
    pub gate: QuantLinear,
    pub up: QuantLinear,
    pub down: QuantLinear,
    silu: CodeLut,
}

impl SwiGluMlp {
    pub fn new(name: impl Into<String>, gate: QuantLinear, up: QuantLinear, down: QuantLinear) -> Self {
        assert_eq!(gate.out_features, up.out_features, "gate/up hidden mismatch");
        assert_eq!(down.in_features, gate.out_features, "down input != hidden");
        SwiGluMlp {
            name: name.into(),
            gate,
            up,
            down,
            // SiLU(x)=x·sigmoid(x); 0 on the left, identity on the right (handled in integer units below).
            silu: CodeLut::build(|x| x / (1.0 + (-x).exp()), -30.0, 30.0, 16, 20, true),
        }
    }
}

impl Module for SwiGluMlp {
    fn name(&self) -> &str {
        &self.name
    }

    fn forward_int(&self, x: &IntTensor) -> IntTensor {
        let g = self.gate.forward_int(x);
        let u = self.up.forward_int(x);
        let hidden = self.gate.out_features;
        let rows = g.len() / hidden;
        // right-tail identity factor: code = round(g_real / code_scale) = (g.scale · code_scale⁻¹)·g_data
        let id_factor = g.scale.mul(self.silu.code_scale.recip());
        let mut h = vec![0i64; g.len()];
        #[allow(clippy::needless_range_loop)] // g, u and h indexed in lockstep; the index keeps their length contract
        for k in 0..g.len() {
            let gd = g.data[k] as i64;
            let idx = self.silu.index(gd, g.scale);
            let silu_code = if idx < 0 {
                0
            } else if idx >= self.silu.n() as i64 {
                id_factor.apply(gd) // identity right tail, in code units
            } else {
                self.silu.code_at(idx)
            };
            h[k] = silu_code * u.data[k] as i64;
        }
        let h_scale = self.silu.code_scale.mul(u.scale);
        let hmax = h.iter().fold(0i64, |a, &z| a.max(z.abs()));
        let shape = if rows == 1 { vec![hidden] } else { vec![rows, hidden] };
        let hint = if hmax == 0 {
            IntTensor::new(vec![0; g.len()], shape, Scale::ZERO)
        } else {
            let hd: Vec<i32> = h.iter().map(|&z| requant_dyn(z, ACT_LEVELS, hmax)).collect();
            IntTensor::new(hd, shape, Scale::from_ratio(hmax as u128, ACT_LEVELS as u128).mul(h_scale))
        };
        self.down.forward_int(&hint)
    }

    fn forward_ref(&self, x: &[f32]) -> Vec<f32> {
        let g = self.gate.forward_ref(x);
        let u = self.up.forward_ref(x);
        let act: Vec<f32> =
            g.iter().zip(&u).map(|(&gk, &uk)| ((gk as f64 / (1.0 + (-gk as f64).exp())) * uk as f64) as f32).collect();
        self.down.forward_ref(&act)
    }
}

/// Mixture-of-Experts MLP: a router scores all `E` experts; only the **top-k** expert SwiGLU MLPs are computed per
/// token (sparse), and the output is their **softmax-weighted** sum, computed with the SAME integer-code `exp` +
/// fixed-point normalize as attention — so the data path stays zero-float. Generic across MoE families via one knob:
/// `norm_topk_prob` — `false` (OLMoE) normalizes each weight by the softmax over ALL experts; `true` (Mixtral,
/// Qwen-MoE) re-normalizes the selected top-k weights to sum to 1. Routing is per token (one row).
/// Code bits of every MoE router's softmax table: `exp(x)·2^ROUTER_CODE_BITS` for `x ∈ [−30, 0]`, so every routing
/// code is an integer in `[0, 2^ROUTER_CODE_BITS]` for any model and any input, so routing codes are exactly
/// comparable across models.
pub const ROUTER_CODE_BITS: u32 = 20;

pub struct MoeMlp {
    pub name: String,
    pub router: QuantLinear, // dim -> E
    pub experts: Vec<SwiGluMlp>,
    pub top_k: usize,
    /// Re-normalize the top-k weights to sum to 1 (`true`: Mixtral/Qwen-MoE) vs normalize over all experts
    /// (`false`: OLMoE). The numerators are identical either way; only the denominator (all-E vs top-k) differs.
    norm_topk_prob: bool,
    exp_code: CodeLut, // integer-code exp for the router softmax (run-time)
    /// The router's score function: softmax (the default) or sigmoid with a selection bias (GLM-4.5).
    score: RouterScore,
    /// An always-on expert added, unweighted, to the routed sum (GLM-4.5's `shared_experts`), or `None`.
    shared: Option<SwiGluMlp>,
}

impl MoeMlp {
    /// Build a MoE MLP. Defaults to `norm_topk_prob=false` (OLMoE); use [`MoeMlp::with_norm_topk_prob`] for families
    /// that re-normalize the top-k (Mixtral/Qwen-MoE). The default keeps existing OLMoE callers byte-identical.
    pub fn new(name: impl Into<String>, router: QuantLinear, experts: Vec<SwiGluMlp>, top_k: usize) -> Self {
        assert!(top_k >= 1 && top_k <= experts.len(), "top_k in 1..=E");
        assert_eq!(router.out_features, experts.len(), "router out_features != number of experts");
        MoeMlp {
            name: name.into(),
            router,
            experts,
            top_k,
            norm_topk_prob: false,
            exp_code: CodeLut::build(|x| x.exp(), -30.0, 0.0, 16, ROUTER_CODE_BITS, false),
            score: RouterScore::Softmax,
            shared: None,
        }
    }

    /// A MoE layer whose routed experts are all elsewhere (a router-only coordinator). It routes and combines
    /// like any `MoeMlp`, and a shared expert added later runs here; the routed experts come only through
    /// [`MoeMlp::forward_int_with_experts`]. Any local use of a routed expert fails closed.
    pub fn remote(name: impl Into<String>, router: QuantLinear, top_k: usize) -> Self {
        let e = router.out_features;
        assert!(top_k >= 1 && top_k <= e, "top_k in 1..=E");
        MoeMlp {
            name: name.into(),
            router,
            experts: Vec::new(),
            top_k,
            norm_topk_prob: false,
            exp_code: CodeLut::build(|x| x.exp(), -30.0, 0.0, 16, ROUTER_CODE_BITS, false),
            score: RouterScore::Softmax,
            shared: None,
        }
    }

    /// The number of routed experts: the router's outputs.
    pub fn n_experts(&self) -> usize {
        self.router.out_features
    }

    /// Whether the routed experts are elsewhere ([`MoeMlp::remote`]).
    pub fn is_remote(&self) -> bool {
        self.experts.is_empty()
    }

    /// Routed expert `ex`, held here; a remote layer fails closed.
    fn expert(&self, ex: usize) -> &SwiGluMlp {
        assert!(
            !self.is_remote(),
            "{}: the routed experts are remote (MoeMlp::remote); compute them through forward_int_with_experts",
            self.name
        );
        &self.experts[ex]
    }

    /// Score the experts with a sigmoid and choose them by score plus `bias`, the per-expert selection bias
    /// (`e_score_correction_bias`); the weights are the chosen scores without it, normalized to sum 1 with
    /// `norm_topk_prob`, else the raw scores. See [`crate::router`].
    pub fn with_sigmoid_router(mut self, bias: &[f32]) -> Self {
        assert_eq!(bias.len(), self.n_experts(), "{}: one selection bias per expert", self.name);
        self.score = RouterScore::Sigmoid(SigmoidScore::new(bias));
        self
    }

    /// Add an always-on expert whose output is added, unweighted, to the routed sum: `add(routed, shared(x))`.
    pub fn with_shared_expert(mut self, shared: SwiGluMlp) -> Self {
        self.shared = Some(shared);
        self
    }

    /// The router's score function.
    pub fn router_score(&self) -> &RouterScore {
        &self.score
    }

    /// The shared expert, if there is one. A distributed run computes it where the `MoeMlp` runs.
    pub fn shared_expert(&self) -> Option<&SwiGluMlp> {
        self.shared.as_ref()
    }

    /// `routed + shared(x)` at a common scale (the residual add), or `routed` unchanged without a shared expert.
    fn add_shared(&self, routed: IntTensor, x: &IntTensor) -> IntTensor {
        self.add_shared_observed(routed, x, &mut |_, _| {})
    }

    /// [`MoeMlp::add_shared`] that reports each execution of the shared expert to `on_shared(rows, elapsed)`: the rows
    /// it processed and its wall time. Without a shared expert nothing runs and
    /// nothing is reported. The timing only reads the clock; the integers are the same.
    fn add_shared_observed(
        &self,
        routed: IntTensor,
        x: &IntTensor,
        on_shared: &mut dyn FnMut(usize, std::time::Duration),
    ) -> IntTensor {
        match &self.shared {
            None => routed,
            Some(s) => {
                let t = std::time::Instant::now();
                let y = s.forward_int(x);
                on_shared(x.len() / self.dim(), t.elapsed());
                add(&routed, &y)
            }
        }
    }

    /// The router's softmax code table (read-only). Every routing code is one of these entries (`code_at` clamps the
    /// index to the table) or 0.
    pub fn router_codes(&self) -> &[i64] {
        &self.exp_code.codes
    }

    /// Set `norm_topk_prob` (re-normalize the top-k weights to sum to 1). A config-driven knob the generic MoE builder
    /// reads from the checkpoint; `false` (the default) is byte-unchanged.
    pub fn with_norm_topk_prob(mut self, v: bool) -> Self {
        self.norm_topk_prob = v;
        self
    }

    fn dim(&self) -> usize {
        self.router.in_features
    }

    /// Indices of the top-k experts by score (descending; ties resolve to the lower index — deterministic).
    fn topk<T: PartialOrd + Copy>(scores: &[T], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..scores.len()).collect();
        idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        idx.truncate(k);
        idx
    }
}

/// The router decision for one token: the selected experts (top-k, deterministic order), their softmax numerators
/// `ec`, and the normalizer `sumc`. Integer-exact and deterministic — **identical on any node that holds the router
/// weights**, which is exactly what lets the selected experts be computed anywhere (expert-parallel) and gathered back
/// bit-identically.
pub struct MoeRoute {
    pub experts: Vec<usize>,
    pub ec: Vec<i64>,
    pub sumc: i64,
    /// Boundary margin ∈ (0,1]: the softmax weight of the strongest **excluded** expert divided by that of the weakest
    /// **selected** one. → 1 means the top-k cut is a near-tie (the two are nearly interchangeable); → 0 means the cut
    /// is decisive. A read-only measure of how decisive each routing decision is.
    /// A sigmoid router leaves it at 0: its cut is by score plus bias, not by the weights.
    pub margin: f64,
}

impl MoeMlp {
    /// **Expert-parallel seam, step 1:** the routing decision for one token (router + integer-code softmax + top-k +
    /// normalizer). A coordinator computes this; it tells each node which of its experts (if any) this token needs.
    pub fn route(&self, x: &IntTensor) -> MoeRoute {
        let logits = self.router.forward_int(x);
        if let RouterScore::Sigmoid(s) = &self.score {
            return self.route_sigmoid(s, &logits);
        }
        let e = logits.data.len();
        let lmax = logits.data.iter().copied().fold(i32::MIN, i32::max);
        let mut ecs = vec![0i64; e];
        let mut sum_all = 0i64;
        #[allow(clippy::needless_range_loop)] // logits.data and ecs indexed in lockstep
        for j in 0..e {
            let idx = self.exp_code.index((logits.data[j] - lmax) as i64, logits.scale);
            ecs[j] = if idx < 0 { 0 } else { self.exp_code.code_at(idx) };
            sum_all += ecs[j];
        }
        let top = Self::topk(&logits.data, self.top_k);
        // normalizer: top-k codes (norm_topk_prob, Mixtral/Qwen-MoE) or all-E codes (OLMoE). Numerators are the same.
        let sumc: i64 = if self.norm_topk_prob { top.iter().map(|&ex| ecs[ex]).sum() } else { sum_all };
        // boundary margin: strongest excluded / weakest selected, by softmax code (ec is monotone
        // in logit). (0,1]; →1 = near-tie boundary, →0 = decisive.
        let weakest_sel = top.iter().map(|&ex| ecs[ex]).min().unwrap_or(0);
        let strongest_excl = (0..e).filter(|j| !top.contains(j)).map(|j| ecs[j]).max().unwrap_or(0);
        let margin = if weakest_sel > 0 { strongest_excl as f64 / weakest_sel as f64 } else { 0.0 };
        let ec = top.iter().map(|&ex| ecs[ex]).collect();
        MoeRoute { experts: top, ec, sumc, margin }
    }

    /// The sigmoid router's decision: score codes `σ(logit)·2^ROUTER_CODE_BITS`, top-k by code plus coded bias (ties
    /// to the lower index), the chosen codes as numerators over their sum (`norm_topk_prob`) or over
    /// `2^ROUTER_CODE_BITS`.
    fn route_sigmoid(&self, s: &SigmoidScore, logits: &IntTensor) -> MoeRoute {
        let codes = s.codes(logits);
        let top = Self::topk(&s.choice(&codes), self.top_k);
        let ec: Vec<i64> = top.iter().map(|&ex| codes[ex]).collect();
        let sumc = if self.norm_topk_prob { ec.iter().sum() } else { 1i64 << ROUTER_CODE_BITS };
        MoeRoute { experts: top, ec, sumc, margin: 0.0 }
    }

    /// Merge per-token outputs into one `[rows, out_dim]` tensor at a shared scale (the sequence-path reducer;
    /// the same convention as the plain sequence forward).
    fn merge_rows(outs: &[IntTensor], out_dim: usize) -> IntTensor {
        let rows = outs.len();
        let common = outs.iter().map(|o| o.scale).fold(Scale::ZERO, |a, b| if b.ge(a) { b } else { a });
        if common.mant == 0 {
            return IntTensor::new(vec![0; rows * out_dim], vec![rows, out_dim], Scale::ZERO);
        }
        let cinv = common.recip();
        let mut a = vec![0i64; rows * out_dim];
        for (t, o) in outs.iter().enumerate() {
            let f = o.scale.mul(cinv);
            for d in 0..out_dim {
                a[t * out_dim + d] = f.apply(o.data[d] as i64);
            }
        }
        let gmax = a.iter().fold(0i64, |m, &z| m.max(z.abs()));
        if gmax == 0 {
            return IntTensor::new(vec![0; rows * out_dim], vec![rows, out_dim], Scale::ZERO);
        }
        let cd: Vec<i32> = a.iter().map(|&z| requant_dyn(z, ACT_LEVELS, gmax)).collect();
        IntTensor::new(cd, vec![rows, out_dim], Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(common))
    }

    /// **Expert-parallel seam, step 2:** combine the selected experts' outputs into the token's MoE output, where
    /// `ys[i]` is `forward_int` of `route.experts[i]`. The gather is scale-exact (bring each to a common scale, sum the
    /// `ec`-weighted contributions, requant, fold the `1/sumc` softmax normalize) and does NOT depend on where each
    /// `ys[i]` was computed — so a distributed gather reproduces the single-host result bit-identically. `dim` sizes the
    /// degenerate (empty / zero) output.
    pub fn combine(route: &MoeRoute, ys: &[IntTensor], dim: usize) -> IntTensor {
        if route.sumc <= 0 || ys.is_empty() {
            return IntTensor::new(vec![0; dim], vec![dim], Scale::ZERO);
        }
        let out_dim = ys[0].len();
        let common = ys.iter().map(|y| y.scale).fold(Scale::ZERO, |a, b| if b.ge(a) { b } else { a });
        if common.mant == 0 {
            return IntTensor::new(vec![0; out_dim], vec![out_dim], Scale::ZERO);
        }
        let cinv = common.recip();
        let mut acc = vec![0i64; out_dim];
        for (i, y) in ys.iter().enumerate() {
            let f = y.scale.mul(cinv); // bring expert output to the common scale
            let w = route.ec[i]; // this expert's exp code (softmax numerator)
            #[allow(clippy::needless_range_loop)]
            // index keeps the length contract (a short expert output panics; a zip would truncate)
            for d in 0..out_dim {
                acc[d] += w * f.apply(y.data[d] as i64);
            }
        }
        let gmax = acc.iter().fold(0i64, |m, &z| m.max(z.abs()));
        if gmax == 0 {
            return IntTensor::new(vec![0; out_dim], vec![out_dim], Scale::ZERO);
        }
        let cd: Vec<i32> = acc.iter().map(|&z| requant_dyn(z, ACT_LEVELS, gmax)).collect();
        // real = acc · common / sumc; fold the requant ratio, the common scale, and the 1/sumc softmax normalize
        let out_scale = Scale::from_ratio(gmax as u128, ACT_LEVELS as u128)
            .mul(common)
            .mul(Scale::from_ratio(1, route.sumc as u128));
        IntTensor::new(cd, vec![out_dim], out_scale)
    }

    /// Route ONE token `[dim]` → `[out_dim]`: the sparse top-k softmax-weighted expert sum. This is exactly
    /// `combine(route(x), experts…)` — the SAME code path the distributed expert-parallel run uses, so single-host and
    /// distributed are bit-identical by construction. Decode calls it directly; the sequence path calls it per token.
    fn forward_one_int(&self, x: &IntTensor) -> IntTensor {
        let dim = self.dim();
        debug_assert_eq!(x.len(), dim, "{}: forward_one expects one token [dim]", self.name);
        let route = self.route(x);
        if route.sumc <= 0 {
            return IntTensor::new(vec![0; dim], vec![dim], Scale::ZERO);
        }
        let ys: Vec<IntTensor> = route.experts.iter().map(|&ex| self.expert(ex).forward_int(x)).collect();
        Self::combine(&route, &ys, dim)
    }

    /// **Expert-parallel seam, full forward.** Like the `Module::forward_int` sequence path, but each token's selected
    /// experts are computed by `experts_fn(ids, row)` — returning the per-expert outputs **in `ids` order** — instead of
    /// the local `self.experts`. Routing (`route`) and the softmax `combine` gather stay with the caller; only the expert
    /// FFN moves. With the local backend (`|ids,row| ids.iter().map(|&e| self.experts[e].forward_int(row)).collect()`)
    /// this is **byte-identical** to `forward_int`; a distributed backend (workers holding expert shards) therefore
    /// reproduces the single-host result bit-for-bit, because `combine` is placement-independent.
    pub fn forward_int_with_experts(
        &self,
        x: &IntTensor,
        experts_fn: impl FnMut(&[usize], &IntTensor) -> Vec<IntTensor>,
    ) -> IntTensor {
        self.forward_int_with_experts_observed(x, experts_fn, |_, _| {})
    }

    /// [`MoeMlp::forward_int_with_experts`] that also reports each execution of the shared expert to
    /// `on_shared(rows, elapsed)`. The integers are those of `forward_int_with_experts`.
    pub fn forward_int_with_experts_observed(
        &self,
        x: &IntTensor,
        mut experts_fn: impl FnMut(&[usize], &IntTensor) -> Vec<IntTensor>,
        mut on_shared: impl FnMut(usize, std::time::Duration),
    ) -> IntTensor {
        let dim = self.dim();
        assert_eq!(x.len() % dim, 0, "{}: MoE input length must be a multiple of dim", self.name);
        let rows = x.len() / dim;
        let mut row_out = |row: &IntTensor| -> IntTensor {
            let route = self.route(row);
            if route.sumc <= 0 {
                return IntTensor::new(vec![0; dim], vec![dim], Scale::ZERO);
            }
            let ys = experts_fn(&route.experts, row);
            Self::combine(&route, &ys, dim)
        };
        let routed = if rows <= 1 {
            row_out(x)
        } else {
            let outs: Vec<IntTensor> = (0..rows)
                .map(|t| {
                    let row = IntTensor::new(x.data[t * dim..(t + 1) * dim].to_vec(), vec![dim], x.scale);
                    row_out(&row)
                })
                .collect();
            Self::merge_rows(&outs, outs[0].len())
        };
        self.add_shared_observed(routed, x, &mut on_shared)
    }

    /// The routed part of the forward, without the shared expert: one token through `forward_one_int`, a sequence token
    /// by token, merged at one scale.
    fn forward_routed(&self, x: &IntTensor) -> IntTensor {
        let dim = self.dim();
        assert_eq!(x.len() % dim, 0, "{}: MoE input length must be a multiple of dim", self.name);
        let rows = x.len() / dim;
        if rows <= 1 {
            return self.forward_one_int(x);
        }
        let outs: Vec<IntTensor> = (0..rows)
            .map(|t| {
                let row = IntTensor::new(x.data[t * dim..(t + 1) * dim].to_vec(), vec![dim], x.scale);
                self.forward_one_int(&row)
            })
            .collect();
        let out_dim = outs[0].len();
        let common = outs.iter().map(|o| o.scale).fold(Scale::ZERO, |a, b| if b.ge(a) { b } else { a });
        if common.mant == 0 {
            return IntTensor::new(vec![0; rows * out_dim], vec![rows, out_dim], Scale::ZERO);
        }
        let cinv = common.recip();
        let mut a = vec![0i64; rows * out_dim];
        for (t, o) in outs.iter().enumerate() {
            let f = o.scale.mul(cinv); // bring each token's output to the common scale
            for d in 0..out_dim {
                a[t * out_dim + d] = f.apply(o.data[d] as i64);
            }
        }
        let gmax = a.iter().fold(0i64, |m, &z| m.max(z.abs()));
        if gmax == 0 {
            return IntTensor::new(vec![0; rows * out_dim], vec![rows, out_dim], Scale::ZERO);
        }
        let cd: Vec<i32> = a.iter().map(|&z| requant_dyn(z, ACT_LEVELS, gmax)).collect();
        IntTensor::new(cd, vec![rows, out_dim], Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(common))
    }

    /// The float reference for ONE token.
    fn forward_one_ref(&self, x: &[f32]) -> Vec<f32> {
        let logits = self.router.forward_ref(x);
        let (weights, top) = match &self.score {
            RouterScore::Softmax => {
                let lmax = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f64> = logits.iter().map(|&l| ((l - lmax) as f64).exp()).collect();
                (exps, Self::topk(&logits, self.top_k))
            }
            RouterScore::Sigmoid(s) => {
                let (scores, choice) = s.reference(&logits);
                (scores, Self::topk(&choice, self.top_k))
            }
        };
        // normalizer: top-k (norm_topk_prob: Mixtral, Qwen-MoE, GLM) or all experts (OLMoE); a sigmoid router without
        // norm_topk_prob keeps its raw scores
        let sum: f64 = match (&self.score, self.norm_topk_prob) {
            (_, true) => top.iter().map(|&ex| weights[ex]).sum(),
            (RouterScore::Softmax, false) => weights.iter().sum(),
            (RouterScore::Sigmoid(_), false) => 1.0,
        };
        let dim = self.dim();
        let mut out = vec![0f64; dim];
        for &ex in &top {
            let w = weights[ex] / sum;
            let ye = self.expert(ex).forward_ref(x);
            for d in 0..ye.len() {
                out[d] += w * ye[d] as f64;
            }
        }
        if let Some(s) = &self.shared {
            for (o, y) in out.iter_mut().zip(s.forward_ref(x)) {
                *o += y as f64;
            }
        }
        out.iter().map(|&v| v as f32).collect()
    }
}

impl Module for MoeMlp {
    fn name(&self) -> &str {
        &self.name
    }

    /// Route a token or a whole sequence. For one token (decode) this is `forward_one_int` verbatim (byte-unchanged).
    /// For a sequence (prefill, `[rows·dim]`) each token is routed independently, then the per-token outputs are merged
    /// into ONE tensor at a shared scale — the engine's one-scale-per-tensor convention, the same global `requant_dyn`
    /// as `SwiGluMlp`. Routing is per token, so the result is independent of how tokens are batched.
    fn forward_int(&self, x: &IntTensor) -> IntTensor {
        let routed = self.forward_routed(x);
        self.add_shared(routed, x)
    }

    fn forward_ref(&self, x: &[f32]) -> Vec<f32> {
        let dim = self.dim();
        let rows = (x.len() / dim).max(1);
        if rows == 1 {
            return self.forward_one_ref(x);
        }
        let mut out = Vec::with_capacity(rows * dim);
        for t in 0..rows {
            out.extend(self.forward_one_ref(&x[t * dim..(t + 1) * dim]));
        }
        out
    }

    /// Routing trace: the selected experts for each of the `rows` tokens in `x` (same decision the forward makes).
    fn route_trace(&self, x: &IntTensor) -> Option<Vec<Vec<usize>>> {
        let dim = self.dim();
        let rows = (x.len() / dim).max(1);
        Some(
            (0..rows)
                .map(|t| {
                    let row = IntTensor::new(x.data[t * dim..(t + 1) * dim].to_vec(), vec![dim], x.scale);
                    self.route(&row).experts
                })
                .collect(),
        )
    }

    /// The distributed-expert seam: hand back self so a distributed forward can drive route / remote experts / combine.
    fn as_moe(&self) -> Option<&MoeMlp> {
        Some(self)
    }
}

/// Token embedding: a weight table quantized to b=12.
pub struct Embedding {
    pub name: String,
    pub vocab: usize,
    pub dim: usize,
    codes: Vec<i32>,
    pub scale: Scale,
}

impl Embedding {
    pub fn from_weights(name: impl Into<String>, w: &[f32], vocab: usize, dim: usize) -> Self {
        assert_eq!(w.len(), vocab * dim);
        let peak = w.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
        let (scale, codes) = if peak == 0.0 {
            (Scale::ZERO, vec![0i32; w.len()])
        } else {
            let step = peak / WLEVELS as f64;
            (
                Scale::from_f64(step),
                w.iter().map(|&v| ((v as f64 / step).round() as i64).clamp(-WLEVELS, WLEVELS) as i32).collect(),
            )
        };
        Embedding { name: name.into(), vocab, dim, codes, scale }
    }

    pub fn lookup(&self, token: usize) -> IntTensor {
        assert!(token < self.vocab, "token {token} out of vocab {}", self.vocab);
        IntTensor::vector(self.codes[token * self.dim..(token + 1) * self.dim].to_vec(), self.scale)
    }

    pub fn lookup_seq(&self, tokens: &[usize]) -> IntTensor {
        let mut data = Vec::with_capacity(tokens.len() * self.dim);
        for &t in tokens {
            assert!(t < self.vocab, "token {t} out of vocab {}", self.vocab);
            data.extend_from_slice(&self.codes[t * self.dim..(t + 1) * self.dim]);
        }
        IntTensor::new(data, vec![tokens.len(), self.dim], self.scale)
    }
}
