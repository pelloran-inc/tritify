//! kt-int — the zero-float integer op-kernels.
//!
//! A transformer forward decomposes into a handful of op-primitives. Two of them are the only places a
//! naive integer-quantized model still touches floating point:
//!
//!   1. the **rescale** after an (already exact) integer matmul:  `y = accum · (s_x·s_w)` — an
//!      arbitrary-mantissa float multiply; and
//!   2. the **pointwise nonlinearities** — GELU/SiLU, softmax's `exp`, LayerNorm/RMSNorm's `rsqrt` — each a
//!      transcendental FPU op.
//!
//! This crate removes both, so the runtime is integer-only:
//!
//!   * [`requant`] replaces the float scale with an integer mantissa and a power-of-two shift,
//!     `scale = mant · 2^(-shift)`, and evaluates `round(accum · mant / 2^shift)` as a widening integer
//!     multiply + arithmetic shift (x86 `MUL`, ARM `UMULH` + shift — the standard TFLite/gemmlowp requant).
//!   * [`Lut`] tabulates any pointwise nonlinearity ONCE offline; at run time the input becomes an integer
//!     index and the value is a table read — no `exp`/`gelu` is ever evaluated at run time.
//!   * [`rsqrt`] computes `1/sqrt(v)` with no `sqrt`: a frexp (exponent extract = CLZ+shift, an integer op),
//!     a small mantissa table, and an exact power-of-two scale.
//!
//! Float appears ONLY in offline calibration (`frexp_scale`, `Lut::build`) and in the tests (as the reference
//! the integer path is checked against). The run-time functions themselves do no float arithmetic.
//!
//! These kernels are model-agnostic: GPT-2, Llama, BERT, ViT, SAM all compose from the same primitives.
//! The unit tests below map the kernels' precision envelope: fixed-point requant is exact at full mantissa and
//! within one ULP down to ≤ 8 mantissa bits, and the table-driven nonlinearities converge to their float
//! definition as table resolution grows.
//!
//! # Example
//!
//! ```
//! use kt_int::{isqrt_u128, Scale};
//!
//! // A fixed-point scale `mant · 2^-shift`, applied to an integer: no float at run time.
//! let three_quarters = Scale::from_ratio(3, 4);
//! assert_eq!(three_quarters.apply(8), 6);
//! // Scales compose exactly: 3/4 · 4/3 is the identity on an integer.
//! assert_eq!(three_quarters.mul(Scale::from_ratio(4, 3)).apply(9), 9);
//! assert_eq!(isqrt_u128(144), 12);
//! ```

// ───────────────────────────── fixed-point requantization ─────────────────────────────

/// `frexp(x) = (m, e)` with `x = m · 2^e` and `m ∈ [0.5, 1)` for a positive normal `x` (matches C/Python
/// `frexp`). Offline calibration only. Exact — pure IEEE-754 bit surgery, no `log2`.
pub fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i32;
    if exp_field == 0 {
        // subnormal: scale up by 2^53, recurse, correct the exponent.
        let (m, e) = frexp(x * 9007199254740992.0);
        return (m, e - 53);
    }
    let e = exp_field - 1022; // put the mantissa in [0.5, 1)
    let mantissa = (bits & !(0x7ffu64 << 52)) | (1022u64 << 52);
    (f64::from_bits(mantissa), e)
}

/// Decompose a positive float `scale` into `(mant, shift)` with `scale ≈ mant · 2^(-shift)`, `mant` a
/// `mant_bits`-bit non-negative integer. `mant_bits = 53` is exact to f64; fewer bits truncates the mantissa
/// (used only to measure how few scale bits still reproduce the output). Offline calibration only.
pub fn frexp_scale(scale: f64, mant_bits: u32) -> (i64, i32) {
    if !(scale > 0.0 && scale.is_finite()) {
        return (0, 0);
    }
    let (m, e) = frexp(scale);
    let mant = (m * (1u64 << mant_bits) as f64).round() as i64;
    let shift = mant_bits as i32 - e;
    (mant, shift)
}

/// The requantization runtime kernel — **zero float**. Given an exact integer matmul accumulation `accum` and a scale
/// expressed as `(mant, shift)` (from [`frexp_scale`], offline), return `round(accum · mant / 2^shift)`
/// computed as a widening (`i128`) integer multiply and a round-half-away-from-zero arithmetic shift.
pub fn requant(accum: i64, mant: i64, shift: i32) -> i64 {
    let prod: i128 = accum as i128 * mant as i128;
    if shift <= 0 {
        return (prod << (-shift)) as i64; // exact power-of-two upscale, no rounding
    }
    let half: i128 = 1i128 << (shift - 1);
    let r = if prod >= 0 { (prod + half) >> shift } else { -(((-prod) + half) >> shift) };
    r as i64
}

// ───────────────────────────── table-driven nonlinearities ─────────────────────────────

/// What to return when the input falls outside the tabulated range.
#[derive(Clone, Copy)]
pub enum Tail {
    /// Use the nearest end entry of the table.
    Clip,
    /// Return the input unchanged (e.g. GELU's large-`x` tail, where GELU(x) = x).
    Identity,
    /// Return a fixed value (e.g. GELU's left tail → 0).
    Value(f64),
}

/// A pointwise nonlinearity tabulated once offline over `[lo, hi]` at `2^idx_bits` points. At run time the
/// value is an **integer index** into `table` — no transcendental is evaluated. The interchange value is
/// shown as `f64` here; in the full integer runtime it is a fixed-point integer.
pub struct Lut {
    pub table: Vec<f64>,
    pub lo: f64,
    pub hi: f64,
    pub n: usize,
    pub inv_step: f64,
    pub below: Tail,
    pub above: Tail,
}

impl Lut {
    /// Build the table offline (float allowed here). `f` is the scalar nonlinearity.
    pub fn build(f: impl Fn(f64) -> f64, lo: f64, hi: f64, idx_bits: u32, below: Tail, above: Tail) -> Lut {
        let n = 1usize << idx_bits;
        let table = (0..n).map(|i| f(lo + (hi - lo) * (i as f64 / (n - 1) as f64))).collect();
        Lut { table, lo, hi, n, inv_step: (n - 1) as f64 / (hi - lo), below, above }
    }

    /// Run-time lookup: `idx = round((x - lo) · inv_step)`, an integer; then one table read + tail handling.
    pub fn apply(&self, x: f64) -> f64 {
        let idx = ((x - self.lo) * self.inv_step).round() as i64;
        if idx < 0 {
            return match self.below {
                Tail::Clip => self.table[0],
                Tail::Identity => x,
                Tail::Value(v) => v,
            };
        }
        if idx as usize > self.n - 1 {
            return match self.above {
                Tail::Clip => self.table[self.n - 1],
                Tail::Identity => x,
                Tail::Value(v) => v,
            };
        }
        self.table[idx as usize]
    }
}

// ───────────────────────────── rsqrt without sqrt ─────────────────────────────

/// `1/sqrt(v)` with **no `sqrt`**. Decompose `v = m · 2^e` (frexp = exponent extract, a CLZ+shift on a
/// fixed-point word). Fold the parity of `e` into the mantissa domain [0.5, 2): even `e` → `m ∈ [0.5,1)`,
/// odd `e` → `2m ∈ [1,2)`, `e` made even. Then `rsqrt(v) = 2^(-e/2) · rsqrt(m)`, the mantissa `rsqrt(m)` a
/// small offline LUT, the `2^(-e/2)` an exact power-of-two. The `mant_lut` must tabulate `1/sqrt(·)` on
/// `[0.5, 2)`.
pub fn rsqrt(v: f64, mant_lut: &Lut) -> f64 {
    let (mut m, mut e) = frexp(v);
    if e & 1 != 0 {
        m *= 2.0; // m ∈ [1,2)
        e -= 1; // e now even
    }
    let r_m = mant_lut.apply(m);
    // 2^(-e/2): an exact power-of-two (integer exponent) scale.
    let p = -(e / 2);
    r_m * (2.0f64).powi(p)
}

/// The GPT-2 `gelu_new` (tanh) approximation — the float reference used to build/check the GELU table.
pub fn gelu_new(x: f64) -> f64 {
    const C: f64 = 0.7978845608028654; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

// ───────────────────────── Integer fixed-point scale (the zero-float glue) ─────────────────────────
// A per-tensor scale is `mant · 2^(-shift)` (mant ≥ 0, ~31 bits ⇒ ≥ f32 precision). It lets the runtime carry
// activation scales and compose them with INTEGER arithmetic only — no float multiply on the data path. The
// scale VALUE is bookkeeping (one integer pair per tensor); applying it to the data is `(value·mant)>>shift`.

/// A non-negative scale `mant · 2^(-shift)`. `mant == 0` means exactly zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scale {
    pub mant: i64,
    pub shift: i32,
}

impl Scale {
    pub const ZERO: Scale = Scale { mant: 0, shift: 0 };

    #[inline]
    fn bits_u128(x: u128) -> i32 {
        (128 - x.leading_zeros()) as i32
    }

    /// Normalize a positive ratio `num/den` into `mant · 2^(-shift)` with `mant` ≈ 31 bits. Pure integer,
    /// overflow-safe: `num`,`den` are first reduced below 2^64 (ratio preserved), then `mant = round(num·2^s/den)`
    /// with `s = 31 − (bits(num) − bits(den))`, whose intermediate stays below 2^96.
    pub fn from_ratio(mut num: u128, mut den: u128) -> Scale {
        if num == 0 {
            return Scale::ZERO;
        }
        assert!(den > 0, "Scale::from_ratio den must be > 0");
        while Self::bits_u128(num) > 64 || Self::bits_u128(den) > 64 {
            num >>= 1;
            den >>= 1;
        }
        if den == 0 {
            den = 1;
        }
        let s = 31 - (Self::bits_u128(num) - Self::bits_u128(den));
        let mant = if s >= 0 {
            let numer = num << s; // < 2^96 (proven: bits(num)+s = 31+bits(den) ≤ 95)
            (numer + den / 2) / den
        } else {
            let dd = den << (-s);
            (num + dd / 2) / dd
        };
        Scale { mant: mant as i64, shift: s }
    }

    /// A scale from a positive finite float (offline calibration / weight scales — float allowed offline).
    pub fn from_f64(x: f64) -> Scale {
        if !(x > 0.0 && x.is_finite()) {
            return Scale::ZERO;
        }
        // x = m·2^e, m∈[0.5,1). mant = round(m·2^31); shift = 31 − e.
        let (m, e) = frexp(x);
        Scale { mant: (m * (1u64 << 31) as f64).round() as i64, shift: 31 - e }
    }

    /// `self >= other` by magnitude — pure integer, overflow-safe (both mants are non-negative). Equivalent to
    /// comparing `mant·2^(-shift)` without floating point: `self >= other` iff `ma·2^(sb-sa) >= mb`.
    #[inline]
    pub fn ge(self, o: Scale) -> bool {
        if self.mant == 0 {
            return o.mant == 0;
        }
        if o.mant == 0 {
            return true;
        }
        let (ma, mb) = (self.mant as u128, o.mant as u128);
        let d = o.shift - self.shift; // self >= other  ⟺  ma·2^d >= mb
        if d >= 0 {
            if d as u32 + Self::bits_u128(ma) as u32 > 127 {
                return true; // ma·2^d overflows u128 ⇒ astronomically larger than mb
            }
            (ma << d) >= mb
        } else {
            let e = (-d) as u32;
            if e + Self::bits_u128(mb) as u32 > 127 {
                return false; // mb·2^e overflows u128 ⇒ mb astronomically larger
            }
            ma >= (mb << e)
        }
    }

    /// Product of two scales (renormalized), integer.
    #[allow(clippy::should_implement_trait)] // named on purpose: exact fixed-point arithmetic stays explicit at call sites
    pub fn mul(self, o: Scale) -> Scale {
        if self.mant == 0 || o.mant == 0 {
            return Scale::ZERO;
        }
        let p = Scale::from_ratio(self.mant as u128 * o.mant as u128, 1);
        Scale { mant: p.mant, shift: p.shift + self.shift + o.shift }
    }

    /// `round(value · self)` — the integer requant. Widening `i128` multiply + rounded arithmetic shift.
    pub fn apply(self, value: i64) -> i64 {
        if self.mant == 0 {
            return 0;
        }
        let prod = value as i128 * self.mant as i128;
        if self.shift <= 0 {
            return (prod << (-self.shift)) as i64;
        }
        let half = 1i128 << (self.shift - 1);
        let r = if prod >= 0 { (prod + half) >> self.shift } else { -(((-prod) + half) >> self.shift) };
        r as i64
    }

    /// Reciprocal `1 / (mant·2^(-shift)) = 2^shift / mant`, as a normalized `Scale`. Integer.
    pub fn recip(self) -> Scale {
        assert!(self.mant != 0, "recip of zero scale");
        // 1/(mant·2^(-shift)) = 2^shift / mant = (2^40/mant) · 2^(shift-40)
        let r = Scale::from_ratio(1u128 << 40, self.mant as u128); // ≈ 2^40 / mant = r.mant·2^(-r.shift)
        Scale { mant: r.mant, shift: r.shift - self.shift + 40 }
    }

    /// Sum of two scales, integer: align to the finer (larger) shift, add mantissas, renormalize. When the two
    /// differ by more than ~2^96 the smaller is below the 31-bit mantissa and is dropped (exact at this precision).
    #[allow(clippy::should_implement_trait)] // named on purpose: exact fixed-point arithmetic stays explicit at call sites
    pub fn add(self, o: Scale) -> Scale {
        if self.mant == 0 {
            return o;
        }
        if o.mant == 0 {
            return self;
        }
        let s = self.shift.max(o.shift);
        let (da, db) = ((s - self.shift) as u32, (s - o.shift) as u32);
        if da >= 96 {
            return self; // o negligible (self's value ≫ o's)
        }
        if db >= 96 {
            return o; // self negligible
        }
        let sum = ((self.mant as u128) << da) + ((o.mant as u128) << db); // value = sum · 2^(-s)
        let fr = Scale::from_ratio(sum, 1);
        Scale { mant: fr.mant, shift: fr.shift + s }
    }

    /// Inverse square root `1 / sqrt(mant·2^(-shift))`, as a normalized `Scale`. Pure integer: make the exponent
    /// even, take an integer square root with 31 fractional bits, and reciprocate. No floating point.
    pub fn rsqrt(self) -> Scale {
        assert!(self.mant > 0, "rsqrt of non-positive scale");
        let (mut m, mut s) = (self.mant as u128, self.shift);
        if s & 1 != 0 {
            m <<= 1; // value = m·2^(-s) unchanged: m doubled, shift+1 → shift even
            s += 1;
        }
        let k: u32 = 31;
        let r = isqrt_u128(m << (2 * k)); // ≈ sqrt(m) · 2^k
                                          // sqrt(value) = r · 2^(-(k + s/2)); rsqrt = its reciprocal.
        Scale { mant: r as i64, shift: k as i32 + s / 2 }.recip()
    }

    /// The real value `mant · 2^(-shift)` as f64 — for tests / interchange, never the data path.
    pub fn to_f64(self) -> f64 {
        (self.mant as f64) * (2.0f64).powi(-self.shift)
    }
}

/// Integer floor square root of a u128, via Newton's method (pure integer). `isqrt_u128(n) = ⌊√n⌋`.
pub fn isqrt_u128(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    let bits = 128 - n.leading_zeros();
    let mut x = 1u128 << bits.div_ceil(2); // ≥ ⌊√n⌋, so Newton descends monotonically
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// A pointwise nonlinearity tabulated as INTEGER codes: `value ≈ code · code_scale`. The table is built once
/// (float allowed offline); at run time the input (an integer `data` at an integer [`Scale`]) is turned into an
/// integer index with integer arithmetic, and the value is a code read — no float on the data path. `right_id`
/// marks the identity right tail (e.g. SiLU(x)→x for large x), handled by the caller in integer units.
pub struct CodeLut {
    pub codes: Vec<i64>,
    /// `inv_step = (n−1)/(hi−lo)` as an integer [`Scale`] and `round(lo·inv_step)`, both fixed at build so that
    /// [`CodeLut::index`] runs on integers only.
    inv_step_scale: Scale,
    lo_index: i64,
    n: usize,
    pub code_scale: Scale,
    pub right_identity: bool,
}

impl CodeLut {
    /// Tabulate `f` over `[lo,hi]` at `2^idx_bits` points, codes scaled by `2^code_bits`.
    pub fn build(
        f: impl Fn(f64) -> f64,
        lo: f64,
        hi: f64,
        idx_bits: u32,
        code_bits: u32,
        right_identity: bool,
    ) -> CodeLut {
        let n = 1usize << idx_bits;
        let k = (1u64 << code_bits) as f64;
        let codes = (0..n).map(|i| (f(lo + (hi - lo) * (i as f64 / (n - 1) as f64)) * k).round() as i64).collect();
        let inv_step = (n - 1) as f64 / (hi - lo);
        CodeLut {
            codes,
            inv_step_scale: Scale::from_f64(inv_step),
            lo_index: (lo * inv_step).round() as i64,
            n,
            code_scale: Scale::from_f64(1.0 / k),
            right_identity,
        }
    }

    /// Integer index of an input value `data·scale`: `round((data·scale − lo)·inv_step)`, computed with the
    /// integer `Scale` (one scalar `apply` + an integer subtract). Integers only: both constants were fixed at build.
    pub fn index(&self, data: i64, scale: Scale) -> i64 {
        scale.mul(self.inv_step_scale).apply(data) - self.lo_index
    }

    /// Number of table entries.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Read a code at an index, clamping into range (tails handled by the caller for the identity side).
    pub fn code_at(&self, idx: i64) -> i64 {
        self.codes[idx.clamp(0, self.n as i64 - 1) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `CodeLut::index` with its two constants fixed at build gives, for every table the engine builds and a spread
    /// of inputs, exactly the index the run-time float expressions gave: the same function of the same constants.
    #[test]
    fn code_lut_index_with_build_time_constants_equals_the_run_time_expressions() {
        // (lo, hi, idx_bits): exp in attention/SDPA/the router (-30..0), SiLU (-30..30), GELU (-10..10), the
        // sigmoid router's (-16..16), and two others
        let tables = [
            (-30.0, 0.0, 16),
            (-30.0, 30.0, 16),
            (-10.0, 10.0, 16),
            (-16.0, 16.0, 16),
            (-8.0, 8.0, 12),
            (-1.0, 3.0, 10),
        ];
        let mut s = 0x5EEDu64;
        for &(lo, hi, bits) in &tables {
            let lut = CodeLut::build(|x| x, lo, hi, bits, 20, false);
            let inv_step = ((1usize << bits) - 1) as f64 / (hi - lo);
            for _ in 0..20000 {
                let data = (lcg(&mut s) >> 20) as i64 - (1i64 << 43);
                let scale = Scale::from_f64(1e-9 + (lcg(&mut s) >> 40) as f64 / (1u64 << 30) as f64);
                // the index as `index` computed it before: both expressions evaluated on every call
                let before = scale.mul(Scale::from_f64(inv_step)).apply(data) - (lo * inv_step).round() as i64;
                assert_eq!(lut.index(data, scale), before, "lo {lo} hi {hi} bits {bits} data {data}");
            }
        }
    }

    // Deterministic LCG so the test is byte-reproducible on any machine.
    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    }

    /// Fixed, spread-out set of (accum, scale) pairs covering the operating regime: |accum| up to
    /// ~2^41 (observed max ~1.52e12 = 41 bits), with dequantizing scales ≤ 1 (the realistic
    /// requant direction: a large integer accumulation scaled back down to a bounded activation). Keeping the
    /// product bounded means the f64 baseline used as the check reference is itself exact to well under ½.
    fn requant_cases() -> Vec<(i64, f64)> {
        let mut s = 0xC0FFEEu64;
        let mut v = Vec::new();
        for _ in 0..20000 {
            let bits = (lcg(&mut s) % 42) as u32; // 0..41 bits
            let mag = (lcg(&mut s) & ((1u64 << bits).saturating_sub(1).max(1))) as i64;
            let accum = if lcg(&mut s) & 1 == 0 { mag } else { -mag };
            // scale = 2^(uniform in [-40, 0]) · (1 + fractional), an arbitrary-mantissa float ≤ ~2.
            let exp = -((lcg(&mut s) % 41) as i32);
            let frac = 1.0 + (lcg(&mut s) >> 40) as f64 / (1u64 << 24) as f64;
            let scale = frac * (2.0f64).powi(exp);
            v.push((accum, scale));
        }
        v
    }

    /// Kernel-level precision behaviour of the requant: at full precision (53-bit mantissa) the integer path
    /// is EXACT; as the scale mantissa is truncated the output drifts, and we report the max absolute /
    /// relative deviation from the 53-bit output as a function of `mant_bits`. (Honest scope: how few scale
    /// bits a *whole model* tolerates (≈ 8) is an end-to-end result measured after argmax, not a per-element
    /// bit-identity; that is a whole-model property, out of scope for this kernel test.)
    #[test]
    fn requant_precision_sweep() {
        let cases = requant_cases();
        let full: Vec<i64> = cases
            .iter()
            .map(|&(a, sc)| {
                let (m, sh) = frexp_scale(sc, 53);
                requant(a, m, sh)
            })
            .collect();

        for &b in &[53u32, 32, 24, 16, 12, 10, 8, 6, 4] {
            let mut max_abs = 0i64;
            let mut max_rel = 0.0f64;
            for (&(a, sc), &want) in cases.iter().zip(&full) {
                let (m, sh) = frexp_scale(sc, b);
                let got = requant(a, m, sh);
                let d = (got - want).abs();
                max_abs = max_abs.max(d);
                if want != 0 {
                    max_rel = max_rel.max(d as f64 / (want.abs() as f64));
                }
            }
            println!("requant mant_bits={:>2}  max_abs_dev={:>14}  max_rel_dev={:.3e}", b, max_abs, max_rel);
            if b == 53 {
                assert_eq!(max_abs, 0, "the 53-bit integer requant must be exact");
            }
        }
    }

    /// The zero-float integer requant reproduces the float baseline `round(accum · scale)`. Reported as an
    /// exact-match rate; any residual is a single-ULP float-rounding artifact of the *reference*, never of
    /// the (exact-rational) integer path, so we also assert the gap is never more than 1.
    #[test]
    fn requant_matches_float_baseline() {
        let cases = requant_cases();
        let mut exact = 0usize;
        for &(a, sc) in &cases {
            let (m, sh) = frexp_scale(sc, 53);
            let got = requant(a, m, sh);
            let reference = (a as f64 * sc).round() as i64;
            assert!((got - reference).abs() <= 1, "accum={a} scale={sc} got={got} ref={reference}");
            if got == reference {
                exact += 1;
            }
        }
        let rate = 100.0 * exact as f64 / cases.len() as f64;
        println!("requant vs float baseline: {:.3}% exact ({}/{})", rate, exact, cases.len());
        assert!(rate > 99.0, "exact-match rate unexpectedly low: {rate:.3}%");
    }

    /// GELU table: max|LUT(x) − GELU(x)| over the input range falls monotonically with resolution and is
    /// tiny well before f32's epsilon (2^-24 ≈ 6.0e-8). This is the kernel-level property; how many index bits
    /// a bit-identical token hash needs end-to-end is a whole-model property, out of scope for this test.
    #[test]
    fn gelu_lut_converges() {
        let (lo, hi) = (-26.25, 62.65); // GPT-2 GELU input range
        let mut last = f64::INFINITY;
        for &bits in &[10u32, 12, 14, 16, 18, 20] {
            let lut = Lut::build(gelu_new, lo, hi, bits, Tail::Value(0.0), Tail::Identity);
            let mut max_err = 0.0f64;
            let mut s = 1u64; // same sample stream at every resolution
            for _ in 0..100000 {
                let x = lo + (hi - lo) * ((lcg(&mut s) >> 40) as f64 / (1u64 << 24) as f64);
                max_err = max_err.max((lut.apply(x) - gelu_new(x)).abs());
            }
            println!("gelu LUT idx_bits={:>2}  entries={:>8}  max_abs_err={:.3e}", bits, lut.n, max_err);
            last = max_err;
        }
        // The highest-resolution (20-bit) table must sit far below f32 epsilon (2^-24 ≈ 6.0e-8 → ·range).
        assert!(last < 1e-4, "20-bit GELU table error too high: {last:.3e}");
    }

    /// rsqrt without sqrt: matches `1/sqrt(v)` over the LayerNorm range to the table resolution,
    /// using only a frexp + a [0.5,2) mantissa table + a power-of-two. Reported as max relative error vs bits.
    #[test]
    fn rsqrt_no_sqrt_converges() {
        let (lo, hi) = (0.025f64, 12589.68); // LayerNorm var+eps range
        let mut last = f64::INFINITY;
        for &bits in &[10u32, 12, 14, 16, 18, 20] {
            let mant_lut = Lut::build(|m| 1.0 / m.sqrt(), 0.5, 2.0, bits, Tail::Clip, Tail::Clip);
            let mut max_rel = 0.0f64;
            let mut s = 3u64; // same sample stream at every resolution
            for _ in 0..100000 {
                // log-uniform sample across the wide range
                let t = (lcg(&mut s) >> 40) as f64 / (1u64 << 24) as f64;
                let v = lo * (hi / lo).powf(t);
                let got = rsqrt(v, &mant_lut);
                let want = 1.0 / v.sqrt();
                max_rel = max_rel.max(((got - want) / want).abs());
            }
            println!("rsqrt LUT idx_bits={:>2}  max_rel_err={:.3e}", bits, max_rel);
            last = max_rel;
        }
        assert!(last < 1e-5, "20-bit rsqrt rel error too high: {last:.3e}");
    }

    /// The requant function itself contains no floating-point path: a smoke check that a representative
    /// exact scale (3.0 → mant·2^-shift) gives exactly `3·accum` with no rounding drift.
    #[test]
    fn requant_exact_integer_scale() {
        let (m, sh) = frexp_scale(3.0, 53);
        for a in [-1_000_000i64, -7, 0, 1, 42, 999_999, 1_000_000_000_000] {
            assert_eq!(requant(a, m, sh), 3 * a, "exact x3 scale failed at accum={a}");
        }
    }

    // ── Scale fixed-point primitive, checked against f64 over a wide range ──
    #[test]
    fn scale_from_ratio_matches_f64() {
        let mut s = 0x5ca1eu64;
        let mut max_rel = 0.0f64;
        for _ in 0..200000 {
            let num = (lcg(&mut s) % (1u64 << 50)).max(1) as u128;
            let den = (lcg(&mut s) % (1u64 << 50)).max(1) as u128;
            let sc = Scale::from_ratio(num, den);
            let want = num as f64 / den as f64;
            let rel = ((sc.to_f64() - want) / want).abs();
            max_rel = max_rel.max(rel);
        }
        println!("Scale::from_ratio max rel err vs f64 = {max_rel:.3e}");
        assert!(max_rel < 1e-8, "from_ratio too imprecise: {max_rel:.3e}"); // 31-bit mantissa ≥ f32
    }

    #[test]
    fn scale_mul_and_apply_match_f64() {
        let mut s = 0xA5A5u64;
        let mut max_rel_mul = 0.0f64;
        let mut max_apply_err = 0i64;
        for _ in 0..200000 {
            let a = Scale::from_f64(1e-6 + (lcg(&mut s) >> 40) as f64 / (1u64 << 20) as f64);
            let b = Scale::from_f64(1e-3 + (lcg(&mut s) >> 40) as f64 / (1u64 << 24) as f64);
            let p = a.mul(b);
            let want = a.to_f64() * b.to_f64();
            max_rel_mul = max_rel_mul.max(((p.to_f64() - want) / want).abs());
            // apply: round(value · scale) vs the f64 reference
            let value = (lcg(&mut s) >> 24) as i64 - (1i64 << 39);
            let got = p.apply(value);
            let reference = (value as f64 * p.to_f64()).round() as i64;
            max_apply_err = max_apply_err.max((got - reference).abs());
        }
        println!("Scale::mul max rel err = {max_rel_mul:.3e} · apply max |Δ| vs f64 = {max_apply_err}");
        assert!(max_rel_mul < 1e-8, "mul too imprecise: {max_rel_mul:.3e}");
        assert!(max_apply_err <= 1, "apply off by > 1 from the f64-rounded reference");
    }

    #[test]
    fn scale_recip_matches_f64() {
        let mut s = 0x3c3cu64;
        let mut max_rel = 0.0f64;
        for _ in 0..100000 {
            let x = 1e-7 + (lcg(&mut s) >> 30) as f64 / (1u64 << 20) as f64;
            let r = Scale::from_f64(x).recip();
            let want = 1.0 / x;
            max_rel = max_rel.max(((r.to_f64() - want) / want).abs());
        }
        println!("Scale::recip max rel err vs f64 = {max_rel:.3e}");
        assert!(max_rel < 1e-8, "recip too imprecise: {max_rel:.3e}");
    }

    // Inverse-RMS error: Scale::rsqrt (make exponent even, integer isqrt with 31 fractional bits, reciprocate —
    // all the roundings composed) vs the f64 reference 1/sqrt(v), measured over a defined range of positive
    // mean-square inputs v. This is the measured figure reported in the whitepaper tolerance table.
    #[test]
    fn scale_rsqrt_matches_f64() {
        let mut s = 0x1557a1u64;
        let mut max_rel = 0.0f64;
        for _ in 0..200000 {
            // log-uniform positive v in [2^-20, 2^20] (covers the RMSNorm mean-square operating range)
            let frac = (lcg(&mut s) >> 40) as f64 / (1u64 << 24) as f64; // [0,1)
            let v = 2.0f64.powf(-20.0 + 40.0 * frac);
            let got = Scale::from_f64(v).rsqrt().to_f64();
            let want = 1.0 / v.sqrt();
            max_rel = max_rel.max(((got - want) / want).abs());
        }
        println!("Scale::rsqrt max rel err vs f64 over v in [2^-20,2^20] = {max_rel:.3e}");
        // measured 5.77e-10 (~2^-30.7); frozen regression bound with platform margin
        assert!(max_rel < 2e-9, "rsqrt rel error too high: {max_rel:.3e}");
    }

    #[test]
    fn scale_zero_and_exact() {
        assert_eq!(Scale::from_ratio(0, 5).apply(123), 0);
        assert_eq!(Scale::ZERO.apply(999), 0);
        // exact integer scale 4 → 4·value
        let four = Scale::from_ratio(4, 1);
        for v in [-7i64, 0, 1, 1000, 1_000_000_000] {
            assert_eq!(four.apply(v), 4 * v, "x4 failed at {v}");
        }
    }
}
