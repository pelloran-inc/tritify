//! Causal multi-head attention with grouped-query (GQA) and rotary position embeddings (RoPE) — integer.
//!
//! Q/K/V and the output projection are exact-integer [`QuantLinear`]. The **whole attention core is integer on
//! the data path**: RoPE is a signed-[`Scale`] rotation (cos/sin precomputed offline), the scores are an exact
//! integer dot, the softmax uses an integer-code `exp` LUT + a fixed-point normalize, and the weighted sum is
//! an integer accumulation brought to a common `Scale`. Shapes: `[seq, dim] → [seq, dim]`. The float
//! `forward_ref` is the verification twin.

use crate::layers::{requant_dyn, QuantLinear, RmsNorm, ACT_LEVELS};
use crate::module::Module;
use crate::tensor::IntTensor;
use kt_int::{CodeLut, Lut, Scale, Tail};

/// The optional q / k / v projection biases (Qwen-style decoders carry them; Llama-style ones do not).
pub type QkvBias<'a> = (Option<&'a [f32]>, Option<&'a [f32]>, Option<&'a [f32]>);

pub struct Attention {
    pub name: String,
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub wq: QuantLinear,
    pub wk: QuantLinear,
    pub wv: QuantLinear,
    pub wo: QuantLinear,
    // optional QK-norm (RMSNorm over the q/k projection, applied BEFORE RoPE) — OLMoE/Olmo/Qwen3 use it; `None` = a
    // plain dense attention, byte-unchanged. Generic: the norm dimension comes from the checkpoint's q_norm/k_norm shape.
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    max_seq: usize,
    rope_theta: f64,
    // the rotated leading dims of each head: `head_dim` (full RoPE) or fewer (partial, GLM-4.5); the rest pass through
    rot_dim: usize,
    // the planted interleaved-RoPE mistake: rotate the pairs (2j, 2j+1) instead of (j, j + rot_dim/2)
    #[cfg(feature = "plants")]
    interleaved: bool,
    // RoPE: f64 (reference) + signed Scale (integer run-time), per (position, j<half), half = rot_dim / 2.
    cos: Vec<f64>,
    sin: Vec<f64>,
    cos_mag: Vec<Scale>,
    cos_sign: Vec<i8>,
    sin_mag: Vec<Scale>,
    sin_sign: Vec<i8>,
    inv_sqrt: Scale,   // 1/sqrt(head_dim)
    exp_ref: Lut,      // f64 exp (reference)
    exp_code: CodeLut, // integer-code exp (run-time)
}

fn sign_scale(x: f64) -> (Scale, i8) {
    (Scale::from_f64(x.abs()), if x < 0.0 { -1 } else { 1 })
}

/// The RoPE tables over the first `rot_dim` dims: per (position, j < rot_dim/2), `θ^(−2j/rot_dim)`.
struct RopeTables {
    cos: Vec<f64>,
    sin: Vec<f64>,
    cos_mag: Vec<Scale>,
    cos_sign: Vec<i8>,
    sin_mag: Vec<Scale>,
    sin_sign: Vec<i8>,
}

fn rope_tables(max_seq: usize, rot_dim: usize, rope_theta: f64) -> RopeTables {
    let half = rot_dim / 2;
    let n = max_seq * half;
    let (mut cos, mut sin) = (vec![0.0f64; n], vec![0.0f64; n]);
    let (mut cos_mag, mut cos_sign) = (vec![Scale::ZERO; n], vec![1i8; n]);
    let (mut sin_mag, mut sin_sign) = (vec![Scale::ZERO; n], vec![1i8; n]);
    for p in 0..max_seq {
        for j in 0..half {
            let inv_freq = rope_theta.powf(-(2.0 * j as f64) / rot_dim as f64);
            let ang = p as f64 * inv_freq;
            let (c, s) = (ang.cos(), ang.sin());
            let idx = p * half + j;
            cos[idx] = c;
            sin[idx] = s;
            let (cm, cs) = sign_scale(c);
            let (sm, ss) = sign_scale(s);
            cos_mag[idx] = cm;
            cos_sign[idx] = cs;
            sin_mag[idx] = sm;
            sin_sign[idx] = ss;
        }
    }
    RopeTables { cos, sin, cos_mag, cos_sign, sin_mag, sin_sign }
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        wq: QuantLinear,
        wk: QuantLinear,
        wv: QuantLinear,
        wo: QuantLinear,
        max_seq: usize,
        rope_theta: f64,
    ) -> Self {
        assert_eq!(n_heads % n_kv_heads, 0, "n_heads must be a multiple of n_kv_heads");
        assert_eq!(head_dim % 2, 0, "head_dim must be even for RoPE");
        let RopeTables { cos, sin, cos_mag, cos_sign, sin_mag, sin_sign } = rope_tables(max_seq, head_dim, rope_theta);
        Attention {
            name: name.into(),
            dim,
            n_heads,
            n_kv_heads,
            head_dim,
            wq,
            wk,
            wv,
            wo,
            q_norm: None,
            k_norm: None,
            max_seq,
            rope_theta,
            rot_dim: head_dim,
            #[cfg(feature = "plants")]
            interleaved: false,
            cos,
            sin,
            cos_mag,
            cos_sign,
            sin_mag,
            sin_sign,
            inv_sqrt: Scale::from_f64(1.0 / (head_dim as f64).sqrt()),
            exp_ref: Lut::build(|x| x.exp(), -30.0, 0.0, 16, Tail::Value(0.0), Tail::Clip),
            exp_code: CodeLut::build(|x| x.exp(), -30.0, 0.0, 16, 20, false),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        name: impl Into<String>,
        dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        wq_w: &[f32],
        wk_w: &[f32],
        wv_w: &[f32],
        wo_w: &[f32],
        qkv_bias: QkvBias<'_>,
        max_seq: usize,
        rope_theta: f64,
    ) -> Self {
        let (qdim, kvdim) = (n_heads * head_dim, n_kv_heads * head_dim);
        let wq = QuantLinear::from_weights("wq", wq_w, qdim, dim).with_bias(qkv_bias.0);
        let wk = QuantLinear::from_weights("wk", wk_w, kvdim, dim).with_bias(qkv_bias.1);
        let wv = QuantLinear::from_weights("wv", wv_w, kvdim, dim).with_bias(qkv_bias.2);
        let wo = QuantLinear::from_weights("wo", wo_w, dim, qdim);
        Attention::new(name, dim, n_heads, n_kv_heads, head_dim, wq, wk, wv, wo, max_seq, rope_theta)
    }

    /// Attach optional QK-norm (RMSNorm applied to the full q/k projection before RoPE). `None` leaves the attention
    /// a plain dense attention (byte-unchanged). Builders pass the checkpoint's `q_norm`/`k_norm` weights here.
    pub fn with_qk_norm(mut self, q_norm: Option<RmsNorm>, k_norm: Option<RmsNorm>) -> Self {
        self.q_norm = q_norm;
        self.k_norm = k_norm;
        self
    }

    /// Rotate only the first `rot_dim` dims of each head (partial RoPE, GLM-4.5's `partial_rotary_factor`): pairs
    /// `(j, j + rot_dim/2)` for `j < rot_dim/2` at `θ^(−2j/rot_dim)`; dims `rot_dim..head_dim` pass through unrotated.
    /// `rot_dim = head_dim` is the full RoPE, the same tables as without this call.
    pub fn with_rotary_dim(mut self, rot_dim: usize) -> Self {
        assert!(
            rot_dim > 0 && rot_dim.is_multiple_of(2) && rot_dim <= self.head_dim,
            "{}: rotary dim {rot_dim} must be even and in 2..={}",
            self.name,
            self.head_dim
        );
        let RopeTables { cos, sin, cos_mag, cos_sign, sin_mag, sin_sign } =
            rope_tables(self.max_seq, rot_dim, self.rope_theta);
        (self.cos, self.sin, self.cos_mag, self.cos_sign, self.sin_mag, self.sin_sign) =
            (cos, sin, cos_mag, cos_sign, sin_mag, sin_sign);
        self.rot_dim = rot_dim;
        self
    }

    /// **Planted mistake (interleaved RoPE):** rotate the pairs `(2j, 2j+1)`, GPT-J style, instead of rotate_half's
    /// `(j, j + rot_dim/2)`. Only the fidelity gate's control uses it.
    #[cfg(feature = "plants")]
    pub fn planted_interleaved_pairs(mut self) -> Self {
        self.interleaved = true;
        self
    }

    /// The interleaved plant's integer and float rotations: the pairs `(2j, 2j+1)` at the same frequencies.
    #[cfg(feature = "plants")]
    fn rope_interleaved(&self, p: usize, rot: &mut dyn FnMut(usize, usize, usize)) {
        for j in 0..self.rot_dim / 2 {
            rot(2 * j, 2 * j + 1, p * (self.rot_dim / 2) + j);
        }
    }

    /// The number of leading dims of each head that RoPE rotates.
    pub fn rotary_dim(&self) -> usize {
        self.rot_dim
    }

    /// Integer RoPE on one head-vector slice (rotate_half over the first `rot_dim` dims) at position `p`, in place
    /// (values stay at the vector's own scale; cos/sin are dimensionless signed `Scale`s).
    pub(crate) fn rope_int(&self, v: &mut [i32], p: usize) {
        #[cfg(feature = "plants")]
        if self.interleaved {
            return self.rope_interleaved(p, &mut |ja, jb, idx| {
                let (cm, cs) = (self.cos_mag[idx], self.cos_sign[idx] as i64);
                let (sm, ss) = (self.sin_mag[idx], self.sin_sign[idx] as i64);
                let (a, b) = (v[ja] as i64, v[jb] as i64);
                v[ja] = (cs * cm.apply(a) - ss * sm.apply(b)) as i32;
                v[jb] = (cs * cm.apply(b) + ss * sm.apply(a)) as i32;
            });
        }
        let half = self.rot_dim / 2;
        for j in 0..half {
            let idx = p * half + j;
            let (cm, cs) = (self.cos_mag[idx], self.cos_sign[idx] as i64);
            let (sm, ss) = (self.sin_mag[idx], self.sin_sign[idx] as i64);
            let (a, b) = (v[j] as i64, v[j + half] as i64);
            // RoPE preserves magnitude (|cos|,|sin| ≤ 1), so the result stays within the input's i32 range.
            v[j] = (cs * cm.apply(a) - ss * sm.apply(b)) as i32; // a·cos − b·sin
            v[j + half] = (cs * cm.apply(b) + ss * sm.apply(a)) as i32; // b·cos + a·sin
        }
    }

    /// f64 RoPE (reference), over the first `rot_dim` dims.
    pub(crate) fn rope_ref(&self, v: &mut [f64], p: usize) {
        #[cfg(feature = "plants")]
        if self.interleaved {
            return self.rope_interleaved(p, &mut |ja, jb, idx| {
                let (c, s) = (self.cos[idx], self.sin[idx]);
                let (a, b) = (v[ja], v[jb]);
                v[ja] = a * c - b * s;
                v[jb] = b * c + a * s;
            });
        }
        let half = self.rot_dim / 2;
        for j in 0..half {
            let (c, s) = (self.cos[p * half + j], self.sin[p * half + j]);
            let (a, b) = (v[j], v[j + half]);
            v[j] = a * c - b * s;
            v[j + half] = b * c + a * s;
        }
    }
}

/// Per-layer KV cache for incremental decode. Holds, per past step, the post-RoPE K codes with that step's K
/// [`Scale`], and the V codes with that step's V [`Scale`] — because each decode step quantizes its own K/V (one
/// row) with its own dynamic scale, unlike a whole-sequence prefill where all positions share one scale.
pub struct LayerKv {
    k: Vec<i32>,
    kscale: Vec<Scale>,
    v: Vec<i32>,
    vscale: Vec<Scale>,
    kstride: usize,
}

impl LayerKv {
    #[must_use]
    pub fn new(kstride: usize) -> Self {
        LayerKv { k: Vec::new(), kscale: Vec::new(), v: Vec::new(), vscale: Vec::new(), kstride }
    }
    #[must_use]
    pub fn steps(&self) -> usize {
        self.kscale.len()
    }

    /// Serialize the cache to exact, machine-independent bytes (integer codes + per-step `Scale`s) — the resumable
    /// state a worker hands over so a spare can continue the *same* decode bit-identically.
    pub fn to_bytes(&self) -> Vec<u8> {
        let steps = self.kscale.len();
        let mut b = Vec::with_capacity(8 + self.k.len() * 4 + self.v.len() * 4 + steps * 24);
        b.extend_from_slice(&(self.kstride as u32).to_le_bytes());
        b.extend_from_slice(&(steps as u32).to_le_bytes());
        for &x in &self.k {
            b.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &self.v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        for s in &self.kscale {
            b.extend_from_slice(&s.mant.to_le_bytes());
            b.extend_from_slice(&s.shift.to_le_bytes());
        }
        for s in &self.vscale {
            b.extend_from_slice(&s.mant.to_le_bytes());
            b.extend_from_slice(&s.shift.to_le_bytes());
        }
        b
    }

    /// Reconstruct a cache from [`to_bytes`](Self::to_bytes) — exact, no float, so the restored cache is identical.
    pub fn from_bytes(b: &[u8]) -> LayerKv {
        fn u32r(b: &[u8], c: &mut usize) -> u32 {
            let v = u32::from_le_bytes(b[*c..*c + 4].try_into().unwrap());
            *c += 4;
            v
        }
        fn i32r(b: &[u8], c: &mut usize) -> i32 {
            let v = i32::from_le_bytes(b[*c..*c + 4].try_into().unwrap());
            *c += 4;
            v
        }
        fn i64r(b: &[u8], c: &mut usize) -> i64 {
            let v = i64::from_le_bytes(b[*c..*c + 8].try_into().unwrap());
            *c += 8;
            v
        }
        let mut c = 0usize;
        let kstride = u32r(b, &mut c) as usize;
        let steps = u32r(b, &mut c) as usize;
        let n = steps * kstride;
        let k: Vec<i32> = (0..n).map(|_| i32r(b, &mut c)).collect();
        let v: Vec<i32> = (0..n).map(|_| i32r(b, &mut c)).collect();
        let kscale: Vec<Scale> = (0..steps).map(|_| Scale { mant: i64r(b, &mut c), shift: i32r(b, &mut c) }).collect();
        let vscale: Vec<Scale> = (0..steps).map(|_| Scale { mant: i64r(b, &mut c), shift: i32r(b, &mut c) }).collect();
        LayerKv { k, kscale, v, vscale, kstride }
    }
}

impl Attention {
    /// The number of positions the RoPE tables cover: a sequence may not be longer.
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }
}

impl Attention {
    /// **Incremental decode of ONE token** at absolute position `pos`, against the growing `cache`. Appends this
    /// token's post-RoPE K and its V to the cache, then attends over the whole cache (causal: positions `0..=pos`).
    /// Each cached step carries its own dynamic K/V `Scale`; they are brought to a common scale by exact `Scale`
    /// arithmetic before the softmax and the weighted sum, so the output is an exact integer function of the cache
    /// — deterministic, and identical whether the layers run on one machine or are split across workers. Input is
    /// the (normalized) one-token hidden `[dim]`; returns the attention output `[dim]` (post-`wo`).
    pub fn decode_step(&self, x: &IntTensor, cache: &mut LayerKv, pos: usize) -> IntTensor {
        assert_eq!(x.len(), self.dim, "{}: decode_step expects one token [dim]", self.name);
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        let group = nh / nkv;
        let (qstride, kstride) = (nh * hd, nkv * hd);
        debug_assert_eq!(kstride, cache.kstride);

        let q = self.wq.forward_int(x);
        let q = match &self.q_norm {
            Some(n) => n.forward_int(&q),
            None => q,
        };
        let k = self.wk.forward_int(x);
        let k = match &self.k_norm {
            Some(n) => n.forward_int(&k),
            None => k,
        };
        let v = self.wv.forward_int(x);

        // RoPE this token's Q and K at the absolute position `pos`.
        let mut qd = q.data.clone();
        let mut kd = k.data.clone();
        for h in 0..nh {
            self.rope_int(&mut qd[h * hd..h * hd + hd], pos);
        }
        for h in 0..nkv {
            self.rope_int(&mut kd[h * hd..h * hd + hd], pos);
        }

        // Append this step's post-RoPE K and its V to the cache, each with its own dynamic Scale.
        cache.k.extend_from_slice(&kd);
        cache.kscale.push(k.scale);
        cache.v.extend_from_slice(&v.data);
        cache.vscale.push(v.scale);
        let steps = cache.steps();

        // V scales → one common reference (shared across heads); vf[j] rescales code j from its scale to v_ref.
        let v_ref = cache.vscale.iter().copied().fold(Scale::ZERO, |a, b| if b.ge(a) { b } else { a });
        let vf: Vec<Scale> = if v_ref.mant == 0 {
            vec![Scale::ZERO; steps]
        } else {
            let vref_inv = v_ref.recip();
            cache.vscale.iter().map(|s| s.mul(vref_inv)).collect()
        };

        let mut head_out = vec![0i64; nh * hd];
        let mut scale_h = vec![Scale::ZERO; nh];
        for h in 0..nh {
            let kv = h / group;
            // scores: exact integer dot per cached step, each at its own q.scale·kscale_j·inv_sqrt; brought to a
            // common score scale s_ref (the max) exactly before the exp LUT (so softmax sees consistent units).
            let mut dots = vec![0i64; steps];
            let mut sj = vec![Scale::ZERO; steps];
            for j in 0..steps {
                let koff = j * kstride + kv * hd;
                let mut dot = 0i64;
                for d in 0..hd {
                    dot += qd[h * hd + d] as i64 * cache.k[koff + d] as i64;
                }
                dots[j] = dot;
                sj[j] = q.scale.mul(cache.kscale[j]).mul(self.inv_sqrt);
            }
            let s_ref = sj.iter().copied().fold(Scale::ZERO, |a, b| if b.ge(a) { b } else { a });
            if s_ref.mant == 0 {
                continue;
            }
            let sref_inv = s_ref.recip();
            let mut score = vec![0i64; steps];
            let mut dmax = i64::MIN;
            for j in 0..steps {
                score[j] = sj[j].mul(sref_inv).apply(dots[j]);
                if score[j] > dmax {
                    dmax = score[j];
                }
            }
            // softmax: integer-code exp of (score − dmax) at s_ref, fixed-point normalize.
            let mut ec = vec![0i64; steps];
            let mut sumc = 0i64;
            for j in 0..steps {
                let idx = self.exp_code.index(score[j] - dmax, s_ref);
                ec[j] = if idx < 0 { 0 } else { self.exp_code.code_at(idx) };
                sumc += ec[j];
            }
            if sumc <= 0 {
                continue;
            }
            // weighted V (each code brought to v_ref): head output = Σ_j ec_j · v_int_j.
            for d in 0..hd {
                let mut acc = 0i64;
                for j in 0..steps {
                    acc += ec[j] * vf[j].apply(cache.v[j * kstride + kv * hd + d] as i64);
                }
                head_out[h * hd + d] = acc;
            }
            scale_h[h] = v_ref.mul(Scale::from_ratio(1, sumc as u128));
        }

        // bring the heads to a common Scale, dynamic-quant the concat, then the output projection.
        let common = scale_h.iter().copied().fold(Scale::ZERO, |a, b| if b.ge(a) { b } else { a });
        let concat = if common.mant == 0 {
            IntTensor::new(vec![0; qstride], vec![qstride], Scale::ZERO)
        } else {
            let cinv = common.recip();
            let mut vv = vec![0i64; qstride];
            for h in 0..nh {
                let f = scale_h[h].mul(cinv);
                for d in 0..hd {
                    vv[h * hd + d] = f.apply(head_out[h * hd + d]);
                }
            }
            let gmax = vv.iter().fold(0i64, |m, &z| m.max(z.abs()));
            if gmax == 0 {
                IntTensor::new(vec![0; qstride], vec![qstride], Scale::ZERO)
            } else {
                let cd: Vec<i32> = vv.iter().map(|&z| requant_dyn(z, ACT_LEVELS, gmax)).collect();
                IntTensor::new(cd, vec![qstride], Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(common))
            }
        };
        self.wo.forward_int(&concat)
    }
}

impl Module for Attention {
    fn name(&self) -> &str {
        &self.name
    }

    fn forward_int(&self, x: &IntTensor) -> IntTensor {
        assert_eq!(x.len() % self.dim, 0, "{}: input not a multiple of dim", self.name);
        let seq = x.len() / self.dim;
        assert!(seq <= self.max_seq, "sequence {seq} exceeds max_seq {}", self.max_seq);
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        let group = nh / nkv;
        let (qstride, kstride) = (nh * hd, nkv * hd);

        let q = self.wq.forward_int(x);
        let q = match &self.q_norm {
            Some(n) => n.forward_int(&q),
            None => q,
        };
        let k = self.wk.forward_int(x);
        let k = match &self.k_norm {
            Some(n) => n.forward_int(&k),
            None => k,
        };
        let v = self.wv.forward_int(x);
        let s_score = q.scale.mul(k.scale).mul(self.inv_sqrt);

        let mut qd: Vec<i32> = q.data.clone();
        let mut kd: Vec<i32> = k.data.clone();
        let vd: Vec<i64> = v.data.iter().map(|&z| z as i64).collect();
        for i in 0..seq {
            for h in 0..nh {
                self.rope_int(&mut qd[i * qstride + h * hd..i * qstride + h * hd + hd], i);
            }
            for h in 0..nkv {
                self.rope_int(&mut kd[i * kstride + h * hd..i * kstride + h * hd + hd], i);
            }
        }

        // Per-head attention is independent → deterministic parallel fill over heads. Each head writes its own
        // contiguous slice of a head-major output buffer `head_out` ([nh, seq, hd]) and its per-position softmax
        // denominators `head_sum` ([nh, seq]); the per-element arithmetic is unchanged, so the result is
        // bit-identical to the serial loop. `head_sum` is folded back into the original [seq, nh] `scale_ih`
        // order below so the common-scale reduction sees the exact same Scale sequence.
        let mut head_out = vec![0i64; nh * seq * hd];
        let mut head_sum = vec![0i64; nh * seq];
        let compute_head = |h: usize, out: &mut [i64], sum: &mut [i64]| {
            let kv = h / group;
            for i in 0..seq {
                let qoff = i * qstride + h * hd;
                // scores (exact integer dot), causal j = 0..=i
                let mut dots = vec![0i64; i + 1];
                let mut dmax = i64::MIN;
                for (j, dj) in dots.iter_mut().enumerate() {
                    let koff = j * kstride + kv * hd;
                    let mut dot = 0i64;
                    for d in 0..hd {
                        dot += qd[qoff + d] as i64 * kd[koff + d] as i64;
                    }
                    *dj = dot;
                    if dot > dmax {
                        dmax = dot;
                    }
                }
                // softmax: integer-code exp of (dot−dmax), fixed-point normalize
                let mut ecodes = vec![0i64; i + 1];
                let mut sumc = 0i64;
                for (j, ec) in ecodes.iter_mut().enumerate() {
                    let idx = self.exp_code.index(dots[j] - dmax, s_score);
                    *ec = if idx < 0 { 0 } else { self.exp_code.code_at(idx) };
                    sumc += *ec;
                }
                if sumc > 0 {
                    for d in 0..hd {
                        let mut acc = 0i64;
                        for (j, &ec) in ecodes.iter().enumerate() {
                            acc += ec * vd[j * kstride + kv * hd + d];
                        }
                        out[i * hd + d] = acc;
                    }
                    // real = (Σ ec·v_data) · v_scale / sumc  (exp code_scale cancels in the normalize)
                    sum[i] = sumc;
                }
            }
        };
        // Spread the heads across cores only when the work is large enough to amortize spawning (two buffers are
        // partitioned in lockstep — a chunk of heads is a contiguous slice of each). Deterministic / bit-identical.
        let work = nh.saturating_mul(seq).saturating_mul(seq).saturating_mul(hd);
        let nthreads = crate::layers::resolve_threads(nh);
        if nthreads > 1 && hd != 0 && work >= (1 << 20) {
            let chunk = nh.div_ceil(nthreads);
            std::thread::scope(|s| {
                for (ci, (outs, sums)) in
                    head_out.chunks_mut(chunk * seq * hd).zip(head_sum.chunks_mut(chunk * seq)).enumerate()
                {
                    let (base, compute_head) = (ci * chunk, &compute_head);
                    s.spawn(move || {
                        for (hi, (o, su)) in outs.chunks_mut(seq * hd).zip(sums.chunks_mut(seq)).enumerate() {
                            compute_head(base + hi, o, su);
                        }
                    });
                }
            });
        } else {
            for (h, (o, su)) in head_out.chunks_mut(seq * hd).zip(head_sum.chunks_mut(seq)).enumerate() {
                compute_head(h, o, su);
            }
        }
        // Rebuild the per-(position,head) normalization scales in the original [seq, nh] order.
        let mut scale_ih = vec![Scale::ZERO; seq * nh];
        for i in 0..seq {
            for h in 0..nh {
                let sumc = head_sum[h * seq + i];
                if sumc > 0 {
                    scale_ih[i * nh + h] = v.scale.mul(Scale::from_ratio(1, sumc as u128));
                }
            }
        }

        // bring per-(head,position) results to a common Scale, then dynamic-quant the concat.
        let shape = vec![seq, qstride];
        let common = scale_ih.iter().copied().fold(Scale::ZERO, |a, b| if b.ge(a) { b } else { a });
        let concat = if common.mant == 0 {
            IntTensor::new(vec![0; seq * qstride], shape, Scale::ZERO)
        } else {
            let cinv = common.recip();
            let mut vv = vec![0i64; seq * qstride];
            for i in 0..seq {
                for h in 0..nh {
                    let f = scale_ih[i * nh + h].mul(cinv);
                    for d in 0..hd {
                        vv[i * qstride + h * hd + d] = f.apply(head_out[h * seq * hd + i * hd + d]);
                    }
                }
            }
            let gmax = vv.iter().fold(0i64, |m, &z| m.max(z.abs()));
            if gmax == 0 {
                IntTensor::new(vec![0; seq * qstride], shape, Scale::ZERO)
            } else {
                let cd: Vec<i32> = vv.iter().map(|&z| requant_dyn(z, ACT_LEVELS, gmax)).collect();
                IntTensor::new(cd, shape, Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(common))
            }
        };
        self.wo.forward_int(&concat)
    }

    fn forward_ref(&self, x: &[f32]) -> Vec<f32> {
        let seq = x.len() / self.dim;
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        let group = nh / nkv;
        let (qstride, kstride) = (nh * hd, nkv * hd);
        let qf = self.wq.forward_ref(x);
        let qf = match &self.q_norm {
            Some(n) => n.forward_ref(&qf),
            None => qf,
        };
        let mut q: Vec<f64> = qf.iter().map(|&z| z as f64).collect();
        let kf = self.wk.forward_ref(x);
        let kf = match &self.k_norm {
            Some(n) => n.forward_ref(&kf),
            None => kf,
        };
        let mut k: Vec<f64> = kf.iter().map(|&z| z as f64).collect();
        let v: Vec<f64> = self.wv.forward_ref(x).iter().map(|&z| z as f64).collect();
        for i in 0..seq {
            for h in 0..nh {
                self.rope_ref(&mut q[i * qstride + h * hd..i * qstride + h * hd + hd], i);
            }
            for h in 0..nkv {
                self.rope_ref(&mut k[i * kstride + h * hd..i * kstride + h * hd + hd], i);
            }
        }
        let inv_sqrt = 1.0 / (hd as f64).sqrt();
        let mut out = vec![0f32; seq * qstride];
        for h in 0..nh {
            let kv = h / group;
            for i in 0..seq {
                let qoff = i * qstride + h * hd;
                let mut scores = vec![0.0f64; i + 1];
                let mut maxs = f64::NEG_INFINITY;
                for (j, sj) in scores.iter_mut().enumerate() {
                    let koff = j * kstride + kv * hd;
                    let dot: f64 = (0..hd).map(|d| q[qoff + d] * k[koff + d]).sum();
                    *sj = dot * inv_sqrt;
                    if *sj > maxs {
                        maxs = *sj;
                    }
                }
                let mut denom = 0.0f64;
                for sj in scores.iter_mut() {
                    *sj = self.exp_ref.apply(*sj - maxs);
                    denom += *sj;
                }
                for (j, &pj) in scores.iter().enumerate() {
                    let p = pj / denom;
                    let voff = j * kstride + kv * hd;
                    for d in 0..hd {
                        out[i * qstride + h * hd + d] += (p * v[voff + d]) as f32;
                    }
                }
            }
        }
        self.wo.forward_ref(&out) // output projection (the fix: the reference must apply wo too)
    }
}
