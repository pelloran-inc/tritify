//! Plain scaled-dot-product attention — the vision/decoder variant (SAM 2.1, ViT): **non-causal**, **no RoPE**,
//! and usable as **self-** or **cross-attention** (query stream and key/value stream are separate inputs; pass
//! the same tensor for self-attention). Standard multi-head (no GQA). This is the exact-integer counterpart of
//! the `F.scaled_dot_product_attention` op used by SAM 2.1 / ViT-style models.
//!
//! The whole core is integer on the data path, identical in spirit to [`crate::attention::Attention`] minus the
//! rotary step and the causal mask: Q/K/V and the output projection are exact-integer [`QuantLinear`]; the scores
//! are an exact integer dot scaled by the integer `scale = 1/sqrt(head_dim)`; the softmax is a max-shifted
//! integer-code `exp` LUT + a fixed-point normalize; the weighted sum is an integer accumulation brought to a
//! common `Scale`. Projections may down/up-project (`wq/wk/wv: in → n_heads·head_dim`, `wo: n_heads·head_dim →
//! out`), which covers SAM's `downsample_rate`. The float `forward_ref` is the verification twin.

use crate::layers::{requant_dyn, QuantLinear, ACT_LEVELS};
use crate::module::Module;
use crate::tensor::IntTensor;
use kt_int::{CodeLut, Lut, Scale, Tail};

/// The optional q / k / v / o projection biases of an SDPA block.
pub type QkvoBias<'a> = (Option<&'a [f32]>, Option<&'a [f32]>, Option<&'a [f32]>, Option<&'a [f32]>);

/// The bare scaled-dot-product attention **core**, decoupled from the projections: given already-projected,
/// head-interleaved `q `[sq, n_heads·head_dim]`, `k`/`v` `[sk, n_heads·head_dim]`, it computes
/// `softmax(q·kᵀ·scale)·v` and returns the concatenated heads `[sq, n_heads·head_dim]` (no output projection).
/// `scale = 1/sqrt(head_dim)`. This is shared by [`SdpaAttention`] (plain self/cross attention) and by SAM's
/// Hiera `MultiScaleAttention`, whose fused QKV and query-pooling sit *between* projection and this core — so the
/// core must stay projection-free. Integer on the data path; [`SdpaCore::forward_ref`] is the float twin.
pub struct SdpaCore {
    scale: Scale,      // 1/sqrt(head_dim)
    exp_ref: Lut,      // f64 exp (reference)
    exp_code: CodeLut, // integer-code exp (run-time)
}

impl SdpaCore {
    pub fn new(head_dim: usize) -> Self {
        SdpaCore {
            scale: Scale::from_f64(1.0 / (head_dim as f64).sqrt()),
            exp_ref: Lut::build(|x| x.exp(), -30.0, 0.0, 16, Tail::Value(0.0), Tail::Clip),
            exp_code: CodeLut::build(|x| x.exp(), -30.0, 0.0, 16, 20, false),
        }
    }

    /// `q`:[sq, nh·hd], `k`,`v`:[sk, nh·hd] → `[sq, nh·hd]`.
    pub fn forward_int(&self, q: &IntTensor, k: &IntTensor, v: &IntTensor, nh: usize, hd: usize) -> IntTensor {
        let inner = nh * hd;
        let sq = q.len() / inner;
        let sk = k.len() / inner;
        let s_score = q.scale.mul(k.scale).mul(self.scale);
        let qd: Vec<i32> = q.data.clone();
        let kd: Vec<i32> = k.data.clone();
        let vd: Vec<i64> = v.data.iter().map(|&z| z as i64).collect();

        // Per-head attention is independent → deterministic parallel fill over heads (head-major `head_out`
        // [nh, sq, hd] + per-position softmax denominators `head_sum` [nh, sq], two buffers partitioned in
        // lockstep). The per-element arithmetic is unchanged, so the result is bit-identical to the serial loop;
        // `head_sum` is folded back into the original [sq, nh] `scale_ih` order so the common-scale reduction
        // sees the identical Scale sequence.
        let mut head_out = vec![0i64; nh * sq * hd];
        let mut head_sum = vec![0i64; nh * sq];
        let compute_head = |h: usize, out: &mut [i64], sum: &mut [i64]| {
            for i in 0..sq {
                let qoff = i * inner + h * hd;
                let mut dots = vec![0i64; sk];
                let mut dmax = i64::MIN;
                for (j, dj) in dots.iter_mut().enumerate() {
                    let koff = j * inner + h * hd;
                    let mut dot = 0i64;
                    for d in 0..hd {
                        dot += qd[qoff + d] as i64 * kd[koff + d] as i64;
                    }
                    *dj = dot;
                    if dot > dmax {
                        dmax = dot;
                    }
                }
                let mut ecodes = vec![0i64; sk];
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
                            acc += ec * vd[j * inner + h * hd + d];
                        }
                        out[i * hd + d] = acc;
                    }
                    sum[i] = sumc;
                }
            }
        };
        let work = nh.saturating_mul(sq).saturating_mul(sk).saturating_mul(hd);
        let nthreads = crate::layers::resolve_threads(nh);
        if nthreads > 1 && hd != 0 && work >= (1 << 20) {
            let chunk = nh.div_ceil(nthreads);
            std::thread::scope(|s| {
                for (ci, (outs, sums)) in
                    head_out.chunks_mut(chunk * sq * hd).zip(head_sum.chunks_mut(chunk * sq)).enumerate()
                {
                    let (base, compute_head) = (ci * chunk, &compute_head);
                    s.spawn(move || {
                        for (hi, (o, su)) in outs.chunks_mut(sq * hd).zip(sums.chunks_mut(sq)).enumerate() {
                            compute_head(base + hi, o, su);
                        }
                    });
                }
            });
        } else {
            for (h, (o, su)) in head_out.chunks_mut(sq * hd).zip(head_sum.chunks_mut(sq)).enumerate() {
                compute_head(h, o, su);
            }
        }
        let mut scale_ih = vec![Scale::ZERO; sq * nh];
        for i in 0..sq {
            for h in 0..nh {
                let sumc = head_sum[h * sq + i];
                if sumc > 0 {
                    scale_ih[i * nh + h] = v.scale.mul(Scale::from_ratio(1, sumc as u128));
                }
            }
        }

        let shape = vec![sq, inner];
        let common = scale_ih.iter().copied().fold(Scale::ZERO, |a, b| if b.ge(a) { b } else { a });
        if common.mant == 0 {
            return IntTensor::new(vec![0; sq * inner], shape, Scale::ZERO);
        }
        let cinv = common.recip();
        let mut vv = vec![0i64; sq * inner];
        for i in 0..sq {
            for h in 0..nh {
                let f = scale_ih[i * nh + h].mul(cinv);
                for d in 0..hd {
                    vv[i * inner + h * hd + d] = f.apply(head_out[h * sq * hd + i * hd + d]);
                }
            }
        }
        let gmax = vv.iter().fold(0i64, |m, &z| m.max(z.abs()));
        if gmax == 0 {
            return IntTensor::new(vec![0; sq * inner], shape, Scale::ZERO);
        }
        let cd: Vec<i32> = vv.iter().map(|&z| requant_dyn(z, ACT_LEVELS, gmax)).collect();
        IntTensor::new(cd, shape, Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(common))
    }

    /// Float reference: `q`:[sq, nh·hd], `k`,`v`:[sk, nh·hd] → `[sq, nh·hd]`.
    pub fn forward_ref(&self, q: &[f64], k: &[f64], v: &[f64], nh: usize, hd: usize) -> Vec<f64> {
        let inner = nh * hd;
        let sq = q.len() / inner;
        let sk = k.len() / inner;
        let sc = 1.0 / (hd as f64).sqrt();
        let mut out = vec![0f64; sq * inner];
        for h in 0..nh {
            for i in 0..sq {
                let qoff = i * inner + h * hd;
                let mut scores = vec![0.0f64; sk];
                let mut maxs = f64::NEG_INFINITY;
                for (j, sj) in scores.iter_mut().enumerate() {
                    let koff = j * inner + h * hd;
                    let dot: f64 = (0..hd).map(|d| q[qoff + d] * k[koff + d]).sum();
                    *sj = dot * sc;
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
                    let voff = j * inner + h * hd;
                    for d in 0..hd {
                        out[i * inner + h * hd + d] += p * v[voff + d];
                    }
                }
            }
        }
        out
    }
}

pub struct SdpaAttention {
    pub name: String,
    pub n_heads: usize,
    pub head_dim: usize,
    pub q_dim: usize,   // query-stream embedding dim (wq.in_features)
    pub kv_dim: usize,  // key/value-stream embedding dim (wk/wv.in_features)
    pub out_dim: usize, // output embedding dim (wo.out_features)
    pub wq: QuantLinear,
    pub wk: QuantLinear,
    pub wv: QuantLinear,
    pub wo: QuantLinear,
    core: SdpaCore,
}

impl SdpaAttention {
    pub fn new(
        name: impl Into<String>,
        n_heads: usize,
        head_dim: usize,
        wq: QuantLinear,
        wk: QuantLinear,
        wv: QuantLinear,
        wo: QuantLinear,
    ) -> Self {
        let inner = n_heads * head_dim;
        assert_eq!(wq.out_features, inner, "wq out must be n_heads·head_dim");
        assert_eq!(wk.out_features, inner, "wk out must be n_heads·head_dim");
        assert_eq!(wv.out_features, inner, "wv out must be n_heads·head_dim");
        assert_eq!(wo.in_features, inner, "wo in must be n_heads·head_dim");
        assert_eq!(wk.in_features, wv.in_features, "k/v must share the kv stream");
        SdpaAttention {
            name: name.into(),
            n_heads,
            head_dim,
            q_dim: wq.in_features,
            kv_dim: wk.in_features,
            out_dim: wo.out_features,
            wq,
            wk,
            wv,
            wo,
            core: SdpaCore::new(head_dim),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        name: impl Into<String>,
        n_heads: usize,
        head_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        out_dim: usize,
        wq_w: &[f32],
        wk_w: &[f32],
        wv_w: &[f32],
        wo_w: &[f32],
        bias: QkvoBias<'_>,
    ) -> Self {
        let inner = n_heads * head_dim;
        let wq = QuantLinear::from_weights("wq", wq_w, inner, q_dim).with_bias(bias.0);
        let wk = QuantLinear::from_weights("wk", wk_w, inner, kv_dim).with_bias(bias.1);
        let wv = QuantLinear::from_weights("wv", wv_w, inner, kv_dim).with_bias(bias.2);
        let wo = QuantLinear::from_weights("wo", wo_w, out_dim, inner).with_bias(bias.3);
        SdpaAttention::new(name, n_heads, head_dim, wq, wk, wv, wo)
    }

    /// Integer scaled-dot-product attention. `q_in` is `[sq, q_dim]`, `kv_in` is `[sk, kv_dim]`; for
    /// self-attention pass the same tensor for both. Output is `[sq, out_dim]`.
    pub fn forward_int(&self, q_in: &IntTensor, kv_in: &IntTensor) -> IntTensor {
        self.forward_qkv_int(q_in, kv_in, kv_in)
    }

    /// Integer SDPA with **separate** key and value streams — SAM's mask-decoder attention adds the positional
    /// embedding to the query and key but keeps the value un-shifted, so `k_in` and `v_in` differ. `q_in`
    /// `[sq,q_dim]`, `k_in`/`v_in` `[sk,kv_dim]`. Output `[sq,out_dim]`.
    pub fn forward_qkv_int(&self, q_in: &IntTensor, k_in: &IntTensor, v_in: &IntTensor) -> IntTensor {
        assert_eq!(q_in.len() % self.q_dim, 0, "{}: q not a multiple of q_dim", self.name);
        assert_eq!(k_in.len() % self.kv_dim, 0, "{}: k not a multiple of kv_dim", self.name);
        assert_eq!(v_in.len() % self.kv_dim, 0, "{}: v not a multiple of kv_dim", self.name);
        let q = self.wq.forward_int(q_in);
        let k = self.wk.forward_int(k_in);
        let v = self.wv.forward_int(v_in);
        let concat = self.core.forward_int(&q, &k, &v, self.n_heads, self.head_dim);
        self.wo.forward_int(&concat)
    }

    /// Float reference (the verification twin).
    pub fn forward_ref(&self, q_in: &[f32], kv_in: &[f32]) -> Vec<f32> {
        self.forward_qkv_ref(q_in, kv_in, kv_in)
    }

    pub fn forward_qkv_ref(&self, q_in: &[f32], k_in: &[f32], v_in: &[f32]) -> Vec<f32> {
        let q: Vec<f64> = self.wq.forward_ref(q_in).iter().map(|&z| z as f64).collect();
        let k: Vec<f64> = self.wk.forward_ref(k_in).iter().map(|&z| z as f64).collect();
        let v: Vec<f64> = self.wv.forward_ref(v_in).iter().map(|&z| z as f64).collect();
        let out = self.core.forward_ref(&q, &k, &v, self.n_heads, self.head_dim);
        self.wo.forward_ref(&out.iter().map(|&z| z as f32).collect::<Vec<_>>())
    }
}

/// Self-attention convenience: the single-input [`Module`] path feeds the same tensor as query and key/value, so
/// the gate ([`crate::verify::verify`]) can check it like any other op.
impl Module for SdpaAttention {
    fn name(&self) -> &str {
        &self.name
    }
    fn forward_int(&self, x: &IntTensor) -> IntTensor {
        SdpaAttention::forward_int(self, x, x)
    }
    fn forward_ref(&self, x: &[f32]) -> Vec<f32> {
        SdpaAttention::forward_ref(self, x, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    }
    fn vecf(s: &mut u64, n: usize) -> Vec<f32> {
        (0..n).map(|_| ((lcg(s) >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) as f32).collect()
    }
    fn probe(s: &mut u64, rows: usize, cols: usize) -> IntTensor {
        let real: Vec<f64> =
            (0..rows * cols).map(|_| (lcg(s) >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0).collect();
        IntTensor::from_reals(&real, vec![rows, cols], ACT_LEVELS)
    }

    #[test]
    fn self_attention_tracks_float_and_is_deterministic() {
        let (dim, nh, hd, seq) = (32usize, 4usize, 8usize, 6usize);
        let inner = nh * hd;
        let mut s = 0x5A1Fu64;
        let (wq, wk, wv, wo) = (
            vecf(&mut s, inner * dim),
            vecf(&mut s, inner * dim),
            vecf(&mut s, inner * dim),
            vecf(&mut s, dim * inner),
        );
        let p = probe(&mut s, seq, dim);
        let attn =
            SdpaAttention::from_weights("sa", nh, hd, dim, dim, dim, &wq, &wk, &wv, &wo, (None, None, None, None));
        let oi = Module::forward_int(&attn, &p).to_f32();
        let r = Module::forward_ref(&attn, &p.to_f32());
        let (mut mae, mut peak) = (0.0f64, 0.0f64);
        for (&g, &rf) in oi.iter().zip(&r) {
            mae = mae.max((g as f64 - rf as f64).abs());
            peak = peak.max((rf as f64).abs());
        }
        println!("SdpaAttention(self): out={} max_abs_err={:.3e} (peak_ref={:.3e})", oi.len(), mae, peak);
        assert_eq!(Module::forward_int(&attn, &p).data, Module::forward_int(&attn, &p).data, "deterministic");
        assert!(mae <= peak * 3e-2, "self-attention drifted from float");
    }

    #[test]
    fn cross_attention_tracks_float_and_is_deterministic() {
        // query stream (sq tokens, q_dim) attends to a different kv stream (sk tokens, kv_dim).
        let (q_dim, kv_dim, out_dim, nh, hd, sq, sk) = (24usize, 40usize, 24usize, 3usize, 8usize, 5usize, 7usize);
        let inner = nh * hd;
        let mut s = 0xC305u64;
        let (wq, wk, wv, wo) = (
            vecf(&mut s, inner * q_dim),
            vecf(&mut s, inner * kv_dim),
            vecf(&mut s, inner * kv_dim),
            vecf(&mut s, out_dim * inner),
        );
        let q_in = probe(&mut s, sq, q_dim);
        let kv_in = probe(&mut s, sk, kv_dim);
        let attn = SdpaAttention::from_weights(
            "xa",
            nh,
            hd,
            q_dim,
            kv_dim,
            out_dim,
            &wq,
            &wk,
            &wv,
            &wo,
            (None, None, None, None),
        );
        let oi = attn.forward_int(&q_in, &kv_in);
        let r = attn.forward_ref(&q_in.to_f32(), &kv_in.to_f32());
        let got = oi.to_f32();
        assert_eq!(oi.shape, vec![sq, out_dim]);
        let (mut mae, mut peak) = (0.0f64, 0.0f64);
        for (&g, &rf) in got.iter().zip(&r) {
            mae = mae.max((g as f64 - rf as f64).abs());
            peak = peak.max((rf as f64).abs());
        }
        println!("SdpaAttention(cross): out={:?} max_abs_err={:.3e} (peak_ref={:.3e})", oi.shape, mae, peak);
        assert_eq!(oi.data, attn.forward_int(&q_in, &kv_in).data, "deterministic");
        assert!(mae <= peak * 3e-2, "cross-attention drifted from float");
    }
}
