//! SAM 2.1 **video** memory stack — the 2D axial rotary position embedding that the memory attention
//! uses. Faithful to `Sam2VideoVisionRotaryEmbedding` + `rotate_pairwise` in `modeling_sam2_video.py`.
//!
//! The memory-attention q/k are rotated by a per-spatial-position angle before the dot product. The angles are
//! **axial**: half the rope dimension indexes the x-coordinate, half the y-coordinate, each at a geometric
//! series of frequencies; `repeat_interleave(2)` pairs them so each `(2j, 2j+1)` dim pair rotates by one angle
//! (standard interleaved RoPE). The tables are input-independent at the fixed `feat × feat` grid, so they are
//! precomputed in f64 (the sanctioned table-precompute, like the image positional tables).
//!
//! Slice 2 adds `VideoRoPEAttention` — `Sam2VideoRoPEAttention`: project q/k/v, rotate q and k by the 2D RoPE,
//! then the shared integer SDPA core + output projection. The rotation is integer on the data path (signed
//! fixed-point cos/sin applied per dim pair), the float twin uses the f64 tables.

use kt_core::conv::{Conv2d, DepthwiseConv2d};
use kt_core::layers::{add, LayerNorm, QuantLinear, ACT_LEVELS};
use kt_core::module::Module;
use kt_core::sdpa::SdpaCore;
use kt_core::IntTensor;
use kt_int::{CodeLut, Scale};

use crate::hiera::WeightStore;
use crate::nonlin::{apply_gelu, build_gelu_lut, gelu_f64, relu};
use crate::spatial::{chw_to_hwc, hwc_to_chw, Spatial};

/// Video / memory-attention config (facebook/sam2.1-hiera-base-plus).
#[derive(Clone)]
pub struct VideoConfig {
    pub mem_hidden: usize,     // memory_attention_hidden_size
    pub mem_downsample: usize, // memory_attention_downsample_rate
    pub mem_heads: usize,      // memory_attention_num_attention_heads
    pub mem_layers: usize,     // memory_attention_num_layers
    pub rope_theta: f64,
    pub feat_x: usize,
    pub feat_y: usize,
    pub ffn: usize, // memory_attention_feed_forward_hidden_size
}

impl VideoConfig {
    pub fn base_plus() -> Self {
        VideoConfig {
            mem_hidden: 256,
            mem_downsample: 1,
            mem_heads: 1,
            mem_layers: 4,
            rope_theta: 10000.0,
            feat_x: 64,
            feat_y: 64,
            ffn: 2048,
        }
    }
    /// The rotary dimension = internal_dim / heads (must be divisible by 4 for axial splitting).
    pub fn rope_dim(&self) -> usize {
        self.mem_hidden / (self.mem_downsample * self.mem_heads)
    }
}

/// The 2D axial RoPE cos/sin tables, each `[feat_x·feat_y, rope_dim]` row-major. Matches
/// `Sam2VideoVisionRotaryEmbedding.create_inv_freq` + `.cos()/.sin()`.
pub fn rope_tables(cfg: &VideoConfig) -> (Vec<f64>, Vec<f64>) {
    let dim = cfg.rope_dim();
    assert!(dim.is_multiple_of(4), "rope dim must be divisible by 4");
    let nf = dim / 4; // frequencies per axis
    let freqs: Vec<f64> = (0..nf).map(|k| 1.0 / cfg.rope_theta.powf((4 * k) as f64 / dim as f64)).collect();
    let (ex, ey) = (cfg.feat_x, cfg.feat_y);
    let seq = ex * ey;
    let (mut cos, mut sin) = (vec![0.0f64; seq * dim], vec![0.0f64; seq * dim]);
    for pos in 0..seq {
        let x = (pos % ex) as f64;
        let y = (pos / ex) as f64;
        // inv_freq has 2·nf entries: the first nf are x·freq, the next nf are y·freq.
        for j in 0..2 * nf {
            let v = if j < nf { x * freqs[j] } else { y * freqs[j - nf] };
            // repeat_interleave(2): this angle fills dims 2j and 2j+1.
            for &d in &[2 * j, 2 * j + 1] {
                cos[pos * dim + d] = v.cos();
                sin[pos * dim + d] = v.sin();
            }
        }
    }
    (cos, sin)
}

/// `rotate_pairwise`: each dim pair `(2j, 2j+1)` → `(-x[2j+1], x[2j])`. The companion to the RoPE tables so that
/// `x·cos + rotate_pairwise(x)·sin` is a per-pair rotation.
pub fn rotate_pairwise(x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; x.len()];
    let mut i = 0;
    while i + 1 < x.len() {
        out[i] = -x[i + 1];
        out[i + 1] = x[i];
        i += 2;
    }
    out
}

/// Apply 2D RoPE to one token vector at spatial position `pos` (row `pos` of the tables).
pub fn apply_rope_row(x: &[f64], cos: &[f64], sin: &[f64], pos: usize, dim: usize) -> Vec<f64> {
    let (c, s) = (&cos[pos * dim..(pos + 1) * dim], &sin[pos * dim..(pos + 1) * dim]);
    let r = rotate_pairwise(x);
    (0..dim).map(|d| x[d] * c[d] + r[d] * s[d]).collect()
}

fn sign_scale(v: f64) -> (Scale, i8) {
    (Scale::from_f64(v.abs()), if v < 0.0 { -1 } else { 1 })
}

/// `Sam2VideoRoPEAttention` — attention with 2D rotary position encoding on q and k. Q/K/V and the output are
/// exact-integer [`QuantLinear`]; the rotation is applied integer on the data path (signed fixed-point cos/sin);
/// the attention is the shared integer [`SdpaCore`]. Non-causal, single head (base-plus: internal 256, 1 head).
pub struct VideoRoPEAttention {
    pub n_heads: usize,
    pub head_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub out_dim: usize,
    wq: QuantLinear,
    wk: QuantLinear,
    wv: QuantLinear,
    wo: QuantLinear,
    core: SdpaCore,
    dim: usize, // rope dim = n_heads·head_dim
    // rope tables [seq, dim]
    cos: Vec<f64>,
    sin: Vec<f64>,
    cmag: Vec<Scale>,
    csign: Vec<i8>,
    smag: Vec<Scale>,
    ssign: Vec<i8>,
}

impl VideoRoPEAttention {
    pub fn from_store(store: &WeightStore, prefix: &str, cfg: &VideoConfig, kv_dim: usize) -> Self {
        let internal = cfg.mem_hidden / cfg.mem_downsample;
        let n_heads = cfg.mem_heads;
        let head_dim = internal / n_heads;
        let hidden = cfg.mem_hidden;
        let ql = |n: &str, out: usize, inp: usize| {
            QuantLinear::from_weights(n, store.w(&format!("{prefix}.{n}.weight")), out, inp)
                .with_bias(Some(store.w(&format!("{prefix}.{n}.bias"))))
        };
        let (cos, sin) = rope_tables(cfg);
        let dim = cfg.rope_dim();
        let (cmag, csign): (Vec<Scale>, Vec<i8>) = cos.iter().map(|&v| sign_scale(v)).unzip();
        let (smag, ssign): (Vec<Scale>, Vec<i8>) = sin.iter().map(|&v| sign_scale(v)).unzip();
        VideoRoPEAttention {
            n_heads,
            head_dim,
            q_dim: hidden,
            kv_dim,
            out_dim: hidden,
            wq: ql("q_proj", internal, hidden),
            wk: ql("k_proj", internal, kv_dim),
            wv: ql("v_proj", internal, kv_dim),
            wo: ql("o_proj", hidden, internal),
            core: SdpaCore::new(head_dim),
            dim,
            cos,
            sin,
            cmag,
            csign,
            smag,
            ssign,
        }
    }

    /// Integer 2D-RoPE on one `dim`-vector at spatial position `pos`, in place.
    fn rope_int_row(&self, v: &mut [i64], pos: usize) {
        let base = pos * self.dim;
        let orig: Vec<i64> = v.to_vec();
        for d in 0..self.dim {
            let rot = if d % 2 == 0 { -orig[d + 1] } else { orig[d - 1] };
            let c = self.csign[base + d] as i64 * self.cmag[base + d].apply(orig[d]);
            let s = self.ssign[base + d] as i64 * self.smag[base + d].apply(rot);
            v[d] = c + s;
        }
    }

    /// Integer forward. `q_in` `[sq, q_dim]`, `k_in`/`v_in` `[sk, kv_dim]`; q rotated at positions `0..sq`, k at
    /// rows `0..sk-num_k_exclude` (the excluded tail = object pointers, no RoPE) using position `j` — or
    /// `j % k_period` when `k_period > 0` (the memory repeats the per-frame grid, `rope_k_repeat`). Output
    /// `[sq, out_dim]`.
    pub fn forward_int(
        &self,
        q_in: &IntTensor,
        k_in: &IntTensor,
        v_in: &IntTensor,
        num_k_exclude: usize,
        k_period: usize,
    ) -> IntTensor {
        let q = self.wq.forward_int(q_in);
        let k = self.wk.forward_int(k_in);
        let v = self.wv.forward_int(v_in);
        let (sq, sk) = (q.len() / self.dim, k.len() / self.dim);
        let mut qd: Vec<i64> = q.data.iter().map(|&z| z as i64).collect();
        let mut kd: Vec<i64> = k.data.iter().map(|&z| z as i64).collect();
        for i in 0..sq {
            self.rope_int_row(&mut qd[i * self.dim..(i + 1) * self.dim], i);
        }
        for j in 0..sk - num_k_exclude {
            let pos = if k_period > 0 { j % k_period } else { j };
            self.rope_int_row(&mut kd[j * self.dim..(j + 1) * self.dim], pos);
        }
        let qr = IntTensor::new(qd.iter().map(|&z| z as i32).collect(), q.shape.clone(), q.scale);
        let kr = IntTensor::new(kd.iter().map(|&z| z as i32).collect(), k.shape.clone(), k.scale);
        let concat = self.core.forward_int(&qr, &kr, &v, self.n_heads, self.head_dim);
        self.wo.forward_int(&concat)
    }

    /// Float twin.
    pub fn forward_ref(
        &self,
        q_in: &[f32],
        k_in: &[f32],
        v_in: &[f32],
        num_k_exclude: usize,
        k_period: usize,
    ) -> Vec<f32> {
        let q = self.wq.forward_ref(q_in);
        let k = self.wk.forward_ref(k_in);
        let v = self.wv.forward_ref(v_in);
        let (sq, sk) = (q.len() / self.dim, k.len() / self.dim);
        let mut qd: Vec<f64> = q.iter().map(|&z| z as f64).collect();
        let mut kd: Vec<f64> = k.iter().map(|&z| z as f64).collect();
        for i in 0..sq {
            let r = apply_rope_row(&qd[i * self.dim..(i + 1) * self.dim], &self.cos, &self.sin, i, self.dim);
            qd[i * self.dim..(i + 1) * self.dim].copy_from_slice(&r);
        }
        for j in 0..sk - num_k_exclude {
            let pos = if k_period > 0 { j % k_period } else { j };
            let r = apply_rope_row(&kd[j * self.dim..(j + 1) * self.dim], &self.cos, &self.sin, pos, self.dim);
            kd[j * self.dim..(j + 1) * self.dim].copy_from_slice(&r);
        }
        let out = self.core.forward_ref(
            &qd,
            &kd,
            &v.iter().map(|&z| z as f64).collect::<Vec<_>>(),
            self.n_heads,
            self.head_dim,
        );
        self.wo.forward_ref(&out.iter().map(|&z| z as f32).collect::<Vec<_>>())
    }
}

/// One `Sam2VideoMemoryAttentionLayer`: self-attention (RoPE) + cross-attention to memory (RoPE, repeated over
/// frames) + ReLU MLP, pre-LayerNorm with residual adds.
pub struct MemoryAttentionLayer {
    self_attn: VideoRoPEAttention,
    cross_attn: VideoRoPEAttention,
    ln1: LayerNorm,
    ln2: LayerNorm,
    ln3: LayerNorm,
    linear1: QuantLinear,
    linear2: QuantLinear,
}
impl MemoryAttentionLayer {
    fn from_store(store: &WeightStore, prefix: &str, cfg: &VideoConfig) -> Self {
        let ln = |n: &str| {
            LayerNorm::new(n, store.w(&format!("{prefix}.{n}.weight")), store.w(&format!("{prefix}.{n}.bias")), 1e-6)
        };
        MemoryAttentionLayer {
            self_attn: VideoRoPEAttention::from_store(store, &format!("{prefix}.self_attn"), cfg, cfg.mem_hidden),
            cross_attn: VideoRoPEAttention::from_store(store, &format!("{prefix}.cross_attn_image"), cfg, 64),
            ln1: ln("layer_norm1"),
            ln2: ln("layer_norm2"),
            ln3: ln("layer_norm3"),
            linear1: QuantLinear::from_weights(
                "linear1",
                store.w(&format!("{prefix}.linear1.weight")),
                cfg.ffn,
                cfg.mem_hidden,
            )
            .with_bias(Some(store.w(&format!("{prefix}.linear1.bias")))),
            linear2: QuantLinear::from_weights(
                "linear2",
                store.w(&format!("{prefix}.linear2.weight")),
                cfg.mem_hidden,
                cfg.ffn,
            )
            .with_bias(Some(store.w(&format!("{prefix}.linear2.bias")))),
        }
    }
    fn forward_int(
        &self,
        queries: &IntTensor,
        memory: &IntTensor,
        mem_pe: &IntTensor,
        num_ptr: usize,
        k_period: usize,
    ) -> IntTensor {
        let q = self.ln1.forward_int(queries);
        let a = self.self_attn.forward_int(&q, &q, &q, 0, 0);
        let queries = add(queries, &a);
        let q = self.ln2.forward_int(&queries);
        let key = add(memory, mem_pe);
        let a = self.cross_attn.forward_int(&q, &key, memory, num_ptr, k_period);
        let queries = add(&queries, &a);
        let q = self.ln3.forward_int(&queries);
        let m = self.linear2.forward_int(&relu(&self.linear1.forward_int(&q)));
        add(&queries, &m)
    }
    fn forward_ref(
        &self,
        queries: &[f32],
        memory: &[f32],
        mem_pe: &[f32],
        num_ptr: usize,
        k_period: usize,
    ) -> Vec<f32> {
        let addf = |a: &[f32], b: &[f32]| -> Vec<f32> { a.iter().zip(b).map(|(&x, &y)| x + y).collect() };
        let q = self.ln1.forward_ref(queries);
        let a = self.self_attn.forward_ref(&q, &q, &q, 0, 0);
        let queries = addf(queries, &a);
        let q = self.ln2.forward_ref(&queries);
        let key = addf(memory, mem_pe);
        let a = self.cross_attn.forward_ref(&q, &key, memory, num_ptr, k_period);
        let queries = addf(&queries, &a);
        let q = self.ln3.forward_ref(&queries);
        let m = self.linear2.forward_ref(&relu_f32(&self.linear1.forward_ref(&q)));
        addf(&queries, &m)
    }
}

fn relu_f32(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// `Sam2VideoMemoryAttention`: `current_vision_features (+ 0.1·pos)` conditioned on `memory` through
/// `mem_layers` memory-attention layers + a final LayerNorm. `k_period` = the per-frame grid size (feat²), so
/// the memory RoPE repeats per frame.
pub struct MemoryAttention {
    layers: Vec<MemoryAttentionLayer>,
    final_ln: LayerNorm,
    k_period: usize,
    /// the 0.1 the current positions are scaled by, as an integer [`Scale`] fixed at build
    pos_scale: Scale,
}
impl MemoryAttention {
    pub fn from_store(store: &WeightStore, cfg: &VideoConfig) -> Self {
        let layers = (0..cfg.mem_layers)
            .map(|i| MemoryAttentionLayer::from_store(store, &format!("memory_attention.layers.{i}"), cfg))
            .collect();
        MemoryAttention {
            layers,
            final_ln: LayerNorm::new(
                "mem_final_ln",
                store.w("memory_attention.layer_norm.weight"),
                store.w("memory_attention.layer_norm.bias"),
                1e-6,
            ),
            k_period: cfg.feat_x * cfg.feat_y,
            pos_scale: Scale::from_f64(0.1),
        }
    }
    /// `current` `[seq,256]`, `current_pos` `[seq,256]`, `memory` `[M,64]`, `mem_pe` `[M,64]`.
    pub fn forward_int(
        &self,
        current: &IntTensor,
        current_pos: &IntTensor,
        memory: &IntTensor,
        mem_pe: &IntTensor,
        num_ptr: usize,
    ) -> IntTensor {
        let pos01 =
            IntTensor::new(current_pos.data.clone(), current_pos.shape.clone(), current_pos.scale.mul(self.pos_scale));
        let mut out = add(current, &pos01);
        for layer in &self.layers {
            out = layer.forward_int(&out, memory, mem_pe, num_ptr, self.k_period);
        }
        self.final_ln.forward_int(&out)
    }
    pub fn forward_ref(
        &self,
        current: &[f32],
        current_pos: &[f32],
        memory: &[f32],
        mem_pe: &[f32],
        num_ptr: usize,
    ) -> Vec<f32> {
        let mut out: Vec<f32> = current.iter().zip(current_pos).map(|(&c, &p)| c + 0.1 * p).collect();
        for layer in &self.layers {
            out = layer.forward_ref(&out, memory, mem_pe, num_ptr, self.k_period);
        }
        self.final_ln.forward_ref(&out)
    }
}

/// Sine positional encoding for the memory (`Sam2VideoPositionEmbeddingSine`, num_features=32, normalize): the
/// `maskmem_pos_enc` the memory attention adds to the memory keys. Input-independent at the fixed `h×w` grid →
/// returned as `[h·w, 2·num_feats]` tokens (channels: `num_feats` for y, then `num_feats` for x). The y/x
/// coordinates are normalized to `2π·(i+1)/(n+eps)`, each frequency contributing a `(sin, cos)` pair.
pub fn sine_pos_table(h: usize, w: usize, num_feats: usize, temperature: f64) -> Vec<f64> {
    let twopi = 2.0 * std::f64::consts::PI;
    let eps = 1e-6;
    let chans = 2 * num_feats;
    let mut out = vec![0.0f64; h * w * chans];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * chans;
            let ye = (y as f64 + 1.0) / (h as f64 + eps) * twopi;
            let xe = (x as f64 + 1.0) / (w as f64 + eps) * twopi;
            for j in 0..num_feats / 2 {
                let freq = temperature.powf(2.0 * j as f64 / num_feats as f64);
                let (ay, ax) = (ye / freq, xe / freq);
                // channels 0..num_feats = y features
                out[base + 2 * j] = ay.sin();
                out[base + 2 * j + 1] = ay.cos();
                // channels num_feats.. = x features
                out[base + num_feats + 2 * j] = ax.sin();
                out[base + num_feats + 2 * j + 1] = ax.cos();
            }
        }
    }
    out
}

/// The object-pointer head (`object_pointer_proj` + `no_object_pointer`): a 3-layer ReLU MLP applied to the
/// best mask token, gated by the object-score (`is_obj_appearing = obj_score > 0`) — when no object is present
/// the learned `no_object_pointer` is used instead. Produces the `[256]` pointer stored in the memory bank.
pub struct ObjectPointerHead {
    proj_in: QuantLinear,
    mid: QuantLinear,
    proj_out: QuantLinear,
    no_object_pointer: Vec<f32>,
}
impl ObjectPointerHead {
    pub fn from_store(store: &WeightStore) -> Self {
        let p = "object_pointer_proj";
        ObjectPointerHead {
            proj_in: QuantLinear::from_weights("opi", store.w(&format!("{p}.proj_in.weight")), 256, 256)
                .with_bias(Some(store.w(&format!("{p}.proj_in.bias")))),
            mid: QuantLinear::from_weights("opm", store.w(&format!("{p}.layers.0.weight")), 256, 256)
                .with_bias(Some(store.w(&format!("{p}.layers.0.bias")))),
            proj_out: QuantLinear::from_weights("opo", store.w(&format!("{p}.proj_out.weight")), 256, 256)
                .with_bias(Some(store.w(&format!("{p}.proj_out.bias")))),
            no_object_pointer: store.w("no_object_pointer").to_vec(),
        }
    }
    /// `mask_token` `[256]` (the best mask-token output), `obj_score` the object-score logit → pointer `[256]`.
    pub fn forward_int(&self, mask_token: &IntTensor, obj_score: f32) -> Vec<f32> {
        if obj_score <= 0.0 {
            return self.no_object_pointer.clone();
        }
        let h = relu(&self.proj_in.forward_int(mask_token));
        let h = relu(&self.mid.forward_int(&h));
        self.proj_out.forward_int(&h).to_f32()
    }
    pub fn forward_ref(&self, mask_token: &[f32], obj_score: f32) -> Vec<f32> {
        if obj_score <= 0.0 {
            return self.no_object_pointer.clone();
        }
        let r = |x: &[f32]| -> Vec<f32> { x.iter().map(|&v| v.max(0.0)).collect() };
        let h = r(&self.proj_in.forward_ref(mask_token));
        let h = r(&self.mid.forward_ref(&h));
        self.proj_out.forward_ref(&h)
    }
}

// ───────────────────────── memory encoder (fuser + mask downsampler) ─────────────────────────

/// Channels-first LayerNorm on an integer `[C,H,W]` tensor.
fn ln2d(ln: &LayerNorm, x: &IntTensor, c: usize, h: usize, w: usize) -> IntTensor {
    let sp = chw_to_hwc(x);
    hwc_to_chw(&Spatial::from_tokens(&ln.forward_int(&sp.as_tokens()), 1, h, w, c))
}

/// Per-channel learned scale (`gamma`): multiply column `c` of a `[rows, C]` token matrix by `gamma[c]`.
struct LayerScale {
    codes: Vec<i32>,
    scale: Scale,
    gamma: Vec<f32>,
}
impl LayerScale {
    fn new(gamma: &[f32]) -> Self {
        let peak = gamma.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
        let (scale, codes) = if peak == 0.0 {
            (Scale::ZERO, vec![0i32; gamma.len()])
        } else {
            let step = peak / (kt_core::layers::WLEVELS as f64);
            (Scale::from_f64(step), gamma.iter().map(|&v| ((v as f64 / step).round() as i64) as i32).collect())
        };
        LayerScale { codes, scale, gamma: gamma.to_vec() }
    }
    fn apply_int(&self, x: &IntTensor) -> IntTensor {
        let c = self.codes.len();
        let rows = x.len() / c;
        let mut v = vec![0i64; x.len()];
        for r in 0..rows {
            for ci in 0..c {
                v[r * c + ci] = x.data[r * c + ci] as i64 * self.codes[ci] as i64;
            }
        }
        let gmax = v.iter().fold(0i64, |m, &z| m.max(z.abs()));
        if gmax == 0 {
            return IntTensor::new(vec![0; x.len()], x.shape.clone(), Scale::ZERO);
        }
        let out: Vec<i32> = v.iter().map(|&z| kt_core::layers::requant_dyn(z, ACT_LEVELS, gmax)).collect();
        IntTensor::new(
            out,
            x.shape.clone(),
            Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(self.scale).mul(x.scale),
        )
    }
    fn apply_ref(&self, x: &[f32], c: usize) -> Vec<f32> {
        x.iter().enumerate().map(|(i, &v)| v * self.gamma[i % c]).collect()
    }
}

/// One ConvNeXt fuser block: depthwise 7×7 conv → LN2d → pointwise(256→1024) → GELU → pointwise(1024→256) →
/// per-channel layer-scale → residual.
struct FuserBlock {
    dwconv: DepthwiseConv2d,
    ln: LayerNorm,
    pw1: QuantLinear,
    pw2: QuantLinear,
    scale: LayerScale,
    gelu: CodeLut,
}
impl FuserBlock {
    fn from_store(store: &WeightStore, prefix: &str, dim: usize) -> Self {
        FuserBlock {
            dwconv: DepthwiseConv2d::from_weights(
                "dw",
                store.w(&format!("{prefix}.depthwise_conv.weight")),
                dim,
                7,
                1,
                3,
            )
            .with_bias(Some(store.w(&format!("{prefix}.depthwise_conv.bias")))),
            ln: LayerNorm::new(
                "fln",
                store.w(&format!("{prefix}.layer_norm.weight")),
                store.w(&format!("{prefix}.layer_norm.bias")),
                1e-6,
            ),
            pw1: QuantLinear::from_weights("pw1", store.w(&format!("{prefix}.pointwise_conv1.weight")), 1024, dim)
                .with_bias(Some(store.w(&format!("{prefix}.pointwise_conv1.bias")))),
            pw2: QuantLinear::from_weights("pw2", store.w(&format!("{prefix}.pointwise_conv2.weight")), dim, 1024)
                .with_bias(Some(store.w(&format!("{prefix}.pointwise_conv2.bias")))),
            scale: LayerScale::new(store.w(&format!("{prefix}.scale"))),
            gelu: build_gelu_lut(),
        }
    }
    fn forward_int(&self, x: &IntTensor, c: usize, h: usize, w: usize) -> IntTensor {
        let dw = ln2d(&self.ln, &self.dwconv.forward_int(x), c, h, w);
        let hwc = chw_to_hwc(&dw); // [1,h,w,c]
        let t = self.pw2.forward_int(&apply_gelu(&self.gelu, &self.pw1.forward_int(&hwc.as_tokens())));
        let t = self.scale.apply_int(&t);
        let back = hwc_to_chw(&Spatial::from_tokens(&t, 1, h, w, c));
        add(x, &back)
    }
    fn forward_ref(&self, x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
        let dw = self.dwconv.forward_ref(x, h, w);
        // ln2d channels-first
        let hwc = crate::spatial::chw_to_hwc_raw(&dw, c, h, w);
        let dwn = crate::spatial::hwc_to_chw_raw(&self.ln.forward_ref(&hwc), h, w, c);
        let hwc = crate::spatial::chw_to_hwc_raw(&dwn, c, h, w); // tokens [h*w,c]
        let g: Vec<f32> = self.pw1.forward_ref(&hwc).iter().map(|&v| gelu_f64(v as f64) as f32).collect();
        let t = self.scale.apply_ref(&self.pw2.forward_ref(&g), c);
        let back = crate::spatial::hwc_to_chw_raw(&t, h, w, c);
        x.iter().zip(back.iter()).map(|(&a, &b)| a + b).collect()
    }
}

/// The memory encoder: downsample the mask (4× stride-2), add to the projected image features, fuse (ConvNeXt),
/// project to the 64-channel memory. `Sam2VideoMemoryEncoder` (the feature part; sine pos is a separate table).
pub struct MemoryEncoder {
    down_convs: Vec<Conv2d>,
    down_lns: Vec<LayerNorm>,
    down_final: Conv2d,
    feature_projection: Conv2d,
    fuser: Vec<FuserBlock>,
    projection: Conv2d,
    gelu: CodeLut,
}
impl MemoryEncoder {
    pub fn from_store(store: &WeightStore) -> Self {
        let p = "memory_encoder";
        let chans = [(1usize, 4usize), (4, 16), (16, 64), (64, 256)];
        let down_convs = (0..4)
            .map(|i| {
                Conv2d::from_weights(
                    "dc",
                    store.w(&format!("{p}.mask_downsampler.layers.{i}.conv.weight")),
                    chans[i].1,
                    chans[i].0,
                    3,
                    3,
                    2,
                    1,
                )
                .with_bias(Some(store.w(&format!("{p}.mask_downsampler.layers.{i}.conv.bias"))))
            })
            .collect();
        let down_lns = (0..4)
            .map(|i| {
                LayerNorm::new(
                    "dln",
                    store.w(&format!("{p}.mask_downsampler.layers.{i}.layer_norm.weight")),
                    store.w(&format!("{p}.mask_downsampler.layers.{i}.layer_norm.bias")),
                    1e-6,
                )
            })
            .collect();
        let down_final = Conv2d::from_weights(
            "dfin",
            store.w(&format!("{p}.mask_downsampler.final_conv.weight")),
            256,
            256,
            1,
            1,
            1,
            0,
        )
        .with_bias(Some(store.w(&format!("{p}.mask_downsampler.final_conv.bias"))));
        let feature_projection =
            Conv2d::from_weights("fp", store.w(&format!("{p}.feature_projection.weight")), 256, 256, 1, 1, 1, 0)
                .with_bias(Some(store.w(&format!("{p}.feature_projection.bias"))));
        let fuser =
            (0..2).map(|i| FuserBlock::from_store(store, &format!("{p}.memory_fuser.layers.{i}"), 256)).collect();
        let projection = Conv2d::from_weights("proj", store.w(&format!("{p}.projection.weight")), 64, 256, 1, 1, 1, 0)
            .with_bias(Some(store.w(&format!("{p}.projection.bias"))));
        MemoryEncoder {
            down_convs,
            down_lns,
            down_final,
            feature_projection,
            fuser,
            projection,
            gelu: build_gelu_lut(),
        }
    }

    fn downsample_int(&self, mask: &IntTensor) -> IntTensor {
        let (mut x, mut h, mut w) = (mask.clone(), mask.shape[1], mask.shape[2]);
        for i in 0..4 {
            let c = self.down_convs[i].c_out;
            let conv = self.down_convs[i].forward_int(&x);
            let (oh, ow) = (conv.shape[1], conv.shape[2]);
            x = apply_gelu(&self.gelu, &ln2d(&self.down_lns[i], &conv, c, oh, ow));
            h = oh;
            w = ow;
        }
        let _ = (h, w);
        self.down_final.forward_int(&x)
    }

    /// `vision` `[256,64,64]`, `mask` `[1,1024,1024]` → memory features `[64,64,64]`.
    pub fn forward_int(&self, vision: &IntTensor, mask: &IntTensor) -> IntTensor {
        let m = self.downsample_int(mask); // [256,64,64]
        let v = self.feature_projection.forward_int(vision); // [256,64,64]
        let mut x = add(&v, &m);
        for blk in &self.fuser {
            x = blk.forward_int(&x, 256, 64, 64);
        }
        self.projection.forward_int(&x) // [64,64,64]
    }

    fn downsample_ref(&self, mask: &[f32], mh: usize, mw: usize) -> (Vec<f32>, usize, usize) {
        let (mut x, mut h, mut w) = (mask.to_vec(), mh, mw);
        for i in 0..4 {
            let c = self.down_convs[i].c_out;
            let conv = self.down_convs[i].forward_ref(&x, h, w);
            let (oh, ow) = self.down_convs[i].out_dims(h, w);
            let hwc = crate::spatial::chw_to_hwc_raw(&conv, c, oh, ow);
            let n = crate::spatial::hwc_to_chw_raw(&self.down_lns[i].forward_ref(&hwc), oh, ow, c);
            x = n.iter().map(|&v| gelu_f64(v as f64) as f32).collect();
            h = oh;
            w = ow;
        }
        (self.down_final.forward_ref(&x, h, w), h, w)
    }

    pub fn forward_ref(&self, vision: &[f32], mask: &[f32]) -> Vec<f32> {
        let (m, _, _) = self.downsample_ref(mask, 1024, 1024);
        let v = self.feature_projection.forward_ref(vision, 64, 64);
        let mut x: Vec<f32> = v.iter().zip(m.iter()).map(|(&a, &b)| a + b).collect();
        for blk in &self.fuser {
            x = blk.forward_ref(&x, 256, 64, 64);
        }
        self.projection.forward_ref(&x, 64, 64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_pairwise_rotates_each_pair() {
        // (a,b) → (-b, a)
        assert_eq!(rotate_pairwise(&[1.0, 2.0, 3.0, 4.0]), vec![-2.0, 1.0, -4.0, 3.0]);
    }

    #[test]
    fn rope_tables_have_expected_shape_and_bounds() {
        let cfg = VideoConfig::base_plus();
        let dim = cfg.rope_dim();
        assert_eq!(dim, 256);
        let (cos, sin) = rope_tables(&cfg);
        assert_eq!(cos.len(), 64 * 64 * dim);
        assert_eq!(sin.len(), 64 * 64 * dim);
        // position 0 → all angles 0 → cos 1, sin 0.
        assert!(cos[0..dim].iter().all(|&c| (c - 1.0).abs() < 1e-12));
        assert!(sin[0..dim].iter().all(|&s| s.abs() < 1e-12));
        // every value in [-1,1].
        assert!(cos.iter().chain(sin.iter()).all(|&v| (-1.0..=1.0).contains(&v)));
    }
}
