//! The Hiera image encoder — a faithful, exact-integer port of `Sam2HieraDetModel` + `Sam2VisionNeck`'s
//! backbone from `transformers`' `modeling_sam2.py`. Patch-embed (Conv2d) → windowed/global MultiScaleAttention
//! with query-pooling at stage changes → FeedForward (GELU) → residual, over 4 stages. Every op routes through
//! the generic `kt-core` kernels. Every op carries a float twin, and the port was validated block by
//! block against the reference `transformers` model.
//!
//! Layout convention: activations are carried as [`Spatial`] `[B,H,W,C]` (channels-last), exactly as the HF code
//! keeps `hidden_states`. The window batch (`B = num_windows`) is handled by running the attention core once per
//! window and stacking the results with [`kt_core::layers::concat_rows`].

use std::collections::HashMap;

use kt_core::conv::Conv2d;
use kt_core::layers::{add, concat_rows, LayerNorm, QuantLinear, ACT_LEVELS};
use kt_core::module::Module;
use kt_core::sdpa::SdpaCore;
use kt_core::IntTensor;
use kt_int::CodeLut;

use crate::nonlin::{apply_gelu, build_gelu_lut, gelu_f64};
use crate::spatial::{
    chw_to_hwc, chw_to_hwc_raw, max_pool2d, max_pool2d_raw, window_partition, window_partition_raw, window_unpartition,
    window_unpartition_raw, Spatial,
};

// ───────────────────────── weight store ─────────────────────────
/// A name → (flat row-major f32 values, shape) map. Populated either from the model's safetensors (the real
/// loader) or, for verification, from the reference weight dump.
pub struct WeightStore(pub HashMap<String, (Vec<f32>, Vec<usize>)>);

impl WeightStore {
    pub fn w(&self, name: &str) -> &[f32] {
        &self.0.get(name).unwrap_or_else(|| panic!("missing weight: {name}")).0
    }
    pub fn has(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    /// Pull every tensor a [`kt_core::plugin::ModelSource`] holds into a store — the generic-converter path
    /// (the plugin's `build` gets a `ModelSource`, not a file path).
    pub fn from_source(src: &dyn kt_core::plugin::ModelSource) -> Self {
        let mut m = HashMap::new();
        for name in src.tensor_names() {
            if let Some((v, s)) = src.tensor(&name) {
                m.insert(name, (v, s));
            }
        }
        WeightStore(m)
    }

    /// Load all tensors from a `.safetensors` file (dequantizing F32 / F16 / BF16 to f32) — the real-checkpoint
    /// loader, so the demo needs no precomputed weight dump. The header is `[u64 len][JSON]`, then the raw buffer.
    pub fn from_safetensors(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if bytes.len() < 8 {
            return Err("safetensors too short".into());
        }
        let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen]).map_err(|e| format!("header: {e}"))?;
        let base = 8 + hlen;
        let obj = hdr.as_object().ok_or("header is not an object")?;
        let mut map = HashMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v["dtype"].as_str().ok_or("no dtype")?;
            let shape: Vec<usize> =
                v["shape"].as_array().ok_or("no shape")?.iter().map(|x| x.as_u64().unwrap() as usize).collect();
            let offs = v["data_offsets"].as_array().ok_or("no data_offsets")?;
            let (s, e) = (offs[0].as_u64().unwrap() as usize, offs[1].as_u64().unwrap() as usize);
            let raw = &bytes[base + s..base + e];
            let vals: Vec<f32> = match dtype {
                "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                "BF16" => raw
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect(),
                "F16" => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
                other => return Err(format!("unsupported dtype {other} for {name}")),
            };
            map.insert(name.clone(), (vals, shape));
        }
        Ok(WeightStore(map))
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let (sign, exp, mant) = ((h >> 15) & 1, (h >> 10) & 0x1f, h & 0x3ff);
    let s = (sign as u32) << 31;
    let bits = if exp == 0 {
        if mant == 0 {
            s
        } else {
            // subnormal
            let mut e = -1i32;
            let mut m = mant as u32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            s | (((127 - 15 + e + 1) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        s | 0x7f80_0000 | ((mant as u32) << 13)
    } else {
        s | (((exp as i32 - 15 + 127) as u32) << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(bits)
}

// ───────────────────────── config ─────────────────────────
#[derive(Clone)]
pub struct HieraConfig {
    pub num_channels: usize,
    pub patch_k: usize,
    pub patch_s: usize,
    pub patch_p: usize,
    pub patch_grid: usize, // image_size / patch_stride (e.g. 1024/4 = 256)
    pub embed_dim_per_stage: Vec<usize>,
    pub blocks_per_stage: Vec<usize>,
    pub heads_per_stage: Vec<usize>,
    pub window_per_stage: Vec<usize>,
    pub global_attention_blocks: Vec<usize>,
    pub query_stride: usize,
    pub num_query_pool_stages: usize,
    pub mlp_ratio: f64,
    pub eps: f64,
}

impl HieraConfig {
    /// facebook/sam2.1-hiera-base-plus (read from its config.json).
    pub fn base_plus() -> Self {
        HieraConfig {
            num_channels: 3,
            patch_k: 7,
            patch_s: 4,
            patch_p: 3,
            patch_grid: 256,
            embed_dim_per_stage: vec![112, 224, 448, 896],
            blocks_per_stage: vec![2, 3, 16, 3],
            heads_per_stage: vec![2, 4, 8, 16],
            window_per_stage: vec![8, 4, 14, 7],
            global_attention_blocks: vec![12, 16, 20],
            query_stride: 2,
            num_query_pool_stages: 3,
            mlp_ratio: 4.0,
            eps: 1e-6,
        }
    }
    /// stage-end block indices (cumsum(blocks_per_stage) − 1) — the feature maps the neck consumes.
    pub fn stage_ends(&self) -> Vec<usize> {
        let mut acc = 0;
        self.blocks_per_stage
            .iter()
            .map(|&n| {
                acc += n;
                acc - 1
            })
            .collect()
    }
}

// ───────────────────────── patch embed ─────────────────────────
pub struct PatchEmbed {
    conv: Conv2d,
}
impl PatchEmbed {
    fn from_store(store: &WeightStore, cfg: &HieraConfig) -> Self {
        let c_out = cfg.embed_dim_per_stage[0];
        let w = store.w("vision_encoder.backbone.patch_embed.projection.weight");
        let b = store.w("vision_encoder.backbone.patch_embed.projection.bias");
        let conv = Conv2d::from_weights(
            "patch_embed",
            w,
            c_out,
            cfg.num_channels,
            cfg.patch_k,
            cfg.patch_k,
            cfg.patch_s,
            cfg.patch_p,
        )
        .with_bias(Some(b));
        PatchEmbed { conv }
    }
    /// `pixel`: `[C,H,W]` → `Spatial[1, H', W', embed]`.
    fn forward(&self, pixel: &IntTensor) -> Spatial {
        let o = self.conv.forward_int(pixel); // [embed, H', W']
        chw_to_hwc(&o)
    }
    /// Float twin — `pixel` `[C, in_h, in_w]` f32 → `([H'·W', embed] tokens, H', W')`.
    fn forward_ref(&self, pixel: &[f32], in_h: usize, in_w: usize) -> (Vec<f32>, usize, usize) {
        let (oh, ow) = self.conv.out_dims(in_h, in_w);
        let chw = self.conv.forward_ref(pixel, in_h, in_w); // [embed, oh, ow]
        (chw_to_hwc_raw(&chw, self.conv.c_out, oh, ow), oh, ow)
    }
}

// ───────────────────────── MultiScaleAttention ─────────────────────────
pub struct MultiScaleAttention {
    n_heads: usize,
    head_dim: usize,
    dim_out: usize,
    qkv: QuantLinear,
    proj: QuantLinear,
    core: SdpaCore,
}

/// Split a fused qkv activation `[N, 3·dim_out]` (HF layout `reshape(N,3,nheads,head_dim)`) into q,k,v, each
/// `[N, dim_out]`, all sharing the fused output's `Scale`.
fn split_qkv(qkv: &IntTensor, dim_out: usize) -> (IntTensor, IntTensor, IntTensor) {
    let n = qkv.len() / (3 * dim_out);
    let mut q = vec![0i32; n * dim_out];
    let mut k = vec![0i32; n * dim_out];
    let mut v = vec![0i32; n * dim_out];
    for t in 0..n {
        let base = t * 3 * dim_out;
        q[t * dim_out..(t + 1) * dim_out].copy_from_slice(&qkv.data[base..base + dim_out]);
        k[t * dim_out..(t + 1) * dim_out].copy_from_slice(&qkv.data[base + dim_out..base + 2 * dim_out]);
        v[t * dim_out..(t + 1) * dim_out].copy_from_slice(&qkv.data[base + 2 * dim_out..base + 3 * dim_out]);
    }
    let s = qkv.scale;
    (
        IntTensor::new(q, vec![n, dim_out], s),
        IntTensor::new(k, vec![n, dim_out], s),
        IntTensor::new(v, vec![n, dim_out], s),
    )
}

fn rows(t: &IntTensor, start_row: usize, count: usize, cols: usize) -> IntTensor {
    IntTensor::new(t.data[start_row * cols..(start_row + count) * cols].to_vec(), vec![count, cols], t.scale)
}

/// f32 twin of [`split_qkv`] — split a fused `[N, 3·dim_out]` activation into q, k, v.
fn split_qkv_f32(qkv: &[f32], dim_out: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = qkv.len() / (3 * dim_out);
    let (mut q, mut k, mut v) = (vec![0f32; n * dim_out], vec![0f32; n * dim_out], vec![0f32; n * dim_out]);
    for t in 0..n {
        let base = t * 3 * dim_out;
        q[t * dim_out..(t + 1) * dim_out].copy_from_slice(&qkv[base..base + dim_out]);
        k[t * dim_out..(t + 1) * dim_out].copy_from_slice(&qkv[base + dim_out..base + 2 * dim_out]);
        v[t * dim_out..(t + 1) * dim_out].copy_from_slice(&qkv[base + 2 * dim_out..base + 3 * dim_out]);
    }
    (q, k, v)
}

impl MultiScaleAttention {
    fn from_store(store: &WeightStore, prefix: &str, dim: usize, dim_out: usize, n_heads: usize) -> Self {
        let head_dim = dim_out / n_heads;
        let qkv = QuantLinear::from_weights("qkv", store.w(&format!("{prefix}.qkv.weight")), 3 * dim_out, dim)
            .with_bias(Some(store.w(&format!("{prefix}.qkv.bias"))));
        let proj = QuantLinear::from_weights("proj", store.w(&format!("{prefix}.proj.weight")), dim_out, dim_out)
            .with_bias(Some(store.w(&format!("{prefix}.proj.bias"))));
        MultiScaleAttention { n_heads, head_dim, dim_out, qkv, proj, core: SdpaCore::new(head_dim) }
    }

    /// `x`: `[B,H,W,dim]` (B = window batch). With `query_stride`, q is max-pooled per window before attention,
    /// so the output spatial dims shrink. Returns `[B, H', W', dim_out]`.
    fn forward(&self, x: &Spatial, query_stride: Option<usize>) -> Spatial {
        let (batch, h, w) = (x.b, x.h, x.w);
        let inner = self.dim_out;
        let (nh, hd) = (self.n_heads, self.head_dim);
        let qkv = self.qkv.forward_int(&x.as_tokens()); // [B·H·W, 3·inner]
        let (q0, k, v) = split_qkv(&qkv, inner);
        // query pooling (per window)
        let (q, hq, wq) = match query_stride {
            Some(qs) => {
                let qsp = Spatial::from_tokens(&q0, batch, h, w, inner);
                let p = max_pool2d(&qsp, qs);
                (p.as_tokens(), p.h, p.w)
            }
            None => (q0, h, w),
        };
        let (sq, sk) = (hq * wq, h * w);
        // attention core per window, then stack to one feature map.
        let mut parts = Vec::with_capacity(batch);
        for b in 0..batch {
            let qb = rows(&q, b * sq, sq, inner);
            let kb = rows(&k, b * sk, sk, inner);
            let vb = rows(&v, b * sk, sk, inner);
            parts.push(self.core.forward_int(&qb, &kb, &vb, nh, hd));
        }
        let concat = concat_rows(&parts); // [B·sq, inner]
        let out = self.proj.forward_int(&concat);
        Spatial::from_tokens(&out, batch, hq, wq, inner)
    }

    /// Float twin of `MultiScaleAttention::forward` — `x` tokens `[batch·h·w, dim]` → `([batch·hq·wq, dim_out], hq, wq)`.
    pub fn forward_ref(
        &self,
        x: &[f32],
        batch: usize,
        h: usize,
        w: usize,
        query_stride: Option<usize>,
    ) -> (Vec<f32>, usize, usize) {
        let inner = self.dim_out;
        let (nh, hd) = (self.n_heads, self.head_dim);
        let qkv = self.qkv.forward_ref(x);
        let (q0, k, v) = split_qkv_f32(&qkv, inner);
        let (q, hq, wq) = match query_stride {
            Some(qs) => max_pool2d_raw(&q0, batch, h, w, inner, qs),
            None => (q0, h, w),
        };
        let (sq, sk) = (hq * wq, h * w);
        let mut concat = vec![0f32; batch * sq * inner];
        for b in 0..batch {
            let qb: Vec<f64> = q[b * sq * inner..(b + 1) * sq * inner].iter().map(|&z| z as f64).collect();
            let kb: Vec<f64> = k[b * sk * inner..(b + 1) * sk * inner].iter().map(|&z| z as f64).collect();
            let vb: Vec<f64> = v[b * sk * inner..(b + 1) * sk * inner].iter().map(|&z| z as f64).collect();
            let ob = self.core.forward_ref(&qb, &kb, &vb, nh, hd);
            for (i, &z) in ob.iter().enumerate() {
                concat[b * sq * inner + i] = z as f32;
            }
        }
        (self.proj.forward_ref(&concat), hq, wq)
    }
}

// ───────────────────────── FeedForward (GELU MLP) ─────────────────────────
pub struct FeedForward {
    proj_in: QuantLinear,
    proj_out: QuantLinear,
    gelu: CodeLut,
}
impl FeedForward {
    fn from_store(store: &WeightStore, prefix: &str, dim: usize, hidden: usize) -> Self {
        let proj_in = QuantLinear::from_weights("proj_in", store.w(&format!("{prefix}.proj_in.weight")), hidden, dim)
            .with_bias(Some(store.w(&format!("{prefix}.proj_in.bias"))));
        let proj_out =
            QuantLinear::from_weights("proj_out", store.w(&format!("{prefix}.proj_out.weight")), dim, hidden)
                .with_bias(Some(store.w(&format!("{prefix}.proj_out.bias"))));
        let gelu = build_gelu_lut();
        FeedForward { proj_in, proj_out, gelu }
    }
    fn forward_int(&self, x: &IntTensor) -> IntTensor {
        let h = self.proj_in.forward_int(x);
        let g = apply_gelu(&self.gelu, &h);
        self.proj_out.forward_int(&g)
    }
    /// Float twin — exact GELU (the reference the integer-code LUT approximates).
    fn forward_ref(&self, x: &[f32]) -> Vec<f32> {
        let h = self.proj_in.forward_ref(x);
        let g: Vec<f32> = h.iter().map(|&v| gelu_f64(v as f64) as f32).collect();
        self.proj_out.forward_ref(&g)
    }
}

// ───────────────────────── MultiScaleBlock ─────────────────────────
pub struct MultiScaleBlock {
    pub dim: usize,
    pub dim_out: usize,
    pub window_size: usize,
    pub query_stride: Option<usize>,
    ln1: LayerNorm,
    attn: MultiScaleAttention,
    ln2: LayerNorm,
    mlp: FeedForward,
    proj: Option<QuantLinear>,
}

impl MultiScaleBlock {
    #[allow(clippy::too_many_arguments)]
    fn from_store(
        store: &WeightStore,
        idx: usize,
        dim: usize,
        dim_out: usize,
        n_heads: usize,
        window_size: usize,
        query_stride: Option<usize>,
        cfg: &HieraConfig,
    ) -> Self {
        let prefix = format!("vision_encoder.backbone.blocks.{idx}");
        let ln1 = LayerNorm::new(
            "ln1",
            store.w(&format!("{prefix}.layer_norm1.weight")),
            store.w(&format!("{prefix}.layer_norm1.bias")),
            cfg.eps,
        );
        let attn = MultiScaleAttention::from_store(store, &format!("{prefix}.attn"), dim, dim_out, n_heads);
        let ln2 = LayerNorm::new(
            "ln2",
            store.w(&format!("{prefix}.layer_norm2.weight")),
            store.w(&format!("{prefix}.layer_norm2.bias")),
            cfg.eps,
        );
        let hidden = (dim_out as f64 * cfg.mlp_ratio) as usize;
        let mlp = FeedForward::from_store(store, &format!("{prefix}.mlp"), dim_out, hidden);
        let proj = if dim != dim_out {
            Some(
                QuantLinear::from_weights("blk_proj", store.w(&format!("{prefix}.proj.weight")), dim_out, dim)
                    .with_bias(Some(store.w(&format!("{prefix}.proj.bias")))),
            )
        } else {
            None
        };
        MultiScaleBlock { dim, dim_out, window_size, query_stride, ln1, attn, ln2, mlp, proj }
    }

    fn forward(&self, x: Spatial) -> Spatial {
        let (h, w) = (x.h, x.w);
        let mut residual = x.clone();
        let x1 = Spatial::from_tokens(&self.ln1.forward_int(&x.as_tokens()), 1, h, w, self.dim);

        // residual skip gets the projection + pool when the stage changes the dim
        if self.dim != self.dim_out {
            let p = self.proj.as_ref().unwrap().forward_int(&x1.as_tokens());
            let psp = Spatial::from_tokens(&p, 1, h, w, self.dim_out);
            residual = match self.query_stride {
                Some(qs) => max_pool2d(&psp, qs),
                None => psp,
            };
        }

        // window partition (window_size == 0 ⇒ global attention over the whole map)
        let (attn_in, pad_hw) =
            if self.window_size > 0 { window_partition(&x1, self.window_size) } else { (x1, (0usize, 0usize)) };
        let attn_out = self.attn.forward(&attn_in, self.query_stride);

        // if q-pooled, the window shrinks and the grid is the pooled residual grid
        let (mut window_size, mut pad_hw2, mut hw2) = (self.window_size, pad_hw, (h, w));
        if let Some(qs) = self.query_stride {
            window_size = self.window_size / qs;
            let (hh, ww) = (residual.h, residual.w);
            let pad_h = if window_size > 0 { (window_size - hh % window_size) % window_size } else { 0 };
            let pad_w = if window_size > 0 { (window_size - ww % window_size) % window_size } else { 0 };
            pad_hw2 = (hh + pad_h, ww + pad_w);
            hw2 = (hh, ww);
        }
        let x2 = if self.window_size > 0 { window_unpartition(&attn_out, window_size, pad_hw2, hw2) } else { attn_out };

        // residual + attn, then + MLP(LN(·))
        let added = add(&residual.as_tokens(), &x2.as_tokens());
        let mlp = self.mlp.forward_int(&self.ln2.forward_int(&added));
        let out = add(&added, &mlp);
        Spatial::from_tokens(&out, 1, residual.h, residual.w, self.dim_out)
    }

    /// Float twin of [`MultiScaleBlock::forward`] — `x` tokens `[h·w, dim]` → `([h'·w', dim_out], h', w')`.
    fn forward_ref(&self, x: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
        let x1 = self.ln1.forward_ref(x);
        // residual skip (proj + pool at a stage change)
        let (mut res, mut rh, mut rw) = (x.to_vec(), h, w);
        if self.dim != self.dim_out {
            let p = self.proj.as_ref().unwrap().forward_ref(&x1);
            match self.query_stride {
                Some(qs) => {
                    let (pp, ph, pw) = max_pool2d_raw(&p, 1, h, w, self.dim_out, qs);
                    res = pp;
                    rh = ph;
                    rw = pw;
                }
                None => res = p,
            }
        }
        // window partition
        let (attn_in, bo, pad_hw) = if self.window_size > 0 {
            window_partition_raw(&x1, 1, h, w, self.dim, self.window_size)
        } else {
            (x1, 1, (0, 0))
        };
        let (aw_h, aw_w) = if self.window_size > 0 { (self.window_size, self.window_size) } else { (h, w) };
        let (ao, _ah, _aw) = self.attn.forward_ref(&attn_in, bo, aw_h, aw_w, self.query_stride);
        // post-window
        let (mut window_size, mut pad_hw2, mut hw2) = (self.window_size, pad_hw, (h, w));
        if let Some(qs) = self.query_stride {
            window_size = self.window_size / qs;
            let (hh, ww) = (rh, rw);
            let pad_h = if window_size > 0 { (window_size - hh % window_size) % window_size } else { 0 };
            let pad_w = if window_size > 0 { (window_size - ww % window_size) % window_size } else { 0 };
            pad_hw2 = (hh + pad_h, ww + pad_w);
            hw2 = (hh, ww);
        }
        let x2 = if self.window_size > 0 {
            window_unpartition_raw(&ao, bo, window_size, pad_hw2, hw2, self.dim_out)
        } else {
            ao
        };
        let added: Vec<f32> = res.iter().zip(x2.iter()).map(|(&a, &b)| a + b).collect();
        let mlp = self.mlp.forward_ref(&self.ln2.forward_ref(&added));
        let out: Vec<f32> = added.iter().zip(mlp.iter()).map(|(&a, &b)| a + b).collect();
        (out, rh, rw)
    }
}

// ───────────────────────── HieraEncoder ─────────────────────────
pub struct HieraEncoder {
    patch_embed: PatchEmbed,
    pos_embed: IntTensor, // [patch_grid², embed0] — precomputed positional table (added after patch-embed)
    pos_embed_f32: Vec<f32>, // the same table as f32, for the float twin
    blocks: Vec<MultiScaleBlock>,
    cfg: HieraConfig,
}

impl HieraEncoder {
    /// `pos_embed_tokens` is the precomputed `_get_pos_embed` table (input-independent at the fixed image size),
    /// as `[patch_grid², embed0]` row-major reals — supplied by the caller (the loader computes it; for
    /// verification it can be supplied directly, like a precomputed RoPE table).
    pub fn from_store(store: &WeightStore, cfg: &HieraConfig, pos_embed_tokens: &[f64]) -> Self {
        let patch_embed = PatchEmbed::from_store(store, cfg);
        let embed0 = cfg.embed_dim_per_stage[0];
        let grid = cfg.patch_grid;
        assert_eq!(pos_embed_tokens.len(), grid * grid * embed0, "pos_embed table size mismatch");
        let pos_embed = IntTensor::from_reals(pos_embed_tokens, vec![grid * grid, embed0], ACT_LEVELS);
        let pos_embed_f32: Vec<f32> = pos_embed_tokens.iter().map(|&v| v as f32).collect();

        let mut blocks = Vec::new();
        let mut total = 0usize;
        for stage in 0..cfg.blocks_per_stage.len() {
            for block in 0..cfg.blocks_per_stage[stage] {
                let dim = if stage > 0 && block == 0 {
                    cfg.embed_dim_per_stage[stage - 1]
                } else {
                    cfg.embed_dim_per_stage[stage]
                };
                let dim_out = cfg.embed_dim_per_stage[stage];
                let mut window =
                    if stage > 0 && block == 0 { cfg.window_per_stage[stage - 1] } else { cfg.window_per_stage[stage] };
                if cfg.global_attention_blocks.contains(&total) {
                    window = 0;
                }
                let query_stride = if stage > 0 && stage <= cfg.num_query_pool_stages && block == 0 {
                    Some(cfg.query_stride)
                } else {
                    None
                };
                let n_heads = cfg.heads_per_stage[stage];
                blocks.push(MultiScaleBlock::from_store(
                    store,
                    total,
                    dim,
                    dim_out,
                    n_heads,
                    window,
                    query_stride,
                    cfg,
                ));
                total += 1;
            }
        }
        HieraEncoder { patch_embed, pos_embed, pos_embed_f32, blocks, cfg: cfg.clone() }
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
    pub fn stage_ends(&self) -> Vec<usize> {
        self.cfg.stage_ends()
    }

    /// Run patch-embed + pos-embed + every block, returning each block's output (for module-by-module
    /// verification against a float reference).
    pub fn forward_capture(&self, pixel: &IntTensor) -> Vec<Spatial> {
        let pe = self.patch_embed.forward(pixel);
        let grid = self.cfg.patch_grid;
        let embed0 = self.cfg.embed_dim_per_stage[0];
        let x0 = add(&pe.as_tokens(), &self.pos_embed);
        let mut x = Spatial::from_tokens(&x0, 1, grid, grid, embed0);
        let mut outs = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            x = blk.forward(x);
            outs.push(x.clone());
        }
        outs
    }

    /// The stage-end feature maps (what the FPN neck consumes), high→low resolution by stage order.
    pub fn forward(&self, pixel: &IntTensor) -> Vec<Spatial> {
        let outs = self.forward_capture(pixel);
        self.stage_ends().iter().map(|&i| outs[i].clone()).collect()
    }

    /// Float twin of [`HieraEncoder::forward_capture`] — every block output as `(tokens, h, w, c)`.
    pub fn forward_capture_ref(&self, pixel: &[f32]) -> Vec<(Vec<f32>, usize, usize, usize)> {
        let in_sz = self.cfg.patch_grid * self.cfg.patch_s;
        let (mut x, mut h, mut w) = self.patch_embed.forward_ref(pixel, in_sz, in_sz);
        for (xi, &p) in x.iter_mut().zip(self.pos_embed_f32.iter()) {
            *xi += p;
        }
        let mut outs = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let (o, nh, nw) = blk.forward_ref(&x, h, w);
            x = o;
            h = nh;
            w = nw;
            outs.push((x.clone(), h, w, blk.dim_out));
        }
        outs
    }

    /// Float twin of [`HieraEncoder::forward`] — the stage-end feature maps as `(tokens, h, w, c)`.
    pub fn forward_ref(&self, pixel: &[f32]) -> Vec<(Vec<f32>, usize, usize, usize)> {
        let outs = self.forward_capture_ref(pixel);
        self.stage_ends().iter().map(|&i| outs[i].clone()).collect()
    }
}
