//! The two-way mask decoder — a faithful exact-integer port of `Sam2MaskDecoder` + `Sam2TwoWayTransformer`
//! (transformers `modeling_sam2.py`). Output tokens (obj-score, IoU, 4 mask tokens) are concatenated with the
//! sparse prompt tokens and run through two `TwoWayAttentionBlock`s (token self-attention, token→image cross,
//! ReLU MLP, image→token cross) plus a final token→image attention — all via [`SdpaAttention`]. The image tokens
//! are then upscaled 2×2 ([`ConvTranspose2d`]) with lateral adds from the high-res neck features and GELU, and a
//! per-mask hypernetwork MLP produces the mask filters, whose matmul with the upscaled embedding gives the mask
//! logits. The verification twin is the real float `Sam2Model` mask output.

use kt_core::conv::ConvTranspose2d;
use kt_core::layers::{add, concat_rows, matmul, LayerNorm, QuantLinear, ACT_LEVELS};
use kt_core::module::Module;
use kt_core::sdpa::SdpaAttention;
use kt_core::IntTensor;
use kt_int::CodeLut;

use crate::hiera::WeightStore;
use crate::nonlin::{apply_gelu, build_gelu_lut, gelu_f64, relu};
use crate::spatial::{chw_to_hwc, chw_to_hwc_raw, hwc_to_chw, hwc_to_chw_raw, Spatial};

/// Elementwise f32 add (the float twin's residual/positional add).
fn add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}
/// f32 ReLU.
fn relu_f32(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}
/// Float twin of the channels-first LayerNorm.
fn layernorm2d_ref(ln: &LayerNorm, x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let hwc = chw_to_hwc_raw(x, c, h, w); // [h*w, c] tokens
    let t = ln.forward_ref(&hwc);
    hwc_to_chw_raw(&t, h, w, c)
}

const HID: usize = 256;
const EPS: f64 = 1e-6;

fn row_range(t: &IntTensor, start: usize, count: usize) -> IntTensor {
    let cols = *t.shape.last().unwrap();
    IntTensor::new(t.data[start * cols..(start + count) * cols].to_vec(), vec![count, cols], t.scale)
}

fn token(store: &WeightStore, name: &str, rows: usize) -> IntTensor {
    let w: Vec<f64> = store.w(name).iter().map(|&v| v as f64).collect();
    IntTensor::from_reals(&w, vec![rows, HID], ACT_LEVELS)
}

fn attn_from(store: &WeightStore, prefix: &str, head_dim: usize) -> SdpaAttention {
    SdpaAttention::from_weights(
        prefix,
        8,
        head_dim,
        HID,
        HID,
        HID,
        store.w(&format!("{prefix}.q_proj.weight")),
        store.w(&format!("{prefix}.k_proj.weight")),
        store.w(&format!("{prefix}.v_proj.weight")),
        store.w(&format!("{prefix}.o_proj.weight")),
        (
            Some(store.w(&format!("{prefix}.q_proj.bias"))),
            Some(store.w(&format!("{prefix}.k_proj.bias"))),
            Some(store.w(&format!("{prefix}.v_proj.bias"))),
            Some(store.w(&format!("{prefix}.o_proj.bias"))),
        ),
    )
}

// ───────────────────────── ReLU MLP (Sam2FeedForward) ─────────────────────────
struct Mlp {
    proj_in: QuantLinear,
    middle: Vec<QuantLinear>,
    proj_out: QuantLinear,
}
impl Mlp {
    fn from_store(
        store: &WeightStore,
        prefix: &str,
        in_dim: usize,
        hidden: usize,
        out_dim: usize,
        num_layers: usize,
    ) -> Self {
        let proj_in =
            QuantLinear::from_weights("proj_in", store.w(&format!("{prefix}.proj_in.weight")), hidden, in_dim)
                .with_bias(Some(store.w(&format!("{prefix}.proj_in.bias"))));
        let middle = (0..num_layers.saturating_sub(2))
            .map(|i| {
                QuantLinear::from_weights("mid", store.w(&format!("{prefix}.layers.{i}.weight")), hidden, hidden)
                    .with_bias(Some(store.w(&format!("{prefix}.layers.{i}.bias"))))
            })
            .collect();
        let proj_out =
            QuantLinear::from_weights("proj_out", store.w(&format!("{prefix}.proj_out.weight")), out_dim, hidden)
                .with_bias(Some(store.w(&format!("{prefix}.proj_out.bias"))));
        Mlp { proj_in, middle, proj_out }
    }
    fn forward_int(&self, x: &IntTensor) -> IntTensor {
        let mut h = relu(&self.proj_in.forward_int(x));
        for l in &self.middle {
            h = relu(&l.forward_int(&h));
        }
        self.proj_out.forward_int(&h)
    }
    fn forward_ref(&self, x: &[f32]) -> Vec<f32> {
        let mut h = relu_f32(&self.proj_in.forward_ref(x));
        for l in &self.middle {
            h = relu_f32(&l.forward_ref(&h));
        }
        self.proj_out.forward_ref(&h)
    }
}

// ───────────────────────── TwoWayAttentionBlock ─────────────────────────
struct TwoWayBlock {
    skip_first_layer_pe: bool,
    self_attn: SdpaAttention,
    ln1: LayerNorm,
    cross_t2i: SdpaAttention,
    ln2: LayerNorm,
    mlp: Mlp,
    ln3: LayerNorm,
    ln4: LayerNorm,
    cross_i2t: SdpaAttention,
}
impl TwoWayBlock {
    fn from_store(store: &WeightStore, prefix: &str, skip: bool) -> Self {
        let ln = |n: &str| {
            LayerNorm::new(n, store.w(&format!("{prefix}.{n}.weight")), store.w(&format!("{prefix}.{n}.bias")), EPS)
        };
        TwoWayBlock {
            skip_first_layer_pe: skip,
            self_attn: attn_from(store, &format!("{prefix}.self_attn"), 32), // downsample_rate 1 → internal 256
            ln1: ln("layer_norm1"),
            cross_t2i: attn_from(store, &format!("{prefix}.cross_attn_token_to_image"), 16), // ds 2 → internal 128
            ln2: ln("layer_norm2"),
            mlp: Mlp::from_store(store, &format!("{prefix}.mlp"), HID, 2048, HID, 2),
            ln3: ln("layer_norm3"),
            ln4: ln("layer_norm4"),
            cross_i2t: attn_from(store, &format!("{prefix}.cross_attn_image_to_token"), 16),
        }
    }
    /// Returns (queries, keys).
    fn forward(
        &self,
        queries: IntTensor,
        keys: IntTensor,
        query_pe: &IntTensor,
        key_pe: &IntTensor,
    ) -> (IntTensor, IntTensor) {
        // self attention on the tokens
        let queries = if self.skip_first_layer_pe {
            self.self_attn.forward_qkv_int(&queries, &queries, &queries)
        } else {
            let q = add(&queries, query_pe);
            let a = self.self_attn.forward_qkv_int(&q, &q, &queries);
            add(&queries, &a)
        };
        let queries = self.ln1.forward_int(&queries);
        // cross attention: tokens → image
        let q = add(&queries, query_pe);
        let k = add(&keys, key_pe);
        let a = self.cross_t2i.forward_qkv_int(&q, &k, &keys);
        let queries = self.ln2.forward_int(&add(&queries, &a));
        // MLP
        let m = self.mlp.forward_int(&queries);
        let queries = self.ln3.forward_int(&add(&queries, &m));
        // cross attention: image → tokens (q = key, k = query, v = queries)
        let q = add(&queries, query_pe);
        let k = add(&keys, key_pe);
        let a = self.cross_i2t.forward_qkv_int(&k, &q, &queries);
        let keys = self.ln4.forward_int(&add(&keys, &a));
        (queries, keys)
    }

    /// Float twin of [`TwoWayBlock::forward`].
    fn forward_ref(&self, queries: Vec<f32>, keys: Vec<f32>, query_pe: &[f32], key_pe: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let queries = if self.skip_first_layer_pe {
            self.self_attn.forward_qkv_ref(&queries, &queries, &queries)
        } else {
            let q = add_f32(&queries, query_pe);
            let a = self.self_attn.forward_qkv_ref(&q, &q, &queries);
            add_f32(&queries, &a)
        };
        let queries = self.ln1.forward_ref(&queries);
        let q = add_f32(&queries, query_pe);
        let k = add_f32(&keys, key_pe);
        let a = self.cross_t2i.forward_qkv_ref(&q, &k, &keys);
        let queries = self.ln2.forward_ref(&add_f32(&queries, &a));
        let m = self.mlp.forward_ref(&queries);
        let queries = self.ln3.forward_ref(&add_f32(&queries, &m));
        let q = add_f32(&queries, query_pe);
        let k = add_f32(&keys, key_pe);
        let a = self.cross_i2t.forward_qkv_ref(&k, &q, &queries);
        let keys = self.ln4.forward_ref(&add_f32(&keys, &a));
        (queries, keys)
    }
}

// ───────────────────────── TwoWayTransformer ─────────────────────────
struct TwoWayTransformer {
    layers: Vec<TwoWayBlock>,
    final_attn: SdpaAttention,
    ln_final: LayerNorm,
}
impl TwoWayTransformer {
    fn from_store(store: &WeightStore, prefix: &str) -> Self {
        let layers = (0..2).map(|i| TwoWayBlock::from_store(store, &format!("{prefix}.layers.{i}"), i == 0)).collect();
        TwoWayTransformer {
            layers,
            final_attn: attn_from(store, &format!("{prefix}.final_attn_token_to_image"), 16),
            ln_final: LayerNorm::new(
                "ln_final",
                store.w(&format!("{prefix}.layer_norm_final_attn.weight")),
                store.w(&format!("{prefix}.layer_norm_final_attn.bias")),
                EPS,
            ),
        }
    }
    /// `point_embeddings` `[T,256]`, `image_embeddings` + `image_pos` `[HW,256]` → (queries `[T,256]`, keys `[HW,256]`).
    fn forward(
        &self,
        point_embeddings: &IntTensor,
        image_embeddings: &IntTensor,
        image_pos: &IntTensor,
    ) -> (IntTensor, IntTensor) {
        let mut queries = point_embeddings.clone();
        let mut keys = image_embeddings.clone();
        for layer in &self.layers {
            let (q, k) = layer.forward(queries, keys, point_embeddings, image_pos);
            queries = q;
            keys = k;
        }
        let q = add(&queries, point_embeddings);
        let k = add(&keys, image_pos);
        let a = self.final_attn.forward_qkv_int(&q, &k, &keys);
        let queries = self.ln_final.forward_int(&add(&queries, &a));
        (queries, keys)
    }

    /// Float twin of [`TwoWayTransformer::forward`].
    fn forward_ref(
        &self,
        point_embeddings: &[f32],
        image_embeddings: &[f32],
        image_pos: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut queries = point_embeddings.to_vec();
        let mut keys = image_embeddings.to_vec();
        for layer in &self.layers {
            let (q, k) = layer.forward_ref(queries, keys, point_embeddings, image_pos);
            queries = q;
            keys = k;
        }
        let q = add_f32(&queries, point_embeddings);
        let k = add_f32(&keys, image_pos);
        let a = self.final_attn.forward_qkv_ref(&q, &k, &keys);
        let queries = self.ln_final.forward_ref(&add_f32(&queries, &a));
        (queries, keys)
    }
}

/// Channels-first LayerNorm (`Sam2LayerNorm` data_format="channels_first"): normalize over channels at each
/// spatial position.
fn layernorm2d(ln: &LayerNorm, x: &IntTensor) -> IntTensor {
    let (c, h, w) = (x.shape[0], x.shape[1], x.shape[2]);
    let sp = chw_to_hwc(x); // [1,h,w,c]
    let t = ln.forward_int(&sp.as_tokens()); // [h*w, c]
    hwc_to_chw(&Spatial::from_tokens(&t, 1, h, w, c))
}

/// A `[1, c]` row repeated `n` times as `[n, c]`: the same codes at the same scale. Quantizing the broadcast floats
/// would give these codes too (the same maximum, so the same scale), so a constant row is quantized once at build.
pub fn broadcast_row(row: &IntTensor, n: usize) -> IntTensor {
    let c = row.data.len();
    IntTensor::new(row.data.repeat(n), vec![n, c], row.scale)
}

// ───────────────────────── MaskDecoder ─────────────────────────
pub struct MaskOutput {
    pub masks: IntTensor,       // [4, 256*256] logits (all mask tokens)
    pub iou: IntTensor,         // [1, 4]
    pub obj_score: IntTensor, // [1, 1] object_score_logits as integer codes (pred_obj_score_head of the obj-score token)
    pub mask_tokens: IntTensor, // [4, 256] the mask-token outputs (for the object pointer)
    pub h: usize,
    pub w: usize,
}
impl MaskOutput {
    /// multimask: the best of masks 1..4 by IoU score, compared as the integer codes (one row, one positive scale:
    /// the order, ties included, is the dequantized values' order).
    pub fn best_index(&self) -> usize {
        (1..4).max_by_key(|&a| self.iou.data[a]).unwrap()
    }
    /// Whether an object is present (`object_score_logits > 0`): the sign of the integer code.
    pub fn obj_appearing(&self) -> bool {
        self.obj_score.data[0] > 0
    }
    /// The object score as a float, for display and for the float video tracker.
    pub fn obj_score_f32(&self) -> f32 {
        self.obj_score.to_f64()[0] as f32
    }
    /// binary mask (logit > 0) for the best mask, row-major `[h*w]`.
    pub fn binary_best(&self) -> Vec<u8> {
        let k = self.best_index();
        let n = self.h * self.w;
        self.masks.data[k * n..(k + 1) * n].iter().map(|&v| u8::from(v > 0)).collect()
    }
}

/// Float twin of [`MaskOutput`] — the reference the integer mask is checked against.
pub struct FMaskOutput {
    pub masks: Vec<f32>, // [4, h*w] logits
    pub iou: Vec<f32>,   // [4]
    pub h: usize,
    pub w: usize,
}
impl FMaskOutput {
    pub fn best_index(&self) -> usize {
        (1..4).max_by(|&a, &b| self.iou[a].partial_cmp(&self.iou[b]).unwrap()).unwrap()
    }
    pub fn binary_best(&self) -> Vec<u8> {
        let k = self.best_index();
        let n = self.h * self.w;
        self.masks[k * n..(k + 1) * n].iter().map(|&v| u8::from(v > 0.0)).collect()
    }
}

pub struct MaskDecoder {
    obj_tok: IntTensor,
    iou_tok: IntTensor,
    mask_tok: IntTensor,
    transformer: TwoWayTransformer,
    upscale_conv1: ConvTranspose2d,
    upscale_conv2: ConvTranspose2d,
    upscale_ln: LayerNorm,
    hypernet: Vec<Mlp>,
    iou_head: Mlp,
    obj_score_head: Mlp,
    gelu: CodeLut,
}

impl MaskDecoder {
    pub fn from_store(store: &WeightStore) -> Self {
        let p = "mask_decoder";
        MaskDecoder {
            obj_tok: token(store, &format!("{p}.obj_score_token.weight"), 1),
            iou_tok: token(store, &format!("{p}.iou_token.weight"), 1),
            mask_tok: token(store, &format!("{p}.mask_tokens.weight"), 4),
            transformer: TwoWayTransformer::from_store(store, &format!("{p}.transformer")),
            upscale_conv1: ConvTranspose2d::from_weights(
                "up1",
                store.w(&format!("{p}.upscale_conv1.weight")),
                256,
                64,
                2,
            )
            .with_bias(Some(store.w(&format!("{p}.upscale_conv1.bias")))),
            upscale_conv2: ConvTranspose2d::from_weights(
                "up2",
                store.w(&format!("{p}.upscale_conv2.weight")),
                64,
                32,
                2,
            )
            .with_bias(Some(store.w(&format!("{p}.upscale_conv2.bias")))),
            upscale_ln: LayerNorm::new(
                "up_ln",
                store.w(&format!("{p}.upscale_layer_norm.weight")),
                store.w(&format!("{p}.upscale_layer_norm.bias")),
                EPS,
            ),
            hypernet: (0..4)
                .map(|i| Mlp::from_store(store, &format!("{p}.output_hypernetworks_mlps.{i}"), HID, HID, 32, 3))
                .collect(),
            iou_head: Mlp::from_store(store, &format!("{p}.iou_prediction_head"), HID, HID, 4, 3),
            obj_score_head: Mlp::from_store(store, &format!("{p}.pred_obj_score_head"), HID, HID, 1, 3),
            gelu: build_gelu_lut(),
        }
    }

    /// `image_embed_tokens` `[64*64,256]` (already with no_memory_embedding), `image_pos` `[64*64,256]`,
    /// `sparse` `[ns,256]`, `dense` `[1,256]` (the no-mask dense embedding, quantized once and broadcast here),
    /// `feat0` `[32,256,256]`, `feat1` `[64,128,128]`.
    pub fn forward(
        &self,
        image_embed_tokens: &IntTensor,
        image_pos: &IntTensor,
        sparse: &IntTensor,
        dense: &IntTensor,
        feat0: &IntTensor,
        feat1: &IntTensor,
    ) -> MaskOutput {
        // tokens = [obj, iou, mask(4), sparse]
        let output_tokens = concat_rows(&[self.obj_tok.clone(), self.iou_tok.clone(), self.mask_tok.clone()]);
        let tokens = concat_rows(&[output_tokens, sparse.clone()]);

        // image embeddings + dense no-mask embedding (broadcast over the 64×64 grid)
        let hw = image_embed_tokens.len() / HID;
        let dense_full = broadcast_row(dense, hw);
        let image_embed = add(image_embed_tokens, &dense_full);

        let (queries, keys) = self.transformer.forward(&tokens, &image_embed, image_pos);
        let iou_token_out = row_range(&queries, 1, 1); // [1,256]
        let mask_tokens_out = row_range(&queries, 2, 4); // [4,256]

        // upscale the image embedding
        let image_chw = hwc_to_chw(&Spatial::from_tokens(&keys, 1, 64, 64, HID)); // [256,64,64]
        let u1 = add(&self.upscale_conv1.forward_int(&image_chw), feat1); // [64,128,128] + feat_s1
        let u1 = apply_gelu(&self.gelu, &layernorm2d(&self.upscale_ln, &u1));
        let u2 = add(&self.upscale_conv2.forward_int(&u1), feat0); // [32,256,256] + feat_s0
        let u2 = apply_gelu(&self.gelu, &u2); // upscaled_embedding [32,256,256]
        let (oh, ow) = (u2.shape[1], u2.shape[2]);

        // hypernetwork → [4,32]; masks = hyper_in @ upscaled[32, oh*ow]
        let hyper_parts: Vec<IntTensor> =
            (0..4).map(|i| self.hypernet[i].forward_int(&row_range(&mask_tokens_out, i, 1))).collect();
        let hyper_in = concat_rows(&hyper_parts); // [4,32]
        let upscaled_flat = IntTensor::new(u2.data.clone(), vec![32, oh * ow], u2.scale);
        let masks = matmul(&hyper_in, &upscaled_flat); // [4, oh*ow]

        let iou = self.iou_head.forward_int(&iou_token_out); // [1,4]
        let obj_score = self.obj_score_head.forward_int(&row_range(&queries, 0, 1));
        MaskOutput { masks, iou, obj_score, mask_tokens: mask_tokens_out, h: oh, w: ow }
    }

    /// Float twin of [`MaskDecoder::forward`] — same structure in f32; the reference the integer mask is checked
    /// against. `feat0` `[32,256,256]`, `feat1` `[64,128,128]` channels-first.
    pub fn forward_ref(
        &self,
        image_embed_tokens: &[f32],
        image_pos: &[f32],
        sparse: &[f32],
        dense_vec: &[f32],
        feat0: &[f32],
        feat1: &[f32],
    ) -> FMaskOutput {
        let mut tokens = Vec::with_capacity((6 + sparse.len() / HID) * HID);
        tokens.extend_from_slice(&self.obj_tok.to_f32());
        tokens.extend_from_slice(&self.iou_tok.to_f32());
        tokens.extend_from_slice(&self.mask_tok.to_f32());
        tokens.extend_from_slice(sparse);

        let hw = image_embed_tokens.len() / HID;
        let mut image_embed = image_embed_tokens.to_vec();
        for t in 0..hw {
            for c in 0..HID {
                image_embed[t * HID + c] += dense_vec[c];
            }
        }
        let (queries, keys) = self.transformer.forward_ref(&tokens, &image_embed, image_pos);
        let iou_token_out = &queries[HID..2 * HID];
        let mask_tokens_out = &queries[2 * HID..6 * HID];

        let image_chw = hwc_to_chw_raw(&keys, 64, 64, HID); // [256,64,64]
        let u1 = add_f32(&self.upscale_conv1.forward_ref(&image_chw, 64, 64), feat1); // [64,128,128]
        let u1 = layernorm2d_ref(&self.upscale_ln, &u1, 64, 128, 128);
        let u1: Vec<f32> = u1.iter().map(|&v| gelu_f64(v as f64) as f32).collect();
        let u2 = add_f32(&self.upscale_conv2.forward_ref(&u1, 128, 128), feat0); // [32,256,256]
        let u2: Vec<f32> = u2.iter().map(|&v| gelu_f64(v as f64) as f32).collect();
        let (oh, ow) = (256usize, 256usize);
        let n = oh * ow;

        let mut hyper = vec![0f32; 4 * 32];
        for i in 0..4 {
            let h = self.hypernet[i].forward_ref(&mask_tokens_out[i * HID..(i + 1) * HID]);
            hyper[i * 32..(i + 1) * 32].copy_from_slice(&h);
        }
        let mut masks = vec![0f32; 4 * n];
        for k in 0..4 {
            for p in 0..n {
                let mut acc = 0f64;
                for c in 0..32 {
                    acc += hyper[k * 32 + c] as f64 * u2[c * n + p] as f64;
                }
                masks[k * n + p] = acc as f32;
            }
        }
        let iou_t = self.iou_head.forward_ref(iou_token_out);
        FMaskOutput { masks, iou: iou_t, h: oh, w: ow }
    }
}
