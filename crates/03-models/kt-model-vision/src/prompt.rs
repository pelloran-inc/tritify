//! The prompt encoder — a faithful port of `Sam2PromptEncoder` for point prompts (the common single-point case: one foreground
//! point, no box, no mask). Produces the sparse token embeddings (the point + a padding "not-a-point" token) and
//! the dense embedding (the learned "no-mask" vector, broadcast over the 64×64 grid).
//!
//! The point positional encoding is a small random-Fourier feature map (`Sam2PositionalEmbedding`): normalize
//! the pixel coordinate to `[0, 1]`, map to `[−1, 1]`, project through the learned `[2,128]` matrix, scale by 2π, then
//! `[sin, cos]`. This is input preparation (O(#points), not the data path, like a tokenizer), so it is computed
//! in f64 and then quantized to the activation grid — exactly as the sine/positional tables elsewhere.

use kt_core::conv::Conv2d;
use kt_core::layers::{LayerNorm, ACT_LEVELS};
use kt_core::module::Module;
use kt_core::IntTensor;
use kt_int::CodeLut;

use crate::hiera::WeightStore;
use crate::nonlin::{apply_gelu, build_gelu_lut, gelu_f64};
use crate::spatial::{chw_to_hwc, chw_to_hwc_raw, hwc_to_chw, hwc_to_chw_raw, Spatial};

pub struct PromptEncoder {
    pe: Vec<f64>,          // shared_embedding.positional_embedding [2,128]
    point_embed: Vec<f32>, // [4,256]
    not_a_point: Vec<f32>, // [256]
    no_mask: Vec<f32>,     // [256]
    hidden: usize,         // 256
    image_size: f64,       // 1024
}

impl PromptEncoder {
    pub fn from_store(store: &WeightStore) -> Self {
        let pe: Vec<f64> =
            store.w("prompt_encoder.shared_embedding.positional_embedding").iter().map(|&v| v as f64).collect();
        PromptEncoder {
            pe,
            point_embed: store.w("prompt_encoder.point_embed.weight").to_vec(),
            not_a_point: store.w("prompt_encoder.not_a_point_embed.weight").to_vec(),
            no_mask: store.w("prompt_encoder.no_mask_embed.weight").to_vec(),
            hidden: 256,
            image_size: 1024.0,
        }
    }

    /// Random-Fourier positional encoding of a pixel coordinate (already +0.5-shifted), → `[256]` reals.
    fn pos_enc(&self, px: f64, py: f64) -> Vec<f64> {
        let half = self.hidden / 2; // 128
        let (cx, cy) = (2.0 * (px / self.image_size) - 1.0, 2.0 * (py / self.image_size) - 1.0);
        let mut out = vec![0.0f64; self.hidden];
        for j in 0..half {
            let f = 2.0 * std::f64::consts::PI * (cx * self.pe[j] + cy * self.pe[half + j]);
            out[j] = f.sin();
            out[half + j] = f.cos();
        }
        out
    }

    /// The sparse-embedding real values `[num_tokens·256]` and `num_tokens` (shared by the integer and float
    /// paths). Points in 1024-pixel space; labels 1 = foreground, 0 = background, −1 = padding; `pad` appends one
    /// padding point (the no-box case).
    fn point_rows(&self, points: &[(f64, f64)], labels: &[i32], pad: bool) -> (Vec<f64>, usize) {
        let mut pts: Vec<(f64, f64)> = points.iter().map(|&(x, y)| (x + 0.5, y + 0.5)).collect();
        let mut lbs: Vec<i32> = labels.to_vec();
        if pad {
            pts.push((0.5, 0.5)); // (0,0)+0.5
            lbs.push(-1);
        }
        let mut rows = vec![0.0f64; pts.len() * self.hidden];
        for (t, (&(px, py), &lb)) in pts.iter().zip(lbs.iter()).enumerate() {
            let row = &mut rows[t * self.hidden..(t + 1) * self.hidden];
            if lb == -1 {
                for (r, &v) in row.iter_mut().zip(self.not_a_point.iter()) {
                    *r = v as f64;
                }
            } else {
                let penc = self.pos_enc(px, py);
                let pe = &self.point_embed[(lb as usize) * self.hidden..(lb as usize + 1) * self.hidden];
                for (r, (&p, &e)) in row.iter_mut().zip(penc.iter().zip(pe.iter())) {
                    *r = p + e as f64;
                }
            }
        }
        (rows, pts.len())
    }

    /// Embed point prompts → sparse embeddings `[num_tokens, 256]` (quantized).
    pub fn embed_points(&self, points: &[(f64, f64)], labels: &[i32], pad: bool) -> IntTensor {
        let (rows, n) = self.point_rows(points, labels, pad);
        IntTensor::from_reals(&rows, vec![n, self.hidden], ACT_LEVELS)
    }

    /// Float twin — sparse embeddings as `[num_tokens·256]` reals.
    pub fn embed_points_ref(&self, points: &[(f64, f64)], labels: &[i32], pad: bool) -> Vec<f32> {
        self.point_rows(points, labels, pad).0.iter().map(|&v| v as f32).collect()
    }

    /// Box-prompt sparse rows: each box `(x1,y1,x2,y2)` → 3 tokens — top-left corner + `point_embed[2]`,
    /// bottom-right corner + `point_embed[3]`, and a `not_a_point` padding token. Faithful to `_embed_boxes`.
    fn box_rows(&self, boxes: &[(f64, f64, f64, f64)]) -> (Vec<f64>, usize) {
        let h = self.hidden;
        let mut rows = Vec::with_capacity(boxes.len() * 3 * h);
        for &(x1, y1, x2, y2) in boxes {
            let mut c1 = self.pos_enc(x1 + 0.5, y1 + 0.5);
            for (r, &e) in c1.iter_mut().zip(self.point_embed[2 * h..3 * h].iter()) {
                *r += e as f64;
            }
            let mut c2 = self.pos_enc(x2 + 0.5, y2 + 0.5);
            for (r, &e) in c2.iter_mut().zip(self.point_embed[3 * h..4 * h].iter()) {
                *r += e as f64;
            }
            rows.extend(c1);
            rows.extend(c2);
            rows.extend(self.not_a_point.iter().map(|&v| v as f64));
        }
        (rows, boxes.len() * 3)
    }

    /// Embed box prompts → sparse embeddings `[num_boxes·3, 256]` (quantized).
    pub fn embed_boxes(&self, boxes: &[(f64, f64, f64, f64)]) -> IntTensor {
        let (rows, n) = self.box_rows(boxes);
        IntTensor::from_reals(&rows, vec![n, self.hidden], ACT_LEVELS)
    }

    /// Float twin — box sparse embeddings as reals.
    pub fn embed_boxes_ref(&self, boxes: &[(f64, f64, f64, f64)]) -> Vec<f32> {
        self.box_rows(boxes).0.iter().map(|&v| v as f32).collect()
    }

    /// The dense "no-mask" embedding `[256]`, which the decoder broadcasts over every spatial position.
    pub fn no_mask_dense(&self) -> &[f32] {
        &self.no_mask
    }
}

/// The mask-input dense embedding (`Sam2MaskEmbedding`): a provided low-resolution mask `[1,256,256]` →
/// a dense embedding `[256,64,64]` that replaces the no-mask broadcast. Two strided convs (each halving the
/// resolution) with channels-first LayerNorm + GELU, then a 1×1 conv to the hidden size.
pub struct MaskEmbedding {
    conv1: Conv2d, // 1 → 4,  k2 s2
    conv2: Conv2d, // 4 → 16, k2 s2
    conv3: Conv2d, // 16 → 256, k1
    ln1: LayerNorm,
    ln2: LayerNorm,
    gelu: CodeLut,
}

impl MaskEmbedding {
    pub fn from_store(store: &WeightStore) -> Self {
        let p = "prompt_encoder.mask_embed";
        let cv = |n: &str, co: usize, ci: usize, k: usize, s: usize| {
            Conv2d::from_weights(n, store.w(&format!("{p}.{n}.weight")), co, ci, k, k, s, 0)
                .with_bias(Some(store.w(&format!("{p}.{n}.bias"))))
        };
        let ln =
            |n: &str| LayerNorm::new(n, store.w(&format!("{p}.{n}.weight")), store.w(&format!("{p}.{n}.bias")), 1e-6);
        MaskEmbedding {
            conv1: cv("conv1", 4, 1, 2, 2),
            conv2: cv("conv2", 16, 4, 2, 2),
            conv3: cv("conv3", 256, 16, 1, 1),
            ln1: ln("layer_norm1"),
            ln2: ln("layer_norm2"),
            gelu: build_gelu_lut(),
        }
    }

    /// Channels-first LayerNorm on an integer `[C,H,W]` tensor.
    fn ln2d_int(ln: &LayerNorm, x: &IntTensor, c: usize, h: usize, w: usize) -> IntTensor {
        let sp = chw_to_hwc(x);
        let t = ln.forward_int(&sp.as_tokens());
        hwc_to_chw(&Spatial::from_tokens(&t, 1, h, w, c))
    }

    /// `mask`: integer `[1,256,256]` → dense `[256,64,64]` channels-first.
    pub fn forward_int(&self, mask: &IntTensor) -> IntTensor {
        let h = self.conv1.forward_int(mask); // [4,128,128]
        let h = Self::ln2d_int(&self.ln1, &h, 4, 128, 128);
        let h = apply_gelu(&self.gelu, &h);
        let h = self.conv2.forward_int(&h); // [16,64,64]
        let h = Self::ln2d_int(&self.ln2, &h, 16, 64, 64);
        let h = apply_gelu(&self.gelu, &h);
        self.conv3.forward_int(&h) // [256,64,64]
    }

    /// Float twin — `mask` `[1,mh,mw]` f32 → dense `[256, mh/4, mw/4]` channels-first.
    pub fn forward_ref(&self, mask: &[f32], mh: usize, mw: usize) -> Vec<f32> {
        let h = self.conv1.forward_ref(mask, mh, mw);
        let (h2, w2) = (mh / 2, mw / 2);
        let h = hwc_to_chw_raw(&self.ln1.forward_ref(&chw_to_hwc_raw(&h, 4, h2, w2)), h2, w2, 4);
        let h: Vec<f32> = h.iter().map(|&v| gelu_f64(v as f64) as f32).collect();
        let h = self.conv2.forward_ref(&h, h2, w2);
        let (h3, w3) = (h2 / 2, w2 / 2);
        let h = hwc_to_chw_raw(&self.ln2.forward_ref(&chw_to_hwc_raw(&h, 16, h3, w3)), h3, w3, 16);
        let h: Vec<f32> = h.iter().map(|&v| gelu_f64(v as f64) as f32).collect();
        self.conv3.forward_ref(&h, h3, w3)
    }
}
