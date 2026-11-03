//! The SAM 2.1 architecture plugin — makes the image segmenter a first-class citizen of the generic converter,
//! exactly like the decoder LM. It implements the same [`kt_core::plugin::ModelArch`] contract (detect →
//! plan → build) and hands back a [`VerifiableModel`] whose `self_check` runs the whole zero-float pipeline on a
//! built-in probe (a synthetic image + point), so `kt-convert` dispatches to it with no SAM-specific code.

use kt_core::check::{CheckReport, VerifiableModel};
use kt_core::layers::{add, ACT_LEVELS};
use kt_core::plugin::{ArchMeta, LayerSpec, ModelArch, ModelPlan, ModelSource};
use kt_core::verify::hash_int;
use kt_core::IntTensor;

use crate::decoder::{broadcast_row, FMaskOutput, MaskOutput};
use crate::spatial::chw_to_hwc_raw;
use crate::{
    chw_to_hwc, hiera_pos_embed, image_wide_pos, preprocess_rgb, synthetic_test_image, HieraConfig, HieraEncoder,
    MaskDecoder, PromptEncoder, VisionNeck, WeightStore,
};

/// The assembled, runnable SAM 2.1 image segmenter (encoder + neck + prompt encoder + mask decoder + the
/// precomputed positional tables), ready to turn a preprocessed image + point into a mask.
pub struct SamSegmenter {
    enc: HieraEncoder,
    neck: VisionNeck,
    prompt: PromptEncoder,
    dec: MaskDecoder,
    no_mem: Vec<f32>,
    image_pos: Vec<f64>,
    // the constant inputs of the integer path, quantized once at build: the no-memory embedding `[1,256]`, the
    // image-wide positions `[64*64,256]`, the no-mask dense embedding `[1,256]`
    no_mem_q: IntTensor,
    image_pos_q: IntTensor,
    dense_q: IntTensor,
}

impl SamSegmenter {
    /// Segment: `pixel_values` `[3,1024,1024]` + a point (1024-pixel space) → mask logits / IoU / best index.
    pub fn segment(&self, pixel_values: &[f32], point: (f64, f64)) -> MaskOutput {
        let pixel = IntTensor::from_reals(
            &pixel_values.iter().map(|&v| v as f64).collect::<Vec<_>>(),
            vec![3, 1024, 1024],
            ACT_LEVELS,
        );
        let stages = self.enc.forward(&pixel);
        let feats = self.neck.image_features(&stages);
        let feat2_tokens = chw_to_hwc(&feats[2]).as_tokens();
        let hw = feat2_tokens.len() / 256;
        let no_mem_full = broadcast_row(&self.no_mem_q, hw);
        let image_embed = add(&feat2_tokens, &no_mem_full);
        let sparse = self.prompt.embed_points(&[point], &[1], true);
        self.dec.forward(&image_embed, &self.image_pos_q, &sparse, &self.dense_q, &feats[0], &feats[1])
    }

    /// Float twin of [`SamSegmenter::segment`] — the internal reference (no PyTorch) the integer mask is checked
    /// against. Mirrors `segment` op-for-op through the `forward_ref` paths.
    pub fn segment_ref(&self, pixel_values: &[f32], point: (f64, f64)) -> FMaskOutput {
        let stages = self.enc.forward_ref(pixel_values); // Vec<(tokens,h,w,c)>
        let feats = self.neck.image_features_ref(&stages); // Vec<(chw,c,h,w)>
        let (f2d, f2c, f2h, f2w) = &feats[2];
        let mut image_embed = chw_to_hwc_raw(f2d, *f2c, *f2h, *f2w); // [h*w, 256]
        let hw = image_embed.len() / 256;
        for t in 0..hw {
            for c in 0..256 {
                image_embed[t * 256 + c] += self.no_mem[c];
            }
        }
        let image_pos: Vec<f32> = self.image_pos.iter().map(|&v| v as f32).collect();
        let sparse = self.prompt.embed_points_ref(&[point], &[1], true);
        let dense = self.prompt.no_mask_dense().to_vec();
        self.dec.forward_ref(&image_embed, &image_pos, &sparse, &dense, &feats[0].0, &feats[1].0)
    }
}

impl VerifiableModel for SamSegmenter {
    fn task(&self) -> &str {
        "image-segmentation"
    }
    fn self_check(&self) -> CheckReport {
        let (img, h, w) = synthetic_test_image();
        let px = preprocess_rgb(&img, h, w, 1024);
        let int_out = self.segment(&px, (580.0, 510.0));
        let flt_out = self.segment_ref(&px, (580.0, 510.0)); // internal float twin — no PyTorch
        let hash = hash_int(&int_out.masks);
        let deterministic = hash == hash_int(&self.segment(&px, (580.0, 510.0)).masks);

        // integer logits vs the float twin: per-pixel sign agreement + logit MAE across the 3 output masks.
        let n = int_out.h * int_out.w;
        let s = int_out.masks.scale.to_f64();
        let (mut agree, mut mae) = (0usize, 0.0f64);
        for k in 0..3 {
            let ik = &int_out.masks.data[(k + 1) * n..(k + 2) * n];
            let fk = &flt_out.masks[(k + 1) * n..(k + 2) * n];
            for i in 0..n {
                let (ir, fr) = (ik[i] as f64 * s, fk[i] as f64);
                mae = mae.max((ir - fr).abs());
                if (ik[i] > 0) == (fk[i] > 0.0) {
                    agree += 1;
                }
            }
        }
        // best-mask IoU, integer vs float twin, and the exact differing-pixel count of the SELECTED mask.
        let (bi, bf) = (int_out.binary_best(), flt_out.binary_best());
        let (ib, fb) = (int_out.best_index(), flt_out.best_index());
        let same_candidate = ib == fb;
        let (mut inter, mut uni, mut sel_diff) = (0usize, 0usize, 0usize);
        for (&a, &b) in bi.iter().zip(bf.iter()) {
            if a == 1 || b == 1 {
                uni += 1;
            }
            if a == 1 && b == 1 {
                inter += 1;
            }
            if a != b {
                sel_diff += 1;
            }
        }
        let iou = if uni == 0 { 1.0 } else { inter as f64 / uni as f64 };
        CheckReport {
            task: self.task().into(),
            metric_name: "mask_iou_vs_float".into(),
            metric: iou,
            max_abs_err: mae,
            deterministic,
            hash,
            detail: format!(
                "int best {} / float best {} (same={same_candidate}); selected-mask differing px {sel_diff}/{n}; all-3 mask-sign agree {}/{}; IoU int-vs-float-twin {iou:.5}",
                ib - 1, fb - 1, agree, 3 * n
            ),
        }
    }
}

pub struct Sam2Arch;

impl ModelArch for Sam2Arch {
    fn meta(&self) -> ArchMeta {
        ArchMeta {
            family: "sam2-image (Hiera)".into(),
            hf_arch_ids: vec!["Sam2Model".into(), "Sam2VideoModel".into()],
            summary: "SAM 2.1 image: Hiera encoder + FPN neck + prompt encoder + two-way mask decoder (image path)"
                .into(),
        }
    }

    fn plan(&self, _src: &dyn ModelSource) -> Result<ModelPlan, String> {
        // A coarse graph preview (op-kind counts). Dims vary per stage; representative values shown.
        let cfg = HieraConfig::base_plus();
        let mut specs = vec![LayerSpec::Conv2d { c_in: 3, c_out: cfg.embed_dim_per_stage[0], k: cfg.patch_k }];
        for _ in 0..cfg.blocks_per_stage.iter().sum::<usize>() {
            specs.push(LayerSpec::LayerNorm { dim: cfg.embed_dim_per_stage[0] });
            specs.push(LayerSpec::Sdpa { heads: cfg.heads_per_stage[0], head_dim: 56 });
            specs.push(LayerSpec::LayerNorm { dim: cfg.embed_dim_per_stage[0] });
            specs.push(LayerSpec::Linear {
                in_features: cfg.embed_dim_per_stage[0],
                out_features: cfg.embed_dim_per_stage[0] * 4,
            });
            specs.push(LayerSpec::Linear {
                in_features: cfg.embed_dim_per_stage[0] * 4,
                out_features: cfg.embed_dim_per_stage[0],
            });
        }
        for cin in [896, 448, 224, 112, 256, 256] {
            specs.push(LayerSpec::Conv2d { c_in: cin, c_out: 256, k: 1 });
        }
        // two-way decoder: self + 2 cross per layer ×2 + final
        for _ in 0..7 {
            specs.push(LayerSpec::Sdpa { heads: 8, head_dim: 32 });
            specs.push(LayerSpec::LayerNorm { dim: 256 });
        }
        specs.push(LayerSpec::ConvTranspose2d { c_in: 256, c_out: 64, k: 2 });
        specs.push(LayerSpec::ConvTranspose2d { c_in: 64, c_out: 32, k: 2 });
        Ok(ModelPlan { name: "sam2-image".into(), family: self.meta().family, layers: specs })
    }

    fn build(&self, src: &dyn ModelSource) -> Result<Box<dyn VerifiableModel>, String> {
        let store = WeightStore::from_source(src);
        if !store.has("vision_encoder.backbone.patch_embed.projection.weight") {
            return Err("source has no SAM vision backbone tensors".into());
        }
        let cfg = HieraConfig::base_plus();
        let enc = HieraEncoder::from_store(&store, &cfg, &hiera_pos_embed(&store, 256, 112));
        let neck = VisionNeck::base_plus_from_store(&store);
        let prompt = PromptEncoder::from_store(&store);
        let dec = MaskDecoder::from_store(&store);
        let no_mem = store.w("no_memory_embedding").to_vec();
        let image_pos = image_wide_pos(&store, 64);
        let row = |v: &[f32]| {
            IntTensor::from_reals(&v.iter().map(|&x| x as f64).collect::<Vec<_>>(), vec![1, v.len()], ACT_LEVELS)
        };
        let no_mem_q = row(&no_mem);
        let image_pos_q = IntTensor::from_reals(&image_pos, vec![64 * 64, 256], ACT_LEVELS);
        let dense_q = row(prompt.no_mask_dense());
        Ok(Box::new(SamSegmenter { enc, neck, prompt, dec, no_mem, image_pos, no_mem_q, image_pos_q, dense_q }))
    }
}
