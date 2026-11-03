//! The FPN neck — a faithful exact-integer port of `Sam2VisionNeck` + the `conv_s0/conv_s1` projections that
//! `Sam2Model.get_image_features` applies. It turns the four Hiera stage-end feature maps into the three image
//! feature levels the mask decoder consumes (high→low resolution):
//!   - level 0 `[32, 256, 256]`  = `conv_s0(convs[3](stage0))`
//!   - level 1 `[64, 128, 128]`  = `conv_s1(convs[2](stage1))`
//!   - level 2 `[256, 64, 64]`   = `convs[1](stage2) + nearest_2x(convs[0](stage3))`
//!
//! All convolutions are 1×1 (`kt_core::conv::Conv2d`), run channels-first; the single top-down connection
//! (level 2) is a nearest-neighbour 2× upsample + integer add. `fpn_top_down_levels = [2,3]` means only level 2
//! receives the top-down term (level 3 is the base; levels 0,1 are lateral-only). The positional encodings the
//! neck also returns are for the video memory path, not the image mask decoder, so they are not produced here.

use kt_core::conv::Conv2d;
use kt_core::layers::add;
use kt_core::IntTensor;

use crate::spatial::{hwc_to_chw, hwc_to_chw_raw, nearest_up2x_raw, nearest_upsample_2x, Spatial};

pub struct VisionNeck {
    convs: Vec<Conv2d>, // convs[0..3]: backbone_channel_list [896,448,224,112] → 256 (stage3,2,1,0)
    conv_s0: Conv2d,    // 256 → 32  (level-0 high-res projection)
    conv_s1: Conv2d,    // 256 → 64  (level-1 high-res projection)
}

impl VisionNeck {
    /// Build from the real weights for facebook/sam2.1-hiera-base-plus (fpn_hidden_size=256, channel list
    /// `[896, 448, 224, 112]`, conv_s0→32, conv_s1→64).
    pub fn base_plus_from_store(store: &crate::hiera::WeightStore) -> Self {
        let chans = [896usize, 448, 224, 112];
        let fpn = 256usize;
        let convs = (0..4)
            .map(|i| {
                let p = format!("vision_encoder.neck.convs.{i}");
                Conv2d::from_weights(&p, store.w(&format!("{p}.weight")), fpn, chans[i], 1, 1, 1, 0)
                    .with_bias(Some(store.w(&format!("{p}.bias"))))
            })
            .collect();
        let conv_s0 = Conv2d::from_weights("conv_s0", store.w("mask_decoder.conv_s0.weight"), 32, fpn, 1, 1, 1, 0)
            .with_bias(Some(store.w("mask_decoder.conv_s0.bias")));
        let conv_s1 = Conv2d::from_weights("conv_s1", store.w("mask_decoder.conv_s1.weight"), 64, fpn, 1, 1, 1, 0)
            .with_bias(Some(store.w("mask_decoder.conv_s1.bias")));
        VisionNeck { convs, conv_s0, conv_s1 }
    }

    /// `stages` = the four Hiera stage-end maps `[s0(256²), s1(128²), s2(64²), s3(32²)]` (channels-last).
    /// Returns the three decoder image features, channels-first, high→low: `[ [32,256,256], [64,128,128],
    /// [256,64,64] ]`.
    pub fn image_features(&self, stages: &[Spatial]) -> Vec<IntTensor> {
        assert_eq!(stages.len(), 4, "neck expects 4 stage outputs");
        let l3 = self.convs[0].forward_int(&hwc_to_chw(&stages[3])); // [256,32,32]
        let l2 = self.convs[1].forward_int(&hwc_to_chw(&stages[2])); // [256,64,64]
        let l1 = self.convs[2].forward_int(&hwc_to_chw(&stages[1])); // [256,128,128]
        let l0 = self.convs[3].forward_int(&hwc_to_chw(&stages[0])); // [256,256,256]

        let lvl2 = add(&l2, &nearest_upsample_2x(&l3)); // top-down add (only level 2)

        let feat0 = self.conv_s0.forward_int(&l0); // [32,256,256]
        let feat1 = self.conv_s1.forward_int(&l1); // [64,128,128]
        let feat2 = lvl2; // [256,64,64]
        vec![feat0, feat1, feat2]
    }

    /// Float twin of [`VisionNeck::image_features`] — `stages[i] = (tokens [h·w,c], h, w, c)` channels-last →
    /// three features as channels-first `(data [c,h,w], c, h, w)`.
    pub fn image_features_ref(
        &self,
        stages: &[(Vec<f32>, usize, usize, usize)],
    ) -> Vec<(Vec<f32>, usize, usize, usize)> {
        let chw = |i: usize| -> (Vec<f32>, usize, usize) {
            let (d, h, w, c) = &stages[i];
            (hwc_to_chw_raw(d, *h, *w, *c), *h, *w)
        };
        let (c3, h3, w3) = chw(3);
        let l3 = self.convs[0].forward_ref(&c3, h3, w3);
        let (c2, h2, w2) = chw(2);
        let l2 = self.convs[1].forward_ref(&c2, h2, w2);
        let (c1, h1, w1) = chw(1);
        let l1 = self.convs[2].forward_ref(&c1, h1, w1);
        let (c0, h0, w0) = chw(0);
        let l0 = self.convs[3].forward_ref(&c0, h0, w0);
        let up = nearest_up2x_raw(&l3, 256, h3, w3); // [256, 2h3, 2w3] = [256, h2, w2]
        let lvl2: Vec<f32> = l2.iter().zip(up.iter()).map(|(&a, &b)| a + b).collect();
        let feat0 = self.conv_s0.forward_ref(&l0, h0, w0);
        let feat1 = self.conv_s1.forward_ref(&l1, h1, w1);
        vec![(feat0, 32, h0, w0), (feat1, 64, h1, w1), (lvl2, 256, h2, w2)]
    }
}
