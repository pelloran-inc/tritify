//! Standalone SAM 2.1 segmentation from the real checkpoint — the open demo. Load the weights straight from
//! `model.safetensors`, compute the positional tables in Rust, optionally generate + preprocess the image in
//! Rust, run the whole zero-float integer pipeline (Hiera encoder → FPN neck → prompt encoder → mask decoder),
//! and emit the mask + a reproducing SHA-256 — no PyTorch anywhere.
//!
//!   cargo run --release --example sam2_segment -- <model.safetensors> <synthetic|pixel_values.f32> [ref_dir]
//!
//! `synthetic` generates the synthetic tan-rectangle image and preprocesses it in Rust (image → mask, fully Rust);
//! a `.f32` path uses a preprocessed tensor instead. If `ref_dir` is given, the mask is scored (IoU) against a
//! reference `pred_masks.f32` from the float model. The point defaults to the built-in prompt (580,510).

use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

use kt_core::layers::{add, ACT_LEVELS};
use kt_core::IntTensor;
use kt_model_vision::{
    chw_to_hwc, hiera_pos_embed, image_wide_pos, preprocess_rgb, synthetic_test_image, HieraConfig, HieraEncoder,
    MaskDecoder, PromptEncoder, VisionNeck, WeightStore,
};

fn read_f32(p: &Path) -> Vec<f32> {
    fs::read(p)
        .unwrap_or_else(|e| panic!("read {p:?}: {e}"))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn iou(a: &[u8], b: &[u8]) -> f64 {
    let (mut i, mut u) = (0usize, 0usize);
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x == 1 || y == 1 {
            u += 1;
        }
        if x == 1 && y == 1 {
            i += 1;
        }
    }
    if u == 0 {
        1.0
    } else {
        i as f64 / u as f64
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: sam2_segment <model.safetensors> <synthetic|pixel_values.f32> [ref_dir]");
        std::process::exit(2);
    }

    eprintln!("loading checkpoint {} …", args[1]);
    let store = WeightStore::from_safetensors(&args[1]).unwrap();

    let cfg = HieraConfig::base_plus();
    let enc = HieraEncoder::from_store(&store, &cfg, &hiera_pos_embed(&store, 256, 112));
    let neck = VisionNeck::base_plus_from_store(&store);
    let prompt = PromptEncoder::from_store(&store);
    let dec = MaskDecoder::from_store(&store);

    let pixel_f32 = if args[2] == "synthetic" {
        eprintln!("generating + preprocessing the synthetic image in Rust …");
        let (img, h, w) = synthetic_test_image();
        preprocess_rgb(&img, h, w, 1024)
    } else {
        read_f32(Path::new(&args[2]))
    };
    let pixel = IntTensor::from_reals(
        &pixel_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        vec![3, 1024, 1024],
        ACT_LEVELS,
    );

    eprintln!("running encoder → neck → prompt → decoder …");
    let stages = enc.forward(&pixel);
    let feats = neck.image_features(&stages);

    let feat2_tokens = chw_to_hwc(&feats[2]).as_tokens();
    let no_mem = store.w("no_memory_embedding");
    let hw = feat2_tokens.len() / 256;
    let no_mem_full = IntTensor::from_reals(
        &(0..hw).flat_map(|_| no_mem.iter().map(|&v| v as f64)).collect::<Vec<_>>(),
        vec![hw, 256],
        ACT_LEVELS,
    );
    let image_embed = add(&feat2_tokens, &no_mem_full);
    let image_pos = IntTensor::from_reals(&image_wide_pos(&store, 64), vec![64 * 64, 256], ACT_LEVELS);

    let sparse = prompt.embed_points(&[(580.0, 510.0)], &[1], true);
    let dense_vec = prompt.no_mask_dense().to_vec();
    let dense =
        IntTensor::from_reals(&dense_vec.iter().map(|&v| v as f64).collect::<Vec<_>>(), vec![1, 256], ACT_LEVELS);

    let out = dec.forward(&image_embed, &image_pos, &sparse, &dense, &feats[0], &feats[1]);

    let mask = out.binary_best();
    let fg: usize = mask.iter().map(|&v| v as usize).sum();
    let mut hasher = Sha256::new();
    for &v in &out.masks.data {
        hasher.update(v.to_le_bytes());
    }
    let hash: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();

    println!(
        "best mask {} of 3   foreground {fg}/{} ({:.1}%)",
        out.best_index() - 1,
        mask.len(),
        100.0 * fg as f64 / mask.len() as f64
    );
    println!("mask-logits SHA-256: {hash}");

    if let Some(refdir) = args.get(3) {
        let pm = read_f32(&Path::new(refdir).join("pred_masks.f32")); // [3,256,256]
        let iou_ref = read_f32(&Path::new(refdir).join("iou_scores.f32")); // [3]
        let n = 256 * 256;
        let best_f = (0..3).max_by(|&a, &b| iou_ref[a].partial_cmp(&iou_ref[b]).unwrap()).unwrap();
        let binary_f: Vec<u8> = pm[best_f * n..(best_f + 1) * n].iter().map(|&v| u8::from(v > 0.0)).collect();
        println!("best-mask IoU vs float SAM: {:.5}", iou(&binary_f, &mask));
    }
}
