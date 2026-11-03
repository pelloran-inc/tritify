//! End-to-end verification of the full SAM 2.1 image path: run the integer encoder → neck → prompt encoder →
//! mask decoder on the real input and compare the predicted mask to the float `Sam2Model` — the headline
//! (best-mask IoU vs float == 1.0, deterministic) — plus a reproducing SHA-256 over the integer mask logits.
//! Usage:  cargo run --release --example verify_decoder -- <sam_ref_dir>

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use kt_core::layers::{add, ACT_LEVELS};
use kt_core::IntTensor;
use kt_model_vision::{
    chw_to_hwc, hiera_pos_embed, image_wide_pos, HieraConfig, HieraEncoder, MaskDecoder, PromptEncoder, VisionNeck,
    WeightStore,
};
use sha2::{Digest, Sha256};

fn read_f32(p: &Path) -> Vec<f32> {
    fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn read_manifest(p: &Path) -> Vec<(String, Vec<usize>)> {
    fs::read_to_string(p)
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (n, s) = l.split_once('\t').unwrap();
            (n.to_string(), s.split(',').map(|x| x.parse().unwrap()).collect())
        })
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
    let dir = std::env::args().nth(1).expect("usage: verify_decoder <sam_ref_dir>");
    let dir = Path::new(&dir);
    let wdir = dir.join("weights");
    let mut store = HashMap::new();
    for (name, shape) in read_manifest(&wdir.join("manifest.txt")) {
        let vals = read_f32(&wdir.join(format!("{name}.f32")));
        store.insert(name, (vals, shape));
    }
    let store = WeightStore(store);
    let load_ref = |n: &str| -> Vec<f32> { read_f32(&dir.join(format!("{n}.f32"))) };

    let cfg = HieraConfig::base_plus();
    let pixel = IntTensor::from_reals(
        &load_ref("pixel_values").iter().map(|&v| v as f64).collect::<Vec<_>>(),
        vec![3, 1024, 1024],
        ACT_LEVELS,
    );
    // positional tables computed in Rust (not loaded from the dump) → tests the fully-standalone path.
    let pos_f64 = hiera_pos_embed(&store, 256, 112);

    let enc = HieraEncoder::from_store(&store, &cfg, &pos_f64);
    let neck = VisionNeck::base_plus_from_store(&store);
    let prompt = PromptEncoder::from_store(&store);
    let dec = MaskDecoder::from_store(&store);

    // encoder → neck
    let stages = enc.forward(&pixel);
    let feats = neck.image_features(&stages); // [feat0 32x256x256, feat1 64x128x128, feat2 256x64x64]

    // image embedding tokens = feat2 + no_memory_embedding (broadcast)
    let feat2_tokens = chw_to_hwc(&feats[2]).as_tokens(); // [4096,256]
    let no_mem = store.w("no_memory_embedding"); // [256]
    let hw = feat2_tokens.len() / 256;
    let no_mem_full = IntTensor::from_reals(
        &(0..hw).flat_map(|_| no_mem.iter().map(|&v| v as f64)).collect::<Vec<_>>(),
        vec![hw, 256],
        ACT_LEVELS,
    );
    let image_embed_tokens = add(&feat2_tokens, &no_mem_full);

    // image-wide positional embedding, computed in Rust (Fourier) → tokens [4096,256]
    let image_pos = IntTensor::from_reals(&image_wide_pos(&store, 64), vec![64 * 64, 256], ACT_LEVELS);

    // prompt (point [580,510], label 1, padded)
    let sparse = prompt.embed_points(&[(580.0, 510.0)], &[1], true);
    let dense_vec = prompt.no_mask_dense().to_vec();
    let dense =
        IntTensor::from_reals(&dense_vec.iter().map(|&v| v as f64).collect::<Vec<_>>(), vec![1, 256], ACT_LEVELS);

    let out = dec.forward(&image_embed_tokens, &image_pos, &sparse, &dense, &feats[0], &feats[1]);

    // float reference masks [3,256,256] + iou [3] (indices 1..4 of the 4)
    let pm = load_ref("pred_masks"); // [1,1,3,256,256]
    let iou_ref = load_ref("iou_scores"); // [1,1,3]
    let n = 256 * 256;
    let best_f = (0..3).max_by(|&a, &b| iou_ref[a].partial_cmp(&iou_ref[b]).unwrap()).unwrap();
    let binary_f: Vec<u8> = pm[best_f * n..(best_f + 1) * n].iter().map(|&v| u8::from(v > 0.0)).collect();

    // mine: masks rows 1..4 (multimask), best by my iou
    let best_mine = out.best_index(); // 1..4 index into the 4
    let binary_mine = out.binary_best();

    // per-pixel sign agreement across all 3 output masks (1..4)
    let mut agree = 0usize;
    let total = 3 * n;
    for k in 0..3 {
        let mine_k = &out.masks.data[(k + 1) * n..(k + 2) * n];
        let ref_k = &pm[k * n..(k + 1) * n];
        for (i, &rv) in ref_k.iter().enumerate() {
            if (mine_k[i] > 0) == (rv > 0.0) {
                agree += 1;
            }
        }
    }

    // object score logits (video tracker output) vs the dumped reference.
    let obj_ref = read_f32(&dir.join("object_score_logits.f32"))[0];
    println!(
        "object_score_logits: mine {:.4}  float {:.4}  (abs diff {:.3e})",
        out.obj_score_f32(),
        obj_ref,
        (out.obj_score_f32() - obj_ref).abs()
    );
    println!("float best mask index (of 3): {best_f}   mine: {} (→ mask {})", out.best_index(), best_mine - 1);
    println!("per-pixel mask-sign agreement vs float: {}/{} = {:.5}", agree, total, agree as f64 / total as f64);
    println!("best-mask IoU vs float: {:.5}", iou(&binary_f, &binary_mine));

    let mut hasher = Sha256::new();
    for &v in &out.masks.data {
        hasher.update(v.to_le_bytes());
    }
    let hash = hasher.finalize();
    println!("mask-logits SHA-256: {}", hash.iter().map(|b| format!("{b:02x}")).collect::<String>());

    let pass = iou(&binary_f, &binary_mine) >= 0.999 && (out.best_index() - 1) == best_f;
    println!(
        "\n{}",
        if pass {
            "PASS — zero-float integer SAM mask == float SAM mask (IoU 1.0)."
        } else {
            "MISMATCH — investigate."
        }
    );
}
