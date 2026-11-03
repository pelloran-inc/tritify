//! Reference-driven verification of the FPN neck: run the integer Hiera encoder + neck on the real input and
//! compare the three decoder image features to the float `Sam2Model`'s `get_image_features` output
//! (fpn_hidden_0/1/2). Usage:  cargo run --release --example verify_neck -- <sam_ref_dir>

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use kt_core::layers::ACT_LEVELS;
use kt_core::IntTensor;
use kt_model_vision::{chw_to_hwc, HieraConfig, HieraEncoder, VisionNeck, WeightStore};

fn read_f32(p: &Path) -> Vec<f32> {
    fs::read(p)
        .unwrap_or_else(|e| panic!("read {p:?}: {e}"))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
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

fn main() {
    let dir = std::env::args().nth(1).expect("usage: verify_neck <sam_ref_dir>");
    let dir = Path::new(&dir);
    let wdir = dir.join("weights");

    let mut store = HashMap::new();
    for (name, shape) in read_manifest(&wdir.join("manifest.txt")) {
        let vals = read_f32(&wdir.join(format!("{name}.f32")));
        assert_eq!(vals.len(), shape.iter().product::<usize>(), "size mismatch {name}");
        store.insert(name, (vals, shape));
    }
    let store = WeightStore(store);
    let load_ref = |n: &str| -> Vec<f32> { read_f32(&dir.join(format!("{n}.f32"))) };

    let cfg = HieraConfig::base_plus();
    let pixel_f32 = load_ref("pixel_values");
    let pixel = IntTensor::from_reals(
        &pixel_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        vec![3, 1024, 1024],
        ACT_LEVELS,
    );
    let pos_f64: Vec<f64> = load_ref("hiera_pos_embed").iter().map(|&v| v as f64).collect();

    let enc = HieraEncoder::from_store(&store, &cfg, &pos_f64);
    let neck = VisionNeck::base_plus_from_store(&store);

    let stages = enc.forward(&pixel); // 4 stage-end maps
    let feats = neck.image_features(&stages); // 3 features, channels-first

    println!("{:>6}  {:>14}  {:>12}  {:>12}  {:>9}", "level", "shape(HWxC)", "max_abs_err", "peak_ref", "rel");
    let mut worst = 0.0f64;
    for (i, feat) in feats.iter().enumerate() {
        // channels-first [C,H,W] → tokens [H*W, C]
        let tok = chw_to_hwc(feat).as_tokens().to_f64();
        let rf = load_ref(&format!("fpn_hidden_{i}")); // [H*W,1,C] == [H*W*C]
        assert_eq!(tok.len(), rf.len(), "level {i} size mismatch");
        let (mut mae, mut peak) = (0.0f64, 0.0f64);
        for (&a, &b) in tok.iter().zip(rf.iter()) {
            mae = mae.max((a - b as f64).abs());
            peak = peak.max((b as f64).abs());
        }
        let rel = if peak > 0.0 { mae / peak } else { 0.0 };
        worst = worst.max(rel);
        let (c, h, w) = (feat.shape[0], feat.shape[1], feat.shape[2]);
        println!("{:>6}  {:>14}  {:>12.3e}  {:>12.3e}  {:>9.2e}", i, format!("{}x{}", h * w, c), mae, peak, rel);
    }
    println!("\nworst relative error across the 3 feature levels: {worst:.3e}");
    println!(
        "{}",
        if worst < 2e-2 {
            "PASS — the integer FPN neck tracks the float reference."
        } else {
            "DRIFT — investigate."
        }
    );
}
