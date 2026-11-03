//! Reference-driven verification of the Hiera encoder port: load the real sam2.1-hiera-base-plus backbone
//! weights and the float `Sam2Model` intermediates (dumped by the Python harness as raw little-endian f32 +
//! plaintext manifests), run the zero-float integer encoder, and compare EACH of the 24 block outputs to the
//! real float reference. Usage:
//!
//!   cargo run --release --example verify_hiera -- <sam_ref_dir>
//!
//! where <sam_ref_dir> contains `weights/manifest.txt` + `weights/*.f32`, `ref_manifest.txt`, and the
//! `*.f32` reference tensors. (Dev-only: the data lives outside the repo.)

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use kt_core::layers::ACT_LEVELS;
use kt_core::IntTensor;
use kt_model_vision::{HieraConfig, HieraEncoder, WeightStore};

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn read_manifest(path: &Path) -> Vec<(String, Vec<usize>)> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {path:?}: {e}"))
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (name, shape) = l.split_once('\t').expect("manifest line");
            (name.to_string(), shape.split(',').map(|s| s.parse().unwrap()).collect())
        })
        .collect()
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: verify_hiera <sam_ref_dir>");
    let dir = Path::new(&dir);

    // --- weights ---
    let wdir = dir.join("weights");
    let mut store = HashMap::new();
    for (name, shape) in read_manifest(&wdir.join("manifest.txt")) {
        let vals = read_f32(&wdir.join(format!("{name}.f32")));
        assert_eq!(vals.len(), shape.iter().product::<usize>(), "size mismatch for {name}");
        store.insert(name, (vals, shape));
    }
    let store = WeightStore(store);
    println!("loaded {} weight tensors", store.0.len());

    // --- reference intermediates ---
    let refs: HashMap<String, Vec<usize>> = read_manifest(&dir.join("ref_manifest.txt")).into_iter().collect();
    let load_ref =
        |name: &str| -> (Vec<f32>, Vec<usize>) { (read_f32(&dir.join(format!("{name}.f32"))), refs[name].clone()) };

    // --- build the encoder ---
    let cfg = HieraConfig::base_plus();
    let (pixel_f32, pshape) = load_ref("pixel_values"); // [1,3,1024,1024]
    assert_eq!(pshape, vec![1, 3, 1024, 1024]);
    let pixel_f64: Vec<f64> = pixel_f32.iter().map(|&v| v as f64).collect();
    let pixel = IntTensor::from_reals(&pixel_f64, vec![3, 1024, 1024], ACT_LEVELS);

    let (pos_f32, _) = load_ref("hiera_pos_embed"); // [1,256,256,112] → tokens [256*256,112]
    let pos_f64: Vec<f64> = pos_f32.iter().map(|&v| v as f64).collect();

    let enc = HieraEncoder::from_store(&store, &cfg, &pos_f64);
    println!("built encoder: {} blocks, stage_ends={:?}\n", enc.num_blocks(), enc.stage_ends());

    // --- run + verify block by block ---
    let outs = enc.forward_capture(&pixel);
    // determinism: a second run must be bit-identical
    let outs2 = enc.forward_capture(&pixel);

    println!("{:>5}  {:>14}  {:>12}  {:>12}  {:>9}  det", "block", "shape", "max_abs_err", "peak_ref", "rel");
    let mut worst_rel = 0.0f64;
    for i in 0..enc.num_blocks() {
        let got = &outs[i];
        let (rf, rshape) = load_ref(&format!("block_{i:02}_out"));
        assert_eq!(
            got.data.len(),
            rf.len(),
            "block {i} length mismatch (shape {:?} vs ref {:?})",
            got_shape(got),
            rshape
        );
        let g = got.as_tokens().to_f64();
        let (mut mae, mut peak) = (0.0f64, 0.0f64);
        for (&a, &b) in g.iter().zip(rf.iter()) {
            mae = mae.max((a - b as f64).abs());
            peak = peak.max((b as f64).abs());
        }
        let rel = if peak > 0.0 { mae / peak } else { 0.0 };
        worst_rel = worst_rel.max(rel);
        let det = outs[i].data == outs2[i].data;
        println!(
            "{:>5}  {:>14}  {:>12.3e}  {:>12.3e}  {:>9.2e}  {}",
            i,
            format!("{}x{}x{}", got.h, got.w, got.c),
            mae,
            peak,
            rel,
            if det { "ok" } else { "FAIL" }
        );
    }
    println!("\nworst relative error across all 24 blocks: {worst_rel:.3e}");
    println!(
        "{}",
        if worst_rel < 2e-2 {
            "PASS — the integer Hiera encoder tracks the float reference."
        } else {
            "DRIFT — investigate."
        }
    );
}

fn got_shape(s: &kt_model_vision::Spatial) -> Vec<usize> {
    vec![s.b, s.h, s.w, s.c]
}
