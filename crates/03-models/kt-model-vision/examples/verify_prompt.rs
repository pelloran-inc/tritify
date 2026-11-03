//! Reference-driven verification of the prompt encoder: build the sparse (point + pad) and dense (no-mask)
//! embeddings for the point prompt and compare to the float `Sam2PromptEncoder` output.
//! Usage:  cargo run --release --example verify_prompt -- <sam_ref_dir>

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use kt_core::layers::ACT_LEVELS;
use kt_core::IntTensor;
use kt_model_vision::{MaskEmbedding, PromptEncoder, WeightStore};

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

fn cmp(label: &str, got: &[f64], rf: &[f32]) -> f64 {
    assert_eq!(got.len(), rf.len(), "{label}: size mismatch {} vs {}", got.len(), rf.len());
    let (mut mae, mut peak) = (0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(rf.iter()) {
        mae = mae.max((a - b as f64).abs());
        peak = peak.max((b as f64).abs());
    }
    let rel = if peak > 0.0 { mae / peak } else { 0.0 };
    println!("  {label:20}  max_abs_err {mae:.3e}  peak {peak:.3e}  rel {rel:.2e}");
    rel
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: verify_prompt <sam_ref_dir>");
    let dir = Path::new(&dir);
    let wdir = dir.join("weights");
    let mut store = HashMap::new();
    for (name, shape) in read_manifest(&wdir.join("manifest.txt")) {
        let vals = read_f32(&wdir.join(format!("{name}.f32")));
        store.insert(name, (vals, shape));
    }
    let store = WeightStore(store);
    let pe = PromptEncoder::from_store(&store);

    // point: [580,510] (1024-space), label 1 (foreground), padded (no box).
    let sparse = pe.embed_points(&[(580.0, 510.0)], &[1], true); // [2,256]
    let sparse_ref = read_f32(&dir.join("sparse_embeddings.f32")); // [1,1,2,256] = [2,256]

    // dense: no_mask broadcast over 64x64; reference dense_embeddings [1,256,64,64]; compare the constant vector.
    let dense_ref = read_f32(&dir.join("dense_embeddings.f32")); // [256,64,64]
    let dense_vec_ref: Vec<f32> = (0..256).map(|c| dense_ref[c * 64 * 64]).collect(); // [:,0,0]
    let dense_const = dense_ref.chunks(64 * 64).all(|ch| ch.iter().all(|&v| (v - ch[0]).abs() < 1e-6));

    // box prompt: [x1,y1,x2,y2] = [360,300,800,720] (1024-space) → 3 tokens.
    let sparse_box = pe.embed_boxes(&[(360.0, 300.0, 800.0, 720.0)]); // [3,256]
    let sparse_box_ref = read_f32(&dir.join("sparse_box.f32")); // [1,1,3,256] = [3,256]

    println!("prompt encoder verification:");
    let r1 = cmp("sparse (point+pad)", &sparse.to_f64(), &sparse_ref);
    let r2 =
        cmp("dense (no-mask vec)", &pe.no_mask_dense().iter().map(|&v| v as f64).collect::<Vec<_>>(), &dense_vec_ref);
    let r3 = cmp("sparse (box, 3 tok)", &sparse_box.to_f64(), &sparse_box_ref);

    // mask input: a fixed [1,256,256] mask → dense [256,64,64].
    let me = MaskEmbedding::from_store(&store);
    let mask_in = read_f32(&dir.join("mask_input.f32")); // [256,256]
    let mask_t =
        IntTensor::from_reals(&mask_in.iter().map(|&v| v as f64).collect::<Vec<_>>(), vec![1, 256, 256], ACT_LEVELS);
    let dense_mask = me.forward_int(&mask_t); // [256,64,64]
    let dense_mask_ref = read_f32(&dir.join("dense_mask.f32"));
    let r4 = cmp("mask dense [256,64,64]", &dense_mask.to_f64(), &dense_mask_ref);

    println!("  dense constant over spatial (reference): {dense_const}");
    let worst = r1.max(r2).max(r3).max(r4);
    println!("\nworst relative error: {worst:.3e}");
    println!(
        "{}",
        if worst < 5e-3 && dense_const {
            "PASS — the prompt encoder matches the float reference."
        } else {
            "DRIFT — investigate."
        }
    );
}
