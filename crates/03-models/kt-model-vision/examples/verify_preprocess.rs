//! Verify the Rust preprocessing (generate the synthetic image → bilinear resize + normalize) against the SAM 2.1
//! processor's dumped pixel_values.  Usage: cargo run --release --example verify_preprocess -- <sam_ref_dir>

use kt_model_vision::{preprocess_rgb, synthetic_test_image};
use std::fs;
use std::path::Path;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: verify_preprocess <sam_ref_dir>");
    let (img, h, w) = synthetic_test_image();
    let got = preprocess_rgb(&img, h, w, 1024); // [3,1024,1024]
    let rf: Vec<f32> = fs::read(Path::new(&dir).join("pixel_values.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got.len(), rf.len(), "size {} vs {}", got.len(), rf.len());
    let (mut mae, mut peak) = (0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(rf.iter()) {
        mae = mae.max((a as f64 - b as f64).abs());
        peak = peak.max((b as f64).abs());
    }
    println!("preprocess vs processor: max_abs_err {mae:.3e}  peak {peak:.3e}  rel {:.2e}", mae / peak);
    println!(
        "{}",
        if mae / peak < 1e-2 {
            "PASS — Rust preprocessing matches the SAM processor."
        } else {
            "DRIFT — bilinear convention differs; investigate."
        }
    );
}
