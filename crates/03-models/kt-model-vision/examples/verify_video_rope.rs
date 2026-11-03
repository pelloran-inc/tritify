//! Verify the Rust 2D-RoPE cos/sin tables against the real Sam2VideoVisionRotaryEmbedding dump.
//! Usage:  cargo run --release --example verify_video_rope -- <sam_ref_dir>

use kt_model_vision::{rope_tables, VideoConfig};
use std::fs;
use std::path::Path;

fn read_f32(p: &Path) -> Vec<f32> {
    fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn cmp(label: &str, got: &[f64], rf: &[f32]) -> f64 {
    assert_eq!(got.len(), rf.len(), "{label}: {} vs {}", got.len(), rf.len());
    let mut mae = 0.0f64;
    for (&a, &b) in got.iter().zip(rf.iter()) {
        mae = mae.max((a - b as f64).abs());
    }
    println!("  {label:10}  max_abs_err {mae:.3e}");
    mae
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: verify_video_rope <sam_ref_dir>");
    let dir = Path::new(&dir);
    let (cos, sin) = rope_tables(&VideoConfig::base_plus());
    println!("2D-RoPE table verification:");
    let a = cmp("rope_cos", &cos, &read_f32(&dir.join("rope_cos.f32")));
    let b = cmp("rope_sin", &sin, &read_f32(&dir.join("rope_sin.f32")));
    let worst = a.max(b);
    println!("\nworst max_abs_err: {worst:.3e}");
    println!(
        "{}",
        if worst < 1e-5 { "PASS — Rust 2D-RoPE tables match the real model." } else { "DRIFT — investigate." }
    );
}
