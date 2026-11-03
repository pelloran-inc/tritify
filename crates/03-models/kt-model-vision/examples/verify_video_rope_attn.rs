//! Verify VideoRoPEAttention (self-attention) against the real Sam2VideoRoPEAttention op.
//! Usage:  cargo run --release --example verify_video_rope_attn -- <model.safetensors> <sam_ref_dir>

use kt_core::layers::ACT_LEVELS;
use kt_core::IntTensor;
use kt_model_vision::{VideoConfig, VideoRoPEAttention, WeightStore};
use std::fs;
use std::path::Path;

fn read_f32(p: &Path) -> Vec<f32> {
    fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn cmp(label: &str, got: &[f64], rf: &[f32]) {
    let (mut mae, mut peak) = (0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(rf.iter()) {
        mae = mae.max((a - b as f64).abs());
        peak = peak.max((b as f64).abs());
    }
    println!("  {label:14}  max_abs_err {mae:.3e}  peak {peak:.3e}  rel {:.2e}", mae / peak);
}

fn main() {
    let model = std::env::args().nth(1).expect("usage: <model.safetensors> <sam_ref_dir>");
    let dir = std::env::args().nth(2).expect("ref dir");
    let dir = Path::new(&dir);
    let store = WeightStore::from_safetensors(&model).unwrap();
    let cfg = VideoConfig::base_plus();
    let attn = VideoRoPEAttention::from_store(&store, "memory_attention.layers.0.self_attn", &cfg, 256);

    let x_f32 = read_f32(&dir.join("vrope_x.f32")); // [4096,256]
    let x = IntTensor::from_reals(&x_f32.iter().map(|&v| v as f64).collect::<Vec<_>>(), vec![4096, 256], ACT_LEVELS);
    let rf = read_f32(&dir.join("vrope_self_out.f32"));

    println!("VideoRoPEAttention self-attn verification:");
    let oi = attn.forward_int(&x, &x, &x, 0, 0);
    cmp("integer", &oi.to_f64(), &rf);
    let or = attn.forward_ref(&x_f32, &x_f32, &x_f32, 0, 0);
    cmp("float twin", &or.iter().map(|&v| v as f64).collect::<Vec<_>>(), &rf);
    // determinism
    let det = oi.data == attn.forward_int(&x, &x, &x, 0, 0).data;
    println!("  deterministic: {det}");
}
