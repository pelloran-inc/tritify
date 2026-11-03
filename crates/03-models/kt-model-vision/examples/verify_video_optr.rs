//! Verify the object-pointer head (object_pointer_proj) against the real op. Usage: <model.safetensors> <ref_dir>
use kt_core::layers::ACT_LEVELS;
use kt_core::IntTensor;
use kt_model_vision::{ObjectPointerHead, WeightStore};
use std::fs;
use std::path::Path;

fn read_f32(p: &Path) -> Vec<f32> {
    fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let model = std::env::args().nth(1).unwrap();
    let dir = std::env::args().nth(2).unwrap();
    let dir = Path::new(&dir);
    let head = ObjectPointerHead::from_store(&WeightStore::from_safetensors(&model).unwrap());
    let tok = read_f32(&dir.join("optr_tok.f32"));
    let rf = read_f32(&dir.join("optr_out.f32"));
    let ti = IntTensor::from_reals(&tok.iter().map(|&v| v as f64).collect::<Vec<_>>(), vec![1, 256], ACT_LEVELS);
    let oi = head.forward_int(&ti, 5.0); // obj present
    let or = head.forward_ref(&tok, 5.0);
    let rep = |label: &str, got: &[f32]| {
        let (mut mae, mut peak) = (0.0f64, 0.0f64);
        for (&a, &b) in got.iter().zip(rf.iter()) {
            mae = mae.max((a as f64 - b as f64).abs());
            peak = peak.max((b as f64).abs());
        }
        println!("  {label:10} max_abs_err {mae:.3e} peak {peak:.3e} rel {:.2e}", mae / peak);
    };
    println!("object-pointer head verification:");
    rep("integer", &oi);
    rep("float twin", &or);
}
