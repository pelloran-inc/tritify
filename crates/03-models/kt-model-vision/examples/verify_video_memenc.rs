//! Verify the memory encoder (mask downsampler + feature projection + ConvNeXt fuser + projection) against the
//! real Sam2VideoMemoryEncoder. Usage: verify_video_memenc <model.safetensors> <sam_ref_dir>

use kt_core::layers::ACT_LEVELS;
use kt_core::IntTensor;
use kt_model_vision::{MemoryEncoder, WeightStore};
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
    println!("  {label:12}  max_abs_err {mae:.3e}  peak {peak:.3e}  rel {:.2e}", mae / peak.max(1e-12));
}

fn main() {
    let model = std::env::args().nth(1).unwrap();
    let dir = std::env::args().nth(2).unwrap();
    let dir = Path::new(&dir);
    let enc = MemoryEncoder::from_store(&WeightStore::from_safetensors(&model).unwrap());

    let vision = read_f32(&dir.join("menc_vision.f32"));
    let mask = read_f32(&dir.join("menc_mask.f32"));
    let rf = read_f32(&dir.join("menc_out.f32"));
    let vt =
        IntTensor::from_reals(&vision.iter().map(|&v| v as f64).collect::<Vec<_>>(), vec![256, 64, 64], ACT_LEVELS);
    let mt =
        IntTensor::from_reals(&mask.iter().map(|&v| v as f64).collect::<Vec<_>>(), vec![1, 1024, 1024], ACT_LEVELS);

    println!("memory encoder verification:");
    let oi = enc.forward_int(&vt, &mt);
    cmp("integer", &oi.to_f64(), &rf);
    let or = enc.forward_ref(&vision, &mask);
    cmp("float twin", &or.iter().map(|&v| v as f64).collect::<Vec<_>>(), &rf);
    println!("  deterministic: {}", oi.data == enc.forward_int(&vt, &mt).data);
}
