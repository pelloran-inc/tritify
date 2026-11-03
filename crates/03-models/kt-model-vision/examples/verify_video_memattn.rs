//! Verify the memory-attention transformer (4 layers) against the real Sam2VideoMemoryAttention on synthetic
//! current-features + a 2-frame memory. Usage: verify_video_memattn <model.safetensors> <sam_ref_dir>

use kt_core::layers::ACT_LEVELS;
use kt_core::IntTensor;
use kt_model_vision::{MemoryAttention, VideoConfig, WeightStore};
use std::fs;
use std::path::Path;

fn read_f32(p: &Path) -> Vec<f32> {
    fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn it(v: &[f32], shape: Vec<usize>) -> IntTensor {
    IntTensor::from_reals(&v.iter().map(|&x| x as f64).collect::<Vec<_>>(), shape, ACT_LEVELS)
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
    let store = WeightStore::from_safetensors(&model).unwrap();
    let mem_attn = MemoryAttention::from_store(&store, &VideoConfig::base_plus());

    let cur = read_f32(&dir.join("mem_cur.f32"));
    let cur_pos = read_f32(&dir.join("mem_cur_pos.f32"));
    let mem = read_f32(&dir.join("mem_mem.f32"));
    let mem_pe = read_f32(&dir.join("mem_mem_pe.f32"));
    let rf = read_f32(&dir.join("mem_out.f32"));

    let (ci, cpi, mi, mpi) = (
        it(&cur, vec![4096, 256]),
        it(&cur_pos, vec![4096, 256]),
        it(&mem, vec![8192, 64]),
        it(&mem_pe, vec![8192, 64]),
    );
    println!("memory-attention transformer verification (2-frame memory):");
    let oi = mem_attn.forward_int(&ci, &cpi, &mi, &mpi, 0);
    cmp("integer", &oi.to_f64(), &rf);
    let or = mem_attn.forward_ref(&cur, &cur_pos, &mem, &mem_pe, 0);
    cmp("float twin", &or.iter().map(|&v| v as f64).collect::<Vec<_>>(), &rf);
    println!("  deterministic: {}", oi.data == mem_attn.forward_int(&ci, &cpi, &mi, &mpi, 0).data);
}
