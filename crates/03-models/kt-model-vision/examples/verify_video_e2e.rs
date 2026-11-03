//! End-to-end video check: given the REAL frame-1 memory (spatial memory of frame 0 + object-pointer tokens),
//! does the integer memory attention reproduce the real conditioned feature map? This proves the video compute
//! path (the stateful bank bookkeeping that assembles the memory is deterministic glue, provided here from the
//! real run). Usage: verify_video_e2e <model.safetensors> <sam_ref_dir>

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
    let mem_attn =
        MemoryAttention::from_store(&WeightStore::from_safetensors(&model).unwrap(), &VideoConfig::base_plus());

    let cur = it(&read_f32(&dir.join("vid1_current.f32")), vec![4096, 256]);
    let cpos = it(&read_f32(&dir.join("vid1_cur_pos.f32")), vec![4096, 256]);
    let mem_v = read_f32(&dir.join("vid1_memory.f32"));
    let m = mem_v.len() / 64; // 4100 (4096 spatial + 4 object-pointer tokens)
    let mem = it(&mem_v, vec![m, 64]);
    let mpos = it(&read_f32(&dir.join("vid1_mempos.f32")), vec![m, 64]);
    let rf = read_f32(&dir.join("vid1_conditioned.f32")); // [4096,256]
    let num_ptr = 4; // from vid1_meta.json (object pointer tokens, excluded from RoPE)

    println!(
        "video end-to-end: integer memory attention on the real frame-1 memory (M={m}, {num_ptr} pointer tokens):"
    );
    let oi = mem_attn.forward_int(&cur, &cpos, &mem, &mpos, num_ptr);
    cmp("integer", &oi.to_f64(), &rf);
    let or = mem_attn.forward_ref(
        &read_f32(&dir.join("vid1_current.f32")),
        &read_f32(&dir.join("vid1_cur_pos.f32")),
        &mem_v,
        &read_f32(&dir.join("vid1_mempos.f32")),
        num_ptr,
    );
    cmp("float twin", &or.iter().map(|&v| v as f64).collect::<Vec<_>>(), &rf);
    println!("  deterministic: {}", oi.data == mem_attn.forward_int(&cur, &cpos, &mem, &mpos, num_ptr).data);
}
