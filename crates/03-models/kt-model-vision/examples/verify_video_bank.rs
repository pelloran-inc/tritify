//! Verify the video memory-bank assembly glue: from the real frame-0 bank entry (maskmem + pos + object
//! pointer), assemble the frame-1 memory and check it matches the real assembled memory the model fed to the
//! memory attention. Usage: verify_video_bank <model.safetensors> <sam_ref_dir>

use kt_model_vision::{FrameMemory, VideoMemoryBank, WeightStore};
use std::fs;
use std::path::Path;

fn read_f32(p: &Path) -> Vec<f32> {
    fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn cmp(label: &str, got: &[f32], rf: &[f32]) {
    assert_eq!(got.len(), rf.len(), "{label}: {} vs {}", got.len(), rf.len());
    let (mut mae, mut peak) = (0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(rf.iter()) {
        mae = mae.max((a as f64 - b as f64).abs());
        peak = peak.max((b as f64).abs());
    }
    println!("  {label:16} max_abs_err {mae:.3e} peak {peak:.3e} rel {:.2e}", mae / peak.max(1e-12));
}

fn main() {
    let model = std::env::args().nth(1).unwrap();
    let dir = std::env::args().nth(2).unwrap();
    let dir = Path::new(&dir);
    let bank = VideoMemoryBank::from_store(&WeightStore::from_safetensors(&model).unwrap(), 7, 64, 256, 16);

    let f0 = FrameMemory {
        maskmem: read_f32(&dir.join("f0_maskmem.f32")),               // [4096,64]
        maskmem_pos: read_f32(&dir.join("f0_maskmem_pos.f32")),       // [4096,64]
        object_pointer: read_f32(&dir.join("f0_object_pointer.f32")), // [256]
        is_conditioning: true,
        frame_idx: 0,
    };
    let (mem, pos, num_ptr) = bank.assemble(&[f0], 1, 2, 4096);

    let ref_mem = read_f32(&dir.join("vid1_memory.f32")); // [4100,64]
    let ref_pos = read_f32(&dir.join("vid1_mempos.f32"));
    println!(
        "video memory-bank assembly (frame 1 from frame-0 entry): {} tokens, {num_ptr} pointer tokens",
        mem.len() / 64
    );
    cmp("combined memory", &mem, &ref_mem);
    cmp("combined pos", &pos, &ref_pos);
    let n_sp = 4096 * 64;
    cmp("  spatial pos", &pos[..n_sp], &ref_pos[..n_sp]);
    cmp("  pointer pos", &pos[n_sp..], &ref_pos[n_sp..]);
    println!("  mine   pointer pos[0..4]  = {:?}", &pos[n_sp..n_sp + 4]);
    println!("  float  pointer pos[0..4]  = {:?}", &ref_pos[n_sp..n_sp + 4]);
}
