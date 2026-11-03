//! Verify the Rust memory sine positional encoding vs the real Sam2VideoPositionEmbeddingSine dump.
//! Usage: verify_video_sinepos <sam_ref_dir>
use kt_model_vision::sine_pos_table;
use std::fs;
use std::path::Path;

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let dir = Path::new(&dir);
    let got = sine_pos_table(64, 64, 32, 10000.0); // [4096,64] tokens (t=y*64+x, channel)
    let dump: Vec<f32> = fs::read(dir.join("maskmem_pos_enc.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect(); // [1,64,64,64] = [c,y,x]
                    // dump token-major: ref[t, c] = dump[c*4096 + t]
    let (mut mae, mut peak) = (0.0f64, 0.0f64);
    for t in 0..4096 {
        for c in 0..64 {
            let a = got[t * 64 + c];
            let b = dump[c * 4096 + t] as f64;
            mae = mae.max((a - b).abs());
            peak = peak.max(b.abs());
        }
    }
    println!("memory sine-pos: max_abs_err {mae:.3e} peak {peak:.3e} rel {:.2e}", mae / peak);
    println!("{}", if mae < 1e-5 { "PASS — Rust memory sine-pos matches the real model." } else { "DRIFT." });
}
