//! Verify the Rust-computed positional tables (bicubic Hiera pos-embed + Fourier image-wide pos) against the
//! float `Sam2Model` dumps — so the demo can compute them itself instead of loading a precomputed dump.
//! Usage:  cargo run --release --example verify_pos -- <sam_ref_dir>

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use kt_model_vision::{hiera_pos_embed, image_wide_pos, WeightStore};

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
fn report(label: &str, got: &[f64], rf: &[f32]) {
    assert_eq!(got.len(), rf.len(), "{label}: size {} vs {}", got.len(), rf.len());
    let (mut mae, mut peak) = (0.0f64, 0.0f64);
    for (&a, &b) in got.iter().zip(rf.iter()) {
        mae = mae.max((a - b as f64).abs());
        peak = peak.max((b as f64).abs());
    }
    println!(
        "  {label:22}  max_abs_err {mae:.3e}  peak {peak:.3e}  rel {:.2e}",
        if peak > 0.0 { mae / peak } else { 0.0 }
    );
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: verify_pos <sam_ref_dir>");
    let dir = Path::new(&dir);
    let wdir = dir.join("weights");
    let mut store = HashMap::new();
    for (name, shape) in read_manifest(&wdir.join("manifest.txt")) {
        let vals = read_f32(&wdir.join(format!("{name}.f32")));
        store.insert(name, (vals, shape));
    }
    let store = WeightStore(store);

    // Hiera pos-embed: my tokens [256*256,112] vs dump [1,256,256,112] (same token-major, channel-inner layout).
    let hp = hiera_pos_embed(&store, 256, 112);
    report("hiera_pos_embed", &hp, &read_f32(&dir.join("hiera_pos_embed.f32")));

    // Image-wide pos: my tokens [64*64,256] (token=r*64+c, channel) vs dump [1,256,64,64] (channel,r,c).
    let iw = image_wide_pos(&store, 64);
    let dref = read_f32(&dir.join("image_pos_embed.f32")); // [256,64,64]
    let mut dref_tok = vec![0.0f32; 64 * 64 * 256];
    for ch in 0..256 {
        for t in 0..64 * 64 {
            dref_tok[t * 256 + ch] = dref[ch * 64 * 64 + t];
        }
    }
    report("image_wide_pos", &iw, &dref_tok);

    println!("\n(small errors = quantization/interpolation float noise; these feed the already-verified pipeline)");
}
