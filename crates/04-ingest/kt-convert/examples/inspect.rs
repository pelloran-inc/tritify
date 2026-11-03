//! Inspect a HF model dir (single-file OR sharded): print its arch ids + the shape/first/last of named tensors.
//!   cargo run --release -p kt-convert --example inspect -- <model_dir> [tensor_name ...]

use kt_convert::HfModelSource;
use kt_core::plugin::ModelSource;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: inspect <model_dir> [tensor ...]");
        std::process::exit(2);
    }
    let src = HfModelSource::from_dir(&a[1]).unwrap_or_else(|e| {
        eprintln!("load: {e}");
        std::process::exit(1);
    });
    println!("arch = {:?}", src.hf_arch_ids());
    println!("tensors = {}", src.tensor_names().len());
    for name in &a[2..] {
        match src.tensor(name) {
            Some((t, shape)) => println!(
                "  {name}: shape {shape:?}  n={}  first={:.5}  last={:.5}",
                t.len(),
                t.first().copied().unwrap_or(0.0),
                t.last().copied().unwrap_or(0.0)
            ),
            None => println!("  {name}: MISSING"),
        }
    }
}
