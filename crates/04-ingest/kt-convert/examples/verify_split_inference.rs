//! Verifiable split inference — the core of the distributed engine, proven.
//!
//! A decoder-only model is split **layer-wise** across workers: worker A embeds the tokens and runs blocks
//! `0..k`, serializes the integer residual stream (`IntTensor::to_bytes`, the "wire"), and hands it to worker B,
//! which deserializes it and runs blocks `k..n` plus the LM head. Because every block keeps the *whole* tensor,
//! splitting by layers changes nothing numerically: the final integer logits are **bit-identical** to a single
//! machine's — carried by the same reproducing SHA-256. This is the property float stacks (that split the same
//! way) cannot offer: distributed, but provably the same computation.
//!
//!   cargo run --release -p kt-convert --example verify_split_inference -- <model_dir> [seq_len]

use kt_convert::HfModelSource;
use kt_core::verify::hash_int;
use kt_core::{CausalLM, IntTensor};
use kt_model_decoder::DecoderArch;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: verify_split_inference <model_dir> [seq_len]");
        std::process::exit(2);
    }
    let dir = &args[1];
    let seq: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);

    let src = HfModelSource::from_dir(dir).unwrap_or_else(|e| {
        eprintln!("load: {e}");
        std::process::exit(1);
    });
    eprintln!("building exact-integer decoder from {dir} …");
    let model = DecoderArch::build_decoder(&src).unwrap_or_else(|e| {
        eprintln!("build: {e}");
        std::process::exit(1);
    });
    let vocab = model.vocab();
    let n = model.n_layers();
    let tokens: Vec<usize> = (0..seq).map(|i| i % vocab.max(1)).collect();

    // --- single machine: the reference ---
    let full = model.forward_int(&tokens);
    let ref_hash = hash_int(&full);

    println!("tritify — verifiable split inference");
    println!("  model      : {dir}");
    println!("  seq / layers: {seq} / {n}");
    println!("  single-machine reproducing hash: {ref_hash}");
    println!("  --- 2-worker layer splits (activation serialized over the 'wire' between A and B) ---");

    let mut all_ok = true;
    for &k in &[1usize, n / 4, n / 2, (3 * n) / 4, n.saturating_sub(1)] {
        if k == 0 || k >= n {
            continue;
        }
        // worker A: embed + blocks[0..k], then serialize the integer residual stream
        let x_a = model.blocks_int(model.embed_int(&tokens), 0, k);
        let wire = x_a.to_bytes();
        // worker B: deserialize (exact bytes) + blocks[k..n] + head
        let x_b_in = IntTensor::from_bytes(&wire);
        let out = model.head_int(&model.blocks_int(x_b_in, k, n));
        let split_hash = hash_int(&out);
        let ok = split_hash == ref_hash && out.data == full.data;
        all_ok &= ok;
        println!(
            "  split at layer {k:>3} | wire {:>8} B | {}",
            wire.len(),
            if ok { "bit-identical to single machine ✓" } else { "DIFFERS ✗" }
        );
    }

    // --- a 3-worker split, to show it composes ---
    if n >= 3 {
        let (k1, k2) = (n / 3, (2 * n) / 3);
        let w1 = model.blocks_int(model.embed_int(&tokens), 0, k1).to_bytes();
        let w2 = model.blocks_int(IntTensor::from_bytes(&w1), k1, k2).to_bytes();
        let out3 = model.head_int(&model.blocks_int(IntTensor::from_bytes(&w2), k2, n));
        let ok3 = hash_int(&out3) == ref_hash;
        all_ok &= ok3;
        println!("  3-worker split at {k1},{k2} | {}", if ok3 { "bit-identical ✓" } else { "DIFFERS ✗" });
    }

    println!(
        "  → {}",
        if all_ok {
            "every split reproduces the single-machine hash exactly — distributed == one machine, provably."
        } else {
            "FAIL: a split diverged"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
}
