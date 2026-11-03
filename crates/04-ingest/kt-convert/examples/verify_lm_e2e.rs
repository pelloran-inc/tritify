//! End-to-end LM check at a real sequence length — the counterpart of the SAM `sam2_segment` demo for a
//! decoder-only language model. It builds the concrete exact-integer [`kt_core::DecoderModel`] from a
//! Hugging Face checkpoint, runs its integer forward on a probe of `seq` tokens **twice — single-threaded then
//! multithreaded** (via the [`kt_core::set_max_threads`] knob), checks the two outputs are bit-identical,
//! times both, verifies argmax-fidelity against the float reference, and prints the reproducing hash.
//!
//! Unlike the built-in 8-token self-check, a few-hundred-token sequence pushes the attention past the multithread
//! gate, so this is where the **decoder** attention parallelism actually runs (SAM exercises only `SdpaCore`).
//!
//!   cargo run --release -p kt-convert --example verify_lm_e2e -- <model_dir> [seq_len]
//!
//! The thread count is configurable (`TRITIFY_THREADS=<n>` or `set_max_threads`); this demo drives it directly.

use kt_convert::HfModelSource;
use kt_core::{set_max_threads, CausalLM};
use kt_model_decoder::DecoderArch;
use std::time::Instant;

fn argmax_rows(logits: &[f32], vocab: usize) -> Vec<usize> {
    logits
        .chunks(vocab)
        .map(|row| {
            let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            bi
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: verify_lm_e2e <model_dir> [seq_len]");
        std::process::exit(2);
    }
    let dir = &args[1];
    let seq: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(512);

    let src = HfModelSource::from_dir(dir).unwrap_or_else(|e| {
        eprintln!("load error: {e}");
        std::process::exit(1);
    });

    eprintln!("building exact-integer decoder from {dir} …");
    let t = Instant::now();
    let model = DecoderArch::build_decoder(&src).unwrap_or_else(|e| {
        eprintln!("build error: {e}");
        std::process::exit(1);
    });
    let build_ms = t.elapsed().as_secs_f64() * 1000.0;
    let vocab = model.vocab();
    let tokens: Vec<usize> = (0..seq).map(|i| i % vocab.max(1)).collect();
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());

    // --- A/B on the identical model: single-threaded, then multithreaded (auto). ---
    eprintln!("integer forward (seq={seq}) — single-threaded …");
    set_max_threads(1);
    let t = Instant::now();
    let li_serial = model.forward_int(&tokens);
    let serial_ms = t.elapsed().as_secs_f64() * 1000.0;

    eprintln!("integer forward (seq={seq}) — multithread (auto) …");
    set_max_threads(0);
    let t = Instant::now();
    let li_mt = model.forward_int(&tokens);
    let mt_ms = t.elapsed().as_secs_f64() * 1000.0;

    let bit_identical = li_serial.data == li_mt.data;
    let hash = model.logits_hash(&tokens);

    // --- argmax fidelity vs the float reference (computed once). ---
    eprintln!("float reference …");
    let lr = model.forward_ref(&tokens);
    let lif = li_mt.to_f32();
    let (ai, ar) = (argmax_rows(&lif, vocab), argmax_rows(&lr, vocab));
    let agree = ai.iter().zip(&ar).filter(|(a, b)| a == b).count();
    let max_abs_err = lif.iter().zip(&lr).fold(0.0f64, |a, (&x, &y)| a.max((x as f64 - y as f64).abs()));

    println!("tritify — LM end-to-end check");
    println!("  model       : {dir}");
    println!("  seq / vocab : {seq} / {vocab}");
    println!("  cores       : {cores}");
    println!("  build       : {build_ms:.0} ms");
    println!(
        "  forward_int : single-thread {serial_ms:.1} ms   multithread {mt_ms:.1} ms   → {:.2}x",
        serial_ms / mt_ms
    );
    println!("  bit-identical (serial == multithread): {bit_identical}");
    println!("  argmax      : {agree}/{seq} == float  ({:.4})", agree as f64 / seq as f64);
    println!("  max_abs_err : {max_abs_err:.3e}");
    println!("  repro hash  : {hash}");
    if !bit_identical {
        eprintln!("FAIL: multithread output differs from single-thread");
        std::process::exit(1);
    }
    println!("  → threads change speed, never a single output bit. Re-run anywhere: same hash.");
}
