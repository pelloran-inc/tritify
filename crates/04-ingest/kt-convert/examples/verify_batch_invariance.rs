//! Batch / schedule invariance check for the decoder LM — does a token's logits depend on the *schedule*
//! (full-sequence prefill vs. incremental / prefix decode)?
//!
//! tritify has no KV cache: `forward_int(tokens)` is a whole-sequence prefill. "Decode step i" is therefore
//! `forward_int(T[0..=i])` and its last row is position i's logits as produced incrementally. A correct causal
//! model gives the *same* logits at position i whether or not later tokens are present (position i depends only on
//! 0..i). This harness measures whether tritify's INTEGER path has that invariance, with the float reference as a
//! causal-exact control.
//!
//!   cargo run --release -p kt-convert --example verify_batch_invariance -- <model_dir> [N] [seed]

use kt_convert::HfModelSource;
use kt_core::verify::hash_int;
use kt_core::{set_max_threads, CausalLM};
use kt_model_decoder::DecoderArch;

fn argmax(row: &[f32]) -> usize {
    let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: verify_batch_invariance <model_dir> [N] [seed]");
        std::process::exit(2);
    }
    let dir = &args[1];
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(42);

    let src = HfModelSource::from_dir(dir).unwrap_or_else(|e| {
        eprintln!("load error: {e}");
        std::process::exit(1);
    });
    eprintln!("building exact-integer decoder from {dir} …");
    let model = DecoderArch::build_decoder(&src).unwrap_or_else(|e| {
        eprintln!("build error: {e}");
        std::process::exit(1);
    });
    let vocab = model.vocab();

    // A fixed pseudo-random token sequence (seeded LCG), spanning the vocab for magnitude diversity — NOT an
    // arange, so no single position dominates the per-tensor max by construction.
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let tokens: Vec<usize> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as usize) % vocab.max(1)
        })
        .collect();

    // Threads are bit-identical to single-thread (established); use auto for speed.
    set_max_threads(0);

    eprintln!("full-sequence prefill (N={n}) …");
    let full = model.forward_int(&tokens);
    let full_hash = hash_int(&full);
    let full_f = full.to_f32();
    let full_ref = model.forward_ref(&tokens);

    // Per-position comparison: full[i] vs decode-step-i = forward_int(T[0..=i]).last_row.
    let (mut n_exact, mut n_argmax, mut maxdiff) = (0usize, 0usize, 0.0f64);
    let (mut n_exact_ref, mut n_argmax_ref, mut maxdiff_ref) = (0usize, 0usize, 0.0f64);
    let mut per_pos_match: Vec<bool> = vec![false; n]; // integer argmax match
    let mut checkpoints: Vec<(usize, String)> = Vec::new();

    for i in 0..n {
        eprint!("\r  decode step {}/{n}      ", i + 1);
        let pref = &tokens[0..=i];

        // integer schedule
        let pint = model.forward_int(pref);
        let pf = pint.to_f32();
        let last = &pf[i * vocab..(i + 1) * vocab];
        let fullrow = &full_f[i * vocab..(i + 1) * vocab];
        let exact = last == fullrow;
        let am = argmax(last) == argmax(fullrow);
        let d = last.iter().zip(fullrow).fold(0.0f64, |a, (&x, &y)| a.max((x as f64 - y as f64).abs()));
        if exact {
            n_exact += 1;
        }
        if am {
            n_argmax += 1;
        }
        per_pos_match[i] = am;
        maxdiff = maxdiff.max(d);

        // float control (causal-exact reference)
        let prref = model.forward_ref(pref);
        let lastr = &prref[i * vocab..(i + 1) * vocab];
        let fullrr = &full_ref[i * vocab..(i + 1) * vocab];
        if lastr == fullrr {
            n_exact_ref += 1;
        }
        if argmax(lastr) == argmax(fullrr) {
            n_argmax_ref += 1;
        }
        maxdiff_ref =
            maxdiff_ref.max(lastr.iter().zip(fullrr).fold(0.0f64, |a, (&x, &y)| a.max((x as f64 - y as f64).abs())));

        if i + 1 == n / 4 || i + 1 == n / 2 || i + 1 == (3 * n) / 4 || i + 1 == n {
            checkpoints.push((i + 1, hash_int(&pint)));
        }
    }
    eprintln!();

    // Barry-style split: prefill K = N/2, decode the rest one-by-one. Stats over the decoded tail i in K..N.
    let k = n / 2;
    let tail_argmax = (k..n).filter(|&i| per_pos_match[i]).count();
    let tail = n - k;

    println!("tritify — batch / schedule invariance (prefill vs incremental decode)");
    println!("  model        : {dir}");
    println!("  N / vocab    : {n} / {vocab}   seed {seed}");
    println!("  full hash    : {full_hash}");
    println!("  --- INTEGER path: full prefill  vs  decode-step forward_int(T[0..=i]).last ---");
    println!("  exact-equal  : {n_exact}/{n} positions bit-identical real logits");
    println!("  argmax-match : {n_argmax}/{n}  ({:.4})", n_argmax as f64 / n as f64);
    println!("  max_abs_diff : {maxdiff:.3e}");
    println!(
        "  split(K={k}) : decoded-tail argmax-match {tail_argmax}/{tail}  ({:.4})",
        tail_argmax as f64 / tail.max(1) as f64
    );
    println!("  --- FLOAT reference control (causal-exact; expected invariant) ---");
    println!("  exact-equal  : {n_exact_ref}/{n}");
    println!("  argmax-match : {n_argmax_ref}/{n}  ({:.4})", n_argmax_ref as f64 / n as f64);
    println!("  max_abs_diff : {maxdiff_ref:.3e}");
    println!("  --- prefix checkpoints (reproducing hash_int of forward_int(T[0..len])) ---");
    for (len, h) in &checkpoints {
        println!("    len {len:>3} : {h}");
    }
    let invariant = n_exact == n;
    println!(
        "  VERDICT      : integer path is {} to the prefill/decode schedule",
        if invariant { "INVARIANT" } else { "NOT invariant" }
    );
}
