//! KV-cache incremental decode — correctness. Decodes a fixed token sequence one token at a time
//! against a growing KV cache (O(cache)/step), and checks:
//!   determinism     — the decode logits stream is bit-identical run-to-run, and serial == multithread.
//!   distributed     — a layer-split decode (cache sharded across workers, activation over the wire) is identical.
//!   diagnostic      — decode-vs-prefill argmax agreement (expected high, not 1.0: the dynamic-quant non-invariance).
//!
//!   cargo run --release -p kt-convert --example verify_kv_decode -- <model_dir> [seq_len]

use kt_convert::HfModelSource;
use kt_core::verify::hash_int;
use kt_core::{set_max_threads, CausalLM, IntTensor};
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

/// Single-machine decode of `tokens`: returns (concatenated logit codes, a reproducing hash, per-position argmax).
fn decode_single(model: &kt_core::DecoderModel, tokens: &[usize], vocab: usize) -> (Vec<i32>, String, Vec<usize>) {
    let mut cache = model.new_cache();
    let mut stream = Vec::with_capacity(tokens.len() * vocab);
    let mut am = Vec::with_capacity(tokens.len());
    let mut scale0 = None;
    for (pos, &t) in tokens.iter().enumerate() {
        let li = model.decode_step(t, &mut cache, pos);
        if scale0.is_none() {
            scale0 = Some(li.scale);
        }
        am.push(argmax(&li.to_f32()));
        stream.extend_from_slice(&li.data);
    }
    let h = hash_int(&IntTensor::new(stream.clone(), vec![tokens.len(), vocab], scale0.unwrap()));
    (stream, h, am)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: verify_kv_decode <model_dir> [seq_len]");
        std::process::exit(2);
    }
    let dir = &args[1];
    let seq: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);

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

    // determinism + serial == multithread
    set_max_threads(1);
    let (s1, h1, am_dec) = decode_single(&model, &tokens, vocab);
    let (_s1b, h1b, _) = decode_single(&model, &tokens, vocab);
    set_max_threads(0);
    let (s_mt, h_mt, _) = decode_single(&model, &tokens, vocab);

    // layer-split decode: worker A = blocks 0..k (cache A), worker B = blocks k..n (cache B)
    let k = n / 2;
    let mut cache_a = model.new_cache();
    cache_a.truncate(k);
    let mut cache_b = model.new_cache();
    cache_b.truncate(n - k);
    let mut s_split = Vec::with_capacity(seq * vocab);
    let mut wire_bytes = 0usize;
    for (pos, &t) in tokens.iter().enumerate() {
        let x_a = model.decode_blocks(model.embed_int(&[t]), &mut cache_a, 0, k, pos);
        let wire = x_a.to_bytes();
        wire_bytes = wire.len();
        let x_b = model.decode_blocks(IntTensor::from_bytes(&wire), &mut cache_b, k, n, pos);
        let li = model.head_int(&x_b);
        s_split.extend_from_slice(&li.data);
    }

    // diagnostic — decode vs whole-sequence prefill argmax
    let prefill = model.forward_int(&tokens).to_f32();
    let am_pre: Vec<usize> = (0..seq).map(|i| argmax(&prefill[i * vocab..(i + 1) * vocab])).collect();
    let agree = am_dec.iter().zip(&am_pre).filter(|(a, b)| a == b).count();

    println!("tritify — KV-cache incremental decode check");
    println!("  model / seq / layers : {dir} / {seq} / {n}");
    println!("  decode reproducing hash : {h1}");
    println!(
        "  determinism     : run-to-run {} | serial==MT {}",
        if h1 == h1b { "identical ✓" } else { "DIFFERS ✗" },
        if h1 == h_mt && s1 == s_mt { "identical ✓" } else { "DIFFERS ✗" }
    );
    println!(
        "  split==single   : {} (k={k}, wire {wire_bytes} B/step)",
        if s_split == s1 { "bit-identical ✓" } else { "DIFFERS ✗" }
    );
    println!(
        "  diagnostic      : decode vs prefill argmax {agree}/{seq} = {:.4} (non-invariance, expected <1)",
        agree as f64 / seq as f64
    );

    let ok = h1 == h1b && h1 == h_mt && s1 == s_mt && s_split == s1;
    println!("  → {}", if ok { "KV-cache decode is deterministic and split-reproducible." } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}
