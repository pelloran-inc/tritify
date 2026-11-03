//! One fixed chat request and one fixed embedding input, run under every compute backend this host
//! offers. Each backend must produce the same tokens, the same logits hash and the same embedding hash, and a repeat
//! must too.
//!
//! `--gpu-variants` adds the per-call GPU, for a speed comparison. A CUDA device joins through
//! `kt-runtime` when the build has the `cuda` feature, and `--cuda-emu` adds the digit backend's host executor.
//!
//! `--plant` adds three planted partitions of the CPU linear layer, at the engine's own thread plan; each must differ.
//! `--plant-genuine` counts them as genuine backends, so the comparison must fail (its own firing control).
//!
//! The program compares every genuine backend with the first and exits 1 with `VERDICT: MISMATCH` on any difference;
//! otherwise it ends with `VERDICT: all <n> backends identical`.
//!
//!   cargo run --release -p kt-convert --example generate_backends [--features kt-runtime/cuda] -- <model_dir>
//!     [max_tokens=32] [--gpu-variants] [--cuda-emu] [--plant | --plant-genuine]

use kt_convert::text::{Message, Tokenizer};
use kt_convert::{load_text_model, HfModelSource};
use kt_core::backend::{install_backend, reset_backend, LinearBackend, Portable};
use kt_core::generate::{embed_mean, generate_greedy};
use kt_core::verify::hash_int;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!(
            "usage: generate_backends <model_dir> [max_tokens=32] [--gpu-variants] [--cuda-emu] [--plant | --plant-genuine]"
        );
        std::process::exit(2);
    }
    let max_new: usize = a.get(2).filter(|s| !s.starts_with("--")).and_then(|s| s.parse().ok()).unwrap_or(32);
    let src = HfModelSource::from_dir(&a[1]).expect("load checkpoint");
    let model = load_text_model(&src).expect("build");
    let tok = Tokenizer::from_dir(&a[1]).expect("tokenizer");
    let chat = [
        Message { role: "system".into(), content: "You are a concise assistant.".into() },
        Message { role: "user".into(), content: "What is the capital of France? Answer in one sentence.".into() },
    ];
    let prompt = tok.encode(&tok.chat_prompt(&chat), false).expect("encode");
    let emb_in: Vec<usize> =
        tok.encode("Exact integers, the same bits everywhere.", false).unwrap().iter().map(|&t| t as usize).collect();
    println!("generate_backends · {} · prompt {} tokens · eos {:?} · max {max_new}", a[1], prompt.len(), tok.eos_ids());

    let host = kt_runtime::detect();
    // (backend, planted): every genuine backend must agree with the first; a planted one must not
    let mut backends: Vec<(Arc<dyn LinearBackend>, bool)> = vec![(Arc::new(Portable), false)];
    backends.extend(kt_runtime::available_backends(&host).into_iter().map(|b| (b, false)));
    if a.iter().any(|s| s == "--gpu-variants") {
        backends.extend(
            kt_gpu::Gpu::open(None)
                .map(|g| (Arc::new(kt_gpu::GpuLinear::per_call(g)) as Arc<dyn LinearBackend>, false)),
        );
    }
    if a.iter().any(|s| s == "--cuda-emu") {
        backends.push((Arc::new(kt_cuda::emulator()), false));
    }
    // `--plant` adds three planted partitions, which must differ. `--plant-genuine` adds them as if they were
    // genuine backends, so the comparison must report a MISMATCH: the firing control of the comparison itself.
    let as_genuine = a.iter().any(|s| s == "--plant-genuine");
    if as_genuine || a.iter().any(|s| s == "--plant") {
        use kt_runtime::planted::{BadPartition, PlantedPartition};
        for bad in [BadPartition::FloorOffsets, BadPartition::DropLastRun, BadPartition::SwappedIndex] {
            backends.push((Arc::new(PlantedPartition { bad, threads: None }), !as_genuine));
        }
    }
    backends.push((Arc::new(Portable), false)); // the repeat
    let (mut first, mut genuine, mut failures) = (None, 0usize, Vec::new());
    for (b, planted) in backends {
        let name = b.name();
        install_backend(b);
        let t = Instant::now();
        let g = generate_greedy(&model, &prompt, max_new, tok.eos_ids(), |_| {});
        let ms = t.elapsed().as_millis();
        let e = hash_int(&embed_mean(&model, &emb_in));
        let text = tok.decode(&g.tokens).unwrap().replace('\n', "\\n");
        println!(
            "  {name:<18} tokens {:>3} {:?}  logits_hash {}  embedding {}  {ms} ms\n  {:<18} text: {text}",
            g.tokens.len(),
            g.finish,
            g.logits_hash,
            e,
            ""
        );
        let result = (g.tokens.clone(), g.logits_hash.clone(), e);
        let reference = first.get_or_insert_with(|| result.clone());
        let same = *reference == result;
        if planted && same {
            println!("  MISMATCH: the planted backend {name} gives the reference's result; the plant is blind");
            failures.push(name.to_string());
        } else if !planted {
            genuine += 1;
            if !same {
                println!("  MISMATCH: {name} differs from the first backend in its tokens, logits or embedding");
                failures.push(name.to_string());
            }
        }
    }
    reset_backend();
    println!("  digit-backend fallbacks to portable: {}", kt_cuda::fallbacks());
    if failures.is_empty() {
        println!("VERDICT: all {genuine} backends identical");
    } else {
        println!("VERDICT: MISMATCH ({})", failures.join(", "));
        std::process::exit(1);
    }
}
