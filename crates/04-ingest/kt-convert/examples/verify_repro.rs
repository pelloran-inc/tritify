//! Verify determinism yourself: one model and one input, greedy, on every compute backend this host offers. Every
//! position's logits must be bit-identical on every backend. Then write a reference and check it on another machine.
//!
//! ```text
//! cargo run --release -p kt-convert --example verify_repro -- (<model_dir> | --synthetic [--seed N])
//!   [--prompt FILE] [--tokens 32] [--reference-out FILE | --check FILE]
//!   [--model-name NAME --revision REV] [--perturb POS] [--plant]
//! ```
//!
//! - **The input:** by default, one fixed chat request (the same one `generate_backends` runs, so with 32 tokens its
//!   `logits_hash` is the one `REPRODUCE.md` lists). `--prompt FILE` takes your own text instead.
//! - **The comparison:** every backend's run against the first, position by position (`kt_convert::repro`). On any
//!   difference it names the first position whose logits differ.
//! - **A reference:** `--reference-out FILE` stores the run, bound to the checkpoint's files and the input's ids.
//!   `--check FILE` (on any machine) runs every backend against it, and refuses a reference for another checkpoint or
//!   input.
//! - **The red-tests:** `--perturb POS` flips one bit of the last backend's logits at POS in its hash only, so the
//!   report must name exactly POS. `--plant` adds two deliberately wrong CPU partitions (three workers forced), each
//!   of which must be found.
//!
//! Exit 0: bit-identical everywhere (and equal to the reference). 1: a difference. 2: usage, or a refused reference.
use std::sync::Arc;

use kt_convert::repro::{self, Difference, Reference, Run};
use kt_convert::text::{Message, Tokenizer};
use kt_convert::{load_text_model, HfModelSource};
use kt_core::backend::{install_backend, reset_backend, LinearBackend, Portable};
use kt_core::model::DecoderModel;

/// What a model source gives: the model, its identity (and the per-file hashes behind it), the prompt ids, the stop
/// ids, and a label.
type Loaded = (DecoderModel, (String, Vec<(String, String)>), Vec<u32>, Vec<u32>, String);

fn arg(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned())
}

fn usage() -> ! {
    eprintln!(
        "usage: verify_repro (<model_dir> | --synthetic [--seed N]) [--prompt FILE] [--tokens 32] \
         [--reference-out FILE | --check FILE] [--model-name NAME --revision REV] [--perturb POS] [--plant]"
    );
    std::process::exit(2)
}

fn describe(d: &Difference) -> String {
    match d {
        Difference::Position { pos, a, b } => {
            format!("first differing position {pos}: {}… against {}…", &a[..16], &b[..16])
        }
        Difference::Token { index } => format!("the generated tokens differ at index {index}"),
        Difference::Length { a, b } => format!("the runs have {a} and {b} positions"),
        Difference::RunHash => "the whole-run logits_hash differs".into(),
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let synthetic = a.iter().any(|x| x == "--synthetic");
    let dir = a.get(1).filter(|s| !s.starts_with("--")).cloned();
    if synthetic == dir.is_some() {
        usage();
    }
    let max_new: usize = arg(&a, "--tokens").map_or(32, |s| s.parse().unwrap_or_else(|_| usage()));
    let perturb: Option<usize> = arg(&a, "--perturb").map(|s| s.parse().unwrap_or_else(|_| usage()));
    if arg(&a, "--reference-out").is_some() && arg(&a, "--check").is_some() {
        usage();
    }

    // the model, its identity, the prompt and the stop ids
    let (model, checkpoint, prompt, stop, label): Loaded = if synthetic {
        let seed: u64 = arg(&a, "--seed").map_or(7, |s| s.parse().unwrap_or_else(|_| usage()));
        let (m, id) = repro::synthetic_model(seed);
        if arg(&a, "--prompt").is_some() {
            eprintln!("--prompt needs a checkpoint with a tokenizer; the synthetic model takes its fixed ids");
            std::process::exit(2);
        }
        (m, (id, vec![]), vec![3, 14, 15, 9, 26, 5, 35, 8, 9, 7, 9, 3], vec![], format!("synthetic seed {seed}"))
    } else {
        let d = dir.clone().expect("a model directory");
        let src = HfModelSource::from_dir(&d).unwrap_or_else(|e| {
            eprintln!("{d}: {e}");
            std::process::exit(2)
        });
        let m = load_text_model(&src).unwrap_or_else(|e| {
            eprintln!("{d}: {e}");
            std::process::exit(2)
        });
        let tok = Tokenizer::from_dir(&d).unwrap_or_else(|e| {
            eprintln!("{d}: {e}");
            std::process::exit(2)
        });
        let prompt = match arg(&a, "--prompt") {
            Some(f) => {
                let text = std::fs::read_to_string(&f).unwrap_or_else(|e| {
                    eprintln!("--prompt {f}: {e}");
                    std::process::exit(2)
                });
                tok.encode(&text, false).expect("encode the prompt")
            }
            None => {
                // the fixed chat request of generate_backends
                let chat = [
                    Message { role: "system".into(), content: "You are a concise assistant.".into() },
                    Message {
                        role: "user".into(),
                        content: "What is the capital of France? Answer in one sentence.".into(),
                    },
                ];
                tok.encode(&tok.chat_prompt(&chat), false).expect("encode the chat")
            }
        };
        if prompt.is_empty() {
            eprintln!("the prompt is empty");
            std::process::exit(2);
        }
        let id = repro::checkpoint_identity(&d).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2)
        });
        (m, id, prompt, tok.eos_ids().to_vec(), d)
    };
    let host = format!("{} {}", std::env::consts::ARCH, std::env::consts::OS);
    println!(
        "verify_repro · {label} · checkpoint {}… · prompt {} tokens (ids {}…) · {max_new} greedy tokens · {host}",
        &checkpoint.0[..16],
        prompt.len(),
        &repro::ids_sha256(&prompt)[..16]
    );

    // a reference to check against: refused before any run if it is for another checkpoint or input
    let reference = arg(&a, "--check").map(|f| {
        let r = std::fs::read_to_string(&f).map_err(|e| e.to_string()).and_then(|t| Reference::from_json(&t));
        let r = r.unwrap_or_else(|e| {
            eprintln!("--check {f}: {e}");
            std::process::exit(2)
        });
        if let Err(why) = r.applies(&checkpoint.0, &prompt, max_new) {
            eprintln!("--check {f}: refused: {why}");
            std::process::exit(2);
        }
        println!(
            "  reference {f}: {} {} · written on {} · {} positions",
            r.model,
            r.revision,
            r.written_on,
            r.hashes.len()
        );
        r
    });

    // the backends: portable first, every backend the host offers, the planted ones if asked, and a repeat
    let mut backends: Vec<(Arc<dyn LinearBackend>, bool)> = vec![(Arc::new(Portable), false)];
    backends.extend(kt_runtime::available_backends(&kt_runtime::detect()).into_iter().map(|b| (b, false)));
    if a.iter().any(|x| x == "--plant") {
        use kt_runtime::planted::{BadPartition, PlantedPartition};
        // Three workers forced, so each plant splits every call, however small (the engine's own plan keeps small
        // calls on one worker, where a partition defect cannot show), and unevenly (a floor/ceil offset defect is
        // invisible when the outputs divide evenly). The third plant, a row/output index swap, is the identity on a
        // one-row call, and a greedy decode is one row per call, so it cannot show here and is not used.
        for bad in [BadPartition::FloorOffsets, BadPartition::DropLastRun] {
            backends.push((Arc::new(PlantedPartition { bad, threads: Some(3) }), true));
        }
    }
    backends.push((Arc::new(Portable), false)); // the repeat
    let last_genuine = backends.iter().rposition(|(_, planted)| !planted).expect("a genuine backend");

    let (mut first, mut agreed, mut failures): (Option<Run>, Vec<String>, Vec<String>) = (None, vec![], vec![]);
    for (i, (b, planted)) in backends.into_iter().enumerate() {
        let name = b.name();
        install_backend(b);
        let t = std::time::Instant::now();
        let r = repro::run(&model, &prompt, max_new, &stop, if i == last_genuine { perturb } else { None });
        let ms = t.elapsed().as_millis();
        let base = first.get_or_insert_with(|| r.clone());
        let vs_first = repro::first_difference(base, &r);
        let vs_ref = reference.as_ref().and_then(|x| x.check(&r));
        let kind = if planted {
            "planted"
        } else if name.starts_with("gpu:") {
            "gpu, whole model"
        } else {
            ""
        };
        let status = match (planted, &vs_first, &vs_ref) {
            (true, None, _) => {
                failures.push(format!("{name}: the plant is blind"));
                "MISMATCH: a planted partition gave the same bits; the check is blind".to_string()
            }
            (true, Some(d), _) => format!("found, as it must be: {}", describe(d)),
            (false, Some(d), _) => {
                failures.push(name.clone());
                format!("MISMATCH against the first backend: {}", describe(d))
            }
            (false, None, Some(d)) => {
                failures.push(name.clone());
                format!("MISMATCH against the reference: {}", describe(d))
            }
            (false, None, None) => {
                agreed.push(name.clone());
                if reference.is_some() { "bit-identical, and equal to the reference" } else { "bit-identical" }.into()
            }
        };
        println!(
            "  {name:<16}{} {} positions · {} tokens · logits_hash {} · {ms} ms · {status}",
            if kind.is_empty() { String::new() } else { format!(" ({kind})") },
            r.hashes.len(),
            r.generation.tokens.len(),
            r.generation.logits_hash,
        );
    }
    reset_backend();

    let genuine = agreed.len() + failures.iter().filter(|f| !f.contains("plant is blind")).count();
    if !failures.is_empty() {
        println!("VERDICT: MISMATCH ({})", failures.join("; "));
        std::process::exit(1);
    }
    if let Some(f) = arg(&a, "--reference-out") {
        let run = first.expect("a run");
        let r = Reference::new(
            checkpoint,
            &arg(&a, "--model-name").unwrap_or_else(|| label.clone()),
            &arg(&a, "--revision").unwrap_or_else(|| "-".into()),
            &prompt,
            max_new,
            &run,
            &host,
            agreed.clone(),
        );
        std::fs::write(&f, r.to_json()).unwrap_or_else(|e| {
            eprintln!("--reference-out {f}: {e}");
            std::process::exit(2)
        });
        println!("  wrote the reference {f} ({} positions)", r.hashes.len());
    }
    println!(
        "VERDICT: all {genuine} backends bit-identical at every position{}",
        if reference.is_some() { ", and equal to the reference" } else { "" }
    );
}
