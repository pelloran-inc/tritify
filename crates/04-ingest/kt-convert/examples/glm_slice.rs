//! Build the first `layers` layers of a MoE checkpoint (GLM-4.5-Air's slices) and record its integer run on a text
//! probe: the logits hash on every CPU variant of the host, twice each, each MoE layer's routing per token and the
//! non-degeneracy facts, and, with `--out`, the final-normed hidden state, the logits and the routes for the float
//! comparison. With `--per-layer`, also the residual stream after every layer (`layer_<i>.f32`).
//!
//!   cargo run --release -p kt-convert --example glm_slice -- <dir> <layers> <probe text> [tokens=32] [--out DIR]
//!       [--per-layer]      (needs --out)
//!       [--plant p1..p5]   (with `--features kt-convert/plants`: the planted architecture mistakes)
//!
//! The probe is the first `tokens` ids of the text in the checkpoint's own `tokenizer.json`, without special tokens.

use kt_convert::text::Tokenizer;
use kt_convert::HfModelSource;
use kt_core::backend::{install_backend, reset_backend, LinearBackend, Portable};
use kt_core::plugin::ModelSource;
use kt_core::verify::hash_int;
use kt_core::CausalLM;
use kt_model_moe::MoeArch;
use serde_json::json;
use std::sync::Arc;

/// The checkpoint with `num_hidden_layers` capped, so a slice builds without the rest of the model.
struct CappedLayers<'a> {
    inner: &'a HfModelSource,
    cap: usize,
}
impl ModelSource for CappedLayers<'_> {
    fn hf_arch_ids(&self) -> Vec<String> {
        self.inner.hf_arch_ids()
    }
    fn config_usize(&self, key: &str) -> Option<usize> {
        if key == "num_hidden_layers" {
            return Some(self.cap);
        }
        self.inner.config_usize(key)
    }
    fn config_f64(&self, key: &str) -> Option<f64> {
        self.inner.config_f64(key)
    }
    fn config_bool(&self, key: &str) -> Option<bool> {
        self.inner.config_bool(key)
    }
    fn tensor(&self, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        self.inner.tensor(name)
    }
    fn tensor_names(&self) -> Vec<String> {
        self.inner.tensor_names()
    }
}

fn die(msg: String) -> ! {
    eprintln!("glm_slice: {msg}");
    std::process::exit(1);
}

fn write_f32(path: &std::path::Path, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap_or_else(|e| die(format!("{}: {e}", path.display())));
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out = a.iter().position(|s| s == "--out").map(|i| std::path::PathBuf::from(&a[i + 1]));
    let plant = a.iter().position(|s| s == "--plant").map(|i| a[i + 1].clone());
    let per_layer = a.iter().any(|s| s == "--per-layer");
    if per_layer && out.is_none() {
        die("--per-layer needs --out DIR".into());
    }
    let pos: Vec<&String> = a.iter().skip(1).take_while(|s| !s.starts_with("--")).collect();
    if pos.len() < 3 {
        die("usage: glm_slice <dir> <layers> <probe text> [tokens=32] [--out DIR]".into());
    }
    let (dir, layers, text) = (pos[0], pos[1].parse::<usize>().unwrap_or_else(|_| die("layers".into())), pos[2]);
    let n_tok = pos.get(3).map(|s| s.parse().unwrap_or_else(|_| die("tokens".into()))).unwrap_or(32usize);

    let tok = Tokenizer::from_dir(dir).unwrap_or_else(|e| die(format!("tokenizer: {e}")));
    let all = tok.encode(text, false).unwrap_or_else(|e| die(format!("encode: {e}")));
    if all.len() < n_tok {
        die(format!("the probe text has {} tokens, fewer than {n_tok}", all.len()));
    }
    let ids: Vec<u32> = all[..n_tok].to_vec();
    let probe: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
    println!("probe: {n_tok} ids {ids:?}");
    println!("probe text (raw decode): {:?}", tok.decode_raw(&ids).unwrap_or_default());

    let src = HfModelSource::from_dir(dir).unwrap_or_else(|e| die(format!("load: {e}")));
    let total = src.config_usize("num_hidden_layers").unwrap_or_else(|| die("num_hidden_layers".into()));
    if layers == 0 || layers > total {
        die(format!("layers must be in 1..={total}"));
    }
    let t0 = std::time::Instant::now();
    let capped = CappedLayers { inner: &src, cap: layers };
    let model = match plant.as_deref() {
        None => MoeArch::build_decoder_moe(&capped),
        #[cfg(feature = "plants")]
        Some(p) => {
            let plant = kt_model_moe::Plant::parse(p).unwrap_or_else(|| die(format!("unknown plant {p}")));
            println!("PLANTED: {} ({plant:?})", plant.name());
            MoeArch::build_decoder_moe_planted(&capped, plant)
        }
        #[cfg(not(feature = "plants"))]
        Some(_) => die("--plant needs a build with --features kt-convert/plants".into()),
    }
    .unwrap_or_else(|e| die(format!("build: {e}")));
    println!("built {layers}/{total} layers of {:?} in {:.1} s", src.hf_arch_ids(), t0.elapsed().as_secs_f64());

    // the slice's integer logits over the probe on every CPU variant of this host, hashed twice each
    let host = kt_runtime::detect();
    println!(
        "host {} / {} · cpu {:?}",
        host.arch,
        host.os,
        host.cpu_variants.iter().map(|v| v.name()).collect::<Vec<_>>()
    );
    let mut backends: Vec<Arc<dyn LinearBackend>> = vec![Arc::new(Portable)];
    backends.extend(kt_runtime::available_backends(&host).into_iter().filter(|b| !b.name().starts_with("gpu:")));
    let mut variants = Vec::new();
    for b in backends {
        let name = b.name().to_string();
        install_backend(b);
        let t = std::time::Instant::now();
        let (h1, h2) = (hash_int(&model.forward_int(&probe)), hash_int(&model.forward_int(&probe)));
        println!("  {name:<18} slice hash = {h1}  det={}  {} ms (two runs)", h1 == h2, t.elapsed().as_millis());
        variants.push(json!({"variant": name, "hash": h1, "deterministic": h1 == h2}));
    }
    reset_backend();
    let logits = model.forward_int(&probe);
    let (hash, again) = (hash_int(&logits), hash_int(&model.forward_int(&probe)));
    println!("slice hash ({layers} layers, {n_tok} tokens) = {hash}  det={}", hash == again);

    // the routes and the non-degeneracy facts
    let routes = model.forward_int_routes(&probe);
    let hidden = model.hidden_int(&probe);
    let h = hidden.to_f32();
    let dim = model.dim();
    let mean = h.iter().map(|&v| v as f64).sum::<f64>() / h.len() as f64;
    let std = (h.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / h.len() as f64).sqrt();
    let rows: Vec<&[i32]> = hidden.data.chunks(dim).collect();
    let distinct_rows = (0..rows.len()).all(|i| (0..i).all(|j| rows[i] != rows[j]));
    println!("hidden: std {std:.6e}, {} rows pairwise distinct: {distinct_rows}", rows.len());
    let mut moe_layers = Vec::new();
    for (k, layer) in routes.iter().enumerate() {
        let sets: std::collections::BTreeSet<Vec<usize>> = layer
            .iter()
            .map(|r| {
                let mut s = r.experts.clone();
                s.sort_unstable();
                s
            })
            .collect();
        let union: std::collections::BTreeSet<usize> = layer.iter().flat_map(|r| r.experts.iter().copied()).collect();
        let all_weighted = layer.iter().all(|r| r.sumc > 0);
        println!(
            "MoE layer {k}: sumc > 0 for every token: {all_weighted}; {} distinct top-k sets; union of {} experts",
            sets.len(),
            union.len()
        );
        moe_layers.push(json!({
            "routed_layer": k,
            "experts": layer.iter().map(|r| r.experts.clone()).collect::<Vec<_>>(),
            "numerators": layer.iter().map(|r| r.ec.clone()).collect::<Vec<_>>(),
            "sumc": layer.iter().map(|r| r.sumc).collect::<Vec<_>>(),
            "distinct_sets": sets.len(),
            "union": union.len(),
            "all_weighted": all_weighted,
        }));
    }
    let vocab = logits.len() / n_tok;
    let argmax: Vec<usize> = logits
        .data
        .chunks(vocab)
        .map(|r| r.iter().enumerate().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(&a.0))).map(|(i, _)| i).unwrap())
        .collect();
    let argmax_varies = argmax.iter().any(|&t| t != argmax[0]);
    println!("argmax per position: {argmax:?}  varies: {argmax_varies}");

    if let Some(dir_out) = out {
        std::fs::create_dir_all(&dir_out).unwrap_or_else(|e| die(format!("{}: {e}", dir_out.display())));
        write_f32(&dir_out.join("hidden.f32"), &h);
        write_f32(&dir_out.join("logits.f32"), &logits.to_f32());
        // the residual stream after every layer, through the split forward (`blocks_int` one layer at a time)
        let split_same = per_layer.then(|| {
            let mut x = model.embed_int(&probe);
            for i in 0..model.n_layers() {
                x = model.blocks_int(x, i, i + 1);
                write_f32(&dir_out.join(format!("layer_{i}.f32")), &x.to_f32());
            }
            hash_int(&model.head_int(&x)) == hash
        });
        if let Some(same) = split_same {
            println!(
                "per-layer: {layers} residual streams written; the split forward's logits = the one-shot's: {same}"
            );
        }
        let mut meta = json!({
            "layers": layers, "tokens": n_tok, "ids": ids, "text": text, "dim": dim, "vocab": vocab,
            "hash": hash, "deterministic": hash == again, "hidden": "final-normed, [tokens, dim], f32 LE",
            "logits": "[tokens, vocab], f32 LE", "hidden_std": std, "hidden_rows_distinct": distinct_rows,
            "argmax": argmax, "argmax_varies": argmax_varies, "moe_layers": moe_layers, "variants": variants,
        });
        if let Some(same) = split_same {
            meta["per_layer"] = json!({"files": "layer_<i>.f32, the residual stream after layer i, [tokens, dim], f32 LE",
                                       "split_logits_equal_one_shot": same});
        }
        let path = dir_out.join("slice.json");
        std::fs::write(&path, serde_json::to_string_pretty(&meta).unwrap())
            .unwrap_or_else(|e| die(format!("{}: {e}", path.display())));
        println!("wrote {}", dir_out.display());
    }
}
