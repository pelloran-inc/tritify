//! Write a renamed copy of a decoder checkpoint and the specs that serve it, to show that a model is data,
//! not code. The copy changes three things and leaves every weight byte as it was:
//! - GGUF-style tensor names (`blk.{i}.attn_q.weight` …): a header-only rewrite of each `.safetensors` file
//! - renamed config keys (`n_embd` …)
//! - an architecture id no handler knows (`KtRenamedLM`)
//!
//! Three specs are written next to it:
//! - `renamed.spec.json` maps everything back; it must reproduce the original's hash
//! - `plant_drop.spec.json` lacks the `ffn_down` rule; the build must fail and name the tensor
//! - `plant_swap.spec.json` swaps `ffn_gate` ↔ `ffn_up`; the shapes match, so it builds, and only the hash shows it
//!
//!   cargo run --release -p kt-convert --example spec_rename -- <decoder_dir> <out_dir>

use kt_convert::spec::{ModelSpec, Rename, SourceDecl};
use serde_json::{json, Map, Value};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

/// canonical (HF) → GGUF-style, as llama.cpp names a Llama/Qwen checkpoint.
const TENSORS: [(&str, &str); 15] = [
    ("model.embed_tokens.weight", "token_embd.weight"),
    ("model.norm.weight", "output_norm.weight"),
    ("lm_head.weight", "output.weight"),
    ("model.layers.{i}.input_layernorm.weight", "blk.{i}.attn_norm.weight"),
    ("model.layers.{i}.self_attn.q_proj.weight", "blk.{i}.attn_q.weight"),
    ("model.layers.{i}.self_attn.q_proj.bias", "blk.{i}.attn_q.bias"),
    ("model.layers.{i}.self_attn.k_proj.weight", "blk.{i}.attn_k.weight"),
    ("model.layers.{i}.self_attn.k_proj.bias", "blk.{i}.attn_k.bias"),
    ("model.layers.{i}.self_attn.v_proj.weight", "blk.{i}.attn_v.weight"),
    ("model.layers.{i}.self_attn.v_proj.bias", "blk.{i}.attn_v.bias"),
    ("model.layers.{i}.self_attn.o_proj.weight", "blk.{i}.attn_output.weight"),
    ("model.layers.{i}.post_attention_layernorm.weight", "blk.{i}.ffn_norm.weight"),
    ("model.layers.{i}.mlp.gate_proj.weight", "blk.{i}.ffn_gate.weight"),
    ("model.layers.{i}.mlp.up_proj.weight", "blk.{i}.ffn_up.weight"),
    ("model.layers.{i}.mlp.down_proj.weight", "blk.{i}.ffn_down.weight"),
];
const KEYS: [(&str, &str); 9] = [
    ("hidden_size", "n_embd"),
    ("num_hidden_layers", "n_layer"),
    ("num_attention_heads", "n_head"),
    ("num_key_value_heads", "n_head_kv"),
    ("intermediate_size", "n_ff"),
    ("vocab_size", "n_vocab"),
    ("rms_norm_eps", "norm_eps"),
    ("rope_theta", "rope_freq_base"),
    ("max_position_embeddings", "n_ctx_train"),
];

/// canonical → GGUF-style for one name, through the same patterns the spec uses (the `{i}` is the layer index).
fn gguf_name(canonical: &str) -> String {
    for (c, g) in TENSORS {
        let (cl, cr) = c.split_once("{i}").unwrap_or((c, ""));
        if c == canonical {
            return g.to_string();
        }
        if let Some(rest) = canonical.strip_prefix(cl) {
            let n = rest.bytes().take_while(u8::is_ascii_digit).count();
            if n > 0 && &rest[n..] == cr {
                return g.replace("{i}", &rest[..n]);
            }
        }
    }
    panic!("no GGUF name for tensor '{canonical}': extend TENSORS")
}

/// Rewrite one `.safetensors` file's header names; the data section is copied byte for byte (offsets are relative).
fn rename_file(src: &Path, dst: &Path) {
    let mut f = fs::File::open(src).expect("open shard");
    let mut len = [0u8; 8];
    f.read_exact(&mut len).unwrap();
    let mut hdr = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut hdr).unwrap();
    let hdr: Map<String, Value> = serde_json::from_slice(&hdr).expect("header JSON");
    let renamed: Map<String, Value> =
        hdr.into_iter().map(|(k, v)| if k == "__metadata__" { (k, v) } else { (gguf_name(&k), v) }).collect();
    let h = serde_json::to_vec(&Value::Object(renamed)).unwrap();
    let mut out = fs::File::create(dst).expect("create shard");
    out.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
    out.write_all(&h).unwrap();
    std::io::copy(&mut f, &mut out).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        eprintln!("usage: spec_rename <decoder_dir> <out_dir>");
        std::process::exit(2);
    }
    let (src, out) = (Path::new(&a[1]), Path::new(&a[2]));
    fs::create_dir_all(out).unwrap();
    let out = out.canonicalize().unwrap();

    // weights: every .safetensors file, header renamed (an index file's weight_map is renamed the same way)
    for e in fs::read_dir(src).unwrap().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name.ends_with(".safetensors") {
            rename_file(&e.path(), &out.join(&name));
        } else if name == "model.safetensors.index.json" {
            let mut idx: Value = serde_json::from_str(&fs::read_to_string(e.path()).unwrap()).unwrap();
            let map = idx["weight_map"].as_object().unwrap().iter().map(|(k, v)| (gguf_name(k), v.clone())).collect();
            idx["weight_map"] = Value::Object(map);
            fs::write(out.join(&name), idx.to_string()).unwrap();
        } else if name.starts_with("tokenizer") || name == "generation_config.json" {
            fs::copy(e.path(), out.join(&name)).unwrap();
        }
    }
    // config: renamed keys, an architecture no handler knows
    let mut cfg: Map<String, Value> =
        serde_json::from_str(&fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
    let arch = cfg["architectures"][0].as_str().unwrap_or("Qwen2ForCausalLM").to_string();
    for (c, g) in KEYS {
        if let Some(v) = cfg.remove(c) {
            cfg.insert(g.into(), v);
        }
    }
    cfg.insert("architectures".into(), json!(["KtRenamedLM"]));
    fs::write(out.join("config.json"), Value::Object(cfg).to_string()).unwrap();

    // the specs
    let spec = ModelSpec {
        id: "renamed-qwen".into(),
        group: "decoder".into(),
        arch: Some(arch),
        source: SourceDecl { format: "safetensors".into(), path: out.to_string_lossy().into() },
        config: KEYS.iter().map(|(c, g)| (c.to_string(), json!(g))).collect(),
        weights: TENSORS.iter().map(|(c, g)| Rename { from: g.to_string(), to: c.to_string() }).collect(),
        eos_token_ids: None,
        chat_template: Some("chatml".into()),
    };
    let mut drop = spec.clone();
    drop.weights.retain(|r| !r.from.contains("ffn_down"));
    let mut swap = spec.clone();
    for r in swap.weights.iter_mut() {
        if r.from.contains("ffn_gate") {
            r.to = "model.layers.{i}.mlp.up_proj.weight".into();
        } else if r.from.contains("ffn_up") {
            r.to = "model.layers.{i}.mlp.gate_proj.weight".into();
        }
    }
    for (file, s) in [("renamed.spec.json", &spec), ("plant_drop.spec.json", &drop), ("plant_swap.spec.json", &swap)] {
        s.validate().expect("valid spec");
        fs::write(out.join(file), s.to_json()).unwrap();
    }
    println!("renamed copy and specs written to {}", out.display());
}
