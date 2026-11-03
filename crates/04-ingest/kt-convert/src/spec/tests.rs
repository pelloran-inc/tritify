//! Model specs on a tiny synthetic checkpoint written to disk. Covered:
//! - an auto-detected spec is neutral
//! - a renamed copy plus one spec builds the same bits
//! - a dropped rule fails closed, and a swapped rule is caught by the hash
//! - the registry, and every refusal

use super::*;
use crate::convert;
use serde_json::{json, Map};
use std::path::PathBuf;

/// The canonical (HF) name → GGUF-style name used by the renamed copy, and the config keys renamed with it.
const RENAMES: [(&str, &str); 13] = [
    ("model.embed_tokens.weight", "token_embd.weight"),
    ("model.norm.weight", "output_norm.weight"),
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
];
const MORE: [(&str, &str); 1] = [("model.layers.{i}.mlp.down_proj.weight", "blk.{i}.ffn_down.weight")];
const KEYS: [(&str, &str); 6] = [
    ("hidden_size", "n_embd"),
    ("num_hidden_layers", "n_layer"),
    ("num_attention_heads", "n_head"),
    ("num_key_value_heads", "n_head_kv"),
    ("intermediate_size", "n_ff"),
    ("vocab_size", "n_vocab"),
];

/// A fresh directory per call: the tests run in parallel and must not share files.
fn dir(tag: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("kt-spec-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn safetensors(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut hdr = Map::new();
    let mut off = 0u64;
    for (name, shape, data) in tensors {
        let end = off + 4 * data.len() as u64;
        hdr.insert(name.clone(), json!({"dtype": "F32", "shape": shape, "data_offsets": [off, end]}));
        off = end;
    }
    let h = serde_json::to_vec(&Value::Object(hdr)).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    for (_, _, d) in tensors {
        out.extend(d.iter().flat_map(|x| x.to_le_bytes()));
    }
    out
}

/// A 2-layer Qwen2-shaped checkpoint; `renamed` writes GGUF-style tensor names, renamed config keys and an unknown
/// architecture id instead of the canonical ones.
fn write_checkpoint(renamed: bool) -> PathBuf {
    let (dim, nh, nkv, hd, hidden, vocab, layers) = (16usize, 4usize, 2usize, 4usize, 32usize, 20usize, 2usize);
    let (qd, kvd) = (nh * hd, nkv * hd);
    let mut cfg = Map::new();
    cfg.insert("architectures".into(), json!([if renamed { "KtRenamedLM" } else { "Qwen2ForCausalLM" }]));
    for (k, v) in [
        ("hidden_size", dim),
        ("num_hidden_layers", layers),
        ("num_attention_heads", nh),
        ("num_key_value_heads", nkv),
        ("intermediate_size", hidden),
        ("vocab_size", vocab),
    ] {
        let key = if renamed { KEYS.iter().find(|(c, _)| *c == k).unwrap().1 } else { k };
        cfg.insert(key.into(), json!(v));
    }
    cfg.insert("head_dim".into(), json!(hd));
    cfg.insert("rms_norm_eps".into(), json!(1e-6));
    cfg.insert("rope_theta".into(), json!(10000.0));
    cfg.insert("max_position_embeddings".into(), json!(64));
    let mut s = 0x5032_0d00u64;
    let mut t = |name: String, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let v = (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) as f32
            })
            .collect();
        (name, shape, v)
    };
    let mut tensors =
        vec![t("model.embed_tokens.weight".into(), vec![vocab, dim]), t("model.norm.weight".into(), vec![dim])];
    for i in 0..layers {
        let p = format!("model.layers.{i}");
        for (n, sh) in [
            ("input_layernorm.weight", vec![dim]),
            ("self_attn.q_proj.weight", vec![qd, dim]),
            ("self_attn.q_proj.bias", vec![qd]),
            ("self_attn.k_proj.weight", vec![kvd, dim]),
            ("self_attn.k_proj.bias", vec![kvd]),
            ("self_attn.v_proj.weight", vec![kvd, dim]),
            ("self_attn.v_proj.bias", vec![kvd]),
            ("self_attn.o_proj.weight", vec![dim, qd]),
            ("post_attention_layernorm.weight", vec![dim]),
            ("mlp.gate_proj.weight", vec![hidden, dim]),
            ("mlp.up_proj.weight", vec![hidden, dim]),
            ("mlp.down_proj.weight", vec![dim, hidden]),
        ] {
            tensors.push(t(format!("{p}.{n}"), sh));
        }
    }
    if renamed {
        for (name, _, _) in tensors.iter_mut() {
            *name = rename_out(name);
        }
    }
    let d = dir(if renamed { "renamed" } else { "canonical" });
    std::fs::write(d.join("config.json"), Value::Object(cfg).to_string()).unwrap();
    std::fs::write(d.join("model.safetensors"), safetensors(&tensors)).unwrap();
    d
}

/// canonical → GGUF-style (the writer's direction).
fn rename_out(canonical: &str) -> String {
    for (c, g) in RENAMES.iter().chain(MORE.iter()) {
        if let Some(caps) = Pattern::parse(c).unwrap().matches(canonical) {
            return Pattern::parse(g).unwrap().fill(&caps);
        }
    }
    canonical.to_string()
}

/// The spec that maps the renamed copy back (GGUF-style → canonical).
fn renamed_spec(path: &Path) -> ModelSpec {
    ModelSpec {
        id: "renamed".into(),
        group: "decoder".into(),
        arch: Some("Qwen2ForCausalLM".into()),
        source: SourceDecl { format: "safetensors".into(), path: path.to_str().unwrap().into() },
        config: KEYS.iter().map(|(c, g)| (c.to_string(), json!(g))).collect(),
        weights: RENAMES
            .iter()
            .chain(MORE.iter())
            .map(|(c, g)| Rename { from: g.to_string(), to: c.to_string() })
            .collect(),
        eos_token_ids: None,
        chat_template: None,
    }
}

fn reference_hash() -> String {
    let d = write_checkpoint(false);
    convert(&HfModelSource::from_dir(d.to_str().unwrap()).unwrap()).unwrap().check.hash
}

#[test]
fn an_auto_detected_spec_is_neutral() {
    let d = write_checkpoint(false);
    let spec = ModelSpec::detect(d.to_str().unwrap()).unwrap();
    assert_eq!((spec.group.as_str(), spec.weights.len(), spec.config.len()), ("decoder", 0, 0));
    assert_eq!(ModelSpec::from_json(&spec.to_json()).unwrap(), spec, "the spec survives a JSON round trip");
    assert_eq!(convert_spec(&spec).unwrap().check.hash, reference_hash());
}

#[test]
fn a_renamed_checkpoint_needs_only_its_spec() {
    let d = write_checkpoint(true);
    assert!(HfModelSource::from_dir(d.to_str().unwrap()).is_ok_and(|s| detect_group(&s).is_none()), "unknown as it is");
    let r = convert_spec(&renamed_spec(&d)).unwrap();
    assert!(r.check.deterministic);
    assert_eq!(r.check.hash, reference_hash(), "the renamed copy computes the same bits");
}

#[test]
fn a_dropped_rule_fails_closed_and_a_swapped_rule_is_caught_by_the_hash() {
    let d = write_checkpoint(true);
    let mut drop = renamed_spec(&d);
    drop.weights.retain(|r| !r.from.contains("ffn_down"));
    let err = convert_spec(&drop).expect_err("a missing tensor must fail the build");
    assert!(err.contains("down_proj"), "the error names the missing tensor: {err}");

    let mut swap = renamed_spec(&d);
    for r in swap.weights.iter_mut() {
        if r.from.contains("ffn_gate") {
            r.to = "model.layers.{i}.mlp.up_proj.weight".into();
        } else if r.from.contains("ffn_up") {
            r.to = "model.layers.{i}.mlp.gate_proj.weight".into();
        }
    }
    let r = convert_spec(&swap).expect("same shapes: it builds");
    assert_ne!(r.check.hash, reference_hash(), "a swapped mapping must not reproduce the hash");
}

#[test]
fn a_literal_override_changes_what_the_handler_reads() {
    let d = write_checkpoint(false);
    let mut one = ModelSpec::detect(d.to_str().unwrap()).unwrap();
    one.config.insert("num_hidden_layers".into(), json!(1));
    let src = SpecSource::open(&one).unwrap();
    assert_eq!(src.config_usize("num_hidden_layers"), Some(1));
    assert_eq!(src.config_usize("hidden_size"), Some(16), "unmapped keys pass through");
    assert_ne!(convert_spec(&one).unwrap().check.hash, reference_hash());
}

#[test]
fn the_registry_lists_group_to_models_and_sees_a_new_file() {
    let d = dir("registry");
    let spec = |id: &str, group: &str| {
        format!(r#"{{"id": "{id}", "group": "{group}", "source": {{"format": "safetensors", "path": "."}}}}"#)
    };
    std::fs::write(d.join("a.json"), spec("qwen", "decoder")).unwrap();
    std::fs::write(d.join("b.json"), spec("olmoe", "moe")).unwrap();
    std::fs::write(d.join("notes.txt"), "not a spec").unwrap();
    let r = SpecRegistry::load_dir(&d).unwrap();
    assert_eq!(r.groups()["decoder"], vec!["qwen"]);
    assert_eq!(r.groups()["moe"], vec!["olmoe"]);
    assert!(r.groups()["vision"].is_empty());
    std::fs::write(d.join("c.json"), spec("sam", "vision")).unwrap();
    assert_eq!(SpecRegistry::load_dir(&d).unwrap().groups()["vision"], vec!["sam"], "a new file needs no rebuild");
    std::fs::write(d.join("d.json"), spec("qwen", "decoder")).unwrap();
    assert!(SpecRegistry::load_dir(&d).err().unwrap().contains("duplicate"));
    std::fs::remove_file(d.join("d.json")).unwrap();
    std::fs::write(d.join("e.json"), spec("x", "audio")).unwrap();
    assert!(SpecRegistry::load_dir(&d).err().unwrap().contains("unknown group"));
}

#[test]
fn a_spec_is_data_and_every_mistake_is_refused() {
    let base = r#""id": "x", "group": "decoder", "source": {"format": "safetensors", "path": "."}"#;
    for (bad, why) in [
        (format!(r#"{{{base}, "wieghts": []}}"#), "unknown field"),
        (r#"{"id": "x", "group": "decoder", "source": {"format": "gguf", "path": "."}}"#.to_string(), "gguf"),
        (format!(r#"{{{base}, "weights": [{{"from": "blk.{{n}}.w", "to": "x.{{n}}.w"}}]}}"#), "placeholder"),
        (format!(r#"{{{base}, "weights": [{{"from": "blk.{{i}}.w", "to": "x.w"}}]}}"#), "placeholders"),
        (format!(r#"{{{base}, "weights": [{{"from": "blk.{{i}}{{e}}.w", "to": "x.{{i}}{{e}}"}}]}}"#), "text between"),
        (format!(r#"{{{base}, "config": {{"hidden_size": {{"eval": "1+1"}}}}}}"#), "config"),
        (format!(r#"{{{base}, "chat_template": "{{{{ messages }}}}"}}"#), "chat_template"),
        (r#"{"id": " ", "group": "decoder", "source": {"format": "safetensors", "path": "."}}"#.to_string(), "id"),
    ] {
        let e = ModelSpec::from_json(&bad).err().unwrap_or_else(|| panic!("accepted: {bad}"));
        assert!(e.contains(why), "{why}: {e}");
    }
}

#[test]
fn patterns_match_whole_names_and_carry_indices() {
    let p = Pattern::parse("blk.{i}.ffn_{e}.w").unwrap();
    let caps = p.matches("blk.12.ffn_3.w").unwrap();
    assert_eq!(Pattern::parse("layers.{i}.experts.{e}").unwrap().fill(&caps), "layers.12.experts.3");
    assert!(p.matches("blk.12.ffn_3.w.extra").is_none());
    assert!(p.matches("blk.x.ffn_3.w").is_none());
}
