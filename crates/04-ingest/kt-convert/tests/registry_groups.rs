//! The one registry holds every model group. A checkpoint of each group is
//! recognised by `default_registry`, and a MoE checkpoint converts end to end through `convert()` — the same
//! detect → plan → build → self-check path as a decoder or a vision model. Firing control: unregister `MoeArch` in
//! `default_registry` and `a_moe_checkpoint_converts_through_the_registry` fails ("no installed plugin").

use kt_convert::{convert, default_registry, HfModelSource};
use serde_json::{json, Map, Value};

/// A multi-tensor `.safetensors` byte buffer (all F32).
fn safetensors(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut hdr = Map::new();
    let mut off = 0u64;
    for (name, shape, data) in tensors {
        let end = off + 4 * data.len() as u64;
        hdr.insert(name.clone(), json!({ "dtype": "F32", "shape": shape, "data_offsets": [off, end] }));
        off = end;
    }
    let hjson = serde_json::to_vec(&Value::Object(hdr)).unwrap();
    let mut out = (hjson.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&hjson);
    for (_, _, data) in tensors {
        out.extend(data.iter().flat_map(|x| x.to_le_bytes()));
    }
    out
}

/// Deterministic small weights seeded by the tensor name; norm weights sit near 1.0.
fn weights(name: &str, len: usize) -> Vec<f32> {
    let mut s = name.bytes().fold(1469598103934665603u64, |s, b| (s ^ b as u64).wrapping_mul(1099511628211));
    let norm = name.ends_with("norm.weight");
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 40) as f64 / (1u64 << 24) as f64) as f32 - 0.5;
            if norm {
                1.0 + 0.1 * u
            } else {
                0.2 * u
            }
        })
        .collect()
}

/// A tiny OLMoE-shaped checkpoint: 2 layers, 4 experts, top-2, full-projection QK-norm, tied embeddings.
fn tiny_olmoe() -> (String, Vec<u8>) {
    let (dim, layers, heads, head_dim, hidden, vocab, experts) = (16, 2, 2, 8, 8, 24, 4);
    let cfg = json!({
        "architectures": ["OlmoeForCausalLM"],
        "hidden_size": dim, "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": heads, "head_dim": head_dim, "intermediate_size": hidden, "vocab_size": vocab,
        "num_experts": experts, "num_experts_per_tok": 2, "norm_topk_prob": false,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 64
    })
    .to_string();
    let qd = heads * head_dim;
    let mut shapes: Vec<(String, Vec<usize>)> =
        vec![("model.embed_tokens.weight".into(), vec![vocab, dim]), ("model.norm.weight".into(), vec![dim])];
    for i in 0..layers {
        let p = format!("model.layers.{i}");
        shapes.push((format!("{p}.input_layernorm.weight"), vec![dim]));
        shapes.push((format!("{p}.post_attention_layernorm.weight"), vec![dim]));
        for w in ["q_proj", "k_proj", "v_proj"] {
            shapes.push((format!("{p}.self_attn.{w}.weight"), vec![qd, dim]));
        }
        shapes.push((format!("{p}.self_attn.o_proj.weight"), vec![dim, qd]));
        shapes.push((format!("{p}.self_attn.q_norm.weight"), vec![qd]));
        shapes.push((format!("{p}.self_attn.k_norm.weight"), vec![qd]));
        shapes.push((format!("{p}.mlp.gate.weight"), vec![experts, dim]));
        for e in 0..experts {
            shapes.push((format!("{p}.mlp.experts.{e}.gate_proj.weight"), vec![hidden, dim]));
            shapes.push((format!("{p}.mlp.experts.{e}.up_proj.weight"), vec![hidden, dim]));
            shapes.push((format!("{p}.mlp.experts.{e}.down_proj.weight"), vec![dim, hidden]));
        }
    }
    let tensors: Vec<_> = shapes
        .into_iter()
        .map(|(n, s)| {
            let w = weights(&n, s.iter().product());
            (n, s, w)
        })
        .collect();
    (cfg, safetensors(&tensors))
}

#[test]
fn the_default_registry_holds_all_three_groups() {
    let reg = default_registry();
    assert_eq!(reg.families().len(), 3, "decoder + MoE + vision: {:?}", reg.families());
    let pick = |arch: &str| {
        let cfg = json!({ "architectures": [arch] }).to_string();
        let src = HfModelSource::from_parts(&cfg, safetensors(&[])).unwrap();
        reg.select(&src).map(|p| p.meta().family)
    };
    assert!(pick("Qwen2ForCausalLM").is_some_and(|f| f.contains("decoder")));
    assert!(pick("OlmoeForCausalLM").is_some_and(|f| f.contains("mixture-of-experts")));
    assert!(pick("Sam2Model").is_some_and(|f| f.contains("sam2")));
}

#[test]
fn a_moe_checkpoint_converts_through_the_registry() {
    let (cfg, st) = tiny_olmoe();
    let src = HfModelSource::from_parts(&cfg, st).expect("load");
    let r = convert(&src).expect("the converter must build a MoE model through the registry");
    println!("family={} | {} | {}", r.family, r.plan_summary, r.check.summary());
    assert!(r.family.contains("mixture-of-experts"), "family: {}", r.family);
    assert_eq!(r.check.task, "causal-lm");
    assert!(r.check.deterministic, "the converted MoE model must be deterministic");
    assert_eq!(r.check.metric, 1.0, "integer argmax must match float on every probe position");
    assert_eq!(r.check.hash, convert(&src).unwrap().check.hash, "the reproducing hash reproduces");
}
