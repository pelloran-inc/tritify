//! GLM-4.5's checkpoint layout through the builder, on a synthetic checkpoint: the sigmoid router and its
//! selection bias, the shared expert, the dense first layer, q/k/v biases and partial RoPE are each read and each change
//! the output; every unsupported knob fails closed.

use super::*;
use kt_core::plugin::ModelSource;

/// A tiny GLM-4.5-shaped checkpoint. `q_proj` is wider than the hidden size (4 heads · 8 > 16), as in GLM-4.5-Air.
#[derive(Clone)]
pub(crate) struct MockGlm {
    dense_first: usize,
    n_shared: usize,
    rotary: Option<f64>,
    attention_bias: bool,
    biases_present: bool,
    sel_bias: f32,
    n_group: usize,
    scaling: f64,
    eps: Option<f64>,
}

const DIM: usize = 16;
const LAYERS: usize = 3;
const HEADS: usize = 4;
const KV: usize = 2;
const HD: usize = 8;
const MOE_H: usize = 8;
const DENSE_H: usize = 24;
const VOCAB: usize = 24;
const E: usize = 8;

impl MockGlm {
    pub(crate) fn glm() -> Self {
        MockGlm {
            dense_first: 1,
            n_shared: 1,
            rotary: Some(0.5),
            attention_bias: true,
            biases_present: true,
            sel_bias: 0.5,
            n_group: 1,
            scaling: 1.0,
            eps: Some(1e-5),
        }
    }
}

fn gen(name: &str, len: usize, around: f32, spread: f32) -> Vec<f32> {
    let mut s = 1469598103934665603u64;
    for b in name.bytes() {
        s = (s ^ b as u64).wrapping_mul(1099511628211);
    }
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            around + (((s >> 40) as f64 / (1u64 << 24) as f64) as f32 - 0.5) * spread
        })
        .collect()
}

impl ModelSource for MockGlm {
    fn hf_arch_ids(&self) -> Vec<String> {
        vec!["Glm4MoeForCausalLM".into()]
    }
    fn config_usize(&self, key: &str) -> Option<usize> {
        match key {
            "hidden_size" => Some(DIM),
            "num_hidden_layers" => Some(LAYERS),
            "num_attention_heads" => Some(HEADS),
            "num_key_value_heads" => Some(KV),
            "head_dim" => Some(HD),
            "moe_intermediate_size" => Some(MOE_H),
            "intermediate_size" => Some(DENSE_H),
            "vocab_size" => Some(VOCAB),
            "n_routed_experts" => Some(E),
            "num_experts_per_tok" => Some(2),
            "n_shared_experts" => Some(self.n_shared),
            "first_k_dense_replace" => Some(self.dense_first),
            "n_group" => Some(self.n_group),
            "topk_group" => Some(1),
            _ => None,
        }
    }
    fn config_f64(&self, key: &str) -> Option<f64> {
        match key {
            "rms_norm_eps" => self.eps,
            "rope_theta" => Some(1e6),
            "partial_rotary_factor" => self.rotary,
            "routed_scaling_factor" => Some(self.scaling),
            _ => None,
        }
    }
    fn config_bool(&self, key: &str) -> Option<bool> {
        match key {
            "norm_topk_prob" => Some(true),
            "attention_bias" => Some(self.attention_bias),
            _ => None,
        }
    }
    fn tensor(&self, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        let (qd, kvd) = (HEADS * HD, KV * HD);
        let layer = name.strip_prefix("model.layers.").and_then(|r| r.split('.').next()).and_then(|i| i.parse().ok());
        let w = |shape: Vec<usize>| Some((gen(name, shape.iter().product(), 0.0, 0.2), shape));
        match name {
            "model.embed_tokens.weight" | "lm_head.weight" => return w(vec![VOCAB, DIM]),
            "model.norm.weight" => return Some((gen(name, DIM, 1.0, 0.1), vec![DIM])),
            _ => {}
        }
        let dense = layer.is_some_and(|i: usize| i < self.dense_first);
        let width = |n: &str| {
            if n.contains(".mlp.experts.") {
                MOE_H
            } else if n.contains(".mlp.shared_experts.") {
                MOE_H * self.n_shared
            } else {
                DENSE_H
            }
        };
        if name.ends_with("layernorm.weight") {
            Some((gen(name, DIM, 1.0, 0.1), vec![DIM]))
        } else if name.ends_with(".q_proj.weight") {
            w(vec![qd, DIM])
        } else if name.ends_with(".k_proj.weight") || name.ends_with(".v_proj.weight") {
            w(vec![kvd, DIM])
        } else if name.ends_with(".o_proj.weight") {
            w(vec![DIM, qd])
        } else if name.ends_with("_proj.bias") {
            let n = if name.ends_with(".q_proj.bias") { qd } else { kvd };
            if self.biases_present {
                w(vec![n])
            } else {
                None
            }
        } else if name.ends_with(".mlp.gate.weight") && !dense {
            w(vec![E, DIM])
        } else if name.ends_with(".mlp.gate.e_score_correction_bias") && !dense {
            Some((gen(name, E, 0.0, 2.0 * self.sel_bias), vec![E]))
        } else if name.ends_with(".gate_proj.weight") || name.ends_with(".up_proj.weight") {
            (!dense || !name.contains(".experts")).then(|| w(vec![width(name), DIM]))?
        } else if name.ends_with(".down_proj.weight") {
            (!dense || !name.contains(".experts")).then(|| w(vec![DIM, width(name)]))?
        } else {
            None
        }
    }
}

fn hash(src: &MockGlm) -> String {
    match MoeArch.build(src) {
        Ok(m) => m.self_check().hash,
        Err(e) => panic!("build: {e}"),
    }
}

fn refused(src: &MockGlm) -> String {
    match MoeArch.build(src) {
        Ok(_) => panic!("this checkpoint must fail closed"),
        Err(e) => e,
    }
}

#[test]
fn glm_detects_plans_builds_and_verifies() {
    let src = MockGlm::glm();
    assert!(MoeArch.detect(&src));
    let plan = MoeArch.plan(&src).expect("plan");
    // embedding; per layer rmsnorm, attention, rope, rmsnorm and the FFN (dense: one SwiGLU; MoE: E experts and the
    // shared one); the final rmsnorm and the LM head
    assert_eq!(plan.layers.len(), 1 + LAYERS * 4 + 1 + (LAYERS - 1) * (E + 1) + 2);
    let model = MoeArch.build(&src).expect("build");
    let rep = model.self_check();
    println!(
        "GLM(mock) self-check: metric={} max_abs_err={:.3e} det={}",
        rep.metric, rep.max_abs_err, rep.deterministic
    );
    assert!(rep.deterministic);
    assert!(rep.max_abs_err < 1.0, "int must track float, got {}", rep.max_abs_err);
    assert_eq!(rep.hash, MoeArch.build(&src).unwrap().self_check().hash, "the hash reproduces");
}

#[test]
fn every_glm_piece_is_read_and_changes_the_output() {
    let full = hash(&MockGlm::glm());
    let variants: [(&str, MockGlm); 5] = [
        ("no selection bias", MockGlm { sel_bias: 0.0, ..MockGlm::glm() }),
        ("no shared expert", MockGlm { n_shared: 0, ..MockGlm::glm() }),
        ("full RoPE", MockGlm { rotary: None, ..MockGlm::glm() }),
        ("no q/k/v biases", MockGlm { attention_bias: false, biases_present: false, ..MockGlm::glm() }),
        ("no dense first layer", MockGlm { dense_first: 0, ..MockGlm::glm() }),
    ];
    for (what, v) in variants {
        assert_ne!(hash(&v), full, "{what}: the knob must change the output");
    }
}

#[test]
fn glms_recorded_epsilon_default_is_the_value_its_config_states() {
    // GLM-4.5-Air states 1e-5, transformers' Glm4MoeConfig default; omitted, it builds the same model
    assert_eq!(hash(&MockGlm { eps: None, ..MockGlm::glm() }), hash(&MockGlm::glm()));
}

#[test]
fn glm_knobs_the_build_does_not_implement_fail_closed() {
    assert!(refused(&MockGlm { n_group: 2, ..MockGlm::glm() }).contains("n_group"));
    assert!(refused(&MockGlm { scaling: 2.5, ..MockGlm::glm() }).contains("routed_scaling_factor"));
    assert!(refused(&MockGlm { biases_present: false, ..MockGlm::glm() }).contains("attention_bias"));
    assert!(refused(&MockGlm { rotary: Some(0.125), ..MockGlm::glm() }).contains("partial_rotary_factor"));
}

#[cfg(feature = "plants")]
#[test]
fn every_plant_changes_the_output() {
    use kt_core::VerifiableModel;
    let src = MockGlm::glm();
    let full = hash(&src);
    for p in Plant::ALL {
        let planted = MoeArch::build_decoder_moe_planted(&src, p).expect("planted build").self_check().hash;
        assert_ne!(planted, full, "plant {}: it must change the output", p.name());
        assert_eq!(Plant::parse(p.name()), Some(p));
    }
}
