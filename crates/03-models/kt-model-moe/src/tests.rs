//! Synthetic-checkpoint tests for the generic MoE plugin — fast, hashed, no external model. A mock [`ModelSource`]
//! synthesizes correctly-shaped deterministic tensors by name, so the whole build → verify path runs in-process.

use super::*;
use kt_core::plugin::ModelSource;

/// A tiny MoE config whose tensors are generated on demand. OLMoE-style naming (`mlp.experts.{e}.gate_proj`).
pub(crate) struct MockMoe {
    arch: &'static str,
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    vocab: usize,
    n_exp: usize,
    top_k: usize,
    qk_norm: bool,
    per_head: bool, // Qwen3-style per-head QK-norm + a distinct moe_intermediate_size (decoy intermediate_size)
    norm_topk: Option<bool>,
    shared: bool,
    eps: Option<f64>, // `rms_norm_eps`; `None` omits it from the config
}

impl MockMoe {
    pub(crate) fn tiny(arch: &'static str) -> Self {
        MockMoe {
            arch,
            dim: 16,
            n_layers: 2,
            n_heads: 2,
            kv_heads: 2,
            head_dim: 8,
            hidden: 8,
            vocab: 24,
            n_exp: 4,
            top_k: 2,
            qk_norm: true,
            per_head: false,
            norm_topk: None,
            shared: false,
            eps: Some(1e-6),
        }
    }
    fn qdim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    fn kvdim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// Deterministic small values seeded by the tensor name; norm weights sit near 1.0 so the model is well-behaved.
fn gen(name: &str, len: usize, is_norm: bool) -> Vec<f32> {
    let mut s = 1469598103934665603u64; // FNV-ish seed over the name
    for b in name.bytes() {
        s = (s ^ b as u64).wrapping_mul(1099511628211);
    }
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 40) as f64) / ((1u64 << 24) as f64); // [0,1)
            if is_norm {
                1.0 + (u as f32 - 0.5) * 0.1
            } else {
                (u as f32 - 0.5) * 0.2
            }
        })
        .collect()
}

impl ModelSource for MockMoe {
    fn hf_arch_ids(&self) -> Vec<String> {
        vec![self.arch.into()]
    }

    fn config_usize(&self, key: &str) -> Option<usize> {
        match key {
            "hidden_size" => Some(self.dim),
            "num_hidden_layers" => Some(self.n_layers),
            "num_attention_heads" => Some(self.n_heads),
            "num_key_value_heads" => Some(self.kv_heads),
            "head_dim" => Some(self.head_dim),
            // per-head mode exercises moe_intermediate_size: experts use `hidden`, while intermediate_size is a decoy
            // (3×) — the build only succeeds if the builder reads moe_intermediate_size for the experts.
            "moe_intermediate_size" if self.per_head => Some(self.hidden),
            "intermediate_size" => Some(if self.per_head { self.hidden * 3 } else { self.hidden }),
            "vocab_size" => Some(self.vocab),
            "num_experts" => Some(self.n_exp),
            "num_experts_per_tok" => Some(self.top_k),
            "num_shared_experts" if self.shared => Some(2),
            _ => None,
        }
    }
    fn config_f64(&self, key: &str) -> Option<f64> {
        match key {
            "rms_norm_eps" => self.eps,
            "rope_theta" => Some(10000.0),
            _ => None,
        }
    }
    fn config_bool(&self, key: &str) -> Option<bool> {
        match key {
            "norm_topk_prob" => self.norm_topk,
            _ => None,
        }
    }

    fn tensor(&self, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        let (dim, hidden, vocab) = (self.dim, self.hidden, self.vocab);
        let (qdim, kvdim) = (self.qdim(), self.kvdim());
        // global tensors
        match name {
            "model.embed_tokens.weight" => return Some((gen(name, vocab * dim, false), vec![vocab, dim])),
            "model.norm.weight" => return Some((gen(name, dim, true), vec![dim])),
            "lm_head.weight" => return None, // tied embeddings (exercises the tied path)
            _ => {}
        }
        // per-layer tensors
        let shape: Vec<usize> =
            if name.ends_with(".input_layernorm.weight") || name.ends_with(".post_attention_layernorm.weight") {
                vec![dim]
            } else if name.ends_with(".self_attn.q_proj.weight") {
                vec![qdim, dim]
            } else if name.ends_with(".self_attn.k_proj.weight") || name.ends_with(".self_attn.v_proj.weight") {
                vec![kvdim, dim]
            } else if name.ends_with(".self_attn.o_proj.weight") {
                vec![dim, qdim]
            } else if name.ends_with(".self_attn.q_norm.weight") {
                if !self.qk_norm {
                    return None;
                }
                vec![if self.per_head { self.head_dim } else { qdim }]
            } else if name.ends_with(".self_attn.k_norm.weight") {
                if !self.qk_norm {
                    return None;
                }
                vec![if self.per_head { self.head_dim } else { kvdim }]
            } else if name.ends_with(".mlp.gate.weight") {
                vec![self.n_exp, dim]
            } else if name.ends_with(".gate_proj.weight") || name.ends_with(".up_proj.weight") {
                vec![hidden, dim]
            } else if name.ends_with(".down_proj.weight") {
                vec![dim, hidden]
            } else {
                return None;
            };
        let is_norm = name.ends_with("norm.weight");
        let len = shape.iter().product();
        Some((gen(name, len, is_norm), shape))
    }
}

#[test]
fn detects_olmoe_and_builds_and_verifies() {
    let src = MockMoe::tiny("OlmoeForCausalLM");
    let arch = MoeArch;
    assert!(arch.detect(&src), "OLMoE arch id must detect");
    let plan = arch.plan(&src).expect("plan");
    println!("{}", plan.summary());
    // one embedding + per layer (rmsnorm, attn, rope, rmsnorm, E×swiglu) + final rmsnorm + linear
    assert_eq!(plan.layers.len(), 1 + src.n_layers * (4 + src.n_exp) + 2);

    let model = arch.build(&src).expect("build MoE");
    let rep = model.self_check();
    println!(
        "OLMoE(mock) self-check: metric={} max_abs_err={:.3e} det={} hash={}",
        rep.metric, rep.max_abs_err, rep.deterministic, rep.hash
    );
    assert!(rep.deterministic, "integer path must be deterministic");
    assert!(rep.max_abs_err < 1.0, "int must track float (tiny well-behaved model), got {}", rep.max_abs_err);
    assert_eq!(rep.hash, model.self_check().hash, "self-check hash reproduces");
}

#[test]
fn norm_topk_prob_knob_is_wired_and_changes_output() {
    let arch = MoeArch;
    let mut off = MockMoe::tiny("OlmoeForCausalLM");
    off.norm_topk = Some(false);
    let mut on = MockMoe::tiny("OlmoeForCausalLM");
    on.norm_topk = Some(true);
    let h_off = arch.build(&off).expect("build off").self_check().hash;
    let h_on = arch.build(&on).expect("build on").self_check().hash;
    // same weights, different normalizer → the knob must actually change the integer output.
    assert_ne!(h_off, h_on, "norm_topk_prob must change the output (knob wired)");
}

#[test]
fn detects_a_moe_by_experts_count_without_a_known_arch_id() {
    // an unseen MoE arch id, but it carries an experts count + the standard OLMoE-style layout → still builds.
    let src = MockMoe::tiny("SomeNewMoEForCausalLM");
    let arch = MoeArch;
    assert!(arch.detect(&src), "a MoE-shaped config must detect even with an unknown arch id");
    let model = arch.build(&src).expect("build via fallback scheme");
    assert!(model.self_check().deterministic);
}

#[test]
fn an_omitted_epsilon_is_the_architectures_transformers_default() {
    // OLMoE's config sets no `rms_norm_eps` and transformers' OlmoeConfig gives 1e-5, so the build equals
    // one that states 1e-5. A stated value is used as stated: 1e-6 still builds the old model.
    let hash =
        |eps| MoeArch.build(&MockMoe { eps, ..MockMoe::tiny("OlmoeForCausalLM") }).expect("build").self_check().hash;
    let omitted = hash(None);
    assert_eq!(omitted, hash(Some(1e-5)), "an omitted epsilon must be OLMoE's 1e-5");
    assert_ne!(omitted, hash(Some(1e-6)), "a stated 1e-6 must build a different model");
}

#[test]
fn an_architecture_without_a_recorded_default_must_state_its_epsilon() {
    // the fallback scheme builds an unseen MoE (above) only when its config states the epsilon
    let msg = match MoeArch.build(&MockMoe { eps: None, ..MockMoe::tiny("SomeNewMoEForCausalLM") }) {
        Ok(_) => panic!("an omitted epsilon without a recorded default must fail closed"),
        Err(e) => e,
    };
    assert!(msg.contains("missing config key: rms_norm_eps"), "error should name the key, got: {msg}");
}

#[test]
fn does_not_detect_a_dense_model() {
    struct Dense;
    impl ModelSource for Dense {
        fn hf_arch_ids(&self) -> Vec<String> {
            vec!["Qwen2ForCausalLM".into()]
        }
        fn config_usize(&self, _k: &str) -> Option<usize> {
            None
        } // no experts count
        fn config_f64(&self, _k: &str) -> Option<f64> {
            None
        }
        fn tensor(&self, _n: &str) -> Option<(Vec<f32>, Vec<usize>)> {
            None
        }
    }
    assert!(!MoeArch.detect(&Dense), "a dense model must NOT be claimed by the MoE plugin");
}

#[test]
fn defers_shared_experts_cleanly() {
    let mut src = MockMoe::tiny("Qwen2MoeForCausalLM");
    src.shared = true;
    // Box<dyn VerifiableModel> isn't Debug, so match rather than expect_err.
    let msg = match MoeArch.build(&src) {
        Ok(_) => panic!("shared experts must fail closed, not build wrong"),
        Err(e) => e,
    };
    assert!(msg.contains("shared experts"), "error should name shared experts, got: {msg}");
}

#[test]
fn builds_qwen3_style_per_head_qknorm_and_moe_intermediate() {
    // Qwen3-MoE shape: per-head QK-norm (head_dim-length q_norm/k_norm) + a dedicated moe_intermediate_size (with a
    // decoy intermediate_size = 3×). Generic/adaptive: the builder reads the QK-norm granularity from the tensor shape
    // and the expert size from moe_intermediate_size — no per-model branch. The build succeeds only if both are right.
    let mut src = MockMoe::tiny("Qwen3MoeForCausalLM");
    src.per_head = true;
    src.norm_topk = Some(true);
    let arch = MoeArch;
    assert!(arch.detect(&src), "Qwen3-MoE arch id must detect");
    let model = arch.build(&src).expect("build Qwen3-style MoE (per-head QK-norm + moe_intermediate_size)");
    let rep = model.self_check();
    println!(
        "Qwen3(mock) self-check: metric={} max_abs_err={:.3e} det={}",
        rep.metric, rep.max_abs_err, rep.deterministic
    );
    assert!(rep.deterministic, "integer path must be deterministic");
    assert!(rep.max_abs_err < 1.0, "int must track float, got {}", rep.max_abs_err);
}
