//! tritify architecture plugin — **Llama/Qwen/Mistral-style decoder-only transformers**.
//!
//! The reference plugin and the template every other family copies. It implements the generic
//! [`kt_core::ModelArch`] contract and nothing else: recognise the family from `config.architectures`,
//! declare the op graph ([`DecoderArch::plan`]), and instantiate the runnable [`kt_core::CausalLM`] from
//! the source's weights ([`DecoderArch::build`]). The engine (`kt-core`) never changes to add a family —
//! you add a crate like this one.
//!
//! One plugin covers Qwen2.5, Llama-3.x, Mistral, SmolLM2 and Phi: they share RMSNorm + RoPE + (grouped-query)
//! attention + SwiGLU, differing only in config numbers and in whether the attention projections carry a bias
//! (Qwen2 does; Llama does not — handled transparently).
//!
//! # Example
//!
//! ```
//! use kt_model_decoder::DecoderArch;
//! use kt_core::{ModelArch, ModelSource};
//!
//! // A checkpoint's config is enough to detect the family and plan the op graph (no weights needed).
//! struct Qwen2;
//! impl ModelSource for Qwen2 {
//!     fn hf_arch_ids(&self) -> Vec<String> { vec!["Qwen2ForCausalLM".into()] }
//!     fn config_usize(&self, key: &str) -> Option<usize> {
//!         match key {
//!             "hidden_size" => Some(1536),
//!             "num_hidden_layers" => Some(28),
//!             "num_attention_heads" => Some(12),
//!             "num_key_value_heads" => Some(2),
//!             "intermediate_size" => Some(8960),
//!             "vocab_size" => Some(151936),
//!             _ => None,
//!         }
//!     }
//!     fn config_f64(&self, _key: &str) -> Option<f64> { None }
//!     fn tensor(&self, _name: &str) -> Option<(Vec<f32>, Vec<usize>)> { None }
//! }
//! assert!(DecoderArch.detect(&Qwen2));
//! let plan = DecoderArch.plan(&Qwen2).unwrap();
//! assert_eq!(plan.layers.len(), 1 + 28 * 5 + 2); // embedding · 5 ops per block · final norm + LM head
//! ```

use kt_core::plugin::{ArchMeta, LayerSpec, ModelArch, ModelPlan, ModelSource};
use kt_core::{DecoderConfig, DecoderModel, DecoderWeights, LayerWeights, VerifiableModel};

/// Cap the RoPE table size (memory) regardless of the model's max_position_embeddings.
const MAX_SEQ_CAP: usize = 4096;

pub struct DecoderArch;

impl DecoderArch {
    pub const ARCH_IDS: &'static [&'static str] =
        &["Qwen2ForCausalLM", "LlamaForCausalLM", "MistralForCausalLM", "Phi3ForCausalLM", "SmolLM2ForCausalLM"];

    fn cfg(src: &dyn ModelSource, key: &str) -> Result<usize, String> {
        src.config_usize(key).ok_or_else(|| format!("missing config key: {key}"))
    }

    /// Build the concrete, runnable integer [`DecoderModel`] from a source — the exact construction the
    /// [`ModelArch::build`] trait method boxes, exposed concretely so callers (e.g. an end-to-end LM check)
    /// can drive its [`kt_core::CausalLM`] forwards directly at a chosen sequence length.
    pub fn build_decoder(src: &dyn ModelSource) -> Result<DecoderModel, String> {
        let dim = Self::cfg(src, "hidden_size")?;
        let n_layers = Self::cfg(src, "num_hidden_layers")?;
        let n_heads = Self::cfg(src, "num_attention_heads")?;
        let n_kv_heads = src.config_usize("num_key_value_heads").unwrap_or(n_heads);
        let hidden = Self::cfg(src, "intermediate_size")?;
        let vocab = Self::cfg(src, "vocab_size")?;
        let head_dim = src.config_usize("head_dim").unwrap_or(dim / n_heads);
        // no default: a config that omits the epsilon is refused, not built at a guessed one
        let eps = src.config_f64("rms_norm_eps").ok_or("missing config key: rms_norm_eps")?;
        let rope_theta = src.config_f64("rope_theta").unwrap_or(10000.0);
        let max_seq = src.config_usize("max_position_embeddings").unwrap_or(MAX_SEQ_CAP).min(MAX_SEQ_CAP);
        let (qdim, kvdim) = (n_heads * head_dim, n_kv_heads * head_dim);

        let w2 = |name: String, rows: usize, cols: usize| -> Result<Vec<f32>, String> {
            let (t, shape) = src.tensor(&name).ok_or_else(|| format!("missing tensor {name}"))?;
            if shape != vec![rows, cols] {
                return Err(format!("{name}: shape {shape:?} != [{rows}, {cols}]"));
            }
            Ok(t)
        };
        let w1 = |name: String, n: usize| -> Result<Vec<f32>, String> {
            let (t, shape) = src.tensor(&name).ok_or_else(|| format!("missing tensor {name}"))?;
            if shape != vec![n] {
                return Err(format!("{name}: shape {shape:?} != [{n}]"));
            }
            Ok(t)
        };
        let bias = |name: String| -> Option<Vec<f32>> { src.tensor(&name).map(|(t, _)| t) };

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("model.layers.{i}");
            layers.push(LayerWeights {
                input_norm: w1(format!("{p}.input_layernorm.weight"), dim)?,
                wq: w2(format!("{p}.self_attn.q_proj.weight"), qdim, dim)?,
                wk: w2(format!("{p}.self_attn.k_proj.weight"), kvdim, dim)?,
                wv: w2(format!("{p}.self_attn.v_proj.weight"), kvdim, dim)?,
                wo: w2(format!("{p}.self_attn.o_proj.weight"), dim, qdim)?,
                wq_b: bias(format!("{p}.self_attn.q_proj.bias")),
                wk_b: bias(format!("{p}.self_attn.k_proj.bias")),
                wv_b: bias(format!("{p}.self_attn.v_proj.bias")),
                post_norm: w1(format!("{p}.post_attention_layernorm.weight"), dim)?,
                gate: w2(format!("{p}.mlp.gate_proj.weight"), hidden, dim)?,
                up: w2(format!("{p}.mlp.up_proj.weight"), hidden, dim)?,
                down: w2(format!("{p}.mlp.down_proj.weight"), dim, hidden)?,
            });
        }

        let embed = w2("model.embed_tokens.weight".into(), vocab, dim)?;
        let final_norm = w1("model.norm.weight".into(), dim)?;
        let lm_head = match src.tensor("lm_head.weight") {
            Some((t, shape)) => {
                if shape != vec![vocab, dim] {
                    return Err(format!("lm_head.weight: shape {shape:?} != [{vocab}, {dim}]"));
                }
                t
            }
            None => embed.clone(), // tied embeddings
        };

        let cfg =
            DecoderConfig { dim, n_layers, n_heads, n_kv_heads, head_dim, hidden, vocab, eps, rope_theta, max_seq };
        let weights = DecoderWeights { embed, layers, final_norm, lm_head, cfg };
        Ok(DecoderModel::build(&weights))
    }
}

impl ModelArch for DecoderArch {
    fn meta(&self) -> ArchMeta {
        ArchMeta {
            family: "decoder-only (Llama/Qwen/Mistral)".into(),
            hf_arch_ids: Self::ARCH_IDS.iter().map(|s| s.to_string()).collect(),
            summary: "RMSNorm + RoPE + (grouped-query) attention + SwiGLU MLP, tied/untied lm_head".into(),
        }
    }

    fn plan(&self, src: &dyn ModelSource) -> Result<ModelPlan, String> {
        let dim = Self::cfg(src, "hidden_size")?;
        let layers = Self::cfg(src, "num_hidden_layers")?;
        let heads = Self::cfg(src, "num_attention_heads")?;
        let kv_heads = src.config_usize("num_key_value_heads").unwrap_or(heads);
        let hidden = Self::cfg(src, "intermediate_size")?;
        let vocab = Self::cfg(src, "vocab_size")?;
        let head_dim = src.config_usize("head_dim").unwrap_or(dim / heads);

        let mut specs = vec![LayerSpec::Embedding { vocab, dim }];
        for _ in 0..layers {
            specs.push(LayerSpec::RmsNorm { dim });
            specs.push(LayerSpec::Attention { heads, kv_heads, head_dim });
            specs.push(LayerSpec::Rope { head_dim });
            specs.push(LayerSpec::RmsNorm { dim });
            specs.push(LayerSpec::SwiGluMlp { dim, hidden });
        }
        specs.push(LayerSpec::RmsNorm { dim });
        specs.push(LayerSpec::Linear { in_features: dim, out_features: vocab });
        Ok(ModelPlan { name: "decoder".into(), family: self.meta().family, layers: specs })
    }

    fn build(&self, src: &dyn ModelSource) -> Result<Box<dyn VerifiableModel>, String> {
        Ok(Box::new(Self::build_decoder(src)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Cfg {
        arch: &'static str,
        eps: Option<f64>,
    }
    impl ModelSource for Cfg {
        fn hf_arch_ids(&self) -> Vec<String> {
            vec![self.arch.into()]
        }
        fn config_usize(&self, key: &str) -> Option<usize> {
            match key {
                "hidden_size" => Some(1536),
                "num_hidden_layers" => Some(28),
                "num_attention_heads" => Some(12),
                "num_key_value_heads" => Some(2),
                "intermediate_size" => Some(8960),
                "vocab_size" => Some(151936),
                _ => None,
            }
        }
        fn config_f64(&self, k: &str) -> Option<f64> {
            if k == "rms_norm_eps" {
                return self.eps;
            }
            None
        }
        fn tensor(&self, _n: &str) -> Option<(Vec<f32>, Vec<usize>)> {
            None
        }
    }

    #[test]
    fn detects_qwen2_and_plans_the_graph() {
        let arch = DecoderArch;
        let src = Cfg { arch: "Qwen2ForCausalLM", eps: None };
        assert!(arch.detect(&src));
        let plan = arch.plan(&src).expect("plan");
        println!("{}", plan.summary());
        assert_eq!(plan.layers.len(), 1 + 28 * 5 + 2);
        assert!(matches!(plan.layers[0], LayerSpec::Embedding { .. }));
        assert!(matches!(plan.layers.last().unwrap(), LayerSpec::Linear { .. }));
    }

    #[test]
    fn does_not_detect_a_foreign_family() {
        let arch = DecoderArch;
        let src = Cfg { arch: "BertModel", eps: None };
        assert!(!arch.detect(&src));
    }

    #[test]
    fn a_config_without_rms_norm_eps_is_refused() {
        // No silent epsilon default. Omitted, the build stops at the key; stated, it passes the key and stops
        // only at the first tensor this config-only source lacks.
        let err = |eps| match DecoderArch::build_decoder(&Cfg { arch: "Qwen2ForCausalLM", eps }) {
            Ok(_) => panic!("a config-only source must not build"),
            Err(e) => e,
        };
        assert_eq!(err(None), "missing config key: rms_norm_eps");
        assert_eq!(err(Some(1e-6)), "missing tensor model.layers.0.input_layernorm.weight");
    }
}
