//! kt-convert — the converter driver. Ties the pieces together: detect the family, plan the graph, build
//! the exact-integer causal-LM (calibrating on a few tokens), run it and its float reference on a probe token
//! sequence, and report the argmax-token agreement + reproducing hash. Point it at a Hugging Face model
//! directory (`config.json` + `model.safetensors`) and it does the rest.
//!
//! Every model group registers here — decoder, Mixture-of-Experts and vision — so each one is built, verified and
//! hashed by the same path. Adding a family means registering its plugin in [`default_registry`]; nothing else
//! here changes. The three group handlers are re-exported, so a higher layer (`kt-cluster`) reaches every group
//! through ingest.
//!
//! # Example
//!
//! ```
//! use kt_convert::default_registry;
//! use kt_core::ModelSource;
//!
//! struct Cfg(&'static str);
//! impl ModelSource for Cfg {
//!     fn hf_arch_ids(&self) -> Vec<String> { vec![self.0.into()] }
//!     fn config_usize(&self, _key: &str) -> Option<usize> { None }
//!     fn config_f64(&self, _key: &str) -> Option<f64> { None }
//!     fn tensor(&self, _name: &str) -> Option<(Vec<f32>, Vec<usize>)> { None }
//! }
//!
//! // The registry picks the first installed plugin that recognises a checkpoint.
//! let registry = default_registry();
//! assert_eq!(registry.families().len(), 3); // decoder + MoE + vision
//! assert!(registry.select(&Cfg("Qwen2ForCausalLM")).is_some());
//! assert!(registry.select(&Cfg("Sam2Model")).is_some());
//! assert!(registry.select(&Cfg("OlmoeForCausalLM")).is_some());
//! ```

pub mod checkpoint;
pub mod hf;
pub mod lazy;
pub mod spec;
pub mod text;

pub use hf::HfModelSource;
pub use kt_model_decoder::DecoderArch;
pub use kt_model_moe::MoeArch;
pub use kt_model_vision::Sam2Arch;

use kt_core::plugin::{ModelArch, ModelSource};
use kt_core::{CheckReport, DecoderModel, Registry};

/// The installed model groups: decoder, Mixture-of-Experts and vision. Extend by registering more plugin crates
/// here — the generic flow (detect → plan → build → self-check) is identical whatever the task (decoder LM, MoE
/// LM, image segmenter, …).
pub fn default_registry() -> Registry {
    let mut r = Registry::new();
    r.register(Box::new(DecoderArch));
    r.register(Box::new(MoeArch));
    r.register(Box::new(Sam2Arch));
    r
}

/// Build a text-generating model (a decoder or a MoE decoder) from a checkpoint. The groups are tried in the
/// registry's order. Vision checkpoints are not text models and are refused.
pub fn load_text_model(src: &dyn ModelSource) -> Result<DecoderModel, String> {
    if DecoderArch.detect(src) {
        DecoderArch::build_decoder(src)
    } else if MoeArch.detect(src) {
        MoeArch::build_decoder_moe(src)
    } else {
        Err(format!("not a text-generation model (architectures {:?})", src.hf_arch_ids()))
    }
}

/// Result of a conversion attempt: which family matched, the declared op graph, and the model's own
/// task-appropriate self-check (int vs float where available, determinism, reproducing hash).
#[derive(Debug)]
pub struct ConvertReport {
    pub family: String,
    pub plan_summary: String,
    pub check: CheckReport,
    /// The decision agreement over [`FIDELITY_POSITIONS`] probe positions, for a language model; `None` for a model
    /// whose self-check already compares its decision (SAM's mask IoU).
    pub fidelity: Option<(f64, String)>,
}

/// The probe length of the binding fidelity measurement. At 32 positions one disagreement already gives 31/32 =
/// 0.96875, so the default threshold of 0.99 means every position must agree: a strict functional test, not a
/// proportional fraction. The probe is a fixed sequence of token ids, not a representative quality benchmark.
pub const FIDELITY_POSITIONS: usize = 32;

/// Run the full generic pipeline: select plugin → plan → build → self-check. Task-agnostic: the built model
/// verifies itself on its own built-in probe, so this function never needs to know whether it is converting a
/// language model or a vision model.
pub fn convert(src: &dyn ModelSource) -> Result<ConvertReport, String> {
    let reg = default_registry();
    let plugin =
        reg.select(src).ok_or_else(|| format!("no installed plugin matches architectures {:?}", src.hf_arch_ids()))?;
    let plan = plugin.plan(src)?;
    let model = plugin.build(src)?;
    let check = model.self_check();
    let fidelity = model.fidelity(FIDELITY_POSITIONS);
    Ok(ConvertReport { family: plugin.meta().family, plan_summary: plan.summary(), check, fidelity })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Map, Value};

    /// Build a multi-tensor `.safetensors` byte buffer (all F32).
    fn make_safetensors(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut hdr = Map::new();
        let mut off = 0usize;
        for (name, shape, data) in tensors {
            let nbytes = data.len() * 4;
            hdr.insert(
                name.clone(),
                json!({ "dtype": "F32", "shape": shape, "data_offsets": [off as u64, (off + nbytes) as u64] }),
            );
            off += nbytes;
        }
        let hjson = serde_json::to_vec(&Value::Object(hdr)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(hjson.len() as u64).to_le_bytes());
        out.extend_from_slice(&hjson);
        for (_, _, data) in tensors {
            for &x in data {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    }
    fn vecf(s: &mut u64, n: usize) -> Vec<f32> {
        (0..n).map(|_| ((lcg(s) >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) as f32).collect()
    }

    // A full tiny Qwen2-shaped model (1 layer, with q/k/v bias) as config + safetensors.
    fn tiny_model() -> (String, Vec<u8>) {
        let (dim, nh, nkv, hd, hidden, vocab) = (16usize, 4usize, 2usize, 4usize, 32usize, 20usize);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let cfg = json!({
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": dim, "num_hidden_layers": 1, "num_attention_heads": nh,
            "num_key_value_heads": nkv, "head_dim": hd, "intermediate_size": hidden,
            "vocab_size": vocab, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 64
        })
        .to_string();

        let mut s = 0x1234u64;
        let t = |s: &mut u64, name: &str, shape: Vec<usize>| {
            let n: usize = shape.iter().product();
            (name.to_string(), shape, vecf(s, n))
        };
        let tensors = vec![
            t(&mut s, "model.embed_tokens.weight", vec![vocab, dim]),
            t(&mut s, "model.layers.0.input_layernorm.weight", vec![dim]),
            t(&mut s, "model.layers.0.self_attn.q_proj.weight", vec![qd, dim]),
            t(&mut s, "model.layers.0.self_attn.q_proj.bias", vec![qd]),
            t(&mut s, "model.layers.0.self_attn.k_proj.weight", vec![kvd, dim]),
            t(&mut s, "model.layers.0.self_attn.k_proj.bias", vec![kvd]),
            t(&mut s, "model.layers.0.self_attn.v_proj.weight", vec![kvd, dim]),
            t(&mut s, "model.layers.0.self_attn.v_proj.bias", vec![kvd]),
            t(&mut s, "model.layers.0.self_attn.o_proj.weight", vec![dim, qd]),
            t(&mut s, "model.layers.0.post_attention_layernorm.weight", vec![dim]),
            t(&mut s, "model.layers.0.mlp.gate_proj.weight", vec![hidden, dim]),
            t(&mut s, "model.layers.0.mlp.up_proj.weight", vec![hidden, dim]),
            t(&mut s, "model.layers.0.mlp.down_proj.weight", vec![dim, hidden]),
            t(&mut s, "model.norm.weight", vec![dim]),
            // no lm_head.weight → tied to embed_tokens
        ];
        (cfg, make_safetensors(&tensors))
    }

    #[test]
    fn end_to_end_full_tiny_model_tracks_float() {
        let (cfg, st) = tiny_model();
        let src = HfModelSource::from_parts(&cfg, st).expect("load");
        let r = convert(&src).expect("convert");
        println!("family={} | {} | {}", r.family, r.plan_summary, r.check.summary());
        assert!(r.check.deterministic, "converted model must be deterministic");
        assert_eq!(r.check.task, "causal-lm");
        assert_eq!(r.check.metric, 1.0, "integer argmax must match float on every probe position");
    }

    #[test]
    fn unknown_architecture_is_rejected_cleanly() {
        let cfg = json!({ "architectures": ["MambaForCausalLM"], "hidden_size": 8, "vocab_size": 16 }).to_string();
        let st = make_safetensors(&[("x".to_string(), vec![1], vec![0.0])]);
        let src = HfModelSource::from_parts(&cfg, st).unwrap();
        let err = convert(&src).unwrap_err();
        assert!(err.contains("no installed plugin"), "got: {err}");
    }

    #[test]
    fn bf16_roundtrips_through_the_loader() {
        let mut st = Vec::new();
        let mut hdr = Map::new();
        hdr.insert("w".to_string(), json!({"dtype":"BF16","shape":[1],"data_offsets":[0u64,2u64]}));
        let hjson = serde_json::to_vec(&Value::Object(hdr)).unwrap();
        st.extend_from_slice(&(hjson.len() as u64).to_le_bytes());
        st.extend_from_slice(&hjson);
        st.extend_from_slice(&0x3FC0u16.to_le_bytes()); // BF16 1.5
        let cfg = json!({"architectures":["X"],"hidden_size":1}).to_string();
        let src = HfModelSource::from_parts(&cfg, st).unwrap();
        let (vals, shape) = src.tensor("w").unwrap();
        assert_eq!(shape, vec![1]);
        assert_eq!(vals[0], 1.5);
    }
}
