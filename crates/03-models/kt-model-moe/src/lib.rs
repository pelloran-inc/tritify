//! tritify architecture plugin — **generic Mixture-of-Experts decoders** (OLMoE / Mixtral / Qwen-MoE …).
//!
//! This is the MoE sibling of `kt-model-decoder`. It implements the same [`kt_core::ModelArch`] contract and
//! reuses the SAME engine: the only structural difference from a dense model is that each block's feed-forward is a
//! [`MoeMlp`] (router top-k + sparse SwiGLU experts) instead of a dense `SwiGluMlp`, and the attention may carry
//! **QK-norm**. Both are built through the generic [`DecoderModel::assemble`] seam, so the integer/float forwards — and
//! the verify gate — are identical to the dense path. Dense models keep building through `kt-model-decoder`,
//! byte-unchanged.
//!
//! **Generic by configuration, not by model name.** A MoE family is one `Scheme` entry below: its tensor-naming
//! (where the router + experts live) and its default knobs. A new softmax-router / SwiGLU-expert family is a new row —
//! no build logic changes. The per-model knobs the builder reads from `config.json`:
//!   - `E` (expert count) and `top_k` (`num_experts_per_tok`);
//!   - `norm_topk_prob` (renormalize the top-k weights — Mixtral/Qwen-MoE `true`, OLMoE `false`);
//!   - QK-norm present? (detected by the presence of the `q_norm`/`k_norm` tensors);
//!   - q/k/v biases (detected by the tensors; required when `attention_bias` is set) and `partial_rotary_factor`;
//!   - `first_k_dense_replace` (dense layers before the first MoE layer, of width `intermediate_size`) and
//!     `n_shared_experts` (an unweighted shared expert, for a scheme that names one);
//!   - `rms_norm_eps` as stated; when a config omits it, transformers' default for the architecture (OLMoE `1e-5`),
//!     and an architecture with no recorded default fails closed.
//!
//! **Honest scope.** Covered + verified end-to-end by a run: **OLMoE** (64 experts, full-projection QK-norm) and
//! **Qwen3-MoE** (Qwen3-30B-A3B, 128 experts, per-head QK-norm + `moe_intermediate_size` — handled generically: the
//! QK-norm granularity is read from the checkpoint and the expert FFN size from `moe_intermediate_size`, no per-model
//! branch). Covered + wired (naming + knobs) but pending a checkpoint to run-verify: Mixtral, Qwen2-MoE, and
//! **GLM-4.5** (a sigmoid router with a selection bias, an unweighted shared expert, a dense first layer,
//! q/k/v biases, partial RoPE; verified on slices of GLM-4.5-Air). Detected and **cleanly deferred** (a clear error,
//! never a silently wrong build): Qwen2-MoE's gated shared expert, any other family's shared experts (DeepSeek), and a
//! sigmoid router with more than one expert group or a routed scaling factor other than 1.
//!
//! # Example
//!
//! ```
//! use kt_model_moe::MoeArch;
//! use kt_core::{ModelArch, ModelSource};
//!
//! struct Cfg(&'static str);
//! impl ModelSource for Cfg {
//!     fn hf_arch_ids(&self) -> Vec<String> { vec![self.0.into()] }
//!     fn config_usize(&self, _key: &str) -> Option<usize> { None }
//!     fn config_f64(&self, _key: &str) -> Option<f64> { None }
//!     fn tensor(&self, _name: &str) -> Option<(Vec<f32>, Vec<usize>)> { None }
//! }
//!
//! // The generic handler recognises every MoE family it has a naming scheme for, and leaves dense models alone.
//! assert!(MoeArch.detect(&Cfg("OlmoeForCausalLM")));
//! assert!(MoeArch.detect(&Cfg("Qwen3MoeForCausalLM")));
//! assert!(MoeArch.detect(&Cfg("Glm4MoeForCausalLM")));
//! assert!(!MoeArch.detect(&Cfg("Qwen2ForCausalLM")));
//! ```

use kt_core::attention::Attention;
use kt_core::layers::{Embedding, MoeMlp, QuantLinear, RmsNorm, SwiGluMlp};
use kt_core::plugin::{ArchMeta, LayerSpec, ModelArch, ModelPlan, ModelSource};
use kt_core::{DecoderLayer, DecoderModel, Module, VerifiableModel};

/// Cap the RoPE table (memory) regardless of the model's `max_position_embeddings`.
const MAX_SEQ_CAP: usize = 4096;

/// One MoE family's tensor-naming + default knobs. The builder is family-agnostic; it reads the scheme.
struct Scheme {
    family: &'static str,
    arch_ids: &'static [&'static str],
    /// The per-layer MoE module name: `mlp` (OLMoE/Qwen-MoE) or `block_sparse_moe` (Mixtral).
    moe_mod: &'static str,
    /// Router weight leaf under the MoE module (`gate` → `…{moe_mod}.gate.weight`, shape `[E, dim]`).
    router: &'static str,
    /// Expert SwiGLU leaf names (OLMoE/Qwen-MoE: `gate_proj`/`up_proj`/`down_proj`; Mixtral: `w1`/`w3`/`w2`).
    e_gate: &'static str,
    e_up: &'static str,
    e_down: &'static str,
    /// Config keys that give the expert count `E` (first present wins).
    experts_keys: &'static [&'static str],
    /// `norm_topk_prob` default when the config omits it.
    default_norm_topk: bool,
    /// Has an end-to-end run on a real checkpoint of this family passed the verify gate?
    verified: bool,
    /// The sigmoid router's selection-bias leaf under the router (`e_score_correction_bias`), or `None` for a softmax
    /// router.
    sigmoid_bias: Option<&'static str>,
    /// The always-on, unweighted shared expert's leaf under the MoE module (`shared_experts`), or `None`.
    shared: Option<&'static str>,
}

const SCHEMES: &[Scheme] = &[
    Scheme {
        family: "OLMoE",
        arch_ids: &["OlmoeForCausalLM"],
        moe_mod: "mlp",
        router: "gate",
        e_gate: "gate_proj",
        e_up: "up_proj",
        e_down: "down_proj",
        experts_keys: &["num_experts"],
        default_norm_topk: false,
        verified: true,
        sigmoid_bias: None,
        shared: None,
    },
    Scheme {
        family: "Mixtral",
        arch_ids: &["MixtralForCausalLM"],
        moe_mod: "block_sparse_moe",
        router: "gate",
        e_gate: "w1",
        e_up: "w3",
        e_down: "w2",
        experts_keys: &["num_local_experts"],
        default_norm_topk: true,
        verified: false,
        sigmoid_bias: None,
        shared: None,
    },
    Scheme {
        family: "Qwen-MoE",
        arch_ids: &["Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM"],
        moe_mod: "mlp",
        router: "gate",
        e_gate: "gate_proj",
        e_up: "up_proj",
        e_down: "down_proj",
        experts_keys: &["num_experts"],
        default_norm_topk: true,
        verified: true, // Qwen3-30B-A3B (128 experts, per-head QK-norm) run-verified end to end
        sigmoid_bias: None,
        shared: None,
    },
    Scheme {
        family: "GLM-4.5 MoE",
        arch_ids: &["Glm4MoeForCausalLM"],
        moe_mod: "mlp",
        router: "gate",
        e_gate: "gate_proj",
        e_up: "up_proj",
        e_down: "down_proj",
        experts_keys: &["n_routed_experts"],
        default_norm_topk: true,
        verified: false, // verified on slices of GLM-4.5-Air, not end to end
        sigmoid_bias: Some("e_score_correction_bias"),
        shared: Some("shared_experts"),
    },
];

/// The RMSNorm epsilon of an architecture whose config omits `rms_norm_eps`: transformers' config-class default at
/// `f90e93c4`. Keyed by the architecture id, not the scheme, so a checkpoint built through the fallback scheme
/// does not inherit OLMoE's value. An architecture missing here must state its epsilon, or the build fails closed.
const DEFAULT_EPS: &[(&str, f64)] = &[
    ("OlmoeForCausalLM", 1e-5),   // configuration_olmoe.py:72
    ("Glm4MoeForCausalLM", 1e-5), // configuration_glm4_moe.py:100
];

/// Knobs this build does not yet implement — detected so a build FAILS CLOSED rather than silently producing a wrong
/// model. Each is a tracked next-slice item.
const SHARED_EXPERT_KEYS: &[&str] = &["num_shared_experts", "n_shared_experts", "shared_expert_intermediate_size"];

pub struct MoeArch;

/// **Planted architecture mistakes,** each a plausible misreading of GLM-4.5. The fidelity gate must tell
/// every one from the correct build.
#[cfg(feature = "plants")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plant {
    /// The selection bias left out of the choice
    NoSelectionBias,
    /// RoPE over the whole head instead of the first `partial_rotary_factor` of it
    FullRope,
    /// RoPE on the pairs `(2j, 2j+1)` instead of rotate_half's `(j, j + rd/2)`
    InterleavedPairs,
    /// The shared expert dropped
    NoSharedExpert,
    /// The q/k/v biases dropped
    NoAttentionBias,
}

#[cfg(feature = "plants")]
impl Plant {
    pub const ALL: [Plant; 5] = [
        Plant::NoSelectionBias,
        Plant::FullRope,
        Plant::InterleavedPairs,
        Plant::NoSharedExpert,
        Plant::NoAttentionBias,
    ];

    /// The plant's name on the command line: `p1`–`p5`.
    pub fn parse(s: &str) -> Option<Plant> {
        Plant::ALL.into_iter().find(|p| p.name() == s)
    }

    pub fn name(self) -> &'static str {
        match self {
            Plant::NoSelectionBias => "p1",
            Plant::FullRope => "p2",
            Plant::InterleavedPairs => "p3",
            Plant::NoSharedExpert => "p4",
            Plant::NoAttentionBias => "p5",
        }
    }
}

/// What the builder applies; every piece is on in a real build, and only a plant turns one off.
#[derive(Clone, Copy)]
struct Pieces {
    selection_bias: bool,
    partial_rope: bool,
    interleaved: bool,
    shared: bool,
    attention_bias: bool,
    /// Build no routed expert: each MoE layer is a [`MoeMlp::remote`] (a router-only coordinator).
    router_only: bool,
}

const REAL: Pieces = Pieces {
    selection_bias: true,
    partial_rope: true,
    interleaved: false,
    shared: true,
    attention_bias: true,
    router_only: false,
};

impl MoeArch {
    /// Pick the family scheme for a source: by `config.architectures` first; else, if a standard experts-count key is
    /// present, fall back to the OLMoE-style layout (`mlp.experts.{e}.gate_proj`) so an unseen MoE checkpoint with the
    /// standard tensor names still builds.
    fn scheme(src: &dyn ModelSource) -> Option<&'static Scheme> {
        let ids = src.hf_arch_ids();
        if let Some(s) = SCHEMES.iter().find(|s| ids.iter().any(|a| s.arch_ids.contains(&a.as_str()))) {
            return Some(s);
        }
        // no known arch id — is it MoE-shaped at all?
        let moe_shaped = ["num_experts", "num_local_experts", "n_routed_experts"]
            .iter()
            .any(|k| src.config_usize(k).is_some_and(|v| v > 0));
        moe_shaped.then(|| &SCHEMES[0]) // OLMoE-style default layout
    }

    fn cfg(src: &dyn ModelSource, key: &str) -> Result<usize, String> {
        src.config_usize(key).ok_or_else(|| format!("missing config key: {key}"))
    }

    /// Build the concrete, runnable integer [`DecoderModel`] for a MoE source (experts lazy-loaded through the source).
    pub fn build_decoder_moe(src: &dyn ModelSource) -> Result<DecoderModel, String> {
        Self::build_pieces(src, REAL)
    }

    /// [`MoeArch::build_decoder_moe`] without the routed experts: every other tensor is built, including the
    /// routers with their biases and the shared experts, and each MoE layer is a [`MoeMlp::remote`]. The routed experts
    /// are never read from the source. Such a model generates only through `decode_step_remote_moe` and
    /// `forward_int_remote_moe`; a local forward fails closed.
    pub fn build_decoder_moe_router_only(src: &dyn ModelSource) -> Result<DecoderModel, String> {
        Self::build_pieces(src, Pieces { router_only: true, ..REAL })
    }

    /// [`MoeArch::build_decoder_moe`] with one planted architecture mistake.
    #[cfg(feature = "plants")]
    pub fn build_decoder_moe_planted(src: &dyn ModelSource, plant: Plant) -> Result<DecoderModel, String> {
        let mut p = REAL;
        match plant {
            Plant::NoSelectionBias => p.selection_bias = false,
            Plant::FullRope => p.partial_rope = false,
            Plant::InterleavedPairs => p.interleaved = true,
            Plant::NoSharedExpert => p.shared = false,
            Plant::NoAttentionBias => p.attention_bias = false,
        }
        Self::build_pieces(src, p)
    }

    fn build_pieces(src: &dyn ModelSource, pieces: Pieces) -> Result<DecoderModel, String> {
        let sch = Self::scheme(src).ok_or("not a recognised MoE family")?;
        if !sch.verified {
            // honest diagnostic: the naming scheme is wired, but no run has verified this family end-to-end yet.
            eprintln!("[kt-model-moe] note: '{}' naming is wired but not yet run-verified; the verify gate still decides PASS/FAIL", sch.family);
        }

        // fail closed on knobs we don't implement yet (never silently wrong). A scheme with an unweighted shared
        // expert reads `n_shared_experts`; a gated one (Qwen2-MoE's `shared_expert_intermediate_size`) stays refused.
        for k in SHARED_EXPERT_KEYS {
            if sch.shared.is_some() && *k == "n_shared_experts" {
                continue;
            }
            if src.config_usize(k).is_some_and(|v| v > 0) {
                return Err(format!(
                    "{}: shared experts ({k}) not yet supported — tracked next-slice knob",
                    sch.family
                ));
            }
        }

        let dim = Self::cfg(src, "hidden_size")?;
        let n_layers = Self::cfg(src, "num_hidden_layers")?;
        let n_heads = Self::cfg(src, "num_attention_heads")?;
        let n_kv_heads = src.config_usize("num_key_value_heads").unwrap_or(n_heads);
        // the EXPERT FFN hidden size: Qwen3-MoE (and others) use a dedicated `moe_intermediate_size` for experts
        // distinct from the dense `intermediate_size`; fall back to `intermediate_size` (OLMoE/Mixtral).
        let hidden = src
            .config_usize("moe_intermediate_size")
            .or_else(|| src.config_usize("intermediate_size"))
            .ok_or_else(|| format!("{}: missing moe_intermediate_size / intermediate_size", sch.family))?;
        let vocab = Self::cfg(src, "vocab_size")?;
        let head_dim = src.config_usize("head_dim").unwrap_or(dim / n_heads);
        // the epsilon as stated, else the architecture's transformers default; never a guessed one
        let eps = match src.config_f64("rms_norm_eps") {
            Some(eps) => eps,
            None => {
                let ids = src.hf_arch_ids();
                DEFAULT_EPS.iter().find(|(a, _)| ids.iter().any(|i| i == a)).map(|&(_, eps)| eps).ok_or_else(|| {
                    format!(
                        "{}: missing config key: rms_norm_eps (no transformers default recorded for {ids:?})",
                        sch.family
                    )
                })?
            }
        };
        let rope_theta = src.config_f64("rope_theta").unwrap_or(10000.0);
        let max_seq = src.config_usize("max_position_embeddings").unwrap_or(MAX_SEQ_CAP).min(MAX_SEQ_CAP);
        let (qdim, kvdim) = (n_heads * head_dim, n_kv_heads * head_dim);

        let n_exp = sch
            .experts_keys
            .iter()
            .find_map(|k| src.config_usize(k))
            .ok_or_else(|| format!("{}: missing expert count ({:?})", sch.family, sch.experts_keys))?;
        let top_k = Self::cfg(src, "num_experts_per_tok")?;
        if top_k < 1 || top_k > n_exp {
            return Err(format!("{}: num_experts_per_tok {top_k} not in 1..={n_exp}", sch.family));
        }
        let norm_topk = src.config_bool("norm_topk_prob").unwrap_or(sch.default_norm_topk);
        if sch.sigmoid_bias.is_some() {
            // the sigmoid router as built: one expert group, no rescaling of the weights
            for (k, want) in [("n_group", 1usize), ("topk_group", 1)] {
                if src.config_usize(k).is_some_and(|v| v != want) {
                    return Err(format!("{}: {k} other than {want} is not supported", sch.family));
                }
            }
            if src.config_f64("routed_scaling_factor").is_some_and(|v| v != 1.0) {
                return Err(format!("{}: routed_scaling_factor other than 1.0 is not supported", sch.family));
            }
        }
        // the dense layers before the first MoE layer (GLM-4.5: `first_k_dense_replace`), with their own width
        let dense_first = src.config_usize("first_k_dense_replace").unwrap_or(0);
        let dense_hidden = src.config_usize("intermediate_size").unwrap_or(hidden);
        // partial RoPE: the rotated leading dims of each head, `int(head_dim · partial_rotary_factor)` as transformers
        let rot_dim = match src.config_f64("partial_rotary_factor") {
            Some(f) => {
                let rd = (head_dim as f64 * f) as usize;
                if rd == 0 || !rd.is_multiple_of(2) || rd > head_dim {
                    return Err(format!(
                        "{}: partial_rotary_factor {f} gives rotary dim {rd} (head_dim {head_dim})",
                        sch.family
                    ));
                }
                rd
            }
            None => head_dim,
        };
        let n_shared = src.config_usize("n_shared_experts").unwrap_or(0);

        // typed tensor readers (shape-checked; fail closed on a missing/mis-shaped tensor).
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
        let opt1 = |name: String| -> Option<(Vec<f32>, Vec<usize>)> { src.tensor(&name) };

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("model.layers.{i}");

            // attention + q/k/v biases when the checkpoint has them (GLM-4.5) + optional QK-norm, each detected by the
            // tensor's presence.
            let bias = |leaf: &str, n: usize| -> Result<Option<Vec<f32>>, String> {
                match opt1(format!("{p}.self_attn.{leaf}.bias")) {
                    Some((b, shape)) if shape == vec![n] => Ok(Some(b)),
                    Some((_, shape)) => Err(format!("{p}.self_attn.{leaf}.bias: shape {shape:?} != [{n}]")),
                    None => Ok(None),
                }
            };
            let (bq, bk, bv) = (bias("q_proj", qdim)?, bias("k_proj", kvdim)?, bias("v_proj", kvdim)?);
            if src.config_bool("attention_bias") == Some(true) && (bq.is_none() || bk.is_none() || bv.is_none()) {
                return Err(format!("{p}: attention_bias is set but a q/k/v bias is missing"));
            }
            let attn = Attention::from_weights(
                "attn",
                dim,
                n_heads,
                n_kv_heads,
                head_dim,
                &w2(format!("{p}.self_attn.q_proj.weight"), qdim, dim)?,
                &w2(format!("{p}.self_attn.k_proj.weight"), kvdim, dim)?,
                &w2(format!("{p}.self_attn.v_proj.weight"), kvdim, dim)?,
                &w2(format!("{p}.self_attn.o_proj.weight"), dim, qdim)?,
                if pieces.attention_bias { (bq.as_deref(), bk.as_deref(), bv.as_deref()) } else { (None, None, None) },
                max_seq,
                rope_theta,
            );
            let attn = if rot_dim == head_dim || !pieces.partial_rope { attn } else { attn.with_rotary_dim(rot_dim) };
            #[cfg(feature = "plants")]
            let attn = if pieces.interleaved { attn.planted_interleaved_pairs() } else { attn };
            #[cfg(not(feature = "plants"))]
            debug_assert!(!pieces.interleaved, "the interleaved-pairs plant exists only with the plants feature");
            // QK-norm — **generic & adaptive**: present iff the q_norm/k_norm tensors exist, and the *granularity* is
            // read from the tensor shape — `[qdim]`/`[kvdim]` = full-projection (OLMoE/Olmo2), `[head_dim]` = per-head
            // (Qwen3). `RmsNorm` normalizes row-wise over its own weight length, so the SAME attention applies either
            // one with no code branch: a head_dim-length norm on the full q simply normalizes each head.
            let attn = match opt1(format!("{p}.self_attn.q_norm.weight")) {
                Some((qw, qs)) => {
                    let per_head = qs == vec![head_dim];
                    if qs != vec![qdim] && !per_head {
                        return Err(format!("{p}.self_attn.q_norm.weight: shape {qs:?} is neither [{qdim}] (full) nor [{head_dim}] (per-head)"));
                    }
                    let kn = if per_head { head_dim } else { kvdim }; // k_norm matches the chosen granularity
                    let kw = w1(format!("{p}.self_attn.k_norm.weight"), kn).map_err(|e| {
                        format!("q_norm present ({} norm) but {e}", if per_head { "per-head" } else { "full" })
                    })?;
                    attn.with_qk_norm(Some(RmsNorm::new("q_norm", &qw, eps)), Some(RmsNorm::new("k_norm", &kw, eps)))
                }
                None => attn,
            };

            let swiglu = |name: String, pre: &str, h: usize| -> Result<SwiGluMlp, String> {
                Ok(SwiGluMlp::new(
                    name,
                    QuantLinear::from_weights("gate", &w2(format!("{pre}.{}.weight", sch.e_gate), h, dim)?, h, dim),
                    QuantLinear::from_weights("up", &w2(format!("{pre}.{}.weight", sch.e_up), h, dim)?, h, dim),
                    QuantLinear::from_weights("down", &w2(format!("{pre}.{}.weight", sch.e_down), dim, h)?, dim, h),
                ))
            };
            let mp = format!("{p}.{}", sch.moe_mod);
            if i < dense_first {
                // a dense layer before the first MoE layer: one SwiGLU of the dense width, computed where the model runs
                layers.push(DecoderLayer {
                    input_norm: RmsNorm::new("in_norm", &w1(format!("{p}.input_layernorm.weight"), dim)?, eps),
                    attn,
                    post_norm: RmsNorm::new(
                        "post_norm",
                        &w1(format!("{p}.post_attention_layernorm.weight"), dim)?,
                        eps,
                    ),
                    ffn: Box::new(swiglu(format!("l{i}.mlp"), &mp, dense_hidden)?),
                });
                continue;
            }

            // the MoE feed-forward: router + E sparse SwiGLU experts.
            let router = QuantLinear::from_weights(
                "router",
                &w2(format!("{mp}.{}.weight", sch.router), n_exp, dim)?,
                n_exp,
                dim,
            );
            let mut experts = Vec::with_capacity(if pieces.router_only { 0 } else { n_exp });
            for e in (0..n_exp).filter(|_| !pieces.router_only) {
                let ep = format!("{mp}.experts.{e}");
                experts.push(SwiGluMlp::new(
                    format!("l{i}.e{e}"),
                    QuantLinear::from_weights(
                        "gate",
                        &w2(format!("{ep}.{}.weight", sch.e_gate), hidden, dim)?,
                        hidden,
                        dim,
                    ),
                    QuantLinear::from_weights(
                        "up",
                        &w2(format!("{ep}.{}.weight", sch.e_up), hidden, dim)?,
                        hidden,
                        dim,
                    ),
                    QuantLinear::from_weights(
                        "down",
                        &w2(format!("{ep}.{}.weight", sch.e_down), dim, hidden)?,
                        dim,
                        hidden,
                    ),
                ));
            }
            let moe = if pieces.router_only {
                MoeMlp::remote(format!("l{i}.moe"), router, top_k)
            } else {
                MoeMlp::new(format!("l{i}.moe"), router, experts, top_k)
            };
            let mut moe = moe.with_norm_topk_prob(norm_topk);
            if let Some(leaf) = sch.sigmoid_bias {
                let bias = w1(format!("{mp}.{}.{leaf}", sch.router), n_exp)?;
                let zero = vec![0.0f32; n_exp];
                moe = moe.with_sigmoid_router(if pieces.selection_bias { &bias } else { &zero });
            }
            if let (Some(leaf), true) = (sch.shared, n_shared > 0 && pieces.shared) {
                moe =
                    moe.with_shared_expert(swiglu(format!("l{i}.shared"), &format!("{mp}.{leaf}"), hidden * n_shared)?);
            }
            let ffn: Box<dyn Module + Send + Sync> = Box::new(moe);

            layers.push(DecoderLayer {
                input_norm: RmsNorm::new("in_norm", &w1(format!("{p}.input_layernorm.weight"), dim)?, eps),
                attn,
                post_norm: RmsNorm::new("post_norm", &w1(format!("{p}.post_attention_layernorm.weight"), dim)?, eps),
                ffn,
            });
        }

        let embed = w2("model.embed_tokens.weight".into(), vocab, dim)?;
        let final_norm = RmsNorm::new("final_norm", &w1("model.norm.weight".into(), dim)?, eps);
        let lm_head_w = match src.tensor("lm_head.weight") {
            Some((t, shape)) => {
                if shape != vec![vocab, dim] {
                    return Err(format!("lm_head.weight: shape {shape:?} != [{vocab}, {dim}]"));
                }
                t
            }
            None => embed.clone(), // tied embeddings
        };
        let lm_head = QuantLinear::from_weights("lm_head", &lm_head_w, vocab, dim);
        let emb = Embedding::from_weights("embed", &embed, vocab, dim);
        Ok(DecoderModel::assemble(emb, layers, final_norm, lm_head, dim, vocab))
    }
}

impl ModelArch for MoeArch {
    fn meta(&self) -> ArchMeta {
        ArchMeta {
            family: "mixture-of-experts (OLMoE/Mixtral/Qwen-MoE/GLM-4.5)".into(),
            hf_arch_ids: SCHEMES.iter().flat_map(|s| s.arch_ids.iter().map(|x| x.to_string())).collect(),
            summary: "RMSNorm + RoPE + (GQA) attention with optional QK-norm + router top-k sparse SwiGLU experts"
                .into(),
        }
    }

    /// MoE detection is by config shape too: a known arch id, OR an experts count present (an unseen MoE checkpoint
    /// with the standard layout still matches).
    fn detect(&self, src: &dyn ModelSource) -> bool {
        MoeArch::scheme(src).is_some()
    }

    fn plan(&self, src: &dyn ModelSource) -> Result<ModelPlan, String> {
        let sch = MoeArch::scheme(src).ok_or("not a recognised MoE family")?;
        let dim = Self::cfg(src, "hidden_size")?;
        let layers = Self::cfg(src, "num_hidden_layers")?;
        let heads = Self::cfg(src, "num_attention_heads")?;
        let kv_heads = src.config_usize("num_key_value_heads").unwrap_or(heads);
        let hidden =
            src.config_usize("moe_intermediate_size").or_else(|| src.config_usize("intermediate_size")).unwrap_or(0);
        let vocab = Self::cfg(src, "vocab_size")?;
        let head_dim = src.config_usize("head_dim").unwrap_or(dim / heads);
        let n_exp = sch.experts_keys.iter().find_map(|k| src.config_usize(k)).unwrap_or(0);
        let dense_first = src.config_usize("first_k_dense_replace").unwrap_or(0);
        let dense_hidden = src.config_usize("intermediate_size").unwrap_or(hidden);
        let n_shared = if sch.shared.is_some() { src.config_usize("n_shared_experts").unwrap_or(0) } else { 0 };

        let mut specs = vec![LayerSpec::Embedding { vocab, dim }];
        for i in 0..layers {
            specs.push(LayerSpec::RmsNorm { dim });
            specs.push(LayerSpec::Attention { heads, kv_heads, head_dim });
            specs.push(LayerSpec::Rope { head_dim });
            specs.push(LayerSpec::RmsNorm { dim });
            if i < dense_first {
                specs.push(LayerSpec::SwiGluMlp { dim, hidden: dense_hidden });
                continue;
            }
            // one SwiGLU spec per expert — the plan shows the full expert bank the MoE routes over — and the shared one
            for _ in 0..n_exp {
                specs.push(LayerSpec::SwiGluMlp { dim, hidden });
            }
            if n_shared > 0 {
                specs.push(LayerSpec::SwiGluMlp { dim, hidden: hidden * n_shared });
            }
        }
        specs.push(LayerSpec::RmsNorm { dim });
        specs.push(LayerSpec::Linear { in_features: dim, out_features: vocab });
        Ok(ModelPlan { name: format!("moe-{}", sch.family), family: self.meta().family, layers: specs })
    }

    fn build(&self, src: &dyn ModelSource) -> Result<Box<dyn VerifiableModel>, String> {
        Ok(Box::new(Self::build_decoder_moe(src)?))
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_glm;
#[cfg(test)]
mod tests_remote;
