//! THE PLUGIN CONTRACT. A model family is added by implementing [`ModelArch`] in its own crate — the engine
//! never changes. A plugin does exactly three things:
//!   1. [`ModelArch::detect`] — say whether an [`ModelSource`] is this family (from `config.architectures`).
//!   2. [`ModelArch::plan`]   — declare the generic op graph it will build ([`ModelPlan`] of [`LayerSpec`]s),
//!      so the converter can show/inspect it before touching weights.
//!   3. [`ModelArch::build`]  — instantiate the runnable [`Module`](crate::Module) from the source's weights.
//!
//! A [`Registry`] holds the installed plugins; [`Registry::select`] picks the first whose `detect` matches.

use crate::check::VerifiableModel;

/// How a plugin reads a model: its declared architecture ids, scalar config values, and named weight tensors.
/// A real HF loader (config.json + safetensors) implements this; tests implement a synthetic mock.
pub trait ModelSource {
    /// `config.architectures` (e.g. `["Qwen2ForCausalLM"]`).
    fn hf_arch_ids(&self) -> Vec<String>;
    fn config_usize(&self, key: &str) -> Option<usize>;
    fn config_f64(&self, key: &str) -> Option<f64>;
    /// A boolean config value (e.g. `norm_topk_prob`). Default `None` so existing sources need not implement it; a
    /// plugin then falls back to a family default. A real HF loader reads the JSON bool.
    fn config_bool(&self, _key: &str) -> Option<bool> {
        None
    }
    /// A named weight tensor as float values + shape (e.g. `"model.layers.0.mlp.gate_proj.weight"`).
    fn tensor(&self, name: &str) -> Option<(Vec<f32>, Vec<usize>)>;
    /// Every tensor name the source holds — lets a plugin that needs many tensors (e.g. a vision backbone) pull
    /// them all without enumerating by hand. Default empty for sources that only serve named lookups.
    fn tensor_names(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Metadata a plugin advertises about itself.
#[derive(Clone, Debug)]
pub struct ArchMeta {
    pub family: String,
    /// The `config.architectures` ids this plugin claims.
    pub hf_arch_ids: Vec<String>,
    pub summary: String,
}

/// A declarative description of one op in the graph — what the plugin intends to build, before weights.
#[derive(Clone, Debug, PartialEq)]
pub enum LayerSpec {
    Embedding { vocab: usize, dim: usize },
    RmsNorm { dim: usize },
    LayerNorm { dim: usize },
    Linear { in_features: usize, out_features: usize },
    Rope { head_dim: usize },
    Attention { heads: usize, kv_heads: usize, head_dim: usize },
    Sdpa { heads: usize, head_dim: usize },
    SwiGluMlp { dim: usize, hidden: usize },
    Conv2d { c_in: usize, c_out: usize, k: usize },
    ConvTranspose2d { c_in: usize, c_out: usize, k: usize },
}

/// The op graph a plugin declares for a model.
#[derive(Clone, Debug)]
pub struct ModelPlan {
    pub name: String,
    pub family: String,
    pub layers: Vec<LayerSpec>,
}

impl ModelPlan {
    /// A one-line human summary (counts per op kind) — printed by the converter for inspection.
    pub fn summary(&self) -> String {
        use std::collections::BTreeMap;
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for l in &self.layers {
            let kind = match l {
                LayerSpec::Embedding { .. } => "embedding",
                LayerSpec::RmsNorm { .. } => "rmsnorm",
                LayerSpec::LayerNorm { .. } => "layernorm",
                LayerSpec::Linear { .. } => "linear",
                LayerSpec::Rope { .. } => "rope",
                LayerSpec::Attention { .. } => "attention",
                LayerSpec::Sdpa { .. } => "sdpa",
                LayerSpec::SwiGluMlp { .. } => "mlp",
                LayerSpec::Conv2d { .. } => "conv2d",
                LayerSpec::ConvTranspose2d { .. } => "convtranspose2d",
            };
            *counts.entry(kind).or_insert(0) += 1;
        }
        let body: Vec<String> = counts.iter().map(|(k, n)| format!("{n} {k}")).collect();
        format!("{} ({}): {}", self.name, self.family, body.join(", "))
    }
}

/// The contract a model family implements. Lives in its own crate; the engine depends only on this trait.
pub trait ModelArch {
    fn meta(&self) -> ArchMeta;

    /// Is `src` this family? (Match `src.hf_arch_ids()` against `meta().hf_arch_ids`.)
    fn detect(&self, src: &dyn ModelSource) -> bool {
        let ids = src.hf_arch_ids();
        let mine = self.meta().hf_arch_ids;
        ids.iter().any(|a| mine.iter().any(|m| m == a))
    }

    /// Declare the op graph for this source (reads config, not weights).
    fn plan(&self, src: &dyn ModelSource) -> Result<ModelPlan, String>;

    /// Instantiate the runnable, self-verifying model from the source's weights (activations are quantized
    /// dynamically at run time, so no calibration input is needed). The concrete task — causal LM, image
    /// segmenter, … — lives inside the returned [`VerifiableModel`]; the engine only ever self-checks it.
    fn build(&self, src: &dyn ModelSource) -> Result<Box<dyn VerifiableModel>, String>;
}

/// Holds the installed plugins and selects one for a given source.
#[derive(Default)]
pub struct Registry {
    plugins: Vec<Box<dyn ModelArch>>,
}

impl Registry {
    pub fn new() -> Self {
        Registry { plugins: Vec::new() }
    }

    pub fn register(&mut self, plugin: Box<dyn ModelArch>) {
        self.plugins.push(plugin);
    }

    /// The first installed plugin that recognises `src`, if any.
    pub fn select(&self, src: &dyn ModelSource) -> Option<&dyn ModelArch> {
        self.plugins.iter().map(|b| b.as_ref()).find(|p| p.detect(src))
    }

    pub fn families(&self) -> Vec<String> {
        self.plugins.iter().map(|p| p.meta().family).collect()
    }
}
