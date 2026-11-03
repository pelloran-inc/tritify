//! Declarative model specs: a model is added by one record, not by code. A [`ModelSpec`] names:
//! - the model's group (`decoder`, `moe` or `vision`)
//! - its checkpoint
//! - how the checkpoint's config keys and tensor names map to the names the group's generic handler reads
//! - optional overrides
//!
//! [`SpecSource`] applies the mapping, so the handler sees the names it already knows and no handler changes for a new
//! model. A spec is JSON and is only ever read as data: unknown fields are refused and nothing in it is executed.
//!
//! The names a handler reads are the Hugging Face names (`model.layers.{i}.self_attn.q_proj.weight`, `hidden_size`, …).
//! A checkpoint that uses them needs an empty mapping, which is what [`ModelSpec::detect`] writes. A checkpoint that
//! names things its own way, for example GGUF-style `blk.{i}.attn_q.weight` and `n_embd`, gets rename rules instead.
//!
//! # Example
//!
//! ```
//! use kt_convert::spec::ModelSpec;
//!
//! let spec = ModelSpec::from_json(r#"{
//!     "id": "my-qwen", "group": "decoder", "arch": "Qwen2ForCausalLM",
//!     "source": {"format": "safetensors", "path": "/models/my-qwen"},
//!     "config": {"hidden_size": "n_embd", "num_hidden_layers": 2},
//!     "weights": [{"from": "blk.{i}.attn_q.weight", "to": "model.layers.{i}.self_attn.q_proj.weight"}]
//! }"#).unwrap();
//! assert_eq!(spec.group, "decoder");
//! // declarative only: a misspelt field, an unknown group or another file format is refused
//! assert!(ModelSpec::from_json(&r#"{"id": "x", "group": "decoder", "source": {"format": "safetensors", "path": "."}, "wieghts": []}"#).is_err());
//! assert!(ModelSpec::from_json(r#"{"id": "x", "group": "audio", "source": {"format": "safetensors", "path": "."}}"#).is_err());
//! assert!(ModelSpec::from_json(r#"{"id": "x", "group": "decoder", "source": {"format": "gguf", "path": "."}}"#).is_err());
//! ```

use crate::hf::HfModelSource;
use crate::text::{ChatTemplate, Tokenizer};
use crate::{ConvertReport, DecoderArch, MoeArch, Sam2Arch};
use kt_core::plugin::{ModelArch, ModelSource};
use kt_core::{DecoderModel, VerifiableModel};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// The model groups, in the registry's order.
pub const GROUPS: [&str; 3] = ["decoder", "moe", "vision"];

/// One model, declared.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub id: String,
    /// `decoder`, `moe` or `vision`.
    pub group: String,
    /// The architecture id the group handler keys on (`Qwen2ForCausalLM`, `OlmoeForCausalLM`, …); by default the
    /// checkpoint's own `config.architectures`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
    pub source: SourceDecl,
    /// Canonical config key → the checkpoint's key (a string), or a literal value (a number or a bool).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub config: BTreeMap<String, Value>,
    /// Tensor rename rules, checkpoint name → canonical name; the first rule that matches a name applies.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub weights: Vec<Rename>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eos_token_ids: Option<Vec<u32>>,
    /// `chatml` or `plain`; by default what the tokenizer's vocabulary suggests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
}

/// Where the weights are.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceDecl {
    /// `safetensors` (a directory with `config.json` and one or more `.safetensors` files).
    pub format: String,
    pub path: String,
}

/// A tensor rename rule. `{i}` and `{e}` match a decimal index and carry it over.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rename {
    pub from: String,
    pub to: String,
}

impl ModelSpec {
    /// Parse and validate a spec.
    pub fn from_json(json: &str) -> Result<ModelSpec, String> {
        let spec: ModelSpec = serde_json::from_str(json).map_err(|e| format!("model spec: {e}"))?;
        spec.validate()?;
        Ok(spec)
    }

    /// Read and validate a spec file.
    pub fn load(path: impl AsRef<Path>) -> Result<ModelSpec, String> {
        let p = path.as_ref();
        ModelSpec::from_json(&std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?)
            .map_err(|e| format!("{}: {e}", p.display()))
    }

    /// The spec as pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a spec serializes")
    }

    /// Check everything a spec can get wrong before it touches a checkpoint.
    pub fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("id is empty".into());
        }
        if !GROUPS.contains(&self.group.as_str()) {
            return Err(format!("unknown group '{}' (decoder, moe, vision)", self.group));
        }
        match self.source.format.as_str() {
            "safetensors" => {}
            "gguf" => return Err("format 'gguf' is not supported yet: convert to safetensors; GGUF-style tensor names are supported through rename rules".into()),
            f => return Err(format!("unknown source format '{f}' (safetensors)")),
        }
        for (k, v) in &self.config {
            if !(v.is_string() || v.is_number() || v.is_boolean()) {
                return Err(format!("config '{k}': a checkpoint key (string) or a literal number/bool"));
            }
        }
        for r in &self.weights {
            let (from, to) = (Pattern::parse(&r.from)?, Pattern::parse(&r.to)?);
            if from.holes() != to.holes() {
                return Err(format!("rule {} → {}: both sides need the same placeholders", r.from, r.to));
            }
        }
        if let Some(t) = &self.chat_template {
            if !["chatml", "plain", "glm4", "glm4-nothink"].contains(&t.as_str()) {
                return Err(format!("chat_template '{t}' (chatml, plain, glm4, glm4-nothink)"));
            }
        }
        Ok(())
    }

    /// Write the spec of a checkpoint that uses the canonical names: its group by registry detection, an empty
    /// mapping, and the end-of-sequence ids and chat layout its tokenizer files give.
    pub fn detect(dir: &str) -> Result<ModelSpec, String> {
        let src = HfModelSource::from_dir(dir)?;
        let group =
            detect_group(&src).ok_or_else(|| format!("no group recognises architectures {:?}", src.hf_arch_ids()))?;
        let tok = (group != "vision").then(|| Tokenizer::from_dir(dir).ok()).flatten();
        let id = Path::new(dir).file_name().and_then(|s| s.to_str()).unwrap_or("model").to_lowercase();
        Ok(ModelSpec {
            id,
            group: group.into(),
            arch: None,
            source: SourceDecl { format: "safetensors".into(), path: dir.into() },
            config: BTreeMap::new(),
            weights: Vec::new(),
            eos_token_ids: tok.as_ref().map(|t| t.eos_ids().to_vec()).filter(|e| !e.is_empty()),
            chat_template: tok.map(|t| match t.chat_template() {
                ChatTemplate::ChatMl => "chatml".into(),
                ChatTemplate::Plain => "plain".into(),
                ChatTemplate::Glm4 { thinking: true } => "glm4".into(),
                ChatTemplate::Glm4 { thinking: false } => "glm4-nothink".into(),
            }),
        })
    }
}

/// The group a checkpoint belongs to, by the registry's detection order (decoder, MoE, vision).
pub fn detect_group(src: &dyn ModelSource) -> Option<&'static str> {
    GROUPS.into_iter().find(|g| handler(g).is_ok_and(|h| h.detect(src)))
}

/// The one generic handler of a group.
pub fn handler(group: &str) -> Result<Box<dyn ModelArch>, String> {
    match group {
        "decoder" => Ok(Box::new(DecoderArch)),
        "moe" => Ok(Box::new(MoeArch)),
        "vision" => Ok(Box::new(Sam2Arch)),
        g => Err(format!("unknown group '{g}'")),
    }
}

/// A rename pattern: literal text and `{i}` / `{e}` holes, each matching one or more decimal digits.
#[derive(Debug)]
struct Pattern(Vec<Part>);

#[derive(Debug, PartialEq)]
enum Part {
    Lit(String),
    Hole(char),
}

impl Pattern {
    fn parse(s: &str) -> Result<Pattern, String> {
        let mut parts = Vec::new();
        let mut rest = s;
        while let Some(i) = rest.find('{') {
            let close = rest[i..].find('}').ok_or_else(|| format!("rule '{s}': unclosed '{{'"))? + i;
            let hole = match &rest[i + 1..close] {
                "i" => 'i',
                "e" => 'e',
                h => return Err(format!("rule '{s}': unknown placeholder '{{{h}}}' ({{i}}, {{e}})")),
            };
            if i > 0 {
                parts.push(Part::Lit(rest[..i].to_string()));
            } else if matches!(parts.last(), Some(Part::Hole(_))) {
                return Err(format!("rule '{s}': two placeholders need text between them"));
            }
            parts.push(Part::Hole(hole));
            rest = &rest[close + 1..];
        }
        if rest.contains('}') {
            return Err(format!("rule '{s}': stray '}}'"));
        }
        if !rest.is_empty() {
            parts.push(Part::Lit(rest.to_string()));
        }
        for w in parts.windows(2) {
            if let (Part::Hole(_), Part::Lit(l)) = (&w[0], &w[1]) {
                if l.starts_with(|c: char| c.is_ascii_digit()) {
                    return Err(format!("rule '{s}': a placeholder may not be followed by a digit"));
                }
            }
        }
        if parts.is_empty() {
            return Err("a rule side is empty".into());
        }
        Ok(Pattern(parts))
    }

    fn holes(&self) -> Vec<char> {
        let mut h: Vec<char> =
            self.0.iter().filter_map(|p| if let Part::Hole(c) = p { Some(*c) } else { None }).collect();
        h.sort_unstable();
        h
    }

    /// The captured indices if `name` matches the whole pattern.
    fn matches(&self, name: &str) -> Option<HashMap<char, String>> {
        let (mut pos, mut caps) = (0, HashMap::new());
        for p in &self.0 {
            match p {
                Part::Lit(l) => {
                    if !name[pos..].starts_with(l.as_str()) {
                        return None;
                    }
                    pos += l.len();
                }
                Part::Hole(h) => {
                    let n = name[pos..].bytes().take_while(u8::is_ascii_digit).count();
                    if n == 0 {
                        return None;
                    }
                    caps.insert(*h, name[pos..pos + n].to_string());
                    pos += n;
                }
            }
        }
        (pos == name.len()).then_some(caps)
    }

    fn fill(&self, caps: &HashMap<char, String>) -> String {
        self.0
            .iter()
            .map(|p| match p {
                Part::Lit(l) => l.clone(),
                Part::Hole(h) => caps[h].clone(),
            })
            .collect()
    }
}

/// A checkpoint seen through a spec: config keys and tensor names translated to the canonical ones.
pub struct SpecSource {
    inner: HfModelSource,
    spec: ModelSpec,
    /// canonical tensor name → the checkpoint's name
    names: HashMap<String, String>,
}

impl SpecSource {
    /// Open the spec's checkpoint and index every tensor under its canonical name. Two checkpoint tensors that map
    /// to one canonical name are an error.
    pub fn open(spec: &ModelSpec) -> Result<SpecSource, String> {
        spec.validate()?;
        let inner = HfModelSource::from_dir(&spec.source.path)?;
        let rules: Vec<(Pattern, Pattern)> = spec
            .weights
            .iter()
            .map(|r| Ok((Pattern::parse(&r.from)?, Pattern::parse(&r.to)?)))
            .collect::<Result<_, String>>()?;
        let mut names = HashMap::new();
        for ck in inner.tensor_names() {
            let canon =
                rules.iter().find_map(|(f, t)| f.matches(&ck).map(|c| t.fill(&c))).unwrap_or_else(|| ck.clone());
            if let Some(prev) = names.insert(canon.clone(), ck.clone()) {
                return Err(format!("tensors '{prev}' and '{ck}' both map to '{canon}'"));
            }
        }
        Ok(SpecSource { inner, spec: spec.clone(), names })
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }
}

impl ModelSource for SpecSource {
    fn hf_arch_ids(&self) -> Vec<String> {
        self.spec.arch.clone().map_or_else(|| self.inner.hf_arch_ids(), |a| vec![a])
    }
    fn config_usize(&self, key: &str) -> Option<usize> {
        match self.spec.config.get(key) {
            Some(Value::String(k)) => self.inner.config_usize(k),
            Some(v) => v.as_u64().map(|v| v as usize),
            None => self.inner.config_usize(key),
        }
    }
    fn config_f64(&self, key: &str) -> Option<f64> {
        match self.spec.config.get(key) {
            Some(Value::String(k)) => self.inner.config_f64(k),
            Some(v) => v.as_f64(),
            None => self.inner.config_f64(key),
        }
    }
    fn config_bool(&self, key: &str) -> Option<bool> {
        match self.spec.config.get(key) {
            Some(Value::String(k)) => self.inner.config_bool(k),
            Some(v) => v.as_bool(),
            None => self.inner.config_bool(key),
        }
    }
    fn tensor(&self, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        self.inner.tensor(self.names.get(name)?)
    }
    fn tensor_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.names.keys().cloned().collect();
        v.sort_unstable();
        v
    }
}

/// Convert and self-verify a model through its spec, by its group's handler.
pub fn convert_spec(spec: &ModelSpec) -> Result<ConvertReport, String> {
    let h = handler(&spec.group)?;
    let src = SpecSource::open(spec)?;
    let plan = h.plan(&src)?;
    let model: Box<dyn VerifiableModel> = h.build(&src)?;
    let fidelity = model.fidelity(crate::FIDELITY_POSITIONS);
    Ok(ConvertReport { family: h.meta().family, plan_summary: plan.summary(), check: model.self_check(), fidelity })
}

/// The text-generating model of a `decoder` or `moe` spec.
pub fn load_text_model_spec(spec: &ModelSpec) -> Result<DecoderModel, String> {
    let src = SpecSource::open(spec)?;
    match spec.group.as_str() {
        "decoder" => DecoderArch::build_decoder(&src),
        "moe" => MoeArch::build_decoder_moe(&src),
        g => Err(format!("group '{g}' is not a text-generation group")),
    }
}

/// The tokenizer of a text spec: the checkpoint's `tokenizer.json`, with the spec's overrides applied.
pub fn tokenizer_for(spec: &ModelSpec) -> Result<Tokenizer, String> {
    let mut t = Tokenizer::from_dir(&spec.source.path)?;
    if let Some(e) = &spec.eos_token_ids {
        t.set_eos_ids(e.clone());
    }
    match spec.chat_template.as_deref() {
        Some("chatml") => t.set_chat_template(ChatTemplate::ChatMl),
        Some("plain") => t.set_chat_template(ChatTemplate::Plain),
        Some("glm4") => t.set_chat_template(ChatTemplate::Glm4 { thinking: true }),
        Some("glm4-nothink") => t.set_chat_template(ChatTemplate::Glm4 { thinking: false }),
        _ => {}
    }
    Ok(t)
}

/// A directory of spec files, listed group → models for the product and the UI.
pub struct SpecRegistry {
    specs: BTreeMap<String, ModelSpec>,
}

impl SpecRegistry {
    /// Load every `*.json` file in `dir`. A file that is not a valid spec, or a repeated id, is an error.
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<SpecRegistry, String> {
        let dir = dir.as_ref();
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        files.sort();
        let mut specs = BTreeMap::new();
        for f in files {
            let s = ModelSpec::load(&f)?;
            if specs.contains_key(&s.id) {
                return Err(format!("{}: duplicate model id '{}'", f.display(), s.id));
            }
            specs.insert(s.id.clone(), s);
        }
        Ok(SpecRegistry { specs })
    }

    /// Every group with the ids of its models (a group may be empty).
    pub fn groups(&self) -> BTreeMap<&'static str, Vec<&str>> {
        GROUPS
            .into_iter()
            .map(|g| (g, self.specs.values().filter(|s| s.group == g).map(|s| s.id.as_str()).collect()))
            .collect()
    }

    pub fn get(&self, id: &str) -> Option<&ModelSpec> {
        self.specs.get(id)
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }
}

#[cfg(test)]
mod tests;
