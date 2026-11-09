//! Reproducibility you can check yourself: a greedy run's logits hashed at every position, compared across compute
//! backends and against a stored reference that is bound to the checkpoint and the input.
//!
//! - **A run** ([`run`]) feeds the prompt position by position, then generates greedily, exactly as
//!   [`kt_core::generate::generate_greedy`] does. It records every position's logits hash ([`logits_hash`]: the
//!   scale and every code), the generated tokens, and the whole run's `logits_hash` (the same value
//!   `generate_greedy` reports).
//! - **Two runs** are compared position by position ([`first_difference`]): the report names the first position whose
//!   logits differ, not only that the runs differ.
//! - **A reference** ([`Reference`], format [`FORMAT`]) stores one run, bound to the checkpoint's files
//!   ([`checkpoint_identity`]) and to the input's ids. Another machine checks its own run against it
//!   ([`Reference::check`]); a reference for another checkpoint or another input is refused.
//!
//! Every value here is an integer or a hash of integers, so the same model and input give the same bytes on every
//! machine, backend and thread count.
use std::path::Path;

use kt_core::generate::{generate_greedy_by, Generation};
use kt_core::model::DecoderModel;
use kt_core::tensor::IntTensor;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The reference file's format id.
pub const FORMAT: &str = "tritify-repro/1";

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// SHA-256 of one logits tensor: its scale (`mant` as `i64`, `shift` as `i32`), then every code as `i32`, all
/// little-endian: the bytes one step adds to `generate_greedy`'s `logits_hash`. The scale is included, so two
/// logits that differ only in their scale do not collide.
pub fn logits_hash(t: &IntTensor) -> String {
    let mut h = Sha256::new();
    h.update(t.scale.mant.to_le_bytes());
    h.update(t.scale.shift.to_le_bytes());
    for &v in &t.data {
        h.update(v.to_le_bytes());
    }
    hex(&h.finalize())
}

/// The SHA-256 of a token sequence, comma-joined: what a reference's input is bound to.
pub fn ids_sha256(ids: &[u32]) -> String {
    hex(&Sha256::digest(ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",").as_bytes()))
}

/// One greedy run: every position's logits hash, in position order (the prompt's, then each generated token's), and
/// the generation itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub hashes: Vec<String>,
    pub generation: Generation,
}

/// Run `prompt` then up to `max_new` greedy tokens on the installed backend, hashing every position's logits.
/// `perturb` flips the lowest bit of the first logit code at that position *in its hash only*: a test of the
/// comparison, which must then report exactly that position; the run itself is unchanged.
pub fn run(model: &DecoderModel, prompt: &[u32], max_new: usize, stop: &[u32], perturb: Option<usize>) -> Run {
    let mut cache = model.new_cache();
    let mut hashes = Vec::with_capacity(prompt.len() + max_new);
    let generation = generate_greedy_by(
        prompt,
        max_new,
        stop,
        |_| {},
        |t, pos| {
            let l = model.decode_step(t, &mut cache, pos);
            if perturb == Some(pos) {
                let mut p = l.clone();
                if let Some(v) = p.data.first_mut() {
                    *v ^= 1;
                }
                hashes.push(logits_hash(&p));
            } else {
                hashes.push(logits_hash(&l));
            }
            l
        },
    );
    Run { hashes, generation }
}

/// Where two runs first differ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Difference {
    /// The logits at this position differ (the two hashes are given).
    Position { pos: usize, a: String, b: String },
    /// The generated tokens differ at this index (their logits were equal up to it, so this only shows if the hashes
    /// were not compared, or one run stopped early).
    Token { index: usize },
    /// One run has more positions than the other, which are equal as far as both go.
    Length { a: usize, b: usize },
    /// Every position and token equal, but the whole-run hash differs (it cannot, unless the stop rule did).
    RunHash,
}

/// The first difference between two runs, or `None` if they are bit-identical at every position.
pub fn first_difference(a: &Run, b: &Run) -> Option<Difference> {
    if let Some(pos) = a.hashes.iter().zip(&b.hashes).position(|(x, y)| x != y) {
        return Some(Difference::Position { pos, a: a.hashes[pos].clone(), b: b.hashes[pos].clone() });
    }
    if let Some(index) = a.generation.tokens.iter().zip(&b.generation.tokens).position(|(x, y)| x != y) {
        return Some(Difference::Token { index });
    }
    if a.hashes.len() != b.hashes.len() || a.generation.tokens.len() != b.generation.tokens.len() {
        return Some(Difference::Length { a: a.hashes.len(), b: b.hashes.len() });
    }
    if a.generation.logits_hash != b.generation.logits_hash {
        return Some(Difference::RunHash);
    }
    None
}

/// The checkpoint's identity: SHA-256 over its files the loader reads (`config.json`, every `*.safetensors`,
/// `tokenizer.json`), as `name:sha256` lines in name order. Returns the identity and the per-file hashes.
pub fn checkpoint_identity(dir: impl AsRef<Path>) -> Result<(String, Vec<(String, String)>), String> {
    let dir = dir.as_ref();
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n == "config.json" || n == "tokenizer.json" || n.ends_with(".safetensors"))
        .collect();
    names.sort();
    if !names.iter().any(|n| n.ends_with(".safetensors")) {
        return Err(format!("{}: no .safetensors file", dir.display()));
    }
    let mut files = Vec::with_capacity(names.len());
    for n in names {
        let p = dir.join(&n);
        let mut f = std::fs::File::open(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let mut h = Sha256::new();
        std::io::copy(&mut f, &mut h).map_err(|e| format!("{}: {e}", p.display()))?;
        files.push((n, hex(&h.finalize())));
    }
    let lines: String = files.iter().map(|(n, s)| format!("{n}:{s}\n")).collect();
    Ok((hex(&Sha256::digest(lines.as_bytes())), files))
}

/// A stored run, bound to the checkpoint and the input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reference {
    pub format: String,
    /// [`checkpoint_identity`] (or a synthetic model's identity).
    pub checkpoint: String,
    /// The per-file hashes behind it, as a record.
    pub checkpoint_files: Vec<(String, String)>,
    /// The model's name and revision, as the writer gave them (a record; the binding is `checkpoint`).
    pub model: String,
    pub revision: String,
    /// [`ids_sha256`] of the prompt.
    pub prompt_sha256: String,
    pub prompt_len: usize,
    pub max_new: usize,
    pub tokens: Vec<u32>,
    pub hashes: Vec<String>,
    pub logits_hash: String,
    /// Where it was written: the architecture and every backend that agreed (a record only).
    pub written_on: String,
    pub backends: Vec<String>,
}

/// Why a reference does not apply to a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    Format(String),
    Checkpoint { reference: String, this: String },
    Input,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::Format(s) => write!(f, "the reference is in format {s}, not {FORMAT}"),
            Refusal::Checkpoint { reference, this } => {
                write!(f, "the reference is for checkpoint {reference}, not this checkpoint {this}")
            }
            Refusal::Input => write!(f, "the reference is for another input (prompt or token budget)"),
        }
    }
}

impl Reference {
    /// The reference of `run` on this checkpoint and input.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        checkpoint: (String, Vec<(String, String)>),
        model: &str,
        revision: &str,
        prompt: &[u32],
        max_new: usize,
        run: &Run,
        written_on: &str,
        backends: Vec<String>,
    ) -> Reference {
        Reference {
            format: FORMAT.into(),
            checkpoint: checkpoint.0,
            checkpoint_files: checkpoint.1,
            model: model.into(),
            revision: revision.into(),
            prompt_sha256: ids_sha256(prompt),
            prompt_len: prompt.len(),
            max_new,
            tokens: run.generation.tokens.clone(),
            hashes: run.hashes.clone(),
            logits_hash: run.generation.logits_hash.clone(),
            written_on: written_on.into(),
            backends,
        }
    }

    /// The JSON text (pretty, one stable field order: the same run gives the same bytes).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a reference serialises") + "\n"
    }

    /// Parse a reference.
    pub fn from_json(s: &str) -> Result<Reference, String> {
        serde_json::from_str(s).map_err(|e| format!("not a reference: {e}"))
    }

    /// `Err` unless this reference is in this format, for this checkpoint and for this input.
    pub fn applies(&self, checkpoint: &str, prompt: &[u32], max_new: usize) -> Result<(), Refusal> {
        if self.format != FORMAT {
            return Err(Refusal::Format(self.format.clone()));
        }
        if self.checkpoint != checkpoint {
            return Err(Refusal::Checkpoint { reference: self.checkpoint.clone(), this: checkpoint.into() });
        }
        if self.prompt_sha256 != ids_sha256(prompt) || self.prompt_len != prompt.len() || self.max_new != max_new {
            return Err(Refusal::Input);
        }
        Ok(())
    }

    /// The run this reference stores, for [`first_difference`].
    pub fn as_run(&self) -> Run {
        Run {
            hashes: self.hashes.clone(),
            generation: Generation {
                tokens: self.tokens.clone(),
                // the finish reason is not part of the comparison: the tokens and every hash are
                finish: kt_core::generate::Finish::Length,
                logits_hash: self.logits_hash.clone(),
            },
        }
    }

    /// Compare `run` with this reference: `None` if bit-identical at every position.
    pub fn check(&self, run: &Run) -> Option<Difference> {
        first_difference(&self.as_run(), run)
    }
}

/// A small decoder from a fixed seed, for a check with no download (and for the tests): 2 layers over 64 ids. Its
/// identity is its description, so a reference for it is bound to the same seed.
pub fn synthetic_model(seed: u64) -> (DecoderModel, String) {
    use kt_core::model::{DecoderConfig, DecoderWeights, LayerWeights};
    let cfg = DecoderConfig {
        dim: 32,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        hidden: 64,
        vocab: 64,
        eps: 1e-6,
        rope_theta: 10000.0,
        max_seq: 256,
    };
    let mut s = seed;
    let mut v = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) as f32
            })
            .collect()
    };
    let (dim, qd, kvd, hid, voc) =
        (cfg.dim, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim, cfg.hidden, cfg.vocab);
    let layers = (0..cfg.n_layers)
        .map(|_| LayerWeights {
            input_norm: v(dim),
            wq: v(qd * dim),
            wk: v(kvd * dim),
            wv: v(kvd * dim),
            wo: v(dim * qd),
            wq_b: None,
            wk_b: None,
            wv_b: None,
            post_norm: v(dim),
            gate: v(hid * dim),
            up: v(hid * dim),
            down: v(dim * hid),
        })
        .collect();
    let w = DecoderWeights { embed: v(voc * dim), layers, final_norm: v(dim), lm_head: v(voc * dim), cfg };
    let id = hex(&Sha256::digest(format!("synthetic-decoder/1 seed={seed} dim=32 layers=2 vocab=64").as_bytes()));
    (DecoderModel::build(&w), id)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROMPT: [u32; 9] = [3, 14, 15, 9, 26, 5, 35, 8, 9];

    fn reference(model: &DecoderModel, id: &str, max_new: usize) -> Reference {
        let r = run(model, &PROMPT, max_new, &[], None);
        Reference::new((id.into(), vec![]), "synthetic", "-", &PROMPT, max_new, &r, "test", vec!["portable".into()])
    }

    #[test]
    fn the_run_is_generate_greedy_and_hashes_every_position() {
        let (m, _) = synthetic_model(7);
        let r = run(&m, &PROMPT, 12, &[], None);
        let g = kt_core::generate::generate_greedy(&m, &PROMPT, 12, &[], |_| {});
        assert_eq!(r.generation, g, "the same generation and the same logits_hash");
        // the prompt's positions and every generated token but the last (whose logits nothing reads)
        assert_eq!(r.hashes.len(), PROMPT.len() + 12 - 1);
        assert_eq!(run(&m, &PROMPT, 12, &[], None), r, "the same bits twice");
    }

    #[test]
    fn red_test_a_one_bit_perturbation_is_found_at_its_position() {
        let (m, _) = synthetic_model(7);
        let base = run(&m, &PROMPT, 12, &[], None);
        for pos in [0usize, 4, PROMPT.len() - 1, PROMPT.len() + 5] {
            let p = run(&m, &PROMPT, 12, &[], Some(pos));
            match first_difference(&base, &p) {
                Some(Difference::Position { pos: at, .. }) => assert_eq!(at, pos),
                d => panic!("a perturbation at {pos} reported as {d:?}"),
            }
            assert_eq!(p.generation, base.generation, "the perturbation touches the hash only");
        }
        assert_eq!(first_difference(&base, &base.clone()), None);
    }

    #[test]
    fn the_reference_round_trips_and_refuses_another_model_or_input() {
        let (m, id) = synthetic_model(7);
        let r = reference(&m, &id, 12);
        let text = r.to_json();
        assert_eq!(reference(&m, &id, 12).to_json(), text, "the same run, the same bytes");
        let back = Reference::from_json(&text).unwrap();
        assert_eq!(back, r);
        assert_eq!(back.applies(&id, &PROMPT, 12), Ok(()));
        assert_eq!(back.check(&run(&m, &PROMPT, 12, &[], None)), None, "a fresh run reproduces it");
        // refused: another checkpoint, another prompt, another budget, another format
        let (_, other) = synthetic_model(8);
        assert!(matches!(back.applies(&other, &PROMPT, 12), Err(Refusal::Checkpoint { .. })));
        assert_eq!(back.applies(&id, &PROMPT[..8], 12), Err(Refusal::Input));
        assert_eq!(back.applies(&id, &PROMPT, 11), Err(Refusal::Input));
        let mut f = back.clone();
        f.format = "tritify-repro/0".into();
        assert!(matches!(f.applies(&id, &PROMPT, 12), Err(Refusal::Format(_))));
        // a changed hash fails at exactly its position
        let mut bad = back.clone();
        bad.hashes[5] = "0".repeat(64);
        assert!(matches!(bad.check(&run(&m, &PROMPT, 12, &[], None)), Some(Difference::Position { pos: 5, .. })));
    }

    #[test]
    fn another_seed_is_another_model_and_differs_at_position_zero() {
        let (a, _) = synthetic_model(7);
        let (b, _) = synthetic_model(8);
        assert!(matches!(
            first_difference(&run(&a, &PROMPT, 4, &[], None), &run(&b, &PROMPT, 4, &[], None)),
            Some(Difference::Position { pos: 0, .. })
        ));
    }

    #[test]
    fn the_logits_hash_includes_the_scale() {
        // a real logits tensor, and the same codes under another scale
        let (m, _) = synthetic_model(7);
        let t = m.decode_step(3, &mut m.new_cache(), 0);
        let mut u = t.clone();
        u.scale.mant += 1;
        assert_ne!(logits_hash(&t), logits_hash(&u));
        assert_eq!(kt_core::verify::hash_int(&t), kt_core::verify::hash_int(&u), "hash_int sees the codes only");
    }
}
