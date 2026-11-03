//! Text generation and embeddings, in integers. Generation is greedy, through one canonical path:
//! - the prompt is fed token by token through [`DecoderModel::decode_step`] (the KV-cached path)
//! - each next token is the argmax of the integer logits, the lowest index winning a tie
//! - it stops on an end-of-sequence id or at `max_new`
//!
//! Every step's logits go into a SHA-256 (`logits_hash`). That is a stronger fingerprint than the tokens: two runs
//! agree on it only if every logit of every step is the same integer. An embedding is the mean over positions of
//! the final-normed hidden state, rounded half away from zero, at the hidden state's own scale.
//!
//! # Example
//!
//! ```no_run
//! # fn model() -> kt_core::DecoderModel { unimplemented!() }
//! use kt_core::generate::{generate_greedy, Finish};
//!
//! let m = model(); // any DecoderModel, e.g. built by kt-convert
//! let g = generate_greedy(&m, &[1, 2, 3], 16, &[0], |_token| {});
//! assert!(g.tokens.len() <= 16);
//! assert!(g.finish == Finish::Stop || g.tokens.len() == 16);
//! assert_eq!(g.logits_hash.len(), 64);
//! ```

use crate::model::DecoderModel;
use crate::tensor::IntTensor;
use sha2::{Digest, Sha256};

/// Why generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// An end-of-sequence id came up (it is not part of `tokens`).
    Stop,
    /// `max_new` tokens were produced.
    Length,
}

/// The result of a greedy generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generation {
    pub tokens: Vec<u32>,
    pub finish: Finish,
    /// SHA-256 over every step's logits that chose a token (or the end-of-sequence id): per step the scale
    /// (`mant` as `i64`, `shift` as `i32`), then every logit code as `i32`, all little-endian.
    pub logits_hash: String,
}

/// The index of the largest code; the lowest index wins a tie. A tensor has one scale, so comparing codes is comparing
/// the logits themselves.
pub fn argmax_codes(codes: &[i32]) -> usize {
    let mut best = 0;
    for (i, &v) in codes.iter().enumerate() {
        if v > codes[best] {
            best = i;
        }
    }
    best
}

fn hash_step(h: &mut Sha256, t: &IntTensor) {
    h.update(t.scale.mant.to_le_bytes());
    h.update(t.scale.shift.to_le_bytes());
    for &v in &t.data {
        h.update(v.to_le_bytes());
    }
}

/// Generate up to `max_new` tokens after `prompt` (which must not be empty), stopping early on any id in `stop`.
/// `on_token` sees each new token as it is produced, for streaming.
pub fn generate_greedy(
    model: &DecoderModel,
    prompt: &[u32],
    max_new: usize,
    stop: &[u32],
    on_token: impl FnMut(u32),
) -> Generation {
    let mut cache = model.new_cache();
    generate_greedy_by(prompt, max_new, stop, on_token, |t, pos| model.decode_step(t, &mut cache, pos))
}

/// [`generate_greedy`] with the decode step given: `step(token, pos)` feeds `token` at position `pos` and returns that
/// position's logits (for example, a step that computes the routed experts elsewhere). The prompt is fed position by
/// position, then each chosen token, exactly as `generate_greedy` does.
pub fn generate_greedy_by(
    prompt: &[u32],
    max_new: usize,
    stop: &[u32],
    mut on_token: impl FnMut(u32),
    mut step: impl FnMut(usize, usize) -> IntTensor,
) -> Generation {
    assert!(!prompt.is_empty(), "generate: the prompt is empty");
    let mut logits = None;
    for (pos, &t) in prompt.iter().enumerate() {
        logits = Some(step(t as usize, pos));
    }
    let (mut h, mut tokens, mut finish) = (Sha256::new(), Vec::new(), Finish::Length);
    let mut pos = prompt.len();
    while tokens.len() < max_new {
        let l = logits.take().expect("logits of the last position");
        hash_step(&mut h, &l);
        let next = argmax_codes(&l.data) as u32;
        if stop.contains(&next) {
            finish = Finish::Stop;
            break;
        }
        tokens.push(next);
        on_token(next);
        if tokens.len() < max_new {
            logits = Some(step(next as usize, pos));
            pos += 1;
        }
    }
    Generation { tokens, finish, logits_hash: h.finalize().iter().map(|b| format!("{b:02x}")).collect() }
}

/// The embedding of `tokens`: the mean over positions of the final-normed hidden state (`[dim]`, same scale).
pub fn embed_mean(model: &DecoderModel, tokens: &[usize]) -> IntTensor {
    assert!(!tokens.is_empty(), "embed: no tokens");
    let hidden = model.hidden_int(tokens);
    let (seq, dim) = (tokens.len(), model.dim());
    let n = seq as i64;
    let data = (0..dim)
        .map(|c| {
            let sum: i64 = (0..seq).map(|r| hidden.data[r * dim + c] as i64).sum();
            (if sum >= 0 { (sum + n / 2) / n } else { -((-sum + n / 2) / n) }) as i32
        })
        .collect();
    IntTensor::new(data, vec![dim], hidden.scale)
}
