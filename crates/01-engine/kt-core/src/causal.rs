//! The causal-LM contract — a plugin's `build` output for autoregressive decoders: tokens → logits.
//!
//! This is the model-level abstraction (as opposed to [`crate::module::Module`], which is a per-activation
//! op). The converter verifies a `CausalLM` by running both forwards on a probe token sequence and comparing
//! the argmax token streams plus the reproducing logits hash.

use crate::tensor::IntTensor;

pub trait CausalLM {
    /// Exact-integer forward: tokens → logits `[seq, vocab]`.
    fn forward_int(&self, tokens: &[usize]) -> IntTensor;
    /// Float reference forward: tokens → logits `[seq, vocab]` (the verification twin).
    fn forward_ref(&self, tokens: &[usize]) -> Vec<f32>;
    fn vocab(&self) -> usize;

    /// Per-position argmax token stream from the integer forward.
    fn argmax_int(&self, tokens: &[usize]) -> Vec<usize> {
        argmax_rows(&self.forward_int(tokens).to_f32(), tokens.len(), self.vocab())
    }
    /// Per-position argmax token stream from the float reference.
    fn argmax_ref(&self, tokens: &[usize]) -> Vec<usize> {
        argmax_rows(&self.forward_ref(tokens), tokens.len(), self.vocab())
    }
    /// Reproducing SHA-256 of the integer logits — identical run-to-run and machine-to-machine.
    fn logits_hash(&self, tokens: &[usize]) -> String {
        crate::verify::hash_int(&self.forward_int(tokens))
    }
}

pub(crate) fn argmax_rows(logits: &[f32], seq: usize, vocab: usize) -> Vec<usize> {
    (0..seq)
        .map(|i| {
            let row = &logits[i * vocab..(i + 1) * vocab];
            row.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (j, &v)| if v > bv { (j, v) } else { (bi, bv) })
                .0
        })
        .collect()
}
