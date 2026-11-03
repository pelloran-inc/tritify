//! How a MoE router scores its experts.
//!
//! A softmax router (OLMoE, Mixtral, Qwen-MoE) ranks experts by logit and weights them by `exp`; that path lives in
//! [`crate::layers::MoeMlp::route`] and is unchanged. A **sigmoid** router (GLM-4.5) scores each expert on its own:
//! `σ(logit)`, and chooses the top-k by `σ(logit) + bias`, where the per-expert bias only steers the choice — the
//! weights are the chosen experts' `σ` without it. In integer form:
//! - the score code is `σ(logit)·2^ROUTER_CODE_BITS`, read from a [`CodeLut`] over `[−SIGMOID_RANGE, SIGMOID_RANGE]`
//! - the bias is coded once at build time, `round(bias·2^ROUTER_CODE_BITS)`
//! - the choice is `code + bias` in `i64`; the ranking and its ties (to the lower index) are the softmax router's

use crate::layers::ROUTER_CODE_BITS;
use crate::tensor::IntTensor;
use kt_int::CodeLut;

/// The sigmoid table spans `[−SIGMOID_RANGE, SIGMOID_RANGE]`; outside it the code is 0 or `2^ROUTER_CODE_BITS`.
pub const SIGMOID_RANGE: f64 = 16.0;
/// Index bits of the sigmoid table (the softmax table's resolution).
pub const SIGMOID_IDX_BITS: u32 = 16;

/// The score function of a MoE router.
pub enum RouterScore {
    /// `exp(logit − max)`, top-k by logit (OLMoE, Mixtral, Qwen-MoE).
    Softmax,
    /// `σ(logit)`, top-k by `σ(logit) + bias` (GLM-4.5).
    Sigmoid(SigmoidScore),
}

/// A sigmoid router's table and its coded selection bias.
pub struct SigmoidScore {
    table: CodeLut,
    bias: Vec<i64>,
    bias_ref: Vec<f64>,
}

impl SigmoidScore {
    /// Code the selection bias (`e_score_correction_bias`, one per expert) and build the table.
    pub fn new(bias: &[f32]) -> Self {
        let k = (1u64 << ROUTER_CODE_BITS) as f64;
        SigmoidScore {
            table: CodeLut::build(
                |x| 1.0 / (1.0 + (-x).exp()),
                -SIGMOID_RANGE,
                SIGMOID_RANGE,
                SIGMOID_IDX_BITS,
                ROUTER_CODE_BITS,
                false,
            ),
            bias: bias.iter().map(|&b| (b as f64 * k).round() as i64).collect(),
            bias_ref: bias.iter().map(|&b| b as f64).collect(),
        }
    }

    /// The number of experts the bias covers.
    pub fn experts(&self) -> usize {
        self.bias.len()
    }

    /// The coded bias, `round(bias·2^ROUTER_CODE_BITS)`.
    pub fn bias_codes(&self) -> &[i64] {
        &self.bias
    }

    /// The score codes `σ(logit)·2^ROUTER_CODE_BITS`, one per expert, from the router's integer logits.
    pub fn codes(&self, logits: &IntTensor) -> Vec<i64> {
        logits.data.iter().map(|&l| self.table.code_at(self.table.index(l as i64, logits.scale))).collect()
    }

    /// The choice keys: score code plus coded bias.
    pub fn choice(&self, codes: &[i64]) -> Vec<i64> {
        codes.iter().zip(&self.bias).map(|(&c, &b)| c + b).collect()
    }

    /// The float reference: `σ(logit)` and the choice `σ(logit) + bias`.
    pub fn reference(&self, logits: &[f32]) -> (Vec<f64>, Vec<f64>) {
        let s: Vec<f64> = logits.iter().map(|&l| 1.0 / (1.0 + (-(l as f64)).exp())).collect();
        let c = s.iter().zip(&self.bias_ref).map(|(&v, &b)| v + b).collect();
        (s, c)
    }
}
