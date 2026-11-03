//! The **task-agnostic** output of a plugin's `build`: a runnable model the converter can verify without knowing
//! what task it performs. This is what makes the converter generic — a decoder LM (tokens → logits) and an image
//! segmenter (image+prompt → mask) are built by different plugins, but both hand back a [`VerifiableModel`], and
//! the converter does the same thing to each: ask it to self-check and print the report + reproducing hash.
//!
//! A model self-checks by running its own deterministic built-in probe through the integer path (and, where it
//! has an internal float twin, the float reference), comparing, and hashing. The task-specific input and output
//! never leak into the converter.

/// A task-appropriate verification result. `metric` is normalized so `1.0` = perfect agreement for the task.
#[derive(Clone, Debug)]
pub struct CheckReport {
    pub task: String,        // e.g. "causal-lm", "image-segmentation"
    pub metric_name: String, // e.g. "argmax_agreement", "deterministic"
    pub metric: f64,         // 1.0 = perfect
    pub max_abs_err: f64,    // integer vs float, where the model has an internal float twin (else 0.0)
    pub deterministic: bool, // same integer output run-to-run
    pub hash: String,        // reproducing SHA-256 of the integer output
    pub detail: String,      // one-line human summary
}

impl CheckReport {
    /// A one-line summary for the converter to print.
    pub fn summary(&self) -> String {
        format!(
            "[{}] {}={:.4} max_abs_err={:.3e} deterministic={} | {} | hash={}",
            self.task,
            self.metric_name,
            self.metric,
            self.max_abs_err,
            self.deterministic,
            self.detail,
            &self.hash[..self.hash.len().min(16)]
        )
    }
}

/// A built, runnable model that can verify itself. Every plugin's `build` returns one; the converter only ever
/// calls [`VerifiableModel::self_check`].
pub trait VerifiableModel {
    /// A short task tag ("causal-lm", "image-segmentation", …).
    fn task(&self) -> &str;
    /// Run the built-in probe through the integer path (+ float twin where available), compare, hash.
    fn self_check(&self) -> CheckReport;
    /// The decision agreement with the float twin over a probe of `positions` positions: `(agreement, detail)`.
    /// It is a separate, longer measurement for a binding fidelity threshold; [`VerifiableModel::self_check`] keeps
    /// its own probe, whose hash is the reproducing one. `None` when the model has no such probe; then the
    /// self-check's metric is the one to bind.
    fn fidelity(&self, _positions: usize) -> Option<(f64, String)> {
        None
    }
}
