//! The op contract: every op is a [`Module`] with two paths — the integer run-time and the float reference.

use crate::layers::MoeMlp;
use crate::tensor::IntTensor;

/// A single op in the engine. The two paths are the heart of the design:
/// * `forward_int` is the **zero-float run-time** (what ships), consuming/producing [`IntTensor`].
/// * `forward_ref` is the **float reference** on the same (dequantized) inputs — what the [`crate::verify()`]
///   gate checks the integer path against. An op that cannot state its own reference cannot ship.
pub trait Module {
    fn name(&self) -> &str;

    /// Zero-float integer run-time path.
    fn forward_int(&self, x: &IntTensor) -> IntTensor;

    /// Float reference path on dequantized input — the verification twin, never used at run time.
    fn forward_ref(&self, x: &[f32]) -> Vec<f32>;

    /// Optional observability hook: for a routed op (MoE), return the selected expert indices **per token** for the
    /// input `x` (a `[rows·dim]` sequence → `rows` lists). Dense ops return `None`. Read-only; used only to trace
    /// which experts fire where — never on the run-time data path.
    fn route_trace(&self, _x: &IntTensor) -> Option<Vec<Vec<usize>>> {
        None
    }

    /// Distributed-expert seam: a routed op (MoE) returns a reference to itself so a distributed forward can drive
    /// `route` / expert-compute / `combine` with a remote expert backend. Dense ops return `None` (they run locally).
    /// Read-only; it changes nothing about `forward_int`.
    fn as_moe(&self) -> Option<&MoeMlp> {
        None
    }
}

/// A straight-line stack of modules (the common case inside a block). Non-linear structure (residuals,
/// attention) is expressed by composing `Sequential`s inside a plugin's own block type.
pub struct Sequential {
    pub name: String,
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(name: impl Into<String>) -> Self {
        Sequential { name: name.into(), layers: Vec::new() }
    }

    pub fn push(mut self, m: Box<dyn Module>) -> Self {
        self.layers.push(m);
        self
    }
}

impl Module for Sequential {
    fn name(&self) -> &str {
        &self.name
    }

    fn forward_int(&self, x: &IntTensor) -> IntTensor {
        let mut cur = x.clone();
        for l in &self.layers {
            cur = l.forward_int(&cur);
        }
        cur
    }

    fn forward_ref(&self, x: &[f32]) -> Vec<f32> {
        let mut cur = x.to_vec();
        for l in &self.layers {
            cur = l.forward_ref(&cur);
        }
        cur
    }
}
