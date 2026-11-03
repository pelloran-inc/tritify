//! kt-core — the generic exact-integer inference engine and the architecture-plugin contract.
//!
//! The converter is deliberately split so the community can extend it without touching the engine:
//!
//! ```text
//!   ┌─────────────────────────────────────────────────────────────────────────┐
//!   │  kt-core  (this crate) — GENERIC, model-agnostic                          │
//!   │    • IntTensor            exact-integer activations with an implicit scale │
//!   │    • Module               an op with BOTH an integer run-time path and a   │
//!   │                           float reference path (the verification twin)     │
//!   │    • layers::*            the generic op library (QuantLinear, …)          │
//!   │    • verify               the bit-identity / tolerance gate + output hash  │
//!   │    • ModelArch + Registry THE PLUGIN CONTRACT: map an HF family → a graph  │
//!   └─────────────────────────────────────────────────────────────────────────┘
//!                 ▲                                   ▲
//!     implements ModelArch                implements ModelArch
//!   ┌───────────────────────┐          ┌───────────────────────┐
//!   │ kt-model-decoder      │          │ kt-model-<yours>      │   ← new model groups
//!   │  Llama/Qwen/Mistral…  │          │  a new family, no core │
//!   └───────────────────────┘          │  changes required      │
//!                                       └───────────────────────┘
//! ```
//!
//! A plugin's ONLY job is to recognise a model family ([`ModelArch::detect`]) and emit the generic op graph
//! for it ([`ModelArch::build`]). The engine then runs the integer forward, runs the float reference, and the
//! [`verify()`] gate decides PASS/FAIL (bit-identical outputs, or a declared tolerance) and prints the
//! reproducing hash. New architectures are new plugin crates — the engine never changes.
//!
//! Every [`Module`] carries two paths on purpose: `forward_int` (the zero-float run-time) and `forward_ref`
//! (float, the reference). That pairing is what makes the whole engine self-verifying and keeps us honest:
//! nothing ships that the gate has not checked against float.
//!
//! # Example
//!
//! ```
//! use kt_core::layers::{QuantLinear, ACT_LEVELS};
//! use kt_core::{verify, IntTensor};
//!
//! // An op with an integer run-time path and a float reference twin…
//! let (out, inp) = (4, 8);
//! let w: Vec<f32> = (0..out * inp).map(|i| ((i % 7) as f32 - 3.0) / 10.0).collect();
//! let layer = QuantLinear::from_weights("lin", &w, out, inp);
//! let x: Vec<f64> = (0..inp).map(|i| (i as f64 - 4.0) / 8.0).collect();
//! let probe = IntTensor::from_reals(&x, vec![1, inp], ACT_LEVELS);
//! // …and the verify gate that checks one against the other and hashes the integer output.
//! let report = verify(&layer, &probe);
//! assert!(report.deterministic);
//! assert!(report.passes(1e-2));
//! assert_eq!(report.output_hash.len(), 64);
//! ```

pub mod attention;
pub mod backend;
pub mod causal;
pub mod check;
pub mod conv;
pub mod generate;
pub mod layers;
pub mod model;
pub mod module;
pub mod plugin;
pub mod router;
pub mod sdpa;
pub mod tensor;
pub mod verify;

pub use causal::CausalLM;
pub use check::{CheckReport, VerifiableModel};
pub use layers::{max_threads, set_max_threads};
pub use model::{DecoderConfig, DecoderLayer, DecoderModel, DecoderWeights, LayerWeights};
pub use module::{Module, Sequential};
pub use plugin::{ArchMeta, LayerSpec, ModelArch, ModelPlan, ModelSource, Registry};
pub use sdpa::{SdpaAttention, SdpaCore};
pub use tensor::IntTensor;
pub use verify::{verify, VerifyReport};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_generate;
#[cfg(test)]
mod tests_glm;
#[cfg(test)]
mod tests_partition;
