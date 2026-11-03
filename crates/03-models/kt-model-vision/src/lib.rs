//! kt-model-vision — the SAM 2.1 architecture plugin: a zero-float, exact-integer, deterministic port of the
//! **image** segmentation path (Hiera encoder → FPN neck → prompt encoder → two-way mask decoder), built on the
//! generic `kt-core` ops. It reproduces the float SAM 2.1 segmentation mask bit-identically
//! (mask IoU 1.000 vs float, deterministic), in standalone Rust with no PyTorch.
//!
//! Ported faithfully from the `transformers` reference (`modeling_sam2.py`). Every op carries a float twin the
//! verify gate checks it against, and the port was validated module-by-module against the reference model —
//! covering:
//!   - the generic gap ops in `kt-core` — LayerNorm (with mean) + plain SDPA (self/cross)
//!   - the spatial plumbing ([`spatial`]): window partition/unpartition, q-pooling max-pool, CHW↔HWC
//!   - the Hiera image encoder — patch-embed + multi-scale attention (fused QKV, q-pool, windowing) + MLP
//!   - the FPN neck
//!   - the prompt encoder (point/box positional embeddings)
//!   - the two-way mask decoder (self/cross attention + transposed-conv upscale → mask logits)
//!   - the weight loader, producing a bit-identical mask and a reproducing hash on a real image
//!
//! # Example
//!
//! ```
//! use kt_model_vision::Sam2Arch;
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
//! assert!(Sam2Arch.detect(&Cfg("Sam2Model")));
//! assert!(!Sam2Arch.detect(&Cfg("Qwen2ForCausalLM")));
//! ```

pub mod decoder;
pub mod hiera;
pub mod neck;
pub mod nonlin;
pub mod plugin;
pub mod pos;
pub mod preprocess;
pub mod prompt;
pub mod spatial;
pub mod tracker;
pub mod video;

pub use decoder::{FMaskOutput, MaskDecoder, MaskOutput};
pub use hiera::{HieraConfig, HieraEncoder, WeightStore};
pub use neck::VisionNeck;
pub use plugin::{Sam2Arch, SamSegmenter};
pub use pos::{hiera_pos_embed, image_wide_pos};
pub use preprocess::{preprocess_rgb, synthetic_test_image};
pub use prompt::{MaskEmbedding, PromptEncoder};
pub use spatial::{
    chw_to_hwc, hwc_to_chw, max_pool2d, nearest_upsample_2x, window_partition, window_unpartition, Spatial,
};
pub use tracker::{FrameMemory, VideoMemoryBank};
pub use video::{
    rope_tables, sine_pos_table, MemoryAttention, MemoryEncoder, ObjectPointerHead, VideoConfig, VideoRoPEAttention,
};
