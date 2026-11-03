//! Assemble a REAL OLMoE decoder through the generic `MoeArch` plugin and run the verify gate.
//!
//! The full 16-layer × 64-expert OLMoE is tens of GB once quantized, so by default this builds a **bounded** number of
//! layers (memory-safe) to prove the builder reads the real checkpoint correctly: OLMoE detection, QK-norm (real
//! `q_norm`/`k_norm` shapes), router + 64-expert naming, and a verify-gated prefill+decode. Pass `all` to build every
//! layer (slice D — only on a machine with the RAM headroom).
//!
//!   cargo run --release -p kt-convert --example olmoe_build -- <olmoe_dir> [layers=2|all]

use kt_convert::HfModelSource;
use kt_core::plugin::{ModelArch, ModelSource};
use kt_model_moe::MoeArch;

/// Wrap a source and override `num_hidden_layers` so we can assemble a bounded prefix of a large MoE without the full
/// memory cost. Everything else delegates to the real checkpoint (real tensors, real config).
struct CappedLayers<'a> {
    inner: &'a HfModelSource,
    cap: usize,
}
impl<'a> ModelSource for CappedLayers<'a> {
    fn hf_arch_ids(&self) -> Vec<String> {
        self.inner.hf_arch_ids()
    }
    fn config_usize(&self, key: &str) -> Option<usize> {
        if key == "num_hidden_layers" {
            return Some(self.cap);
        }
        self.inner.config_usize(key)
    }
    fn config_f64(&self, key: &str) -> Option<f64> {
        self.inner.config_f64(key)
    }
    fn config_bool(&self, key: &str) -> Option<bool> {
        self.inner.config_bool(key)
    }
    fn tensor(&self, name: &str) -> Option<(Vec<f32>, Vec<usize>)> {
        self.inner.tensor(name)
    }
    fn tensor_names(&self) -> Vec<String> {
        self.inner.tensor_names()
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: olmoe_build <olmoe_dir> [layers=2|all]");
        std::process::exit(2);
    }
    let dir = &a[1];
    let src = HfModelSource::from_dir(dir).unwrap_or_else(|e| {
        eprintln!("load: {e}");
        std::process::exit(1);
    });

    let total = src.config_usize("num_hidden_layers").expect("num_hidden_layers");
    let cap = match a.get(2).map(String::as_str) {
        Some("all") => total,
        Some(n) => n.parse().unwrap_or(2).min(total),
        None => 2.min(total),
    };

    let arch = MoeArch;
    assert!(arch.detect(&src), "MoeArch must detect OLMoE");
    println!("OLMoE detected: {:?}", src.hf_arch_ids());

    let capped = CappedLayers { inner: &src, cap };
    eprintln!("[olmoe_build] assembling {cap}/{total} real OLMoE layers (experts lazy-loaded) …");
    let model = MoeArch::build_decoder_moe(&capped).unwrap_or_else(|e| {
        eprintln!("build: {e}");
        std::process::exit(1);
    });

    // the engine's own self-check: prefill probe int-vs-float + a reproducing integer hash (uses `DecoderModel`'s gate).
    use kt_core::VerifiableModel;
    let rep = model.self_check();
    println!(
        "OLMoE REAL build ({cap} layers): task={} metric={} max_abs_err={:.3e} det={}",
        rep.task, rep.metric, rep.max_abs_err, rep.deterministic
    );
    println!("  reproducing integer logits hash = {}", rep.hash);
    let ok = rep.deterministic && rep.max_abs_err.is_finite();
    println!(
        "  -> {}",
        if ok { "generic MoeArch assembled a REAL OLMoE decoder; verify gate ran (deterministic)." } else { "CHECK" }
    );
    if !ok {
        std::process::exit(1);
    }
}
