//! The verification gate — the engine's conscience. It runs a module's integer path and its float reference
//! on the same probe input, reports how far apart they are, hashes the integer output, and confirms the
//! integer path is deterministic (same bytes when run again). A conversion only ships if this gate passes.

use crate::module::Module;
use crate::tensor::IntTensor;
use sha2::{Digest, Sha256};

/// Outcome of checking one module (or a whole model) against its float reference.
#[derive(Clone, Debug)]
pub struct VerifyReport {
    pub name: String,
    /// Max absolute difference between the integer path (dequantized) and the float reference.
    pub max_abs_err: f64,
    /// Max relative difference (against the reference magnitude).
    pub max_rel_err: f64,
    /// SHA-256 of the integer output words — the reproducing hash.
    pub output_hash: String,
    /// The integer path produced byte-identical output on a repeat run.
    pub deterministic: bool,
    /// Why the two outputs could not be compared, when they could not: their lengths differ, the integer output's
    /// length does not match its shape, or a value is NaN or infinite. Then no error was computed (both are NaN).
    pub refused: Option<String>,
}

impl VerifyReport {
    /// A PASS at a declared tolerance: the outputs were comparable, the integer path tracks float within `tol`, and
    /// it is deterministic. A refused comparison never passes.
    pub fn passes(&self, tol: f64) -> bool {
        self.refused.is_none() && self.deterministic && self.max_abs_err <= tol
    }
}

/// SHA-256 of an integer tensor's words (little-endian) — machine-independent because the data are integers.
pub fn hash_int(t: &IntTensor) -> String {
    let mut h = Sha256::new();
    for &v in &t.data {
        h.update(v.to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{:02x}", b)).collect()
}

/// Run `m.forward_int` vs `m.forward_ref` on `probe` and report the gap, the hash, and determinism.
pub fn verify(m: &dyn Module, probe: &IntTensor) -> VerifyReport {
    let out = m.forward_int(probe);
    let out2 = m.forward_int(probe);
    let deterministic = out.data == out2.data && out.scale == out2.scale;

    let got = out.to_f32();
    let reference = m.forward_ref(&probe.to_f32());
    let refused = comparable(&out, &got, &reference).err();
    let (mut max_abs, mut max_rel) = (0.0f64, 0.0f64);
    if refused.is_some() {
        (max_abs, max_rel) = (f64::NAN, f64::NAN);
    } else {
        for (&g, &r) in got.iter().zip(&reference) {
            let d = (g as f64 - r as f64).abs();
            max_abs = max_abs.max(d);
            if r != 0.0 {
                max_rel = max_rel.max(d / (r as f64).abs());
            }
        }
    }

    VerifyReport {
        name: m.name().to_string(),
        max_abs_err: max_abs,
        max_rel_err: max_rel,
        output_hash: hash_int(&out),
        deterministic,
        refused,
    }
}

/// Whether the integer output and the float reference can be compared value for value: equal lengths, the integer
/// output's length matching its shape, and every value finite.
fn comparable(out: &IntTensor, got: &[f32], reference: &[f32]) -> Result<(), String> {
    let n: usize = out.shape.iter().product();
    if out.data.len() != n {
        return Err(format!("the integer output has {} values for its shape {:?} ({n})", out.data.len(), out.shape));
    }
    if got.len() != reference.len() {
        return Err(format!("the integer output has {} values, the float reference {}", got.len(), reference.len()));
    }
    if let Some(i) = reference.iter().position(|r| !r.is_finite()) {
        return Err(format!("the float reference is not finite at {i}: {}", reference[i]));
    }
    if let Some(i) = got.iter().position(|g| !g.is_finite()) {
        return Err(format!("the dequantized integer output is not finite at {i}: {}", got[i]));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kt_int::Scale;

    /// A module whose integer output and float reference are given, so a test can make them disagree in shape.
    struct Fixed {
        out: IntTensor,
        reference: Vec<f32>,
    }
    impl Module for Fixed {
        fn name(&self) -> &str {
            "fixed"
        }
        fn forward_int(&self, _x: &IntTensor) -> IntTensor {
            self.out.clone()
        }
        fn forward_ref(&self, _x: &[f32]) -> Vec<f32> {
            self.reference.clone()
        }
    }

    fn run(out: IntTensor, reference: Vec<f32>) -> VerifyReport {
        let probe = IntTensor { data: vec![1], shape: vec![1], scale: Scale::from_f64(1.0) };
        verify(&Fixed { out, reference }, &probe)
    }
    fn ints(data: Vec<i32>) -> IntTensor {
        let n = data.len();
        IntTensor { data, shape: vec![n], scale: Scale::from_f64(1.0) }
    }

    #[test]
    fn comparable_outputs_are_compared() {
        let r = run(ints(vec![1, 2, 3]), vec![1.0, 2.0, 3.5]);
        assert_eq!(r.refused, None);
        assert_eq!(r.max_abs_err, 0.5);
        assert!(r.passes(0.5) && !r.passes(0.4));
    }

    #[test]
    fn a_missing_value_is_refused() {
        let r = run(ints(vec![1, 2]), vec![1.0, 2.0, 3.0]);
        assert!(r.refused.as_deref().unwrap().contains("2 values, the float reference 3"), "{:?}", r.refused);
        assert!(r.max_abs_err.is_nan() && !r.passes(f64::INFINITY));
    }

    #[test]
    fn an_extra_value_is_refused() {
        let r = run(ints(vec![1, 2, 3, 4]), vec![1.0, 2.0, 3.0]);
        assert!(r.refused.as_deref().unwrap().contains("4 values, the float reference 3"), "{:?}", r.refused);
        assert!(!r.passes(f64::INFINITY));
    }

    #[test]
    fn an_output_whose_length_does_not_match_its_shape_is_refused() {
        let out = IntTensor { data: vec![1, 2, 3], shape: vec![2, 2], scale: Scale::from_f64(1.0) };
        let r = run(out, vec![1.0, 2.0, 3.0]);
        assert!(r.refused.as_deref().unwrap().contains("for its shape"), "{:?}", r.refused);
        assert!(!r.passes(f64::INFINITY));
    }

    #[test]
    fn a_nan_reference_is_refused() {
        let r = run(ints(vec![1, 2, 3]), vec![1.0, f32::NAN, 3.0]);
        assert!(r.refused.as_deref().unwrap().contains("not finite at 1"), "{:?}", r.refused);
        assert!(!r.passes(f64::INFINITY));
    }

    #[test]
    fn an_infinite_reference_is_refused() {
        let r = run(ints(vec![1, 2, 3]), vec![1.0, 2.0, f32::NEG_INFINITY]);
        assert!(r.refused.as_deref().unwrap().contains("not finite at 2"), "{:?}", r.refused);
        assert!(!r.passes(f64::INFINITY));
    }
}
