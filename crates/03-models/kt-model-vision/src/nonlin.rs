//! Pointwise nonlinearities shared by the SAM modules: exact-erf GELU (as an integer-code LUT) and ReLU (exact
//! on the integer grid — `max(0,·)` on codes is `max(0,·)` on reals since the scale is positive).

use kt_core::IntTensor;
use kt_int::CodeLut;

/// erf via Abramowitz & Stegun 7.1.26 (|err| ≤ 1.5e-7).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}
/// Exact GELU: `0.5·x·(1+erf(x/√2))` (nn.GELU()).
pub fn gelu_f64(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Build the GELU integer-code LUT: left tail → 0, right tail → identity (GELU(x)=x for large x).
pub fn build_gelu_lut() -> CodeLut {
    CodeLut::build(gelu_f64, -10.0, 10.0, 16, 20, true)
}

/// Apply a GELU code-LUT to an activation → a new activation at the LUT's code scale (zero float on the data
/// path: a table read gives the integer code; the right tail is the integer identity factor).
pub fn apply_gelu(lut: &CodeLut, x: &IntTensor) -> IntTensor {
    let n = lut.n() as i64;
    let id_factor = x.scale.mul(lut.code_scale.recip());
    let mut g = vec![0i32; x.len()];
    for (gi, &xd) in g.iter_mut().zip(x.data.iter()) {
        let d = xd as i64;
        let idx = lut.index(d, x.scale);
        let code = if idx < 0 {
            0
        } else if idx >= n {
            id_factor.apply(d)
        } else {
            lut.code_at(idx)
        };
        *gi = code as i32;
    }
    IntTensor::new(g, x.shape.clone(), lut.code_scale)
}

/// ReLU — exact on the integer grid, scale unchanged.
pub fn relu(x: &IntTensor) -> IntTensor {
    IntTensor::new(x.data.iter().map(|&d| d.max(0)).collect(), x.shape.clone(), x.scale)
}
