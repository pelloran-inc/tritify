//! The activation tensor: exact integer data with one per-tensor **integer** scale (`kt_int::Scale`).
//! `real = data · scale` where the scale is `mant · 2^(-shift)` — carried and composed with integer arithmetic
//! only (no float on the data path). The `to_f32`/`to_f64` helpers materialize reals for the reference and
//! verification paths; the run-time ops never call them.

use kt_int::Scale;

/// An activation carried between ops. Real value of element `i` is `data[i] · scale`.
#[derive(Clone, Debug)]
pub struct IntTensor {
    pub data: Vec<i32>,
    pub shape: Vec<usize>,
    pub scale: Scale,
}

impl IntTensor {
    pub fn new(data: Vec<i32>, shape: Vec<usize>, scale: Scale) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>().max(0));
        IntTensor { data, shape, scale }
    }

    pub fn vector(data: Vec<i32>, scale: Scale) -> Self {
        let n = data.len();
        IntTensor { data, shape: vec![n], scale }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn to_f32(&self) -> Vec<f32> {
        let s = self.scale.to_f64();
        self.data.iter().map(|&q| (q as f64 * s) as f32).collect()
    }

    /// Dequantize to f64 (reference / interchange paths only).
    pub fn to_f64(&self) -> Vec<f64> {
        let s = self.scale.to_f64();
        self.data.iter().map(|&q| q as f64 * s).collect()
    }

    /// Largest magnitude of the integer data (an integer max-reduction).
    pub fn max_abs_data(&self) -> i64 {
        self.data.iter().fold(0i64, |a, &v| a.max((v as i64).abs()))
    }

    /// Serialize to a self-describing little-endian byte buffer: the integer codes, the shape, and the exact
    /// `Scale` (mantissa + shift). This is the **interchange form passed between workers** in a distributed split:
    /// an activation carried as exact integers, so the receiving worker continues the *same* computation and the
    /// final output is bit-identical to a single machine. Machine-independent (integers + fixed widths).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(28 + self.shape.len() * 8 + self.data.len() * 4);
        b.extend_from_slice(&(self.shape.len() as u32).to_le_bytes());
        for &s in &self.shape {
            b.extend_from_slice(&(s as u64).to_le_bytes());
        }
        b.extend_from_slice(&self.scale.mant.to_le_bytes());
        b.extend_from_slice(&self.scale.shift.to_le_bytes());
        b.extend_from_slice(&(self.data.len() as u64).to_le_bytes());
        for &d in &self.data {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b
    }

    /// Reconstruct an [`IntTensor`] from [`to_bytes`](Self::to_bytes) — the exact bytes, no float, so the decoded
    /// tensor is identical to the one serialized.
    pub fn from_bytes(b: &[u8]) -> IntTensor {
        fn u32r(b: &[u8], c: &mut usize) -> u32 {
            let v = u32::from_le_bytes(b[*c..*c + 4].try_into().unwrap());
            *c += 4;
            v
        }
        fn u64r(b: &[u8], c: &mut usize) -> u64 {
            let v = u64::from_le_bytes(b[*c..*c + 8].try_into().unwrap());
            *c += 8;
            v
        }
        fn i64r(b: &[u8], c: &mut usize) -> i64 {
            let v = i64::from_le_bytes(b[*c..*c + 8].try_into().unwrap());
            *c += 8;
            v
        }
        fn i32r(b: &[u8], c: &mut usize) -> i32 {
            let v = i32::from_le_bytes(b[*c..*c + 4].try_into().unwrap());
            *c += 4;
            v
        }
        let mut c = 0usize;
        let nshape = u32r(b, &mut c) as usize;
        let shape: Vec<usize> = (0..nshape).map(|_| u64r(b, &mut c) as usize).collect();
        let (mant, shift) = (i64r(b, &mut c), i32r(b, &mut c));
        let ndata = u64r(b, &mut c) as usize;
        let data: Vec<i32> = (0..ndata).map(|_| i32r(b, &mut c)).collect();
        IntTensor::new(data, shape, Scale { mant, shift })
    }

    /// **Dynamic per-tensor quantization** of real values: `scale = max|real| / levels` (as an integer
    /// `Scale`), `data = round(real / scale)`. Used to bring real inputs into the integer representation at the
    /// model's entry boundary (and by the float reference twins); the integer op data path builds output
    /// `Scale`s directly and never passes through reals.
    pub fn from_reals(real: &[f64], shape: Vec<usize>, levels: i64) -> IntTensor {
        let max = real.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        if max == 0.0 {
            return IntTensor::new(vec![0; real.len()], shape, Scale::ZERO);
        }
        let step = max / levels as f64;
        let scale = Scale::from_f64(step);
        let inv = levels as f64 / max; // = 1/step
        let data = real.iter().map(|&v| (v * inv).round() as i32).collect();
        IntTensor::new(data, shape, scale)
    }

    /// Quantize a probe float vector to an `IntTensor` at a given integer level (dynamic per-tensor).
    pub fn quantize_probe(x: &[f32], levels: i64) -> Self {
        let real: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let n = real.len();
        IntTensor::from_reals(&real, vec![n], levels)
    }
}
