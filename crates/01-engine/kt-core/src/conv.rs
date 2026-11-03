//! 2-D convolution — the one new primitive the vision models (SAM 2.1, ViT patch-embed, CNNs) need beyond the
//! decoder op set. Same discipline as [`crate::layers::QuantLinear`]: weights quantized to b=12, the input
//! dynamically quantized per-tensor by its own max, an **exact integer accumulation**, then an integer `Scale`
//! requant — zero float on the data path. Layout: input `[c_in, H, W]`, weights `[c_out, c_in, kh, kw]`,
//! output `[c_out, OH, OW]`.

use crate::layers::{par_blocks, quant_to, requant_dyn, ACT_LEVELS, WLEVELS};
use crate::tensor::IntTensor;
use kt_int::Scale;

pub struct Conv2d {
    pub name: String,
    pub c_in: usize,
    pub c_out: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub pad: usize,
    w_codes: Vec<i32>, // [c_out, c_in, kh, kw]
    w_scale: Scale,
    bias: Option<(Vec<i64>, Scale)>,
}

impl Conv2d {
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        name: impl Into<String>,
        w: &[f32],
        c_out: usize,
        c_in: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> Self {
        assert_eq!(w.len(), c_out * c_in * kh * kw);
        let peak = w.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
        let (w_scale, w_codes) = if peak == 0.0 {
            (Scale::ZERO, vec![0i32; w.len()])
        } else {
            let step = peak / WLEVELS as f64;
            (
                Scale::from_f64(step),
                w.iter().map(|&v| ((v as f64 / step).round() as i64).clamp(-WLEVELS, WLEVELS) as i32).collect(),
            )
        };
        Conv2d { name: name.into(), c_in, c_out, kh, kw, stride, pad, w_codes, w_scale, bias: None }
    }

    pub fn with_bias(mut self, bias: Option<&[f32]>) -> Self {
        if let Some(b) = bias {
            assert_eq!(b.len(), self.c_out, "{}: bias len mismatch", self.name);
            let bmax = b.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
            if bmax > 0.0 {
                let step = bmax / ACT_LEVELS as f64;
                self.bias =
                    Some((b.iter().map(|&v| (v as f64 / step).round() as i64).collect(), Scale::from_f64(step)));
            }
        }
        self
    }

    pub fn out_dims(&self, h: usize, w: usize) -> (usize, usize) {
        ((h + 2 * self.pad - self.kh) / self.stride + 1, (w + 2 * self.pad - self.kw) / self.stride + 1)
    }

    /// Exact-integer conv (zero float on the data path). `x` is `[c_in, H, W]`.
    pub fn forward_int(&self, x: &IntTensor) -> IntTensor {
        assert_eq!(x.shape.len(), 3, "{}: expected [c_in,H,W]", self.name);
        let (ci, h, w) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(ci, self.c_in);
        let (oh, ow) = self.out_dims(h, w);
        let xmax = x.max_abs_data();
        if xmax == 0 {
            return IntTensor::new(vec![0; self.c_out * oh * ow], vec![self.c_out, oh, ow], Scale::ZERO);
        }
        let qx = quant_to(&x.data, xmax);
        let c = Scale::from_ratio(xmax as u128, WLEVELS as u128).mul(x.scale).mul(self.w_scale);
        if c.mant == 0 {
            return IntTensor::new(vec![0; self.c_out * oh * ow], vec![self.c_out, oh, ow], Scale::ZERO);
        }
        let bias_in_c: Option<Vec<i64>> = self.bias.as_ref().map(|(codes, bs)| {
            let f = bs.mul(c.recip());
            codes.iter().map(|&bc| f.apply(bc)).collect()
        });

        let mut combined = vec![0i64; self.c_out * oh * ow];
        let plane = oh * ow;
        // Output channels are independent → deterministic parallel fill (one disjoint output plane each). Per
        // output element the (ic,ky,kx) accumulation order is unchanged, so the result is bit-identical.
        let fill = |oc: usize, out: &mut [i64]| {
            let wbase = oc * self.c_in * self.kh * self.kw;
            let b = bias_in_c.as_ref().map_or(0, |v| v[oc]);
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut accum = 0i64;
                    for ic in 0..self.c_in {
                        let xplane = ic * h * w;
                        let wplane = wbase + ic * self.kh * self.kw;
                        for ky in 0..self.kh {
                            let iy = oy * self.stride + ky;
                            if iy < self.pad || iy - self.pad >= h {
                                continue;
                            }
                            let iy = iy - self.pad;
                            for kx in 0..self.kw {
                                let ix = ox * self.stride + kx;
                                if ix < self.pad || ix - self.pad >= w {
                                    continue;
                                }
                                let ix = ix - self.pad;
                                accum +=
                                    qx[xplane + iy * w + ix] as i64 * self.w_codes[wplane + ky * self.kw + kx] as i64;
                            }
                        }
                    }
                    out[oy * ow + ox] = accum + b;
                }
            }
        };
        let work =
            self.c_out.saturating_mul(plane).saturating_mul(self.c_in).saturating_mul(self.kh).saturating_mul(self.kw);
        par_blocks(&mut combined, plane, self.c_out, work, fill);
        let gmax = combined.iter().fold(0i64, |m, &z| m.max(z.abs()));
        if gmax == 0 {
            return IntTensor::new(vec![0; self.c_out * oh * ow], vec![self.c_out, oh, ow], Scale::ZERO);
        }
        let out: Vec<i32> = combined.iter().map(|&z| requant_dyn(z, ACT_LEVELS, gmax)).collect();
        IntTensor::new(out, vec![self.c_out, oh, ow], Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(c))
    }

    /// Float reference conv (on the b=12 dequantized weights).
    pub fn forward_ref(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (oh, ow) = self.out_dims(h, w);
        let ws = self.w_scale.to_f64();
        let bias_real: Option<Vec<f64>> = self.bias.as_ref().map(|(codes, bs)| {
            let s = bs.to_f64();
            codes.iter().map(|&c| c as f64 * s).collect()
        });
        let mut out = vec![0f32; self.c_out * oh * ow];
        for oc in 0..self.c_out {
            let wbase = oc * self.c_in * self.kh * self.kw;
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0f64;
                    for ic in 0..self.c_in {
                        let xplane = ic * h * w;
                        let wplane = wbase + ic * self.kh * self.kw;
                        for ky in 0..self.kh {
                            let iy = oy * self.stride + ky;
                            if iy < self.pad || iy - self.pad >= h {
                                continue;
                            }
                            let iy = iy - self.pad;
                            for kx in 0..self.kw {
                                let ix = ox * self.stride + kx;
                                if ix < self.pad || ix - self.pad >= w {
                                    continue;
                                }
                                let ix = ix - self.pad;
                                acc += x[xplane + iy * w + ix] as f64
                                    * (self.w_codes[wplane + ky * self.kw + kx] as f64 * ws);
                            }
                        }
                    }
                    out[oc * oh * ow + oy * ow + ox] = (acc + bias_real.as_ref().map_or(0.0, |b| b[oc])) as f32;
                }
            }
        }
        out
    }
}

/// Transposed 2-D convolution (`nn.ConvTranspose2d`) with `stride == kernel` and no padding — SAM's mask-decoder
/// 2× upscaler (`upscale_conv1`/`upscale_conv2`, kernel=stride=2). With stride==kernel the output blocks don't
/// overlap: `out[oc, s·y+ky, s·x+kx] = Σ_ic x[ic,y,x]·W[ic,oc,ky,kx] (+ bias[oc])`. Weight layout is PyTorch's
/// `[c_in, c_out, kh, kw]`. Same integer discipline as [`Conv2d`]: b=12 weights, dynamic per-tensor input quant,
/// exact integer accumulation, integer `Scale` requant.
pub struct ConvTranspose2d {
    pub name: String,
    pub c_in: usize,
    pub c_out: usize,
    pub k: usize,      // kernel == stride
    w_codes: Vec<i32>, // [c_in, c_out, k, k]
    w_scale: Scale,
    bias: Option<(Vec<i64>, Scale)>,
}

impl ConvTranspose2d {
    pub fn from_weights(name: impl Into<String>, w: &[f32], c_in: usize, c_out: usize, k: usize) -> Self {
        assert_eq!(w.len(), c_in * c_out * k * k);
        let peak = w.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
        let (w_scale, w_codes) = if peak == 0.0 {
            (Scale::ZERO, vec![0i32; w.len()])
        } else {
            let step = peak / WLEVELS as f64;
            (
                Scale::from_f64(step),
                w.iter().map(|&v| ((v as f64 / step).round() as i64).clamp(-WLEVELS, WLEVELS) as i32).collect(),
            )
        };
        ConvTranspose2d { name: name.into(), c_in, c_out, k, w_codes, w_scale, bias: None }
    }

    pub fn with_bias(mut self, bias: Option<&[f32]>) -> Self {
        if let Some(b) = bias {
            assert_eq!(b.len(), self.c_out, "{}: bias len mismatch", self.name);
            let bmax = b.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
            if bmax > 0.0 {
                let step = bmax / ACT_LEVELS as f64;
                self.bias =
                    Some((b.iter().map(|&v| (v as f64 / step).round() as i64).collect(), Scale::from_f64(step)));
            }
        }
        self
    }

    /// `x`: `[c_in, H, W]` → `[c_out, k·H, k·W]`.
    pub fn forward_int(&self, x: &IntTensor) -> IntTensor {
        assert_eq!(x.shape.len(), 3, "{}: expected [c_in,H,W]", self.name);
        let (ci, h, w) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(ci, self.c_in);
        let (oh, ow) = (self.k * h, self.k * w);
        let xmax = x.max_abs_data();
        if xmax == 0 {
            return IntTensor::new(vec![0; self.c_out * oh * ow], vec![self.c_out, oh, ow], Scale::ZERO);
        }
        let qx = quant_to(&x.data, xmax);
        let c = Scale::from_ratio(xmax as u128, WLEVELS as u128).mul(x.scale).mul(self.w_scale);
        if c.mant == 0 {
            return IntTensor::new(vec![0; self.c_out * oh * ow], vec![self.c_out, oh, ow], Scale::ZERO);
        }
        let bias_in_c: Option<Vec<i64>> = self.bias.as_ref().map(|(codes, bs)| {
            let f = bs.mul(c.recip());
            codes.iter().map(|&bc| f.apply(bc)).collect()
        });
        let mut combined = vec![0i64; self.c_out * oh * ow];
        let plane = oh * ow;
        // Output channels are independent → deterministic parallel fill (one disjoint output plane each). The
        // scatter is re-nested with `oc` outermost; per output element the contributions are still summed over
        // `ic` ascending (stride==kernel → no overlap within a plane), so the result is bit-identical.
        let fill = |oc: usize, out: &mut [i64]| {
            for ic in 0..self.c_in {
                let xplane = ic * h * w;
                let wbase = ((ic * self.c_out + oc) * self.k) * self.k;
                for y in 0..h {
                    for x_ in 0..w {
                        let xv = qx[xplane + y * w + x_];
                        if xv == 0 {
                            continue;
                        }
                        for ky in 0..self.k {
                            let oy = self.k * y + ky;
                            for kx in 0..self.k {
                                let ox = self.k * x_ + kx;
                                out[oy * ow + ox] += xv as i64 * self.w_codes[wbase + ky * self.k + kx] as i64;
                            }
                        }
                    }
                }
            }
            if let Some(bic) = &bias_in_c {
                let b = bic[oc];
                for p in out.iter_mut() {
                    *p += b;
                }
            }
        };
        let work = self
            .c_out
            .saturating_mul(self.c_in)
            .saturating_mul(h)
            .saturating_mul(w)
            .saturating_mul(self.k)
            .saturating_mul(self.k);
        par_blocks(&mut combined, plane, self.c_out, work, fill);
        let gmax = combined.iter().fold(0i64, |m, &z| m.max(z.abs()));
        if gmax == 0 {
            return IntTensor::new(vec![0; self.c_out * oh * ow], vec![self.c_out, oh, ow], Scale::ZERO);
        }
        let out: Vec<i32> = combined.iter().map(|&z| requant_dyn(z, ACT_LEVELS, gmax)).collect();
        IntTensor::new(out, vec![self.c_out, oh, ow], Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(c))
    }

    /// Float reference.
    pub fn forward_ref(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (oh, ow) = (self.k * h, self.k * w);
        let ws = self.w_scale.to_f64();
        let bias_real: Option<Vec<f64>> = self.bias.as_ref().map(|(codes, bs)| {
            let s = bs.to_f64();
            codes.iter().map(|&cc| cc as f64 * s).collect()
        });
        let mut out = vec![0f64; self.c_out * oh * ow];
        for ic in 0..self.c_in {
            let xplane = ic * h * w;
            for y in 0..h {
                for x_ in 0..w {
                    let xv = x[xplane + y * w + x_] as f64;
                    for oc in 0..self.c_out {
                        let wbase = ((ic * self.c_out + oc) * self.k) * self.k;
                        let obase = oc * oh * ow;
                        for ky in 0..self.k {
                            for kx in 0..self.k {
                                out[obase + (self.k * y + ky) * ow + self.k * x_ + kx] +=
                                    xv * (self.w_codes[wbase + ky * self.k + kx] as f64 * ws);
                            }
                        }
                    }
                }
            }
        }
        let mut o = vec![0f32; self.c_out * oh * ow];
        for oc in 0..self.c_out {
            let b = bias_real.as_ref().map_or(0.0, |bb| bb[oc]);
            for p in 0..oh * ow {
                o[oc * oh * ow + p] = (out[oc * oh * ow + p] + b) as f32;
            }
        }
        o
    }
}

/// Depthwise 2-D convolution (`nn.Conv2d` with `groups = channels`): each output channel is the convolution of
/// the one matching input channel — weight layout `[C, 1, k, k]`. SAM 2.1's memory fuser uses a 7×7 depthwise
/// conv. Same integer discipline as [`Conv2d`].
pub struct DepthwiseConv2d {
    pub name: String,
    pub c: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    w_codes: Vec<i32>, // [c, k, k]
    w_scale: Scale,
    bias: Option<(Vec<i64>, Scale)>,
}

impl DepthwiseConv2d {
    pub fn from_weights(name: impl Into<String>, w: &[f32], c: usize, k: usize, stride: usize, pad: usize) -> Self {
        assert_eq!(w.len(), c * k * k);
        let peak = w.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
        let (w_scale, w_codes) = if peak == 0.0 {
            (Scale::ZERO, vec![0i32; w.len()])
        } else {
            let step = peak / WLEVELS as f64;
            (
                Scale::from_f64(step),
                w.iter().map(|&v| ((v as f64 / step).round() as i64).clamp(-WLEVELS, WLEVELS) as i32).collect(),
            )
        };
        DepthwiseConv2d { name: name.into(), c, k, stride, pad, w_codes, w_scale, bias: None }
    }

    pub fn with_bias(mut self, bias: Option<&[f32]>) -> Self {
        if let Some(b) = bias {
            assert_eq!(b.len(), self.c, "{}: bias len mismatch", self.name);
            let bmax = b.iter().fold(0.0f64, |a, &v| a.max((v as f64).abs()));
            if bmax > 0.0 {
                let step = bmax / ACT_LEVELS as f64;
                self.bias =
                    Some((b.iter().map(|&v| (v as f64 / step).round() as i64).collect(), Scale::from_f64(step)));
            }
        }
        self
    }

    pub fn out_dims(&self, h: usize, w: usize) -> (usize, usize) {
        ((h + 2 * self.pad - self.k) / self.stride + 1, (w + 2 * self.pad - self.k) / self.stride + 1)
    }

    /// `x`: `[C, H, W]` → `[C, OH, OW]`.
    pub fn forward_int(&self, x: &IntTensor) -> IntTensor {
        let (c, h, w) = (x.shape[0], x.shape[1], x.shape[2]);
        assert_eq!(c, self.c);
        let (oh, ow) = self.out_dims(h, w);
        let xmax = x.max_abs_data();
        if xmax == 0 {
            return IntTensor::new(vec![0; c * oh * ow], vec![c, oh, ow], Scale::ZERO);
        }
        let qx = quant_to(&x.data, xmax);
        let cf = Scale::from_ratio(xmax as u128, WLEVELS as u128).mul(x.scale).mul(self.w_scale);
        if cf.mant == 0 {
            return IntTensor::new(vec![0; c * oh * ow], vec![c, oh, ow], Scale::ZERO);
        }
        let bias_in_c: Option<Vec<i64>> = self.bias.as_ref().map(|(codes, bs)| {
            let f = bs.mul(cf.recip());
            codes.iter().map(|&bc| f.apply(bc)).collect()
        });
        let mut combined = vec![0i64; c * oh * ow];
        let plane = oh * ow;
        // Channels are independent → deterministic parallel fill (one disjoint output plane each). Per output
        // element the (ky,kx) accumulation order is unchanged, so the result is bit-identical.
        let fill = |ch: usize, out: &mut [i64]| {
            let xplane = ch * h * w;
            let wbase = ch * self.k * self.k;
            let b = bias_in_c.as_ref().map_or(0, |v| v[ch]);
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0i64;
                    for ky in 0..self.k {
                        let iy = oy * self.stride + ky;
                        if iy < self.pad || iy - self.pad >= h {
                            continue;
                        }
                        let iy = iy - self.pad;
                        for kx in 0..self.k {
                            let ix = ox * self.stride + kx;
                            if ix < self.pad || ix - self.pad >= w {
                                continue;
                            }
                            let ix = ix - self.pad;
                            acc += qx[xplane + iy * w + ix] as i64 * self.w_codes[wbase + ky * self.k + kx] as i64;
                        }
                    }
                    out[oy * ow + ox] = acc + b;
                }
            }
        };
        let work = c.saturating_mul(plane).saturating_mul(self.k).saturating_mul(self.k);
        par_blocks(&mut combined, plane, c, work, fill);
        let gmax = combined.iter().fold(0i64, |m, &z| m.max(z.abs()));
        if gmax == 0 {
            return IntTensor::new(vec![0; c * oh * ow], vec![c, oh, ow], Scale::ZERO);
        }
        let out: Vec<i32> = combined.iter().map(|&z| requant_dyn(z, ACT_LEVELS, gmax)).collect();
        IntTensor::new(out, vec![c, oh, ow], Scale::from_ratio(gmax as u128, ACT_LEVELS as u128).mul(cf))
    }

    /// Float reference.
    pub fn forward_ref(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (oh, ow) = self.out_dims(h, w);
        let ws = self.w_scale.to_f64();
        let bias_real: Option<Vec<f64>> = self.bias.as_ref().map(|(codes, bs)| {
            let s = bs.to_f64();
            codes.iter().map(|&cc| cc as f64 * s).collect()
        });
        let mut out = vec![0f32; self.c * oh * ow];
        for ch in 0..self.c {
            let xplane = ch * h * w;
            let wbase = ch * self.k * self.k;
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0f64;
                    for ky in 0..self.k {
                        let iy = oy * self.stride + ky;
                        if iy < self.pad || iy - self.pad >= h {
                            continue;
                        }
                        let iy = iy - self.pad;
                        for kx in 0..self.k {
                            let ix = ox * self.stride + kx;
                            if ix < self.pad || ix - self.pad >= w {
                                continue;
                            }
                            let ix = ix - self.pad;
                            acc +=
                                x[xplane + iy * w + ix] as f64 * (self.w_codes[wbase + ky * self.k + kx] as f64 * ws);
                        }
                    }
                    out[ch * oh * ow + oy * ow + ox] = (acc + bias_real.as_ref().map_or(0.0, |b| b[ch])) as f32;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    }
    fn vecf(s: &mut u64, n: usize) -> Vec<f32> {
        (0..n).map(|_| ((lcg(s) >> 40) as f64 / (1u64 << 24) as f64 * 2.0 - 1.0) as f32).collect()
    }

    #[test]
    fn conv2d_tracks_float_and_is_deterministic() {
        let (c_in, h, w, c_out, k, stride, pad) = (3usize, 8usize, 8usize, 4usize, 3usize, 2usize, 1usize);
        let mut s = 0xC0Fu64;
        let wt = vecf(&mut s, c_out * c_in * k * k);
        let b = vecf(&mut s, c_out);
        let xf = vecf(&mut s, c_in * h * w);
        let x = IntTensor::from_reals(&xf.iter().map(|&v| v as f64).collect::<Vec<_>>(), vec![c_in, h, w], ACT_LEVELS);

        let conv = Conv2d::from_weights("conv", &wt, c_out, c_in, k, k, stride, pad).with_bias(Some(&b));
        let oi = conv.forward_int(&x);
        let r = conv.forward_ref(&x.to_f32(), h, w);
        let got = oi.to_f32();
        let (mut mae, mut peak) = (0.0f64, 0.0f64);
        for (&g, &rf) in got.iter().zip(&r) {
            mae = mae.max((g as f64 - rf as f64).abs());
            peak = peak.max((rf as f64).abs());
        }
        println!("Conv2d: out={:?} max_abs_err={:.3e} (peak_ref={:.3e})", oi.shape, mae, peak);
        assert_eq!(oi.data, conv.forward_int(&x).data, "deterministic");
        assert!(mae <= peak * 2e-2, "conv2d drifted from float");
    }
}
