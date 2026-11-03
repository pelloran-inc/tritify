//! Spatial tensor plumbing for the Hiera image encoder — the exact reshape/pool operations SAM 2.1 uses, ported
//! faithfully from `transformers`' `modeling_sam2.py` (`window_partition`, `window_unpartition`, `do_pool`,
//! and the patch-embed channels-first→last permute).
//!
//! These are **value-preserving index operations** (they rearrange or select, they do not do arithmetic), so
//! they carry the tensor's [`Scale`] through unchanged. Max-pool is exact: every element of a tensor shares one
//! scale, so the max code equals the max real. They are the highest-risk part of the port (a single transposed
//! axis silently breaks bit-identity), so each is covered by an exact round-trip / known-value unit test below.

use kt_core::IntTensor;
use kt_int::Scale;

/// A feature map in **channels-last** layout `[B, H, W, C]`, flat row-major (B outer → H → W → C inner), which is
/// how the Hiera encoder carries activations (`modeling_sam2.py` keeps hidden states as `(B, H, W, C)`).
#[derive(Clone)]
pub struct Spatial {
    pub data: Vec<i32>,
    pub scale: Scale,
    pub b: usize,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}

impl Spatial {
    pub fn new(data: Vec<i32>, scale: Scale, b: usize, h: usize, w: usize, c: usize) -> Self {
        assert_eq!(data.len(), b * h * w * c, "Spatial: data len != b·h·w·c");
        Spatial { data, scale, b, h, w, c }
    }

    /// View the `[B·H·W, C]` token matrix as an [`IntTensor`] (rows = tokens, cols = channels) for the generic
    /// ops (LayerNorm, Linear, attention) which work row-wise over the last dim.
    pub fn as_tokens(&self) -> IntTensor {
        IntTensor::new(self.data.clone(), vec![self.b * self.h * self.w, self.c], self.scale)
    }

    /// Rebuild a `Spatial` from a `[B·H·W, C]` token [`IntTensor`] with the given spatial dims.
    pub fn from_tokens(t: &IntTensor, b: usize, h: usize, w: usize, c: usize) -> Self {
        assert_eq!(t.data.len(), b * h * w * c, "from_tokens: len mismatch");
        Spatial { data: t.data.clone(), scale: t.scale, b, h, w, c }
    }
}

// ───────── generic raw reshape ops (shared by the integer `Spatial` path and the f32 float-twin path) ─────────
// These are value-preserving index permutations/selections — identical for any element type. The integer path
// (below) wraps them with a `Scale`; the float twin calls them directly on `f32`.

/// `[C,H,W]` → `[H,W,C]` (channels-first → channels-last).
pub fn chw_to_hwc_raw<T: Copy + Default>(d: &[T], c: usize, h: usize, w: usize) -> Vec<T> {
    let mut out = vec![T::default(); c * h * w];
    for ci in 0..c {
        for y in 0..h {
            for x in 0..w {
                out[(y * w + x) * c + ci] = d[(ci * h + y) * w + x];
            }
        }
    }
    out
}

/// `[H,W,C]` → `[C,H,W]` (channels-last → channels-first).
pub fn hwc_to_chw_raw<T: Copy + Default>(d: &[T], h: usize, w: usize, c: usize) -> Vec<T> {
    let mut out = vec![T::default(); c * h * w];
    for y in 0..h {
        for x in 0..w {
            for ci in 0..c {
                out[(ci * h + y) * w + x] = d[(y * w + x) * c + ci];
            }
        }
    }
    out
}

/// Nearest-neighbour 2× upsample of `[C,H,W]` → `[C,2H,2W]`.
pub fn nearest_up2x_raw<T: Copy + Default>(d: &[T], c: usize, h: usize, w: usize) -> Vec<T> {
    let (oh, ow) = (h * 2, w * 2);
    let mut out = vec![T::default(); c * oh * ow];
    for ci in 0..c {
        for y in 0..h {
            for x in 0..w {
                let v = d[(ci * h + y) * w + x];
                for dy in 0..2 {
                    for dx in 0..2 {
                        out[(ci * oh + 2 * y + dy) * ow + 2 * x + dx] = v;
                    }
                }
            }
        }
    }
    out
}

/// Partition `[B,H,W,C]` into `win×win` windows (zero-padding H,W to a multiple of `win`). Returns
/// `(windows [bo,win,win,C], bo, (padded_h, padded_w))`.
pub fn window_partition_raw<T: Copy + Default>(
    d: &[T],
    b: usize,
    h: usize,
    w: usize,
    c: usize,
    win: usize,
) -> (Vec<T>, usize, (usize, usize)) {
    let (ph, pw) = ((win - h % win) % win, (win - w % win) % win);
    let (hp, wp) = (h + ph, w + pw);
    let (nwh, nww) = (hp / win, wp / win);
    let bo = b * nwh * nww;
    let mut out = vec![T::default(); bo * win * win * c];
    for bi in 0..b {
        for wh in 0..nwh {
            for ww in 0..nww {
                let wi = (bi * nwh + wh) * nww + ww;
                for wy in 0..win {
                    for wx in 0..win {
                        let (sy, sx) = (wh * win + wy, ww * win + wx);
                        if sy < h && sx < w {
                            let dst = ((wi * win + wy) * win + wx) * c;
                            let src = ((bi * h + sy) * w + sx) * c;
                            out[dst..dst + c].copy_from_slice(&d[src..src + c]);
                        }
                    }
                }
            }
        }
    }
    (out, bo, (hp, wp))
}

/// Inverse of [`window_partition_raw`]: windows `[bo,win,win,C]` → `[b,h,w,C]` (crops the padding).
pub fn window_unpartition_raw<T: Copy + Default>(
    d: &[T],
    bo: usize,
    win: usize,
    pad_hw: (usize, usize),
    hw: (usize, usize),
    c: usize,
) -> Vec<T> {
    let (hp, wp) = pad_hw;
    let (h, w) = hw;
    let (nwh, nww) = (hp / win, wp / win);
    let b = bo / (nwh * nww);
    let mut out = vec![T::default(); b * h * w * c];
    for bi in 0..b {
        for wh in 0..nwh {
            for ww in 0..nww {
                let wi = (bi * nwh + wh) * nww + ww;
                for wy in 0..win {
                    for wx in 0..win {
                        let (y, x) = (wh * win + wy, ww * win + wx);
                        if y < h && x < w {
                            let dst = ((bi * h + y) * w + x) * c;
                            let src = ((wi * win + wy) * win + wx) * c;
                            out[dst..dst + c].copy_from_slice(&d[src..src + c]);
                        }
                    }
                }
            }
        }
    }
    out
}

/// Channel-wise max-pool `[B,H,W,C]`, `kernel=stride=stride`, `ceil_mode=False`. Returns `(out, oh, ow)`.
pub fn max_pool2d_raw<T: Copy + Default + PartialOrd>(
    d: &[T],
    b: usize,
    h: usize,
    w: usize,
    c: usize,
    stride: usize,
) -> (Vec<T>, usize, usize) {
    let (oh, ow) = (h / stride, w / stride);
    let mut out = vec![T::default(); b * oh * ow * c];
    for bi in 0..b {
        for oy in 0..oh {
            for ox in 0..ow {
                for ci in 0..c {
                    let mut m = d[((bi * h + oy * stride) * w + ox * stride) * c + ci];
                    for ky in 0..stride {
                        for kx in 0..stride {
                            let v = d[((bi * h + oy * stride + ky) * w + ox * stride + kx) * c + ci];
                            if v > m {
                                m = v;
                            }
                        }
                    }
                    out[((bi * oh + oy) * ow + ox) * c + ci] = m;
                }
            }
        }
    }
    (out, oh, ow)
}

// ───────── integer `Spatial` wrappers (carry the `Scale`) ─────────

/// `nn.Conv2d` output is channels-first `[C, H, W]`; the Hiera patch-embed `permute(0,2,3,1)` → `[1, H, W, C]`.
pub fn chw_to_hwc(t: &IntTensor) -> Spatial {
    assert_eq!(t.shape.len(), 3, "chw_to_hwc expects [C,H,W]");
    let (c, h, w) = (t.shape[0], t.shape[1], t.shape[2]);
    Spatial::new(chw_to_hwc_raw(&t.data, c, h, w), t.scale, 1, h, w, c)
}

/// Inverse of [`chw_to_hwc`]: a single-image `Spatial[1,H,W,C]` → channels-first `[C,H,W]` [`IntTensor`].
pub fn hwc_to_chw(sp: &Spatial) -> IntTensor {
    assert_eq!(sp.b, 1, "hwc_to_chw expects a single image (B=1)");
    IntTensor::new(hwc_to_chw_raw(&sp.data, sp.h, sp.w, sp.c), vec![sp.c, sp.h, sp.w], sp.scale)
}

/// Nearest-neighbour 2× upsample of a channels-first `[C,H,W]` tensor → `[C,2H,2W]`.
pub fn nearest_upsample_2x(t: &IntTensor) -> IntTensor {
    assert_eq!(t.shape.len(), 3, "nearest_upsample_2x expects [C,H,W]");
    let (c, h, w) = (t.shape[0], t.shape[1], t.shape[2]);
    IntTensor::new(nearest_up2x_raw(&t.data, c, h, w), vec![c, h * 2, w * 2], t.scale)
}

/// Partition into `window_size × window_size` windows; returns the windows and the padded `(h,w)`.
pub fn window_partition(sp: &Spatial, window_size: usize) -> (Spatial, (usize, usize)) {
    let (out, bo, pad) = window_partition_raw(&sp.data, sp.b, sp.h, sp.w, sp.c, window_size);
    (Spatial::new(out, sp.scale, bo, window_size, window_size, sp.c), pad)
}

/// Inverse of [`window_partition`]: reassemble and crop the padding.
pub fn window_unpartition(
    windows: &Spatial,
    window_size: usize,
    pad_hw: (usize, usize),
    hw: (usize, usize),
) -> Spatial {
    let out = window_unpartition_raw(&windows.data, windows.b, window_size, pad_hw, hw, windows.c);
    Spatial::new(
        out,
        windows.scale,
        windows.b / ((pad_hw.0 / window_size) * (pad_hw.1 / window_size)),
        hw.0,
        hw.1,
        windows.c,
    )
}

/// `do_pool`: channel-wise max-pool with `kernel = stride`. Exact on the integer grid (shared scale).
pub fn max_pool2d(sp: &Spatial, stride: usize) -> Spatial {
    if stride <= 1 {
        return sp.clone();
    }
    let (out, oh, ow) = max_pool2d_raw(&sp.data, sp.b, sp.h, sp.w, sp.c, stride);
    Spatial::new(out, sp.scale, sp.b, oh, ow, sp.c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq_spatial(b: usize, h: usize, w: usize, c: usize) -> Spatial {
        // distinct values so any mis-indexing is caught: data[i] = i.
        let data: Vec<i32> = (0..(b * h * w * c) as i32).collect();
        Spatial::new(data, Scale::from_f64(1.0), b, h, w, c)
    }

    #[test]
    fn window_partition_unpartition_round_trips_no_padding() {
        // H,W exact multiples of window → unpartition(partition(x)) == x.
        let sp = seq_spatial(2, 8, 12, 5);
        let (win, (hp, wp)) = window_partition(&sp, 4);
        assert_eq!((win.b, win.h, win.w, win.c), (2 * 2 * 3, 4, 4, 5));
        assert_eq!((hp, wp), (8, 12));
        let back = window_unpartition(&win, 4, (hp, wp), (8, 12));
        assert_eq!((back.b, back.h, back.w, back.c), (2, 8, 12, 5));
        assert_eq!(back.data, sp.data, "round-trip must be identity");
    }

    #[test]
    fn window_partition_unpartition_round_trips_with_padding() {
        // H,W NOT multiples of window → padded internally, cropped back on unpartition → still identity.
        let sp = seq_spatial(1, 7, 10, 3);
        let (win, pad_hw) = window_partition(&sp, 4);
        assert_eq!((win.h, win.w), (4, 4));
        assert_eq!(pad_hw, (8, 12));
        let back = window_unpartition(&win, 4, pad_hw, (7, 10));
        assert_eq!((back.b, back.h, back.w, back.c), (1, 7, 10, 3));
        assert_eq!(back.data, sp.data, "padded round-trip must still be identity on the original region");
    }

    #[test]
    fn max_pool2d_known_values() {
        // single channel, 4x4 counting grid, stride 2 → each 2x2 block's max is its bottom-right corner.
        let sp = seq_spatial(1, 4, 4, 1);
        let pooled = max_pool2d(&sp, 2);
        assert_eq!((pooled.b, pooled.h, pooled.w, pooled.c), (1, 2, 2, 1));
        // grid values 0..15 row-major; block (0,0)={0,1,4,5}->5, (0,1)={2,3,6,7}->7, (1,0)={8,9,12,13}->13, (1,1)->15
        assert_eq!(pooled.data, vec![5, 7, 13, 15]);
    }

    #[test]
    fn max_pool2d_multichannel_picks_per_channel_max() {
        // 2x2 spatial, 2 channels, stride 2 → one output pixel per channel = channel max over the 4 positions.
        // layout [H,W,C]: positions (0,0),(0,1),(1,0),(1,1) each with [c0,c1].
        let data = vec![1, 9, /* (0,0) */ 3, 2, /* (0,1) */ 8, 4, /* (1,0) */ 0, 7 /* (1,1) */];
        let sp = Spatial::new(data, Scale::from_f64(1.0), 1, 2, 2, 2);
        let pooled = max_pool2d(&sp, 2);
        assert_eq!(pooled.data, vec![8, 9], "per-channel max: c0=max(1,3,8,0)=8, c1=max(9,2,4,7)=9");
    }

    #[test]
    fn chw_to_hwc_permutes_correctly() {
        // C=2,H=2,W=3 channels-first; element (ci,y,x) at ci*6+y*3+x → channels-last (y,x,ci).
        let t = IntTensor::new((0..12).collect(), vec![2, 2, 3], Scale::from_f64(1.0));
        let sp = chw_to_hwc(&t);
        assert_eq!((sp.b, sp.h, sp.w, sp.c), (1, 2, 3, 2));
        // hwc[(y,x,ci)] = chw[ci*6 + y*3 + x]; e.g. (y=0,x=0,c0)=0, (0,0,c1)=6, (0,1,c0)=1, (1,2,c1)=11
        assert_eq!(sp.data[0], 0);
        assert_eq!(sp.data[1], 6);
        assert_eq!(sp.data[2], 1);
        assert_eq!(sp.data[(3 + 2) * 2 + 1], 11); // row 1 · 3 cols + col 2, channel 1
    }

    #[test]
    fn chw_hwc_round_trips() {
        let t = IntTensor::new((0..24).collect(), vec![2, 3, 4], Scale::from_f64(1.0)); // [C=2,H=3,W=4]
        let sp = chw_to_hwc(&t);
        assert_eq!((sp.b, sp.h, sp.w, sp.c), (1, 3, 4, 2));
        let back = hwc_to_chw(&sp);
        assert_eq!(back.shape, vec![2, 3, 4]);
        assert_eq!(back.data, t.data, "chw→hwc→chw must be identity");
    }

    #[test]
    fn nearest_upsample_2x_duplicates() {
        // 1 channel, 2x2 → 4x4, each pixel a 2x2 block.
        let t = IntTensor::new(vec![1, 2, 3, 4], vec![1, 2, 2], Scale::from_f64(1.0));
        let up = nearest_upsample_2x(&t);
        assert_eq!(up.shape, vec![1, 4, 4]);
        assert_eq!(up.data, vec![1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]);
    }

    #[test]
    fn tokens_round_trip() {
        let sp = seq_spatial(1, 3, 4, 6);
        let t = sp.as_tokens();
        assert_eq!(t.shape, vec![12, 6]);
        let back = Spatial::from_tokens(&t, 1, 3, 4, 6);
        assert_eq!(back.data, sp.data);
    }
}
