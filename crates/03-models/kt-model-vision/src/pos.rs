//! The two input-independent positional tables SAM 2.1 adds, computed in Rust so the demo is self-contained (no
//! precomputed dump): the Hiera window positional embedding (`_get_pos_embed`: bicubic-upsample the learned
//! `[112,14,14]` grid + tile the `[112,8,8]` window grid) and the image-wide positional embedding
//! (`get_image_wide_positional_embeddings`: a random-Fourier map over the 64×64 decoder grid). Both depend only
//! on the fixed image size, so they are computed once in f64 — the sanctioned table-precompute, like RoPE.

use crate::hiera::WeightStore;

/// Cubic convolution kernel, a = −0.75 (torch's `F.interpolate(mode="bicubic")` default).
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.75;
    let x = x.abs();
    if x <= 1.0 {
        (A + 2.0) * x * x * x - (A + 3.0) * x * x + 1.0
    } else if x < 2.0 {
        A * x * x * x - 5.0 * A * x * x + 8.0 * A * x - 4.0 * A
    } else {
        0.0
    }
}

/// Separable bicubic resize of a single `[ih,iw]` plane to `[oh,ow]`, `align_corners=False` (edge-replicated
/// taps), matching `F.interpolate(mode="bicubic")`.
fn bicubic_plane(src: &[f64], ih: usize, iw: usize, oh: usize, ow: usize) -> Vec<f64> {
    let sx = iw as f64 / ow as f64;
    let mut tmp = vec![0.0f64; ih * ow];
    for y in 0..ih {
        for ox in 0..ow {
            let s = (ox as f64 + 0.5) * sx - 0.5;
            let i0 = s.floor() as i64;
            let t = s - i0 as f64;
            let mut acc = 0.0;
            for k in -1..=2 {
                let xi = (i0 + k).clamp(0, iw as i64 - 1) as usize;
                acc += cubic(t - k as f64) * src[y * iw + xi];
            }
            tmp[y * ow + ox] = acc;
        }
    }
    let sy = ih as f64 / oh as f64;
    let mut out = vec![0.0f64; oh * ow];
    for oy in 0..oh {
        let s = (oy as f64 + 0.5) * sy - 0.5;
        let i0 = s.floor() as i64;
        let t = s - i0 as f64;
        for ox in 0..ow {
            let mut acc = 0.0;
            for k in -1..=2 {
                let yi = (i0 + k).clamp(0, ih as i64 - 1) as usize;
                acc += cubic(t - k as f64) * tmp[yi * ow + ox];
            }
            out[oy * ow + ox] = acc;
        }
    }
    out
}

/// Hiera positional embedding as `[grid·grid, embed]` tokens (row-major token = y·grid+x, channel inner).
/// `pos_embed` is `[1,embed,bg,bg]` (bg = 14), `pos_embed_window` is `[1,embed,win,win]` (win = 8).
pub fn hiera_pos_embed(store: &WeightStore, grid: usize, embed: usize) -> Vec<f64> {
    let pe = store.w("vision_encoder.backbone.pos_embed"); // [1,embed,bg,bg]
    let pw = store.w("vision_encoder.backbone.pos_embed_window"); // [1,embed,win,win]
    let bg = ((pe.len() / embed) as f64).sqrt().round() as usize;
    let win = ((pw.len() / embed) as f64).sqrt().round() as usize;
    let mut tokens = vec![0.0f64; grid * grid * embed];
    for c in 0..embed {
        let src: Vec<f64> = pe[c * bg * bg..(c + 1) * bg * bg].iter().map(|&v| v as f64).collect();
        let up = bicubic_plane(&src, bg, bg, grid, grid); // [grid,grid]
        for y in 0..grid {
            for x in 0..grid {
                let w = pw[(c * win + y % win) * win + x % win] as f64; // tiled window embed
                tokens[(y * grid + x) * embed + c] = up[y * grid + x] + w;
            }
        }
    }
    tokens
}

/// Image-wide positional embedding as `[size·size, 2·half]` tokens (token = r·size+c), from the random-Fourier
/// `shared_image_embedding.positional_embedding` `[2, half]` (half = 128).
pub fn image_wide_pos(store: &WeightStore, size: usize) -> Vec<f64> {
    let pe: Vec<f64> = store.w("shared_image_embedding.positional_embedding").iter().map(|&v| v as f64).collect();
    let half = pe.len() / 2;
    let hidden = 2 * half;
    let mut tokens = vec![0.0f64; size * size * hidden];
    for r in 0..size {
        for c in 0..size {
            let x = (c as f64 + 0.5) / size as f64;
            let y = (r as f64 + 0.5) / size as f64;
            let (cx, cy) = (2.0 * x - 1.0, 2.0 * y - 1.0);
            let base = (r * size + c) * hidden;
            for j in 0..half {
                let f = 2.0 * std::f64::consts::PI * (cx * pe[j] + cy * pe[half + j]);
                tokens[base + j] = f.sin();
                tokens[base + half + j] = f.cos();
            }
        }
    }
    tokens
}
