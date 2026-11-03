//! Image preprocessing — the SAM 2.1 processor's float input prep, in Rust, so the demo takes a raw image:
//! bilinear resize to 1024×1024 (`F.interpolate` bilinear, `align_corners=False`; antialias is a no-op here
//! because this is upsampling), rescale by 1/255, normalize by the ImageNet mean/std. This is ordinary float
//! input prep — identical whether the model then runs in float or integer — not part of the exact-integer path.

pub const IMAGENET_MEAN: [f64; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f64; 3] = [0.229, 0.224, 0.225];

/// Separable bilinear resize of one `[ih,iw]` plane to `[oh,ow]`, `align_corners=False` (pixel-center mapping,
/// edge-clamped taps).
fn bilinear_plane(src: &[f64], ih: usize, iw: usize, oh: usize, ow: usize) -> Vec<f64> {
    let sx = iw as f64 / ow as f64;
    let mut tmp = vec![0.0f64; ih * ow];
    for y in 0..ih {
        for ox in 0..ow {
            let s = ((ox as f64 + 0.5) * sx - 0.5).max(0.0);
            let i0 = s.floor() as usize;
            let i1 = (i0 + 1).min(iw - 1);
            let t = s - i0 as f64;
            tmp[y * ow + ox] = src[y * iw + i0] * (1.0 - t) + src[y * iw + i1] * t;
        }
    }
    let sy = ih as f64 / oh as f64;
    let mut out = vec![0.0f64; oh * ow];
    for oy in 0..oh {
        let s = ((oy as f64 + 0.5) * sy - 0.5).max(0.0);
        let i0 = s.floor() as usize;
        let i1 = (i0 + 1).min(ih - 1);
        let t = s - i0 as f64;
        for ox in 0..ow {
            out[oy * ow + ox] = tmp[i0 * ow + ox] * (1.0 - t) + tmp[i1 * ow + ox] * t;
        }
    }
    out
}

/// Preprocess an RGB image (channels-last `[h,w,3]`, u8) → `pixel_values` `[3,size,size]` f32, exactly as the
/// SAM 2.1 processor (resize → /255 → ImageNet normalize).
pub fn preprocess_rgb(img_hwc: &[u8], h: usize, w: usize, size: usize) -> Vec<f32> {
    assert_eq!(img_hwc.len(), h * w * 3);
    let mut out = vec![0f32; 3 * size * size];
    for c in 0..3 {
        let plane: Vec<f64> = (0..h * w).map(|p| img_hwc[p * 3 + c] as f64).collect();
        let up = bilinear_plane(&plane, h, w, size, size);
        for (p, &v) in up.iter().enumerate() {
            out[c * size * size + p] = ((v / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]) as f32;
        }
    }
    out
}

/// A synthetic test image: a 512×512 tan rectangle (rows 150..360, cols 180..400 = `[200, 180, 140]`) on
/// black — the reproducible prompt image used throughout the port.
pub fn synthetic_test_image() -> (Vec<u8>, usize, usize) {
    let (h, w) = (512usize, 512usize);
    let mut img = vec![0u8; h * w * 3];
    for y in 150..360 {
        for x in 180..400 {
            let p = (y * w + x) * 3;
            img[p] = 200;
            img[p + 1] = 180;
            img[p + 2] = 140;
        }
    }
    (img, h, w)
}
