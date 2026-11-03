//! The CPU dot-kernel variants. Each is the same exact `i32 · i32 → i64` multiply-accumulate, built for a different
//! instruction set: `portable` for the build target, `neon` as explicit aarch64 intrinsics, and `avx2` / `avx512` as
//! the portable loop compiled with that target feature enabled (function multi-versioning). A variant is handed out
//! only after the CPU reports its feature, so the `unsafe` calls below never run on a CPU that lacks it.

use kt_core::backend::{dot_portable, linear_cpu, DotI32, LinearBackend};

/// A CPU kernel variant. [`CpuVariant::available`] says whether this host can run it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CpuVariant {
    /// The engine's own loop, built for the baseline target (on aarch64 that already includes NEON).
    Portable,
    /// Explicit NEON intrinsics (`vmlal_s32`), aarch64 only.
    Neon,
    /// The loop compiled with AVX2 enabled, x86-64 only.
    Avx2,
    /// The loop compiled with AVX-512F enabled, x86-64 only.
    Avx512,
}

impl CpuVariant {
    /// Every variant, in no particular order.
    pub const ALL: [CpuVariant; 4] = [CpuVariant::Portable, CpuVariant::Neon, CpuVariant::Avx2, CpuVariant::Avx512];

    /// The stable name used in configuration and run records.
    pub fn name(self) -> &'static str {
        match self {
            CpuVariant::Portable => "portable",
            CpuVariant::Neon => "neon",
            CpuVariant::Avx2 => "avx2",
            CpuVariant::Avx512 => "avx512",
        }
    }

    /// Parse a name from [`CpuVariant::name`].
    pub fn from_name(s: &str) -> Option<CpuVariant> {
        CpuVariant::ALL.into_iter().find(|v| v.name() == s)
    }

    /// Does this CPU have the instructions the variant needs?
    pub fn available(self) -> bool {
        match self {
            CpuVariant::Portable => true,
            CpuVariant::Neon => arm::has_neon(),
            CpuVariant::Avx2 => x86::has("avx2"),
            CpuVariant::Avx512 => x86::has("avx512f"),
        }
    }

    /// The dot kernel, or `None` if this host cannot run the variant.
    pub fn dot(self) -> Option<DotI32> {
        if !self.available() {
            return None;
        }
        match self {
            CpuVariant::Portable => Some(dot_portable),
            CpuVariant::Neon => arm::DOT,
            CpuVariant::Avx2 => x86::DOT_AVX2,
            CpuVariant::Avx512 => x86::DOT_AVX512,
        }
    }
}

/// A CPU backend: one dot variant, driven by the engine's threaded [`linear_cpu`].
pub struct CpuBackend {
    variant: CpuVariant,
    dot: DotI32,
}

impl CpuBackend {
    /// The backend for `variant`, or an error if this host cannot run it (never a silent fallback).
    pub fn new(variant: CpuVariant) -> Result<CpuBackend, String> {
        let dot =
            variant.dot().ok_or_else(|| format!("CPU variant '{}' is not available on this host", variant.name()))?;
        Ok(CpuBackend { variant, dot })
    }

    pub fn variant(&self) -> CpuVariant {
        self.variant
    }
}

impl LinearBackend for CpuBackend {
    fn name(&self) -> String {
        self.variant.name().into()
    }
    fn linear(&self, x: &[i32], w: &[i32], rows: usize, out: usize, k: usize) -> Vec<i64> {
        linear_cpu(x, w, rows, out, k, self.dot)
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::DotI32;
    use std::arch::aarch64::{vaddq_s64, vaddvq_s64, vdupq_n_s64, vget_low_s32, vld1q_s32, vmlal_high_s32, vmlal_s32};

    pub fn has_neon() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    pub const DOT: Option<DotI32> = Some(dot_neon);

    /// Four lanes per step: the low and high halves widen-multiply-accumulate into two `i64×2` accumulators; the
    /// tail (`len % 4`) is summed in scalar code.
    fn dot_neon(a: &[i32], b: &[i32]) -> i64 {
        assert_eq!(a.len(), b.len(), "dot: length mismatch");
        let quads = a.len() / 4;
        // SAFETY: NEON is part of every aarch64 target this crate builds for, and is checked before this kernel is
        // handed out. Each load reads elements 4q..4q+4 with q < len/4, so it stays inside both slices.
        let mut sum = unsafe {
            let (mut lo, mut hi) = (vdupq_n_s64(0), vdupq_n_s64(0));
            for q in 0..quads {
                let va = vld1q_s32(a.as_ptr().add(4 * q));
                let vb = vld1q_s32(b.as_ptr().add(4 * q));
                lo = vmlal_s32(lo, vget_low_s32(va), vget_low_s32(vb));
                hi = vmlal_high_s32(hi, va, vb);
            }
            vaddvq_s64(vaddq_s64(lo, hi))
        };
        for i in 4 * quads..a.len() {
            sum += a[i] as i64 * b[i] as i64;
        }
        sum
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod arm {
    use super::DotI32;
    pub fn has_neon() -> bool {
        false
    }
    pub const DOT: Option<DotI32> = None;
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::DotI32;

    pub fn has(feature: &str) -> bool {
        match feature {
            "avx2" => std::arch::is_x86_feature_detected!("avx2"),
            "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
            _ => false,
        }
    }

    pub const DOT_AVX2: Option<DotI32> = Some(dot_avx2);
    pub const DOT_AVX512: Option<DotI32> = Some(dot_avx512);

    /// The portable loop, inlined into each feature-enabled function so it is compiled for that instruction set.
    #[inline(always)]
    fn dot_loop(a: &[i32], b: &[i32]) -> i64 {
        assert_eq!(a.len(), b.len(), "dot: length mismatch");
        a.iter().zip(b).map(|(&x, &y)| x as i64 * y as i64).sum()
    }

    #[target_feature(enable = "avx2")]
    unsafe fn dot_avx2_impl(a: &[i32], b: &[i32]) -> i64 {
        dot_loop(a, b)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn dot_avx512_impl(a: &[i32], b: &[i32]) -> i64 {
        dot_loop(a, b)
    }

    fn dot_avx2(a: &[i32], b: &[i32]) -> i64 {
        // SAFETY: handed out only by `CpuVariant::dot`, after `is_x86_feature_detected!("avx2")`.
        unsafe { dot_avx2_impl(a, b) }
    }

    fn dot_avx512(a: &[i32], b: &[i32]) -> i64 {
        // SAFETY: handed out only by `CpuVariant::dot`, after `is_x86_feature_detected!("avx512f")`.
        unsafe { dot_avx512_impl(a, b) }
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod x86 {
    use super::DotI32;
    pub fn has(_feature: &str) -> bool {
        false
    }
    pub const DOT_AVX2: Option<DotI32> = None;
    pub const DOT_AVX512: Option<DotI32> = None;
}
