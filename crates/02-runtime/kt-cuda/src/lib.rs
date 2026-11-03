//! kt-cuda — the exact INT8 Tensor-Core backend. A b=12 code is three balanced base-81 digits, each four of
//! its trits, so the engine's accumulate `Y[r,o] = Σ_k X[r,k]·W[o,k]` becomes nine `int8 × int8 → int32` products of
//! digit planes, added as `Σ_{i,j} 81^{i+j} · S_ij` in `i64`. Integer products are exact, so the result is the
//! integers `portable` returns, bit for bit. No float appears anywhere.
//!
//! - [`digits`]: the split, the packed tile layout, and every bound as a compile-time assertion.
//! - [`TcLinear`]: the backend on an executor. It keeps packed weights resident by `WeightId` and sends any call outside
//!   the bounds to `portable` ([`fallbacks`] counts them).
//! - [`HostExec`]: the executor on the CPU (`cuda-emu`). It runs the tile loop the CUDA kernel runs, so a host without
//!   CUDA checks everything but the tile instruction.
//!
//! # Example
//!
//! ```
//! use kt_core::backend::{LinearBackend, Portable};
//!
//! // x is 2 × 2, w is 2 × 2 (one weight row per output); the codes include ±WLEVELS.
//! let (x, w) = (vec![265_720, -1, 3, 40], vec![2, 0, -265_720, 81]);
//! assert_eq!(kt_cuda::emulator().linear(&x, &w, 2, 2, 2), Portable.linear(&x, &w, 2, 2, 2));
//! ```

mod backend;
#[cfg(feature = "cuda")]
mod device;
pub mod digits;
mod host;

#[cfg(feature = "plants")]
pub use backend::Plant;
pub use backend::{fallbacks, Exec, TcLinear};
pub use host::HostExec;

use kt_core::backend::LinearBackend;
use std::sync::Arc;

/// The algorithm on the CPU, weights resident under the default budget (8 GiB).
pub fn emulator() -> TcLinear<HostExec> {
    TcLinear::new(HostExec)
}

/// Whether this build drives CUDA devices (the `cuda` feature).
pub const CUDA: bool = cfg!(feature = "cuda");

/// The CUDA devices this build can drive, by name, in index order. Empty without the `cuda` feature.
pub fn devices() -> Vec<String> {
    #[cfg(feature = "cuda")]
    return device::devices();
    #[cfg(not(feature = "cuda"))]
    Vec::new()
}

/// The backend on CUDA device `index` (`None`: device 0), packed weights resident under `budget` (`None`: 4/5 of the
/// device memory free when it opens). `None` without the `cuda` feature or such a device.
pub fn open(index: Option<usize>, budget: Option<u64>) -> Option<Arc<dyn LinearBackend>> {
    #[cfg(feature = "cuda")]
    return device::DeviceExec::open(index.unwrap_or(0)).map(|d| {
        let b = TcLinear::new(d);
        Arc::new(match budget {
            Some(bytes) => b.with_budget(bytes),
            None => b,
        }) as Arc<dyn LinearBackend>
    });
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (index, budget);
        None
    }
}

/// The backend on CUDA device `index` with one planted defect (the firing controls). `None` as for [`open`].
#[cfg(feature = "plants")]
pub fn open_planted(index: Option<usize>, plant: Plant) -> Option<Arc<dyn LinearBackend>> {
    #[cfg(feature = "cuda")]
    return device::DeviceExec::open(index.unwrap_or(0))
        .map(|d| Arc::new(TcLinear::planted(d, plant)) as Arc<dyn LinearBackend>);
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (index, plant);
        None
    }
}

#[cfg(test)]
mod tests;
