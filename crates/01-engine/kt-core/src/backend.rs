//! The compute-backend seam. The engine's hot integer kernel is the exact accumulate of a linear layer,
//! `Y[r,o] = Σ_k X[r,k]·W[o,k]` with `i32` operands and `i64` accumulators. It runs on the backend installed here.
//! The engine ships one backend, [`Portable`], and depends on no other: the runtime layer above it (`kt-runtime`,
//! `kt-gpu`) builds a faster one for the host and installs it with [`install_backend`]. Integer accumulation is exact
//! and order-independent, so a correct backend can differ only in speed, never in a single output bit.
//!
//! # Example
//!
//! ```
//! use kt_core::backend::{self, LinearBackend};
//!
//! // x is rows × k, w is out × k (one weight row per output): Y[r,o] = Σ_k x[r,k]·w[o,k].
//! let (x, w) = (vec![1, 2, 3, 4, 5, 6], vec![1, 0, -1, 2, 2, 2]);
//! let y = backend::backend().linear(&x, &w, 2, 2, 3);
//! assert_eq!(y, vec![-2, 12, -2, 30]);
//! assert_eq!(backend::backend().name(), "portable"); // nothing installed: the engine's own kernel
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// A process-local identity for one weight matrix, given once when its layer is built and never reused. A backend that
/// keeps weights on a device uses it as a cache key: a pointer is no proof, because freed memory is reused. It is a
/// cache key and nothing else: it never enters a hash, an output or any reproducing identity, and two processes give
/// the same weights different ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightId(u64);

impl WeightId {
    /// A new id, distinct from every other in this process.
    pub fn fresh() -> WeightId {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        WeightId(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

/// A weight matrix (`out × k`, row-major codes) with its identity. The codes behind an id never change.
#[derive(Clone, Copy)]
pub struct Weights<'a> {
    pub id: WeightId,
    pub codes: &'a [i32],
}

/// A compute backend for the linear-layer accumulate. Every implementation must return exactly the integers
/// [`Portable`] returns.
pub trait LinearBackend: Send + Sync {
    /// A short, stable name for logs and run records (`portable`, `neon`, `avx2`, `gpu:<adapter>`).
    fn name(&self) -> String;
    /// The exact accumulators. `x` is `rows × k` and `w` is `out × k`, both row-major, and the result is `rows × out`
    /// with `Y[r,o] = Σ_k x[r·k+i]·w[o·k+i]` in `i64`. The engine's operands are b=12 codes (`|v| ≤ WLEVELS`), so a
    /// product is below 2^37 and a sum of fewer than 2^26 of them cannot overflow `i64`.
    fn linear(&self, x: &[i32], w: &[i32], rows: usize, out: usize, k: usize) -> Vec<i64>;

    /// The same accumulate, told which weights these are, so a backend may keep them resident between calls. It must
    /// return exactly what [`LinearBackend::linear`] returns. The default ignores the id.
    fn linear_weights(&self, x: &[i32], w: Weights<'_>, rows: usize, out: usize, k: usize) -> Vec<i64> {
        self.linear(x, w.codes, rows, out, k)
    }
}

/// One output element of the CPU path: an exact `i32 · i32 → i64` dot product of two equal-length slices.
pub type DotI32 = fn(&[i32], &[i32]) -> i64;

/// The engine's own dot kernel: a widening multiply-accumulate the compiler vectorises for the build target
/// (NEON `smlal` on aarch64; the baseline vector ISA on x86-64).
pub fn dot_portable(a: &[i32], b: &[i32]) -> i64 {
    debug_assert_eq!(a.len(), b.len(), "dot: length mismatch");
    a.iter().zip(b).map(|(&x, &y)| x as i64 * y as i64).sum()
}

/// The multiply-accumulates one core does in the time one more worker costs to spawn. A scoped spawn costs about
/// 5.6 µs per thread on the M3 Max and 21 µs on an EPYC 9454P. A faster dot does more work in that time, so this
/// spans about 2^15 to 2^18 by host and variant; 2^17 is the middle (measured with the `cpu_partition` example).
pub const WORK_PER_SPAWN: usize = 1 << 17;

/// A call with less work than this runs serially: there, one spawned worker
/// can cost more than it saves: on an EPYC 9454P, Qwen's `1×896×896` (802,816) takes 70 µs serially with AVX-512 and
/// 130 µs on two workers.
pub const SERIAL_BELOW: usize = 1 << 20;

/// The workers for one call with `elems` independent outputs and `work` multiply-accumulates.
/// - Below [`SERIAL_BELOW`], one worker.
/// - Above it, a call takes about `C/t + s·t` on `t` workers: `C` its single-thread time, `s` the spawn cost of a
///   worker. That is least at `t = √(C/s) = √(work / WORK_PER_SPAWN)`.
/// - The count is at most one worker per output and at most the thread cap ([`crate::set_max_threads`]; `0` = every
///   core the host reports), and always at least 1.
///
/// It chooses speed only: every count gives the same bits.
pub fn plan_threads(elems: usize, work: usize) -> usize {
    if work < SERIAL_BELOW {
        return 1;
    }
    crate::layers::resolve_threads(elems).min((work / WORK_PER_SPAWN).isqrt()).max(1)
}

/// `buf[i] = f(i)` for every index, the buffer cut into contiguous runs on `threads` workers; the calling thread takes
/// the first run. The runs are `chunks_mut` of the one buffer, so they are disjoint and cover it, which the borrow
/// checker proves. A run's first index is the sum of the lengths before it, not a product of the chunk size, so it is
/// the slice's own offset by construction. Every element is written once, by `f`, a pure function of its index, so the
/// result is the serial loop's, bit for bit, for any `threads`.
pub(crate) fn fill_partitioned(buf: &mut [i64], threads: usize, f: impl Fn(usize) -> i64 + Sync) {
    if buf.is_empty() {
        return;
    }
    let chunk = buf.len().div_ceil(threads.clamp(1, buf.len()));
    let run = |start: usize, ys: &mut [i64]| {
        for (e, slot) in ys.iter_mut().enumerate() {
            *slot = f(start + e);
        }
    };
    if chunk == buf.len() {
        return run(0, buf);
    }
    std::thread::scope(|s| {
        let mut runs = buf.chunks_mut(chunk);
        let first = runs.next();
        let mut start = first.as_ref().map_or(0, |ys| ys.len());
        for ys in runs {
            let (run, at) = (&run, start);
            start += ys.len();
            s.spawn(move || run(at, ys));
        }
        if let Some(ys) = first {
            run(0, ys);
        }
    });
}

/// The threaded CPU driver every CPU backend shares. The output `rows × out` is one flat buffer; element `i`
/// is `dot(x[i / out], w[i % out])`. [`plan_threads`] picks the worker count from the call's shape, and the buffer is cut
/// into that many contiguous runs: whole rows when there are many, the outputs of one row when there is one, as in
/// every decode step. `k` is never split, so each element is one call of the variant's `dot`, as before.
pub fn linear_cpu(x: &[i32], w: &[i32], rows: usize, out: usize, k: usize, dot: DotI32) -> Vec<i64> {
    assert_eq!(x.len(), rows * k, "linear: x is not rows·k");
    assert_eq!(w.len(), out * k, "linear: w is not out·k");
    let mut acc = vec![0i64; rows * out];
    let threads = plan_threads(rows * out, rows.saturating_mul(out).saturating_mul(k));
    fill_partitioned(&mut acc, threads, |i| dot(&x[i / out * k..][..k], &w[i % out * k..][..k]));
    acc
}

/// The engine's built-in backend: [`dot_portable`] on the CPU, threaded by [`linear_cpu`].
pub struct Portable;

impl LinearBackend for Portable {
    fn name(&self) -> String {
        "portable".into()
    }
    fn linear(&self, x: &[i32], w: &[i32], rows: usize, out: usize, k: usize) -> Vec<i64> {
        linear_cpu(x, w, rows, out, k, dot_portable)
    }
}

static INSTALLED: RwLock<Option<Arc<dyn LinearBackend>>> = RwLock::new(None);

/// Install the backend every linear layer uses from the next call on; returns the previously installed one. The
/// runtime layer calls this at startup. The output bits do not depend on which backend is installed.
pub fn install_backend(b: Arc<dyn LinearBackend>) -> Option<Arc<dyn LinearBackend>> {
    INSTALLED.write().unwrap_or_else(|e| e.into_inner()).replace(b)
}

/// Return to the engine's own [`Portable`] backend.
pub fn reset_backend() {
    INSTALLED.write().unwrap_or_else(|e| e.into_inner()).take();
}

/// The backend in use: the installed one, or [`Portable`].
pub fn backend() -> Arc<dyn LinearBackend> {
    INSTALLED.read().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_else(|| Arc::new(Portable))
}
