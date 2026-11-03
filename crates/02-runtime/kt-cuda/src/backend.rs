//! [`TcLinear`], the engine's linear accumulate as digit-plane products, on any [`Exec`]. It checks the bounds, packs
//! the activations per call, keeps packed weights resident by `WeightId` under an LRU byte budget (as the resident GPU
//! backend does), and hands the product to the executor. A call outside the bounds goes to `portable`, counted.

use crate::digits::{self, DIGITS, KT_FLUSH, K_MAX};
use kt_core::backend::{LinearBackend, Portable, WeightId, Weights};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Where the digit-plane product runs: the CUDA device, or the host executor of the same algorithm.
pub trait Exec: Send + Sync {
    /// Packed weights as the executor holds them (device memory, or host bytes).
    type Buf: Send + Sync;
    /// `cuda` or `cuda-emu`, for planted names.
    fn kind(&self) -> &'static str;
    /// The backend's name in run records (`cuda:<device>`, `cuda-emu`).
    fn name(&self) -> String;
    /// The resident-weight budget when none is configured.
    fn default_budget(&self) -> u64;
    /// Take packed weight tiles into the executor's memory.
    fn upload(&self, packed: Vec<i8>) -> Self::Buf;
    /// `Y = Σ 81^{i+j} · S_ij` for packed activations `xa` (`rows × k`) and weights `wb` (`out × k`). The `int32` sums
    /// are added into `i64` every `flush` k-tiles; `skip22` drops the `(2,2)` plane (a plant only).
    fn gemm(&self, xa: &[i8], wb: &Self::Buf, shape: (usize, usize, usize), flush: usize, skip22: bool) -> Vec<i64>;
}

/// A planted defect (the `plants` feature only, for the firing controls; never in a product build).
#[cfg(feature = "plants")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plant {
    /// The recombination skips the `(2,2)` digit plane.
    DropPlane22,
    /// The activation split drops the carry out of a rebalanced low digit.
    WrongCarry,
    /// The `int32` accumulators are never flushed into `i64`.
    NoFlush,
    /// The resident-weight cache is keyed by shape, so two layers of one shape share the first one's weights.
    CacheByShape,
}

static FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Calls in this process that a digit backend handed to `portable` because they were outside its bounds (a code beyond
/// `±WLEVELS`, or `k > K_MAX`). The engine's calls never are, so a model run reports 0.
pub fn fallbacks() -> u64 {
    FALLBACKS.load(Ordering::Relaxed)
}

/// A cache key: the weights' id and shape. A planted cache drops the id.
type Key = (Option<WeightId>, usize, usize);

struct Entry<B> {
    buf: Arc<B>,
    bytes: u64,
    last_use: u64,
}

struct Cache<B> {
    map: HashMap<Key, Entry<B>>,
    bytes: u64,
    tick: u64,
}

/// The choices a plant can change. A product build always runs [`EXACT`].
#[derive(Clone, Copy)]
struct Knobs {
    split_x: fn(i32) -> Option<[i8; DIGITS]>,
    flush: usize,
    skip22: bool,
    key_by_id: bool,
}

const EXACT: Knobs = Knobs { split_x: digits::split, flush: KT_FLUSH, skip22: false, key_by_id: true };

/// The linear accumulate as nine digit-plane products on an executor.
pub struct TcLinear<E: Exec> {
    exec: E,
    budget: Option<u64>,
    cache: Mutex<Cache<E::Buf>>,
    knobs: Knobs,
    #[cfg(feature = "plants")]
    plant: Option<Plant>,
}

impl<E: Exec> TcLinear<E> {
    /// Weights resident under the executor's default budget.
    pub fn new(exec: E) -> TcLinear<E> {
        TcLinear {
            exec,
            budget: None,
            cache: Mutex::new(Cache { map: HashMap::new(), bytes: 0, tick: 0 }),
            knobs: EXACT,
            #[cfg(feature = "plants")]
            plant: None,
        }
    }

    /// Cap the bytes of packed weights kept resident. A weight larger than the budget is used for its call and dropped,
    /// so a budget of 1 byte evicts on every call.
    pub fn with_budget(mut self, bytes: u64) -> TcLinear<E> {
        self.budget = Some(bytes);
        self
    }

    /// The backend with one planted defect.
    #[cfg(feature = "plants")]
    pub fn planted(exec: E, plant: Plant) -> TcLinear<E> {
        let knobs = match plant {
            Plant::DropPlane22 => Knobs { skip22: true, ..EXACT },
            Plant::WrongCarry => Knobs { split_x: digits::split_wrong_carry, ..EXACT },
            Plant::NoFlush => Knobs { flush: usize::MAX, ..EXACT },
            Plant::CacheByShape => Knobs { key_by_id: false, ..EXACT },
        };
        TcLinear { knobs, plant: Some(plant), ..TcLinear::new(exec) }
    }

    /// The bytes of packed weights resident now.
    pub fn resident_bytes(&self) -> u64 {
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).bytes
    }

    /// Packed weights for this call: resident if the id was seen, else packed, taken in and cached under the budget.
    /// `None` if a code is outside `±WLEVELS`.
    fn weights(&self, w: Weights<'_>, out: usize, k: usize) -> Option<Arc<E::Buf>> {
        let key = (self.knobs.key_by_id.then_some(w.id), out, k);
        {
            let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            c.tick += 1;
            let tick = c.tick;
            if let Some(e) = c.map.get_mut(&key) {
                e.last_use = tick;
                return Some(e.buf.clone());
            }
        }
        let packed = digits::pack(w.codes, out, k, digits::split)?;
        let bytes = packed.len() as u64;
        let buf = Arc::new(self.exec.upload(packed));
        let budget = self.budget.unwrap_or_else(|| self.exec.default_budget());
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        while c.bytes + bytes > budget {
            let Some(lru) = c.map.iter().min_by_key(|(_, e)| e.last_use).map(|(k, _)| *k) else { break };
            let gone = c.map.remove(&lru).map_or(0, |e| e.bytes);
            c.bytes -= gone;
        }
        if bytes <= budget {
            let last_use = c.tick;
            if let Some(old) = c.map.insert(key, Entry { buf: buf.clone(), bytes, last_use }) {
                c.bytes -= old.bytes; // another thread took the same weights in first
            }
            c.bytes += bytes;
        }
        Some(buf)
    }

    fn run(&self, x: &[i32], w: Option<Weights<'_>>, codes: &[i32], (rows, out, k): (usize, usize, usize)) -> Vec<i64> {
        assert_eq!(x.len(), rows * k, "linear: x is not rows·k");
        assert_eq!(codes.len(), out * k, "linear: w is not out·k");
        let kn = self.knobs;
        let computed = (k <= K_MAX).then(|| {
            let xa = digits::pack(x, rows, k, kn.split_x)?;
            let wb = match w {
                Some(w) => self.weights(w, out, k)?,
                None => Arc::new(self.exec.upload(digits::pack(codes, out, k, digits::split)?)),
            };
            Some(self.exec.gemm(&xa, &wb, (rows, out, k), kn.flush, kn.skip22))
        });
        computed.flatten().unwrap_or_else(|| {
            FALLBACKS.fetch_add(1, Ordering::Relaxed);
            Portable.linear(x, codes, rows, out, k)
        })
    }
}

impl<E: Exec> LinearBackend for TcLinear<E> {
    fn name(&self) -> String {
        #[cfg(feature = "plants")]
        if let Some(p) = self.plant {
            return format!("planted-{}-{p:?}", self.exec.kind());
        }
        match self.budget {
            Some(b) => format!("{} [budget {b} B]", self.exec.name()),
            None => self.exec.name(),
        }
    }
    fn linear(&self, x: &[i32], w: &[i32], rows: usize, out: usize, k: usize) -> Vec<i64> {
        self.run(x, None, w, (rows, out, k))
    }
    fn linear_weights(&self, x: &[i32], w: Weights<'_>, rows: usize, out: usize, k: usize) -> Vec<i64> {
        self.run(x, Some(w), w.codes, (rows, out, k))
    }
}
