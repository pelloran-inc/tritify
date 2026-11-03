//! Firing controls: the CPU linear layer with a planted partition defect, each at the spot a partition can go
//! wrong. Test and example builds only (`cfg(test)` or the `plants` feature); never in a product build.

use kt_core::backend::{dot_portable, plan_threads, LinearBackend};

/// Where the partition is wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BadPartition {
    /// Runs are cut at `⌈len/t⌉` but told they start at `ci · ⌊len/t⌋`.
    FloorOffsets,
    /// The last run is never computed.
    DropLastRun,
    /// Element `i` is read as `(i % rows, i / rows)`: rows and outputs swapped.
    SwappedIndex,
}

/// `linear_cpu` with one planted defect, on [`dot_portable`]. `threads` forces the worker count, as the property test
/// does; `None` uses the engine's own plan, as a model run does.
pub struct PlantedPartition {
    pub bad: BadPartition,
    pub threads: Option<usize>,
}

impl LinearBackend for PlantedPartition {
    fn name(&self) -> String {
        format!("planted-partition-{:?}", self.bad)
    }
    fn linear(&self, x: &[i32], w: &[i32], rows: usize, out: usize, k: usize) -> Vec<i64> {
        let mut y = vec![0i64; rows * out];
        if y.is_empty() {
            return y;
        }
        let planned = plan_threads(rows * out, rows.saturating_mul(out).saturating_mul(k));
        let t = self.threads.unwrap_or(planned).clamp(1, y.len());
        let (len, chunk) = (y.len(), y.len().div_ceil(t));
        let element = |i: usize| match self.bad {
            BadPartition::SwappedIndex => dot_portable(&x[i % rows * k..][..k], &w[i / rows * k..][..k]),
            _ => dot_portable(&x[i / out * k..][..k], &w[i % out * k..][..k]),
        };
        let runs = y.chunks_mut(chunk).count();
        std::thread::scope(|s| {
            for (ci, ys) in y.chunks_mut(chunk).enumerate() {
                let start = match self.bad {
                    BadPartition::FloorOffsets => ci * (len / t),
                    _ => ci * chunk,
                };
                if self.bad == BadPartition::DropLastRun && runs > 1 && ci == runs - 1 {
                    continue;
                }
                let element = &element;
                s.spawn(move || {
                    for (e, slot) in ys.iter_mut().enumerate() {
                        *slot = element(start + e);
                    }
                });
            }
        });
        y
    }
}
