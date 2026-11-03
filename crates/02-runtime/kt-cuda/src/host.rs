//! The host executor: the Tensor-Core product run on the CPU, step for step. It reads the packed tiles with the WMMA
//! index rules, keeps nine `int32` sums per output that wrap as the hardware's do, and adds them into `i64` every
//! `flush` k-tiles. Only the tile instruction itself is not the device's, so on a host without CUDA this checks the
//! packing, the indexing, the flush and the recombination.

use crate::backend::Exec;
use crate::digits::{tile_offset, DIGITS, SCALE, TILE};

/// The algorithm on the CPU (`cuda-emu`). A correctness tool, not a fast path.
pub struct HostExec;

impl Exec for HostExec {
    type Buf = Vec<i8>;

    fn kind(&self) -> &'static str {
        "cuda-emu"
    }
    fn name(&self) -> String {
        "cuda-emu".into()
    }
    fn default_budget(&self) -> u64 {
        8 << 30
    }
    fn upload(&self, packed: Vec<i8>) -> Vec<i8> {
        packed
    }
    fn gemm(
        &self,
        xa: &[i8],
        wb: &Vec<i8>,
        (rows, out, k): (usize, usize, usize),
        flush: usize,
        skip22: bool,
    ) -> Vec<i64> {
        const BLOCK: usize = 64;
        let kts = k.div_ceil(TILE);
        let mut y = vec![0i64; rows * out];
        let work = rows.saturating_mul(out).saturating_mul(k).saturating_mul(DIGITS * DIGITS);
        par_fill(&mut y, BLOCK, work, |b, ys| {
            for (e, slot) in ys.iter_mut().enumerate() {
                let idx = b * BLOCK + e;
                *slot = dot_tiles(xa, wb, idx / out, idx % out, kts, flush, skip22);
            }
            true
        });
        y
    }
}

/// One output of the product. Row `r` of an activation tile (WMMA `matrix_a`, row-major) is `A[r][0..16]`; column `o`
/// of a weight tile (`matrix_b`, column-major) is `B[0..16][o]`, stored at `o·16`. Their dot product is the tile
/// instruction's contribution to `C[r][o]`.
fn dot_tiles(xa: &[i8], wb: &[i8], r: usize, o: usize, kts: usize, flush: usize, skip22: bool) -> i64 {
    let (ra, ob) = (r % TILE * TILE, o % TILE * TILE);
    let (mut acc, mut t0) = (0i64, 0usize);
    while t0 < kts {
        let t1 = kts.min(t0.saturating_add(flush));
        let mut s = [[0i32; DIGITS]; DIGITS];
        for t in t0..t1 {
            let a: [&[i8]; DIGITS] = std::array::from_fn(|i| &xa[tile_offset(r / TILE, t, i, kts) + ra..][..TILE]);
            let b: [&[i8]; DIGITS] = std::array::from_fn(|j| &wb[tile_offset(o / TILE, t, j, kts) + ob..][..TILE]);
            for (si, ai) in s.iter_mut().zip(a) {
                for (sij, bj) in si.iter_mut().zip(b) {
                    *sij = sij.wrapping_add(dot16(ai, bj)); // the hardware's int32 accumulator wraps
                }
            }
        }
        for (i, si) in s.iter().enumerate() {
            for (j, &sij) in si.iter().enumerate() {
                if !(skip22 && i == 2 && j == 2) {
                    acc += SCALE[i + j] * sij as i64;
                }
            }
        }
        t0 = t1;
    }
    acc
}

/// Sixteen digit products: at most 16 · 40² = 25,600 in magnitude.
fn dot16(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// A deterministic parallel fill of `buf` in blocks of `block` elements (the last may be shorter): `fill(i, block_i)`
/// writes block `i` from shared inputs only, so the result is the serial loop's. It honours the engine's thread cap
/// and stays serial for small `work`. Returns whether every `fill` returned `true`.
pub(crate) fn par_fill<T: Send, F>(buf: &mut [T], block: usize, work: usize, fill: F) -> bool
where
    F: Fn(usize, &mut [T]) -> bool + Sync,
{
    if block == 0 || buf.is_empty() {
        return true;
    }
    let nblocks = buf.len().div_ceil(block);
    let cap = kt_core::max_threads();
    let want = if cap == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { cap };
    let threads = want.max(1).min(nblocks);
    if threads < 2 || work < (1 << 20) {
        return buf.chunks_mut(block).enumerate().all(|(i, b)| fill(i, b));
    }
    let chunk = nblocks.div_ceil(threads);
    std::thread::scope(|s| {
        let workers: Vec<_> = buf
            .chunks_mut(chunk * block)
            .enumerate()
            .map(|(ci, group)| {
                let fill = &fill;
                s.spawn(move || group.chunks_mut(block).enumerate().all(|(bi, b)| fill(ci * chunk + bi, b)))
            })
            .collect();
        workers.into_iter().fold(true, |ok, h| h.join().expect("par_fill: a worker panicked") && ok)
    })
}
