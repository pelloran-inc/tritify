//! The digit arithmetic and its bounds. A b=12 code `v` (`|v| ≤ WLEVELS`) is three balanced base-81 digits,
//! `v = d₀ + 81·d₁ + 81²·d₂` with `|dᵢ| ≤ 40`: each digit is four of the code's trits. The product of two codes is then
//! `Σ_{i,j} 81^{i+j}·xᵢ·wⱼ`, nine `int8 × int8` products. Every bound the exactness rests on is a compile-time
//! assertion below, not a note.

use kt_core::layers::WLEVELS;

/// The digit base: four balanced trits, 3⁴.
pub const BASE: i64 = 81;
/// Digits per code.
pub const DIGITS: usize = 3;
/// The largest digit magnitude, `(BASE − 1) / 2`.
pub const DMAX: i64 = 40;
/// The edge of one Tensor-Core tile (WMMA `m16n16k16`).
pub const TILE: usize = 16;
/// Bytes in one packed tile: 16 × 16 digits.
pub const TILE_BYTES: usize = TILE * TILE;
/// The k-tiles an `int32` accumulator takes before it is added into the `i64` sum and reset. Each tile adds at most
/// `TILE · DMAX²` in magnitude, so 83,886 tiles (1,342,176 terms) stay within `i32`. Any longer `k` is split.
pub const KT_FLUSH: usize = (i32::MAX as i64 / (TILE as i64 * DMAX * DMAX)) as usize;
/// The longest `k` the backend computes: the terms sum to at most `k · WLEVELS²` in magnitude, which must fit `i64`.
/// It is the bound of the engine's own sum. A longer `k` goes to `portable`.
pub const K_MAX: usize = (i64::MAX / (WLEVELS * WLEVELS)) as usize;
/// `81^(i+j)` for `i + j` in `0..=4`: the weight of the product of digit planes `i` and `j`.
pub const SCALE: [i64; 2 * DIGITS - 1] = [1, BASE, BASE * BASE, BASE * BASE * BASE, BASE * BASE * BASE * BASE];

// Three digits cover exactly the codes: 40 · (1 + 81 + 81²) = 265,720 = WLEVELS.
const _: () = assert!(DMAX * (1 + BASE + BASE * BASE) == WLEVELS);
// A digit fits `int8`.
const _: () = assert!(DMAX <= i8::MAX as i64);
// The flushed `int32` accumulator cannot overflow: KT_FLUSH tiles of at most TILE · DMAX² each.
const _: () = assert!(KT_FLUSH > 0 && KT_FLUSH as i64 * TILE as i64 * DMAX * DMAX <= i32::MAX as i64);
// One tile more would overflow, so the flush is placed at the bound and not below it.
const _: () = assert!((KT_FLUSH as i64 + 1) * TILE as i64 * DMAX * DMAX > i32::MAX as i64);
// Every partial sum of the recombination is at most K_MAX · WLEVELS² in magnitude, which fits `i64`.
const _: () = assert!(K_MAX as i128 * WLEVELS as i128 * WLEVELS as i128 <= i64::MAX as i128);

/// The three balanced base-81 digits of a code, low digit first. `None` outside `±WLEVELS`, where three digits do not
/// suffice.
pub fn split(v: i32) -> Option<[i8; DIGITS]> {
    if (v as i64).abs() > WLEVELS {
        return None;
    }
    let (mut d, mut r) = ([0i8; DIGITS], v);
    for slot in &mut d {
        let lo = (r + DMAX as i32).rem_euclid(BASE as i32) - DMAX as i32;
        *slot = lo as i8;
        r = (r - lo) / BASE as i32; // exact: r − lo is a multiple of 81
    }
    debug_assert_eq!(r, 0, "split: {v} needs a fourth digit");
    Some(d)
}

/// A split that drops the carry out of a rebalanced low digit (the `WrongCarry` plant).
#[cfg(feature = "plants")]
pub fn split_wrong_carry(v: i32) -> Option<[i8; DIGITS]> {
    if (v as i64).abs() > WLEVELS {
        return None;
    }
    let (mut d, mut r) = ([0i8; DIGITS], v);
    for slot in &mut d {
        *slot = ((r + DMAX as i32).rem_euclid(BASE as i32) - DMAX as i32) as i8;
        r /= BASE as i32; // the plant: truncates, forgetting the carry when the digit was rebalanced
    }
    Some(d)
}

/// The value of three digits.
pub fn join(d: [i8; DIGITS]) -> i64 {
    d.iter().rev().fold(0i64, |acc, &x| acc * BASE + x as i64)
}

/// Offset of one packed tile: row tile `rt`, k tile `kt` of `kts`, digit plane `digit`. Tiles are 256 bytes, so every
/// tile starts on a 256-byte boundary of the buffer.
pub fn tile_offset(rt: usize, kt: usize, digit: usize, kts: usize) -> usize {
    ((rt * kts + kt) * DIGITS + digit) * TILE_BYTES
}

/// Pack `rows × k` codes (row-major) into digit tiles, `[row tile][k tile][digit][16][16]`, rows and `k` zero-padded to
/// 16. Within a tile, row `r` and column `c` sit at `r·16 + c`. For activations a row is an input row (WMMA `matrix_a`,
/// row-major, `ldm` 16); for weights a row is an output, so the tile is `Wᵀ` column-major (`matrix_b`, `ldm` 16).
/// `None` if a code is outside `±WLEVELS`.
pub fn pack(codes: &[i32], rows: usize, k: usize, split: fn(i32) -> Option<[i8; DIGITS]>) -> Option<Vec<i8>> {
    assert_eq!(codes.len(), rows * k, "pack: codes are not rows·k");
    let kts = k.div_ceil(TILE);
    let tile_row = kts * DIGITS * TILE_BYTES; // the bytes of one row tile
    let mut out = vec![0i8; rows.div_ceil(TILE) * tile_row];
    let ok = crate::host::par_fill(&mut out, tile_row, rows * k, |rt, block| {
        for r in rt * TILE..rows.min(rt * TILE + TILE) {
            for (c, &v) in codes[r * k..(r + 1) * k].iter().enumerate() {
                let Some(d) = split(v) else { return false };
                let at = tile_offset(0, c / TILE, 0, kts) + (r % TILE) * TILE + c % TILE;
                for (i, &di) in d.iter().enumerate() {
                    block[at + i * TILE_BYTES] = di;
                }
            }
        }
        true
    });
    ok.then_some(out)
}
