use crate::digits::{join, pack, split, tile_offset, DIGITS, KT_FLUSH, K_MAX, TILE, TILE_BYTES};
use crate::{emulator, fallbacks};
use kt_core::backend::{LinearBackend, Portable, WeightId, Weights};
use kt_core::layers::WLEVELS;

/// Codes uniform in `±WLEVELS` from a seeded LCG.
fn codes(n: usize, mut s: u64) -> Vec<i32> {
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % (2 * WLEVELS as u64 + 1)) as i32 - WLEVELS as i32
        })
        .collect()
}

/// The smallest `k` beyond one flush: `KT_FLUSH + 1` tiles. With every code +`WLEVELS`, each digit product is 1,600,
/// so an unflushed `int32` sum would reach 1,342,192 · 1,600 = 2,147,507,200 > 2^31 − 1.
const K_BEYOND: usize = (KT_FLUSH + 1) * TILE;

#[test]
fn every_code_splits_into_three_digits_and_back() {
    let lv = WLEVELS as i32;
    for v in -lv..=lv {
        let d = split(v).expect("a code in range splits");
        assert!(d.iter().all(|x| x.abs() <= 40), "{v}: {d:?}");
        assert_eq!(join(d), v as i64);
    }
    assert_eq!(split(lv), Some([40, 40, 40]));
    assert_eq!(split(-lv), Some([-40, -40, -40]));
    for v in [lv + 1, -lv - 1, i32::MAX, i32::MIN] {
        assert_eq!(split(v), None, "{v} is outside ±WLEVELS");
    }
}

#[test]
fn packing_puts_each_digit_where_the_tile_rules_read_it() {
    let (rows, k) = (17, 35);
    let c = codes(rows * k, 3);
    let p = pack(&c, rows, k, split).expect("in range");
    let kts = k.div_ceil(TILE);
    assert_eq!(p.len(), rows.div_ceil(TILE) * kts * DIGITS * TILE_BYTES);
    for r in 0..rows.div_ceil(TILE) * TILE {
        for col in 0..kts * TILE {
            let d = if r < rows && col < k { split(c[r * k + col]).unwrap() } else { [0; DIGITS] };
            for (i, &di) in d.iter().enumerate() {
                let at = tile_offset(r / TILE, col / TILE, i, kts) + r % TILE * TILE + col % TILE;
                assert_eq!(p[at], di, "row {r}, column {col}, digit {i}");
            }
        }
    }
}

#[test]
fn the_emulator_equals_portable_on_tails_and_extremes() {
    let emu = emulator();
    for (rows, out, k) in [(1, 1, 1), (3, 5, 7), (2, 17, 33), (16, 16, 16), (17, 33, 50), (1, 9, 4099), (20, 40, 300)] {
        let (x, w) = (codes(rows * k, rows as u64), codes(out * k, 7 + out as u64));
        assert_eq!(emu.linear(&x, &w, rows, out, k), Portable.linear(&x, &w, rows, out, k), "{rows}×{out}×{k}");
    }
    let (lv, k) = (WLEVELS as i32, 1027);
    let x: Vec<i32> = (0..2 * k).map(|i| if i % 3 == 0 { -lv } else { lv }).collect();
    let w: Vec<i32> = (0..3 * k).map(|i| if i % 2 == 0 { lv } else { -lv }).collect();
    assert_eq!(emu.linear(&x, &w, 2, 3, k), Portable.linear(&x, &w, 2, 3, k), "extremes ±WLEVELS");
}

#[test]
fn a_k_beyond_one_flush_is_exact() {
    assert_eq!(K_BEYOND, 1_342_192);
    let lv = WLEVELS as i32;
    let x = vec![lv; K_BEYOND];
    let w = [vec![lv; K_BEYOND], vec![-lv; K_BEYOND]].concat();
    let want = K_BEYOND as i64 * WLEVELS * WLEVELS;
    let y = emulator().linear(&x, &w, 1, 2, K_BEYOND);
    assert_eq!(y, vec![want, -want]);
    assert_eq!(y, Portable.linear(&x, &w, 1, 2, K_BEYOND));
}

#[test]
fn calls_outside_the_bounds_go_to_portable_and_are_counted() {
    let before = fallbacks();
    let (x, w) = (vec![WLEVELS as i32 + 1, 2], vec![3, 4]);
    assert_eq!(emulator().linear(&x, &w, 1, 1, 2), Portable.linear(&x, &w, 1, 1, 2));
    let (x, w) = (vec![1, 2], vec![i32::MIN, 4]);
    let wts = Weights { id: WeightId::fresh(), codes: &w };
    assert_eq!(emulator().linear_weights(&x, wts, 1, 1, 2), Portable.linear(&x, &w, 1, 1, 2));
    assert!(fallbacks() >= before + 2);
    // K_MAX is the last k whose worst sum fits i64; one more would not
    assert_eq!(K_MAX, 130_629_492);
    assert!((K_MAX as i128 + 1) * (WLEVELS as i128).pow(2) > i64::MAX as i128);
}

#[test]
fn resident_weights_give_portable_through_repeats_twins_and_eviction() {
    let (x, pair) =
        (codes(4 * 1023, 1), [(WeightId::fresh(), codes(64 * 1023, 11)), (WeightId::fresh(), codes(64 * 1023, 12))]);
    for (b, resident) in [(emulator(), true), (emulator().with_budget(1), false)] {
        for round in 0..3 {
            for (i, (id, w)) in pair.iter().enumerate() {
                let got = b.linear_weights(&x, Weights { id: *id, codes: w }, 4, 64, 1023);
                assert_eq!(got, Portable.linear(&x, w, 4, 64, 1023), "{}: twin {i}, round {round}", b.name());
            }
        }
        assert_eq!(b.resident_bytes() > 0, resident, "{}", b.name());
    }
}

#[cfg(feature = "plants")]
#[test]
fn every_plant_fires_where_it_should() {
    use crate::{HostExec, Plant, TcLinear};
    let planted = |p| TcLinear::planted(HostExec, p);
    let (x, w) = (codes(4 * 300, 5), codes(20 * 300, 6));
    let want = Portable.linear(&x, &w, 4, 20, 300);
    assert_ne!(planted(Plant::DropPlane22).linear(&x, &w, 4, 20, 300), want, "dropping the (2,2) plane");
    assert_ne!(planted(Plant::WrongCarry).linear(&x, &w, 4, 20, 300), want, "a wrong carry");
    // the flush matters only beyond its bound: a short k is exact without it, one tile more is not
    assert_eq!(planted(Plant::NoFlush).linear(&x, &w, 4, 20, 300), want, "no flush, short k");
    let (lv, xl) = (WLEVELS as i32, vec![WLEVELS as i32; K_BEYOND]);
    let wl = vec![lv; K_BEYOND];
    assert_ne!(planted(Plant::NoFlush).linear(&xl, &wl, 1, 1, K_BEYOND), Portable.linear(&xl, &wl, 1, 1, K_BEYOND));
    let b = planted(Plant::CacheByShape);
    let (twin0, twin1) = (codes(64 * 50, 21), codes(64 * 50, 22));
    let xs = codes(2 * 50, 23);
    assert_eq!(
        b.linear_weights(&xs, Weights { id: WeightId::fresh(), codes: &twin0 }, 2, 64, 50),
        Portable.linear(&xs, &twin0, 2, 64, 50)
    );
    assert_ne!(
        b.linear_weights(&xs, Weights { id: WeightId::fresh(), codes: &twin1 }, 2, 64, 50),
        Portable.linear(&xs, &twin1, 2, 64, 50)
    );
}
