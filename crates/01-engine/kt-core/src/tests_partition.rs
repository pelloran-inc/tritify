//! The CPU linear layer's partition. Every partition of the flat output gives the serial loop's integers,
//! including the degenerate shapes, and the thread plan follows the work and the cap.

use crate::backend::{dot_portable, fill_partitioned, linear_cpu, plan_threads, SERIAL_BELOW, WORK_PER_SPAWN};
use crate::tests::lcg;

/// Codes in `±2^20`, enough to exercise every lane of a widening multiply-accumulate.
fn codes(n: usize, seed: u64) -> Vec<i32> {
    let mut s = seed;
    (0..n).map(|_| ((lcg(&mut s) >> 33) % (1 << 21)) as i32 - (1 << 20)).collect()
}

/// The reference: two nested loops, written independently of the flat index arithmetic.
fn serial(x: &[i32], w: &[i32], rows: usize, out: usize, k: usize) -> Vec<i64> {
    let mut y = Vec::with_capacity(rows * out);
    for r in 0..rows {
        for o in 0..out {
            y.push((0..k).map(|i| x[r * k + i] as i64 * w[o * k + i] as i64).sum());
        }
    }
    y
}

#[test]
fn every_partition_gives_the_serial_loop() {
    for rows in [0usize, 1, 2, 3, 7, 16, 31] {
        for out in [0usize, 1, 2, 17, 64] {
            for k in [0usize, 1, 5, 33] {
                let (x, w) = (codes(rows * k, 1 + rows as u64), codes(out * k, 7 + out as u64));
                let want = serial(&x, &w, rows, out, k);
                for threads in [1usize, 2, 3, 16, 64] {
                    let mut y = vec![-1i64; rows * out];
                    fill_partitioned(&mut y, threads, |i| dot_portable(&x[i / out * k..][..k], &w[i % out * k..][..k]));
                    assert_eq!(y, want, "{rows}×{out}×{k} on {threads} threads");
                }
                assert_eq!(linear_cpu(&x, &w, rows, out, k, dot_portable), want, "{rows}×{out}×{k}, planned");
            }
        }
    }
}

#[test]
fn a_one_row_call_large_enough_is_split_over_its_outputs() {
    // one row, 4864 outputs, k = 896: √(4,358,144 / 2^17) = 5 workers at most
    let (rows, out, k) = (1usize, 4864usize, 896usize);
    let t = plan_threads(rows * out, rows * out * k);
    assert!(t <= 5, "{t} workers where the square-root rule allows 5");
    let (x, w) = (codes(rows * k, 3), codes(out * k, 4));
    assert_eq!(linear_cpu(&x, &w, rows, out, k, dot_portable), serial(&x, &w, rows, out, k));
}

#[test]
fn the_plan_follows_the_work_and_never_exceeds_the_outputs() {
    assert_eq!(plan_threads(0, 0), 1);
    assert_eq!(plan_threads(1, usize::MAX), 1, "one output, one worker");
    assert_eq!(plan_threads(1 << 20, SERIAL_BELOW - 1), 1, "below the floor: serial");
    assert_eq!(plan_threads(1 << 20, 896 * 896), 1, "Qwen's 1×896×896 stays serial");
    assert!(plan_threads(1 << 20, 9 * WORK_PER_SPAWN) <= 3, "√9 = 3");
    assert!(plan_threads(1 << 20, 10_000 * WORK_PER_SPAWN) <= 100, "√10,000 = 100: the square root, not the work");
    assert!(plan_threads(2, usize::MAX) <= 2, "never more workers than outputs");
}
