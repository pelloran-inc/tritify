//! tritify — deterministic, bit-identical, zero-float ternary matmul demo.
//!
//! Runs a ternary GEMM  Y = X·Wᵀ  three ways on the SAME integer data:
//!   CPU  i64  : exact integer accumulate                      (the reference)
//!   GPU  i32  : 16×16 tiled shared-memory kernel              (fast path; valid when the sum fits i32)
//!   GPU  i64  : emulated-64-bit, one thread per output        (lossless path; any magnitude, any device)
//! GPU outputs are gated BIT-IDENTICAL to the CPU reference; then we print a SHA-256 of the output.
//! There is NO floating point anywhere, and the hash reproduces run-to-run and machine-to-machine. Both GPU kernels
//! live in `kt-gpu`, the same backend the engine runs on.
//!
//! Zero-setup: `cargo run --release -p kt-demo` uses a deterministic seeded ternary input, so it runs with no download.
//! Env: M=, N=, K=, SEED= override the dims/seed; TILE=/path/to.tile loads a real exported tile instead
//!      (format: [M,N,K : i32 LE][qx : M*K i32][qw : N*K i32]).

use kt_gpu::Gpu;
use sha2::{Digest, Sha256};

// ---- deterministic input (PCG-style LCG; identical bytes on every machine) ----
fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}
fn gen_input(m: usize, n: usize, k: usize, seed: u64) -> (Vec<i32>, Vec<i32>) {
    let mut s = seed ^ 0x9E3779B97F4A7C15;
    let qx: Vec<i32> = (0..m * k).map(|_| ((lcg(&mut s) >> 33) % 255) as i32 - 127).collect(); // [-127,127]
    let qw: Vec<i32> = (0..n * k).map(|_| ((lcg(&mut s) >> 33) % 3) as i32 - 1).collect(); // {-1,0,1} ternary
    (qx, qw)
}
fn rd_i32(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn load_tile(path: &str) -> (usize, usize, usize, Vec<i32>, Vec<i32>) {
    let bytes = std::fs::read(path).expect("read TILE");
    let (m, n, k) = (rd_i32(&bytes, 0) as usize, rd_i32(&bytes, 4) as usize, rd_i32(&bytes, 8) as usize);
    let base = 12;
    let qx = (0..m * k).map(|i| rd_i32(&bytes, base + i * 4)).collect();
    let qw = (0..n * k).map(|i| rd_i32(&bytes, base + (m * k + i) * 4)).collect();
    (m, n, k, qx, qw)
}

// ---- CPU reference: exact integer accumulate (one output element = one left-to-right i64 sum) ----
fn cpu_ref(qx: &[i32], qw: &[i32], m: usize, n: usize, k: usize) -> Vec<i64> {
    let mut y = vec![0i64; m * n];
    for r in 0..m {
        let xb = r * k;
        for c in 0..n {
            let wb = c * k;
            let mut a = 0i64;
            for i in 0..k {
                a += qx[xb + i] as i64 * qw[wb + i] as i64;
            }
            y[r * n + c] = a;
        }
    }
    y
}
fn hash_i64(y: &[i64]) -> String {
    let mut h = Sha256::new();
    for &v in y {
        h.update(v.to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() {
    let envn = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let seed: u64 = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(42);
    let (m, n, k, qx, qw, src) = if let Ok(p) = std::env::var("TILE") {
        let (m, n, k, qx, qw) = load_tile(&p);
        (m, n, k, qx, qw, format!("TILE {}", p))
    } else {
        let (m, n, k) = (envn("M", 64), envn("N", 768), envn("K", 768));
        let (qx, qw) = gen_input(m, n, k, seed);
        (m, n, k, qx, qw, format!("seeded ternary GEMM (seed={seed})"))
    };

    let y_ref = cpu_ref(&qx, &qw, m, n, k);
    let acc_fits_i32 = y_ref.iter().all(|&v| v >= i32::MIN as i64 && v <= i32::MAX as i64);
    let hash = hash_i64(&y_ref);
    // `--plant`: compare the GPU against a CPU result with one element off by one, so the check must fail
    let mut y_cmp = y_ref.clone();
    if std::env::args().any(|a| a == "--plant") {
        y_cmp[0] += 1;
    }
    let mut mismatch = false;

    println!("tritify — deterministic exact-integer ternary inference (demo)");
    println!("  input      : {src}");
    println!("  dims       : M={m}  N={n}  K={k}   Y = X · Wᵀ");
    println!("  arithmetic : INTEGER ONLY — zero floating point");

    match Gpu::open(None) {
        Some(g) => {
            println!("  gpu        : {} / {}", g.info().backend, g.info().name);
            // tiled i32 (valid only if the accumulation fits i32)
            if acc_fits_i32 {
                let out = g.gemm_i32_tiled(&qx, &qw, m, n, k);
                let ok = out.len() == y_cmp.len() && out.iter().zip(&y_cmp).all(|(&a, &b)| a as i64 == b);
                mismatch |= !ok;
                println!("    tiled i32  : {}", if ok { "BIT-IDENTICAL to CPU ✓" } else { "MISMATCH ✗" });
            } else {
                println!("    tiled i32  : skipped (sum exceeds i32; use the i64 path)");
            }
            // emulated-i64 (lossless, any magnitude)
            let ok64 = g.gemm_i64(&qx, &qw, m, n, k) == y_cmp;
            mismatch |= !ok64;
            println!(
                "    emul. i64  : {}  (lossless, any device)",
                if ok64 { "BIT-IDENTICAL to CPU ✓" } else { "MISMATCH ✗" }
            );
        }
        None => println!("  gpu        : no adapter found — CPU-only (the hash below is still machine-independent)"),
    }

    println!("  reproducing hash (sha256 of the output): {hash}");
    if mismatch {
        eprintln!("FAIL: a GPU result differs from the CPU's");
        std::process::exit(1);
    }
    println!("  → run again, or on another computer: the hash is identical. No float. No trick.");
}
