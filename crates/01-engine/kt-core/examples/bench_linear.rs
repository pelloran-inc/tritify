//! Dev benchmark (NOT shipped): time QuantLinear::forward_int at SAM/decoder-representative dims.
use kt_core::layers::QuantLinear;
use kt_core::{IntTensor, Module};
use std::time::Instant;
fn main() {
    let (out, inf, rows) = (896usize, 896usize, 4096usize); // SAM hidden × token count
    let w: Vec<f32> = (0..out * inf).map(|i| ((i.wrapping_mul(2654435761) % 2000) as f32 - 1000.0) / 1000.0).collect();
    let lin = QuantLinear::from_weights("bench", &w, out, inf);
    let xr: Vec<f64> = (0..rows * inf).map(|i| ((i.wrapping_mul(40503) % 2000) as f64 - 1000.0) / 1000.0).collect();
    let x = IntTensor::from_reals(&xr, vec![rows, inf], 1 << 28);
    std::hint::black_box(&lin.forward_int(&x)); // warmup
    let reps = 5;
    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(&lin.forward_int(&x));
    }
    println!("QuantLinear {out}x{inf} rows={rows}: {:.1} ms/call", t.elapsed().as_secs_f64() * 1000.0 / reps as f64);
}
