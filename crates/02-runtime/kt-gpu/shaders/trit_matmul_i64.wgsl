// tritify — ternary matmul  Y = X · Wᵀ  with EMULATED 64-bit exact accumulation (portable WGSL).
//
// WGSL core has no 64-bit integers. For the full-lossless regime (both operands many-trit) a single product qx·qw can
// exceed i32, so the accumulator is the emulated signed 64-bit of int64.wgsl (prepended by kt-gpu). One thread per
// output Y[m,n]; the result is written as two i32 words (lo, hi); the host reconstructs i64 = (hi<<32)|lo. This is
// the accumulator for the lossless/verifiable regime; the i32 kernel (trit_matmul_tiled.wgsl) is the fast path for
// the ternary regime whenever the accumulation fits i32, and trit_matmul_i64_splitk.wgsl serves products with few outputs.

struct Meta { m : u32, n : u32, k : u32, _pad : u32, }

@group(0) @binding(0) var<storage, read>       qx   : array<i32>;   // M*K
@group(0) @binding(1) var<storage, read>       qw   : array<i32>;   // N*K
@group(0) @binding(2) var<storage, read_write> yout : array<i32>;   // 2*M*N  (lo,hi per element)
@group(0) @binding(3) var<uniform>             params : Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.m * params.n;
    if (idx >= total) { return; }
    let xb = (idx / params.n) * params.k;
    let wb = (idx % params.n) * params.k;
    var acc = vec2<u32>(0u, 0u);
    for (var t : u32 = 0u; t < params.k; t = t + 1u) {
        acc = add64(acc, mul_i_wide(qx[xb + t], qw[wb + t]));
    }
    yout[2u * idx]      = i32(acc.x);   // lo
    yout[2u * idx + 1u] = i32(acc.y);   // hi
}
