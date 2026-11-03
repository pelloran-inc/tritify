// tritify — SPLIT-K exact matmul  Y = X · Wᵀ  for products with few outputs (decode: one row, long dot products). One workgroup
// per output: its 64 threads each sum a strided slice of k in emulated 64-bit (int64.wgsl, prepended by kt-gpu), then a
// tree reduction in workgroup memory, also emulated 64-bit, finishes the sum. Integer addition is associative, so any
// split and any reduction order give the same i64 as the one-thread-per-output kernel, bit for bit.
//
// Dispatch: m·n workgroups laid out (gx, gy) to stay within the per-dimension limit; the output index is
// wid.x + wid.y·gx. The result is written as two i32 words (lo, hi).

struct Meta { m : u32, n : u32, k : u32, _pad : u32, }

@group(0) @binding(0) var<storage, read>       qx   : array<i32>;   // M*K
@group(0) @binding(1) var<storage, read>       qw   : array<i32>;   // N*K
@group(0) @binding(2) var<storage, read_write> yout : array<i32>;   // 2*M*N  (lo,hi per element)
@group(0) @binding(3) var<uniform>             params : Meta;

const WG : u32 = 64u;
var<workgroup> part : array<vec2<u32>, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = wid.x + wid.y * nwg.x;
    let live = idx < params.m * params.n;   // the same for the whole workgroup, so the barriers stay uniform
    var acc = vec2<u32>(0u, 0u);
    if (live) {
        let xb = (idx / params.n) * params.k;
        let wb = (idx % params.n) * params.k;
        for (var t : u32 = lid.x; t < params.k; t = t + WG) {
            acc = add64(acc, mul_i_wide(qx[xb + t], qw[wb + t]));
        }
    }
    // PLANT-HOOK
    part[lid.x] = acc;
    workgroupBarrier();
    for (var s : u32 = WG / 2u; s > 0u; s = s / 2u) {
        if (lid.x < s) { part[lid.x] = add64(part[lid.x], part[lid.x + s]); }
        workgroupBarrier();
    }
    if (live && lid.x == 0u) {
        yout[2u * idx]      = i32(part[0].x);   // lo
        yout[2u * idx + 1u] = i32(part[0].y);   // hi
    }
}
