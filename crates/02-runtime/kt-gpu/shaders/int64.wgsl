// tritify — EMULATED signed 64-bit integers for WGSL, which has no i64. A value is vec2<u32> (lo, hi) in two's
// complement. Exact, and portable to any shader device (no 64-bit-integer extension needed). Prepended by kt-gpu to
// every kernel that accumulates in 64 bits.

// unsigned 32×32 → 64  (lo, hi)
fn mul_u_wide(a : u32, b : u32) -> vec2<u32> {
    let al = a & 0xFFFFu; let ah = a >> 16u;
    let bl = b & 0xFFFFu; let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;
    var lo = ll;
    var hi = hh;
    var carry : u32 = 0u;
    let m1 = lh << 16u; let s1 = lo + m1; if (s1 < lo) { carry = carry + 1u; } lo = s1;
    let m2 = hl << 16u; let s2 = lo + m2; if (s2 < lo) { carry = carry + 1u; } lo = s2;
    hi = hi + (lh >> 16u) + (hl >> 16u) + carry;
    return vec2<u32>(lo, hi);
}
fn add64(a : vec2<u32>, b : vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    let carry = select(0u, 1u, lo < a.x);
    return vec2<u32>(lo, a.y + b.y + carry);
}
fn neg64(a : vec2<u32>) -> vec2<u32> {
    let lo = ~a.x + 1u;
    let carry = select(0u, 1u, lo == 0u);
    return vec2<u32>(lo, ~a.y + carry);
}
// signed 32×32 → 64
fn mul_i_wide(a : i32, b : i32) -> vec2<u32> {
    var p = mul_u_wide(u32(abs(a)), u32(abs(b)));
    if ((a < 0) != (b < 0)) { p = neg64(p); }
    return p;
}
