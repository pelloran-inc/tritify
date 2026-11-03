// tritify — TILED / shared-memory ternary matmul  Y = X·Wᵀ  (i32 exact). A real GEMM kernel (not
// one-thread-per-output): each 16×16 workgroup computes a 16×16 tile of Y, streaming 16-wide K-tiles of X and W
// through workgroup shared memory so each global element is read once per tile and reused 16× from fast on-chip
// memory. Valid (bit-identical to the CPU reference) whenever the accumulation fits i32.
//
// X = qx [M][K] row-major, W = qw [N][K] row-major (Wᵀ), Y = yout [M][N] i32.

struct Meta { m : u32, n : u32, k : u32, _pad : u32, }
@group(0) @binding(0) var<storage, read>       qx   : array<i32>;
@group(0) @binding(1) var<storage, read>       qw   : array<i32>;
@group(0) @binding(2) var<storage, read_write> yout : array<i32>;
@group(0) @binding(3) var<uniform>             params : Meta;

const T : u32 = 16u;
var<workgroup> xs : array<i32, 256>;   // 16×16 tile of X
var<workgroup> ws : array<i32, 256>;   // 16×16 tile of W (Wᵀ layout: [n_local][k])

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x * T + lid.x;   // m
    let col = wid.y * T + lid.y;   // n
    let m = params.m; let n = params.n; let k = params.k;
    var acc : i32 = 0;
    let ntiles = (k + T - 1u) / T;
    for (var t : u32 = 0u; t < ntiles; t = t + 1u) {
        let kx = t * T + lid.y;    // X column for this thread's load
        xs[lid.x * T + lid.y] = select(0, qx[row * k + kx], row < m && kx < k);
        let kw = t * T + lid.x;    // W column (k) for this thread's load
        ws[lid.y * T + lid.x] = select(0, qw[col * k + kw], col < n && kw < k);
        workgroupBarrier();
        for (var i : u32 = 0u; i < T; i = i + 1u) {
            acc = acc + xs[lid.x * T + i] * ws[lid.y * T + i];
        }
        workgroupBarrier();
    }
    if (row < m && col < n) { yout[row * n + col] = acc; }
}
