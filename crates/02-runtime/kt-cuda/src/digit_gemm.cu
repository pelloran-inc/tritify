// The digit-plane product on Tensor Cores (the `cuda` feature of kt-cuda).
//
// Exact by construction: int8 × int8 → int32 tile products (WMMA m16n16k16, compute capability ≥ 7.2), and the nine
// int32 sums of an output tile added into int64 as Σ 81^(i+j) · S_ij every `kt_flush` k-tiles. The packed layout and
// every bound are fixed in Rust (src/digits.rs); this file only reads them:
// - a packed operand is [row tile][k tile][digit][16][16] bytes, each 16 × 16 tile 256 bytes and 256-byte aligned
// - activation tiles are matrix_a, row-major (A[r][k] at r·16 + k); weight tiles are matrix_b, column-major
//   (B[k][o] at o·16 + k), because a weight row is one output
// - kt_flush · 16 · 40² ≤ 2^31 − 1, so no int32 sum wraps (asserted in Rust at compile time)
// No floating-point type appears in this file.

#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <string.h>

using namespace nvcuda;

#define TILE 16
#define DIGITS 3
#define TILE_BYTES (TILE * TILE)
#define WARPS 4          // warps per block; each warp computes one 16 × 16 output tile
#define MAX_GRID_Y 65535 // row tiles per launch

__device__ __forceinline__ const signed char* tile_at(const signed char* base, long long rt, long long kt, int digit,
                                                      long long kts) {
    return base + ((rt * kts + kt) * DIGITS + digit) * TILE_BYTES;
}

template <bool DROP22>
__global__ void digit_gemm(const signed char* __restrict__ xa, const signed char* __restrict__ wb,
                           long long* __restrict__ y, int rows, int out, int nts, int kts, int kt_flush, int rt0) {
    __shared__ __align__(32) int stage[WARPS][TILE_BYTES];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int nt = blockIdx.x * WARPS + warp, rt = rt0 + blockIdx.y;
    if (nt >= nts) return; // a whole warp leaves; no block-wide barrier follows
    const long long scale[2 * DIGITS - 1] = {1LL, 81LL, 6561LL, 531441LL, 43046721LL};
    long long acc[TILE_BYTES / 32];
#pragma unroll
    for (int e = 0; e < TILE_BYTES / 32; ++e) acc[e] = 0;

    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, signed char, wmma::row_major> a[DIGITS];
    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, signed char, wmma::col_major> b[DIGITS];
    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> s[DIGITS][DIGITS];

    int k0 = 0;
    while (k0 < kts) {
        const int k1 = (kt_flush >= kts - k0) ? kts : k0 + kt_flush; // no overflow when kt_flush is INT_MAX
#pragma unroll
        for (int i = 0; i < DIGITS; ++i)
#pragma unroll
            for (int j = 0; j < DIGITS; ++j) wmma::fill_fragment(s[i][j], 0);
        for (int kt = k0; kt < k1; ++kt) {
#pragma unroll
            for (int d = 0; d < DIGITS; ++d) {
                wmma::load_matrix_sync(a[d], tile_at(xa, rt, kt, d, kts), TILE);
                wmma::load_matrix_sync(b[d], tile_at(wb, nt, kt, d, kts), TILE);
            }
#pragma unroll
            for (int i = 0; i < DIGITS; ++i)
#pragma unroll
                for (int j = 0; j < DIGITS; ++j) wmma::mma_sync(s[i][j], a[i], b[j], s[i][j]);
        }
        // flush: each lane owns the elements lane, lane + 32, …, lane + 224 of the 16 × 16 output tile
#pragma unroll
        for (int i = 0; i < DIGITS; ++i)
#pragma unroll
            for (int j = 0; j < DIGITS; ++j) {
                if (DROP22 && i == 2 && j == 2) continue; // the planted defect only
                wmma::store_matrix_sync(stage[warp], s[i][j], TILE, wmma::mem_row_major);
                __syncwarp();
#pragma unroll
                for (int e = 0; e < TILE_BYTES / 32; ++e) acc[e] += scale[i + j] * (long long)stage[warp][lane + 32 * e];
                __syncwarp();
            }
        k0 = k1;
    }
#pragma unroll
    for (int e = 0; e < TILE_BYTES / 32; ++e) {
        const int idx = lane + 32 * e;
        const long long r = (long long)rt * TILE + idx / TILE, c = (long long)nt * TILE + idx % TILE;
        if (r < rows && c < out) y[r * out + c] = acc[e];
    }
}

extern "C" {

int kt_cuda_device_count(int* n) {
    *n = 0;
    return (int)cudaGetDeviceCount(n);
}

int kt_cuda_device_info(int dev, char* name, int len, int* major, int* minor, size_t* free_bytes) {
    cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, dev);
    if (e != cudaSuccess) return (int)e;
    strncpy(name, p.name, (size_t)len - 1);
    name[len - 1] = 0;
    *major = p.major;
    *minor = p.minor;
    if ((e = cudaSetDevice(dev)) != cudaSuccess) return (int)e;
    size_t total = 0;
    return (int)cudaMemGetInfo(free_bytes, &total);
}

int kt_cuda_alloc(int dev, size_t bytes, void** p) {
    cudaError_t e = cudaSetDevice(dev);
    return (int)(e != cudaSuccess ? e : cudaMalloc(p, bytes));
}

int kt_cuda_free(int dev, void* p) {
    cudaError_t e = cudaSetDevice(dev);
    return (int)(e != cudaSuccess ? e : cudaFree(p));
}

int kt_cuda_upload(int dev, void* dst, const void* src, size_t bytes) {
    cudaError_t e = cudaSetDevice(dev);
    return (int)(e != cudaSuccess ? e : cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

// One product: the packed activations go up into `xa_dev`, the kernel writes `y_dev`, and `y_host` (rows × out) comes
// back. The device buffers are the caller's and large enough. Returns a cudaError_t.
int kt_cuda_digit_gemm(int dev, const signed char* xa_host, size_t xa_bytes, void* xa_dev, const void* wb_dev,
                       void* y_dev, long long* y_host, int rows, int out, int kts, int kt_flush, int drop22) {
    cudaError_t e = cudaSetDevice(dev);
    if (e != cudaSuccess) return (int)e;
    if (rows <= 0 || out <= 0) return (int)cudaSuccess;
    if (xa_bytes > 0 && (e = cudaMemcpy(xa_dev, xa_host, xa_bytes, cudaMemcpyHostToDevice)) != cudaSuccess) return (int)e;
    const int mts = (rows + TILE - 1) / TILE, nts = (out + TILE - 1) / TILE;
    for (int rt0 = 0; rt0 < mts; rt0 += MAX_GRID_Y) {
        const dim3 grid((nts + WARPS - 1) / WARPS, (mts - rt0 < MAX_GRID_Y) ? mts - rt0 : MAX_GRID_Y);
        const signed char* xa = (const signed char*)xa_dev;
        const signed char* wb = (const signed char*)wb_dev;
        long long* y = (long long*)y_dev;
        if (drop22)
            digit_gemm<true><<<grid, WARPS * 32>>>(xa, wb, y, rows, out, nts, kts, kt_flush, rt0);
        else
            digit_gemm<false><<<grid, WARPS * 32>>>(xa, wb, y, rows, out, nts, kts, kt_flush, rt0);
        if ((e = cudaGetLastError()) != cudaSuccess) return (int)e;
    }
    // the copy waits for the kernels on the default stream
    return (int)cudaMemcpy(y_host, y_dev, (size_t)rows * (size_t)out * sizeof(long long), cudaMemcpyDeviceToHost);
}

const char* kt_cuda_error_string(int e) { return cudaGetErrorString((cudaError_t)e); }

} // extern "C"
