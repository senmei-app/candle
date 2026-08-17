// Simple row-major GEMM (A: m x k, B: k x n -> C: m x n) used by the HIP backend.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float h2f(float x) { return x; }
__device__ __forceinline__ float h2f(double x) { return (float)x; }
__device__ __forceinline__ float h2f(__half x) { return __half2float(x); }
__device__ __forceinline__ float h2f(__hip_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float h2f(int8_t x) { return (float)x; }
__device__ __forceinline__ float h2f(uint8_t x) { return (float)x; }
__device__ __forceinline__ float h2f(int16_t x) { return (float)x; }
__device__ __forceinline__ float h2f(int32_t x) { return (float)x; }
__device__ __forceinline__ float h2f(int64_t x) { return (float)x; }
__device__ __forceinline__ float h2f(uint32_t x) { return (float)x; }

template <typename T> __device__ __forceinline__ T f2h(float x);
template <> __device__ __forceinline__ float f2h<float>(float x) { return x; }
template <> __device__ __forceinline__ double f2h<double>(float x) { return (double)x; }
template <> __device__ __forceinline__ __half f2h<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __hip_bfloat16 f2h<__hip_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template <typename T>
__device__ void gemm_nt(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float beta) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = m * n;
    if (idx >= total) return;
    const int row = idx / n;
    const int col = idx % n;
    const T* arow = A + (long)row * k;
    const T* bcol = B + col;
    float acc = 0.f;
    for (int i = 0; i < k; ++i) {
        acc += h2f(arow[i]) * h2f(bcol[(long)i * n]);
    }
    const float cval = (beta != 0.f) ? h2f(C[idx]) : 0.f;
    C[idx] = f2h<T>(alpha * acc + beta * cval);
}

#define INSTANTIATE_GEMM(T, name)                                       \
    extern "C" __global__ void name(const T* A, const T* B, T* C,       \
                                    const int m, const int n, const int k, \
                                    const float alpha, const float beta) { \
        gemm_nt<T>(A, B, C, m, n, k, alpha, beta);                       \
    }

INSTANTIATE_GEMM(float, gemm_f32)
INSTANTIATE_GEMM(double, gemm_f64)
INSTANTIATE_GEMM(__half, gemm_f16)
INSTANTIATE_GEMM(__hip_bfloat16, gemm_bf16)
