// HIP compatibility layer for the candle CUDA kernels.
// Maps the small CUDA surface used by candle-kernels onto ROCm/HIP.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

// ROCm's libhipcxx refers to the fp8 types under their CUDA names.
typedef __hip_fp8_e4m3 __nv_fp8_e4m3;
typedef __hip_fp8_e5m2 __nv_fp8_e5m2;

// __CUDA_ARCH__ is only defined during the device compilation pass, mirroring nvcc.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
#define __CUDA_ARCH__ __HIP_ARCH__
#endif

#ifndef __bfloat16
typedef __hip_bfloat16 __bfloat16;
#endif

// NOTE: we do NOT include <cuda/std/limits> here: the full libhipcxx header
// produces code objects that crash when loaded via hipModuleLoadData on this
// ROCm build. A minimal `cuda::std::numeric_limits` shim is provided instead
// (see hip_compat/cuda/std/limits).

// HIP natively supports float/double atomicMax/atomicMin; alias candle's names.
__device__ __forceinline__ float atomicMaxf(float* a, float v) { return atomicMax(a, v); }
__device__ __forceinline__ double atomicMaxf(double* a, double v) { return atomicMax(a, v); }
__device__ __forceinline__ float atomicMinf(float* a, float v) { return atomicMin(a, v); }
__device__ __forceinline__ double atomicMinf(double* a, double v) { return atomicMin(a, v); }

// fp8 intrinsic name mappings (HIP uses __hip_cvt_fp8_to_halfraw / __HIP_E4M3).
#define __NV_E4M3 __HIP_E4M3
#define __NV_E5M2 __HIP_E5M2
#define __nv_cvt_fp8_to_halfraw(x, interp) __half(__hip_cvt_fp8_to_halfraw((x), (interp)))

// HIP shuffle masks are 64-bit; candle passes the 32-bit 0xffffffff literal.
#define __shfl_xor_sync(mask, val, lanemask, width) \
    __shfl_xor_sync((uint64_t)(mask), (val), (lanemask), (width))
