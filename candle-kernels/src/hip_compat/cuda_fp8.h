// Shim so candle kernels including "cuda_fp8.h" resolve to HIP's fp8 header.
#pragma once
#include <hip/hip_fp8.h>
