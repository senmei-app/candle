// Shim so candle kernels including "cuda_bf16.h" resolve to HIP's bf16 header.
#pragma once
#include <hip/hip_bf16.h>
