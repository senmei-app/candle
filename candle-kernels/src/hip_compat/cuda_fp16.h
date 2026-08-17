// Shim so candle kernels including "cuda_fp16.h" resolve to HIP's fp16 header.
#pragma once
#include <hip/hip_fp16.h>
