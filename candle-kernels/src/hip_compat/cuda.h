// Shim so the host-compilation pass of kernels that include <cuda.h> compiles.
// The host pass of candle's kernels only needs the CUDA_VERSION gate to be false.
#pragma once
#define CUDA_VERSION 0
