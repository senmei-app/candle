// Direct conv2d / conv_transpose2d kernels used by the HIP backend (e.g. CUGAN).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float h2f(float x) { return x; }
__device__ __forceinline__ float h2f(double x) { return (float)x; }
__device__ __forceinline__ float h2f(__half x) { return __half2float(x); }
__device__ __forceinline__ float h2f(__hip_bfloat16 x) { return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T f2h(float x);
template <> __device__ __forceinline__ float f2h<float>(float x) { return x; }
template <> __device__ __forceinline__ double f2h<double>(float x) { return (double)x; }
template <> __device__ __forceinline__ __half f2h<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __hip_bfloat16 f2h<__hip_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// input (n, c, h, w), weight (oc, c/group, kh, kw), bias (oc), output (n, oc, oh, ow).
template <typename T>
__device__ void conv2d_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    const int n, const int c, const int h, const int w,
    const int oc, const int kh, const int kw,
    const int oh, const int ow,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dil_h, const int dil_w,
    const int groups) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * oc * oh * ow;
    if (idx >= total) return;
    const int ow_ = idx % ow;
    const int oh_ = (idx / ow) % oh;
    const int oc_ = (idx / (ow * oh)) % oc;
    const int n_ = idx / (ow * oh * oc);

    const int c_per_group = c / groups;
    const int oc_per_group = oc / groups;
    const int group = oc_ / oc_per_group;
    const int oc_in_group = oc_ % oc_per_group;

    float acc = 0.f;
    for (int icg = 0; icg < c_per_group; ++icg) {
        const int ic = group * c_per_group + icg;
        for (int kh_ = 0; kh_ < kh; ++kh_) {
            const int ih = oh_ * stride_h - pad_h + kh_ * dil_h;
            if (ih < 0 || ih >= h) continue;
            for (int kw_ = 0; kw_ < kw; ++kw_) {
                const int iw = ow_ * stride_w - pad_w + kw_ * dil_w;
                if (iw < 0 || iw >= w) continue;
                const float iv = h2f(input[((n_ * c + ic) * h + ih) * w + iw]);
                const float wv = h2f(weight[((oc_ * c_per_group + icg) * kh + kh_) * kw + kw_]);
                acc += iv * wv;
            }
        }
    }
    if (bias != nullptr) acc += h2f(bias[oc_]);
    output[idx] = f2h<T>(acc);
}

// input (n, c, h, w), weight (c, oc/group, kh, kw), bias (oc), output (n, oc, oh, ow).
template <typename T>
__device__ void conv_transpose2d_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    const int n, const int c, const int h, const int w,
    const int oc, const int kh, const int kw,
    const int oh, const int ow,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dil_h, const int dil_w,
    const int groups) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * oc * oh * ow;
    if (idx >= total) return;
    const int ow_ = idx % ow;
    const int oh_ = (idx / ow) % oh;
    const int oc_ = (idx / (ow * oh)) % oc;
    const int n_ = idx / (ow * oh * oc);

    const int c_per_group = c / groups;
    const int oc_per_group = oc / groups;
    const int group = oc_ / oc_per_group;
    const int oc_in_group = oc_ % oc_per_group;

    float acc = 0.f;
    for (int icg = 0; icg < c_per_group; ++icg) {
        const int ic = group * c_per_group + icg;
        for (int kh_ = 0; kh_ < kh; ++kh_) {
            const int num_h = oh_ + pad_h - kh_ * dil_h;
            if (num_h % stride_h != 0) continue;
            const int ih = num_h / stride_h;
            if (ih < 0 || ih >= h) continue;
            for (int kw_ = 0; kw_ < kw; ++kw_) {
                const int num_w = ow_ + pad_w - kw_ * dil_w;
                if (num_w % stride_w != 0) continue;
                const int iw = num_w / stride_w;
                if (iw < 0 || iw >= w) continue;
                const float iv = h2f(input[((n_ * c + ic) * h + ih) * w + iw]);
                const float wv = h2f(weight[((ic * oc_per_group + oc_in_group) * kh + kh_) * kw + kw_]);
                acc += iv * wv;
            }
        }
    }
    if (bias != nullptr) acc += h2f(bias[oc_]);
    output[idx] = f2h<T>(acc);
}

#define INSTANTIATE_CONV(T, name)                                             \
    extern "C" __global__ void name##_conv2d(                                 \
        const T* input, const T* weight, const T* bias, T* output,            \
        const int n, const int c, const int h, const int w,                   \
        const int oc, const int kh, const int kw,                             \
        const int oh, const int ow, const int pad_h, const int pad_w,         \
        const int stride_h, const int stride_w, const int dil_h, const int dil_w, \
        const int groups) {                                                   \
        conv2d_kernel<T>(input, weight, bias, output, n, c, h, w, oc, kh, kw, \
                         oh, ow, pad_h, pad_w, stride_h, stride_w, dil_h, dil_w, groups); \
    }                                                                         \
    extern "C" __global__ void name##_conv_transpose2d(                       \
        const T* input, const T* weight, const T* bias, T* output,            \
        const int n, const int c, const int h, const int w,                   \
        const int oc, const int kh, const int kw,                             \
        const int oh, const int ow, const int pad_h, const int pad_w,         \
        const int stride_h, const int stride_w, const int dil_h, const int dil_w, \
        const int groups) {                                                   \
        conv_transpose2d_kernel<T>(input, weight, bias, output, n, c, h, w, oc, \
                                   kh, kw, oh, ow, pad_h, pad_w, stride_h, stride_w, \
                                   dil_h, dil_w, groups);                     \
    }

INSTANTIATE_CONV(float, hip_conv_f32)
INSTANTIATE_CONV(double, hip_conv_f64)
INSTANTIATE_CONV(__half, hip_conv_f16)
INSTANTIATE_CONV(__hip_bfloat16, hip_conv_bf16)
