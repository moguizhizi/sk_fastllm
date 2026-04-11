#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cmath>

#include "cub_helpers.h"
#include "dispatch_utils.h"
#include "libtorch_stable/quantization/vectorization_utils.cuh"

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
    static constexpr auto i8_min = static_cast<float>(std::numeric_limits<int8_t>::min());
    static constexpr auto i8_max = static_cast<float>(std::numeric_limits<int8_t>::max());

    // To match the rounding mode of CUDA, we use nearbyint.
    // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
    // If that changes in the future, we may need to set the rounding mode
    // explicitly, either at runtime or compile time.
    float dst = std::nearbyint(x);

    // saturate
    // See https://github.com/pytorch/pytorch/issues/127666
    // See https://github.com/llvm/llvm-project/issues/95183
    // hip-clang std::clamp __glibcxx_assert_fail host function when building on
    // Arch/gcc14. The following replaces std::clamp usage with similar logic
    // dst = std::clamp(dst, i8_min, i8_max);
    dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
    return static_cast<int8_t>(dst);
#else
    // CUDA path
    uint32_t dst;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
    return reinterpret_cast<const int8_t &>(dst);
#endif
}

static inline __device__ int32_t float_to_int32_rn(float x) {
#ifdef USE_ROCM
    // int32_max is not exactly representable as float.
    // Therefore, we need to be careful and manually return int32_max on overflow.
    // For symmetry, we also do the same for int32_min, even though it is exactly
    // representable as float and the conversion should be exact.
    static constexpr auto i32_min = std::numeric_limits<int32_t>::min();
    static constexpr auto i32_min_f = static_cast<float>(i32_min);
    static constexpr auto i32_max = std::numeric_limits<int32_t>::max();
    static constexpr auto i32_max_f = static_cast<float>(i32_max);

    // To match the rounding mode of CUDA, we use nearbyint.
    // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
    // If that changes in the future, we may need to set the rounding mode
    // explicitly, either at runtime or compile time.
    float dst = std::nearbyint(x);

    // saturate on the higher end.
    if (dst >= i32_max_f) {
        return i32_max;
    }
    // saturate on the lower end.
    if (dst <= i32_min_f) {
        return i32_min;
    }

    return static_cast<int32_t>(dst);
#else
    // CUDA path
    uint32_t dst;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(dst) : "f"(x));
    return reinterpret_cast<const int32_t &>(dst);
#endif
}

static inline __device__ int8_t int32_to_int8(int32_t x) {
#ifdef USE_ROCM
    static constexpr auto i8_min = static_cast<int32_t>(std::numeric_limits<int8_t>::min());
    static constexpr auto i8_max = static_cast<int32_t>(std::numeric_limits<int8_t>::max());

    // saturate
    // See https://github.com/pytorch/pytorch/issues/127666
    // See https://github.com/llvm/llvm-project/issues/95183
    // hip-clang std::clamp __glibcxx_assert_fail host function when building on
    // Arch/gcc14. The following replaces std::clamp usage with similar logic
    // int32_t dst = std::clamp(x, i8_min, i8_max);
    int32_t dst = (x < i8_min) ? i8_min : (x > i8_max) ? i8_max : x;
    return static_cast<int8_t>(dst);
#else
    // CUDA path
    uint32_t dst;
    asm volatile("cvt.sat.s8.s32 %0, %1;" : "=r"(dst) : "r"(x));
    return reinterpret_cast<const int8_t &>(dst);
#endif
}

template <typename scalar_t, typename scale_t>
__global__ void static_scaled_int8_quant_kernel(
    const scalar_t *__restrict__ input, int8_t *__restrict__ output, const scale_t *scale_ptr, const int hidden_size) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int64_t token_idx = blockIdx.x;
    const float scale = *scale_ptr;

    // Must be performed using 64-bit math to avoid integer overflow.
    const scalar_t *row_in = input + token_idx * hidden_size;
    int8_t *row_out = output + token_idx * hidden_size;

    vectorize_with_alignment<16>(row_in, row_out, hidden_size, tid, stride,
        [=] __device__(int8_t &dst, const scalar_t &src) { dst = float_to_int8_rn(static_cast<float>(src) / scale); });
}
