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

template <typename scalar_t>
__global__ void static_scaled_int8_quant_kernel(
    const scalar_t *__restrict__ input, int8_t *__restrict__ output, const float scale, const int hidden_size) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int64_t token_idx = blockIdx.x;

    // Must be performed using 64-bit math to avoid integer overflow.
    const scalar_t *row_in = input + token_idx * hidden_size;
    int8_t *row_out = output + token_idx * hidden_size;

    vectorize_with_alignment<16>(row_in, row_out, hidden_size, tid, stride,
        [=] __device__(int8_t &dst, const scalar_t &src) { dst = float_to_int8_rn(static_cast<float>(src) / scale); });
}

template <typename scalar_t, typename azp_t>
__global__ void static_scaled_int8_azp_quant_kernel(
    const scalar_t *__restrict__ input, int8_t *__restrict__ output, const float scale, const azp_t azp, const int hidden_size) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int64_t token_idx = blockIdx.x;
    const float inv_s = 1.0f / scale;

    // Must be performed using 64-bit math to avoid integer overflow.
    const scalar_t *row_in = input + token_idx * hidden_size;
    int8_t *row_out = output + token_idx * hidden_size;

    vectorize_with_alignment<16>(row_in, row_out, hidden_size, tid, stride, [=] __device__(int8_t &dst, const scalar_t &src) {
        const auto v = static_cast<float>(src) * inv_s;
        dst = int32_to_int8(float_to_int32_rn(v) + azp);
    });
}

template <typename scalar_t, typename scale_t>
__global__ void dynamic_scaled_int8_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ output,
    scale_t* scale_out, const int hidden_size) {
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t token_idx = blockIdx.x;

  // Must be performed using 64-bit math to avoid integer overflow.
  const scalar_t* row_in = input + token_idx * hidden_size;
  int8_t* row_out = output + token_idx * hidden_size;

  // calculate for absmax
  float thread_max = 0.f;
  vectorize_read_with_alignment<16>(
      row_in, hidden_size, tid, stride, [&] __device__(const scalar_t& src) {
        const float v = fabsf(static_cast<float>(src));
        thread_max = fmaxf(thread_max, v);
      });
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);
  __shared__ float absmax;
  if (tid == 0) {
    absmax = block_max;
    scale_out[blockIdx.x] = absmax / 127.f;
  }
  __syncthreads();

  float inv_s = (absmax == 0.f) ? 0.f : 127.f / absmax;

  vectorize_with_alignment<16>(
      row_in, row_out, hidden_size, tid, stride,
      [=] __device__(int8_t& dst, const scalar_t& src) {
        dst = float_to_int8_rn(static_cast<float>(src) * inv_s);
      });
}

bool static_scaled_int8_quant(const fastllm::Data &input, fastllm::Data &output, const float scale, std::optional<fastllm::Data> const &azp) {
    TORCH_CHECK(!azp || azp.Count(0) == 1);

    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    if (azp.has_value()) {
        float *cudaazp = (float *)FastllmCudaPrepareInput(azp);
    }

    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);

    int input_num_dims = input.dims.size();
    int axis = -1;
    axis = (axis % input_num_dims + input_num_dims) % input_num_dims;

    int const hidden_size = input.dims[axis];
    int const num_tokens = input.Count(0) / hidden_size;

    dim3 const grid(num_tokens);
    dim3 const block(std::min(hidden_size, 256));

    FASTLLM_DISPATCH_FLOATING_TYPES(input.dataType, {
        if (!azp.has_value()) {
            static_scaled_int8_quant_kernel<scalar_t><<<grid, block>>>((scalar_t *)cudaInput, (int8_t *)cudaOutput, scale, hidden_size);
        } else {
            static_scaled_int8_azp_quant_kernel<scalar_t, int32_t>
                <<<grid, block>>>((scalar_t *)cudaInput, (int8_t *)cudaOutput, scale, (int32_t *)cudaazp, hidden_size);
        }
    });

    return true;
}