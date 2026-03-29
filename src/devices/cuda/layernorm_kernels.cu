#include "type_convert.cuh"
#include <cub/cub.cuh>
#include "vectorization_utils.cuh"
#include "cup_helpers.h"
#include "kernel_macros.cuh"
#include <torch/all.h>
#include <torch/extension.h>

#define FASTLLM_DISPATCH_RANK234(NUM_DIMS, ...)                                \
  switch (NUM_DIMS) {                                                          \
    case 2: {                                                                  \
      constexpr int tensor_rank = 2;                                           \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    case 3: {                                                                  \
      constexpr int tensor_rank = 3;                                           \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    case 4: {                                                                  \
      constexpr int tensor_rank = 4;                                           \
      __VA_ARGS__();                                                           \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      TORCH_CHECK(false, "Expects rank 2, 3 or 4 tensors but got ", NUM_DIMS); \
  }

#define FASTLLM_DISPATCH_VEC_SIZE(VEC_SIZE, ...)    \
    switch (VEC_SIZE) {                             \
        case 16: {                                  \
            constexpr int vec_size = 16;            \
            __VA_ARGS__();                          \
            break;                                  \
        }                                           \
        case 8: {                                   \
            constexpr int vec_size = 8;             \
            __VA_ARGS__();                          \
            break;                                  \
        }                                           \
        case 4: {                                   \
            constexpr int vec_size = 4;             \
            __VA_ARGS__();                          \
            break;                                  \
        }                                           \
        case 2: {                                   \
            constexpr int vec_size = 2;             \
            __VA_ARGS__();                          \
            break;                                  \
        }                                           \
        default: {                                  \
            constexpr int vec_size = 1;             \
            __VA_ARGS__();                          \
            break;                                  \
        }                                           \
    }


template <typename scalar_t, int VEC_SIZE, int NUM_DIMS>
__global__ void rms_norm_kernel(
    scalar_t *__restrict__ out,          // [..., hidden_size]
    const scalar_t *__restrict__ input,  // [..., hidden_size]
    const int64_t input_stride_d2,       // input.stride(-2)
    const int64_t input_stride_d3,       // input.stride(-3)
    const int64_t input_stride_d4,       // input.stride(-4)
    const int64_t input_shape_d2,        // input.size(-2)
    const int64_t input_shape_d3,        // input.size(-3)
    const scalar_t *__restrict__ weight, // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size)
{
    __shared__ float s_variance;
    float variance = 0.0f;
    const scalar_t *input_row;
    if constexpr (NUM_DIMS == 2)
    {
        // 2D for layernorm normal case [batch_size, hidden]
        input_row = input + blockIdx.x * input_stride_d2;
    }
    else if constexpr (NUM_DIMS == 3)
    {
        // 3D for q/k norm [batch_size, num_heads, head_size]
        int batch_idx = blockIdx.x / input_shape_d2;
        int head_idx = blockIdx.x % input_shape_d2;
        input_row =
            input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
    }
    else if constexpr (NUM_DIMS == 4)
    {
        // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
        int batch_idx = blockIdx.x / (input_shape_d3 * input_shape_d2);
        int remaining = blockIdx.x % (input_shape_d3 * input_shape_d2);
        int seq_idx = remaining / input_shape_d2;
        int head_idx = remaining % input_shape_d2;
        input_row = input + batch_idx * input_stride_d4 +
                    seq_idx * input_stride_d3 + head_idx * input_stride_d2;
    }

    auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE> &vec)
    {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i)
        {
            float x = static_cast<float>(vec.val[i]);
            variance += x * x;
        }
    };
    auto scalar_op = [&variance](const scalar_t &val)
    {
        float x = static_cast<float>(val);
        variance += x * x;
    };
    fastllm::vectorize_read_with_alignment<VEC_SIZE>(
        input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

    if (threadIdx.x == 0)
    {
        s_variance = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();

    scalar_t *out_row = out + blockIdx.x * hidden_size;
    auto *v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE> *>(input_row);
    auto *v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE> *>(weight);
    auto *v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE> *>(out_row);
    for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x)
    {
        vec_n_t<scalar_t, VEC_SIZE> dst;
        vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];
        vec_n_t<scalar_t, VEC_SIZE> src2 = v_w[i];
#pragma unroll
        for (int j = 0; j < VEC_SIZE; j++)
        {
            float x = static_cast<float>(src1.val[j]);
            dst.val[j] = ((scalar_t)(x * s_variance)) * src2.val[j];
        }
        v_out[i] = dst;
    }
}

#define FASTLLM_LAUNCH_RMSNORM_KERNEL()                                                         \
    FASTLLM_DISPATCH_RANK234(num_dims, [&] {                                                    \
        FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, {                                          \
            const int calculated_vec_size = std::gcd(16 / sizeof(scalar_t), hidden_size);       \
            const int block_size = std::min(hidden_size / calculated_vec_size, max_block_size); \
            dim3 block(block_size);                                                             \
                                                                                                \
            FASTLLM_DISPATCH_VEC_SIZE(calculated_vec_size, [&] {                                \
                LAUNCH_KERNEL(rms_norm_kernel<scalar_t, vec_size, tensor_rank>                  \
                    <<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput,            \
                        input_stride_d2, input_stride_d3, input_stride_d4,                      \
                        input_shape_d2, input_shape_d3, (scalar_t *)cudaWeight,                 \
                        epsilon, num_tokens, hidden_size));                                     \
            });                                                                                 \
        });                                                                                     \
    })

#define FASTLLM_RMSNORM_BODY()                                                           \
    int input_len = input.Count(0);                                                      \
    int output_len = output.Count(0);                                                    \
                                                                                         \
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);                          \
    float *cudaWeight = (float *)FastllmCudaPrepareInput(weight);                        \
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);                       \
                                                                                         \
    int hidden_size = input.Count(input.dims.size() - 1);                                \
                                                                                         \
    int num_tokens = input_len / input.dims[input.dims.size() - 1];                      \
    int num_dims = input.dims.size();                                                    \
                                                                                         \
    axis = -2;                                                                           \
    axis = (axis % num_dims + num_dims) % num_dims;                                      \
    int64_t input_stride_d2 = input.strides[axis];                                       \
                                                                                         \
    axis = -3;                                                                           \
    axis = (axis % num_dims + num_dims) % num_dims;                                      \
    int64_t input_stride_d3 = (num_dims >= 3) ? input.strides[axis] : 0;                 \
                                                                                         \
    axis = -4;                                                                           \
    axis = (axis % num_dims + num_dims) % num_dims;                                      \
    int64_t input_stride_d4 = (num_dims >= 4) ? input.strides[axis] : 0;                 \
                                                                                         \
                                                                                         \
    axis = -2;                                                                           \
    axis = (axis % num_dims + num_dims) % num_dims;                                      \
    int64_t input_shape_d2 = (num_dims >= 3) ? input.dims[axis] : 0;                     \
                                                                                         \
    axis = -3;                                                                           \
    axis = (axis % num_dims + num_dims) % num_dims;                                      \
    int64_t input_shape_d3 = (num_dims >= 4) ? input.dims[axis] : 0;                     \
                                                                                         \
    const int max_block_size = (num_tokens < 256) ? 1024 : 256;                          \
                                                                                         \
    dim3 grid(num_tokens);                                                               \
                                                                                         \
    FASTLLM_LAUNCH_RMSNORM_KERNEL();                                             \
                                                                                         \
    FastllmCudaFinishInput(input, cudaInput);                                            \
    FastllmCudaFinishOutput(output, cudaOutput);                                         \
    return true;

bool rms_norm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps) {
    int axis = 0;
    const float epsilon = eps;
    FASTLLM_RMSNORM_BODY();
}