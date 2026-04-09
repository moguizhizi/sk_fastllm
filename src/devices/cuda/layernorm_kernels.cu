#include <torch/all.h>
#include <torch/extension.h>

#include <cub/cub.cuh>

#include "batch_invariant.hpp"
#include "cup_helpers.h"
#include "kernel_macros.cuh"
#include "type_convert.cuh"
#include "vectorization_utils.cuh"

#define FASTLLM_DISPATCH_RANK234(NUM_DIMS, ...)                                      \
    switch (NUM_DIMS) {                                                              \
        case 2: {                                                                    \
            constexpr int tensor_rank = 2;                                           \
            __VA_ARGS__();                                                           \
            break;                                                                   \
        }                                                                            \
        case 3: {                                                                    \
            constexpr int tensor_rank = 3;                                           \
            __VA_ARGS__();                                                           \
            break;                                                                   \
        }                                                                            \
        case 4: {                                                                    \
            constexpr int tensor_rank = 4;                                           \
            __VA_ARGS__();                                                           \
            break;                                                                   \
        }                                                                            \
        default:                                                                     \
            TORCH_CHECK(false, "Expects rank 2, 3 or 4 tensors but got ", NUM_DIMS); \
    }

#define FASTLLM_DISPATCH_VEC_SIZE(VEC_SIZE, ...) \
    switch (VEC_SIZE) {                          \
        case 16: {                               \
            constexpr int vec_size = 16;         \
            __VA_ARGS__();                       \
            break;                               \
        }                                        \
        case 8: {                                \
            constexpr int vec_size = 8;          \
            __VA_ARGS__();                       \
            break;                               \
        }                                        \
        case 4: {                                \
            constexpr int vec_size = 4;          \
            __VA_ARGS__();                       \
            break;                               \
        }                                        \
        case 2: {                                \
            constexpr int vec_size = 2;          \
            __VA_ARGS__();                       \
            break;                               \
        }                                        \
        default: {                               \
            constexpr int vec_size = 1;          \
            __VA_ARGS__();                       \
            break;                               \
        }                                        \
    }

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists> fused_add_rms_norm_kernel(
    scalar_t *__restrict__ input, // [..., hidden_size]
    const int64_t input_stride,
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
    // Sanity checks on our vector struct and type-punned pointer arithmetic
    static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
    static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

    const int vec_hidden_size = hidden_size / width;
    const int64_t vec_input_stride = input_stride / width;
    __shared__ float s_variance;
    float variance = 0.0f;
    /* These and the argument pointers are all declared `restrict` as they are
       not aliased in practice. Argument pointers should not be dereferenced
       in this kernel as that would be undefined behavior */
    auto *__restrict__ input_v = reinterpret_cast<_f16Vec<scalar_t, width> *>(input);
    auto *__restrict__ residual_v = reinterpret_cast<_f16Vec<scalar_t, width> *>(residual);
    auto *__restrict__ weight_v = reinterpret_cast<const _f16Vec<scalar_t, width> *>(weight);

    for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
        int id = blockIdx.x * vec_hidden_size + idx;
        int64_t strided_id = blockIdx.x * vec_input_stride + idx;
        _f16Vec<scalar_t, width> temp = input_v[strided_id];
        temp += residual_v[id];
        variance += temp.sum_squares();
        residual_v[id] = temp;
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
        int id = blockIdx.x * vec_hidden_size + idx;
        int64_t strided_id = blockIdx.x * vec_input_stride + idx;
        _f16Vec<scalar_t, width> temp = residual_v[id];
        temp *= s_variance;
        temp *= weight_v[idx];
        input_v[strided_id] = temp;
    }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists> fused_add_rms_norm_kernel(
    scalar_t *__restrict__ input, // [..., hidden_size]
    const int64_t input_stride,
    scalar_t *__restrict__ residual,     // [..., hidden_size]
    const scalar_t *__restrict__ weight, // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
    __shared__ float s_variance;
    float variance = 0.0f;

    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        scalar_t z = input[blockIdx.x * input_stride + idx];
        z += residual[blockIdx.x * hidden_size + idx];
        float x = (float)z;
        variance += x * x;
        residual[blockIdx.x * hidden_size + idx] = z;
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
        float x = (float)residual[blockIdx.x * hidden_size + idx];
        input[blockIdx.x * input_stride + idx] = ((scalar_t)(x * s_variance)) * weight[idx];
    }
}

template <typename scalar_t, int VEC_SIZE, int NUM_DIMS>
__global__ void rms_norm_kernel(scalar_t *__restrict__ out, // [..., hidden_size]
    const scalar_t *__restrict__ input,                     // [..., hidden_size]
    const int64_t input_stride_d2,                          // input.stride(-2)
    const int64_t input_stride_d3,                          // input.stride(-3)
    const int64_t input_stride_d4,                          // input.stride(-4)
    const int64_t input_shape_d2,                           // input.size(-2)
    const int64_t input_shape_d3,                           // input.size(-3)
    const scalar_t *__restrict__ weight,                    // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {
    __shared__ float s_variance;
    float variance = 0.0f;
    const scalar_t *input_row;
    if constexpr (NUM_DIMS == 2) {
        // 2D for layernorm normal case [batch_size, hidden]
        input_row = input + blockIdx.x * input_stride_d2;
    } else if constexpr (NUM_DIMS == 3) {
        // 3D for q/k norm [batch_size, num_heads, head_size]
        int batch_idx = blockIdx.x / input_shape_d2;
        int head_idx = blockIdx.x % input_shape_d2;
        input_row = input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
    } else if constexpr (NUM_DIMS == 4) {
        // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
        int batch_idx = blockIdx.x / (input_shape_d3 * input_shape_d2);
        int remaining = blockIdx.x % (input_shape_d3 * input_shape_d2);
        int seq_idx = remaining / input_shape_d2;
        int head_idx = remaining % input_shape_d2;
        input_row = input + batch_idx * input_stride_d4 + seq_idx * input_stride_d3 + head_idx * input_stride_d2;
    }

    auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE> &vec) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            float x = static_cast<float>(vec.val[i]);
            variance += x * x;
        }
    };
    auto scalar_op = [&variance](const scalar_t &val) {
        float x = static_cast<float>(val);
        variance += x * x;
    };
    fastllm::vectorize_read_with_alignment<VEC_SIZE>(input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();

    scalar_t *out_row = out + blockIdx.x * hidden_size;
    auto *v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE> *>(input_row);
    auto *v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE> *>(weight);
    auto *v_out = reinterpret_cast<vec_n_t<scalar_t, VEC_SIZE> *>(out_row);
    for (int i = threadIdx.x; i < hidden_size / VEC_SIZE; i += blockDim.x) {
        vec_n_t<scalar_t, VEC_SIZE> dst;
        vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[i];
        vec_n_t<scalar_t, VEC_SIZE> src2 = v_w[i];
#pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            float x = static_cast<float>(src1.val[j]);
            dst.val[j] = ((scalar_t)(x * s_variance)) * src2.val[j];
        }
        v_out[i] = dst;
    }
}

#define FASTLLM_LAUNCH_RMSNORM_KERNEL()                                                                                                        \
    FASTLLM_DISPATCH_RANK234(num_dims, [&] {                                                                                                   \
        FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, {                                                                                         \
            const int calculated_vec_size = std::gcd(16 / sizeof(scalar_t), hidden_size);                                                      \
            const int block_size = std::min(hidden_size / calculated_vec_size, max_block_size);                                                \
            dim3 block(block_size);                                                                                                            \
                                                                                                                                               \
            FASTLLM_DISPATCH_VEC_SIZE(calculated_vec_size, [&] {                                                                               \
                LAUNCH_KERNEL(rms_norm_kernel<scalar_t, vec_size, tensor_rank><<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput, \
                    input_stride_d2, input_stride_d3, input_stride_d4, input_shape_d2, input_shape_d3, (scalar_t *)cudaWeight, epsilon,        \
                    num_tokens, hidden_size));                                                                                                 \
            });                                                                                                                                \
        });                                                                                                                                    \
    })

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                                         \
  FASTLLM_DISPATCH_FLOAT_TYPES(                                             \
      input.dataType, LAUNCH_KERNEL(fused_add_rms_norm_kernel<scalar_t, width><<<grid, block>>>(                                   \
                (scalar_t *)cudaInput, input_stride,                   \
                (scalar_t *)cudaResidual, (scalar_t *)cudaWeight, \
                epsilon, num_tokens, hidden_size);                          \
      );

#define FASTLLM_RMSNORM_BODY()                                           \
    int input_len = input.Count(0);                                      \
    int output_len = output.Count(0);                                    \
                                                                         \
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);          \
    float *cudaWeight = (float *)FastllmCudaPrepareInput(weight);        \
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);       \
                                                                         \
    int hidden_size = input.Count(input.dims.size() - 1);                \
                                                                         \
    int num_tokens = input_len / input.dims[input.dims.size() - 1];      \
    int num_dims = input.dims.size();                                    \
                                                                         \
    axis = -2;                                                           \
    axis = (axis % num_dims + num_dims) % num_dims;                      \
    int64_t input_stride_d2 = input.strides[axis];                       \
                                                                         \
    axis = -3;                                                           \
    axis = (axis % num_dims + num_dims) % num_dims;                      \
    int64_t input_stride_d3 = (num_dims >= 3) ? input.strides[axis] : 0; \
                                                                         \
    axis = -4;                                                           \
    axis = (axis % num_dims + num_dims) % num_dims;                      \
    int64_t input_stride_d4 = (num_dims >= 4) ? input.strides[axis] : 0; \
                                                                         \
    axis = -2;                                                           \
    axis = (axis % num_dims + num_dims) % num_dims;                      \
    int64_t input_shape_d2 = (num_dims >= 3) ? input.dims[axis] : 0;     \
                                                                         \
    axis = -3;                                                           \
    axis = (axis % num_dims + num_dims) % num_dims;                      \
    int64_t input_shape_d3 = (num_dims >= 4) ? input.dims[axis] : 0;     \
                                                                         \
    const int max_block_size = (num_tokens < 256) ? 1024 : 256;          \
                                                                         \
    dim3 grid(num_tokens);                                               \
                                                                         \
    FASTLLM_LAUNCH_RMSNORM_KERNEL();                                     \
                                                                         \
    FastllmCudaFinishInput(input, cudaInput);                            \
    FastllmCudaFinishOutput(output, cudaOutput);                         \
    return true;

bool rms_norm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps) {
    int axis = 0;
    const float epsilon = eps;
    FASTLLM_RMSNORM_BODY();
}

bool fused_add_rms_norm( // [hidden_size]
    const fastllm::Data &input, const fastllm::Data &weight, fastllm::Data &output, double epsilon) {
    TORCH_CHECK(weight.dataType == input.dataType);
    TORCH_CHECK(input.dataType == residual.dataType);

    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaWeight = (float *)FastllmCudaPrepareInput(weight);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);

    int hidden_size = input.Count(input.dims.size() - 1);
    int64_t input_stride = hidden_size; // Assuming the last dimension is contiguous

    int input_len = input.Count(0);    
    int num_tokens = input_len / input.dims[input.dims.size() - 1];

    dim3 grid(num_tokens);
    /* This kernel is memory-latency bound in many scenarios.
       When num_tokens is large, a smaller block size allows
       for increased block occupancy on CUs and better latency
       hiding on global mem ops. */
    const int max_block_size = (num_tokens < 256) ? 1024 : 256;
    dim3 block(std::min(hidden_size, max_block_size));
    /*If the tensor types are FP16/BF16, try to use the optimized kernel
      with packed + vectorized ops.
      Max optimization is achieved with a width-8 vector of FP16/BF16s
      since we can load at most 128 bits at once in a global memory op.
      However, this requires each tensor's data to be aligned to 16
      bytes.
     */
    auto inp_ptr = reinterpret_cast<std::uintptr_t>(cudaInput);
    auto res_ptr = reinterpret_cast<std::uintptr_t>(cudaResidual);
    auto wt_ptr = reinterpret_cast<std::uintptr_t>(cudaWeight);
    constexpr int vector_width = 8;
    constexpr int req_alignment_bytes = vector_width * 2; // vector_width * sizeof(bfloat16 or float16) (float32
                                                          // falls back to non-vectorized version anyway)
    bool ptrs_are_aligned = inp_ptr % req_alignment_bytes == 0 && res_ptr % req_alignment_bytes == 0 && wt_ptr % req_alignment_bytes == 0;
    bool offsets_are_multiple_of_vector_width = hidden_size % vector_width == 0 && input_stride % vector_width == 0;
    bool batch_invariant_launch = FastLLM::FastLLM_is_batch_invariant();
    if (ptrs_are_aligned && offsets_are_multiple_of_vector_width && !batch_invariant_launch) {
        LAUNCH_FUSED_ADD_RMS_NORM(8);
    } else {
        LAUNCH_FUSED_ADD_RMS_NORM(0);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);

    return true;
}