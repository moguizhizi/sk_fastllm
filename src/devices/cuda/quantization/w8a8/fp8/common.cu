#include "common.cuh"

template <typename scalar_t, typename fp8_type>
__global__ void segmented_max_reduction_strided(
    float *__restrict__ scale, const scalar_t *__restrict__ input, int hidden_size, int64_t in_row_stride, int64_t num_tokens) {
    __shared__ float cache[256];
    const int tid = threadIdx.x;
    int64_t token_idx = blockIdx.x;

    // one block per token. Guard in case gridDim.x > num_tokens.
    if (token_idx >= num_tokens) {
        return;
    }

    const scalar_t *row_ptr = input + token_idx * in_row_stride;

    // each thread scans elements of the row in a strided fashion.
    float thread_max = 0.0f;
    for (int e = tid; e < hidden_size; e += blockDim.x) {
        float v = fabsf(static_cast<float>(row_ptr[e]));
        thread_max = fmaxf(thread_max, v);
    }

    cache[tid] = thread_max;
    __syncthreads();

    // parallel reduction to find row max.
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            cache[tid] = fmaxf(cache[tid], cache[tid + offset]);
        }
        __syncthreads();
    }

    // thread 0 updates global scale (per-tensor) atomically.
    if (tid == 0) {
        atomicMaxFloat(scale, cache[0] / quant_type_max_v<fp8_type>);
    }
}

template <typename scalar_t, typename fp8_type>
__global__ void scaled_fp8_quant_kernel_strided_dynamic(fp8_type *__restrict__ out, const scalar_t *__restrict__ input,
    const float *__restrict__ scale, int hidden_size, int64_t in_row_stride, int64_t out_row_stride) {
    const int64_t token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const scalar_t *token_in = input + token_idx * in_row_stride;
    fp8_type *token_out = out + token_idx * out_row_stride;

    const float reciprocal_scale = 1.0f / (*scale);
    vectorize_with_alignment<16>(token_in, token_out, hidden_size, tid, blockDim.x, [=] __device__(fp8_type & dst, const scalar_t &src) {
        dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src), reciprocal_scale);
    });
}

template <typename scalar_t, typename fp8_type>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel_strided(fp8_type *__restrict__ out, float *__restrict__ scale,
    const scalar_t *__restrict__ input, const float *__restrict__ scale_ub, int hidden_size, int64_t in_row_stride, int64_t out_row_stride) {
    const int64_t token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Use int64 to avoid overflowing an int32 when calculating this offset
    int64_t in_offset = static_cast<int64_t>(token_idx) * in_row_stride;
    int64_t out_offset = static_cast<int64_t>(token_idx) * out_row_stride;
    const scalar_t *token_in = input + in_offset;
    fp8_type *token_out = out + out_offset;

    // 1) per-token absmax
    float absmax_val = 0.f;
    vectorize_read_with_alignment<16>(
        token_in, hidden_size, tid, blockDim.x, [&] __device__(scalar_t v) { absmax_val = fmaxf(absmax_val, fabsf(static_cast<float>(v))); });

    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage tmp;
    const float block_max = BlockReduce(tmp).Reduce(absmax_val, CubMaxOp{}, blockDim.x);

    __shared__ float token_scale;
    if (tid == 0) {
        token_scale = scale_ub ? fminf(block_max, *scale_ub) : block_max;
        token_scale = fmaxf(token_scale / quant_type_max_v<fp8_type>, min_scaling_factor<fp8_type>::val());
        scale[token_idx] = token_scale;
    }
    __syncthreads();

    // 2) quantize
    vectorize_with_alignment<16>(token_in, token_out, hidden_size, tid, blockDim.x, [=] __device__(fp8_type & dst, const scalar_t &src) {
        dst = scaled_fp8_conversion<false, fp8_type>(static_cast<float>(src), token_scale);
    });
}

bool dynamic_scaled_fp8_quant(const fastllm::Data &input,
    fastllm::Data &output, fastllm::Data &scale) // [1]
{
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaScale = (float *)FastllmCudaPrepareInput(scale);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);

    const int hidden_size = input.Count(input.dims.size() - 1);
    const int input_len = input.Count(0);

    const int num_tokens = input_len / input.dims[input.dims.size() - 1];
    const int block_size = 256;

    dim3 grid(num_tokens);
    dim3 block(block_size);

    const int64_t in_row_stride = hidden_size;
    const int64_t out_row_stride = hidden_size;

    FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, {
        FASTLLM_DISPATCH_FP8_TYPES(output.dataType, {
            segmented_max_reduction_strided<scalar_t, fp8_t>
                <<<grid, block>>>(cudaScale, (scalar_t *)cudaInput, hidden_size, in_row_stride, static_cast<int64_t>(num_tokens));

            scaled_fp8_quant_kernel_strided_dynamic<scalar_t, fp8_t>
                <<<grid, block>>>((fp8_t *)cudaOutput, (scalar_t *)cudaInput, cudaScale, hidden_size, in_row_stride, out_row_stride);
        });
    });

    return true;
}
