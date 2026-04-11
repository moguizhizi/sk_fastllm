#include <optional>
#include <tuple>

#include "common.cuh"
#include "kernel_macros.cuh"

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
    const scalar_t *__restrict__ input, const float &scale_ub, const bool has_scale_ub, int hidden_size, int64_t in_row_stride,
    int64_t out_row_stride) {
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
        token_scale = has_scale_ub ? fminf(block_max, scale_ub) : block_max;
        token_scale = fmaxf(token_scale / quant_type_max_v<fp8_type>, min_scaling_factor<fp8_type>::val());
        scale[token_idx] = token_scale;
    }
    __syncthreads();

    // 2) quantize
    vectorize_with_alignment<16>(token_in, token_out, hidden_size, tid, blockDim.x, [=] __device__(fp8_type & dst, const scalar_t &src) {
        dst = scaled_fp8_conversion<false, fp8_type>(static_cast<float>(src), token_scale);
    });
}

// STRIDE_I_ZERO: true if scale_stride_i == 0 (per-tensor or per-channel)
// STRIDE_J_ZERO: true if scale_stride_j == 0 (per-tensor or per-token)
template <typename scalar_t, typename fp8_type, bool STRIDE_I_ZERO, bool STRIDE_J_ZERO>
__global__ void scaled_fp8_quant_kernel_strided_group_shape(fp8_type *__restrict__ out, const scalar_t *__restrict__ input,
    const float *__restrict__ scale, int hidden_size, int64_t in_row_stride, int64_t out_row_stride, int group_m, int group_n,
    int64_t scale_stride_i, int64_t scale_stride_j) {
    const int64_t token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const scalar_t *token_in = input + token_idx * in_row_stride;
    fp8_type *token_out = out + token_idx * out_row_stride;

    // Precompute row-level base offset for scale access (compile-time eliminated
    // when STRIDE_I_ZERO)
    const int64_t scale_row_base = STRIDE_I_ZERO ? 0 : static_cast<int>(token_idx) / group_m * scale_stride_i;

    auto get_inv_scale = [&](int gj) { return 1.0f / scale[scale_row_base + gj * scale_stride_j]; };

    int cached_gj = -1;
    float cached_inv_scale = 0.0f;
    auto get_inv_scale_cached = [&](int gj) {
        if (gj != cached_gj) {
            cached_inv_scale = 1.0f / scale[scale_row_base + gj * scale_stride_j];
            cached_gj = gj;
        }
        return cached_inv_scale;
    };

    constexpr int VEC_SIZE = 16; // FP8 so vectorize to 128 bits
    auto scaled_fp8_conversion_vectorized = [&](const scalar_t *in, fp8_type *out, int size, float inv_scale) {
        vectorize_with_alignment<VEC_SIZE>(in, out, size, tid, blockDim.x, [=] __device__(fp8_type & dst, const scalar_t &src) {
            dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src), inv_scale);
        });
    };

    if (STRIDE_J_ZERO && hidden_size % VEC_SIZE == 0) {
        // Per-tensor or per-token: single scale per row, vectorize full row
        scaled_fp8_conversion_vectorized(token_in, token_out, hidden_size, get_inv_scale(0));
    } else if (group_n % VEC_SIZE == 0) {
        // Multiple column groups with vectorization
        const int num_groups_n = hidden_size / group_n;

        for (int gj = 0; gj < num_groups_n; gj++) {
            scaled_fp8_conversion_vectorized(token_in + gj * group_n, token_out + gj * group_n, group_n, get_inv_scale(gj));
        }
    } else {
        // Scalar path for small column groups (group_n < VEC_SIZE)
        for (int n = tid; n < hidden_size; n += blockDim.x) {
            const int gj = n / group_n;
            token_out[n] = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(token_in[n]), get_inv_scale_cached(gj));
        }
    }
}

bool dynamic_scaled_fp8_quant(const fastllm::Data &input, fastllm::Data &output, fastllm::Data &scale) // [1]
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

bool dynamic_per_token_scaled_fp8_quant(
    const fastllm::Data &input, fastllm::Data &output, fastllm::Data &scale, std::optional<float> const &scale_ub) {
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaScale = (float *)FastllmCudaPrepareInput(scale);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);

    const int hidden_size = input.Count(input.dims.size() - 1);
    const int input_len = input.Count(0);

    const int num_tokens = input_len / input.dims[input.dims.size() - 1];
    const int block_size = 256;

    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, block_size));

    const int64_t in_row_stride = hidden_size;
    const int64_t out_row_stride = hidden_size;

    float scale_ub_val = scale_ub.value_or(0.0f);
    bool has_scale_ub = scale_ub.has_value();

    FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, {
        FASTLLM_DISPATCH_FP8_TYPES(output.dataType, {
            dynamic_per_token_scaled_fp8_quant_kernel_strided<scalar_t, fp8_t><<<grid, block>>>((fp8_t *)cudaOutput, cudaScale,
                (scalar_t *)cudaInput, scale_ub.has_value() ? scale_ub_val, has_scale_ub, hidden_size, in_row_stride, out_row_stride);
        });
    });

    return true;
}

bool static_scaled_fp8_quant(const fastllm::Data &input, fastllm::Data &output, fastllm::Data &scale, // various shapes
    std::optional<std::tuple<int64_t, int64_t>> opt_group_shape_1)                                    // optional explicit (group_m, group_n)
{
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaScale = (float *)FastllmCudaPrepareInput(scale);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);

    int input_num_dims = input.strides.size();
    int axis = -1;
    axis = (axis % input_num_dims + input_num_dims) % input_num_dims;

    int output_num_dims = output.strides.size();
    axis = -1;
    axis = (axis % output_num_dims + output_num_dims) % output_num_dims;

    TORCH_CHECK(input.strides[axis] == 1, "last dimension of input must be contiguous");
    TORCH_CHECK(output.strides[axis] == 1, "last dimension of output must be contiguous");

    input_num_dims = input.dims.size();
    int axis = -1;
    axis = (axis % input_num_dims + input_num_dims) % input_num_dims;

    const int hidden_size = input.dims[axis];            // N (columns)
    const int num_tokens = input.Count(0) / hidden_size; // M (rows)

    // Determine group_m, group_n, and scale strides from scale shape
    // Scale indexing: scale[gi * scale_stride_j + gj * scale_stride_i]
    // where gi = m / group_m, gj = n / group_n
    int group_m, group_n;
    int64_t scale_stride_i, scale_stride_j;

    if (scale.Count(0) == 1) {
        // Per-tensor: one scale for the entire tensor
        group_m = num_tokens;
        group_n = hidden_size;
        scale_stride_i = 0;
        scale_stride_j = 0;
    } else if (scale.dims.size() == 1) {
        // 1D scale: require explicit group_shape to disambiguate per-channel vs
        // per-token (avoids edge case where num_tokens == hidden_size)
        TORCH_CHECK(opt_group_shape.has_value(),
            "1D scale requires explicit group_shape to disambiguate "
            "per-channel vs per-token quantization. "
            "Use group_shape=(-1, 1) for per-channel or group_shape=(1, "
            "-1) for per-token.");

        const auto &[opt_group_m, opt_group_n] = opt_group_shape.value();
        group_m = opt_group_m == -1 ? num_tokens : static_cast<int>(opt_group_m);
        group_n = opt_group_n == -1 ? hidden_size : static_cast<int>(opt_group_n);

        // Validate the explicit group shape matches the 1D scale
        const int64_t scale_len = scale.Count(0);
        const int64_t expected_scale_m = num_tokens / group_m;
        const int64_t expected_scale_n = hidden_size / group_n;
        const int64_t expected_scale_numel = expected_scale_m * expected_scale_n;

        TORCH_CHECK(scale_len == expected_scale_numel, "1D scale length (", scale_len, ") does not match expected size (", expected_scale_numel,
            ") for group_shape (", opt_group_m, ", ", opt_group_n, ") with input shape (", num_tokens, ", ", hidden_size, ")");

        // For 1D scale, determine strides based on which dim is trivial
        // Scale indexing: scale[gi * scale_stride_i + gj * scale_stride_j]
        // where gi = m / group_m (row group), gj = n / group_n (col group)
        if (expected_scale_m == 1) {
            // Per-channel style: one scale in M dim, scale varies along N
            // gi = 0 always, gj varies, so stride_1 traverses the scale
            scale_stride_i = 0;
            scale_stride_j = 1;
        } else if (expected_scale_n == 1) {
            // Per-token style: one scale in N dim, scale varies along M
            // gj = 0 always, gi varies, so stride_0 traverses the scale
            scale_stride_i = 1;
            scale_stride_j = 0;
        } else {
            TORCH_CHECK(false,
                "1D scale can only be used when one of the scale dimensions is 1. "
                "For 2D group scaling, use a 2D scale tensor.");
        }
    } else if (scale.dims.size() == 2) {
        // 2D scale: infer group sizes from scale dimensions (or use explicit if
        // provided)
        const int64_t scale_size_0 = scale.dims[0];
        const int64_t scale_size_1 = scale.dims[1];

        TORCH_CHECK(num_tokens % scale_size_0 == 0, "num_tokens (", num_tokens, ") must be divisible by scale.size(0) (", scale_size_0, ")");
        TORCH_CHECK(hidden_size % scale_size_1 == 0, "hidden_size (", hidden_size, ") must be divisible by scale.size(1) (", scale_size_1, ")");

        // Infer from 2D scale shape
        int inferred_group_m = num_tokens / scale_size_0;
        int inferred_group_n = hidden_size / scale_size_1;

        // Use explicit if provided, otherwise use inferred
        if (opt_group_shape.has_value()) {
            const auto &[opt_group_m, opt_group_n] = opt_group_shape.value();
            group_m = opt_group_m == -1 ? num_tokens : static_cast<int>(opt_group_m);
            group_n = opt_group_n == -1 ? hidden_size : static_cast<int>(opt_group_n);

            // Validate explicit matches inferred
            TORCH_CHECK(group_m == inferred_group_m && group_n == inferred_group_n, "Explicit group_shape (", opt_group_m, ", ", opt_group_n,
                ") does not match inferred group shape (", inferred_group_m, ", ", inferred_group_n, ") from 2D scale tensor shape (",
                scale_size_0, ", ", scale_size_1, ")");
        } else {
            group_m = inferred_group_m;
            group_n = inferred_group_n;
        }

        scale_stride_i = scale.strides[0];
        scale_stride_j = scale.strides[1];
    } else {
        TORCH_CHECK(false, "scale must be 0D, 1D, or 2D tensor, but got ", scale.dim(), "D");
    }

    const int block_size = 256;
    dim3 grid(num_tokens);
    dim3 block(block_size);

    const int64_t in_row_stride = input.strides[0];
    const int64_t out_row_stride = out.strides[0];

    // Dispatch to template-specialized kernel based on stride pattern
    FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, {
        FASTLLM_DISPATCH_FP8_TYPES(out.dataType, {
            FASTLLM_DISPATCH_BOOL(scale_stride_i == 0, S0_ZERO, {
                FASTLLM_DISPATCH_BOOL(scale_stride_j == 0, S1_ZERO, {
                    scaled_fp8_quant_kernel_strided_group_shape<scalar_t, fp8_t, S0_ZERO, S1_ZERO><<<grid, block>>>((fp8_t *)cudaOutput,
                        (scalar_t *)cudaInput, cudaScale, hidden_size, in_row_stride, out_row_stride, group_m, group_n, scale_stride_i,
                        scale_stride_j);
                });
            });
        });
    });

    return true;
}