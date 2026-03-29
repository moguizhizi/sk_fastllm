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
    vllm::vectorize_read_with_alignment<VEC_SIZE>(
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