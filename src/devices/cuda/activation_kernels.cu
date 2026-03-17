
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <fastllm-cuda.cuh>

#include "activation_kernels.cuh"

#ifndef USE_ROCM
#    define FASTLLM_LDG(arg) __ldg(arg)
#else
#    define VLLM_LDG(arg) *(arg)
#endif

#define LAUNCH_KERNEL(...) __VA_ARGS__

#define FASTLLM_DISPATCH_FLOAT_TYPES(TYPE, ACT_FN, PACKED_ACT_FN, BODY)                        \
    switch (TYPE) {                                                                            \
        case fastllm::DataType::FLOAT32: {                                                     \
            using scalar_t = float;                                                            \
            using packed_t = float2;                                                           \
            constexpr scalar_t (*act_fn)(const scalar_t &) = ACT_FN<float>;                    \
            constexpr packed_t (*packed_fn)(const packed_t &) = PACKED_ACT_FN<float2>;         \
            BODY;                                                                              \
            break;                                                                             \
        }                                                                                      \
        case fastllm::DataType::FLOAT16: {                                                     \
            using scalar_t = half;                                                             \
            using packed_t = half2;                                                            \
            constexpr scalar_t (*act_fn)(const scalar_t &) = ACT_FN<half>;                     \
            constexpr packed_t (*packed_fn)(const packed_t &) = PACKED_ACT_FN<half2>;          \
            BODY;                                                                              \
            break;                                                                             \
        }                                                                                      \
        case fastllm::DataType::BFLOAT16: {                                                    \
            using scalar_t = __nv_bfloat16;                                                    \
            using packed_t = __nv_bfloat162;                                                   \
            constexpr scalar_t (*act_fn)(const scalar_t &) = ACT_FN<__nv_bfloat16>;            \
            constexpr packed_t (*packed_fn)(const packed_t &) = PACKED_ACT_FN<__nv_bfloat162>; \
            BODY;                                                                              \
            break;                                                                             \
        }                                                                                      \
    }

struct VecConfig {
    bool use_vec;
    int vec_size;
    int block_size;
};

struct alignas(32) u32x8_t {
    uint32_t u0, u1, u2, u3, u4, u5, u6, u7;
};

template <typename T>
struct PackedTraits;

template <>
struct PackedTraits<__nv_bfloat16> {
    using packed_t = __nv_bfloat162;
};

template <>
struct PackedTraits<half> {
    using packed_t = __half2;
};

template <>
struct PackedTraits<float> {
    using packed_t = float2;
};

template <bool support_256>
struct VecTraits;

template <>
struct VecTraits<true> {
    static constexpr int ARCH_MAX_VEC_SIZE = 32;
    using vec_t = u32x8_t;
};

template <>
struct VecTraits<false> {
    static constexpr int ARCH_MAX_VEC_SIZE = 16;
    using vec_t = int4;
};

__device__ __forceinline__ void ld256(u32x8_t &val, const u32x8_t *ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(CUDA_VERSION) && CUDA_VERSION >= 12090
    asm volatile("ld.global.nc.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
        : "=r"(val.u0), "=r"(val.u1), "=r"(val.u2), "=r"(val.u3), "=r"(val.u4), "=r"(val.u5), "=r"(val.u6), "=r"(val.u7)
        : "l"(ptr));
#else
    const uint4 *uint_ptr = reinterpret_cast<const uint4 *>(ptr);
    uint4 top_half = __ldg(&uint_ptr[0]);
    uint4 bottom_half = __ldg(&uint_ptr[1]);
    val.u0 = top_half.x;
    val.u1 = top_half.y;
    val.u2 = top_half.z;
    val.u3 = top_half.w;
    val.u4 = bottom_half.x;
    val.u5 = bottom_half.y;
    val.u6 = bottom_half.z;
    val.u7 = bottom_half.w;
#endif
}

__device__ __forceinline__ void st256(u32x8_t &val, u32x8_t *ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(CUDA_VERSION) && CUDA_VERSION >= 12090
    asm volatile("st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n"
        :
        : "l"(ptr), "r"(val.u0), "r"(val.u1), "r"(val.u2), "r"(val.u3), "r"(val.u4), "r"(val.u5), "r"(val.u6), "r"(val.u7)
        : "memory");
#else
    uint4 *uint_ptr = reinterpret_cast<uint4 *>(ptr);
    uint_ptr[0] = make_uint4(val.u0, val.u1, val.u2, val.u3);
    uint_ptr[1] = make_uint4(val.u4, val.u5, val.u6, val.u7);
#endif
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_mul(const packed_t &x, const packed_t &y) {
    if constexpr (std::is_same_v<packed_t, __nv_bfloat162> || std::is_same_v<packed_t, __half2>) {
        return __hmul2(x, y);
    } else if constexpr (std::is_same_v<packed_t, float2>) {
        return make_float2(x.x * y.x, x.y * y.y);
    }
}

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t &), bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t &x, const scalar_t &y) {
    return act_first ? ACT_FN(x) * y : x * ACT_FN(y);
}

template <typename packed_t, packed_t (*PACKED_ACT_FN)(const packed_t &), bool act_first>
__device__ __forceinline__ packed_t packed_compute(const packed_t &x, const packed_t &y) {
    return act_first ? packed_mul(PACKED_ACT_FN(x), y) : packed_mul(x, PACKED_ACT_FN(y));
}

template <typename packed_t>
__device__ __forceinline__ packed_t cast_to_packed(const float2 &val) {
    if constexpr (std::is_same_v<packed_t, __nv_bfloat162>) {
        return __float22bfloat162_rn(val);
    } else if constexpr (std::is_same_v<packed_t, __half2>) {
        return __float22half2_rn(val);
    } else if constexpr (std::is_same_v<packed_t, float2>) {
        return float2(val);
    }
}

template <typename packed_t>
__device__ __forceinline__ float2 cast_to_float2(const packed_t &val) {
    if constexpr (std::is_same_v<packed_t, __nv_bfloat162>) {
        return __bfloat1622float2(val);
    } else if constexpr (std::is_same_v<packed_t, __half2>) {
        return __half22float2(val);
    } else if constexpr (std::is_same_v<packed_t, float2>) {
        return float2(val);
    }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T &x) {
    // x * sigmoid(x)
    return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_silu_kernel(const packed_t &val) {
    // x * sigmoid(x)
    float2 fval = cast_to_float2(val);
    fval.x = fval.x / (1.0f + expf(-fval.x));
    fval.y = fval.y / (1.0f + expf(-fval.y));
    return cast_to_packed<packed_t>(fval);
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T &x) {
    // Equivalent to PyTorch GELU with 'none' approximation.
    // Refer to:
    // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
    const float f = (float)x;
    constexpr float ALPHA = M_SQRT1_2;
    return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_gelu_kernel(const packed_t &val) {
    // Equivalent to PyTorch GELU with 'none' approximation.
    // Refer to:
    // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
    constexpr float ALPHA = M_SQRT1_2;
    float2 fval = cast_to_float2(val);
    fval.x = fval.x * 0.5f * (1.0f + ::erf(fval.x * ALPHA));
    fval.y = fval.y * 0.5f * (1.0f + ::erf(fval.y * ALPHA));
    return cast_to_packed<packed_t>(fval);
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T &x) {
    // Equivalent to PyTorch GELU with 'tanh' approximation.
    // Refer to:
    // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
    const float f = (float)x;
    constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float KAPPA = 0.044715;
    float x_cube = f * f * f;
    float inner = BETA * (f + KAPPA * x_cube);
    return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_gelu_tanh_kernel(const packed_t &val) {
    // Equivalent to PyTorch GELU with 'tanh' approximation.
    // Refer to:
    // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
    float2 fval = cast_to_float2(val);
    constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float KAPPA = 0.044715;

    float x_cube = fval.x * fval.x * fval.x;
    float inner = BETA * (fval.x + KAPPA * x_cube);
    fval.x = 0.5f * fval.x * (1.0f + ::tanhf(inner));

    x_cube = fval.y * fval.y * fval.y;
    inner = BETA * (fval.y + KAPPA * x_cube);
    fval.y = 0.5f * fval.y * (1.0f + ::tanhf(inner));
    return cast_to_packed<packed_t>(fval);
}

// Activation and gating kernel template.
template <typename scalar_t, typename packed_t, scalar_t (*ACT_FN)(const scalar_t &), packed_t (*PACKED_ACT_FN)(const packed_t &),
    bool act_first, bool use_vec, bool use_256b>
__global__ void act_and_mul_kernel(scalar_t *__restrict__ out, // [..., d]
    const scalar_t *__restrict__ input,                        // [..., 2, d]
    const int d) {
    const scalar_t *x_ptr = input + blockIdx.x * 2 * d;
    const scalar_t *y_ptr = x_ptr + d;
    scalar_t *out_ptr = out + blockIdx.x * d;

    if constexpr (use_vec) {
        // Fast path: 128-bit/256-bit vectorized loop
        using vec_t = typename VecTraits<use_256b>::vec_t;
        constexpr int ARCH_MAX_VEC_SIZE = VecTraits<use_256b>::ARCH_MAX_VEC_SIZE;
        constexpr int VEC_SIZE = ARCH_MAX_VEC_SIZE / sizeof(packed_t);

        const vec_t *x_vec = reinterpret_cast<const vec_t *>(x_ptr);
        const vec_t *y_vec = reinterpret_cast<const vec_t *>(y_ptr);
        vec_t *out_vec = reinterpret_cast<vec_t *>(out_ptr);
        const int num_vecs = d / 2 / VEC_SIZE;

        for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
            vec_t x, y;
            if constexpr (use_256b) {
                ld256(x, &x_vec[i]);
                ld256(y, &y_vec[i]);
            } else {
                x = FASTLLM_LDG(&x_vec[i]);
                y = FASTLLM_LDG(&y_vec[i]);
            }
            auto *xp = reinterpret_cast<packed_t *>(&x);
            auto *yp = reinterpret_cast<packed_t *>(&y);
#pragma unroll
            for (int j = 0; j < VEC_SIZE; j++) {
                xp[j] = packed_compute<packed_t, PACKED_ACT_FN, act_first>(xp[j], yp[j]);
            }
            if constexpr (use_256b) {
                st256(x, &out_vec[i]);
            } else {
                out_vec[i] = x;
            }
        }
    } else {
        // Scalar fallback for unaligned data or small d
        for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
            const scalar_t x = FASTLLM_LDG(&x_ptr[idx]);
            const scalar_t y = FASTLLM_LDG(&y_ptr[idx]);
            out_ptr[idx] = compute<scalar_t, ACT_FN, act_first>(x, y);
        }
    }
}

VecConfig get_vec_config(uint32_t num_tokens, uint32_t elementSize, uint32_t num_elements) {
    VecConfig cfg{};

    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        printf("cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return cfg;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return cfg;
    }

    int cc_major = prop.major;

    int support_vec = (cc_major >= 10 && num_tokens > 128) ? 32 : 16;

    cfg.vec_size = support_vec / elementSize;

    cfg.use_vec = (num_elements % cfg.vec_size == 0);

    if (cfg.use_vec) {
        cfg.block_size = std::min((int)(num_elements / cfg.vec_size), 1024);
    } else {
        cfg.vec_size = 1;
        cfg.block_size = std::min((int)num_elements, 1024);
    }

    return cfg;
}

bool use_256b(uint32_t num_tokens) {
    int device = -1;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        printf("cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    cudaDeviceProp prop;
    auto error = cudaGetDeviceProperties(&prop, device);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceProperties returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return false;
    }

    int cc_major = prop.major;

    return (num_tokens > 128) && (cc_major >= 10);
}

bool silu_and_mul(const fastllm::Data &input, fastllm::Data &output) {
    int input_len = input.Count(0);
    int output_len = output.Count(0);

    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);
    int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
    int num_tokens = input_len / input.dims[input.dims.size() - 1];
    int elementSize = input.unitSize;
    int output_num_elements = mid;

    VecConfig vec_config = get_vec_config(num_tokens, elementSize, output_num_elements);
    bool use_256b_flag = use_256b(num_tokens);

    dim3 grid(num_tokens);
    dim3 block(vec_config.block_size);

    if (vec_config.use_vec) {
        if (use_256b_flag) {
            FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, silu_kernel, packed_silu_kernel, {
                LAUNCH_KERNEL(act_and_mul_kernel<scalar_t, packed_t, act_fn, packed_fn, true, true, true>
                    <<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput, mid));
            });
        } else {
            FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, silu_kernel, packed_silu_kernel, {
                LAUNCH_KERNEL(act_and_mul_kernel<scalar_t, packed_t, act_fn, packed_fn, true, true, false>
                    <<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput, mid));
            });
        }
    } else {
        FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, silu_kernel, packed_silu_kernel, {
            LAUNCH_KERNEL(act_and_mul_kernel<scalar_t, packed_t, act_fn, packed_fn, true, false, false>
                <<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput, mid));
        });
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool mul_and_silu(const fastllm::Data &input, fastllm::Data &output) // [..., 2 * d]
{
    int input_len = input.Count(0);
    int output_len = output.Count(0);

    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);
    int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
    int num_tokens = input_len / input.dims[input.dims.size() - 1];
    int elementSize = input.unitSize;
    int output_num_elements = mid;

    VecConfig vec_config = get_vec_config(num_tokens, elementSize, output_num_elements);
    bool use_256b_flag = use_256b(num_tokens);

    dim3 grid(num_tokens);
    dim3 block(vec_config.block_size);

    if (vec_config.use_vec) {
        if (use_256b_flag) {
            FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, silu_kernel, packed_silu_kernel, {
                LAUNCH_KERNEL(act_and_mul_kernel<scalar_t, packed_t, act_fn, packed_fn, false, true, true>
                    <<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput, mid));
            });
        } else {
            FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, silu_kernel, packed_silu_kernel, {
                LAUNCH_KERNEL(act_and_mul_kernel<scalar_t, packed_t, act_fn, packed_fn, false, true, false>
                    <<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput, mid));
            });
        }
    } else {
        FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, silu_kernel, packed_silu_kernel, {
            LAUNCH_KERNEL(act_and_mul_kernel<scalar_t, packed_t, act_fn, packed_fn, false, false, false>
                <<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput, mid));
        });
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool gelu_and_mul(const fastllm::Data &input, fastllm::Data &output) {
    int input_len = input.Count(0);
    int output_len = output.Count(0);

    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);
    int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
    int num_tokens = input_len / input.dims[input.dims.size() - 1];
    int elementSize = input.unitSize;
    int output_num_elements = mid;

    VecConfig vec_config = get_vec_config(num_tokens, elementSize, output_num_elements);
    bool use_256b_flag = use_256b(num_tokens);

    dim3 grid(num_tokens);
    dim3 block(vec_config.block_size);

    if (vec_config.use_vec) {
        if (use_256b_flag) {
            FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, gelu_kernel, packed_gelu_kernel, {
                LAUNCH_KERNEL(act_and_mul_kernel<scalar_t, packed_t, act_fn, packed_fn, true, true, true>
                    <<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput, mid));
            });
        } else {
            FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, gelu_kernel, packed_gelu_kernel, {
                LAUNCH_KERNEL(act_and_mul_kernel<scalar_t, packed_t, act_fn, packed_fn, true, true, false>
                    <<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput, mid));
            });
        }
    } else {
        FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, gelu_kernel, packed_gelu_kernel, {
            LAUNCH_KERNEL(act_and_mul_kernel<scalar_t, packed_t, act_fn, packed_fn, true, false, false>
                <<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput, mid));
        });
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool gelu_tanh_and_mul(const fastllm::Data &input, fastllm::Data &output) {
    int input_len = input.Count(0);
    int output_len = output.Count(0);

    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);
    int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
    int num_tokens = input_len / input.dims[input.dims.size() - 1];
    int elementSize = input.unitSize;
    int output_num_elements = mid;

    VecConfig vec_config = get_vec_config(num_tokens, elementSize, output_num_elements);
    bool use_256b_flag = use_256b(num_tokens);

    dim3 grid(num_tokens);
    dim3 block(vec_config.block_size);

    if (vec_config.use_vec) {
        if (use_256b_flag) {
            FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, gelu_tanh_kernel, packed_gelu_tanh_kernel, {
                LAUNCH_KERNEL(act_and_mul_kernel<scalar_t, packed_t, act_fn, packed_fn, true, true, true>
                    <<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput, mid));
            });
        } else {
            FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, gelu_tanh_kernel, packed_gelu_tanh_kernel, {
                LAUNCH_KERNEL(act_and_mul_kernel<scalar_t, packed_t, act_fn, packed_fn, true, true, false>
                    <<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput, mid));
            });
        }
    } else {
        FASTLLM_DISPATCH_FLOAT_TYPES(input.dataType, gelu_tanh_kernel, packed_gelu_tanh_kernel, {
            LAUNCH_KERNEL(act_and_mul_kernel<scalar_t, packed_t, act_fn, packed_fn, true, false, false>
                <<<grid, block>>>((scalar_t *)cudaOutput, (scalar_t *)cudaInput, mid));
        });
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}