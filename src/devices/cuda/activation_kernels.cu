
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "activation_kernels.cuh"

#ifndef USE_ROCM
#    define FASTLLM_LDG(arg) __ldg(arg)
#else
#    define VLLM_LDG(arg) *(arg)
#endif

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

bool use_vec(uint32_t num_tokens, uint32_t elementSize, uint32_t num_elements) {
    if (elementSize == 0) {
        return false;
    }

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
    int support_vec = (cc_major >= 10 && num_tokens > 128) ? 32 : 16;
    int vec_size = support_vec / elementSize;
    bool use_vec = (num_elements % vec_size == 0);

    return use_vec;
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