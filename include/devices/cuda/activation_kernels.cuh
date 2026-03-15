#include <cstdint>

#include "fastllm.h"

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

__device__ __forceinline__ void ld256(u32x8_t &val, const u32x8_t *ptr);
__device__ __forceinline__ void st256(u32x8_t &val, u32x8_t *ptr);

template <typename packed_t>
__device__ __forceinline__ packed_t packed_mul(const packed_t &x, const packed_t &y);

template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t &), bool act_first>
__device__ __forceinline__ scalar_t compute(const scalar_t &x, const scalar_t &y);

template <typename packed_t, packed_t (*PACKED_ACT_FN)(const packed_t &), bool act_first>
__device__ __forceinline__ packed_t packed_compute(const packed_t &x, const packed_t &y);

template <typename T>
__device__ __forceinline__ T silu_kernel(const T &x);

template <typename packed_t>
__device__ __forceinline__ packed_t packed_silu_kernel(const packed_t &val);

template <typename packed_t>
__device__ __forceinline__ float2 cast_to_float2(const packed_t& val);

template <typename packed_t>
__device__ __forceinline__ packed_t cast_to_packed(const float2& val);

// Activation and gating kernel template.
template <typename scalar_t, typename packed_t, scalar_t (*ACT_FN)(const scalar_t &), packed_t (*PACKED_ACT_FN)(const packed_t &),
    bool act_first, bool use_vec, bool use_256b = false>
__global__ void act_and_mul_kernel(scalar_t *__restrict__ out, // [..., d]
    const scalar_t *__restrict__ input,                        // [..., 2, d]
    const int d);

bool use_vec(uint32_t num_tokens, uint32_t elementSize, uint32_t num_elements);
bool use_256b(uint32_t num_tokens);