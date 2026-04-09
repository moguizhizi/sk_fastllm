#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>
#include "fastllm.h"

#define LAUNCH_KERNEL(...) __VA_ARGS__

#define FASTLLM_DISPATCH_FLOAT_TYPES(TYPE, BODY) \
    switch (TYPE) {                              \
        case fastllm::DataType::FLOAT32: {       \
            using scalar_t = float;              \
            using packed_t = float2;             \
            BODY;                                \
            break;                               \
        }                                        \
        case fastllm::DataType::FLOAT16: {       \
            using scalar_t = half;               \
            using packed_t = __half2;            \
            BODY;                                \
            break;                               \
        }                                        \
        case fastllm::DataType::BFLOAT16: {      \
            using scalar_t = __nv_bfloat16;      \
            using packed_t = __nv_bfloat162;     \
            BODY;                                \
            break;                               \
        }                                        \
    }

#define FASTLLM_DISPATCH_FP8_TYPES(TYPE, BODY)   \
    switch (TYPE) {                              \
        case fastllm::DataType::FP8_E4M3: {      \
            using fp8_t = c10::Float8_e4m3fn;    \
            BODY;                                \
            break;                               \
        }                                        \
                                                 \
    }
