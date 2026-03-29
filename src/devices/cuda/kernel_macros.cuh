#pragma once

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
