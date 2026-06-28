// 受 vLLM fused AWQ GEMM 思路启发的 FastLLM 原生 AWQ GEMM 路径。
//
// 本文件不直接调用 vLLM 的 torch::stable::Tensor awq_gemm wrapper。
// FastLLM 的 INT4_GROUP 权重布局为：
//   qweight: [outChannels, inChannels / 2] uint8
//   scales:  [outChannels, groups]
//   mins:    [outChannels, groups]
// 而 vLLM 原始 AWQ GEMM 期望的布局为：
//   qweight: [inChannels, outChannels / 8] int32
//   scales:  [groups, outChannels]
//   qzeros:  [groups, outChannels / 8]
//
// 这里借鉴的是 vLLM 的融合思路：加载 int4 权重、即时反量化并执行 GEMM，
// 但直接消费 FastLLM 原生 INT4_GROUP 布局。

#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <cuda_fp16.h>

#include <cstdlib>
#include <cstdio>

namespace {

bool FastllmVllmKernelTraceEnabled() {
    static const bool enabled = (std::getenv("FASTLLM_TRACE_VLLM_KERNEL") != nullptr);
    return enabled;
}

void FastllmVllmKernelTraceSkip(const char *reason, int numTokens, int inChannels, int outChannels, int groupCnt) {
    if (!FastllmVllmKernelTraceEnabled()) {
        return;
    }
    printf("[Fastllm] vLLM AWQ GEMM skip: %s (tokens=%d ic=%d oc=%d groupCnt=%d)\n",
           reason, numTokens, inChannels, outChannels, groupCnt);
}

// 正确性优先的朴素 fused AWQ GEMM kernel。
// 一个线程计算一个 output[token, outChannel]，直接读取 FastLLM INT4_GROUP：
//   qweight[outChannel, inChannel / 2]，偶数 inChannel 在高 4bit，奇数在低 4bit。
[[maybe_unused]] __global__ void FastllmCudaAwqGemmNaiveKernel(const half *input,
                                                               const uint8_t *qweight,
                                                               const half *scales,
                                                               const half *mins,
                                                               const float *bias,
                                                               half *output,
                                                               int numTokens,
                                                               int inChannels,
                                                               int outChannels,
                                                               int groupCnt,
                                                               int groups) {
    int outChannel = blockIdx.x * blockDim.x + threadIdx.x;
    int token = blockIdx.y * blockDim.y + threadIdx.y;
    if (token >= numTokens || outChannel >= outChannels) {
        return;
    }

    const half *inputRow = input + (size_t)token * inChannels;
    const uint8_t *weightRow = qweight + (size_t)outChannel * (inChannels / 2);
    const half *scaleRow = scales + (size_t)outChannel * groups;
    const half *minRow = mins + (size_t)outChannel * groups;

    float acc = bias == nullptr ? 0.0f : bias[outChannel];
    for (int inChannel = 0; inChannel < inChannels; ++inChannel) {
        uint8_t packed = weightRow[inChannel >> 1];
        int q = (inChannel & 1) ? (packed & 0xF) : (packed >> 4);
        int group = inChannel / groupCnt;
        float scale = __half2float(scaleRow[group]);
        float min = __half2float(minRow[group]);
        acc += __half2float(inputRow[inChannel]) * (min + scale * q);
    }

    output[(size_t)token * outChannels + outChannel] = __float2half(acc);
}

}  // namespace

// 可选 vLLM-inspired AWQ GEMM 路径的入口。
// 只有当本路径已经成功写出 output 时才返回 true；对于不支持的 shape、
// dtype 或尚未实现的部分返回 false，让原始 FastLLM INT4_GROUP fallback
// 继续安全执行。
bool TryFastllmCudaAwqGemm(const fastllm::Data &input, fastllm::Data &weight,
                           const fastllm::Data &bias, fastllm::Data &output,
                           int numTokens, int inChannels, int outChannels) {
#ifndef ENABLE_VLLM_KERNEL
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)numTokens;
    (void)inChannels;
    (void)outChannels;
    return false;
#else
    (void)output;

    if (input.dataType != fastllm::DataType::FLOAT16) {
        FastllmVllmKernelTraceSkip("input is not FLOAT16", numTokens, inChannels, outChannels, weight.groupCnt);
        return false;
    }
    if (weight.dataType != fastllm::DataType::INT4_GROUP) {
        FastllmVllmKernelTraceSkip("weight is not INT4_GROUP", numTokens, inChannels, outChannels, weight.groupCnt);
        return false;
    }
    if (weight.cudaData == nullptr) {
        FastllmVllmKernelTraceSkip("weight cudaData is null", numTokens, inChannels, outChannels, weight.groupCnt);
        return false;
    }
    if (!bias.dims.empty() && bias.dataType != fastllm::DataType::FLOAT32) {
        FastllmVllmKernelTraceSkip("bias is neither empty nor FLOAT32", numTokens, inChannels, outChannels, weight.groupCnt);
        return false;
    }
    if (weight.groupCnt <= 0 || weight.groupCnt % 32 != 0) {
        FastllmVllmKernelTraceSkip("groupCnt is not a positive multiple of 32", numTokens, inChannels, outChannels, weight.groupCnt);
        return false;
    }
    if (inChannels <= 0 || outChannels <= 0 || numTokens <= 0) {
        FastllmVllmKernelTraceSkip("invalid matrix shape", numTokens, inChannels, outChannels, weight.groupCnt);
        return false;
    }
    if (inChannels % weight.groupCnt != 0) {
        FastllmVllmKernelTraceSkip("inChannels is not divisible by groupCnt", numTokens, inChannels, outChannels, weight.groupCnt);
        return false;
    }
    if (outChannels % 64 != 0) {
        FastllmVllmKernelTraceSkip("outChannels is not divisible by 64", numTokens, inChannels, outChannels, weight.groupCnt);
        return false;
    }

    // vLLM-inspired FastLLM 原生 fused AWQ GEMM 的占位实现。
    // 真正实现时应直接消费 weight.cudaData、scales 和 mins，
    // 而不是通过 vLLM 的 torch stable wrapper 转换。
    FastllmVllmKernelTraceSkip("raw-pointer vLLM AWQ kernel bridge is not implemented yet", numTokens, inChannels, outChannels, weight.groupCnt);
    return false;
#endif
}
