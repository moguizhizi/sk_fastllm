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
