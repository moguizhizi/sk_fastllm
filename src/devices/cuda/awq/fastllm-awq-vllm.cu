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
    (void)bias;
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

    // This adapter is intentionally kept as the only bridge between FastLLM's
    // Data-based linear flow and the vLLM AWQ kernel family. The current vLLM
    // awq_gemm entry takes torch::stable::Tensor objects, while FastLLM stores
    // INT4_GROUP weights as raw cudaData + scales/mins. The real kernel hook
    // should be added here after the vLLM kernel is converted to a raw-pointer
    // interface or an explicit qzeros cache is prepared from FastLLM mins.
    FastllmVllmKernelTraceSkip("raw-pointer vLLM AWQ kernel bridge is not implemented yet", numTokens, inChannels, outChannels, weight.groupCnt);
    return false;
#endif
}
