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
#include "fastllm-awq-vllm-kernel.cuh"

#include <cuda_fp16.h>

#include <cstdlib>
#include <cstdio>
#include <vector>

namespace {

static constexpr int INT4GROUP_CUDA_SCALES_IDX = 0;
static constexpr int INT4GROUP_CUDA_MINS_IDX = 1;

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

void FastllmVllmKernelTraceHit(const char *event, int numTokens, int inChannels, int outChannels,
                               int groupCnt, int groups, bool hasBias) {
    if (!FastllmVllmKernelTraceEnabled()) {
        return;
    }
    printf("[Fastllm] vLLM AWQ GEMM %s: tokens=%d ic=%d oc=%d groupCnt=%d groups=%d bias=%s\n",
           event, numTokens, inChannels, outChannels, groupCnt, groups, hasBias ? "yes" : "no");
}

void FastllmVllmKernelTraceCache(const char *name, bool created, size_t count) {
    if (!FastllmVllmKernelTraceEnabled()) {
        return;
    }
    printf("[Fastllm] vLLM AWQ GEMM cache %s: %s (%zu half values)\n",
           name, created ? "created" : "reuse", count);
}

bool FastllmCudaAwqEnsureScalesMinsOnDevice(fastllm::Data &weight, int outChannels) {
    int groups = weight.group;
    if (groups <= 0 ||
        weight.scales.size() != (size_t)outChannels * groups ||
        weight.mins.size() != (size_t)outChannels * groups) {
        return false;
    }

    if ((int)weight.extraCudaData.size() <= INT4GROUP_CUDA_MINS_IDX) {
        weight.extraCudaData.resize(INT4GROUP_CUDA_MINS_IDX + 1, nullptr);
    }

    size_t count = (size_t)outChannels * groups;
    if (weight.extraCudaData[INT4GROUP_CUDA_SCALES_IDX] == nullptr) {
        std::vector<half> hostScales(count);
        for (size_t i = 0; i < count; ++i) {
            hostScales[i] = __float2half(weight.scales[i]);
        }
        half *cudaScales = (half*)FastllmCudaMalloc(count * sizeof(half));
        FastllmCudaCopyFromHostToDevice(cudaScales, hostScales.data(), count * sizeof(half));
        weight.extraCudaData[INT4GROUP_CUDA_SCALES_IDX] = cudaScales;
        FastllmVllmKernelTraceCache("scales", true, count);
    } else {
        FastllmVllmKernelTraceCache("scales", false, count);
    }

    if (weight.extraCudaData[INT4GROUP_CUDA_MINS_IDX] == nullptr) {
        std::vector<half> hostMins(count);
        for (size_t i = 0; i < count; ++i) {
            hostMins[i] = __float2half(weight.mins[i]);
        }
        half *cudaMins = (half*)FastllmCudaMalloc(count * sizeof(half));
        FastllmCudaCopyFromHostToDevice(cudaMins, hostMins.data(), count * sizeof(half));
        weight.extraCudaData[INT4GROUP_CUDA_MINS_IDX] = cudaMins;
        FastllmVllmKernelTraceCache("mins", true, count);
    } else {
        FastllmVllmKernelTraceCache("mins", false, count);
    }

    return true;
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
    if (std::getenv("FASTLLM_DISABLE_VLLM_AWQ_GEMM") != nullptr) {
        FastllmVllmKernelTraceSkip("disabled by FASTLLM_DISABLE_VLLM_AWQ_GEMM",
                                   numTokens, inChannels, outChannels, weight.groupCnt);
        return false;
    }
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
    if (weight.group <= 0 || inChannels / weight.groupCnt != weight.group) {
        FastllmVllmKernelTraceSkip("group metadata does not match inChannels/groupCnt", numTokens, inChannels, outChannels, weight.groupCnt);
        return false;
    }
    if (!FastllmCudaAwqEnsureScalesMinsOnDevice(weight, outChannels)) {
        FastllmVllmKernelTraceSkip("failed to prepare scales/mins", numTokens, inChannels, outChannels, weight.groupCnt);
        return false;
    }

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);
    half *cudaScales = (half*)weight.extraCudaData[INT4GROUP_CUDA_SCALES_IDX];
    half *cudaMins = (half*)weight.extraCudaData[INT4GROUP_CUDA_MINS_IDX];
    const float *cudaBias = bias.dims.empty() ? nullptr : (const float*)bias.cudaData;

    dim3 block(16, 16);
    dim3 grid((outChannels + block.x - 1) / block.x,
              (numTokens + block.y - 1) / block.y);
    FastllmVllmKernelTraceHit("launch naive kernel", numTokens, inChannels, outChannels,
                              weight.groupCnt, weight.group, !bias.dims.empty());
    FastllmCudaAwqGemmNaiveKernel <<< grid, block >>>(
        cudaInput, (const uint8_t*)weight.cudaData, cudaScales, cudaMins, cudaBias,
        cudaOutput, numTokens, inChannels, outChannels, weight.groupCnt, weight.group);
    checkCudaErrors("Error: CUDA error when launching vLLM-inspired AWQ GEMM!", cudaGetLastError());
    FastllmCudaFinishOutput(output, cudaOutput);
    FastllmVllmKernelTraceHit("done", numTokens, inChannels, outChannels,
                              weight.groupCnt, weight.group, !bias.dims.empty());
    return true;
#endif
}
