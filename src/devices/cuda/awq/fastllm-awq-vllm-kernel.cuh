#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

// 正确性优先的朴素 fused AWQ GEMM kernel。
// 一个线程计算一个 output[token, outChannel]，直接读取 FastLLM INT4_GROUP：
//   qweight[outChannel, inChannel / 2]，偶数 inChannel 在高 4bit，奇数在低 4bit。
__global__ void FastllmCudaAwqGemmNaiveKernel(const half *input,
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
