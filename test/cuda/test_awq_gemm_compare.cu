#include "test_utils.h"
#include "awq/fastllm-awq-vllm-kernel.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>
#include <exception>
#include <vector>

namespace {

struct AwqGemmCase {
    int numTokens = 2;
    int inChannels = 128;
    int outChannels = 64;
    int groupCnt = 32;
};

void CheckCuda(cudaError_t state, const char *message) {
    if (state != cudaSuccess) {
        std::printf("[AWQ GEMM] CUDA error: %s: %s\n", message, cudaGetErrorString(state));
        std::exit(1);
    }
}

std::vector<half> ToHalfVector(const std::vector<float> &values) {
    std::vector<half> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = __float2half(values[i]);
    }
    return result;
}

std::vector<float> ToFloatVector(const std::vector<half> &values) {
    std::vector<float> result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = __half2float(values[i]);
    }
    return result;
}

std::vector<float> RoundToHalfFloats(const std::vector<float> &values) {
    return ToFloatVector(ToHalfVector(values));
}

std::vector<float> AwqGemmCpuReference(const std::vector<float> &input,
                                       const std::vector<uint8_t> &qweight,
                                       const std::vector<float> &scales,
                                       const std::vector<float> &mins,
                                       const std::vector<float> *bias,
                                       const AwqGemmCase &shape) {
    using fastllm::cuda_test::Expect;

    int groups = shape.inChannels / shape.groupCnt;
    Expect(shape.numTokens > 0, "numTokens must be positive.");
    Expect(shape.inChannels > 0, "inChannels must be positive.");
    Expect(shape.outChannels > 0, "outChannels must be positive.");
    Expect(shape.groupCnt > 0, "groupCnt must be positive.");
    Expect(shape.inChannels % shape.groupCnt == 0, "inChannels must be divisible by groupCnt.");
    Expect((shape.inChannels & 1) == 0, "inChannels must be even for packed int4 weights.");
    Expect(input.size() == (size_t)shape.numTokens * shape.inChannels, "input size mismatch.");
    Expect(qweight.size() == (size_t)shape.outChannels * (shape.inChannels / 2), "qweight size mismatch.");
    Expect(scales.size() == (size_t)shape.outChannels * groups, "scales size mismatch.");
    Expect(mins.size() == (size_t)shape.outChannels * groups, "mins size mismatch.");
    if (bias != nullptr) {
        Expect(bias->size() == (size_t)shape.outChannels, "bias size mismatch.");
    }

    std::vector<float> output((size_t)shape.numTokens * shape.outChannels, 0.0f);
    for (int token = 0; token < shape.numTokens; ++token) {
        for (int oc = 0; oc < shape.outChannels; ++oc) {
            float acc = bias == nullptr ? 0.0f : (*bias)[oc];
            for (int ic = 0; ic < shape.inChannels; ++ic) {
                uint8_t packed = qweight[(size_t)oc * (shape.inChannels / 2) + ic / 2];
                int q = (ic & 1) ? (packed & 0xF) : (packed >> 4);
                int group = ic / shape.groupCnt;
                float weight = mins[(size_t)oc * groups + group] +
                               scales[(size_t)oc * groups + group] * q;
                acc += input[(size_t)token * shape.inChannels + ic] * weight;
            }
            output[(size_t)token * shape.outChannels + oc] = acc;
        }
    }
    return output;
}

std::vector<float> AwqGemmGpuNaive(const std::vector<half> &input,
                                   const std::vector<uint8_t> &qweight,
                                   const std::vector<half> &scales,
                                   const std::vector<half> &mins,
                                   const std::vector<float> *bias,
                                   const AwqGemmCase &shape) {
    using fastllm::cuda_test::Expect;

    int groups = shape.inChannels / shape.groupCnt;
    size_t inputBytes = input.size() * sizeof(half);
    size_t qweightBytes = qweight.size() * sizeof(uint8_t);
    size_t scaleBytes = scales.size() * sizeof(half);
    size_t minBytes = mins.size() * sizeof(half);
    size_t outputCount = (size_t)shape.numTokens * shape.outChannels;
    size_t outputBytes = outputCount * sizeof(half);

    half *dInput = nullptr;
    uint8_t *dQweight = nullptr;
    half *dScales = nullptr;
    half *dMins = nullptr;
    float *dBias = nullptr;
    half *dOutput = nullptr;

    CheckCuda(cudaMalloc(&dInput, inputBytes), "malloc input");
    CheckCuda(cudaMalloc(&dQweight, qweightBytes), "malloc qweight");
    CheckCuda(cudaMalloc(&dScales, scaleBytes), "malloc scales");
    CheckCuda(cudaMalloc(&dMins, minBytes), "malloc mins");
    CheckCuda(cudaMalloc(&dOutput, outputBytes), "malloc output");
    if (bias != nullptr) {
        CheckCuda(cudaMalloc(&dBias, bias->size() * sizeof(float)), "malloc bias");
    }

    CheckCuda(cudaMemcpy(dInput, input.data(), inputBytes, cudaMemcpyHostToDevice), "copy input");
    CheckCuda(cudaMemcpy(dQweight, qweight.data(), qweightBytes, cudaMemcpyHostToDevice), "copy qweight");
    CheckCuda(cudaMemcpy(dScales, scales.data(), scaleBytes, cudaMemcpyHostToDevice), "copy scales");
    CheckCuda(cudaMemcpy(dMins, mins.data(), minBytes, cudaMemcpyHostToDevice), "copy mins");
    if (bias != nullptr) {
        CheckCuda(cudaMemcpy(dBias, bias->data(), bias->size() * sizeof(float), cudaMemcpyHostToDevice), "copy bias");
    }

    dim3 block(16, 16);
    dim3 grid((shape.outChannels + block.x - 1) / block.x,
              (shape.numTokens + block.y - 1) / block.y);
    FastllmCudaAwqGemmNaiveKernel <<< grid, block >>>(
        dInput, dQweight, dScales, dMins, dBias, dOutput,
        shape.numTokens, shape.inChannels, shape.outChannels, shape.groupCnt, groups);
    CheckCuda(cudaGetLastError(), "launch FastllmCudaAwqGemmNaiveKernel");
    CheckCuda(cudaDeviceSynchronize(), "sync FastllmCudaAwqGemmNaiveKernel");

    std::vector<half> hostOutput(outputCount);
    CheckCuda(cudaMemcpy(hostOutput.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost), "copy output");

    CheckCuda(cudaFree(dInput), "free input");
    CheckCuda(cudaFree(dQweight), "free qweight");
    CheckCuda(cudaFree(dScales), "free scales");
    CheckCuda(cudaFree(dMins), "free mins");
    CheckCuda(cudaFree(dOutput), "free output");
    if (dBias != nullptr) {
        CheckCuda(cudaFree(dBias), "free bias");
    }

    Expect(hostOutput.size() == outputCount, "GPU output size mismatch.");
    return ToFloatVector(hostOutput);
}

bool RunGpuCompare(bool withBias) {
    using namespace fastllm::cuda_test;

    AwqGemmCase shape;
    int groups = shape.inChannels / shape.groupCnt;
    std::vector<float> input = MakeRandomFloats((size_t)shape.numTokens * shape.inChannels, -1.0f, 1.0f, 1001);
    std::vector<uint8_t> qweight = MakeRandomInt4Weights(shape.outChannels, shape.inChannels, 1002);
    std::vector<float> scales = MakeRandomFloats((size_t)shape.outChannels * groups, 0.001f, 0.05f, 1003);
    std::vector<float> mins = MakeRandomFloats((size_t)shape.outChannels * groups, -0.4f, 0.1f, 1004);
    std::vector<float> bias = MakeRandomFloats(shape.outChannels, -0.2f, 0.2f, 1005);

    std::vector<half> inputHalf = ToHalfVector(input);
    std::vector<half> scalesHalf = ToHalfVector(scales);
    std::vector<half> minsHalf = ToHalfVector(mins);
    std::vector<float> inputRounded = RoundToHalfFloats(input);
    std::vector<float> scalesRounded = RoundToHalfFloats(scales);
    std::vector<float> minsRounded = RoundToHalfFloats(mins);

    const std::vector<float> *biasPtr = withBias ? &bias : nullptr;
    std::vector<float> expected = AwqGemmCpuReference(inputRounded, qweight, scalesRounded, minsRounded, biasPtr, shape);
    std::vector<float> actual = AwqGemmGpuNaive(inputHalf, qweight, scalesHalf, minsHalf, biasPtr, shape);

    constexpr float maxAbsTol = 2.0e-2f;
    constexpr float meanAbsTol = 2.0e-3f;
    constexpr float maxRelTol = 5.0e-2f;
    CompareResult result = CompareVectors(expected, actual, maxAbsTol, meanAbsTol, maxRelTol);
    PrintCompareResult(withBias ? "AWQ GEMM GPU compare with bias" : "AWQ GEMM GPU compare no bias",
                       result, maxAbsTol, meanAbsTol, maxRelTol);
    return result.passed;
}

}  // namespace

int main() {
    try {
        bool ok = true;
        ok = RunGpuCompare(false) && ok;
        ok = RunGpuCompare(true) && ok;
        if (!ok) {
            return 1;
        }
        std::printf("[AWQ GEMM] GPU compare PASS\n");
        return 0;
    } catch (const std::exception &e) {
        std::printf("[AWQ GEMM] FAIL: %s\n", e.what());
        return 1;
    }
}
