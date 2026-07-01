#include "test_utils.h"
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "awq/fastllm-awq-vllm-kernel.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
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

std::vector<float> DataHalfToFloat(fastllm::Data &data) {
    data.ToDevice(fastllm::DataDevice::CPU);
    size_t count = (size_t)data.Count(0);
    std::vector<float> result(count);
    const uint16_t *src = (const uint16_t*)data.cpuData;
    for (size_t i = 0; i < count; ++i) {
        result[i] = __half2float(*(const half*)&src[i]);
    }
    return result;
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

bool RunFastllmDataPathCompareCase(const AwqGemmCase &shape, bool withBias,
                                   uint32_t seedBase, const std::string &name) {
    using namespace fastllm::cuda_test;

    int groups = shape.inChannels / shape.groupCnt;
    std::vector<float> input = MakeRandomFloats((size_t)shape.numTokens * shape.inChannels, -1.0f, 1.0f, seedBase + 1);
    std::vector<uint8_t> qweight = MakeRandomInt4Weights(shape.outChannels, shape.inChannels, seedBase + 2);
    std::vector<float> scales = MakeRandomFloats((size_t)shape.outChannels * groups, 0.001f, 0.05f, seedBase + 3);
    std::vector<float> mins = MakeRandomFloats((size_t)shape.outChannels * groups, -0.4f, 0.1f, seedBase + 4);
    std::vector<float> biasValues = MakeRandomFloats(shape.outChannels, -0.2f, 0.2f, seedBase + 5);

    std::vector<float> inputRounded = RoundToHalfFloats(input);
    std::vector<float> scalesRounded = RoundToHalfFloats(scales);
    std::vector<float> minsRounded = RoundToHalfFloats(mins);
    const std::vector<float> *biasPtr = withBias ? &biasValues : nullptr;
    std::vector<float> expected = AwqGemmCpuReference(inputRounded, qweight, scalesRounded, minsRounded, biasPtr, shape);

    fastllm::Data inputData(fastllm::DataType::FLOAT16, {shape.numTokens, shape.inChannels}, input);
    inputData.ToDevice(fastllm::DataDevice::CUDA);

    fastllm::Data weightData(fastllm::DataType::INT4_GROUP, {shape.outChannels, shape.inChannels});
    weightData.groupCnt = shape.groupCnt;
    weightData.group = groups;
    weightData.scales = scales;
    weightData.mins = mins;
    weightData.Allocate(false);
    std::memcpy(weightData.cpuData, qweight.data(), qweight.size());
    weightData.ToDevice(fastllm::DataDevice::CUDA);

    fastllm::Data emptyBias;
    fastllm::Data biasData(fastllm::DataType::FLOAT32, {shape.outChannels}, biasValues);
    if (withBias) {
        biasData.ToDevice(fastllm::DataDevice::CUDA);
    }
    fastllm::Data &bias = withBias ? biasData : emptyBias;

    fastllm::Data outputData(fastllm::DataType::FLOAT16, {shape.numTokens, shape.outChannels});
    outputData.Allocate(false);
    bool usedAwqPath = TryFastllmCudaAwqGemm(inputData, weightData, bias, outputData,
                                             shape.numTokens, shape.inChannels, shape.outChannels);
    Expect(usedAwqPath, "TryFastllmCudaAwqGemm returned false.");
    CheckCuda(cudaDeviceSynchronize(), "sync TryFastllmCudaAwqGemm");

    std::vector<float> actual = DataHalfToFloat(outputData);

    constexpr float maxAbsTol = 2.0e-2f;
    constexpr float meanAbsTol = 2.0e-3f;
    constexpr float maxRelTol = 5.0e-2f;
    CompareResult result = CompareVectors(expected, actual, maxAbsTol, meanAbsTol, maxRelTol);
    PrintCompareResult(name, result, maxAbsTol, meanAbsTol, maxRelTol);
    return result.passed;
}

bool RunFastllmDataPathCompare(bool withBias) {
    AwqGemmCase shape;
    return RunFastllmDataPathCompareCase(
        shape, withBias, 2000,
        withBias ? "AWQ GEMM FastLLM Data path with bias" : "AWQ GEMM FastLLM Data path no bias");
}

bool RunRealShapeCompare() {
    std::vector<AwqGemmCase> cases = {
        {1, 1024, 1024, 128},
        {4, 1024, 2048, 128},
        {1, 4096, 4096, 128},
    };

    bool ok = true;
    for (size_t i = 0; i < cases.size(); ++i) {
        const AwqGemmCase &shape = cases[i];
        std::string prefix = "AWQ GEMM real shape tokens=" + std::to_string(shape.numTokens) +
                             " ic=" + std::to_string(shape.inChannels) +
                             " oc=" + std::to_string(shape.outChannels) +
                             " group=" + std::to_string(shape.groupCnt);
        ok = RunFastllmDataPathCompareCase(shape, false, 5000 + (uint32_t)i * 20, prefix + " no bias") && ok;
        ok = RunFastllmDataPathCompareCase(shape, true, 6000 + (uint32_t)i * 20, prefix + " with bias") && ok;
    }
    return ok;
}

int GetBenchmarkIters() {
    const char *value = std::getenv("FASTLLM_AWQ_BENCH_ITERS");
    if (value == nullptr) {
        return 10;
    }
    int iters = std::atoi(value);
    return iters > 0 ? iters : 10;
}

float BenchmarkRawKernel(const AwqGemmCase &shape, int warmup, int iters) {
    int groups = shape.inChannels / shape.groupCnt;
    std::vector<float> input = fastllm::cuda_test::MakeRandomFloats(
        (size_t)shape.numTokens * shape.inChannels, -1.0f, 1.0f, 3001);
    std::vector<uint8_t> qweight = fastllm::cuda_test::MakeRandomInt4Weights(
        shape.outChannels, shape.inChannels, 3002);
    std::vector<float> scales = fastllm::cuda_test::MakeRandomFloats(
        (size_t)shape.outChannels * groups, 0.001f, 0.05f, 3003);
    std::vector<float> mins = fastllm::cuda_test::MakeRandomFloats(
        (size_t)shape.outChannels * groups, -0.4f, 0.1f, 3004);

    std::vector<half> inputHalf = ToHalfVector(input);
    std::vector<half> scalesHalf = ToHalfVector(scales);
    std::vector<half> minsHalf = ToHalfVector(mins);

    size_t inputBytes = inputHalf.size() * sizeof(half);
    size_t qweightBytes = qweight.size() * sizeof(uint8_t);
    size_t scaleBytes = scalesHalf.size() * sizeof(half);
    size_t minBytes = minsHalf.size() * sizeof(half);
    size_t outputBytes = (size_t)shape.numTokens * shape.outChannels * sizeof(half);

    half *dInput = nullptr;
    uint8_t *dQweight = nullptr;
    half *dScales = nullptr;
    half *dMins = nullptr;
    half *dOutput = nullptr;

    CheckCuda(cudaMalloc(&dInput, inputBytes), "bench malloc input");
    CheckCuda(cudaMalloc(&dQweight, qweightBytes), "bench malloc qweight");
    CheckCuda(cudaMalloc(&dScales, scaleBytes), "bench malloc scales");
    CheckCuda(cudaMalloc(&dMins, minBytes), "bench malloc mins");
    CheckCuda(cudaMalloc(&dOutput, outputBytes), "bench malloc output");
    CheckCuda(cudaMemcpy(dInput, inputHalf.data(), inputBytes, cudaMemcpyHostToDevice), "bench copy input");
    CheckCuda(cudaMemcpy(dQweight, qweight.data(), qweightBytes, cudaMemcpyHostToDevice), "bench copy qweight");
    CheckCuda(cudaMemcpy(dScales, scalesHalf.data(), scaleBytes, cudaMemcpyHostToDevice), "bench copy scales");
    CheckCuda(cudaMemcpy(dMins, minsHalf.data(), minBytes, cudaMemcpyHostToDevice), "bench copy mins");

    dim3 block(16, 16);
    dim3 grid((shape.outChannels + block.x - 1) / block.x,
              (shape.numTokens + block.y - 1) / block.y);
    for (int i = 0; i < warmup; ++i) {
        FastllmCudaAwqGemmNaiveKernel <<< grid, block >>>(
            dInput, dQweight, dScales, dMins, nullptr, dOutput,
            shape.numTokens, shape.inChannels, shape.outChannels, shape.groupCnt, groups);
    }
    CheckCuda(cudaGetLastError(), "bench launch raw warmup");
    CheckCuda(cudaDeviceSynchronize(), "bench sync raw warmup");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CheckCuda(cudaEventCreate(&start), "bench create raw start event");
    CheckCuda(cudaEventCreate(&stop), "bench create raw stop event");
    CheckCuda(cudaEventRecord(start), "bench record raw start");
    for (int i = 0; i < iters; ++i) {
        FastllmCudaAwqGemmNaiveKernel <<< grid, block >>>(
            dInput, dQweight, dScales, dMins, nullptr, dOutput,
            shape.numTokens, shape.inChannels, shape.outChannels, shape.groupCnt, groups);
    }
    CheckCuda(cudaEventRecord(stop), "bench record raw stop");
    CheckCuda(cudaEventSynchronize(stop), "bench sync raw stop");
    CheckCuda(cudaGetLastError(), "bench launch raw timed");

    float elapsedMs = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "bench elapsed raw");

    CheckCuda(cudaEventDestroy(start), "bench destroy raw start event");
    CheckCuda(cudaEventDestroy(stop), "bench destroy raw stop event");
    CheckCuda(cudaFree(dInput), "bench free input");
    CheckCuda(cudaFree(dQweight), "bench free qweight");
    CheckCuda(cudaFree(dScales), "bench free scales");
    CheckCuda(cudaFree(dMins), "bench free mins");
    CheckCuda(cudaFree(dOutput), "bench free output");
    return elapsedMs / iters;
}

float BenchmarkFastllmDataPath(const AwqGemmCase &shape, int warmup, int iters) {
    using fastllm::cuda_test::Expect;

    int groups = shape.inChannels / shape.groupCnt;
    std::vector<float> input = fastllm::cuda_test::MakeRandomFloats(
        (size_t)shape.numTokens * shape.inChannels, -1.0f, 1.0f, 4001);
    std::vector<uint8_t> qweight = fastllm::cuda_test::MakeRandomInt4Weights(
        shape.outChannels, shape.inChannels, 4002);
    std::vector<float> scales = fastllm::cuda_test::MakeRandomFloats(
        (size_t)shape.outChannels * groups, 0.001f, 0.05f, 4003);
    std::vector<float> mins = fastllm::cuda_test::MakeRandomFloats(
        (size_t)shape.outChannels * groups, -0.4f, 0.1f, 4004);

    fastllm::Data inputData(fastllm::DataType::FLOAT16, {shape.numTokens, shape.inChannels}, input);
    inputData.ToDevice(fastllm::DataDevice::CUDA);

    fastllm::Data weightData(fastllm::DataType::INT4_GROUP, {shape.outChannels, shape.inChannels});
    weightData.groupCnt = shape.groupCnt;
    weightData.group = groups;
    weightData.scales = scales;
    weightData.mins = mins;
    weightData.Allocate(false);
    std::memcpy(weightData.cpuData, qweight.data(), qweight.size());
    weightData.ToDevice(fastllm::DataDevice::CUDA);

    fastllm::Data emptyBias;
    fastllm::Data outputData(fastllm::DataType::FLOAT16, {shape.numTokens, shape.outChannels});
    outputData.Allocate(false);
    outputData.ToDevice(fastllm::DataDevice::CUDA, false);

    for (int i = 0; i < warmup; ++i) {
        bool usedAwqPath = TryFastllmCudaAwqGemm(inputData, weightData, emptyBias, outputData,
                                                 shape.numTokens, shape.inChannels, shape.outChannels);
        Expect(usedAwqPath, "bench TryFastllmCudaAwqGemm returned false during warmup.");
    }
    CheckCuda(cudaDeviceSynchronize(), "bench sync data path warmup");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CheckCuda(cudaEventCreate(&start), "bench create data path start event");
    CheckCuda(cudaEventCreate(&stop), "bench create data path stop event");
    CheckCuda(cudaEventRecord(start), "bench record data path start");
    for (int i = 0; i < iters; ++i) {
        bool usedAwqPath = TryFastllmCudaAwqGemm(inputData, weightData, emptyBias, outputData,
                                                 shape.numTokens, shape.inChannels, shape.outChannels);
        Expect(usedAwqPath, "bench TryFastllmCudaAwqGemm returned false.");
    }
    CheckCuda(cudaEventRecord(stop), "bench record data path stop");
    CheckCuda(cudaEventSynchronize(stop), "bench sync data path stop");
    CheckCuda(cudaGetLastError(), "bench launch data path timed");

    float elapsedMs = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "bench elapsed data path");
    CheckCuda(cudaEventDestroy(start), "bench destroy data path start event");
    CheckCuda(cudaEventDestroy(stop), "bench destroy data path stop event");
    return elapsedMs / iters;
}

void RunAwqGemmBenchmark() {
    int iters = GetBenchmarkIters();
    constexpr int warmup = 3;
    std::vector<AwqGemmCase> cases = {
        {1, 1024, 1024, 128},
        {4, 1024, 2048, 128},
        {1, 4096, 4096, 128},
    };

    std::printf("[AWQ GEMM bench] warmup=%d iters=%d\n", warmup, iters);
    for (const AwqGemmCase &shape : cases) {
        float rawMs = BenchmarkRawKernel(shape, warmup, iters);
        float dataPathMs = BenchmarkFastllmDataPath(shape, warmup, iters);
        std::printf("[AWQ GEMM bench] tokens=%d ic=%d oc=%d group=%d raw_kernel=%.4f ms data_path=%.4f ms\n",
                    shape.numTokens, shape.inChannels, shape.outChannels, shape.groupCnt,
                    rawMs, dataPathMs);
    }
}

}  // namespace

int main() {
    try {
        bool ok = true;
        ok = RunGpuCompare(false) && ok;
        ok = RunGpuCompare(true) && ok;
        ok = RunFastllmDataPathCompare(false) && ok;
        ok = RunFastllmDataPathCompare(true) && ok;
        ok = RunRealShapeCompare() && ok;
        if (!ok) {
            return 1;
        }
        RunAwqGemmBenchmark();
        std::printf("[AWQ GEMM] GPU and FastLLM Data path compare PASS\n");
        return 0;
    } catch (const std::exception &e) {
        std::printf("[AWQ GEMM] FAIL: %s\n", e.what());
        return 1;
    }
}
