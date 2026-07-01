#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define FASTLLM_CUDA_NO_MALLOC_CHECK_MACRO
#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "utils/utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <type_traits>
#include <vector>

// vLLM kernel 分支下拆出的模型级 CUDA 算子。
// ENABLE_VLLM_KERNEL=OFF 时不编译本文件，仍走原始 fastllm-cuda.cu 流程。
static constexpr float FASTLLM_CUDA_PI = 3.14159265358979323846f;

__device__ __forceinline__ float FastllmVllmOpsValueToFloat(float value) {
    return value;
}

__device__ __forceinline__ float FastllmVllmOpsValueToFloat(half value) {
    return __half2float(value);
}

__device__ __forceinline__ float FastllmVllmOpsValueToFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template<typename T>
__device__ __forceinline__ T FastllmVllmOpsFloatToValue(float value);

template<>
__device__ __forceinline__ float FastllmVllmOpsFloatToValue<float>(float value) {
    return value;
}

template<>
__device__ __forceinline__ half FastllmVllmOpsFloatToValue<half>(float value) {
    return __float2half(value);
}

template<>
__device__ __forceinline__ __nv_bfloat16 FastllmVllmOpsFloatToValue<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template<>
__device__ __forceinline__ __nv_fp8_e4m3 FastllmVllmOpsFloatToValue<__nv_fp8_e4m3>(float value) {
    return __nv_fp8_e4m3(value);
}

__device__ __forceinline__ float FastllmVllmOpsSoftplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return __expf(x);
    }
    return __logf(1.0f + __expf(x));
}

template <typename T>
__global__ void FastllmVllmOpsClampKernel(T *data, int len, bool hasMin, float minValue, bool hasMax, float maxValue) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) {
        return;
    }
    float x = FastllmVllmOpsValueToFloat(data[idx]);
    if (hasMin && x < minValue) {
        x = minValue;
    }
    if (hasMax && x > maxValue) {
        x = maxValue;
    }
    data[idx] = FastllmVllmOpsFloatToValue<T>(x);
}

bool FastllmCudaClamp(fastllm::Data &input, bool hasMin, float minValue, bool hasMax, float maxValue) {
    if (!hasMin && !hasMax) {
        return true;
    }
    if (input.dataDevice != fastllm::DataDevice::CUDA || input.cudaData == nullptr) {
        return false;
    }
    int len = input.Count(0);
    if (len <= 0) {
        return true;
    }
    int threadPerBlock = std::min(1024, len);
    int blocks = (len - 1) / threadPerBlock + 1;
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmVllmOpsClampKernel<<<blocks, threadPerBlock>>>(
            (float *)input.cudaData, len, hasMin, minValue, hasMax, maxValue);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmVllmOpsClampKernel<<<blocks, threadPerBlock>>>(
            (half *)input.cudaData, len, hasMin, minValue, hasMax, maxValue);
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        FastllmVllmOpsClampKernel<<<blocks, threadPerBlock>>>(
            (__nv_bfloat16 *)input.cudaData, len, hasMin, minValue, hasMax, maxValue);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmVllmOpsRMSNormSiluMulHalfKernel(const half *input, const float *weight,
                                                       const half *gateInput, half *output,
                                                       int channels, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    gateInput = gateInput + o * channels;
    output = output + o * channels;

    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS];
    __shared__ float scale;

    unsigned int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int half2_channels = channels / 2;
    const half2 *input_h2 = reinterpret_cast<const half2 *>(input);
    float sum2 = 0.0f;
    for (int i = tid; i < half2_channels; i += THREAD_PER_BLOCK) {
        half2 v = input_h2[i];
        float2 fv = __half22float2(v);
        sum2 += fv.x * fv.x + fv.y * fv.y;
    }
    if ((channels & 1) && tid == 0) {
        float x = __half2float(input[channels - 1]);
        sum2 += x * x;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = sum2;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane_id == 0) {
            scale = rsqrtf(val / channels + eps);
        }
    }
    __syncthreads();

    float s = scale;
    half2 *output_h2 = reinterpret_cast<half2 *>(output);
    const half2 *gate_h2 = reinterpret_cast<const half2 *>(gateInput);
    for (int i = tid; i < half2_channels; i += THREAD_PER_BLOCK) {
        half2 v = input_h2[i];
        float2 fv = __half22float2(v);

        half2 gateVec = gate_h2[i];
        half gate0In = __low2half(gateVec);
        half gate1In = __high2half(gateVec);

#ifdef CUDA_NO_TENSOR_CORE
        float gate0Float = __half2float(gate0In);
        float gate1Float = __half2float(gate1In);
        half gate0 = __float2half(gate0Float / (1.0f + expf(-gate0Float)));
        half gate1 = __float2half(gate1Float / (1.0f + expf(-gate1Float)));
#else
        half gate0 = __hdiv(gate0In, __hadd(__float2half(1.0f), hexp(-gate0In)));
        half gate1 = __hdiv(gate1In, __hadd(__float2half(1.0f), hexp(-gate1In)));
#endif

        half rms0 = __float2half_rn(fv.x * s * __ldg(&weight[i * 2]));
        half rms1 = __float2half_rn(fv.y * s * __ldg(&weight[i * 2 + 1]));

#ifdef CUDA_NO_TENSOR_CORE
        half out0 = __float2half(__half2float(rms0) * __half2float(gate0));
        half out1 = __float2half(__half2float(rms1) * __half2float(gate1));
#else
        half out0 = __hmul(rms0, gate0);
        half out1 = __hmul(rms1, gate1);
#endif
        output_h2[i] = __halves2half2(out0, out1);
    }

    if ((channels & 1) && tid == 0) {
        int last = channels - 1;
#ifdef CUDA_NO_TENSOR_CORE
        float gateFloat = __half2float(gateInput[last]);
        half gate = __float2half(gateFloat / (1.0f + expf(-gateFloat)));
        half rms = __float2half(__half2float(input[last]) * s * __ldg(&weight[last]));
        output[last] = __float2half(__half2float(rms) * __half2float(gate));
#else
        half gate = __hdiv(gateInput[last], __hadd(__float2half(1.0f), hexp(-gateInput[last])));
        half rms = __float2half_rn(__half2float(input[last]) * s * __ldg(&weight[last]));
        output[last] = __hmul(rms, gate);
#endif
    }
}

bool FastllmCudaRMSNormSiluMulFloat16(const fastllm::Data &input, fastllm::Data &weight,
                                      const fastllm::Data &gateInput, fastllm::Data &output, float eps) {
    if (input.dataDevice != fastllm::DataDevice::CUDA || gateInput.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA || weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (input.dataType != fastllm::DataType::FLOAT16 || gateInput.dataType != fastllm::DataType::FLOAT16 ||
        output.dataType != fastllm::DataType::FLOAT16 || weight.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    if (input.dims.size() == 0 || input.dims != gateInput.dims || output.dims != input.dims ||
        input.strides.empty() || gateInput.strides.empty() || output.strides.empty() ||
        input.strides.back() != 1 || gateInput.strides.back() != 1 || output.strides.back() != 1 ||
        weight.dims.size() != 1 || weight.dims[0] != input.dims.back()) {
        return false;
    }

    int channels = input.dims.back();
    int outer = input.Count(0) / channels;
    const half *cudaInput = (const half *)input.cudaData;
    const float *cudaWeight = (const float *)weight.cudaData;
    const half *cudaGateInput = (const half *)gateInput.cudaData;
    half *cudaOutput = (half *)output.cudaData;

    if (channels < 512) {
        FastllmVllmOpsRMSNormSiluMulHalfKernel<64><<<outer, 64>>>(cudaInput, cudaWeight, cudaGateInput, cudaOutput, channels, eps);
    } else if (channels < 4096) {
        FastllmVllmOpsRMSNormSiluMulHalfKernel<512><<<outer, 512>>>(cudaInput, cudaWeight, cudaGateInput, cudaOutput, channels, eps);
    } else {
        FastllmVllmOpsRMSNormSiluMulHalfKernel<1024><<<outer, 1024>>>(cudaInput, cudaWeight, cudaGateInput, cudaOutput, channels, eps);
    }
    checkCudaErrors("Error: CUDA error in FastllmCudaRMSNormSiluMulFloat16.", cudaGetLastError());
    return true;
}

__global__ void FastllmVllmOpsResetLogitsOfEOSAllKernel(int total, int eos_num, int stride, float *logits, int *eos_ids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int b = idx / eos_num;
    int e = idx - b * eos_num;
    logits[stride * b + eos_ids[e]] = 0;
}

void FastllmResetLogitsOfEOSAll(int batch, fastllm::Data *logits, const std::vector<int> &eos_ids) {
    if (batch <= 0 || eos_ids.empty()) {
        return;
    }
    struct ResetEosAllCache {
        int *cuda_eos_ids = nullptr;
        int capacity = 0;
        std::vector<int> last_eos_ids;
    };
    static thread_local std::map<int, ResetEosAllCache> caches;

    int device = FastllmCudaGetDevice();
    ResetEosAllCache &cache = caches[device];
    if (cache.cuda_eos_ids == nullptr || cache.capacity < (int)eos_ids.size()) {
        if (cache.cuda_eos_ids != nullptr) {
            FastllmCudaFree(cache.cuda_eos_ids);
        }
        cache.cuda_eos_ids = (int *)FastllmCudaMalloc(sizeof(int) * eos_ids.size());
        cache.capacity = (int)eos_ids.size();
        cache.last_eos_ids.clear();
    }
    if (cache.last_eos_ids != eos_ids) {
        FastllmCudaCopyFromHostToDevice(cache.cuda_eos_ids, (void *)eos_ids.data(), sizeof(int) * eos_ids.size());
        cache.last_eos_ids = eos_ids;
    }
    int total = batch * (int)eos_ids.size();
    FastllmVllmOpsResetLogitsOfEOSAllKernel<<<(total + 255) / 256, 256>>>(
        total, (int)eos_ids.size(), logits->Count(0) / batch, (float *)logits->cudaData, cache.cuda_eos_ids);
    checkCudaErrors("Error: CUDA error when reset logits of EOS all!", cudaGetLastError());
}

template<int THREAD_PER_BLOCK>
__global__ void FastllmVllmOpsShiftAppendWindowKernel(
    uint8_t *cache,
    const uint8_t *newToken,
    int channels,
    int window,
    int unitSize) {
    int channel = blockIdx.x * THREAD_PER_BLOCK + threadIdx.x;
    if (channel >= channels) {
        return;
    }

    uint8_t *cacheRow = cache + (size_t)channel * window * unitSize;
    const uint8_t *newTokenRow = newToken + (size_t)channel * unitSize;
    int shiftBytes = (window - 1) * unitSize;
    for (int i = 0; i < shiftBytes; i++) {
        cacheRow[i] = cacheRow[i + unitSize];
    }
    for (int i = 0; i < unitSize; i++) {
        cacheRow[shiftBytes + i] = newTokenRow[i];
    }
}

void FastllmCudaShiftAppendWindow(uint8_t *cache, const uint8_t *newToken, int channels, int window, int unitSize) {
    if (channels <= 0 || window <= 0 || unitSize <= 0) {
        return;
    }
    const int threads = 256;
    FastllmVllmOpsShiftAppendWindowKernel<threads>
        <<<(channels + threads - 1) / threads, threads>>>(cache, newToken, channels, window, unitSize);
    DeviceSync();
}

__global__ void FastllmVllmOpsCopyKernel(uint8_t *input, uint8_t *output, uint64_t len) {
    uint64_t idx = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * blockDim.x;
    if (idx < len) {
        output[idx] = input[idx];
    }
}

bool FastllmCudaCopy(const fastllm::Data &input, fastllm::Data &output) {
    uint64_t len = input.GetBytes();
    if (len == 0) {
        return true;
    }
    uint8_t *cudaInput = (uint8_t *)FastllmCudaPrepareInput(input);
    uint8_t *cudaOutput = (uint8_t *)FastllmCudaPrepareOutput(output);
    const int threads = 256;
    FastllmVllmOpsCopyKernel<<<(len - 1) / threads + 1, threads>>>(cudaInput, cudaOutput, len);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

__global__ void FastllmVllmOpsGegluKernel(float *__restrict__ input, float *__restrict__ output, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int id = idx / mid * spatial + idx % mid;
        float gate = input[id], up = input[id + mid];
        output[idx] = gate * 0.5f * (1.0f + erff(gate / 1.41421356237f)) * up;
    }
}

__global__ void FastllmVllmOpsGegluKernel(half *__restrict__ input, half *__restrict__ output, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int id = idx / mid * spatial + idx % mid;
        float gate = __half2float(input[id]), up = __half2float(input[id + mid]);
        output[idx] = __float2half(gate * 0.5f * (1.0f + erff(gate / 1.41421356237f)) * up);
    }
}

__global__ void FastllmVllmOpsGegluKernel(__nv_bfloat16 *__restrict__ input, __nv_bfloat16 *__restrict__ output,
                                          int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int id = idx / mid * spatial + idx % mid;
        float gate = __bfloat162float(input[id]), up = __bfloat162float(input[id + mid]);
        output[idx] = __float2bfloat16(gate * 0.5f * (1.0f + erff(gate / 1.41421356237f)) * up);
    }
}

bool FastllmCudaGeglu(const fastllm::Data &input, fastllm::Data &output) {
    int len = output.Count(0);
    float *cudaInput = (float *)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(output);
    int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
    int threads = std::min(1024, len);

    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmVllmOpsGegluKernel<<<(len - 1) / threads + 1, threads>>>(cudaInput, cudaOutput, len, spatial, mid);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmVllmOpsGegluKernel<<<(len - 1) / threads + 1, threads>>>((half *)cudaInput, (half *)cudaOutput, len, spatial, mid);
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        FastllmVllmOpsGegluKernel<<<(len - 1) / threads + 1, threads>>>(
            (__nv_bfloat16 *)cudaInput, (__nv_bfloat16 *)cudaOutput, len, spatial, mid);
    } else {
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
        return false;
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

__global__ void FastllmVllmOpsSigmoidMambaSoftplusKernel(
    float *sigmoidData,
    const float *softplusInputData,
    float *softplusOutputData,
    const float *aLog,
    const float *dtBias,
    int channels) {
    int o = blockIdx.x;
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        int idx = o * channels + i;
        float x = sigmoidData[idx];
        sigmoidData[idx] = 1.0f / (1.0f + expf(-x));
        softplusOutputData[idx] = -expf((double)aLog[i]) * FastllmVllmOpsSoftplus(softplusInputData[idx] + dtBias[i]);
    }
}

__global__ void FastllmVllmOpsSigmoidMambaSoftplusKernel(
    half *sigmoidData,
    const half *softplusInputData,
    half *softplusOutputData,
    const float *aLog,
    const float *dtBias,
    int channels) {
    int o = blockIdx.x;
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        int idx = o * channels + i;
#ifdef CUDA_NO_TENSOR_CORE
        float x = __half2float(sigmoidData[idx]);
        sigmoidData[idx] = __float2half(1.0f / (1.0f + expf(-x)));
#else
        half x = sigmoidData[idx];
        sigmoidData[idx] = __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(-x)));
#endif
        softplusOutputData[idx] = __float2half(
            -exp((double)aLog[i]) * FastllmVllmOpsSoftplus(__half2float(softplusInputData[idx]) + dtBias[i]));
    }
}

bool FastllmCudaSigmoidMambaSoftplus(fastllm::Data &sigmoidInputOutput, const fastllm::Data &softplusInput,
                                     fastllm::Data &softplusOutput, const fastllm::Data &aLogData,
                                     const fastllm::Data &dtBiasData) {
    if (sigmoidInputOutput.dataDevice != fastllm::DataDevice::CUDA ||
        softplusInput.dataDevice != fastllm::DataDevice::CUDA ||
        softplusOutput.dataDevice != fastllm::DataDevice::CUDA ||
        aLogData.dataDevice != fastllm::DataDevice::CUDA ||
        dtBiasData.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }

    int dimsLen = softplusInput.dims.size();
    int outer = softplusInput.Count(0) / softplusInput.Count(dimsLen - 1);
    int channels = softplusInput.dims[dimsLen - 1];
    int threads = std::min(64, channels);
    if (sigmoidInputOutput.dataType == fastllm::DataType::FLOAT32) {
        FastllmVllmOpsSigmoidMambaSoftplusKernel<<<outer, threads>>>(
            (float *)sigmoidInputOutput.cudaData, (const float *)softplusInput.cudaData,
            (float *)softplusOutput.cudaData, (const float *)aLogData.cudaData,
            (const float *)dtBiasData.cudaData, channels);
    } else if (sigmoidInputOutput.dataType == fastllm::DataType::FLOAT16) {
        FastllmVllmOpsSigmoidMambaSoftplusKernel<<<outer, threads>>>(
            (half *)sigmoidInputOutput.cudaData, (const half *)softplusInput.cudaData,
            (half *)softplusOutput.cudaData, (const float *)aLogData.cudaData,
            (const float *)dtBiasData.cudaData, channels);
    } else {
        return false;
    }
    checkCudaErrors("Error: CUDA error in FastllmCudaSigmoidMambaSoftplus.", cudaGetLastError());
    return true;
}

template<int THREAD_PER_BLOCK>
__global__ void FastllmVllmOpsGreedySamplingKernel(float *logits, int *output, int vocabSize) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    float *row = logits + (long long)b * vocabSize;

    __shared__ float maxData[THREAD_PER_BLOCK];
    __shared__ int idData[THREAD_PER_BLOCK];
    float localMax = -1.0e30f;
    int localId = 0;
    for (int i = tid; i < vocabSize; i += THREAD_PER_BLOCK) {
        float value = row[i];
        if (value > localMax) {
            localMax = value;
            localId = i;
        }
    }
    maxData[tid] = localMax;
    idData[tid] = localId;
    __syncthreads();

    for (int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s && maxData[tid] < maxData[tid + s]) {
            maxData[tid] = maxData[tid + s];
            idData[tid] = idData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[b] = idData[0];
    }
}

bool FastllmCudaGreedySampling(float *logits, int *output, int batch, int vocabSize) {
    if (batch <= 0) {
        return true;
    }
    if (logits == nullptr || output == nullptr || vocabSize <= 0) {
        fastllm::ErrorInFastLLM("FastllmCudaGreedySampling: invalid input.\n");
        return false;
    }
    FastllmVllmOpsGreedySamplingKernel<256><<<batch, 256>>>(logits, output, vocabSize);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("FastllmCudaGreedySampling: kernel launch failed: %s\n", cudaGetErrorString(status));
        return false;
    }
    return true;
}

template<int THREAD_PER_BLOCK>
__global__ void FastllmVllmOpsGreedySamplingWithScoresKernel(float *logits, int *output, float *scores, int vocabSize) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    float *row = logits + (long long)b * vocabSize;

    __shared__ float maxData[THREAD_PER_BLOCK];
    __shared__ int idData[THREAD_PER_BLOCK];
    float localMax = -1.0e30f;
    int localId = 0;
    for (int i = tid; i < vocabSize; i += THREAD_PER_BLOCK) {
        float value = row[i];
        if (value > localMax) {
            localMax = value;
            localId = i;
        }
    }
    maxData[tid] = localMax;
    idData[tid] = localId;
    __syncthreads();

    for (int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = maxData[tid + s];
            int otherId = idData[tid + s];
            if (maxData[tid] < other || (maxData[tid] == other && idData[tid] > otherId)) {
                maxData[tid] = other;
                idData[tid] = otherId;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[b] = idData[0];
        scores[b] = maxData[0];
    }
}

bool FastllmCudaGreedySamplingWithScores(float *logits, int *output, float *scores, int batch, int vocabSize) {
    if (batch <= 0) {
        return true;
    }
    if (logits == nullptr || output == nullptr || scores == nullptr || vocabSize <= 0) {
        fastllm::ErrorInFastLLM("FastllmCudaGreedySamplingWithScores: invalid input.\n");
        return false;
    }
    FastllmVllmOpsGreedySamplingWithScoresKernel<256><<<batch, 256>>>(logits, output, scores, vocabSize);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("FastllmCudaGreedySamplingWithScores: kernel launch failed: %s\n", cudaGetErrorString(status));
        return false;
    }
    return true;
}

__global__ void FastllmVllmOpsSampleTopKKernel(float *topk, float *temperatures,
                                               int *topKArr, float *topPArr,
                                               float *randoms, int *output,
                                               int maxTopK) {
    int b = blockIdx.x;
    float *base = topk + (long long)b * maxTopK * 2;
    int curTopK = topKArr[b];
    if (curTopK <= 1 || maxTopK <= 1 || temperatures[b] <= 1.0e-6f) {
        output[b] = (int)(base[0] + 1.0e-3f);
        return;
    }
    curTopK = min(curTopK, maxTopK);

    float topP = topPArr[b];
    float invTemp = 1.0f / temperatures[b];
    float maxValue = base[1] * invTemp;
    for (int i = 1; i < curTopK; i++) {
        maxValue = max(maxValue, base[i * 2 + 1] * invTemp);
    }

    float sum = 0.0f;
    for (int i = 0; i < curTopK; i++) {
        sum += expf(base[i * 2 + 1] * invTemp - maxValue);
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        output[b] = (int)(base[0] + 1.0e-3f);
        return;
    }

    float cutoffSum = 0.0f;
    int cutoff = curTopK;
    for (int i = 0; i < curTopK; i++) {
        cutoffSum += expf(base[i * 2 + 1] * invTemp - maxValue) / sum;
        if (cutoffSum > topP) {
            cutoff = i + 1;
            break;
        }
    }
    cutoffSum = max(cutoffSum, 1.0e-20f);

    float rnd = randoms[b];
    rnd = min(max(rnd, 0.0f), 0.99999994f) * cutoffSum;
    float curSum = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        curSum += expf(base[i * 2 + 1] * invTemp - maxValue) / sum;
        if (curSum > rnd || i == cutoff - 1) {
            output[b] = (int)(base[i * 2] + 1.0e-3f);
            return;
        }
    }
    output[b] = (int)(base[0] + 1.0e-3f);
}

bool FastllmCudaSampleTopK(float *topk, float *temperatures,
                           int *topKArr, float *topPArr, float *randoms,
                           int *output,
                           int batch, int maxTopK) {
    if (batch <= 0) {
        return true;
    }
    if (topk == nullptr || temperatures == nullptr || topKArr == nullptr ||
        topPArr == nullptr || randoms == nullptr || output == nullptr || maxTopK <= 0) {
        fastllm::ErrorInFastLLM("FastllmCudaSampleTopK: invalid input.\n");
        return false;
    }
    FastllmVllmOpsSampleTopKKernel<<<batch, 1>>>(topk, temperatures, topKArr, topPArr, randoms, output, maxTopK);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("FastllmCudaSampleTopK: kernel launch failed: %s\n", cudaGetErrorString(status));
        return false;
    }
    return true;
}

__global__ void FastllmVllmOpsMaskAndRemapExpertsKernel(
    int32_t *index,
    float *score,
    int total,
    int expertStart,
    int expertEnd) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }
    int expert = index[tid];
    if (expert >= expertStart && expert < expertEnd) {
        index[tid] = expert - expertStart;
    } else {
        index[tid] = 0;
        score[tid] = 0.0f;
    }
}

bool FastllmCudaMaskAndRemapExpertsForLocalRange(fastllm::Data &index, fastllm::Data &score,
                                                 int expertStart, int expertEnd) {
    if (expertStart < 0 || expertStart >= expertEnd ||
        index.dataDevice != fastllm::DataDevice::CUDA ||
        score.dataDevice != fastllm::DataDevice::CUDA ||
        index.dataType != fastllm::DataType::INT32 ||
        score.dataType != fastllm::DataType::FLOAT32 ||
        index.cudaData == nullptr || score.cudaData == nullptr ||
        index.Count(0) != score.Count(0)) {
        return false;
    }
    int total = index.Count(0);
    if (total <= 0) {
        return true;
    }
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    FastllmVllmOpsMaskAndRemapExpertsKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
        (int32_t *)index.cudaData, (float *)score.cudaData, total, expertStart, expertEnd);
    DeviceSync();
    return true;
}

__device__ __forceinline__ half FastllmVllmOpsSiluHalf(half value) {
#ifdef CUDA_NO_TENSOR_CORE
    float x = __half2float(value);
    return __float2half(x / (1.0f + expf(-x)));
#else
    return __hdiv(value, __hadd(__float2half(1.0f), hexp(-value)));
#endif
}

__global__ void FastllmVllmOpsConv1DPerChannelSiluSingleTokenHalfKernel(
    const half *input,
    const float *weight,
    const float *bias,
    half *output,
    int channels) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }

    const half *curInput = input + c * 4;
    const float *curWeight = weight + c * 4;
    float value = bias ? bias[c] : 0.0f;
    value += __half2float(curInput[0]) * curWeight[0];
    value += __half2float(curInput[1]) * curWeight[1];
    value += __half2float(curInput[2]) * curWeight[2];
    value += __half2float(curInput[3]) * curWeight[3];
    output[c] = FastllmVllmOpsSiluHalf(__float2half_rn(value));
}

__global__ void FastllmVllmOpsShiftAppendConv1DPerChannelSiluSingleTokenHalfKernel(
    half *cache,
    const half *newToken,
    const float *weight,
    const float *bias,
    half *output,
    int batch,
    int channels) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (row >= total) {
        return;
    }

    int c = row % channels;
    half *cacheRow = cache + row * 4;
    const float *curWeight = weight + c * 4;
    half x0 = cacheRow[1];
    half x1 = cacheRow[2];
    half x2 = cacheRow[3];
    half x3 = newToken[row];
    cacheRow[0] = x0;
    cacheRow[1] = x1;
    cacheRow[2] = x2;
    cacheRow[3] = x3;

    float value = bias ? bias[c] : 0.0f;
    value += __half2float(x0) * curWeight[0];
    value += __half2float(x1) * curWeight[1];
    value += __half2float(x2) * curWeight[2];
    value += __half2float(x3) * curWeight[3];
    output[row] = FastllmVllmOpsSiluHalf(__float2half_rn(value));
}

__global__ void FastllmVllmOpsShiftAppendConv1DPerChannelSiluTwoTokenHalfKernel(
    half *cache,
    const half *newTokens,
    const float *weight,
    const float *bias,
    half *output,
    half *firstTokenCache,
    int batch,
    int channels) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (row >= total) {
        return;
    }

    int c = row % channels;
    half *cacheRow = cache + row * 4;
    const half *tokenRow = newTokens + row * 2;
    half t0 = tokenRow[0];
    half t1 = tokenRow[1];
    half x1 = cacheRow[1];
    half x2 = cacheRow[2];
    half x3 = cacheRow[3];

    if (firstTokenCache) {
        half *firstRow = firstTokenCache + row * 4;
        firstRow[0] = x1;
        firstRow[1] = x2;
        firstRow[2] = x3;
        firstRow[3] = t0;
    }

    cacheRow[0] = x2;
    cacheRow[1] = x3;
    cacheRow[2] = t0;
    cacheRow[3] = t1;

    const float *curWeight = weight + c * 4;
    float value0 = bias ? bias[c] : 0.0f;
    value0 += __half2float(x1) * curWeight[0];
    value0 += __half2float(x2) * curWeight[1];
    value0 += __half2float(x3) * curWeight[2];
    value0 += __half2float(t0) * curWeight[3];

    float value1 = bias ? bias[c] : 0.0f;
    value1 += __half2float(x2) * curWeight[0];
    value1 += __half2float(x3) * curWeight[1];
    value1 += __half2float(t0) * curWeight[2];
    value1 += __half2float(t1) * curWeight[3];

    half *outputRow = output + row * 2;
    outputRow[0] = FastllmVllmOpsSiluHalf(__float2half_rn(value0));
    outputRow[1] = FastllmVllmOpsSiluHalf(__float2half_rn(value1));
}

__global__ void FastllmVllmOpsShiftAppendConv1DPerChannelSiluSingleTokenHalfPointerKernel(
    half **caches,
    const half *newToken,
    const float *weight,
    const float *bias,
    half *output,
    int batch,
    int channels) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (row >= total) {
        return;
    }

    int b = row / channels;
    int c = row - b * channels;
    half *cacheRow = caches[b] + c * 4;
    const float *curWeight = weight + c * 4;
    half x0 = cacheRow[1];
    half x1 = cacheRow[2];
    half x2 = cacheRow[3];
    half x3 = newToken[row];
    cacheRow[0] = x0;
    cacheRow[1] = x1;
    cacheRow[2] = x2;
    cacheRow[3] = x3;

    float value = bias ? bias[c] : 0.0f;
    value += __half2float(x0) * curWeight[0];
    value += __half2float(x1) * curWeight[1];
    value += __half2float(x2) * curWeight[2];
    value += __half2float(x3) * curWeight[3];
    output[row] = FastllmVllmOpsSiluHalf(__float2half_rn(value));
}

__global__ void FastllmVllmOpsShiftAppendConv1DPerChannelSiluSingleTokenHalfSlotKernel(
    half *cachePool,
    const int *slotIds,
    const half *newToken,
    const float *weight,
    const float *bias,
    half *output,
    int batch,
    int channels) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (row >= total) {
        return;
    }

    int b = row / channels;
    int c = row - b * channels;
    int slot = slotIds[b];
    half *cacheRow = cachePool + ((size_t)slot * channels + c) * 4;
    const float *curWeight = weight + c * 4;
    half x0 = cacheRow[1];
    half x1 = cacheRow[2];
    half x2 = cacheRow[3];
    half x3 = newToken[row];
    cacheRow[0] = x0;
    cacheRow[1] = x1;
    cacheRow[2] = x2;
    cacheRow[3] = x3;

    float value = bias ? bias[c] : 0.0f;
    value += __half2float(x0) * curWeight[0];
    value += __half2float(x1) * curWeight[1];
    value += __half2float(x2) * curWeight[2];
    value += __half2float(x3) * curWeight[3];
    output[row] = FastllmVllmOpsSiluHalf(__float2half_rn(value));
}

static bool FastllmVllmOpsValidConv1DWeightShape(const fastllm::Data &weight, int channels) {
    return (weight.dims.size() == 2 && weight.dims[0] == channels && weight.dims[1] == 4) ||
           (weight.dims.size() == 3 && weight.dims[0] == channels && weight.dims[1] == 1 && weight.dims[2] == 4);
}

static bool FastllmVllmOpsValidConv1DBias(const fastllm::Data &bias, int channels) {
    return bias.dims.empty() ||
           (bias.dims.size() == 1 && bias.dims[0] == channels &&
            bias.dataDevice == fastllm::DataDevice::CUDA &&
            bias.dataType == fastllm::DataType::FLOAT32);
}

bool FastllmCudaConv1DPerChannelSiluSingleTokenFloat16(const fastllm::Data &input, fastllm::Data &weight,
                                                       fastllm::Data &bias, fastllm::Data &output) {
    if (input.dataDevice != fastllm::DataDevice::CUDA || weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (input.dataType != fastllm::DataType::FLOAT16 || weight.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    if (input.dims.size() != 3 || input.dims[0] != 1 || input.dims[2] != 4 ||
        input.strides.empty() || input.strides.back() != 1 ||
        !FastllmVllmOpsValidConv1DWeightShape(weight, input.dims[1]) ||
        !FastllmVllmOpsValidConv1DBias(bias, input.dims[1])) {
        return false;
    }

    output.dataType = input.dataType;
    output.Resize({1, input.dims[1], 1});
    output.ToDevice(input.dataDevice, input.dataDeviceIds);
    output.Allocate();

    int channels = input.dims[1];
    int threads = 256;
    int blocks = (channels + threads - 1) / threads;
    FastllmVllmOpsConv1DPerChannelSiluSingleTokenHalfKernel<<<blocks, threads>>>(
        (const half *)input.cudaData, (const float *)weight.cudaData,
        bias.dims.empty() ? nullptr : (const float *)bias.cudaData,
        (half *)output.cudaData, channels);
    checkCudaErrors("Error: CUDA error in FastllmCudaConv1DPerChannelSiluSingleTokenFloat16.", cudaGetLastError());
    return true;
}

bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(fastllm::Data &cache, const fastllm::Data &newToken,
                                                                  fastllm::Data &weight, fastllm::Data &bias,
                                                                  fastllm::Data &output) {
    if (cache.dataDevice != fastllm::DataDevice::CUDA || newToken.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (cache.dataType != fastllm::DataType::FLOAT16 || newToken.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    if (cache.dims.size() != 3 || cache.dims[0] <= 0 || cache.dims[2] != 4 ||
        newToken.dims.size() != 3 || newToken.dims[0] != cache.dims[0] ||
        newToken.dims[1] != cache.dims[1] || newToken.dims[2] != 1 ||
        cache.strides.empty() || newToken.strides.empty() ||
        cache.strides.back() != 1 || newToken.strides.back() != 1 ||
        !FastllmVllmOpsValidConv1DWeightShape(weight, cache.dims[1]) ||
        !FastllmVllmOpsValidConv1DBias(bias, cache.dims[1])) {
        return false;
    }

    output.dataType = cache.dataType;
    output.Resize({cache.dims[0], cache.dims[1], 1});
    output.ToDevice(cache.dataDevice, cache.dataDeviceIds);
    output.Allocate();

    int batch = cache.dims[0];
    int channels = cache.dims[1];
    int total = batch * channels;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    FastllmVllmOpsShiftAppendConv1DPerChannelSiluSingleTokenHalfKernel<<<blocks, threads>>>(
        (half *)cache.cudaData, (const half *)newToken.cudaData, (const float *)weight.cudaData,
        bias.dims.empty() ? nullptr : (const float *)bias.cudaData, (half *)output.cudaData, batch, channels);
    checkCudaErrors("Error: CUDA error in FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16.", cudaGetLastError());
    return true;
}

bool FastllmCudaShiftAppendConv1DPerChannelSiluTwoTokenFloat16(fastllm::Data &cache, const fastllm::Data &newTokens,
                                                               fastllm::Data &weight, fastllm::Data &bias,
                                                               fastllm::Data &output, fastllm::Data *firstTokenCache) {
    if (cache.dataDevice != fastllm::DataDevice::CUDA || newTokens.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (cache.dataType != fastllm::DataType::FLOAT16 || newTokens.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    if (cache.dims.size() != 3 || cache.dims[0] <= 0 || cache.dims[2] != 4 ||
        newTokens.dims.size() != 3 || newTokens.dims[0] != cache.dims[0] ||
        newTokens.dims[1] != cache.dims[1] || newTokens.dims[2] != 2 ||
        cache.strides.empty() || newTokens.strides.empty() ||
        cache.strides.back() != 1 || newTokens.strides.back() != 1 ||
        !FastllmVllmOpsValidConv1DWeightShape(weight, cache.dims[1]) ||
        !FastllmVllmOpsValidConv1DBias(bias, cache.dims[1])) {
        return false;
    }

    output.dataType = cache.dataType;
    output.Resize({cache.dims[0], cache.dims[1], 2});
    output.ToDevice(cache.dataDevice, cache.dataDeviceIds);
    output.Allocate();

    if (firstTokenCache != nullptr) {
        firstTokenCache->dataType = cache.dataType;
        firstTokenCache->Resize(cache.dims);
        firstTokenCache->ToDevice(cache.dataDevice, cache.dataDeviceIds);
        firstTokenCache->Allocate();
    }

    int batch = cache.dims[0];
    int channels = cache.dims[1];
    int total = batch * channels;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    FastllmVllmOpsShiftAppendConv1DPerChannelSiluTwoTokenHalfKernel<<<blocks, threads>>>(
        (half *)cache.cudaData, (const half *)newTokens.cudaData, (const float *)weight.cudaData,
        bias.dims.empty() ? nullptr : (const float *)bias.cudaData, (half *)output.cudaData,
        firstTokenCache == nullptr ? nullptr : (half *)firstTokenCache->cudaData, batch, channels);
    checkCudaErrors("Error: CUDA error in FastllmCudaShiftAppendConv1DPerChannelSiluTwoTokenFloat16.", cudaGetLastError());
    return true;
}

bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchDevicePointers(
    void *cudaCachePointers, int batch, const fastllm::Data &firstCache, const fastllm::Data &newToken,
    fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output) {
    if (cudaCachePointers == nullptr || batch <= 0) {
        return false;
    }
    if (firstCache.dataDevice != fastllm::DataDevice::CUDA ||
        newToken.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (firstCache.dataType != fastllm::DataType::FLOAT16 ||
        newToken.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    if (firstCache.dims.size() != 3 || firstCache.dims[0] != 1 || firstCache.dims[2] != 4 ||
        firstCache.strides.empty() || firstCache.strides.back() != 1 ||
        firstCache.cudaData == nullptr ||
        !FastllmVllmOpsValidConv1DWeightShape(weight, firstCache.dims[1]) ||
        !FastllmVllmOpsValidConv1DBias(bias, firstCache.dims[1])) {
        return false;
    }

    int channels = firstCache.dims[1];
    if (newToken.dims.size() != 3 || newToken.dims[0] != batch ||
        newToken.dims[1] != channels || newToken.dims[2] != 1 ||
        newToken.strides.empty() || newToken.strides.back() != 1) {
        return false;
    }

    output.dataType = firstCache.dataType;
    output.Resize({batch, channels, 1});
    output.ToDevice(firstCache.dataDevice, firstCache.dataDeviceIds);
    output.Allocate();

    int total = batch * channels;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    FastllmVllmOpsShiftAppendConv1DPerChannelSiluSingleTokenHalfPointerKernel<<<blocks, threads>>>(
        (half **)cudaCachePointers, (const half *)newToken.cudaData, (const float *)weight.cudaData,
        bias.dims.empty() ? nullptr : (const float *)bias.cudaData, (half *)output.cudaData, batch, channels);
    checkCudaErrors("Error: CUDA error in FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchDevicePointers.",
                    cudaGetLastError());
    return true;
}

bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchPointers(
    const std::vector<fastllm::Data *> &caches, const fastllm::Data &newToken,
    fastllm::Data &weight, fastllm::Data &bias, fastllm::Data &output) {
    if (caches.empty() || caches[0] == nullptr) {
        return false;
    }
    fastllm::Data *first = caches[0];
    if (first->dataDevice != fastllm::DataDevice::CUDA ||
        newToken.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (first->dataType != fastllm::DataType::FLOAT16 ||
        newToken.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    if (first->dims.size() != 3 || first->dims[0] != 1 || first->dims[2] != 4 ||
        first->strides.empty() || first->strides.back() != 1) {
        return false;
    }

    int batch = (int)caches.size();
    int channels = first->dims[1];
    if (newToken.dims.size() != 3 || newToken.dims[0] != batch ||
        newToken.dims[1] != channels || newToken.dims[2] != 1 ||
        newToken.strides.empty() || newToken.strides.back() != 1 ||
        !FastllmVllmOpsValidConv1DWeightShape(weight, channels) ||
        !FastllmVllmOpsValidConv1DBias(bias, channels)) {
        return false;
    }
    for (int i = 0; i < batch; i++) {
        fastllm::Data *cur = caches[i];
        if (cur == nullptr ||
            cur->dataDevice != first->dataDevice ||
            cur->dataType != first->dataType ||
            cur->dims != first->dims ||
            cur->strides.empty() ||
            cur->strides.back() != 1 ||
            cur->cudaData == nullptr) {
            return false;
        }
    }

    void **cpuPointers = new void *[batch];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i] = caches[i]->cudaData;
    }
    void **cudaPointers = (void **)FastllmCudaMalloc(sizeof(void *) * batch);
    cudaError_t state = cudaMemcpy(cudaPointers, cpuPointers, sizeof(void *) * batch, cudaMemcpyHostToDevice);
    delete[] cpuPointers;
    checkCudaErrors("Error: CUDA error when copy conv cache pointers to GPU!", state);

    bool ret = FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchDevicePointers(
        cudaPointers, batch, *first, newToken, weight, bias, output);
    FastllmCudaFree(cudaPointers);
    return ret;
}

bool FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchSlots(
    void *cudaCachePool, void *cudaSlotIds, int batch, const fastllm::Data &firstCache,
    const fastllm::Data &newToken, fastllm::Data &weight, fastllm::Data &bias,
    fastllm::Data &output) {
    if (cudaCachePool == nullptr || cudaSlotIds == nullptr || batch <= 0) {
        return false;
    }
    if (firstCache.dataDevice != fastllm::DataDevice::CUDA ||
        newToken.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (firstCache.dataType != fastllm::DataType::FLOAT16 ||
        newToken.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    if (firstCache.dims.size() != 3 || firstCache.dims[0] != 1 || firstCache.dims[2] != 4 ||
        firstCache.strides.empty() || firstCache.strides.back() != 1 ||
        firstCache.cudaData == nullptr ||
        !FastllmVllmOpsValidConv1DWeightShape(weight, firstCache.dims[1]) ||
        !FastllmVllmOpsValidConv1DBias(bias, firstCache.dims[1])) {
        return false;
    }

    int channels = firstCache.dims[1];
    if (newToken.dims.size() != 3 || newToken.dims[0] != batch ||
        newToken.dims[1] != channels || newToken.dims[2] != 1 ||
        newToken.strides.empty() || newToken.strides.back() != 1) {
        return false;
    }

    output.dataType = firstCache.dataType;
    output.Resize({batch, channels, 1});
    output.ToDevice(firstCache.dataDevice, firstCache.dataDeviceIds);
    output.Allocate();

    int total = batch * channels;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    FastllmVllmOpsShiftAppendConv1DPerChannelSiluSingleTokenHalfSlotKernel<<<blocks, threads>>>(
        (half *)cudaCachePool, (const int *)cudaSlotIds, (const half *)newToken.cudaData,
        (const float *)weight.cudaData, bias.dims.empty() ? nullptr : (const float *)bias.cudaData,
        (half *)output.cudaData, batch, channels);
    checkCudaErrors("Error: CUDA error in FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16BatchSlots.",
                    cudaGetLastError());
    return true;
}

__device__ __forceinline__ float FastllmLlama3InvFreq(float invFreq, float factor, float originalMaxPosition,
                                                      float lowFreqFactor, float highFreqFactor) {
    float wavelen = 2.0f * FASTLLM_CUDA_PI / invFreq;
    float lowWavelen = originalMaxPosition / lowFreqFactor;
    float highWavelen = originalMaxPosition / highFreqFactor;
    float invLlama = wavelen > lowWavelen ? invFreq / factor : invFreq;
    if (!(wavelen < highWavelen) && !(wavelen > lowWavelen)) {
        float smooth = (originalMaxPosition / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
        invLlama = (1.0f - smooth) * invFreq / factor + smooth * invFreq;
    }
    return invLlama;
}

__global__ void FastllmLlama3RopeEncodingKernel(float *data, float *positionIds,
                                                int len, int bs, int spatial, int n, int m, int partStride, int rotateDim,
                                                float ropeTheta, float factor, float originalMaxPosition,
                                                float lowFreqFactor, float highFreqFactor) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int half = rotateDim / 2;
    float position = positionIds[b * partStride + l];
    float invFreq = 1.0f / powf(ropeTheta, (float)(2 * j) / rotateDim);
    invFreq = FastllmLlama3InvFreq(invFreq, factor, originalMaxPosition, lowFreqFactor, highFreqFactor);
    float freq = position * invFreq;
    float curSin = sinf(freq), curCos = cosf(freq);
    float *d = data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + half];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + half] = va * curSin + vb * curCos;
}

__global__ void FastllmLlama3RopeEncodingKernel(half *data, float *positionIds,
                                                int len, int bs, int spatial, int n, int m, int partStride, int rotateDim,
                                                float ropeTheta, float factor, float originalMaxPosition,
                                                float lowFreqFactor, float highFreqFactor) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int halfDim = rotateDim / 2;
    float position = positionIds[b * partStride + l];
    float invFreq = 1.0f / powf(ropeTheta, (float)(2 * j) / rotateDim);
    invFreq = FastllmLlama3InvFreq(invFreq, factor, originalMaxPosition, lowFreqFactor, highFreqFactor);
    float freq = position * invFreq;
    float curSin = sinf(freq), curCos = cosf(freq);
    half *d = data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __half2float(d[i * m]), vb = __half2float(d[i * m + halfDim]);
    d[i * m] = __float2half(va * curCos - vb * curSin);
    d[i * m + halfDim] = __float2half(va * curSin + vb * curCos);
}

__global__ void FastllmLlama3RopeEncodingKernel(__nv_bfloat16 *data, float *positionIds,
                                                int len, int bs, int spatial, int n, int m, int partStride, int rotateDim,
                                                float ropeTheta, float factor, float originalMaxPosition,
                                                float lowFreqFactor, float highFreqFactor) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int halfDim = rotateDim / 2;
    float position = positionIds[b * partStride + l];
    float invFreq = 1.0f / powf(ropeTheta, (float)(2 * j) / rotateDim);
    invFreq = FastllmLlama3InvFreq(invFreq, factor, originalMaxPosition, lowFreqFactor, highFreqFactor);
    float freq = position * invFreq;
    float curSin = sinf(freq), curCos = cosf(freq);
    __nv_bfloat16 *d = data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __bfloat162float(d[i * m]), vb = __bfloat162float(d[i * m + halfDim]);
    d[i * m] = __float2bfloat16(va * curCos - vb * curSin);
    d[i * m + halfDim] = __float2bfloat16(va * curSin + vb * curCos);
}

__global__ void FastllmQwen35InterleavedRopeKernel(float *data, float *positionIds,
                                                   int len, int spatial, int n, int m, int positionStride, int rotateDim,
                                                   int sectionH, int sectionW, float ropeTheta, float ropeScale) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int j = threadIdx.x;
    int half = rotateDim / 2;
    int row = 0;
    if (j % 3 == 1 && j < sectionH * 3) {
        row = 1;
    } else if (j % 3 == 2 && j < sectionW * 3) {
        row = 2;
    }
    float position = positionIds[row * positionStride + l] / ropeScale;
    float freq = position / powf(ropeTheta, (float)(2 * j) / rotateDim);
    float curSin = sinf(freq);
    float curCos = cosf(freq);
    float *d = (float *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + half];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + half] = va * curSin + vb * curCos;
}

__global__ void FastllmQwen35InterleavedRopeKernel(half *data, float *positionIds,
                                                   int len, int spatial, int n, int m, int positionStride, int rotateDim,
                                                   int sectionH, int sectionW, float ropeTheta, float ropeScale) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int j = threadIdx.x;
    int half_dim = rotateDim / 2;
    int row = 0;
    if (j % 3 == 1 && j < sectionH * 3) {
        row = 1;
    } else if (j % 3 == 2 && j < sectionW * 3) {
        row = 2;
    }
    float position = positionIds[row * positionStride + l] / ropeScale;
    float freq = position / powf(ropeTheta, (float)(2 * j) / rotateDim);
    float curSin = sinf(freq);
    float curCos = cosf(freq);
    half *d = (half *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __half2float(d[i * m]), vb = __half2float(d[i * m + half_dim]);
    d[i * m] = __float2half(va * curCos - vb * curSin);
    d[i * m + half_dim] = __float2half(va * curSin + vb * curCos);
}

__global__ void FastllmQwen35InterleavedRopeKernel(__nv_bfloat16 *data, float *positionIds,
                                                   int len, int spatial, int n, int m, int positionStride, int rotateDim,
                                                   int sectionH, int sectionW, float ropeTheta, float ropeScale) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int j = threadIdx.x;
    int half_dim = rotateDim / 2;
    int row = 0;
    if (j % 3 == 1 && j < sectionH * 3) {
        row = 1;
    } else if (j % 3 == 2 && j < sectionW * 3) {
        row = 2;
    }
    float position = positionIds[row * positionStride + l] / ropeScale;
    float freq = position / powf(ropeTheta, (float)(2 * j) / rotateDim);
    float curSin = sinf(freq);
    float curCos = cosf(freq);
    __nv_bfloat16 *d = (__nv_bfloat16 *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __bfloat162float(d[i * m]), vb = __bfloat162float(d[i * m + half_dim]);
    d[i * m] = __float2bfloat16(va * curCos - vb * curSin);
    d[i * m + half_dim] = __float2bfloat16(va * curSin + vb * curCos);
}

// ============================================================
template <int THREAD_PER_BLOCK, typename T, typename TKV>
__global__ void FastllmQKVRMSNormRopeSplitAppendPagedCacheKernel(
    T *qkvData,              // [bs, seqlen, total_dim], 物理布局; 逻辑含义为 batch 个 token
    float *qNormWeight,      // [head_dim]
    float *kNormWeight,      // [head_dim]
    float *positionIds,      // [bs, partStride]
    T *qOutputData,          // [bsz * q_heads, seqlen, head_dim] (permuted output)
    TKV *pagedKData,         // paged K cache raw data
    TKV *pagedVData,         // paged V cache raw data
    int32_t *insertIndexs,   // [batch] page index for each batch (逻辑 batch)
    int32_t *insertPositions,// [batch] page offset for each batch (逻辑 batch)
    int32_t *lastPageLens,   // optional [batch], filled with page offset after append
    int outer,               // bs * seqlen = 总 token 数
    int total_dim,           // (q_heads + k_heads + v_heads) * head_dim
    int q_heads,
    int k_heads,
    int v_heads,
    int head_dim,
    int bs,                  // qkv.dims[0], 物理 batch 维
    int seqlen,              // qkv.dims[1], 物理 seqlen 维
    int partStride,          // positionIds.dims.back()
    int rotateDim,
    float eps,
    float ropeTheta,
    float ropeScale,
    int pageLen,             // page length for paged cache
    int maxPages,            // max pages in paged cache
    int batch,               // 逻辑 batch 数（= insertIndexs 长度）
    int doQKNorm,            // 是否做 QK RMSNorm（0 = 跳过）
    int useLlama3,
    float llama3Factor,
    float llama3OriginalMaxPosition,
    float llama3LowFreqFactor,
    float llama3HighFreqFactor
) {
    int total_heads = q_heads + k_heads + v_heads;
    int block_id = blockIdx.x;
    int token_id = block_id / total_heads;  // [0, outer), 即第几个 token
    int head_id = block_id % total_heads;

    // 物理维度索引（用于定位 qkv 和 positionIds）
    int phys_b = token_id / seqlen;   // qkv 的物理 batch 索引
    int phys_l = token_id % seqlen;   // qkv 的物理 seq 索引

    // 逻辑 batch 索引（用于 insertIndexs / insertPositions）
    // 在 decode 路径: bs=1, seqlen=batch, 逻辑 batch_idx = token_id
    // 在单 batch 路径: bs=1, seqlen=1, batch=1, 逻辑 batch_idx = 0
    int batch_idx = token_id;  // 每个 token 对应一个逻辑 batch（decode 模式下 seqlen_per_batch=1）

    unsigned int tid = threadIdx.x;

    if (batch_idx >= batch) {
        return;
    }

    int insertPageIdx = insertIndexs[batch_idx];
    int insertPageOffset = insertPositions[batch_idx];
    bool validInsert = insertPageIdx >= 0 && insertPageIdx < maxPages &&
                       insertPageOffset >= 0 && insertPageOffset < pageLen;

    if (lastPageLens != nullptr && head_id == 0 && tid == 0) {
        lastPageLens[batch_idx] = validInsert ? insertPageOffset + 1 : 0;
    }

    // 确定当前 head 在 qkv 中的偏移
    int offset_in_total;
    if (head_id < q_heads) {
        offset_in_total = head_id * head_dim;
    } else if (head_id < q_heads + k_heads) {
        offset_in_total = q_heads * head_dim + (head_id - q_heads) * head_dim;
    } else {
        offset_in_total = (q_heads + k_heads) * head_dim + (head_id - q_heads - k_heads) * head_dim;
    }

    T *base = qkvData + token_id * total_dim + offset_in_total;

    if (head_id < q_heads + k_heads) {
        // ======== Q or K head: (optional) RMSNorm + RoPE ========
        if (doQKNorm) {
            float *normWeight = (head_id < q_heads) ? qNormWeight : kNormWeight;

            // Step 1: RMSNorm
            __shared__ float sdata[THREAD_PER_BLOCK];
            __shared__ float scale;

            float local_sum2 = 0.0f;
            for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
                float x = (float)base[i];
                local_sum2 += x * x;
            }
            sdata[tid] = local_sum2;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
                if (tid < s) sdata[tid] += sdata[tid + s];
                __syncthreads();
            }
            if (tid < 32) {
                volatile float *now = sdata;
                now[tid] += now[tid + 32];
                now[tid] += now[tid + 16];
                now[tid] += now[tid + 8];
                now[tid] += now[tid + 4];
                now[tid] += now[tid + 2];
                now[tid] += now[tid + 1];
            }
            __syncthreads();

            if (tid == 0) {
                scale = 1.0f / sqrtf(sdata[0] / head_dim + eps);
            }
            __syncthreads();

            // Apply RMSNorm in-place
            for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
                base[i] = (T)((float)base[i] * scale * normWeight[i]);
            }
            __syncthreads();
        }

        // Step 2: RoPE Encoding
        // Single-token batch decode may arrive as either [1, batch] or [batch, 1].
        // The logical token order is still token_id, matching insertIndexs/insertPositions.
        int half_rotate = rotateDim / 2;
        if ((int)tid < half_rotate) {
            int j = tid;
            int positionOffset = phys_b * partStride + phys_l;
            if (outer == batch && batch > 1) {
                positionOffset = (partStride == 1) ? phys_b : batch_idx;
            }
            float rawPosition = positionIds[positionOffset];
            float invFreq = 1.0f / powf(ropeTheta, (float)(2 * j) / rotateDim);
            float freq;
            if (useLlama3) {
                invFreq = FastllmLlama3InvFreq(invFreq, llama3Factor,
                                                llama3OriginalMaxPosition,
                                                llama3LowFreqFactor,
                                                llama3HighFreqFactor);
                freq = rawPosition * invFreq;
            } else {
                float position = (float)((int)rawPosition) / ropeScale;
                freq = position * invFreq;
            }
            float curSin = sinf(freq);
            float curCos = cosf(freq);

            float va = (float)base[j];
            float vb = (float)base[j + half_rotate];
            base[j]               = (T)(va * curCos - vb * curSin);
            base[j + half_rotate] = (T)(va * curSin + vb * curCos);
        }
        __syncthreads();

        // Step 3: Write output
        if (head_id < q_heads) {
            // Q head: 写入 qOutput，布局 [bsz * q_heads, seqlen, head_dim]
            // Permute: [bs, seqlen, q_heads, head_dim] -> [bs, q_heads, seqlen, head_dim] -> [bs * q_heads, seqlen, head_dim]
            // 即 (phys_b, phys_l, head_id) -> (phys_b * q_heads + head_id, phys_l, :)
            T *dst = qOutputData + ((phys_b * q_heads + head_id) * seqlen + phys_l) * head_dim;
            for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
                dst[i] = base[i];
            }
        } else {
            // K head: 直接写入 paged K cache
            // pagedData layout: [maxPages, pageLen, numHeads, headDim]
            // 用逻辑 batch_idx 索引 insertIndexs / insertPositions
            if (!validInsert) {
                return;
            }
            int kh = head_id - q_heads;
            int pageStride = pageLen * k_heads * head_dim;
            int tokenStride = k_heads * head_dim;
            TKV *dst = pagedKData + (size_t)insertPageIdx * pageStride + insertPageOffset * tokenStride + kh * head_dim;
            for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
                dst[i] = FastllmVllmOpsFloatToValue<TKV>(FastllmVllmOpsValueToFloat(base[i]));
            }
        }
    } else {
        // ======== V head: 直接拷贝到 paged V cache（无需 RMSNorm/RoPE）========
        // 用逻辑 batch_idx 索引 insertIndexs / insertPositions
        if (!validInsert) {
            return;
        }
        int vh = head_id - q_heads - k_heads;
        int pageStride = pageLen * v_heads * head_dim;
        int tokenStride = v_heads * head_dim;
        TKV *dst = pagedVData + (size_t)insertPageIdx * pageStride + insertPageOffset * tokenStride + vh * head_dim;
        for (int i = tid; i < head_dim; i += THREAD_PER_BLOCK) {
            dst[i] = FastllmVllmOpsFloatToValue<TKV>(FastllmVllmOpsValueToFloat(base[i]));
        }
    }
}

bool FastllmCudaQKVRMSNormRopeSplitAppendPagedCache(
    fastllm::Data &qkv,
    fastllm::Data &qNormWeight,
    fastllm::Data &kNormWeight,
    const fastllm::Data &positionIds,
    fastllm::Data &qOutput,
    uint8_t *pagedKData,
    uint8_t *pagedVData,
    int32_t *insertIndexs,
    int32_t *insertPositions,
    int32_t *lastPageLens,
    int q_heads, int k_heads, int head_dim,
    int rotateDim, float eps, float ropeTheta, float ropeScale,
    int pageLen, int maxPages, fastllm::DataType pagedDataType, int batch,
    int doQKNorm,
    int useLlama3, float llama3Factor,
    float llama3OriginalMaxPosition,
    float llama3LowFreqFactor,
    float llama3HighFreqFactor
) {
    float *cudaQKV = (float *) FastllmCudaPrepareInput(qkv);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);

    int bs = qkv.dims[0];
    int seqlen = qkv.dims[1];
    int total_dim = qkv.dims[2];
    int v_heads = k_heads; // v_heads == k_heads
    int outer = bs * seqlen;
    int total_heads = q_heads + k_heads + v_heads;
    int grid_size = outer * total_heads;
    int partStride = (int)positionIds.dims.back();

    // 确保 qOutput 已分配
    float *cudaQOutput = (float*)qOutput.cudaData;

    auto launch = [&](auto TPB, auto *qkvPtr, auto *qOutputPtr, auto *pagedTag) {
        using QT = std::remove_pointer_t<decltype(qkvPtr)>;
        using KVT = std::remove_pointer_t<decltype(pagedTag)>;
        FastllmQKVRMSNormRopeSplitAppendPagedCacheKernel<decltype(TPB)::value, QT, KVT><<<grid_size, decltype(TPB)::value>>>(
            qkvPtr, (float*)qNormWeight.cudaData, (float*)kNormWeight.cudaData,
            cudaPositionIds, qOutputPtr,
            (KVT*)pagedKData, (KVT*)pagedVData, insertIndexs, insertPositions, lastPageLens,
            outer, total_dim, q_heads, k_heads, v_heads, head_dim,
            bs, seqlen, partStride, rotateDim, eps, ropeTheta, ropeScale, pageLen, maxPages, batch, doQKNorm,
            useLlama3, llama3Factor, llama3OriginalMaxPosition,
            llama3LowFreqFactor, llama3HighFreqFactor);
    };

    auto launchByPagedType = [&](auto TPB, auto *qkvPtr, auto *qOutputPtr) {
        if (pagedDataType == fastllm::DataType::FLOAT32) {
            launch(TPB, qkvPtr, qOutputPtr, (float*)nullptr);
        } else if (pagedDataType == fastllm::DataType::FLOAT16) {
            launch(TPB, qkvPtr, qOutputPtr, (half*)nullptr);
        } else if (pagedDataType == fastllm::DataType::BFLOAT16) {
            launch(TPB, qkvPtr, qOutputPtr, (__nv_bfloat16*)nullptr);
        } else if (pagedDataType == fastllm::DataType::FP8_E4M3) {
            launch(TPB, qkvPtr, qOutputPtr, (__nv_fp8_e4m3*)nullptr);
        } else {
            fastllm::ErrorInFastLLM("FastllmCudaQKVRMSNormRopeSplitAppendPagedCache: unsupported pagedDataType.\n");
        }
    };

    if (qkv.dataType == fastllm::DataType::FLOAT32) {
        if (head_dim <= 64) launchByPagedType(std::integral_constant<int, 64>{}, (float*)cudaQKV, (float*)cudaQOutput);
        else if (head_dim <= 128) launchByPagedType(std::integral_constant<int, 128>{}, (float*)cudaQKV, (float*)cudaQOutput);
        else launchByPagedType(std::integral_constant<int, 512>{}, (float*)cudaQKV, (float*)cudaQOutput);
    } else if (qkv.dataType == fastllm::DataType::FLOAT16) {
        if (head_dim <= 64) launchByPagedType(std::integral_constant<int, 64>{}, (half*)cudaQKV, (half*)cudaQOutput);
        else if (head_dim <= 128) launchByPagedType(std::integral_constant<int, 128>{}, (half*)cudaQKV, (half*)cudaQOutput);
        else launchByPagedType(std::integral_constant<int, 512>{}, (half*)cudaQKV, (half*)cudaQOutput);
    } else if (qkv.dataType == fastllm::DataType::BFLOAT16) {
        if (head_dim <= 64) launchByPagedType(std::integral_constant<int, 64>{}, (__nv_bfloat16*)cudaQKV, (__nv_bfloat16*)cudaQOutput);
        else if (head_dim <= 128) launchByPagedType(std::integral_constant<int, 128>{}, (__nv_bfloat16*)cudaQKV, (__nv_bfloat16*)cudaQOutput);
        else launchByPagedType(std::integral_constant<int, 512>{}, (__nv_bfloat16*)cudaQKV, (__nv_bfloat16*)cudaQOutput);
    } else {
        fastllm::ErrorInFastLLM("FastllmCudaQKVRMSNormRopeSplitAppendPagedCache: unsupported qkv dataType.\n");
    }

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    cudaError_t launchError = cudaGetLastError();
    if (launchError != cudaSuccess) {
        printf("FastllmCudaQKVRMSNormRopeSplitAppendPagedCache: kernel launch failed: %s\n",
               cudaGetErrorString(launchError));
        return false;
    }
    // 注意: 不需要 FinishOutput qkv，因为 qkv 内容已经不再需要
    return true;
}

// ============================================================
// Qwen3.5 gated attention decode fusion:
//   input layout: [Q/Gate interleaved per Q head, K, V]
//   Q -> RMSNorm + RoPE -> qOutput
//   Gate -> gateOutput
//   K -> RMSNorm + RoPE -> paged K cache
//   V -> paged V cache
// ============================================================
template <int THREAD_PER_BLOCK, typename T, typename TKV>
__global__ void FastllmQwen35QGateKVRMSNormRopeSplitAppendPagedCacheKernel(
    T *qgatekvData,
    float *qNormWeight,
    float *kNormWeight,
    float *positionIds,
    T *qOutputData,
    T *gateOutputData,
    TKV *pagedKData,
    TKV *pagedVData,
    int32_t *insertIndexs,
    int32_t *insertPositions,
    int32_t *lastPageLens,
    int outer,
    int totalDim,
    int qHeads,
    int kHeads,
    int headDim,
    int bs,
    int seqlen,
    int positionStride,
    int rotaryDim,
    int sectionH,
    int sectionW,
    int useInterleavedRope,
    float eps,
    float ropeTheta,
    float ropeScale,
    int pageLen,
    int batch,
    int doQKNorm) {
    int totalHeads = qHeads + kHeads + kHeads;
    int blockId = blockIdx.x;
    int tokenId = blockId / totalHeads;
    int headId = blockId % totalHeads;
    int physB = tokenId / seqlen;
    int physL = tokenId % seqlen;
    int batchIdx = tokenId;
    unsigned int tid = threadIdx.x;

    if (lastPageLens != nullptr && headId == 0 && tid == 0 && batchIdx < batch) {
        lastPageLens[batchIdx] = insertPositions[batchIdx] + 1;
    }

    int qGateDim = qHeads * headDim * 2;
    T *base = nullptr;
    float *normWeight = nullptr;
    bool isQ = headId < qHeads;
    bool isK = headId >= qHeads && headId < qHeads + kHeads;
    if (isQ) {
        base = qgatekvData + (size_t)tokenId * totalDim + headId * headDim * 2;
        normWeight = qNormWeight;
    } else if (isK) {
        int kh = headId - qHeads;
        base = qgatekvData + (size_t)tokenId * totalDim + qGateDim + kh * headDim;
        normWeight = kNormWeight;
    }

    if (isQ || isK) {
        if (doQKNorm) {
            __shared__ float sdata[THREAD_PER_BLOCK];
            __shared__ float scale;

            float localSum2 = 0.0f;
            for (int i = tid; i < headDim; i += THREAD_PER_BLOCK) {
                float x = FastllmVllmOpsValueToFloat(base[i]);
                localSum2 += x * x;
            }
            sdata[tid] = localSum2;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            if (tid < 32) {
                volatile float *now = sdata;
                now[tid] += now[tid + 32];
                now[tid] += now[tid + 16];
                now[tid] += now[tid + 8];
                now[tid] += now[tid + 4];
                now[tid] += now[tid + 2];
                now[tid] += now[tid + 1];
            }
            __syncthreads();

            if (tid == 0) {
                scale = 1.0f / sqrtf(sdata[0] / headDim + eps);
            }
            __syncthreads();

            for (int i = tid; i < headDim; i += THREAD_PER_BLOCK) {
                base[i] = FastllmVllmOpsFloatToValue<T>(
                    FastllmVllmOpsValueToFloat(base[i]) * scale * normWeight[i]);
            }
            __syncthreads();
        }

        int halfRotate = rotaryDim / 2;
        if ((int)tid < halfRotate) {
            int j = tid;
            float rawPosition;
            if (useInterleavedRope) {
                int row = 0;
                if (j % 3 == 1 && j < sectionH * 3) {
                    row = 1;
                } else if (j % 3 == 2 && j < sectionW * 3) {
                    row = 2;
                }
                int logicalL = (outer == batch && batch > 1) ? batchIdx : physL;
                rawPosition = positionIds[row * positionStride + logicalL];
            } else {
                int positionOffset = physB * positionStride + physL;
                if (outer == batch && batch > 1) {
                    positionOffset = (positionStride == 1) ? physB : batchIdx;
                }
                rawPosition = positionIds[positionOffset];
            }
            float position = rawPosition / ropeScale;
            float freq = position / powf(ropeTheta, (float)(2 * j) / rotaryDim);
            float curSin = sinf(freq);
            float curCos = cosf(freq);
            float va = FastllmVllmOpsValueToFloat(base[j]);
            float vb = FastllmVllmOpsValueToFloat(base[j + halfRotate]);
            base[j] = FastllmVllmOpsFloatToValue<T>(va * curCos - vb * curSin);
            base[j + halfRotate] = FastllmVllmOpsFloatToValue<T>(va * curSin + vb * curCos);
        }
        __syncthreads();

        if (isQ) {
            T *qDst = qOutputData + ((physB * qHeads + headId) * seqlen + physL) * headDim;
            T *gateBase = qgatekvData + (size_t)tokenId * totalDim + headId * headDim * 2 + headDim;
            T *gateDst = gateOutputData + (size_t)tokenId * qHeads * headDim + headId * headDim;
            for (int i = tid; i < headDim; i += THREAD_PER_BLOCK) {
                qDst[i] = base[i];
                gateDst[i] = gateBase[i];
            }
        } else {
            int kh = headId - qHeads;
            int pageIdx = insertIndexs[batchIdx];
            int pageOffset = insertPositions[batchIdx];
            int pageStride = pageLen * kHeads * headDim;
            int tokenStride = kHeads * headDim;
            TKV *kDst = pagedKData + (size_t)pageIdx * pageStride +
                pageOffset * tokenStride + kh * headDim;
            for (int i = tid; i < headDim; i += THREAD_PER_BLOCK) {
                kDst[i] = FastllmVllmOpsFloatToValue<TKV>(FastllmVllmOpsValueToFloat(base[i]));
            }
        }
    } else {
        int vh = headId - qHeads - kHeads;
        T *vBase = qgatekvData + (size_t)tokenId * totalDim + qGateDim + kHeads * headDim + vh * headDim;
        int pageIdx = insertIndexs[batchIdx];
        int pageOffset = insertPositions[batchIdx];
        int pageStride = pageLen * kHeads * headDim;
        int tokenStride = kHeads * headDim;
        TKV *vDst = pagedVData + (size_t)pageIdx * pageStride +
            pageOffset * tokenStride + vh * headDim;
        for (int i = tid; i < headDim; i += THREAD_PER_BLOCK) {
            vDst[i] = FastllmVllmOpsFloatToValue<TKV>(FastllmVllmOpsValueToFloat(vBase[i]));
        }
    }
}

bool FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache(
    fastllm::Data &qgatekv,
    fastllm::Data &qNormWeight,
    fastllm::Data &kNormWeight,
    const fastllm::Data &positionIds,
    fastllm::Data &qOutput,
    fastllm::Data &gateOutput,
    uint8_t *pagedKData,
    uint8_t *pagedVData,
    int32_t *insertIndexs,
    int32_t *insertPositions,
    int32_t *lastPageLens,
    int qHeads, int kHeads, int headDim,
    int rotaryDim, int sectionT, int sectionH, int sectionW,
    float eps, float ropeTheta, float ropeScale,
    int pageLen, fastllm::DataType pagedDataType, int batch,
    int doQKNorm) {
    fastllm::AssertInFastLLM(qgatekv.dims.size() == 3,
                             "FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache expects [bs, seq, dim].\n");
    int bs = qgatekv.dims[0];
    int seqlen = qgatekv.dims[1];
    int totalDim = qgatekv.dims[2];
    int outer = bs * seqlen;
    int expectedDim = qHeads * headDim * 2 + kHeads * headDim * 2;
    fastllm::AssertInFastLLM(totalDim == expectedDim,
                             "FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache got invalid qgatekv dim.\n");
    fastllm::AssertInFastLLM(outer == batch,
                             "FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache is decode-only.\n");
    int useInterleavedRope = positionIds.dims.size() == 2 && positionIds.dims[0] == 3;
    if (useInterleavedRope) {
        fastllm::AssertInFastLLM(sectionT + sectionH + sectionW == rotaryDim / 2,
                                 "Qwen3.5 fused decode RoPE section sizes must sum to rotary_dim / 2.\n");
    }

    float *cudaQGateKV = (float*)FastllmCudaPrepareInput(qgatekv);
    float *cudaPositionIds = (float*)FastllmCudaPrepareInput(positionIds);
    float *cudaQOutput = (float*)qOutput.cudaData;
    float *cudaGateOutput = (float*)gateOutput.cudaData;

    int totalHeads = qHeads + kHeads + kHeads;
    int gridSize = outer * totalHeads;
    int positionStride = (int)positionIds.dims.back();

    auto launch = [&](auto TPB, auto *qgatekvPtr, auto *qOutputPtr, auto *gateOutputPtr, auto *pagedTag) {
        using QT = std::remove_pointer_t<decltype(qgatekvPtr)>;
        using KVT = std::remove_pointer_t<decltype(pagedTag)>;
        FastllmQwen35QGateKVRMSNormRopeSplitAppendPagedCacheKernel<decltype(TPB)::value, QT, KVT>
            <<<gridSize, decltype(TPB)::value>>>(
                qgatekvPtr, (float*)qNormWeight.cudaData, (float*)kNormWeight.cudaData,
                cudaPositionIds, qOutputPtr, gateOutputPtr,
                (KVT*)pagedKData, (KVT*)pagedVData,
                insertIndexs, insertPositions, lastPageLens,
                outer, totalDim, qHeads, kHeads, headDim,
                bs, seqlen, positionStride, rotaryDim,
                sectionH, sectionW, useInterleavedRope,
                eps, ropeTheta, ropeScale, pageLen, batch, doQKNorm);
    };

    auto launchByPagedType = [&](auto TPB, auto *qgatekvPtr, auto *qOutputPtr, auto *gateOutputPtr) {
        if (pagedDataType == fastllm::DataType::FLOAT32) {
            launch(TPB, qgatekvPtr, qOutputPtr, gateOutputPtr, (float*)nullptr);
        } else if (pagedDataType == fastllm::DataType::FLOAT16) {
            launch(TPB, qgatekvPtr, qOutputPtr, gateOutputPtr, (half*)nullptr);
        } else if (pagedDataType == fastllm::DataType::BFLOAT16) {
            launch(TPB, qgatekvPtr, qOutputPtr, gateOutputPtr, (__nv_bfloat16*)nullptr);
        } else if (pagedDataType == fastllm::DataType::FP8_E4M3) {
            launch(TPB, qgatekvPtr, qOutputPtr, gateOutputPtr, (__nv_fp8_e4m3*)nullptr);
        } else {
            fastllm::ErrorInFastLLM("FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache: unsupported pagedDataType.\n");
        }
    };

    if (qgatekv.dataType == fastllm::DataType::FLOAT32) {
        if (headDim <= 64) {
            launchByPagedType(std::integral_constant<int, 64>{}, (float*)cudaQGateKV,
                              (float*)cudaQOutput, (float*)cudaGateOutput);
        } else if (headDim <= 128) {
            launchByPagedType(std::integral_constant<int, 128>{}, (float*)cudaQGateKV,
                              (float*)cudaQOutput, (float*)cudaGateOutput);
        } else {
            launchByPagedType(std::integral_constant<int, 512>{}, (float*)cudaQGateKV,
                              (float*)cudaQOutput, (float*)cudaGateOutput);
        }
    } else if (qgatekv.dataType == fastllm::DataType::FLOAT16) {
        if (headDim <= 64) {
            launchByPagedType(std::integral_constant<int, 64>{}, (half*)cudaQGateKV,
                              (half*)cudaQOutput, (half*)cudaGateOutput);
        } else if (headDim <= 128) {
            launchByPagedType(std::integral_constant<int, 128>{}, (half*)cudaQGateKV,
                              (half*)cudaQOutput, (half*)cudaGateOutput);
        } else {
            launchByPagedType(std::integral_constant<int, 512>{}, (half*)cudaQGateKV,
                              (half*)cudaQOutput, (half*)cudaGateOutput);
        }
    } else if (qgatekv.dataType == fastllm::DataType::BFLOAT16) {
        if (headDim <= 64) {
            launchByPagedType(std::integral_constant<int, 64>{}, (__nv_bfloat16*)cudaQGateKV,
                              (__nv_bfloat16*)cudaQOutput, (__nv_bfloat16*)cudaGateOutput);
        } else if (headDim <= 128) {
            launchByPagedType(std::integral_constant<int, 128>{}, (__nv_bfloat16*)cudaQGateKV,
                              (__nv_bfloat16*)cudaQOutput, (__nv_bfloat16*)cudaGateOutput);
        } else {
            launchByPagedType(std::integral_constant<int, 512>{}, (__nv_bfloat16*)cudaQGateKV,
                              (__nv_bfloat16*)cudaQOutput, (__nv_bfloat16*)cudaGateOutput);
        }
    } else {
        fastllm::ErrorInFastLLM("FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache: unsupported qgatekv dataType.\n");
    }

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    cudaError_t launchError = cudaGetLastError();
    if (launchError != cudaSuccess) {
        printf("FastllmCudaQwen35QGateKVRMSNormRopeSplitAppendPagedCache: kernel launch failed: %s\n",
               cudaGetErrorString(launchError));
        return false;
    }
    return true;
}

__global__ void FastllmAdvanceDecodeMetaKernel(
    int32_t *insertPositions,
    int32_t *lastPageLens,
    int batch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) {
        return;
    }
    int32_t oldLen = lastPageLens[b];
    insertPositions[b] = oldLen;
    lastPageLens[b] = oldLen + 1;
}

bool FastllmCudaAdvanceDecodeMeta(
    int32_t *insertPositions,
    int32_t *lastPageLens,
    int batch) {
    if (batch <= 0) {
        return true;
    }
    if (insertPositions == nullptr || lastPageLens == nullptr) {
        fastllm::ErrorInFastLLM("FastllmCudaAdvanceDecodeMeta: null metadata pointer.\n");
        return false;
    }
    const int threads = 128;
    int blocks = (batch + threads - 1) / threads;
    FastllmAdvanceDecodeMetaKernel<<<blocks, threads, 0, cudaStreamPerThread>>>(
        insertPositions, lastPageLens, batch);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        printf("FastllmCudaAdvanceDecodeMeta: kernel launch failed: %s\n",
               cudaGetErrorString(status));
        return false;
    }
    return true;
}

bool FastllmCudaLlama3RopeEncoding(fastllm::Data &data, const fastllm::Data &positionIds, int rotaryDim,
                                   float ropeTheta, float factor, float originalMaxPosition,
                                   float lowFreqFactor, float highFreqFactor) {
    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int bs = data.dims[0], len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    int halfDim = rotaryDim / 2;

    if (data.dataType == fastllm::DataType::FLOAT32) {
        FastllmLlama3RopeEncodingKernel <<< outer * n, halfDim >>> (
            cudaData, cudaPositionIds, len, bs, spatial, n, m,
            (int)positionIds.dims.back(), rotaryDim, ropeTheta, factor,
            originalMaxPosition, lowFreqFactor, highFreqFactor);
    } else if (data.dataType == fastllm::DataType::FLOAT16) {
        FastllmLlama3RopeEncodingKernel <<< outer * n, halfDim >>> (
            (half*)cudaData, cudaPositionIds, len, bs, spatial, n, m,
            (int)positionIds.dims.back(), rotaryDim, ropeTheta, factor,
            originalMaxPosition, lowFreqFactor, highFreqFactor);
    } else if (data.dataType == fastllm::DataType::BFLOAT16) {
        FastllmLlama3RopeEncodingKernel <<< outer * n, halfDim >>> (
            (__nv_bfloat16*)cudaData, cudaPositionIds, len, bs, spatial, n, m,
            (int)positionIds.dims.back(), rotaryDim, ropeTheta, factor,
            originalMaxPosition, lowFreqFactor, highFreqFactor);
    }
    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}

bool FastllmCudaQwen35InterleavedRope(fastllm::Data &data, const fastllm::Data &positionIds, int rotaryDim,
                                      int sectionT, int sectionH, int sectionW,
                                      float ropeTheta, float ropeScale) {
    fastllm::AssertInFastLLM(data.dims.size() == 4, "Qwen3.5 interleaved RoPE expects [batch, seq, heads, dim] input.");
    fastllm::AssertInFastLLM(data.dims[0] == 1, "Qwen3.5 interleaved RoPE currently supports batch size 1 only.");
    fastllm::AssertInFastLLM(positionIds.dims.size() == 2 && positionIds.dims[0] == 3,
                             "Qwen3.5 interleaved RoPE expects position ids with shape [3, seq].");
    fastllm::AssertInFastLLM(sectionT + sectionH + sectionW == rotaryDim / 2,
                             "Qwen3.5 interleaved RoPE section sizes must sum to rotary_dim / 2.");

    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    int halfDim = rotaryDim / 2;
    int positionStride = (int) positionIds.dims.back();

    if (data.dataType == fastllm::DataType::FLOAT32) {
        FastllmQwen35InterleavedRopeKernel <<< outer * n, halfDim >>> (
            cudaData, cudaPositionIds, len, spatial, n, m, positionStride,
            rotaryDim, sectionH, sectionW, ropeTheta, ropeScale);
    } else if (data.dataType == fastllm::DataType::FLOAT16) {
        FastllmQwen35InterleavedRopeKernel <<< outer * n, halfDim >>> (
            (half*) cudaData, cudaPositionIds, len, spatial, n, m, positionStride,
            rotaryDim, sectionH, sectionW, ropeTheta, ropeScale);
    } else if (data.dataType == fastllm::DataType::BFLOAT16) {
        FastllmQwen35InterleavedRopeKernel <<< outer * n, halfDim >>> (
            (__nv_bfloat16*) cudaData, cudaPositionIds, len, spatial, n, m, positionStride,
            rotaryDim, sectionH, sectionW, ropeTheta, ropeScale);
    }
    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}
