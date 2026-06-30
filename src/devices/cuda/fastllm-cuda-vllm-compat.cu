#define FASTLLM_CUDA_NO_MALLOC_CHECK_MACRO
#include "fastllm-cuda.cuh"

#include "fastllm.h"

#include <atomic>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// vLLM kernel 分支的 CUDA 兼容层。
// 这里用于承接 fastllm-cuda.cu 中的公共 CUDA 接口，避免 vLLM 算子替换时丢失链接符号。

static std::atomic<bool> fastllmCudaNcclActive(false);
static std::atomic<bool> fastllmCudaNcclForceSync(true);

void FastllmCudaSetNcclActive(bool value) {
    fastllmCudaNcclActive.store(value, std::memory_order_relaxed);
}

void FastllmCudaSetNcclForceSync(bool value) {
    fastllmCudaNcclForceSync.store(value, std::memory_order_relaxed);
}

bool FastllmCudaGetNcclForceSync() {
    return fastllmCudaNcclForceSync.load(std::memory_order_relaxed);
}

cudaError_t FastllmCudaCheckedMalloc(void **ret, size_t size, const char *file, int line) {
    (void)file;
    (void)line;
    if (fastllmCudaNcclActive.load(std::memory_order_relaxed)) {
        cudaDeviceSynchronize();
    }
    return cudaMalloc(ret, size);
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmRMSNormPartSum2Kernel(const T *input, float *sumOut, int outer, int channels, int start, int end) {
    int o = blockIdx.x;
    input = input + o * channels;
    int partChannels = end - start;

    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = THREAD_PER_BLOCK / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS > 0 ? NUM_WARPS : 1];

    unsigned int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    float sum2 = 0.0f;
    for (int i = tid; i < partChannels; i += THREAD_PER_BLOCK) {
        float x = (float)input[start + i];
        sum2 += x * x;
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    if (THREAD_PER_BLOCK > WARP_SIZE) {
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
                sumOut[o] = val;
            }
        }
    } else {
        if (tid == 0) {
            sumOut[o] = sum2;
        }
    }
}

template <>
__global__ void FastllmRMSNormPartSum2Kernel<1, half>(const half *input, float *sumOut, int outer, int channels, int start, int end) {
    int o = blockIdx.x;
    const half *base = input + o * channels;
    float sum2 = 0.0f;
    for (int i = start; i < end; i++) {
        float x = __half2float(base[i]);
        sum2 += x * x;
    }
    sumOut[o] = sum2;
}

template <>
__global__ void FastllmRMSNormPartSum2Kernel<1, float>(const float *input, float *sumOut, int outer, int channels, int start, int end) {
    int o = blockIdx.x;
    const float *base = input + o * channels;
    float sum2 = 0.0f;
    for (int i = start; i < end; i++) {
        float x = base[i];
        sum2 += x * x;
    }
    sumOut[o] = sum2;
}

template <>
__global__ void FastllmRMSNormPartSum2Kernel<1, __nv_bfloat16>(
    const __nv_bfloat16 *input, float *sumOut, int outer, int channels, int start, int end) {
    int o = blockIdx.x;
    const __nv_bfloat16 *base = input + o * channels;
    float sum2 = 0.0f;
    for (int i = start; i < end; i++) {
        float x = __bfloat162float(base[i]);
        sum2 += x * x;
    }
    sumOut[o] = sum2;
}

bool FastllmCudaRMSNormPartSum2(const fastllm::Data &input, float *sumOut, int start, int end) {
    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int partChannels = end - start;
    if (outer <= 0 || partChannels <= 0) {
        return true;
    }

    void *cudaInput = (void *)FastllmCudaPrepareInput(input);

    if (input.dataType == fastllm::DataType::FLOAT32) {
        const float *p = (const float *)cudaInput;
        if (partChannels < 64) {
            FastllmRMSNormPartSum2Kernel<1, float><<<outer, 1>>>(p, sumOut, outer, channels, start, end);
        } else if (partChannels < 512) {
            FastllmRMSNormPartSum2Kernel<64, float><<<outer, 64>>>(p, sumOut, outer, channels, start, end);
        } else if (partChannels < 4096) {
            FastllmRMSNormPartSum2Kernel<512, float><<<outer, 512>>>(p, sumOut, outer, channels, start, end);
        } else {
            FastllmRMSNormPartSum2Kernel<1024, float><<<outer, 1024>>>(p, sumOut, outer, channels, start, end);
        }
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        const half *p = (const half *)cudaInput;
        if (partChannels < 64) {
            FastllmRMSNormPartSum2Kernel<1, half><<<outer, 1>>>(p, sumOut, outer, channels, start, end);
        } else if (partChannels < 512) {
            FastllmRMSNormPartSum2Kernel<64, half><<<outer, 64>>>(p, sumOut, outer, channels, start, end);
        } else if (partChannels < 4096) {
            FastllmRMSNormPartSum2Kernel<512, half><<<outer, 512>>>(p, sumOut, outer, channels, start, end);
        } else {
            FastllmRMSNormPartSum2Kernel<1024, half><<<outer, 1024>>>(p, sumOut, outer, channels, start, end);
        }
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        const __nv_bfloat16 *p = (const __nv_bfloat16 *)cudaInput;
        if (partChannels < 64) {
            FastllmRMSNormPartSum2Kernel<1, __nv_bfloat16><<<outer, 1>>>(p, sumOut, outer, channels, start, end);
        } else if (partChannels < 512) {
            FastllmRMSNormPartSum2Kernel<64, __nv_bfloat16><<<outer, 64>>>(p, sumOut, outer, channels, start, end);
        } else if (partChannels < 4096) {
            FastllmRMSNormPartSum2Kernel<512, __nv_bfloat16><<<outer, 512>>>(p, sumOut, outer, channels, start, end);
        } else {
            FastllmRMSNormPartSum2Kernel<1024, __nv_bfloat16><<<outer, 1024>>>(p, sumOut, outer, channels, start, end);
        }
    } else {
        printf("Error: FastllmCudaRMSNormPartSum2 unsupported dtype %d.\n", (int)input.dataType);
        return false;
    }

    FastllmCudaFinishInput(input, cudaInput);
    return true;
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmRMSNormPartApplyKernel(const T *input, const float *weight, T *output, const float *sumIn,
                                              int outer, int channels, int start, int end, int partChannelsGlobal, float eps) {
    int o = blockIdx.x;
    const T *inRow = input + o * channels;
    T *outRow = output + o * channels;
    int partChannels = end - start;

    __shared__ float scale;
    if (threadIdx.x == 0) {
        scale = rsqrtf(sumIn[o] / partChannelsGlobal + eps);
    }
    __syncthreads();

    if (input != output) {
        for (int i = threadIdx.x; i < start; i += THREAD_PER_BLOCK) {
            outRow[i] = inRow[i];
        }
        for (int i = end + threadIdx.x; i < channels; i += THREAD_PER_BLOCK) {
            outRow[i] = inRow[i];
        }
    }

    float s = scale;
    for (int i = threadIdx.x; i < partChannels; i += THREAD_PER_BLOCK) {
        outRow[start + i] = (T)((float)inRow[start + i] * s * __ldg(&weight[i]));
    }
}

bool FastllmCudaRMSNormPartApply(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output,
                                 const float *sumIn, float eps, int start, int end, int partChannelsGlobal) {
    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int partChannels = end - start;
    if (outer <= 0) {
        return true;
    }

    void *cudaInput = (void *)FastllmCudaPrepareInput(input);
    void *cudaOutput = (void *)FastllmCudaPrepareInput(output);

    auto pickThreads = [](int n) -> int {
        if (n < 64) {
            return 64;
        }
        if (n < 512) {
            return 64;
        }
        if (n < 4096) {
            return 512;
        }
        return 1024;
    };
    int threads = pickThreads(partChannels);

    if (input.dataType == fastllm::DataType::FLOAT32) {
        const float *p = (const float *)cudaInput;
        float *o = (float *)cudaOutput;
        if (threads == 64) {
            FastllmRMSNormPartApplyKernel<64, float><<<outer, 64>>>(
                p, (const float *)weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        } else if (threads == 512) {
            FastllmRMSNormPartApplyKernel<512, float><<<outer, 512>>>(
                p, (const float *)weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        } else {
            FastllmRMSNormPartApplyKernel<1024, float><<<outer, 1024>>>(
                p, (const float *)weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        }
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        const half *p = (const half *)cudaInput;
        half *o = (half *)cudaOutput;
        if (threads == 64) {
            FastllmRMSNormPartApplyKernel<64, half><<<outer, 64>>>(
                p, (const float *)weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        } else if (threads == 512) {
            FastllmRMSNormPartApplyKernel<512, half><<<outer, 512>>>(
                p, (const float *)weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        } else {
            FastllmRMSNormPartApplyKernel<1024, half><<<outer, 1024>>>(
                p, (const float *)weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        }
    } else if (input.dataType == fastllm::DataType::BFLOAT16) {
        const __nv_bfloat16 *p = (const __nv_bfloat16 *)cudaInput;
        __nv_bfloat16 *o = (__nv_bfloat16 *)cudaOutput;
        if (threads == 64) {
            FastllmRMSNormPartApplyKernel<64, __nv_bfloat16><<<outer, 64>>>(
                p, (const float *)weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        } else if (threads == 512) {
            FastllmRMSNormPartApplyKernel<512, __nv_bfloat16><<<outer, 512>>>(
                p, (const float *)weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        } else {
            FastllmRMSNormPartApplyKernel<1024, __nv_bfloat16><<<outer, 1024>>>(
                p, (const float *)weight.cudaData, o, sumIn, outer, channels, start, end, partChannelsGlobal, eps);
        }
    } else {
        printf("Error: FastllmCudaRMSNormPartApply unsupported dtype %d.\n", (int)input.dataType);
        return false;
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}
