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

template <typename T>
__global__ void FastllmVllmOpsRecurrentGatedDeltaRuleKernel(
    T *last_recurrent_state,
    const T *g_t,
    const T *k_t,
    const T *v_t,
    const T *b_t,
    const T *q_t,
    T *core_attn_out,
    int n0, int n1, int n2, int n3, int group, float qScale) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    if (batch_idx >= n0 || head_idx >= n1) {
        return;
    }

    int base_idx = batch_idx * n1 + head_idx;
    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];
    float *kv_mem = shared_mem;
    float *delta = &shared_mem[n3];

    float g_val = expf((float)g_t[base_idx]);
    for (int idx = tid; idx < n2 * n3; idx += blockDim.x) {
        int state_idx = base_idx * n2 * n3 + idx;
        last_recurrent_state[state_idx] = (T)((float)last_recurrent_state[state_idx] * g_val);
    }
    __syncthreads();

    if (tid < n3) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float k_val = (float)k_t[base_idx / group * n2 + j];
            int state_idx = base_idx * n2 * n3 + j * n3 + tid;
            sum += (float)last_recurrent_state[state_idx] * k_val;
        }
        kv_mem[tid] = sum;
    }
    __syncthreads();

    float b_val = (float)b_t[base_idx];
    if (tid < n3) {
        float v_val = (float)v_t[base_idx * n3 + tid];
        delta[tid] = (v_val - kv_mem[tid]) * b_val;
    }
    __syncthreads();

    for (int idx = tid; idx < n2 * n3; idx += blockDim.x) {
        int j = idx / n3;
        int k = idx % n3;
        float k_val = (float)k_t[base_idx / group * n2 + j];
        int state_idx = base_idx * n2 * n3 + idx;
        last_recurrent_state[state_idx] = (T)((float)last_recurrent_state[state_idx] + k_val * delta[k]);
    }
    __syncthreads();

    if (tid < n3) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float q_val = FastllmVllmOpsValueToFloat(q_t[base_idx / group * n2 + j]);
            if (qScale != 1.0f) {
                if constexpr (std::is_same_v<T, float>) {
                    q_val *= qScale;
                } else if constexpr (std::is_same_v<T, half>) {
                    half qScaleHalf = __float2half_rn(qScale);
#ifdef CUDA_NO_TENSOR_CORE
                    q_val = __half2float(__float2half(__half2float(q_t[base_idx / group * n2 + j]) * __half2float(qScaleHalf)));
#else
                    q_val = __half2float(__hmul(q_t[base_idx / group * n2 + j], qScaleHalf));
#endif
                } else {
                    q_val = FastllmVllmOpsValueToFloat(FastllmVllmOpsFloatToValue<T>(q_val * qScale));
                }
            }
            int state_idx = base_idx * n2 * n3 + j * n3 + tid;
            sum += (float)last_recurrent_state[state_idx] * q_val;
        }
        core_attn_out[base_idx * n3 + tid] = (T)sum;
    }
}

template <int TILE_V>
__global__ void FastllmVllmOpsRecurrentGatedDeltaRuleHalfTileKernel(
    half *last_recurrent_state,
    const half *g_t,
    const half *k_t,
    const half *v_t,
    const half *b_t,
    const half *q_t,
    half *core_attn_out,
    int n0, int n1, int n2, int n3, int group, float qScale) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int v_base = blockIdx.z * TILE_V;
    if (batch_idx >= n0 || head_idx >= n1 || v_base >= n3) {
        return;
    }

    int tile_v = n3 - v_base;
    if (tile_v > TILE_V) {
        tile_v = TILE_V;
    }

    int tid = threadIdx.x;
    int base_idx = batch_idx * n1 + head_idx;
    int kv_heads = n1 / group;
    int qk_base = (batch_idx * kv_heads + head_idx / group) * n2;
    int state_base = base_idx * n2 * n3;
    int out_base = base_idx * n3 + v_base;

    extern __shared__ float shared_mem[];
    float *state_tile = shared_mem;
    float *delta = state_tile + n2 * tile_v;

    float g_val = expf(__half2float(g_t[base_idx]));
    for (int idx = tid; idx < n2 * tile_v; idx += blockDim.x) {
        int j = idx / tile_v;
        int tv = idx - j * tile_v;
        int state_idx = state_base + j * n3 + v_base + tv;
        float scaled = __half2float(__float2half_rn(__half2float(last_recurrent_state[state_idx]) * g_val));
        state_tile[idx] = scaled;
    }
    __syncthreads();

    float b_val = __half2float(b_t[base_idx]);
    if (tid < tile_v) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float k_val = __half2float(k_t[qk_base + j]);
            sum += state_tile[j * tile_v + tid] * k_val;
        }
        float v_val = __half2float(v_t[out_base + tid]);
        delta[tid] = (v_val - sum) * b_val;
    }
    __syncthreads();

    for (int idx = tid; idx < n2 * tile_v; idx += blockDim.x) {
        int j = idx / tile_v;
        int tv = idx - j * tile_v;
        float k_val = __half2float(k_t[qk_base + j]);
        float updated = __half2float(__float2half_rn(state_tile[idx] + k_val * delta[tv]));
        state_tile[idx] = updated;
        int state_idx = state_base + j * n3 + v_base + tv;
        last_recurrent_state[state_idx] = __float2half_rn(updated);
    }
    __syncthreads();

    if (tid < tile_v) {
        float sum = 0.0f;
        half qScaleHalf = __float2half_rn(qScale);
        for (int j = 0; j < n2; j++) {
            float q_val;
            if (qScale != 1.0f) {
#ifdef CUDA_NO_TENSOR_CORE
                q_val = __half2float(__float2half(__half2float(q_t[qk_base + j]) * __half2float(qScaleHalf)));
#else
                q_val = __half2float(__hmul(q_t[qk_base + j], qScaleHalf));
#endif
            } else {
                q_val = __half2float(q_t[qk_base + j]);
            }
            sum += state_tile[j * tile_v + tid] * q_val;
        }
        core_attn_out[out_base + tid] = __float2half_rn(sum);
    }
}

void FastllmRecurrentGatedDeltaRule(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b,
                                    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out, float qScale) {
    int n0 = last_recurrent_state.dims[0];
    int n1 = last_recurrent_state.dims[1];
    int n2 = last_recurrent_state.dims[2];
    int n3 = last_recurrent_state.dims[3];

    float *d_last_state = (float *)last_recurrent_state.cudaData;
    float *d_g = (float *)g.cudaData;
    float *d_k = (float *)k.cudaData;
    float *d_v = (float *)v.cudaData;
    float *d_b = (float *)b.cudaData;
    float *d_q = (float *)q.cudaData;
    float *d_out = (float *)core_attn_out.cudaData;

    int group = v.dims[1] / q.dims[1];
    dim3 gridDim(n0, n1);
    int threadsPerBlock = std::min(256, std::max(n2 * n3, n3));
    size_t sharedMemSize = 2 * n3 * sizeof(float);

    if (q.dataType == fastllm::DataType::FLOAT32) {
        FastllmVllmOpsRecurrentGatedDeltaRuleKernel<float><<<gridDim, threadsPerBlock, sharedMemSize>>>(
            d_last_state, d_g, d_k, d_v, d_b, d_q, d_out, n0, n1, n2, n3, group, qScale);
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        constexpr int tileV = 8;
        size_t tileSharedMemSize = ((size_t)n2 * tileV + tileV) * sizeof(float);
        if (n2 > 0 && n3 > 0 && tileSharedMemSize <= 48 * 1024) {
            dim3 tileGrid(n0, n1, (n3 + tileV - 1) / tileV);
            int tileThreads = 256;
            FastllmVllmOpsRecurrentGatedDeltaRuleHalfTileKernel<tileV><<<tileGrid, tileThreads, tileSharedMemSize>>>(
                (half *)d_last_state, (half *)d_g, (half *)d_k, (half *)d_v, (half *)d_b, (half *)d_q, (half *)d_out,
                n0, n1, n2, n3, group, qScale);
        } else {
            FastllmVllmOpsRecurrentGatedDeltaRuleKernel<half><<<gridDim, threadsPerBlock, sharedMemSize>>>(
                (half *)d_last_state, (half *)d_g, (half *)d_k, (half *)d_v, (half *)d_b, (half *)d_q, (half *)d_out,
                n0, n1, n2, n3, group, qScale);
        }
    }

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRule.", cudaGetLastError());
}

template <typename T>
__global__ void FastllmRecurrentGatedDeltaRuleBatchPointerKernel(
    T** last_recurrent_states, // batch pointers, each [1, n1, n2, n3]
    const T* g_t,              // [batch, n1]
    const T* k_t,              // [batch, n1 / group, n2]
    const T* v_t,              // [batch, n1, n3]
    const T* b_t,              // [batch, n1]
    const T* q_t,              // [batch, n1 / group, n2]
    T* core_attn_out,          // [batch, n1, n3]
    int batch, int n1, int n2, int n3, int group, float qScale)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (batch_idx >= batch || head_idx >= n1) return;

    T* last_recurrent_state = last_recurrent_states[batch_idx];
    int base_idx = batch_idx * n1 + head_idx;
    int state_head_base = head_idx * n2 * n3;
    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* kv_mem = shared_mem;
    float* delta = &shared_mem[n3];

    float g_val = expf((float)g_t[base_idx]);

    for (int idx = tid; idx < n2 * n3; idx += blockDim.x) {
        int state_idx = state_head_base + idx;
        last_recurrent_state[state_idx] = (T)((float)last_recurrent_state[state_idx] * g_val);
    }
    __syncthreads();

    if (tid < n3) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float k_val = (float)k_t[base_idx / group * n2 + j];
            int state_idx = state_head_base + j * n3 + tid;
            sum += (float)last_recurrent_state[state_idx] * k_val;
        }
        kv_mem[tid] = sum;
    }
    __syncthreads();

    float b_val = (float)b_t[base_idx];
    if (tid < n3) {
        float v_val = (float)v_t[base_idx * n3 + tid];
        delta[tid] = (v_val - kv_mem[tid]) * b_val;
    }
    __syncthreads();

    for (int idx = tid; idx < n2 * n3; idx += blockDim.x) {
        int j = idx / n3;
        int k = idx % n3;
        float k_val = (float)k_t[base_idx / group * n2 + j];
        int state_idx = state_head_base + idx;
        last_recurrent_state[state_idx] = (T)((float)last_recurrent_state[state_idx] + k_val * delta[k]);
    }
    __syncthreads();

    if (tid < n3) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float q_val = FastllmVllmOpsValueToFloat(q_t[base_idx / group * n2 + j]);
            if (qScale != 1.0f) {
                if constexpr (std::is_same_v<T, float>) {
                    q_val *= qScale;
                } else if constexpr (std::is_same_v<T, half>) {
                    half qScaleHalf = __float2half_rn(qScale);
#ifdef CUDA_NO_TENSOR_CORE
                    q_val = __half2float(__float2half(__half2float(q_t[base_idx / group * n2 + j]) * __half2float(qScaleHalf)));
#else
                    q_val = __half2float(__hmul(q_t[base_idx / group * n2 + j], qScaleHalf));
#endif
                } else {
                    q_val = FastllmVllmOpsValueToFloat(FastllmVllmOpsFloatToValue<T>(q_val * qScale));
                }
            }
            int state_idx = state_head_base + j * n3 + tid;
            sum += (float)last_recurrent_state[state_idx] * q_val;
        }
        core_attn_out[base_idx * n3 + tid] = (T)sum;
    }
}

__global__ void FastllmLinearAttentionStateTransposeHalfKernel(
    const half *input, half *output, size_t totalHeads, int kdim, int vdim, bool kvToVk) {
    size_t stride = (size_t)kdim * vdim;
    size_t total = totalHeads * stride;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)blockDim.x * gridDim.x) {
        size_t head = idx / stride;
        size_t rem = idx - head * stride;
        if (kvToVk) {
            int k = rem / vdim;
            int v = rem - (size_t)k * vdim;
            output[head * stride + (size_t)v * kdim + k] = input[idx];
        } else {
            int v = rem / kdim;
            int k = rem - (size_t)v * kdim;
            output[head * stride + (size_t)k * vdim + v] = input[idx];
        }
    }
}

template <int TILE_V>
__global__ void FastllmRecurrentGatedDeltaRuleNormBaTransposedHalfWarpKernel(
    half* last_recurrent_state,  // physical [n0, n1, n3, n2], logical [n0, n1, n2, n3]
    const half* a_t,             // [n0, n1]
    const half* b_t,             // [n0, n1]
    const half* k_t,             // [n0, n1 / group, n2], unnormalized
    const half* v_t,             // [n0, n1, n3]
    const half* q_t,             // [n0, n1 / group, n2], unnormalized
    const float* norm_weight,    // [n2]
    const float* a_log,          // [n1]
    const float* dt_bias,        // [n1]
    half* core_attn_out,         // [n0, n1, n3]
    int n0, int n1, int n2, int n3, int group, float eps, float qScale)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int v_base = blockIdx.z * TILE_V;
    if (batch_idx >= n0 || head_idx >= n1 || v_base >= n3) return;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int base_idx = batch_idx * n1 + head_idx;
    int kv_heads = n1 / group;
    int qk_base = (batch_idx * kv_heads + head_idx / group) * n2;
    int state_base = base_idx * n2 * n3;
    int out_base = base_idx * n3;

    extern __shared__ char shared_buf[];
    float *q_norm = reinterpret_cast<float*>(shared_buf);
    float *k_norm = q_norm + n2;
    float *warp_q = k_norm + n2;
    float *warp_k = warp_q + 2;
    float *scales = warp_k + 2;
    float *ba_values = scales + 2;

    if (tid < 64) {
        int norm_warp = tid >> 5;
        const half2 *q_h2 = reinterpret_cast<const half2*>(q_t + qk_base);
        const half2 *k_h2 = reinterpret_cast<const half2*>(k_t + qk_base);
        half2 qh = q_h2[tid];
        half2 kh = k_h2[tid];
        float2 qf = __half22float2(qh);
        float2 kf = __half22float2(kh);
        float q_sum2 = qf.x * qf.x + qf.y * qf.y;
        float k_sum2 = kf.x * kf.x + kf.y * kf.y;
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_sum2 += __shfl_down_sync(0xffffffff, q_sum2, offset);
            k_sum2 += __shfl_down_sync(0xffffffff, k_sum2, offset);
        }
        if (lane_id == 0) {
            warp_q[norm_warp] = q_sum2;
            warp_k[norm_warp] = k_sum2;
        }
    }
    __syncthreads();

    if (tid < 32) {
        float q_val = tid < 2 ? warp_q[tid] : 0.0f;
        float k_val = tid < 2 ? warp_k[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_val += __shfl_down_sync(0xffffffff, q_val, offset);
            k_val += __shfl_down_sync(0xffffffff, k_val, offset);
        }
        if (tid == 0) {
            scales[0] = rsqrtf(q_val / n2 + eps);
            scales[1] = rsqrtf(k_val / n2 + eps);
        }
    }
    __syncthreads();

    if (tid < 64) {
        const half2 *q_h2 = reinterpret_cast<const half2*>(q_t + qk_base);
        const half2 *k_h2 = reinterpret_cast<const half2*>(k_t + qk_base);
        half2 qh = q_h2[tid];
        half2 kh = k_h2[tid];
        float2 qf = __half22float2(qh);
        float2 kf = __half22float2(kh);
        float w0 = __ldg(&norm_weight[tid * 2]);
        float w1 = __ldg(&norm_weight[tid * 2 + 1]);
        q_norm[tid * 2] = qf.x * scales[0] * w0;
        q_norm[tid * 2 + 1] = qf.y * scales[0] * w1;
        k_norm[tid * 2] = kf.x * scales[1] * w0;
        k_norm[tid * 2 + 1] = kf.y * scales[1] * w1;
    }

    if (tid == 0) {
        float b_raw = __half2float(b_t[base_idx]);
        float g_raw = -__expf(a_log[head_idx]) *
                      FastllmVllmOpsSoftplus(__half2float(a_t[base_idx]) + dt_bias[head_idx]);
        ba_values[0] = 1.0f / (1.0f + __expf(-b_raw));
        ba_values[1] = __expf(g_raw);
    }
    __syncthreads();

    int v_col = v_base + warp_id;
    if (warp_id >= TILE_V || v_col >= n3) {
        return;
    }

    half *state_row = last_recurrent_state + state_base + (size_t)v_col * n2;
    float g_val = ba_values[1];
    float sum_k = 0.0f;
    for (int j = lane_id; j < n2; j += 32) {
        sum_k += (__half2float(state_row[j]) * g_val) * k_norm[j];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_k += __shfl_down_sync(0xffffffff, sum_k, offset);
    }
    float delta = (FastllmVllmOpsValueToFloat(v_t[out_base + v_col]) - __shfl_sync(0xffffffff, sum_k, 0)) * ba_values[0];

    float sum_q = 0.0f;
    for (int j = lane_id; j < n2; j += 32) {
        float updated = __half2float(state_row[j]) * g_val + k_norm[j] * delta;
        state_row[j] = __float2half_rn(updated);
        sum_q += updated * (q_norm[j] * qScale);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_q += __shfl_down_sync(0xffffffff, sum_q, offset);
    }
    if (lane_id == 0) {
        core_attn_out[out_base + v_col] = __float2half_rn(sum_q);
    }
}

template <int TILE_V>
__global__ void FastllmRecurrentGatedDeltaRuleNormTransposedHalfWarpKernel(
    half* last_recurrent_state,  // physical [n0, n1, n3, n2], logical [n0, n1, n2, n3]
    const half* g_t,             // [n0, n1]
    const half* k_t,             // [n0, n1 / group, n2], unnormalized
    const half* v_t,             // [n0, n1, n3]
    const half* b_t,             // [n0, n1]
    const half* q_t,             // [n0, n1 / group, n2], unnormalized
    const float* norm_weight,    // [n2]
    half* core_attn_out,         // [n0, n1, n3]
    int n0, int n1, int n2, int n3, int group, float eps, float qScale)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int v_base = blockIdx.z * TILE_V;
    if (batch_idx >= n0 || head_idx >= n1 || v_base >= n3) return;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int base_idx = batch_idx * n1 + head_idx;
    int kv_heads = n1 / group;
    int qk_base = (batch_idx * kv_heads + head_idx / group) * n2;
    int state_base = base_idx * n2 * n3;
    int out_base = base_idx * n3;

    extern __shared__ char shared_buf[];
    float *q_norm = reinterpret_cast<float*>(shared_buf);
    float *k_norm = q_norm + n2;
    float *warp_q = k_norm + n2;
    float *warp_k = warp_q + 2;
    float *scales = warp_k + 2;

    if (tid < 64) {
        int norm_warp = tid >> 5;
        const half2 *q_h2 = reinterpret_cast<const half2*>(q_t + qk_base);
        const half2 *k_h2 = reinterpret_cast<const half2*>(k_t + qk_base);
        half2 qh = q_h2[tid];
        half2 kh = k_h2[tid];
        float2 qf = __half22float2(qh);
        float2 kf = __half22float2(kh);
        float q_sum2 = qf.x * qf.x + qf.y * qf.y;
        float k_sum2 = kf.x * kf.x + kf.y * kf.y;
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_sum2 += __shfl_down_sync(0xffffffff, q_sum2, offset);
            k_sum2 += __shfl_down_sync(0xffffffff, k_sum2, offset);
        }
        if (lane_id == 0) {
            warp_q[norm_warp] = q_sum2;
            warp_k[norm_warp] = k_sum2;
        }
    }
    __syncthreads();

    if (tid < 32) {
        float q_val = tid < 2 ? warp_q[tid] : 0.0f;
        float k_val = tid < 2 ? warp_k[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_val += __shfl_down_sync(0xffffffff, q_val, offset);
            k_val += __shfl_down_sync(0xffffffff, k_val, offset);
        }
        if (tid == 0) {
            scales[0] = rsqrtf(q_val / n2 + eps);
            scales[1] = rsqrtf(k_val / n2 + eps);
        }
    }
    __syncthreads();

    if (tid < 64) {
        const half2 *q_h2 = reinterpret_cast<const half2*>(q_t + qk_base);
        const half2 *k_h2 = reinterpret_cast<const half2*>(k_t + qk_base);
        half2 qh = q_h2[tid];
        half2 kh = k_h2[tid];
        float2 qf = __half22float2(qh);
        float2 kf = __half22float2(kh);
        float w0 = __ldg(&norm_weight[tid * 2]);
        float w1 = __ldg(&norm_weight[tid * 2 + 1]);
        q_norm[tid * 2] = qf.x * scales[0] * w0;
        q_norm[tid * 2 + 1] = qf.y * scales[0] * w1;
        k_norm[tid * 2] = kf.x * scales[1] * w0;
        k_norm[tid * 2 + 1] = kf.y * scales[1] * w1;
    }
    __syncthreads();

    int v_col = v_base + warp_id;
    if (warp_id >= TILE_V || v_col >= n3) {
        return;
    }

    half *state_row = last_recurrent_state + state_base + (size_t)v_col * n2;
    float g_val = expf(__half2float(g_t[base_idx]));
    float sum_k = 0.0f;
    for (int j = lane_id; j < n2; j += 32) {
        sum_k += (__half2float(state_row[j]) * g_val) * k_norm[j];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_k += __shfl_down_sync(0xffffffff, sum_k, offset);
    }
    float delta = (FastllmVllmOpsValueToFloat(v_t[out_base + v_col]) - __shfl_sync(0xffffffff, sum_k, 0)) *
                  __half2float(b_t[base_idx]);

    float sum_q = 0.0f;
    for (int j = lane_id; j < n2; j += 32) {
        float updated = __half2float(state_row[j]) * g_val + k_norm[j] * delta;
        state_row[j] = __float2half_rn(updated);
        sum_q += updated * (q_norm[j] * qScale);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_q += __shfl_down_sync(0xffffffff, sum_q, offset);
    }
    if (lane_id == 0) {
        core_attn_out[out_base + v_col] = __float2half_rn(sum_q);
    }
}

template <int TILE_V>
__global__ void FastllmRecurrentGatedDeltaRuleBatchPointerHalfTileKernel(
    half** last_recurrent_states, // batch pointers, each [1, n1, n2, n3]
    const half* g_t,              // [batch, n1]
    const half* k_t,              // [batch, n1 / group, n2]
    const half* v_t,              // [batch, n1, n3]
    const half* b_t,              // [batch, n1]
    const half* q_t,              // [batch, n1 / group, n2]
    half* core_attn_out,          // [batch, n1, n3]
    int batch, int n1, int n2, int n3, int group, float qScale)
{
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int v_base = blockIdx.z * TILE_V;
    if (batch_idx >= batch || head_idx >= n1 || v_base >= n3) return;

    int tile_v = n3 - v_base;
    if (tile_v > TILE_V) {
        tile_v = TILE_V;
    }

    int tid = threadIdx.x;
    int base_idx = batch_idx * n1 + head_idx;
    int kv_heads = n1 / group;
    int qk_base = (batch_idx * kv_heads + head_idx / group) * n2;
    int state_head_base = head_idx * n2 * n3;
    int out_base = base_idx * n3 + v_base;
    half *last_recurrent_state = last_recurrent_states[batch_idx];

    extern __shared__ float shared_mem[];
    float *state_tile = shared_mem;
    float *delta = state_tile + n2 * tile_v;

    float g_val = expf(__half2float(g_t[base_idx]));

    for (int idx = tid; idx < n2 * tile_v; idx += blockDim.x) {
        int j = idx / tile_v;
        int tv = idx - j * tile_v;
        int state_idx = state_head_base + j * n3 + v_base + tv;
        float scaled = __half2float(__float2half_rn(__half2float(last_recurrent_state[state_idx]) * g_val));
        state_tile[idx] = scaled;
    }
    __syncthreads();

    float b_val = __half2float(b_t[base_idx]);
    if (tid < tile_v) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float k_val = __half2float(k_t[qk_base + j]);
            sum += state_tile[j * tile_v + tid] * k_val;
        }
        float v_val = __half2float(v_t[out_base + tid]);
        delta[tid] = (v_val - sum) * b_val;
    }
    __syncthreads();

    for (int idx = tid; idx < n2 * tile_v; idx += blockDim.x) {
        int j = idx / tile_v;
        int tv = idx - j * tile_v;
        float k_val = __half2float(k_t[qk_base + j]);
        float updated = __half2float(__float2half_rn(state_tile[idx] + k_val * delta[tv]));
        state_tile[idx] = updated;
        int state_idx = state_head_base + j * n3 + v_base + tv;
        last_recurrent_state[state_idx] = __float2half_rn(updated);
    }
    __syncthreads();

    if (tid < tile_v) {
        float sum = 0.0f;
        half qScaleHalf = __float2half_rn(qScale);
        for (int j = 0; j < n2; j++) {
            float q_val;
            if (qScale != 1.0f) {
#ifdef CUDA_NO_TENSOR_CORE
                q_val = __half2float(__float2half(__half2float(q_t[qk_base + j]) * __half2float(qScaleHalf)));
#else
                q_val = __half2float(__hmul(q_t[qk_base + j], qScaleHalf));
#endif
            } else {
                q_val = __half2float(q_t[qk_base + j]);
            }
            sum += state_tile[j * tile_v + tid] * q_val;
        }
        core_attn_out[out_base + tid] = __float2half_rn(sum);
    }
}


static bool FastllmLinearAttentionStateTransposeFloat16(fastllm::Data &last_recurrent_state, bool kvToVk) {
    if (last_recurrent_state.isFake ||
        last_recurrent_state.dataDevice != fastllm::DataDevice::CUDA ||
        last_recurrent_state.dataType != fastllm::DataType::FLOAT16 ||
        last_recurrent_state.cudaData == nullptr ||
        last_recurrent_state.dims.size() != 4 ||
        last_recurrent_state.dims[0] <= 0 ||
        last_recurrent_state.dims[1] <= 0 ||
        last_recurrent_state.dims[2] <= 0 ||
        last_recurrent_state.dims[3] <= 0) {
        return false;
    }

    int n0 = last_recurrent_state.dims[0];
    int n1 = last_recurrent_state.dims[1];
    int n2 = last_recurrent_state.dims[2];
    int n3 = last_recurrent_state.dims[3];
    size_t totalHeads = (size_t)n0 * n1;
    size_t total = totalHeads * n2 * n3;
    size_t bytes = last_recurrent_state.GetBytes();
    void *oldData = last_recurrent_state.cudaData;
    bool oldBorrowed = last_recurrent_state.cudaDataBorrowed;
    void *newData = last_recurrent_state.directMemory ? FastllmCudaDirectMalloc(bytes) : FastllmCudaMalloc(bytes);
    if (newData == nullptr) {
        return false;
    }

    int threads = 256;
    int blocks = (int)std::min<size_t>((total + threads - 1) / threads, 65535);
    FastllmLinearAttentionStateTransposeHalfKernel<<<blocks, threads>>>(
        (const half*)oldData, (half*)newData, totalHeads, n2, n3, kvToVk
    );
    checkCudaErrors("Error: CUDA error in FastllmLinearAttentionStateTransposeFloat16.", cudaGetLastError());

    if (!oldBorrowed) {
        if (last_recurrent_state.directMemory) {
            FastllmCudaDirectFree(oldData);
        } else {
            FastllmCudaFree(oldData);
        }
    } else if (last_recurrent_state.isPagedKVCache &&
               last_recurrent_state.pagedKVCacheData != nullptr &&
               !last_recurrent_state.pageIndex.empty()) {
        last_recurrent_state.pagedKVCacheData->ReleasePageIndices(last_recurrent_state.pageIndex);
        last_recurrent_state.pageIndex.clear();
        last_recurrent_state.pagedKVCacheData = nullptr;
        last_recurrent_state.isPagedKVCache = false;
    }
    last_recurrent_state.cudaData = newData;
    last_recurrent_state.cudaDataBorrowed = false;
    return true;
}

bool FastllmLinearAttentionStateTransposeKVToVKFloat16(fastllm::Data &last_recurrent_state) {
    return FastllmLinearAttentionStateTransposeFloat16(last_recurrent_state, true);
}

bool FastllmLinearAttentionStateTransposeVKToKVFloat16(fastllm::Data &last_recurrent_state) {
    return FastllmLinearAttentionStateTransposeFloat16(last_recurrent_state, false);
}

bool FastllmRecurrentGatedDeltaRuleNormTransposedFloat16(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b, fastllm::Data &normWeight, fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out, float eps, float qScale) {
    if (q.dataDevice != fastllm::DataDevice::CUDA ||
        k.dataDevice != fastllm::DataDevice::CUDA ||
        v.dataDevice != fastllm::DataDevice::CUDA ||
        g.dataDevice != fastllm::DataDevice::CUDA ||
        b.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        last_recurrent_state.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (q.dataType != fastllm::DataType::FLOAT16 ||
        k.dataType != fastllm::DataType::FLOAT16 ||
        v.dataType != fastllm::DataType::FLOAT16 ||
        g.dataType != fastllm::DataType::FLOAT16 ||
        b.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        last_recurrent_state.dataType != fastllm::DataType::FLOAT16) {
        return false;
    }
    if (last_recurrent_state.dims.size() != 4 ||
        q.dims.size() != 4 || k.dims.size() != 4 || v.dims.size() != 4 ||
        g.dims.size() != 3 || b.dims.size() != 3 ||
        normWeight.dims.size() != 1) {
        return false;
    }

    int n0 = last_recurrent_state.dims[0];
    int n1 = last_recurrent_state.dims[1];
    int n2 = last_recurrent_state.dims[2];
    int n3 = last_recurrent_state.dims[3];
    if (n0 <= 0 || n1 <= 0 || n2 != 128 || n3 <= 0 ||
        normWeight.dims[0] != n2 ||
        q.dims[0] != n0 || k.dims[0] != n0 || v.dims[0] != n0 ||
        q.dims[2] != 1 || k.dims[2] != 1 || v.dims[2] != 1 ||
        q.dims[3] != n2 || k.dims[3] != n2 || v.dims[3] != n3 ||
        v.dims[1] != n1 ||
        g.dims[0] != n0 || b.dims[0] != n0 ||
        g.dims[1] != n1 || b.dims[1] != n1 ||
        g.dims[2] != 1 || b.dims[2] != 1 ||
        q.dims[1] <= 0 || q.dims[1] != k.dims[1] ||
        n1 % q.dims[1] != 0) {
        return false;
    }

    int group = n1 / q.dims[1];
    constexpr int tileV = 8;
    size_t sharedMemSize = (2 * (size_t)n2 + 8) * sizeof(float);
    if (sharedMemSize > 48 * 1024) {
        return false;
    }

    core_attn_out.dataType = last_recurrent_state.dataType;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = last_recurrent_state.dataDeviceIds;
    core_attn_out.Resize({n0, n1, 1, n3});
    core_attn_out.Allocate(false);

    dim3 grid(n0, n1, (n3 + tileV - 1) / tileV);
    FastllmRecurrentGatedDeltaRuleNormTransposedHalfWarpKernel<tileV><<<grid, tileV * 32, sharedMemSize>>>(
        (half*)last_recurrent_state.cudaData,
        (half*)g.cudaData,
        (half*)k.cudaData,
        (half*)v.cudaData,
        (half*)b.cudaData,
        (half*)q.cudaData,
        (float*)normWeight.cudaData,
        (half*)core_attn_out.cudaData,
        n0, n1, n2, n3, group, eps, qScale
    );
    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleNormTransposedFloat16.", cudaGetLastError());
    return true;
}

bool FastllmRecurrentGatedDeltaRuleNormBaTransposedFloat16(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &a, fastllm::Data &b, fastllm::Data &normWeight, fastllm::Data &aLog, fastllm::Data &dtBias, fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out, float eps, float qScale) {
    if (q.dataDevice != fastllm::DataDevice::CUDA ||
        k.dataDevice != fastllm::DataDevice::CUDA ||
        v.dataDevice != fastllm::DataDevice::CUDA ||
        a.dataDevice != fastllm::DataDevice::CUDA ||
        b.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        aLog.dataDevice != fastllm::DataDevice::CUDA ||
        dtBias.dataDevice != fastllm::DataDevice::CUDA ||
        last_recurrent_state.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    if (q.dataType != fastllm::DataType::FLOAT16 ||
        k.dataType != fastllm::DataType::FLOAT16 ||
        v.dataType != fastllm::DataType::FLOAT16 ||
        a.dataType != fastllm::DataType::FLOAT16 ||
        b.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        last_recurrent_state.dataType != fastllm::DataType::FLOAT16) {
        return false;
    }
    if (last_recurrent_state.dims.size() != 4 ||
        q.dims.size() != 4 || k.dims.size() != 4 || v.dims.size() != 4 ||
        a.dims.size() != 3 || b.dims.size() != 3 ||
        normWeight.dims.size() != 1 ||
        aLog.dims.size() != 1 ||
        dtBias.dims.size() != 1) {
        return false;
    }

    int n0 = last_recurrent_state.dims[0];
    int n1 = last_recurrent_state.dims[1];
    int n2 = last_recurrent_state.dims[2];
    int n3 = last_recurrent_state.dims[3];
    if (n0 <= 0 || n1 <= 0 || n2 != 128 || n3 <= 0 ||
        normWeight.dims[0] != n2 ||
        aLog.dims[0] != n1 ||
        dtBias.dims[0] != n1 ||
        q.dims[0] != n0 || k.dims[0] != n0 || v.dims[0] != n0 ||
        q.dims[2] != 1 || k.dims[2] != 1 || v.dims[2] != 1 ||
        q.dims[3] != n2 || k.dims[3] != n2 || v.dims[3] != n3 ||
        v.dims[1] != n1 ||
        a.dims[0] != n0 || b.dims[0] != n0 ||
        a.dims[1] != n1 || b.dims[1] != n1 ||
        a.dims[2] != 1 || b.dims[2] != 1 ||
        q.dims[1] <= 0 || q.dims[1] != k.dims[1] ||
        n1 % q.dims[1] != 0) {
        return false;
    }

    int group = n1 / q.dims[1];
    constexpr int tileV = 8;
    size_t sharedMemSize = (2 * (size_t)n2 + 10) * sizeof(float);
    if (sharedMemSize > 48 * 1024) {
        return false;
    }

    core_attn_out.dataType = last_recurrent_state.dataType;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = last_recurrent_state.dataDeviceIds;
    core_attn_out.Resize({n0, n1, 1, n3});
    core_attn_out.Allocate(false);

    dim3 grid(n0, n1, (n3 + tileV - 1) / tileV);
    FastllmRecurrentGatedDeltaRuleNormBaTransposedHalfWarpKernel<tileV><<<grid, tileV * 32, sharedMemSize>>>(
        (half*)last_recurrent_state.cudaData,
        (half*)a.cudaData,
        (half*)b.cudaData,
        (half*)k.cudaData,
        (half*)v.cudaData,
        (half*)q.cudaData,
        (float*)normWeight.cudaData,
        (float*)aLog.cudaData,
        (float*)dtBias.cudaData,
        (half*)core_attn_out.cudaData,
        n0, n1, n2, n3, group, eps, qScale
    );
    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleNormBaTransposedFloat16.", cudaGetLastError());
    return true;
}

bool FastllmRecurrentGatedDeltaRuleBatchDevicePointers(
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b,
    fastllm::Data &first_recurrent_state, void *cudaStatePointers, int batch,
    fastllm::Data &core_attn_out, float qScale) {
    if (cudaStatePointers == nullptr || batch <= 0 ||
        first_recurrent_state.dims.size() != 4 ||
        first_recurrent_state.dataDevice != fastllm::DataDevice::CUDA ||
        first_recurrent_state.cudaData == nullptr) {
        return false;
    }
    int n1 = first_recurrent_state.dims[1];
    int n2 = first_recurrent_state.dims[2];
    int n3 = first_recurrent_state.dims[3];
    if (q.dims.size() != 4 || k.dims.size() != 4 || v.dims.size() != 4 ||
        g.dims.size() != 3 || b.dims.size() != 3 ||
        q.dims[0] != batch || k.dims[0] != batch || v.dims[0] != batch ||
        g.dims[0] != batch || b.dims[0] != batch ||
        q.dims[2] != 1 || k.dims[2] != 1 || v.dims[2] != 1 ||
        g.dims[2] != 1 || b.dims[2] != 1 ||
        v.dims[1] != n1 || g.dims[1] != n1 || b.dims[1] != n1 ||
        q.dims[3] != n2 || k.dims[3] != n2 || v.dims[3] != n3 ||
        q.dims[1] <= 0 || q.dims[1] != k.dims[1] || n1 % q.dims[1] != 0) {
        return false;
    }
    int group = v.dims[1] / q.dims[1];

    core_attn_out.dataType = first_recurrent_state.dataType;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = first_recurrent_state.dataDeviceIds;
    core_attn_out.Resize({batch, n1, 1, n3});
    core_attn_out.Allocate(false);

    dim3 gridDim(batch, n1);
    int threadsPerBlock = std::min(256, std::max(n2 * n3, n3));
    size_t sharedMemSize = 2 * n3 * sizeof(float);

    if (q.dataType == fastllm::DataType::FLOAT32) {
        FastllmRecurrentGatedDeltaRuleBatchPointerKernel<float><<<gridDim, threadsPerBlock, sharedMemSize>>>(
            (float**)cudaStatePointers, (float*)g.cudaData, (float*)k.cudaData, (float*)v.cudaData,
            (float*)b.cudaData, (float*)q.cudaData, (float*)core_attn_out.cudaData,
            batch, n1, n2, n3, group, qScale
        );
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        constexpr int tileV = 8;
        size_t tileSharedMemSize = ((size_t)n2 * tileV + tileV) * sizeof(float);
        if (n2 > 0 && n3 > 0 && tileSharedMemSize <= 48 * 1024) {
            dim3 tileGrid(batch, n1, (n3 + tileV - 1) / tileV);
            int tileThreads = 256;
            FastllmRecurrentGatedDeltaRuleBatchPointerHalfTileKernel<tileV><<<tileGrid, tileThreads, tileSharedMemSize>>>(
                (half**)cudaStatePointers, (half*)g.cudaData, (half*)k.cudaData, (half*)v.cudaData,
                (half*)b.cudaData, (half*)q.cudaData, (half*)core_attn_out.cudaData,
                batch, n1, n2, n3, group, qScale
            );
        } else {
            FastllmRecurrentGatedDeltaRuleBatchPointerKernel<half><<<gridDim, threadsPerBlock, sharedMemSize>>>(
                (half**)cudaStatePointers, (half*)g.cudaData, (half*)k.cudaData, (half*)v.cudaData,
                (half*)b.cudaData, (half*)q.cudaData, (half*)core_attn_out.cudaData,
                batch, n1, n2, n3, group, qScale
            );
        }
    } else {
        return false;
    }

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleBatchDevicePointers.", cudaGetLastError());
    return true;
}

void FastllmRecurrentGatedDeltaRuleBatch(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b, std::vector<fastllm::Data*> &last_recurrent_states, fastllm::Data &core_attn_out, float qScale) {
    int batch = (int)last_recurrent_states.size();
    void **cpuPointers = new void*[batch];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i] = last_recurrent_states[i]->cudaData;
    }
    void **cudaPointers = (void**)FastllmCudaMalloc(sizeof(void*) * batch);
    cudaError_t state = cudaMemcpy(cudaPointers, cpuPointers, sizeof(void*) * batch, cudaMemcpyHostToDevice);
    delete[] cpuPointers;
    checkCudaErrors("Error: CUDA error when copy recurrent state pointers to GPU!", state);

    FastllmRecurrentGatedDeltaRuleBatchDevicePointers(
        q, k, v, g, b, *last_recurrent_states[0], cudaPointers, batch, core_attn_out, qScale);
    FastllmCudaFree(cudaPointers);
}

__global__ void FastllmRecurrentGatedDeltaRuleBatchFromConvBaHalfKernel(
    half **last_recurrent_states,
    const half *convOutput,
    const half *ba,
    const float *normWeight,
    const float *aLog,
    const float *dtBias,
    half *core_attn_out,
    int batch, int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale,
    half *statePool, const int *slotIds, int stateStride) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    if (batch_idx >= batch || head_idx >= numVHeads) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int numWarps = (blockDim.x + 31) / 32;
    int group = numVHeads / numKHeads;
    int qHead = head_idx / group;
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    int convBase = batch_idx * qkvDim;
    int qOffset = convBase + qHead * headKDim;
    int kOffset = convBase + numKHeads * headKDim + qHead * headKDim;
    int vOffset = convBase + 2 * numKHeads * headKDim + head_idx * headVDim;

    extern __shared__ float shared_mem[];
    float *kv_mem = shared_mem;
    float *delta = kv_mem + headVDim;
    float *warpQ = delta + headVDim;
    float *warpK = warpQ + numWarps;
    float *scales = warpK + numWarps;

    float qSum2 = 0.0f;
    float kSum2 = 0.0f;
    for (int j = tid; j < headKDim; j += blockDim.x) {
        float qx = __half2float(convOutput[qOffset + j]);
        float kx = __half2float(convOutput[kOffset + j]);
        qSum2 += qx * qx;
        kSum2 += kx * kx;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        qSum2 += __shfl_down_sync(0xffffffff, qSum2, offset);
        kSum2 += __shfl_down_sync(0xffffffff, kSum2, offset);
    }
    if (lane_id == 0) {
        warpQ[warp_id] = qSum2;
        warpK[warp_id] = kSum2;
    }
    __syncthreads();
    if (warp_id == 0) {
        float qVal = lane_id < numWarps ? warpQ[lane_id] : 0.0f;
        float kVal = lane_id < numWarps ? warpK[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            qVal += __shfl_down_sync(0xffffffff, qVal, offset);
            kVal += __shfl_down_sync(0xffffffff, kVal, offset);
        }
        if (lane_id == 0) {
            scales[0] = rsqrtf(qVal / headKDim + eps);
            scales[1] = rsqrtf(kVal / headKDim + eps);
        }
    }
    __syncthreads();
    float qNormScale = scales[0];
    float kNormScale = scales[1];

    const half *baRow = ba + batch_idx * 2 * numVHeads;
    float bRaw = __half2float(baRow[head_idx]);
#ifdef CUDA_NO_TENSOR_CORE
    float bVal = 1.0f / (1.0f + expf(-bRaw));
#else
    half bHalfRaw = baRow[head_idx];
    float bVal = __half2float(__hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(-bHalfRaw))));
#endif
    float aRaw = __half2float(baRow[numVHeads + head_idx]);
    float gStored = __half2float(__float2half_rn(-exp((double)aLog[head_idx]) * FastllmVllmOpsSoftplus(aRaw + dtBias[head_idx])));
    float gVal = expf(gStored);

    half *last_recurrent_state = statePool != nullptr ?
        (slotIds == nullptr ? statePool : statePool + (size_t)slotIds[batch_idx] * stateStride) :
        last_recurrent_states[batch_idx];
    int stateHeadBase = head_idx * headKDim * headVDim;

    for (int idx = tid; idx < headKDim * headVDim; idx += blockDim.x) {
        int state_idx = stateHeadBase + idx;
        last_recurrent_state[state_idx] = __float2half_rn(__half2float(last_recurrent_state[state_idx]) * gVal);
    }
    __syncthreads();

    if (tid < headVDim) {
        float sum = 0.0f;
        for (int j = 0; j < headKDim; j++) {
            float kRaw = __half2float(convOutput[kOffset + j]);
            float kNorm = __half2float(__float2half_rn(kRaw * kNormScale * normWeight[j]));
            int state_idx = stateHeadBase + j * headVDim + tid;
            sum += __half2float(last_recurrent_state[state_idx]) * kNorm;
        }
        kv_mem[tid] = sum;
    }
    __syncthreads();

    if (tid < headVDim) {
        float vVal = __half2float(convOutput[vOffset + tid]);
        delta[tid] = (vVal - kv_mem[tid]) * bVal;
    }
    __syncthreads();

    for (int idx = tid; idx < headKDim * headVDim; idx += blockDim.x) {
        int j = idx / headVDim;
        int k = idx % headVDim;
        float kRaw = __half2float(convOutput[kOffset + j]);
        float kNorm = __half2float(__float2half_rn(kRaw * kNormScale * normWeight[j]));
        int state_idx = stateHeadBase + idx;
        float updated = __half2float(last_recurrent_state[state_idx]) + kNorm * delta[k];
        last_recurrent_state[state_idx] = __float2half_rn(updated);
    }
    __syncthreads();

    if (tid < headVDim) {
        float sum = 0.0f;
        half qScaleHalf = __float2half_rn(qScale);
        for (int j = 0; j < headKDim; j++) {
            float qRaw = __half2float(convOutput[qOffset + j]);
            half qNormHalf = __float2half_rn(qRaw * qNormScale * normWeight[j]);
            float qVal;
            if (qScale != 1.0f) {
#ifdef CUDA_NO_TENSOR_CORE
                qVal = __half2float(__float2half(__half2float(qNormHalf) * __half2float(qScaleHalf)));
#else
                qVal = __half2float(__hmul(qNormHalf, qScaleHalf));
#endif
            } else {
                qVal = __half2float(qNormHalf);
            }
            int state_idx = stateHeadBase + j * headVDim + tid;
            sum += __half2float(last_recurrent_state[state_idx]) * qVal;
        }
        core_attn_out[(batch_idx * numVHeads + head_idx) * headVDim + tid] = __float2half_rn(sum);
    }
}

template <int TILE_V>
__global__ void FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedHalfWarpKernel(
    half **last_recurrent_states,
    const half *convOutput,
    const half *ba,
    const float *normWeight,
    const float *aLog,
    const float *dtBias,
    half *core_attn_out,
    int batch, int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale,
    half *statePool, const int *slotIds, int stateStride) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int v_base = blockIdx.z * TILE_V;
    if (batch_idx >= batch || head_idx >= numVHeads || v_base >= headVDim) {
        return;
    }

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int group = numVHeads / numKHeads;
    int qHead = head_idx / group;
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    int convBase = batch_idx * qkvDim;
    int qOffset = convBase + qHead * headKDim;
    int kOffset = convBase + numKHeads * headKDim + qHead * headKDim;
    int vOffset = convBase + 2 * numKHeads * headKDim + head_idx * headVDim;
    int outBase = (batch_idx * numVHeads + head_idx) * headVDim;

    extern __shared__ char shared_buf[];
    float *q_norm = reinterpret_cast<float*>(shared_buf);
    float *k_norm = q_norm + headKDim;
    float *warp_q = k_norm + headKDim;
    float *warp_k = warp_q + 2;
    float *scales = warp_k + 2;
    float *ba_values = scales + 2;

    if (tid < 64) {
        int norm_warp = tid >> 5;
        const half2 *q_h2 = reinterpret_cast<const half2*>(convOutput + qOffset);
        const half2 *k_h2 = reinterpret_cast<const half2*>(convOutput + kOffset);
        half2 qh = q_h2[tid];
        half2 kh = k_h2[tid];
        float2 qf = __half22float2(qh);
        float2 kf = __half22float2(kh);
        float q_sum2 = qf.x * qf.x + qf.y * qf.y;
        float k_sum2 = kf.x * kf.x + kf.y * kf.y;
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_sum2 += __shfl_down_sync(0xffffffff, q_sum2, offset);
            k_sum2 += __shfl_down_sync(0xffffffff, k_sum2, offset);
        }
        if (lane_id == 0) {
            warp_q[norm_warp] = q_sum2;
            warp_k[norm_warp] = k_sum2;
        }
    }
    __syncthreads();

    if (tid < 32) {
        float q_val = tid < 2 ? warp_q[tid] : 0.0f;
        float k_val = tid < 2 ? warp_k[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_val += __shfl_down_sync(0xffffffff, q_val, offset);
            k_val += __shfl_down_sync(0xffffffff, k_val, offset);
        }
        if (tid == 0) {
            scales[0] = rsqrtf(q_val / headKDim + eps);
            scales[1] = rsqrtf(k_val / headKDim + eps);
        }
    }
    __syncthreads();

    if (tid < 64) {
        const half2 *q_h2 = reinterpret_cast<const half2*>(convOutput + qOffset);
        const half2 *k_h2 = reinterpret_cast<const half2*>(convOutput + kOffset);
        half2 qh = q_h2[tid];
        half2 kh = k_h2[tid];
        float2 qf = __half22float2(qh);
        float2 kf = __half22float2(kh);
        float w0 = __ldg(&normWeight[tid * 2]);
        float w1 = __ldg(&normWeight[tid * 2 + 1]);
        q_norm[tid * 2] = qf.x * scales[0] * w0;
        q_norm[tid * 2 + 1] = qf.y * scales[0] * w1;
        k_norm[tid * 2] = kf.x * scales[1] * w0;
        k_norm[tid * 2 + 1] = kf.y * scales[1] * w1;
    }

    if (tid == 0) {
        const half *baRow = ba + (size_t)batch_idx * (numVHeads * 2);
        float bRaw = __half2float(baRow[head_idx]);
        float aRaw = __half2float(baRow[numVHeads + head_idx]);
        float gRaw = -__expf(aLog[head_idx]) * FastllmVllmOpsSoftplus(aRaw + dtBias[head_idx]);
        ba_values[0] = 1.0f / (1.0f + __expf(-bRaw));
        ba_values[1] = __expf(gRaw);
    }
    __syncthreads();

    int v_col = v_base + warp_id;
    if (warp_id >= TILE_V || v_col >= headVDim) {
        return;
    }

    half *last_recurrent_state = statePool != nullptr ?
        (slotIds == nullptr ? statePool : statePool + (size_t)slotIds[batch_idx] * stateStride) :
        last_recurrent_states[batch_idx];
    int stateHeadBase = head_idx * headKDim * headVDim;
    half *state_row = last_recurrent_state + stateHeadBase + (size_t)v_col * headKDim;
    float gVal = ba_values[1];

    float sumK = 0.0f;
    for (int j = lane_id; j < headKDim; j += 32) {
        sumK += (__half2float(state_row[j]) * gVal) * k_norm[j];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumK += __shfl_down_sync(0xffffffff, sumK, offset);
    }
    float delta = (__half2float(convOutput[vOffset + v_col]) -
                   __shfl_sync(0xffffffff, sumK, 0)) * ba_values[0];

    float sumQ = 0.0f;
    for (int j = lane_id; j < headKDim; j += 32) {
        float updated = __half2float(state_row[j]) * gVal + k_norm[j] * delta;
        state_row[j] = __float2half_rn(updated);
        sumQ += updated * (q_norm[j] * qScale);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumQ += __shfl_down_sync(0xffffffff, sumQ, offset);
    }
    if (lane_id == 0) {
        core_attn_out[outBase + v_col] = __float2half_rn(sumQ);
    }
}

template <int TILE_V>
__global__ void FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedHalfWarpKernel(
    const half *convOutput,
    const half *ba,
    const float *normWeight,
    const float *aLog,
    const float *dtBias,
    half *last_recurrent_state,
    half *core_attn_out,
    int seqLen, int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    int head_idx = blockIdx.x;
    int v_base = blockIdx.y * TILE_V;
    if (head_idx >= numVHeads || v_base >= headVDim) {
        return;
    }

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int group = numVHeads / numKHeads;
    int qHead = head_idx / group;
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    int v_col = v_base + warp_id;
    bool activeV = warp_id < TILE_V && v_col < headVDim;

    extern __shared__ char shared_buf[];
    float *q_norm = reinterpret_cast<float*>(shared_buf);
    float *k_norm = q_norm + headKDim;
    float *warp_q = k_norm + headKDim;
    float *warp_k = warp_q + 2;
    float *scales = warp_k + 2;
    float *ba_values = scales + 2;

    int stateHeadBase = head_idx * headKDim * headVDim;
    half *state_row = activeV ?
        last_recurrent_state + stateHeadBase + (size_t)v_col * headKDim :
        last_recurrent_state;

    for (int token = 0; token < seqLen; token++) {
        int convBase = token * qkvDim;
        int qOffset = convBase + qHead * headKDim;
        int kOffset = convBase + numKHeads * headKDim + qHead * headKDim;
        int vOffset = convBase + 2 * numKHeads * headKDim + head_idx * headVDim;
        int outBase = (token * numVHeads + head_idx) * headVDim;

        if (tid < 64) {
            const half2 *q_h2 = reinterpret_cast<const half2*>(convOutput + qOffset);
            const half2 *k_h2 = reinterpret_cast<const half2*>(convOutput + kOffset);
            half2 qh = q_h2[tid];
            half2 kh = k_h2[tid];
            float2 qf = __half22float2(qh);
            float2 kf = __half22float2(kh);
            float q_sum2 = qf.x * qf.x + qf.y * qf.y;
            float k_sum2 = kf.x * kf.x + kf.y * kf.y;
            for (int offset = 16; offset > 0; offset >>= 1) {
                q_sum2 += __shfl_down_sync(0xffffffff, q_sum2, offset);
                k_sum2 += __shfl_down_sync(0xffffffff, k_sum2, offset);
            }
            if (lane_id == 0) {
                int norm_warp = tid >> 5;
                warp_q[norm_warp] = q_sum2;
                warp_k[norm_warp] = k_sum2;
            }
        }
        __syncthreads();

        if (tid < 32) {
            float q_val = tid < 2 ? warp_q[tid] : 0.0f;
            float k_val = tid < 2 ? warp_k[tid] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                q_val += __shfl_down_sync(0xffffffff, q_val, offset);
                k_val += __shfl_down_sync(0xffffffff, k_val, offset);
            }
            if (tid == 0) {
                scales[0] = rsqrtf(q_val / headKDim + eps);
                scales[1] = rsqrtf(k_val / headKDim + eps);
            }
        }
        __syncthreads();

        if (tid < 64) {
            const half2 *q_h2 = reinterpret_cast<const half2*>(convOutput + qOffset);
            const half2 *k_h2 = reinterpret_cast<const half2*>(convOutput + kOffset);
            half2 qh = q_h2[tid];
            half2 kh = k_h2[tid];
            float2 qf = __half22float2(qh);
            float2 kf = __half22float2(kh);
            float w0 = __ldg(&normWeight[tid * 2]);
            float w1 = __ldg(&normWeight[tid * 2 + 1]);
            q_norm[tid * 2] = qf.x * scales[0] * w0;
            q_norm[tid * 2 + 1] = qf.y * scales[0] * w1;
            k_norm[tid * 2] = kf.x * scales[1] * w0;
            k_norm[tid * 2 + 1] = kf.y * scales[1] * w1;
        }

        if (tid == 0) {
            const half *baRow = ba + (size_t)token * (numVHeads * 2);
            float bRaw = __half2float(baRow[head_idx]);
            float aRaw = __half2float(baRow[numVHeads + head_idx]);
            float gRaw = -__expf(aLog[head_idx]) * FastllmVllmOpsSoftplus(aRaw + dtBias[head_idx]);
            ba_values[0] = 1.0f / (1.0f + __expf(-bRaw));
            ba_values[1] = __expf(gRaw);
        }
        __syncthreads();

        if (activeV) {
            float gVal = ba_values[1];
            float sumK = 0.0f;
            for (int j = lane_id; j < headKDim; j += 32) {
                sumK += (__half2float(state_row[j]) * gVal) * k_norm[j];
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                sumK += __shfl_down_sync(0xffffffff, sumK, offset);
            }
            float delta = (__half2float(convOutput[vOffset + v_col]) -
                           __shfl_sync(0xffffffff, sumK, 0)) * ba_values[0];

            float sumQ = 0.0f;
            for (int j = lane_id; j < headKDim; j += 32) {
                float updated = __half2float(state_row[j]) * gVal + k_norm[j] * delta;
                state_row[j] = __float2half_rn(updated);
                sumQ += updated * (q_norm[j] * qScale);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                sumQ += __shfl_down_sync(0xffffffff, sumQ, offset);
            }
            if (lane_id == 0) {
                core_attn_out[outBase + v_col] = __float2half_rn(sumQ);
            }
        }
        __syncthreads();
    }
}

bool FastllmRecurrentGatedDeltaRuleBatchFromConvBaDevicePointers(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &first_recurrent_state, void *cudaStatePointers, int batch,
    fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    if (cudaStatePointers == nullptr || batch <= 0 ||
        convOutput.dataDevice != fastllm::DataDevice::CUDA ||
        ba.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        aLog.dataDevice != fastllm::DataDevice::CUDA ||
        dtBias.dataDevice != fastllm::DataDevice::CUDA ||
        first_recurrent_state.dataDevice != fastllm::DataDevice::CUDA ||
        convOutput.dataType != fastllm::DataType::FLOAT16 ||
        ba.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        first_recurrent_state.dataType != fastllm::DataType::FLOAT16 ||
        convOutput.cudaData == nullptr ||
        ba.cudaData == nullptr ||
        first_recurrent_state.cudaData == nullptr) {
        return false;
    }
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    if (numKHeads <= 0 || numVHeads <= 0 || headKDim <= 0 || headVDim <= 0 ||
        numVHeads % numKHeads != 0 ||
        convOutput.dims.empty() || convOutput.dims.back() != qkvDim ||
        ba.dims.empty() || ba.dims.back() != numVHeads * 2 ||
        normWeight.dims.size() != 1 || normWeight.dims[0] != headKDim ||
        aLog.dims.size() != 1 || aLog.dims[0] != numVHeads ||
        dtBias.dims.size() != 1 || dtBias.dims[0] != numVHeads ||
        first_recurrent_state.dims.size() != 4 ||
        first_recurrent_state.dims[0] != 1 ||
        first_recurrent_state.dims[1] != numVHeads ||
        first_recurrent_state.dims[2] != headKDim ||
        first_recurrent_state.dims[3] != headVDim) {
        return false;
    }

    core_attn_out.dataType = first_recurrent_state.dataType;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = first_recurrent_state.dataDeviceIds;
    core_attn_out.Resize({batch, numVHeads, 1, headVDim});
    core_attn_out.Allocate(false);

    int threadsPerBlock = 256;
    int numWarps = (threadsPerBlock + 31) / 32;
    size_t sharedMemSize = (2 * headVDim + 2 * numWarps + 2) * sizeof(float);
    dim3 gridDim(batch, numVHeads);

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaHalfKernel<<<gridDim, threadsPerBlock, sharedMemSize>>>(
        (half**)cudaStatePointers,
        (const half*)convOutput.cudaData,
        (const half*)ba.cudaData,
        (const float*)normWeight.cudaData,
        (const float*)aLog.cudaData,
        (const float*)dtBias.cudaData,
        (half*)core_attn_out.cudaData,
        batch, numKHeads, numVHeads, headKDim, headVDim, eps, qScale,
        nullptr, nullptr, 0
    );

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleBatchFromConvBaDevicePointers.", cudaGetLastError());
    return true;
}

bool FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedDevicePointers(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &first_recurrent_state, void *cudaStatePointers, int batch,
    fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    if (cudaStatePointers == nullptr || batch <= 0 ||
        convOutput.dataDevice != fastllm::DataDevice::CUDA ||
        ba.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        aLog.dataDevice != fastllm::DataDevice::CUDA ||
        dtBias.dataDevice != fastllm::DataDevice::CUDA ||
        first_recurrent_state.dataDevice != fastllm::DataDevice::CUDA ||
        convOutput.dataType != fastllm::DataType::FLOAT16 ||
        ba.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        first_recurrent_state.dataType != fastllm::DataType::FLOAT16 ||
        !first_recurrent_state.isLinearAttentionTransposed ||
        convOutput.cudaData == nullptr ||
        ba.cudaData == nullptr ||
        first_recurrent_state.cudaData == nullptr) {
        return false;
    }
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    if (numKHeads <= 0 || numVHeads <= 0 || headKDim != 128 || headVDim <= 0 ||
        numVHeads % numKHeads != 0 ||
        convOutput.dims.empty() || convOutput.dims.back() != qkvDim ||
        ba.dims.empty() || ba.dims.back() != numVHeads * 2 ||
        normWeight.dims.size() != 1 || normWeight.dims[0] != headKDim ||
        aLog.dims.size() != 1 || aLog.dims[0] != numVHeads ||
        dtBias.dims.size() != 1 || dtBias.dims[0] != numVHeads ||
        first_recurrent_state.dims.size() != 4 ||
        first_recurrent_state.dims[0] != 1 ||
        first_recurrent_state.dims[1] != numVHeads ||
        first_recurrent_state.dims[2] != headKDim ||
        first_recurrent_state.dims[3] != headVDim) {
        return false;
    }

    core_attn_out.dataType = first_recurrent_state.dataType;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = first_recurrent_state.dataDeviceIds;
    core_attn_out.Resize({batch, numVHeads, 1, headVDim});
    core_attn_out.Allocate(false);

    constexpr int tileV = 8;
    int threadsPerBlock = tileV * 32;
    size_t sharedMemSize = (2 * (size_t)headKDim + 8) * sizeof(float);
    dim3 gridDim(batch, numVHeads, (headVDim + tileV - 1) / tileV);

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedHalfWarpKernel<tileV><<<gridDim, threadsPerBlock, sharedMemSize>>>(
        (half**)cudaStatePointers,
        (const half*)convOutput.cudaData,
        (const half*)ba.cudaData,
        (const float*)normWeight.cudaData,
        (const float*)aLog.cudaData,
        (const float*)dtBias.cudaData,
        (half*)core_attn_out.cudaData,
        batch, numKHeads, numVHeads, headKDim, headVDim, eps, qScale,
        nullptr, nullptr, 0
    );

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedDevicePointers.", cudaGetLastError());
    return true;
}

bool FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    if (convOutput.dataDevice != fastllm::DataDevice::CUDA ||
        ba.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        aLog.dataDevice != fastllm::DataDevice::CUDA ||
        dtBias.dataDevice != fastllm::DataDevice::CUDA ||
        last_recurrent_state.dataDevice != fastllm::DataDevice::CUDA ||
        convOutput.dataType != fastllm::DataType::FLOAT16 ||
        ba.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        last_recurrent_state.dataType != fastllm::DataType::FLOAT16 ||
        !last_recurrent_state.isLinearAttentionTransposed ||
        convOutput.cudaData == nullptr ||
        ba.cudaData == nullptr ||
        last_recurrent_state.cudaData == nullptr) {
        return false;
    }
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    if (numKHeads <= 0 || numVHeads <= 0 || headKDim != 128 || headVDim <= 0 ||
        numVHeads % numKHeads != 0 ||
        convOutput.dims.empty() || convOutput.dims.back() != qkvDim ||
        ba.dims.empty() || ba.dims.back() != numVHeads * 2 ||
        normWeight.dims.size() != 1 || normWeight.dims[0] != headKDim ||
        aLog.dims.size() != 1 || aLog.dims[0] != numVHeads ||
        dtBias.dims.size() != 1 || dtBias.dims[0] != numVHeads ||
        last_recurrent_state.dims.size() != 4 ||
        last_recurrent_state.dims[0] != 1 ||
        last_recurrent_state.dims[1] != numVHeads ||
        last_recurrent_state.dims[2] != headKDim ||
        last_recurrent_state.dims[3] != headVDim) {
        return false;
    }

    core_attn_out.dataType = last_recurrent_state.dataType;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = last_recurrent_state.dataDeviceIds;
    core_attn_out.Resize({1, numVHeads, 1, headVDim});
    core_attn_out.Allocate(false);

    constexpr int tileV = 8;
    int threadsPerBlock = tileV * 32;
    size_t sharedMemSize = (2 * (size_t)headKDim + 8) * sizeof(float);
    dim3 gridDim(1, numVHeads, (headVDim + tileV - 1) / tileV);

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedHalfWarpKernel<tileV><<<gridDim, threadsPerBlock, sharedMemSize>>>(
        nullptr,
        (const half*)convOutput.cudaData,
        (const half*)ba.cudaData,
        (const float*)normWeight.cudaData,
        (const float*)aLog.cudaData,
        (const float*)dtBias.cudaData,
        (half*)core_attn_out.cudaData,
        1, numKHeads, numVHeads, headKDim, headVDim, eps, qScale,
        (half*)last_recurrent_state.cudaData, nullptr, 0
    );

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16.", cudaGetLastError());
    return true;
}

bool FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    if (convOutput.dataDevice != fastllm::DataDevice::CUDA ||
        ba.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        aLog.dataDevice != fastllm::DataDevice::CUDA ||
        dtBias.dataDevice != fastllm::DataDevice::CUDA ||
        last_recurrent_state.dataDevice != fastllm::DataDevice::CUDA ||
        convOutput.dataType != fastllm::DataType::FLOAT16 ||
        ba.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        last_recurrent_state.dataType != fastllm::DataType::FLOAT16 ||
        !last_recurrent_state.isLinearAttentionTransposed ||
        convOutput.cudaData == nullptr ||
        ba.cudaData == nullptr ||
        last_recurrent_state.cudaData == nullptr) {
        return false;
    }
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    if (numKHeads <= 0 || numVHeads <= 0 || headKDim != 128 || headVDim <= 0 ||
        numVHeads % numKHeads != 0 ||
        convOutput.dims.size() != 3 || convOutput.dims[0] != 1 ||
        convOutput.dims.back() != qkvDim ||
        ba.dims.size() != 3 || ba.dims[0] != 1 ||
        ba.dims.back() != numVHeads * 2 ||
        convOutput.dims[1] != ba.dims[1] ||
        convOutput.dims[1] <= 1 || convOutput.dims[1] > 4 ||
        normWeight.dims.size() != 1 || normWeight.dims[0] != headKDim ||
        aLog.dims.size() != 1 || aLog.dims[0] != numVHeads ||
        dtBias.dims.size() != 1 || dtBias.dims[0] != numVHeads ||
        last_recurrent_state.dims.size() != 4 ||
        last_recurrent_state.dims[0] != 1 ||
        last_recurrent_state.dims[1] != numVHeads ||
        last_recurrent_state.dims[2] != headKDim ||
        last_recurrent_state.dims[3] != headVDim) {
        return false;
    }

    int seqLen = convOutput.dims[1];
    core_attn_out.dataType = last_recurrent_state.dataType;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = last_recurrent_state.dataDeviceIds;
    core_attn_out.Resize({1, seqLen, numVHeads, headVDim});
    core_attn_out.Allocate(false);

    constexpr int tileV = 8;
    int threadsPerBlock = tileV * 32;
    size_t sharedMemSize = (2 * (size_t)headKDim + 8) * sizeof(float);
    dim3 gridDim(numVHeads, (headVDim + tileV - 1) / tileV);

    FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedHalfWarpKernel<tileV><<<gridDim, threadsPerBlock, sharedMemSize>>>(
        (const half*)convOutput.cudaData,
        (const half*)ba.cudaData,
        (const float*)normWeight.cudaData,
        (const float*)aLog.cudaData,
        (const float*)dtBias.cudaData,
        (half*)last_recurrent_state.cudaData,
        (half*)core_attn_out.cudaData,
        seqLen, numKHeads, numVHeads, headKDim, headVDim, eps, qScale
    );

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16.", cudaGetLastError());
    return true;
}

bool FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedSlots(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    void *cudaStatePool, void *cudaSlotIds, int batch,
    fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    if (cudaStatePool == nullptr || cudaSlotIds == nullptr || batch <= 0 ||
        convOutput.dataDevice != fastllm::DataDevice::CUDA ||
        ba.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        aLog.dataDevice != fastllm::DataDevice::CUDA ||
        dtBias.dataDevice != fastllm::DataDevice::CUDA ||
        convOutput.dataType != fastllm::DataType::FLOAT16 ||
        ba.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        aLog.dataType != fastllm::DataType::FLOAT32 ||
        dtBias.dataType != fastllm::DataType::FLOAT32 ||
        convOutput.cudaData == nullptr ||
        ba.cudaData == nullptr) {
        return false;
    }
    int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
    if (numKHeads <= 0 || numVHeads <= 0 || headKDim != 128 || headVDim <= 0 ||
        numVHeads % numKHeads != 0 ||
        convOutput.dims.empty() || convOutput.dims.back() != qkvDim ||
        ba.dims.empty() || ba.dims.back() != numVHeads * 2 ||
        normWeight.dims.size() != 1 || normWeight.dims[0] != headKDim ||
        aLog.dims.size() != 1 || aLog.dims[0] != numVHeads ||
        dtBias.dims.size() != 1 || dtBias.dims[0] != numVHeads) {
        return false;
    }

    core_attn_out.dataType = fastllm::DataType::FLOAT16;
    core_attn_out.dataDevice = fastllm::DataDevice::CUDA;
    core_attn_out.dataDeviceIds = convOutput.dataDeviceIds;
    core_attn_out.Resize({batch, numVHeads, 1, headVDim});
    core_attn_out.Allocate(false);

    constexpr int tileV = 8;
    int threadsPerBlock = tileV * 32;
    size_t sharedMemSize = (2 * (size_t)headKDim + 8) * sizeof(float);
    dim3 gridDim(batch, numVHeads, (headVDim + tileV - 1) / tileV);
    int stateStride = numVHeads * headVDim * headKDim;

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedHalfWarpKernel<tileV><<<gridDim, threadsPerBlock, sharedMemSize>>>(
        nullptr,
        (const half*)convOutput.cudaData,
        (const half*)ba.cudaData,
        (const float*)normWeight.cudaData,
        (const float*)aLog.cudaData,
        (const float*)dtBias.cudaData,
        (half*)core_attn_out.cudaData,
        batch, numKHeads, numVHeads, headKDim, headVDim, eps, qScale,
        (half*)cudaStatePool, (const int*)cudaSlotIds, stateStride
    );

    checkCudaErrors("Error: CUDA error in FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedSlots.", cudaGetLastError());
    return true;
}

void FastllmRecurrentGatedDeltaRuleBatchFromConvBa(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    std::vector<fastllm::Data*> &last_recurrent_states, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    int batch = (int)last_recurrent_states.size();
    void **cpuPointers = new void*[batch];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i] = last_recurrent_states[i]->cudaData;
    }
    void **cudaPointers = (void**)FastllmCudaMalloc(sizeof(void*) * batch);
    cudaError_t state = cudaMemcpy(cudaPointers, cpuPointers, sizeof(void*) * batch, cudaMemcpyHostToDevice);
    delete[] cpuPointers;
    checkCudaErrors("Error: CUDA error when copy recurrent state pointers to GPU!", state);

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaDevicePointers(
        convOutput, ba, normWeight, aLog, dtBias, *last_recurrent_states[0], cudaPointers, batch,
        core_attn_out, numKHeads, numVHeads, headKDim, headVDim, eps, qScale);
    FastllmCudaFree(cudaPointers);
}

void FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposed(
    fastllm::Data &convOutput, fastllm::Data &ba, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias,
    std::vector<fastllm::Data*> &last_recurrent_states, fastllm::Data &core_attn_out,
    int numKHeads, int numVHeads, int headKDim, int headVDim,
    float eps, float qScale) {
    int batch = (int)last_recurrent_states.size();
    void **cpuPointers = new void*[batch];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i] = last_recurrent_states[i]->cudaData;
    }
    void **cudaPointers = (void**)FastllmCudaMalloc(sizeof(void*) * batch);
    cudaError_t state = cudaMemcpy(cudaPointers, cpuPointers, sizeof(void*) * batch, cudaMemcpyHostToDevice);
    delete[] cpuPointers;
    checkCudaErrors("Error: CUDA error when copy transposed recurrent state pointers to GPU!", state);

    FastllmRecurrentGatedDeltaRuleBatchFromConvBaTransposedDevicePointers(
        convOutput, ba, normWeight, aLog, dtBias, *last_recurrent_states[0], cudaPointers, batch,
        core_attn_out, numKHeads, numVHeads, headKDim, headVDim, eps, qScale);
    FastllmCudaFree(cudaPointers);
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
