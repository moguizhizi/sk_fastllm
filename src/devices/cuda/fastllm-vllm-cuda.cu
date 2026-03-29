#include "fastllm-cuda.cuh"
#include "activation_kernels.cuh"
#include "layernorm_kernels.cuh"

bool FastllmCudaSwiglu(const fastllm::Data &input, fastllm::Data &output)
{
    return silu_and_mul(input, output);
}

bool FastllmCudaCrossSwiglu(const fastllm::Data &input, fastllm::Data &output)
{
    return cross_silu_and_mul(input, output);
}

bool FastllmCudaExp(const fastllm::Data &input, fastllm::Data &output) {
    return exp(input, output);
}

bool FastllmCudaRelu(const fastllm::Data &input, fastllm::Data &output) {
    return fatrelu(input, output, 0);
}

bool FastllmCudaGelu(const fastllm::Data &input, fastllm::Data &output) {
    return gelu(input, output);
}

bool FastllmCudaGeluNew(const fastllm::Data &input, fastllm::Data &output) {
    return gelu_new(input, output);
}

bool FastllmCudaSilu(const fastllm::Data &input, fastllm::Data &output) {
    return silu(input, output);
}

bool FastllmCudaSigmoid(const fastllm::Data &input, fastllm::Data &output) {
    return sigmoid(input, output);
}

bool FastllmCudaRMSNorm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps) {
    return rms_norm(input, weight, output, eps);
}