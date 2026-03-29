#include "fastllm.h"

bool silu_and_mul(const fastllm::Data &input, fastllm::Data &output);
bool cross_silu_and_mul(const fastllm::Data &input, fastllm::Data &output);
bool fatrelu(const fastllm::Data &input, fastllm::Data &output, double threshold);
bool gelu(const fastllm::Data &input, fastllm::Data &output);
bool gelu_fast(const fastllm::Data &input, fastllm::Data &output);
bool gelu_new(const fastllm::Data &input, fastllm::Data &output);
bool sigmoid(const fastllm::Data &input, fastllm::Data &output);
bool exp(const fastllm::Data &input, fastllm::Data &output);
bool silu(const fastllm::Data &input, fastllm::Data &output);