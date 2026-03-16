#include "fastllm.h"

bool silu_and_mul(const fastllm::Data &input, fastllm::Data &output);
bool mul_and_silu(const fastllm::Data &input, fastllm::Data &output);
bool gelu_and_mul(const fastllm::Data &input, fastllm::Data &output);
bool gelu_tanh_and_mul(const fastllm::Data &input, fastllm::Data &output);


