#include "fastllm.h"

bool rms_norm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps);
bool fused_add_rms_norm(fastllm::Data &input, fastllm::Data &residual, const fastllm::Data &weight, double epsilon);
