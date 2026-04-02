#include "fastllm.h"

bool rms_norm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps);
bool fused_add_rms_norm(const fastllm::Data &input, const fastllm::Data &weight, fastllm::Data &output, double epsilon);