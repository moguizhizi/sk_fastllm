#include "fastllm.h"
#include <cstdint>

bool use_vec(uint32_t num_tokens, uint32_t elementSize, uint32_t num_elements);
bool use_256b(uint32_t num_tokens);