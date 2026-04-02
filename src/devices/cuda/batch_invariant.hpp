#pragma once
#include <cstdlib>
#include <string>
#include <cctype>

namespace FastLLM {

// FastLLM_is_batch_invariant(); returns true
// if env FASTLLM_BATCH_INVARIANT=1
inline bool FastLLM_is_batch_invariant() {
  static bool cached = []() {
    std::string env_key = "FASTLLM_BATCH_INVARIANT";
    const char* val = std::getenv(env_key.c_str());
    return (val && std::atoi(val) != 0) ? 1 : 0;
  }();
  return cached;
}

}  // namespace FastLLM
