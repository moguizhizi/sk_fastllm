#include "fastllm.h"

#define checkCudaErrors(message, val) showError(val, message, __FILE__, __LINE__)
void showError(cudaError_t result, char const* const message, const char* const file, int const line);

void *FastllmCudaPrepareInput(const fastllm::Data &input);
void FastllmCudaFinishInput(const fastllm::Data &input, void *data);
void FastllmCudaFree(void *ret);

int FastllmCudaGetDevice();
void FastllmCudaSetDevice(int gpu_id);
