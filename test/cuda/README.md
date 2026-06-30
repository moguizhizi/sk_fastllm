# CUDA Kernel Tests

这个目录用于放独立 CUDA 算子正确性测试。

测试目标是不依赖完整模型加载，直接比较：

```text
CPU reference vs GPU kernel output
```

每个测试建议输出：

```text
max_abs_error
mean_abs_error
max_rel_error
PASS / FAIL
```

启用方式：

```bash
cmake -S . -B build-fastllm \
  -DUSE_CUDA=ON \
  -DENABLE_VLLM_KERNEL=ON \
  -DENABLE_CUDA_TESTS=ON

cmake --build build-fastllm -j$(nproc)
```

