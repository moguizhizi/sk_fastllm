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
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

cmake -S . -B build-fastllm \
  -DUSE_CUDA=ON \
  -DENABLE_VLLM_KERNEL=ON \
  -DENABLE_CUDA_TESTS=ON \
  -DCUDA_ARCH=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

cmake --build build-fastllm --target test_awq_gemm_compare -j$(nproc)

./build-fastllm/test/cuda/test_awq_gemm_compare
```

5090 环境下 AWQ GEMM 测试期望输出类似：

```text
[AWQ GEMM GPU compare no bias] max_abs=0.000894308/0.02 mean_abs=0.000172735/0.002 max_rel=0.000456023/0.05 max_abs_index=1 PASS
[AWQ GEMM GPU compare with bias] max_abs=0.000922203/0.02 mean_abs=0.00017659/0.002 max_rel=0.000439798/0.05 max_abs_index=85 PASS
[AWQ GEMM] GPU compare PASS
```

如果找不到测试可执行文件：

```bash
find build-fastllm -name test_awq_gemm_compare -type f
```

如果 CMake 找不到 `nvcc`，先确认：

```bash
which nvcc
nvcc --version
ls /usr/local/cuda/bin/nvcc
```
