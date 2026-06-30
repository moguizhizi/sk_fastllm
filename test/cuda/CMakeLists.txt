# Standalone CUDA kernel correctness tests.
#
# Keep these tests independent from full model loading. Each test should compare
# a small CPU reference against the CUDA kernel output and print numerical error
# statistics.

message(STATUS "CUDA kernel tests enabled")

add_library(cuda_test_utils STATIC
    test_utils.cpp
)

target_include_directories(cuda_test_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
