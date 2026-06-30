#include "test_utils.h"

#include <cstdio>
#include <exception>
#include <vector>

namespace {

struct AwqGemmCase {
    int numTokens = 2;
    int inChannels = 128;
    int outChannels = 64;
    int groupCnt = 32;
};

std::vector<float> AwqGemmCpuReference(const std::vector<float> &input,
                                       const std::vector<uint8_t> &qweight,
                                       const std::vector<float> &scales,
                                       const std::vector<float> &mins,
                                       const std::vector<float> *bias,
                                       const AwqGemmCase &shape) {
    using fastllm::cuda_test::Expect;

    int groups = shape.inChannels / shape.groupCnt;
    Expect(shape.numTokens > 0, "numTokens must be positive.");
    Expect(shape.inChannels > 0, "inChannels must be positive.");
    Expect(shape.outChannels > 0, "outChannels must be positive.");
    Expect(shape.groupCnt > 0, "groupCnt must be positive.");
    Expect(shape.inChannels % shape.groupCnt == 0, "inChannels must be divisible by groupCnt.");
    Expect((shape.inChannels & 1) == 0, "inChannels must be even for packed int4 weights.");
    Expect(input.size() == (size_t)shape.numTokens * shape.inChannels, "input size mismatch.");
    Expect(qweight.size() == (size_t)shape.outChannels * (shape.inChannels / 2), "qweight size mismatch.");
    Expect(scales.size() == (size_t)shape.outChannels * groups, "scales size mismatch.");
    Expect(mins.size() == (size_t)shape.outChannels * groups, "mins size mismatch.");
    if (bias != nullptr) {
        Expect(bias->size() == (size_t)shape.outChannels, "bias size mismatch.");
    }

    std::vector<float> output((size_t)shape.numTokens * shape.outChannels, 0.0f);
    for (int token = 0; token < shape.numTokens; ++token) {
        for (int oc = 0; oc < shape.outChannels; ++oc) {
            float acc = bias == nullptr ? 0.0f : (*bias)[oc];
            for (int ic = 0; ic < shape.inChannels; ++ic) {
                uint8_t packed = qweight[(size_t)oc * (shape.inChannels / 2) + ic / 2];
                int q = (ic & 1) ? (packed & 0xF) : (packed >> 4);
                int group = ic / shape.groupCnt;
                float weight = mins[(size_t)oc * groups + group] +
                               scales[(size_t)oc * groups + group] * q;
                acc += input[(size_t)token * shape.inChannels + ic] * weight;
            }
            output[(size_t)token * shape.outChannels + oc] = acc;
        }
    }
    return output;
}

bool RunCpuReferenceSmoke(bool withBias) {
    using namespace fastllm::cuda_test;

    AwqGemmCase shape;
    int groups = shape.inChannels / shape.groupCnt;
    std::vector<float> input = MakeRandomFloats((size_t)shape.numTokens * shape.inChannels, -1.0f, 1.0f, 1001);
    std::vector<uint8_t> qweight = MakeRandomInt4Weights(shape.outChannels, shape.inChannels, 1002);
    std::vector<float> scales = MakeRandomFloats((size_t)shape.outChannels * groups, 0.001f, 0.05f, 1003);
    std::vector<float> mins = MakeRandomFloats((size_t)shape.outChannels * groups, -0.4f, 0.1f, 1004);
    std::vector<float> bias = MakeRandomFloats(shape.outChannels, -0.2f, 0.2f, 1005);

    const std::vector<float> *biasPtr = withBias ? &bias : nullptr;
    std::vector<float> expected = AwqGemmCpuReference(input, qweight, scales, mins, biasPtr, shape);
    std::vector<float> actual = AwqGemmCpuReference(input, qweight, scales, mins, biasPtr, shape);

    CompareResult result = CompareVectors(expected, actual, 0.0f, 0.0f, 0.0f);
    PrintCompareResult(withBias ? "AWQ GEMM CPU reference with bias" : "AWQ GEMM CPU reference no bias",
                       result, 0.0f, 0.0f, 0.0f);
    return result.passed;
}

}  // namespace

int main() {
    try {
        bool ok = true;
        ok = RunCpuReferenceSmoke(false) && ok;
        ok = RunCpuReferenceSmoke(true) && ok;
        if (!ok) {
            return 1;
        }
        std::printf("[AWQ GEMM] CPU reference smoke PASS\n");
        return 0;
    } catch (const std::exception &e) {
        std::printf("[AWQ GEMM] FAIL: %s\n", e.what());
        return 1;
    }
}
