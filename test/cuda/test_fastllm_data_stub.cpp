#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

namespace fastllm {

namespace {

uint16_t FloatToHalfBits(float value) {
    half h = __float2half(value);
    uint16_t bits;
    static_assert(sizeof(bits) == sizeof(h), "half size mismatch");
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

uint64_t PackedBytes(DataType type, uint64_t count) {
    if (type == FLOAT32 || type == INT32 || type == INT32PARAM) {
        return count * 4;
    }
    if (type == FLOAT16 || type == BFLOAT16 || type == INT16) {
        return count * 2;
    }
    if (type == INT4 || type == INT4_GROUP || type == INT4_NOZERO) {
        return (count + 1) / 2;
    }
    return count;
}

} // namespace

FastllmEnv::FastllmEnv() = default;

const FastllmEnv &GetFastllmEnv() {
    static FastllmEnv env;
    return env;
}

Data::Data(DataType type) {
    dataType = type;
    UpdateUnitSize();
}

Data::Data(DataType type, const std::vector<int> &dims) {
    dataType = type;
    Resize(dims);
}

Data::Data(DataType type, int ggmlType, const std::vector<int> &dims) {
    dataType = type;
    this->ggmlType = ggmlType;
    Resize(dims);
}

Data::Data(DataType type, const std::vector<int> &dims, DataDevice device, void *ptr) : Data(type, dims) {
    isFake = true;
    dataDevice = device;
    expansionSize = Count(0);
    expansionBytes = GetBytes();
    if (device == CPU) {
        cpuData = (uint8_t *)ptr;
    } else if (device == CUDA) {
        cudaData = ptr;
        dataDeviceIds = {0};
    }
}

Data::Data(DataType type, const std::vector<int> &dims, const std::vector<float> &data) : Data(type, dims) {
    Allocate();
    size_t count = std::min((size_t)Count(0), data.size());
    if (type == FLOAT32) {
        std::memcpy(cpuData, data.data(), count * sizeof(float));
    } else if (type == FLOAT16) {
        uint16_t *dst = (uint16_t *)cpuData;
        for (size_t i = 0; i < count; i++) {
            dst[i] = FloatToHalfBits(data[i]);
        }
    }
}

Data::~Data() {
    FreeSpace();
    for (void *ptr : extraCudaData) {
        if (ptr != nullptr) {
            FastllmCudaFree(ptr);
        }
    }
    extraCudaData.clear();
}

Data::Data(const Data &ori) {
    dataType = ori.dataType;
    Resize(ori.dims);
    group = ori.group;
    groupCnt = ori.groupCnt;
    scales = ori.scales;
    mins = ori.mins;
    Allocate(false);
    if (ori.dataDevice == CPU && ori.cpuData != nullptr) {
        std::memcpy(cpuData, ori.cpuData, GetBytes());
    }
}

void Data::UpdateUnitSize() {
    if (dataType == FLOAT32 || dataType == INT32 || dataType == INT32PARAM) {
        unitSize = 4;
        unitSizeDiv = 1;
    } else if (dataType == FLOAT16 || dataType == BFLOAT16 || dataType == INT16) {
        unitSize = 2;
        unitSizeDiv = 1;
    } else if (dataType == INT4 || dataType == INT4_GROUP || dataType == INT4_NOZERO) {
        unitSize = 1;
        unitSizeDiv = 2;
    } else {
        unitSize = 1;
        unitSizeDiv = 1;
    }
    expansionBytes = PackedBytes(dataType, expansionSize);
}

void Data::Resize(const std::vector<int> &newDims) {
    dims = newDims;
    strides.assign(dims.size(), 1);
    if (!dims.empty()) {
        strides[dims.size() - 1] = 1;
        for (int i = (int)dims.size() - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * (uint64_t)dims[i + 1];
        }
    }
    expansionSize = Count(0);
    UpdateUnitSize();
}

void Data::Reshape(const std::vector<int> &newDims) {
    Resize(newDims);
}

uint64_t Data::Count(int i) const {
    if (dims.empty()) {
        return 0;
    }
    if (i >= (int)dims.size()) {
        return 1;
    }
    return (uint64_t)dims[i] * strides[i];
}

uint64_t Data::GetBytes() const {
    return PackedBytes(dataType, dims.empty() ? 0 : (uint64_t)dims[0] * strides[0]);
}

void Data::MallocSpace(uint64_t size, bool zero) {
    expansionSize = size;
    expansionBytes = PackedBytes(dataType, size);
    if (dataDevice == CPU) {
        cpuData = new uint8_t[expansionBytes];
        if (zero) {
            std::memset(cpuData, 0, expansionBytes);
        }
    } else if (dataDevice == CUDA) {
        cudaData = FastllmCudaMalloc(expansionBytes);
        cudaDataBorrowed = false;
        if (zero) {
            FastllmCudaMemset0(cudaData, expansionBytes);
        }
    }
}

void Data::FreeSpace() {
    if (isFake) {
        return;
    }
    if (cpuData != nullptr) {
        delete[] cpuData;
        cpuData = nullptr;
    }
    if (cudaData != nullptr) {
        if (!cudaDataBorrowed) {
            FastllmCudaFree(cudaData);
        }
        cudaData = nullptr;
        cudaDataBorrowed = false;
    }
    expansionSize = 0;
    expansionBytes = 0;
}

void Data::Allocate() {
    Allocate(true);
}

void Data::Allocate(bool zero) {
    uint64_t count = Count(0);
    if (!isFake && (cpuData == nullptr && cudaData == nullptr || count > expansionSize)) {
        FreeSpace();
        MallocSpace(count, zero);
    }
}

void Data::Allocate(float) {
    Allocate();
}

void Data::ToDevice(DataDevice device, bool copyData) {
    ToDevice(device, {0}, copyData);
}

void Data::ToDevice(DataDevice device, const std::vector<int> &deviceIds, bool copyData) {
    if (dataDevice == device) {
        dataDeviceIds = device == CUDA ? deviceIds : std::vector<int>();
        return;
    }

    uint64_t bytes = GetBytes();
    if (device == CUDA) {
        void *oldCpu = cpuData;
        cudaData = FastllmCudaMalloc(bytes);
        cudaDataBorrowed = false;
        if (copyData && oldCpu != nullptr && bytes > 0) {
            FastllmCudaCopyFromHostToDevice(cudaData, oldCpu, bytes);
        }
        delete[] cpuData;
        cpuData = nullptr;
        dataDevice = CUDA;
        dataDeviceIds = deviceIds;
    } else {
        uint8_t *newCpu = new uint8_t[bytes];
        if (copyData && cudaData != nullptr && bytes > 0) {
            FastllmCudaCopyFromDeviceToHost(newCpu, cudaData, bytes);
        }
        if (cudaData != nullptr && !cudaDataBorrowed) {
            FastllmCudaFree(cudaData);
        }
        cudaData = nullptr;
        cudaDataBorrowed = false;
        cpuData = newCpu;
        dataDevice = CPU;
        dataDeviceIds.clear();
    }
    expansionSize = Count(0);
    expansionBytes = bytes;
}

void Data::ToDevice(void *, bool copyData) {
    ToDevice(CUDA, {0}, copyData);
}

void Data::Expansion(const std::vector<int> &) {}
void Data::CopyFrom(const Data &) {}
void Data::FakeFrom(const Data &, size_t) {}
void Data::PrintShape() const {}
std::vector<int> Data::Shape() const { return dims; }
void Data::Print(const std::string &) const {}
void Data::CalcWeightSum() {}
void Data::ToCudaTemporary(const std::vector<int> &, bool, void *) {}
void Data::FreeCudaTemporary(const std::vector<int> &, bool) {}
void Data::Repack() {}
void Data::SetKVCache() {}
uint64_t Data::GetFastllmFormateBytes() { return 0; }
void Data::ExportFastllmFormat(uint8_t *) {}
void Data::CreateFromFastllmFormat(uint8_t *, uint64_t) {}
DataType Data::GetDataType() { return dataType; }
DataType Data::GetLinearActDataType(int) { return dataType; }
bool Data::IsTensorParallel() const { return false; }
bool Data::IsTensorParallelReplicated() const { return false; }
bool Data::IsTensorParallelSharded() const { return false; }
void Data::ClearTensorParallelLayout() {}
void Data::ResetMultiDeviceState() {}

std::string GetDataTypeName(DataType type) {
    return std::to_string((int)type);
}

size_t GetDataBytes(DataType type, size_t rows, size_t columns) {
    return PackedBytes(type, rows * columns);
}

} // namespace fastllm
