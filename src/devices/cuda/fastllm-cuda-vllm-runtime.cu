#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <execinfo.h>
#endif

#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "utils/utils.h"

// vLLM kernel 分支下拆出的 CUDA 公共运行时支撑。
// ENABLE_VLLM_KERNEL=OFF 时不编译本文件，仍走原始 fastllm-cuda.cu 流程。
void showError(cudaError_t result, char const *const message, const char *const file, int const line) {
    if (cudaSuccess != result) {
        printf(
            "%s\n  CUDA error = %d, %s at %s:%d\n  '%s'\n", message, result, cudaGetErrorName(result), file, line, cudaGetErrorString(result));
    }
}

static std::map<int, cublasHandle_t> s_fastllmCublasHandleMap;
cublasHandle_t getFastllmCublasHandle() {
    int id = -1;
    cudaGetDevice(&id);
    auto it = s_fastllmCublasHandleMap.find(id);
    if (it != s_fastllmCublasHandleMap.end()) {
        return it->second;
    }
    cublasHandle_t handler = nullptr;
    auto stat = cublasCreate(&handler);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("Error: CUBLAS initialization failed. state %d.\n", stat);
        exit(0);
    } else {
        s_fastllmCublasHandleMap[id] = handler;
    }

    return handler;
}

std::vector<long long> FastllmCudaGetFreeSizes() {
    int deviceCount;
    auto error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return {};
    }
    std::vector<long long> ret;

    // 遍历所有设备
    int id = -1;
    cudaGetDevice(&id);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);
        if (error == cudaSuccess) {
            // printf("Device %d: \"%s\"\n", i, prop.name);
            // printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
            // printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);

            // 获取当前设备的显存使用情况
            cudaSetDevice(i);
            size_t free = 0, total = 0;
            cudaMemGetInfo(&free, &total);
            ret.push_back(free);
            // printf("  Free memory: %zu bytes\n", free);
            // printf("  Remaining memory: %zu bytes\n", total - free);
        } else {
            printf("cudaGetDeviceProperties returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        }
    }
    cudaSetDevice(id);
    return ret;
}

std::vector<long long> FastllmCudaGetTotalSizes() {
    int deviceCount;
    auto error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return {};
    }
    std::vector<long long> ret;

    int id = -1;
    cudaGetDevice(&id);

    for (int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        size_t free = 0, total = 0;
        cudaMemGetInfo(&free, &total);
        ret.push_back(total);
    }
    cudaSetDevice(id);
    return ret;
}

__global__ void GetCudaInfoKernel(int *infos) {
#if defined(__CUDA_ARCH__)
    infos[0] = __CUDA_ARCH__;
#else
    infos[0] = 0; // cuda arch
#endif
}

CudaInfos::CudaInfos() {
    int infoLen = 10;
    int *infos;
    cudaMalloc(&infos, infoLen * sizeof(int));
    GetCudaInfoKernel<<<1, 1>>>(infos);
    int *infosInCpu = new int[infoLen];
    cudaMemcpy(infosInCpu, infos, infoLen * sizeof(int), cudaMemcpyDeviceToHost);

    cudaArch = infosInCpu[0];
    hasTensorCore = cudaArch >= 700;

    cudaFree(infos);
    delete[] infosInCpu;

    printf("CUDA_ARCH: %d\n", cudaArch);
    printf("USE_TENSOR_CORE: %d\n", hasTensorCore);
}

CudaInfos *cudaInfos = nullptr;

CudaInfos *getCudaInfos() {
    if (cudaInfos == nullptr) {
        cudaInfos = new CudaInfos();
    }
    return cudaInfos;
}

void DeviceSync() {
    if (fastllm::GetFastllmEnv().cudaSync) {
        cudaDeviceSynchronize();
    }
}

void ForceDeviceSync() {
    cudaDeviceSynchronize();
}

void *FastllmCudaStreamCreate(bool nonBlocking) {
    cudaStream_t stream;
    unsigned int flags = nonBlocking ? cudaStreamNonBlocking : cudaStreamDefault;
    cudaError_t state = cudaStreamCreateWithFlags(&stream, flags);
    checkCudaErrors("Error: CUDA error when creating stream!", state);
    return (void *)stream;
}

void FastllmCudaStreamDestroy(void *stream) {
    cudaError_t state = cudaStreamDestroy((cudaStream_t)stream);
    checkCudaErrors("Error: CUDA error when destroying stream!", state);
}

void FastllmCudaStreamSynchronize(void *stream) {
    cudaError_t state = cudaStreamSynchronize((cudaStream_t)stream);
    checkCudaErrors("Error: CUDA error when synchronizing stream!", state);
}

void *FastllmCudaEventCreate() {
    cudaEvent_t event;
    cudaError_t state = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    checkCudaErrors("Error: CUDA error when creating event!", state);
    return (void *)event;
}

void FastllmCudaEventDestroy(void *event) {
    cudaError_t state = cudaEventDestroy((cudaEvent_t)event);
    checkCudaErrors("Error: CUDA error when destroying event!", state);
}

void FastllmCudaEventRecord(void *event, void *stream) {
    cudaError_t state = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
    checkCudaErrors("Error: CUDA error when recording event!", state);
}

void FastllmCudaEventSynchronize(void *event) {
    cudaError_t state = cudaEventSynchronize((cudaEvent_t)event);
    checkCudaErrors("Error: CUDA error when synchronizing event!", state);
}

void FastllmCudaStreamWaitEvent(void *stream, void *event) {
    cudaError_t state = cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, 0);
    checkCudaErrors("Error: CUDA error when stream waiting event!", state);
}

static thread_local std::string fastllmCudaGraphLastError;

static bool FastllmCudaGraphSetError(const char *stage, cudaError_t err) {
    if (err == cudaSuccess) {
        fastllmCudaGraphLastError.clear();
        return true;
    }
    fastllmCudaGraphLastError = std::string(stage) + ": " + cudaGetErrorString(err);
    return false;
}

bool FastllmCudaGraphBeginCapture() {
    cudaError_t state = cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal);
    return FastllmCudaGraphSetError("cudaStreamBeginCapture", state);
}

bool FastllmCudaGraphEndCapture(void **graph) {
    cudaGraph_t cudaGraph = nullptr;
    cudaError_t state = cudaStreamEndCapture(cudaStreamPerThread, &cudaGraph);
    if (graph != nullptr) {
        *graph = (void *)cudaGraph;
    }
    return FastllmCudaGraphSetError("cudaStreamEndCapture", state);
}

bool FastllmCudaGraphInstantiate(void *graph, void **exec) {
    cudaGraphExec_t cudaExec = nullptr;
    cudaError_t state = cudaGraphInstantiate(&cudaExec, (cudaGraph_t)graph, nullptr, nullptr, 0);
    if (exec != nullptr) {
        *exec = (void *)cudaExec;
    }
    return FastllmCudaGraphSetError("cudaGraphInstantiate", state);
}

bool FastllmCudaGraphLaunch(void *exec) {
    cudaError_t state = cudaGraphLaunch((cudaGraphExec_t)exec, cudaStreamPerThread);
    return FastllmCudaGraphSetError("cudaGraphLaunch", state);
}

void FastllmCudaGraphDestroy(void *graph) {
    if (graph != nullptr) {
        cudaGraphDestroy((cudaGraph_t)graph);
    }
}

void FastllmCudaGraphExecDestroy(void *exec) {
    if (exec != nullptr) {
        cudaGraphExecDestroy((cudaGraphExec_t)exec);
    }
}

const char *FastllmCudaGraphLastError() {
    return fastllmCudaGraphLastError.c_str();
}

double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1);
    return double(duration.count()) * std::chrono::nanoseconds::period::num / std::chrono::nanoseconds::period::den;
};

void *FastllmCudaPrepareInput(const fastllm::Data &input) {
    void *ret;
    if (input.dataDevice == fastllm::DataDevice::CUDA) {
        ret = (void *)input.cudaData;
    } else {
        ret = (void *)(input.expansionBytes);
        auto state = cudaMemcpy(ret, input.cpuData, input.expansionBytes, cudaMemcpyHostToDevice);
        if (cudaSuccess != state) {
            checkCudaErrors("Error: CUDA error when copy from memory to GPU!", state);
            return nullptr;
        }
    }
    return ret;
}

void FastllmCudaFinishInput(const fastllm::Data &input, void *data) {
    if (input.dataDevice != fastllm::DataDevice::CUDA) {
        FastllmCudaFree(data);
    }
}

void *FastllmCudaPrepareOutput(fastllm::Data &output) {
    void *ret;
    if (output.dataDevice == fastllm::DataDevice::CUDA) {
        ret = (float *)output.cudaData;
    } else {
        ret = (float *)FastllmCudaMalloc(output.expansionBytes);
    }
    return ret;
}

void FastllmCudaFinishOutput(fastllm::Data &output, void *data) {
    if (output.dataDevice != fastllm::DataDevice::CUDA) {
        auto state = cudaMemcpy(output.cpuData, data, output.expansionBytes, cudaMemcpyDeviceToHost);
        checkCudaErrors("Error: CUDA error when copy from GPU to memory!", state);
        FastllmCudaFree(data);
    }

    DeviceSync();
}

struct CudaMemoryBuffer {
    void *data;
    size_t size;
    bool busy;

    CudaMemoryBuffer() {}

    CudaMemoryBuffer(void *data, size_t size, bool busy) : data(data), size(size), busy(busy) {}
};
std::map<int, std::vector<CudaMemoryBuffer>> cudaBuffersMap;
std::map<int, int> cudaBuffersMinId; // 最小的空闲id
std::map<int, size_t> noBusyCnt;
std::map<int, std::vector<CudaMemoryBuffer>> bigBuffersMap;

static size_t fastllmCudaMemPoolAllocated = 0;
static size_t fastllmCudaMemPoolPeak = 0;
static size_t fastllmCudaWeightSlabBytes = 0;

void FastllmCudaSetWeightSlabBytes(size_t bytes) {
    fastllmCudaWeightSlabBytes = bytes;
}

size_t FastllmCudaGetWeightSlabBytes() {
    return fastllmCudaWeightSlabBytes;
}

void *FastllmCudaMallocModelWeight(size_t size) {
    return FastllmCudaMalloc(size);
}

#ifdef CUDA_MEM_DEBUG
#    include <cxxabi.h>
#    include <execinfo.h>
#    include <sys/stat.h>

#    include <algorithm>
#    include <chrono>
#    include <fstream>
#    include <iomanip>
#    include <mutex>
#    include <sstream>
#    include <thread>

struct CudaMemDebugInfo {
    size_t size;
    std::string callstack;
};

static std::mutex cudaMemDebugMutex;
static std::map<void *, CudaMemDebugInfo> cudaMemDebugMap;
static bool cudaMemDebugThreadStarted = false;
static size_t cudaMemDebugPeakUsed = 0;

static std::string CudaMemDebugGetCallStack() {
    const int maxFrames = 128;
    void *frames[maxFrames];
    int numFrames = backtrace(frames, maxFrames);
    char **symbols = backtrace_symbols(frames, numFrames);
    std::string result;
    if (symbols) {
        int skip = 0;
        int end = std::min(numFrames, skip + 16);
        for (int i = skip; i < end; i++) {
            result += "  #" + std::to_string(i - skip) + " " + symbols[i] + "\n";
        }
        free(symbols);
    }
    return result;
}

// caller must hold cudaMemDebugMutex; suffix is appended to filename (e.g. "" or "_peak_12345MB")
static void CudaMemDebugWriteReport(const std::string &suffix) {
    mkdir("Debug", 0755);

    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::tm tm_buf;
    localtime_r(&t, &tm_buf);

    std::ostringstream fnss;
    fnss << "Debug/" << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << "_" << std::setfill('0') << std::setw(3) << ms.count() << suffix << ".txt";
    std::string filename = fnss.str();

    size_t totalSize = 0;
    size_t totalCount = cudaMemDebugMap.size();
    std::map<size_t, size_t> sizeDistribution;
    for (auto &it : cudaMemDebugMap) {
        totalSize += it.second.size;
        sizeDistribution[it.second.size]++;
    }

    size_t bigPoolTotal = 0, bigPoolBusy = 0, bigPoolFreeCount = 0, bigPoolBusyCount = 0;
    size_t smallPoolTotal = 0, smallPoolBusy = 0, smallPoolFreeCount = 0, smallPoolBusyCount = 0;
    for (auto &dev : bigBuffersMap) {
        for (auto &b : dev.second) {
            bigPoolTotal += b.size;
            if (b.busy) {
                bigPoolBusy += b.size;
                bigPoolBusyCount++;
            } else {
                bigPoolFreeCount++;
            }
        }
    }
    for (auto &dev : cudaBuffersMap) {
        for (auto &b : dev.second) {
            smallPoolTotal += b.size;
            if (b.busy) {
                smallPoolBusy += b.size;
                smallPoolBusyCount++;
            } else {
                smallPoolFreeCount++;
            }
        }
    }

    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    size_t usedMem = totalMem - freeMem;

    std::ofstream ofs(filename);
    if (!ofs.is_open()) return;

    ofs << "========== CUDA Memory Debug Report ==========\n";
    ofs << "Time: " << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count() << "\n";
    if (!suffix.empty()) ofs << "Trigger: PEAK memory\n";
    ofs << "\n";

    ofs << "--- Summary ---\n";
    ofs << "GPU Memory: used " << (usedMem >> 20) << " MB, free " << (freeMem >> 20) << " MB / total " << (totalMem >> 20) << " MB\n";
    ofs << "Tracked allocations: " << totalCount << " pointers, total " << std::fixed << std::setprecision(2)
        << (double)totalSize / (1024.0 * 1024.0) << " MB\n\n";

    ofs << "Big buffer pool:   total " << (bigPoolTotal >> 20) << " MB, busy " << (bigPoolBusy >> 20) << " MB"
        << " (busy " << bigPoolBusyCount << ", free " << bigPoolFreeCount << ")\n";
    ofs << "Small buffer pool: total " << (smallPoolTotal >> 20) << " MB, busy " << (smallPoolBusy >> 20) << " MB"
        << " (busy " << smallPoolBusyCount << ", free " << smallPoolFreeCount << ")\n";
    ofs << "Pool allocated total: " << (fastllmCudaMemPoolAllocated >> 20) << " MB, peak: " << (fastllmCudaMemPoolPeak >> 20) << " MB\n\n";

    ofs << "--- Size Distribution (tracked) ---\n";
    std::vector<std::pair<size_t, size_t>> sortedDist(sizeDistribution.begin(), sizeDistribution.end());
    std::sort(sortedDist.begin(), sortedDist.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    for (auto &p : sortedDist) {
        double sizeMB = (double)p.first / (1024.0 * 1024.0);
        if (sizeMB >= 1.0)
            ofs << "  " << std::fixed << std::setprecision(2) << sizeMB << " MB : " << p.second << " blocks\n";
        else
            ofs << "  " << (p.first / 1024) << " KB : " << p.second << " blocks\n";
    }

    ofs << "\n--- Free Buffers in Pool ---\n";
    for (auto &dev : bigBuffersMap) {
        size_t devFreeSize = 0, devFreeCount = 0;
        for (auto &b : dev.second) {
            if (!b.busy) {
                devFreeSize += b.size;
                devFreeCount++;
            }
        }
        if (devFreeCount == 0) continue;
        ofs << "  [Big Pool] Device " << dev.first << ": " << devFreeCount << " free blocks, " << std::fixed << std::setprecision(2)
            << (double)devFreeSize / (1024.0 * 1024.0) << " MB\n";
        for (auto &b : dev.second) {
            if (!b.busy) {
                ofs << "    ptr=" << b.data << ", size=" << std::fixed << std::setprecision(2) << (double)b.size / (1024.0 * 1024.0) << " MB ("
                    << b.size << " bytes)\n";
            }
        }
    }
    for (auto &dev : cudaBuffersMap) {
        size_t devFreeSize = 0, devFreeCount = 0;
        for (auto &b : dev.second) {
            if (!b.busy) {
                devFreeSize += b.size;
                devFreeCount++;
            }
        }
        if (devFreeCount == 0) continue;
        ofs << "  [Small Pool] Device " << dev.first << ": " << devFreeCount << " free blocks, " << std::fixed << std::setprecision(2)
            << (double)devFreeSize / (1024.0 * 1024.0) << " MB\n";
        for (auto &b : dev.second) {
            if (!b.busy) {
                ofs << "    ptr=" << b.data << ", size=" << std::fixed << std::setprecision(2) << (double)b.size / (1024.0 * 1024.0) << " MB ("
                    << b.size << " bytes)\n";
            }
        }
    }

    ofs << "\n--- Unreleased Blocks Detail (" << totalCount << " blocks) ---\n";
    for (auto &it : cudaMemDebugMap) {
        double sizeMB = (double)it.second.size / (1024.0 * 1024.0);
        ofs << "ptr=" << it.first << ", size=" << std::fixed << std::setprecision(2) << sizeMB << " MB (" << it.second.size << " bytes)\n";
        ofs << "  callstack:\n" << it.second.callstack << "\n";
    }

    ofs << "========== End of Report ==========\n";
    ofs.close();

    printf("[CUDA_MEM_DEBUG] Report saved to %s (%zu pointers, %.2f MB tracked, GPU used %zu MB)\n", filename.c_str(), totalCount,
        (double)totalSize / (1024.0 * 1024.0), usedMem >> 20);
    fflush(stdout);
}

static void CudaMemDebugReportThread() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(20));
        std::lock_guard<std::mutex> lock(cudaMemDebugMutex);
        CudaMemDebugWriteReport("");
    }
}

static void CudaMemDebugEnsureThread() {
    if (!cudaMemDebugThreadStarted) {
        cudaMemDebugThreadStarted = true;
        std::thread(CudaMemDebugReportThread).detach();
    }
}

static void CudaMemDebugRecord(void *ptr, size_t size) {
    std::lock_guard<std::mutex> lock(cudaMemDebugMutex);
    CudaMemDebugEnsureThread();
    cudaMemDebugMap[ptr] = {size, CudaMemDebugGetCallStack()};

    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    size_t usedMem = totalMem - freeMem;
    if (usedMem > cudaMemDebugPeakUsed) {
        cudaMemDebugPeakUsed = usedMem;
        std::string suffix = "_peak_" + std::to_string(usedMem >> 20) + "MB";
        CudaMemDebugWriteReport(suffix);
    }
}

static void CudaMemDebugRemove(void *ptr) {
    std::lock_guard<std::mutex> lock(cudaMemDebugMutex);
    cudaMemDebugMap.erase(ptr);
}
#endif // CUDA_MEM_DEBUG

void *FastllmCudaDirectMalloc(size_t size) {
    void *ret;
    cudaError_t state = cudaMalloc(&ret, size);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when allocating %lu kB memory! maybe there's no enough memory left on device.", size >> 10);
        checkCudaErrors("", state);
        return nullptr;
    }
#ifdef CUDA_MEM_DEBUG
    CudaMemDebugRecord(ret, size);
#endif
    return ret;
}

void FastllmCudaDirectFree(void *ret) {
#ifdef CUDA_MEM_DEBUG
    CudaMemDebugRemove(ret);
#endif
    cudaError_t state = cudaFree(ret);
    // checkCudaErrors("Error: CUDA error when release memory!", state);
}

void FastllmCudaMemset0(void *ret, size_t size) {
    cudaMemset(ret, 0, size);
}

void FastllmCudaMemPoolStats() {
    int id = -1;
    cudaGetDevice(&id);
    size_t bigTotal = 0, bigBusy = 0;
    size_t smallTotal = 0, smallBusy = 0;
    auto &bigBuffers = bigBuffersMap[id];
    for (auto &b : bigBuffers) {
        bigTotal += b.size;
        if (b.busy) bigBusy += b.size;
    }
    auto &cudaBuffers = cudaBuffersMap[id];
    for (auto &b : cudaBuffers) {
        smallTotal += b.size;
        if (b.busy) smallBusy += b.size;
    }
    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf(
        "[CUDA_MEM_POOL] dev=%d bigPool: %zu/%zu MB (%zu bufs), smallPool: %zu/%zu MB (%zu bufs), "
        "poolTotal: %zu MB, peak: %zu MB, gpuFree: %zu MB / %zu MB\n",
        id, bigBusy >> 20, bigTotal >> 20, bigBuffers.size(), smallBusy >> 20, smallTotal >> 20, cudaBuffers.size(),
        fastllmCudaMemPoolAllocated >> 20, fastllmCudaMemPoolPeak >> 20, freeMem >> 20, totalMem >> 20);
}

void *FastllmCudaMalloc(size_t size) {
    int id = -1;
    cudaError_t state = cudaSuccess;
    state = cudaGetDevice(&id);
    checkCudaErrors("Error: CUDA error when find device!", state);
    if (size > 1024 * 1024) {
        auto &bigBuffers = bigBuffersMap[id];
        int selId = -1;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].size >= size && !bigBuffers[i].busy &&
                (bigBuffers[i].size <= size * 2 || bigBuffers[i].size - size < 1 * 1024 * 1024)) {
                if (selId == -1 || bigBuffers[selId].size > bigBuffers[i].size) {
                    selId = i;
                }
            }
        }
        if (selId != -1) {
            bigBuffers[selId].busy = true;
#ifdef CUDA_MEM_DEBUG
            CudaMemDebugRecord(bigBuffers[selId].data, size);
#endif
            return bigBuffers[selId].data;
        }

        void *ret;
        state = cudaMalloc(&ret, size);
        if (cudaSuccess != state) {
            printf("Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory left on device.", size >> 20);
            checkCudaErrors("", state);
            return nullptr;
        }
        bigBuffers.push_back(CudaMemoryBuffer(ret, size, true));
#ifdef CUDA_MEM_DEBUG
        CudaMemDebugRecord(ret, size);
#endif
        return ret;
    }
    auto &cudaBuffers = cudaBuffersMap[id];
    for (int i = cudaBuffersMinId[id]; i < cudaBuffers.size(); i++) {
        if (cudaBuffers[i].size >= size && !cudaBuffers[i].busy) {
            cudaBuffers[i].busy = true;
            noBusyCnt[id] -= cudaBuffers[i].size;
            while (cudaBuffersMinId[id] < cudaBuffers.size() && cudaBuffers[cudaBuffersMinId[id]].busy) {
                cudaBuffersMinId[id]++;
            }
#ifdef CUDA_MEM_DEBUG
            CudaMemDebugRecord(cudaBuffers[i].data, size);
#endif
            return cudaBuffers[i].data;
        }
    }
    void *ret;
    state = cudaMalloc(&ret, size);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when allocating %lu KB memory! maybe there's no enough memory left on device.", size >> 10);
        checkCudaErrors("", state);
        return nullptr;
    }
    cudaBuffers.push_back(CudaMemoryBuffer(ret, size, true));
#ifdef CUDA_MEM_DEBUG
    CudaMemDebugRecord(ret, size);
#endif
    return ret;
}

void FastllmCudaForceFree(void *ret) {
    if (ret == nullptr) {
        return;
    }

    int oriId = FastllmCudaGetDevice();
    cudaError_t state = cudaSuccess;

    for (auto &it : cudaBuffersMap) {
        auto &cudaBuffers = it.second;
        for (int i = 0; i < (int)cudaBuffers.size(); i++) {
            if (cudaBuffers[i].data == ret) {
                state = cudaSetDevice(it.first);
                state = cudaFree(cudaBuffers[i].data);
                if (cudaSuccess != state) {
                    printf("Error: CUDA error when force releasing memory on device %d!", it.first);
                }
                cudaBuffers.erase(cudaBuffers.begin() + i);
                noBusyCnt[it.first] = 0;
                cudaBuffersMinId[it.first] = (int)cudaBuffers.size();
                for (int j = 0; j < (int)cudaBuffers.size(); j++) {
                    if (!cudaBuffers[j].busy) {
                        noBusyCnt[it.first] += cudaBuffers[j].size;
                        cudaBuffersMinId[it.first] = std::min(cudaBuffersMinId[it.first], j);
                    }
                }
#ifdef CUDA_MEM_DEBUG
                CudaMemDebugRemove(ret);
#endif
                FastllmCudaSetDevice(oriId);
                checkCudaErrors("CUDA error when force releasing memory!", state);
                return;
            }
        }

        auto &bigBuffers = bigBuffersMap[it.first];
        for (int i = 0; i < (int)bigBuffers.size(); i++) {
            if (bigBuffers[i].data == ret) {
                state = cudaSetDevice(it.first);
                state = cudaFree(bigBuffers[i].data);
                if (cudaSuccess != state) {
                    printf("Error: CUDA error when force releasing big memory on device %d!", it.first);
                }
                bigBuffers.erase(bigBuffers.begin() + i);
#ifdef CUDA_MEM_DEBUG
                CudaMemDebugRemove(ret);
#endif
                FastllmCudaSetDevice(oriId);
                checkCudaErrors("CUDA error when force releasing big memory!", state);
                return;
            }
        }
    }

#ifdef CUDA_MEM_DEBUG
    CudaMemDebugRemove(ret);
#endif
    state = cudaFree(ret);
    FastllmCudaSetDevice(oriId);
    checkCudaErrors("CUDA error when force releasing uncached memory!", state);
}

void FastllmCudaFree(void *ret) {
    if (ret == nullptr) {
        return;
    }
    if (cudaBuffersMap.empty()) return;
    int oriId = FastllmCudaGetDevice();
    cudaError_t state = cudaSuccess;
    for (auto &it : cudaBuffersMap) {
        if (noBusyCnt[it.first] > 1024 * 1024 * 1024) {
            auto &cudaBuffers = it.second;
            std::vector<CudaMemoryBuffer> temp;
            for (int i = 0; i < cudaBuffers.size(); i++) {
                if (!cudaBuffers[i].busy) {
                    state = cudaSetDevice(it.first);
                    state = cudaFree(cudaBuffers[i].data);
                    if (cudaSuccess != state) printf("Error: CUDA error when release memory on device %d!", it.first);
                    checkCudaErrors("", state);
                } else {
                    temp.push_back(cudaBuffers[i]);
                }
            }
            cudaBuffers.clear();
            it.second = temp;
            noBusyCnt[it.first] = 0;
        }
    }

    for (auto &it : cudaBuffersMap) {
        auto &cudaBuffers = it.second;
        for (int i = 0; i < cudaBuffers.size(); i++) {
            if (cudaBuffers[i].data == ret) {
                noBusyCnt[it.first] += cudaBuffers[i].size;
                cudaBuffers[i].busy = false;
                cudaBuffersMinId[it.first] = std::min(cudaBuffersMinId[it.first], i);
#ifdef CUDA_MEM_DEBUG
                CudaMemDebugRemove(ret);
#endif
                return;
            }
        }
        auto &bigBuffers = bigBuffersMap[it.first];
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].data == ret) {
                bigBuffers[i].busy = false;
#ifdef CUDA_MEM_DEBUG
                CudaMemDebugRemove(ret);
#endif
                return;
            }
        }
    }
#ifdef CUDA_MEM_DEBUG
    CudaMemDebugRemove(ret);
#endif
    state = cudaFree(ret);
    FastllmCudaSetDevice(oriId);
    checkCudaErrors("CUDA error when release memory!", state);
}

void FastllmCudaMallocBigBuffer(size_t size) {
    void *ret;
    int id = -1;
    cudaGetDevice(&id);
    auto &bigBuffers = bigBuffersMap[id];
    cudaMalloc(&ret, size);
    auto state = cudaMalloc(&ret, size);
    if (cudaSuccess != state)
        printf("Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory left on device.", size >> 20);
    checkCudaErrors("", state);
    bigBuffers.push_back(CudaMemoryBuffer(ret, size, false));
}

void FastllmCudaClearBigBuffer() {
    int id = -1;
    cudaGetDevice(&id);
    if (bigBuffersMap.empty()) return;
    cudaError_t state = cudaSuccess;
    for (auto &it : bigBuffersMap) {
        auto &bigBuffers = it.second;
        std::vector<CudaMemoryBuffer> temp;
        long long littleMemSum = 0;
        long long littleMemSumLimit = 300 * 1024 * 1024; // 留一小部分复用
        std::vector<std::pair<std::size_t, int>> v;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy) {
                v.push_back(std::make_pair(bigBuffers[i].size, i));
            }
        }
        std::sort(v.begin(), v.end());
        std::set<int> littleMemIds;
        for (int i = 0; i < v.size(); i++) {
            littleMemSum += v[i].first;
            if (littleMemSum > littleMemSumLimit) {
                break;
            }
            littleMemIds.insert(v[i].second);
        }
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy && littleMemIds.find(i) == littleMemIds.end()) {
                state = cudaSetDevice(it.first);
                state = cudaFree(bigBuffers[i].data);
                if (cudaSuccess != state) printf("Error: CUDA error when release memory on device %d!", it.first);
                checkCudaErrors("", state);
            } else {
                temp.push_back(bigBuffers[i]);
            }
        }
        bigBuffers.clear();
        bigBuffers = temp;
    }
    cudaSetDevice(id);
}

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size) {
    cudaError_t state = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copy from memory to GPU!", state);
    // cudaDeviceSynchronize();
}

void FastllmCudaCopyFromPinnedHostToDevice(void *dst, void *src, size_t size) {
    cudaError_t state = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, 0);
    checkCudaErrors("Error: CUDA error when async copy from pinned memory to GPU!", state);
}

void FastllmCudaCopyFromHostToDeviceAsync(void *dst, void *src, size_t size, void *stream) {
    cudaError_t state = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    checkCudaErrors("Error: CUDA error when async copy from memory to GPU!", state);
}

void FastllmCudaCopyFromPinnedHostToDeviceAsync(void *dst, void *src, size_t size, void *stream) {
    cudaError_t state = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    checkCudaErrors("Error: CUDA error when async copy from pinned memory to GPU!", state);
}

void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size) {
    cudaError_t state = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    checkCudaErrors("Error: CUDA error when copy from GPU to memory!", state);
    // cudaDeviceSynchronize();
}

void *FastllmCudaHostMalloc(size_t size) {
    void *ptr = nullptr;
    cudaError_t state = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
    checkCudaErrors("Error: CUDA error when allocating pinned memory!", state);
    return ptr;
}

void FastllmCudaHostFree(void *ptr) {
    if (ptr != nullptr) {
        cudaError_t state = cudaFreeHost(ptr);
        checkCudaErrors("Error: CUDA error when freeing pinned memory!", state);
    }
}

bool FastllmCudaHostRegister(void *ptr, size_t size) {
    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "Warning: cudaHostRegister failed (%s), falling back to unpinned memory\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

void FastllmCudaHostUnregister(void *ptr) {
    if (ptr != nullptr) {
        cudaHostUnregister(ptr);
    }
}

void FastllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size) {
    cudaError_t state = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    checkCudaErrors("Error: CUDA error when copy on GPU!", state);
    // cudaDeviceSynchronize();
}

void FastllmCudaMemcpyBetweenDevices(int dstId, void *dst, int srcId, void *src, size_t size) {
    int canPeerAccess = 0;
    cudaError_t state = cudaDeviceCanAccessPeer(&canPeerAccess, srcId, dstId);
    if (canPeerAccess) {
        state = cudaMemcpyPeer(dst, dstId, src, srcId, size);
    } else {
        uint8_t *cpuData = new uint8_t[size];
        state = cudaSetDevice(srcId);
        state = cudaMemcpy(cpuData, src, size, cudaMemcpyDeviceToHost);

        state = cudaSetDevice(dstId);
        state = cudaMemcpy(dst, cpuData, size, cudaMemcpyHostToDevice);
        delete[] cpuData;
    }
    checkCudaErrors("Error: CUDA error when copy Between GPUs!", state);
    DeviceSync();
}

void FastllmCudaMemcpy2DDeviceToDevice(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) {
    cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice);
    DeviceSync();
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMemcpy2DKernel(uint8_t *dst, size_t dpitch, uint8_t *src, size_t spitch, size_t width, size_t height) {
    int id = blockIdx.x;
    dst += id * dpitch;
    src += id * spitch;
    for (int i = threadIdx.x; i < width; i += THREAD_PER_BLOCK) {
        dst[i] = src[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMemcpyBatchKernel(uint8_t **pointer) {
    int id = blockIdx.x;
    uint8_t *dst = pointer[id * 3];
    uint8_t *src = pointer[id * 3 + 1];
    size_t len = (size_t)(pointer[id * 3 + 2]);
    for (int i = threadIdx.x; i < len; i += THREAD_PER_BLOCK) {
        dst[i] = src[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRepeatKernel(
    void *inputOri, void *outputOri, int outer, int repeatTimes, int inputStride, int outputStride0, int outputStride1, int copyLen) {
    int id = blockIdx.x;
    int i = id / repeatTimes, j = id % repeatTimes;
    uint8_t *output = (uint8_t *)outputOri + i * outputStride0 + j * outputStride1;
    uint8_t *input = (uint8_t *)inputOri + i * inputStride;
    for (int x = threadIdx.x; x < copyLen; x += THREAD_PER_BLOCK) {
        output[x] = input[x];
    }
}

void FastllmCudaRepeat(
    void *input, void *output, int outer, int repeatTimes, int inputStride, int outputStride0, int outputStride1, int copyLen) {
    FastllmRepeatKernel<256>
        <<<outer * repeatTimes, 256>>>(input, output, outer, repeatTimes, inputStride, outputStride0, outputStride1, copyLen);
    DeviceSync();
}

void FastllmCudaMemcpy2DDeviceToDeviceBatch(
    void **dsts, size_t *dpitchs, void **srcs, size_t *spitchs, size_t *widths, size_t *heights, int batch) {
    int total = 0;
    for (int i = 0; i < batch; i++) {
        total += heights[i];
    }
    uint8_t **pointers = (uint8_t **)FastllmCudaMalloc(sizeof(uint8_t *) * total * 3);
    uint8_t **cpuPointers = new uint8_t *[total * 3];
    int cur = 0;
    for (int i = 0; i < batch; i++) {
        for (int h = 0; h < heights[i]; h++) {
            cpuPointers[cur * 3 + 0] = (uint8_t *)dsts[i] + h * dpitchs[i];
            cpuPointers[cur * 3 + 1] = (uint8_t *)srcs[i] + h * spitchs[i];
            cpuPointers[cur * 3 + 2] = (uint8_t *)(widths[i]);

            cur++;
        }
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t *) * total * 3, cudaMemcpyHostToDevice);
    FastllmMemcpyBatchKernel<256><<<total, 256>>>(pointers);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
}

void FastllmCudaSetDevice(int gpu_id) {
    cudaSetDevice(gpu_id);
}

int FastllmCudaGetDevice() {
    int id = -1;
    cudaGetDevice(&id);
    return id;
}

int GetPointerDeviceId(void *ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if (err == cudaSuccess) {
#if (CUDART_VERSION < 10000) && !(defined(USE_ROCM))
        if (attributes.memoryType == cudaMemoryTypeDevice) {
#else
        if (attributes.type == cudaMemoryTypeDevice) {
#endif
            int device = attributes.device;
            // printf("Pointer belongs to device %d\n", device);
            return device;
        } else {
            printf("Pointer is not device memory\n");
            return -1;
        }
    } else {
        printf("Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
}

int FastllmCudaGetDeviceCount() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}
