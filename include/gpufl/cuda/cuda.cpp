#define GPUFL_EXPORTS
#include "gpufl/cuda/cuda.hpp"

#include <deque>

#include "gpufl/core/events.hpp"
#include "gpufl/core/runtime.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/logger.hpp"

namespace gpufl::cuda {
    static std::mutex g_queueMutex;
    static std::deque<PendingKernel> g_pendingQueue;

    static std::mutex g_poolMutex;
    static std::vector<cudaEvent_t> g_eventPool;

    std::string dim3ToString(const dim3 v) {
        std::ostringstream oss;
        oss << "(" << v.x << "," << v.y << "," << v.z << ")";
        return oss.str();
    }

    const cudaDeviceProp& getDevicePropsCached(const int deviceId) {
        static std::mutex mu;
        static std::vector<cudaDeviceProp> cache;
        std::lock_guard<std::mutex> lk(mu);

        // Resize cache if needed (assuming max 32 GPUs to be safe)
        if (cache.empty()) cache.resize(32, {0});

        if (deviceId < 0 || deviceId >= 32) return cache[0];

        if (cache[deviceId].name[0] == 0) {
            cudaGetDeviceProperties(&cache[deviceId], deviceId);
        }
        return cache[deviceId];
    }
    void initEventPool(size_t initialSize) {
        std::lock_guard<std::mutex> lk(g_poolMutex);
        for (size_t i = 0; i < initialSize; ++i) {
            cudaEvent_t e;
            // cudaEventBlockingSync is NOT needed here because we use cudaEventQuery
            cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
            // Actually we NEED timing for duration, so default flags:
            cudaEventCreate(&e);
            g_eventPool.push_back(e);
        }
    }

    void freeEventPool() {
        std::lock_guard<std::mutex> lk(g_poolMutex);
        for (auto e : g_eventPool) {
            cudaEventDestroy(e);
        }
        g_eventPool.clear();
    }

    std::pair<cudaEvent_t, cudaEvent_t> getEventPair() {
        std::lock_guard lk(g_poolMutex);

        cudaEvent_t e1, e2;

        if (g_eventPool.size() < 2) {
            // Pool empty? Allocate new ones (Slow path)
            cudaEventCreate(&e1);
            cudaEventCreate(&e2);
        } else {
            // Fast path
            e1 = g_eventPool.back(); g_eventPool.pop_back();
            e2 = g_eventPool.back(); g_eventPool.pop_back();
        }
        return {e1, e2};
    }

    void returnEventPair(cudaEvent_t start, cudaEvent_t stop) {
        std::lock_guard<std::mutex> lk(g_poolMutex);
        g_eventPool.push_back(start);
        g_eventPool.push_back(stop);
    }

    // --- Deferred Engine ---

    void pushPendingKernel(PendingKernel&& k) {
        std::lock_guard<std::mutex> lk(g_queueMutex);
        g_pendingQueue.push_back(std::move(k));
    }

    void processPendingKernels() {
        Runtime* rt = runtime();
        if (!rt || !rt->logger) return;

        // Take a snapshot of the queue to iterate without blocking pushers too long
        // OR just lock tightly. Since we only pop finished items, locking is fine.
        std::unique_lock<std::mutex> lk(g_queueMutex);

        auto it = g_pendingQueue.begin();
        while (it != g_pendingQueue.end()) {
            PendingKernel& k = *it;

            // KEY: Check if GPU has finished this kernel.
            // cudaEventQuery is NON-BLOCKING.

            if (const cudaError_t status = cudaEventQuery(k.stopEvent); status == cudaSuccess) {
                // --- KERNEL FINISHED ---

                float durationMs = 0.0f;
                cudaEventElapsedTime(&durationMs, k.startEvent, k.stopEvent);

                const auto durationNs = static_cast<int64_t>(durationMs * 1'000'000.0);
                const int64_t startTime = k.cpuDispatchNs;
                const int64_t endTime = startTime + durationNs;

                // Generate Start Event
                {
                    KernelBeginEvent e;
                    e.pid = detail::getPid();
                    e.app = rt->appName;
                    e.name = k.name;
                    e.tag = k.tag;
                    e.tsNs = startTime;
                    e.grid = k.grid;
                    e.block = k.block;
                    e.numRegs = k.numRegs;
                    e.staticSharedBytes = k.staticShared;
                    e.dynSharedBytes = k.dynShared;
                    e.localBytes = k.localBytes;
                    e.constBytes = k.constBytes;
                    e.occupancy = k.occupancy;
                    e.maxActiveBlocks = k.maxActiveBlocks;
                    rt->logger->logKernelBegin(e);
                }

                // Generate END Event
                {
                    KernelEndEvent e;
                    e.pid = detail::getPid();
                    e.app = rt->appName;
                    e.name = k.name;
                    e.tag = k.tag;
                    e.tsNs = endTime; // <--- The visualizer uses this for end time

                    rt->logger->logKernelEnd(e);
                }

                // 3. Cleanup
                returnEventPair(k.startEvent, k.stopEvent);

                // Remove from queue and continue
                it = g_pendingQueue.erase(it);
            }
            else if (status == cudaErrorNotReady) {
                ++it;
            }
            else {
                returnEventPair(k.startEvent, k.stopEvent);
                it = g_pendingQueue.erase(it);
            }
        }
    }
}