#include "gpufl/cuda/cupti_backend.hpp"
#include "gpufl/core/ring_buffer.hpp"
#include "gpufl/core/common.hpp"
#include "gpufl/core/trace_type.hpp"
#include "gpufl/cuda/cuda.hpp"

#include <iostream>
#include <cstring>

namespace gpufl {

    std::atomic<gpufl::CuptiBackend*> g_activeBackend{nullptr};

    // External ring buffer (defined in monitor.cpp)
    extern RingBuffer<ActivityRecord, 1024> g_monitorBuffer;

    void CuptiBackend::Initialize(const MonitorOptions &opts) {
        // Check if another backend is already active
        CuptiBackend* expected = nullptr;
        if (!g_activeBackend.compare_exchange_strong(expected, this, std::memory_order_release)) {
            std::cerr << "[GPUFL Monitor] ERROR: Another CUPTI backend is already active" << std::endl;
            initialized_ = false;
            return;
        }

        opts_ = opts;
        if (opts_.enable_debug_output) {
            std::cout << "[GPUFL Monitor] Subscribing to CUPTI..." << std::endl;
        }
        CUptiResult res = cuptiSubscribe(&subscriber_,
                                         (CUpti_CallbackFunc) GflCallback,
                                         this);
        if (res != CUPTI_SUCCESS) {
            g_activeBackend.store(nullptr, std::memory_order_release);
            initialized_ = false;
            std::cerr << "[GPUFL Monitor] ERROR: Failed to subscribe to CUPTI (code " << res << ")" << std::endl;
            if (res == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) {
                std::cerr << "[GPUFL Monitor] Multiple CUPTI subscribers detected!" << std::endl;
                std::cerr << "[GPUFL Monitor] Possible causes:" << std::endl;
                std::cerr << "  - Running under a debugger with GPU debugging enabled" << std::endl;
                std::cerr << "  - Running under a profiler (Nsight Systems, nvprof, etc.)" << std::endl;
                std::cerr << "  - Another monitoring tool is active" << std::endl;
                std::cerr << "  - Previous gpufl instance wasn't properly shutdown" << std::endl;
                std::cerr << "[GPUFL Monitor] Solution: Run without debugger (Ctrl+F5) or disable GPU debugging" << std::endl;
            } else {
                std::cerr << "[GPUFL Monitor] This may indicate:" << std::endl;
                std::cerr << "  - CUPTI library not found or incompatible" << std::endl;
                std::cerr << "  - Insufficient permissions" << std::endl;
                std::cerr << "  - CUDA driver issues" << std::endl;
            }
            return;
        }
        initialized_ = true;
        if (opts_.enable_debug_output) {
            std::cout << "[GPUFL Monitor] CUPTI subscription successful" <<
                    std::endl;
        }

        // Enable resource domain immediately to catch context creation
        cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RESOURCE);

        // Activity API initialization
        if (opts_.enable_debug_output) {
            std::cout << "[GPUFL Monitor] Registering activity callbacks..." <<
                    std::endl;
        }
        cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);
    }

    void CuptiBackend::Shutdown() {
        if (!initialized_) return;
        cuptiActivityFlushAll(1);
        cuptiUnsubscribe(subscriber_);
        g_activeBackend.store(nullptr, std::memory_order_release);
        initialized_ = false;
    }

    CUptiResult(* CuptiBackend::get_value())(CUpti_ActivityKind) {
        return cuptiActivityEnable;
    }

    void CuptiBackend::Start() {
        if (!initialized_) return;
        active_.store(true);

        // Ensure CUDA context is initialized before enabling callbacks
        if (opts_.enable_debug_output) {
            std::cout << "[GPUFL Monitor] Initializing CUDA context..." <<
                    std::endl;
        }
        cudaError_t err = cudaFree(0);
        if (err != cudaSuccess) {
            std::cerr <<
                    "[GPUFL Monitor] WARNING: Failed to initialize CUDA context: "
                    << cudaGetErrorString(err) << std::endl;
        }

        // Enable domains when starting
        if (opts_.enable_debug_output) {
            std::cout << "[GPUFL Monitor] Enabling CUPTI domains..." <<
                    std::endl;
        }
        cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API);
        cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API);
        cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_STATE);

        // Enable all runtime API callbacks (required for callback-based monitoring)
        if (opts_.enable_debug_output) {
            std::cout << "[GPUFL Monitor] Enabling runtime API callbacks..." <<
                    std::endl;
        }

        // Enable activity for kernels
        if (opts_.enable_debug_output) {
            std::cout << "[GPUFL Monitor] Enabling kernel activity tracking..."
                    << std::endl;
        }
        const CUptiResult res1 =
                cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);

        if (res1 != CUPTI_SUCCESS) {
            const char* s = nullptr;
            cuptiGetResultString((CUptiResult)res1, &s);

            std::cerr <<
                    "[GPUFL Monitor] WARNING: Failed to enable kernel activity: "
                    << res1 << ", " << s << std::endl;
        }
        if (const CUptiResult res2 = get_value()(
                CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
            res2 != CUPTI_SUCCESS) {
            std::cerr <<
                    "[GPUFL Monitor] WARNING: Failed to enable kernel activity: "
                    << res1 << ", " << res2 << std::endl;
        }
        if (opts_.enable_debug_output) {
            std::cout << "[GPUFL Monitor] Start complete" << std::endl;
        }
    }

    void CuptiBackend::Stop() {
        if (!initialized_) return;
        active_.store(false);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        cuptiActivityFlushAll(1);
    }

    // Static callback implementations
    void CUPTIAPI CuptiBackend::BufferRequested(uint8_t **buffer, size_t *size,
                                                size_t *maxNumRecords) {
        auto* backend = g_activeBackend.load(std::memory_order_acquire);
        if (backend->GetOptions().enable_debug_output) {
            std::cout << "[CUPTI] BufferRequested" << std::endl;
        }
        *size = 64 * 1024;
        *buffer = static_cast<uint8_t *>(malloc(*size));
        *maxNumRecords = 0;
    }

    void CUPTIAPI CuptiBackend::BufferCompleted(CUcontext context,
                                                uint32_t streamId,
                                                uint8_t *buffer, size_t size,
                                                const size_t validSize) {
        auto* backend = g_activeBackend.load(std::memory_order_acquire);
        if (backend->GetOptions().enable_debug_output) {
            std::cout << "[CUPTI] BufferCompleted validSize=" << validSize << std::endl;
        }
        CUpti_Activity *record = nullptr;

        static int64_t baseCpuNs = detail::getTimestampNs();
        static uint64_t baseCuptiTs = 0;
        if (baseCuptiTs == 0) cuptiGetTimestamp(&baseCuptiTs);

        if (validSize > 0) {
            while (true) {
                const CUptiResult st = cuptiActivityGetNextRecord(
                    buffer, validSize, &record);
                if (st == CUPTI_SUCCESS) {
                    if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL ||
                        record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {

                        const auto *k = reinterpret_cast<const
                            CUpti_ActivityKernel9 *>(record);

                        ActivityRecord out{};
                        out.type = TraceType::KERNEL;
                        out.stream = nullptr;
                        out.startEvent = nullptr;
                        out.stopEvent = nullptr;

                        // Prefer the true kernel name from activity record
                        std::snprintf(out.name, sizeof(out.name), "%s",
                                      (k->name ? k->name : "kernel"));
                        out.name[sizeof(out.name) - 1] = '\0';

                        // Convert GPU timestamps -> "CPU-ish" ns timeline (same technique you used)
                        out.cpuStartNs =
                                baseCpuNs + static_cast<int64_t>(
                                    k->start - baseCuptiTs);
                        out.durationNs = static_cast<int64_t>(
                            k->end - k->start);

                        // Default: no details unless joined
                        out.hasDetails = false;
                        out.gridX = out.gridY = out.gridZ = 0;
                        out.blockX = out.blockY = out.blockZ = 0;
                        out.dynShared = out.staticShared = out.localBytes = out.constBytes = out.numRegs = 0;
                        out.occupancy = 0.0f;
                        out.maxActiveBlocks = 0;

                        // JOIN callback metadata by correlationId
                        const uint64_t corr = k->correlationId;
                        {
                            std::lock_guard<std::mutex> lk(backend->metaMu_);
                            if (auto it = backend->metaByCorr_.find(corr); it != backend->metaByCorr_.end()) {
                                const LaunchMeta &m = it->second;

                                // If you want, you can overwrite name with callback info if activity name is missing
                                // (Usually activity name is best.)
                                // std::snprintf(out.name, sizeof(out.name), "%s", m.name);

                                if (m.hasDetails) {
                                    out.hasDetails = true;
                                    out.gridX = m.gridX;
                                    out.gridY = m.gridY;
                                    out.gridZ = m.gridZ;
                                    out.blockX = m.blockX;
                                    out.blockY = m.blockY;
                                    out.blockZ = m.blockZ;
                                    out.dynShared = m.dynShared;
                                    out.staticShared = m.staticShared;
                                    out.localBytes = m.localBytes;
                                    out.constBytes = m.constBytes;
                                    out.numRegs = m.numRegs;
                                    out.occupancy = m.occupancy;
                                    out.maxActiveBlocks = m.maxActiveBlocks;
                                }

                                // IMPORTANT: erase to prevent unbounded growth
                                backend->metaByCorr_.erase(it);
                            }
                        }

                        g_monitorBuffer.Push(out);
                    }
                } else if (st == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                    // No more records in this buffer
                    break;
                } else {
                    // Any other error: stop parsing this buffer
                    break;
                }
            }
        }

        free(buffer);
    }

    void CUPTIAPI CuptiBackend::GflCallback(void *userdata,
                                            const CUpti_CallbackDomain domain,
                                            CUpti_CallbackId cbid,
                                            const CUpti_CallbackData *cbInfo) {
        auto *backend = static_cast<CuptiBackend *>(userdata);
        if (backend->GetOptions().enable_debug_output) {
            std::cout << "[CUPTI] callback domain=" << domain
                      << " cbid=" << cbid
                      << " site=" << cbInfo->callbackSite
                      << " corr=" << cbInfo->correlationId
                      << " func=" << (cbInfo->functionName ? cbInfo->functionName : "null")
                      << " sym="  << (cbInfo->symbolName ? cbInfo->symbolName : "null")
                      << std::endl;
        }

        if (domain == CUPTI_CB_DOMAIN_RESOURCE && cbid ==
            CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
            cuptiEnableDomain(1, backend->GetSubscriber(),
                              CUPTI_CB_DOMAIN_RUNTIME_API);
            cuptiEnableDomain(1, backend->GetSubscriber(),
                              CUPTI_CB_DOMAIN_DRIVER_API);
            return;
        }

        if (!backend->IsActive()) return;
        if (domain == CUPTI_CB_DOMAIN_STATE) return;

        // Only care about runtime/driver API for launch metadata
        if (domain != CUPTI_CB_DOMAIN_RUNTIME_API && domain !=
            CUPTI_CB_DOMAIN_DRIVER_API) return;

        bool isKernelLaunch = false;

        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
            if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
                cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
                isKernelLaunch = true;
            }
        } else {
            // DRIVER API
            if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunch ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchGrid ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel ||
                cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz) {
                isKernelLaunch = true;
            }
        }

        if (!isKernelLaunch) return;

        const uint64_t corr = cbInfo->correlationId;

        if (cbInfo->callbackSite == CUPTI_API_ENTER) {
            LaunchMeta meta{};
            meta.apiEnterNs = detail::getTimestampNs();

            const char *nm = cbInfo->symbolName
                                 ? cbInfo->symbolName
                                 : cbInfo->functionName;
            if (!nm) nm = "kernel_launch";
            std::snprintf(meta.name, sizeof(meta.name), "%s", nm);

            if (backend->GetOptions().collect_kernel_details &&
                domain == CUPTI_CB_DOMAIN_RUNTIME_API &&
                cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 &&
                cbInfo->functionParams != nullptr) {
                meta.hasDetails = true;

                const auto *params = (cudaLaunchKernel_v7000_params *) (cbInfo->
                    functionParams);

                meta.gridX = params->gridDim.x;
                meta.gridY = params->gridDim.y;
                meta.gridZ = params->gridDim.z;
                meta.blockX = params->blockDim.x;
                meta.blockY = params->blockDim.y;
                meta.blockZ = params->blockDim.z;
                meta.dynShared = static_cast<int>(params->sharedMem);

                cudaFuncAttributes attrs{};
                if (cudaFuncGetAttributes(&attrs, params->func) ==
                    cudaSuccess) {
                    meta.numRegs = attrs.numRegs;
                    meta.staticShared = static_cast<int>(attrs.sharedSizeBytes);
                    meta.localBytes = static_cast<int>(attrs.localSizeBytes);
                    meta.constBytes = static_cast<int>(attrs.constSizeBytes);

                    int dev = 0;
                    cudaGetDevice(&dev);
                    const auto &prop = cuda::getDevicePropsCached(dev);

                    const int blockSize =
                            meta.blockX * meta.blockY * meta.blockZ;
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                        &meta.maxActiveBlocks, params->func, blockSize,
                        meta.dynShared);

                    if (prop.maxThreadsPerMultiProcessor > 0 && prop.warpSize >
                        0 && blockSize > 0) {
                        const int activeWarps =
                                meta.maxActiveBlocks * (
                                    blockSize / prop.warpSize);
                        const int maxWarps =
                                prop.maxThreadsPerMultiProcessor / prop.
                                warpSize;
                        meta.occupancy = (maxWarps > 0)
                                             ? static_cast<float>(activeWarps) /
                                               static_cast<float>(maxWarps)
                                             : 0.0f;
                    }
                }
            }

            // Store by correlationId
            {
                std::lock_guard<std::mutex> lk(backend->metaMu_);
                backend->metaByCorr_[corr] = meta;
            }
        } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
            const int64_t t = detail::getTimestampNs();
            std::lock_guard<std::mutex> lk(backend->metaMu_);
            auto it = backend->metaByCorr_.find(corr);
            if (it != backend->metaByCorr_.end()) {
                it->second.apiExitNs = t;
            }
        }
    }
} // namespace gpufl
