#pragma once
#include <memory>
#include <mutex>
#include <string>

#include "gpufl/core/sampler.hpp"

namespace gpufl {
    class Logger;

    struct Runtime {
        std::string appName;
        std::shared_ptr<Logger> logger;
        std::shared_ptr<ISystemCollector> collector;

        // background system sampling
        std::atomic<bool> systemSampling{false};
        Sampler sampler;
        std::mutex systemMu;
        std::thread systemThread;
        int systemIntervalMs{0};
    };

    Runtime* runtime();
    void set_runtime(std::unique_ptr<Runtime> rt);
}
