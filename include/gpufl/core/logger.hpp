#pragma once
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "gpufl/core/events.hpp"

namespace gpufl {

    struct DeviceSample;

    class Logger {
    public:
        struct Options {
            std::string basePath;
            std::size_t rotateBytes = 64 * 1024 * 1024; // 64 MiB default
            bool flushAlways = false;
        };

        Logger();
        ~Logger();

        bool open(const Options& opt);
        void close();

        // Structured helpers
        void logInit(const InitEvent& e, const std::string& devicesJson );
        void logShutdown(const ShutdownEvent& e);

        void logKernel(const KernelEvent& e, const std::string& devicesJson);
        void logScopeBegin(const ScopeBeginEvent& e);
        void logScopeEnd(const ScopeEndEvent& e);
        void logScopeSample(const ScopeSampleEvent& e);
        void logSystemStart(const SystemStartEvent& e);
        void logSystemStop(const SystemStopEvent& e);
        void logSystemSample(const SystemSampleEvent& e);

    private:
        struct LogFileState {
            std::ofstream stream;
            std::string basePath;
            int index = 0;
            size_t currentBytes = 0;
        };

        void closeLocked_();

        void writeLine(const std::string& jsonLine);

        // rotation helpers (mutex must be held)
        void ensureOpenLocked();
        void rotateLocked();
        [[nodiscard]] std::string makePathLocked() const;

        std::mutex mu_;
        Options opt_;
        LogFileState file_;
        bool opened_ = false;
    };
}