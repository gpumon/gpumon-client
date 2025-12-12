#ifndef GPUFL_COMMON_HPP
#define GPUFL_COMMON_HPP

#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <vector>
#include <sstream>
#include <cstdint>
#include <atomic>
#include <map>
#include <filesystem>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace gpufl {

    enum class LogCategory {
        Meta,   // Init, Shutdown for all files
        Kernel,
        Scope,
        System
    };

    struct InitOptions {
        std::string appName;
        std::string logPath;
        uint32_t sampleIntervalMs = 0; // 0 to disable background sampling
        size_t maxFileSizeBytes = 2 * 1024 * 1024; // Default 2 MB
    };

    namespace detail {
        // Snapshot of Device State (Identity + Memory)
        struct DeviceSnapshot {
            int deviceId = 0;
            std::string name;
            std::string uuid; // Extracted via Runtime API
            int pciBusId = 0;

            size_t freeMiB = 0;
            size_t totalMiB = 0;
            size_t usedMiB = 0;

            unsigned int gpuUtil = 0;      // %
            unsigned int memUtil = 0;      // %
            unsigned int tempC = 0;        // Celsius
            unsigned int powermW = 0;      // Milliwatts
            unsigned int clockGfx = 0;     // MHz
            unsigned int clockSm = 0;      // MHz
            unsigned int clockMem = 0;     // MHz
        };

        struct LogFileState {
            std::ofstream stream;
            std::string basePath;
            int index = 0;
            size_t currentBytes = 0;
        };

        struct State {
            std::string appName;
            std::map<LogCategory, LogFileState> logFiles;
            std::mutex logMutex;
            int32_t pid;
            std::atomic<bool> initialized;
            uint32_t sampleIntervalMs;
            size_t maxSizeBytes;

            State() : pid(0), initialized(false), sampleIntervalMs(0), maxSizeBytes(0) {}
        };

        inline State& getState() {
            static State state;
            return state;
        }

        inline std::mutex& getInitMutex() {
            static std::mutex m;
            return m;
        }

        inline int32_t getPid() {
        #ifdef _WIN32
            return static_cast<int32_t>(_getpid());
        #else
            return static_cast<int32_t>(getpid());
        #endif
        }

        inline int64_t getTimestampNs() {
            const auto now = std::chrono::steady_clock::now();
            const auto duration = now.time_since_epoch();
            return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
        }

        inline std::string generateLogPath(const std::string& prefix, const std::string& suffix) {
            return prefix + "." + suffix + ".log";
        }

        // ============================================================================
        // JSON & Logging Utilities
        // ============================================================================

        inline void rotateFile(LogFileState& fs) {
            if (fs.stream.is_open()) fs.stream.close();

            std::string path = fs.basePath + ".log";
            if (fs.index > 0) {
                path = fs.basePath + "." + std::to_string(fs.index) + ".log";
            }

            fs.stream.open(path, std::ios::out | std::ios::app);
            fs.currentBytes = 0;

            if (fs.stream.is_open()) {
                fs.stream.seekp(0, std::ios::end);
                fs.currentBytes = fs.stream.tellp();
            }
        }

        inline std::string escapeJson(const std::string& str) {
            std::ostringstream oss;
            for (const char c : str) {
                switch (c) {
                    case '"':  oss << "\\\""; break;
                    case '\\': oss << "\\\\"; break;
                    default:   oss << c; break;
                }
            }
            return oss.str();
        }

        inline void writeLogLine(const LogCategory category, const std::string& jsonLine) {
            State& state = getState();
            std::lock_guard lock(state.logMutex);

            if (!state.initialized) return;

            auto writeTo = [&](LogFileState& fs) {
                if (fs.currentBytes + jsonLine.size() + 1 > state.maxSizeBytes) {
                    fs.index++;
                    rotateFile(fs);
                }

                if (fs.stream.is_open()) {
                    fs.stream << jsonLine << '\n';
                    fs.stream.flush();
                    fs.currentBytes += jsonLine.size() + 1;
                }
            };

            // Meta events (Init/Shutdown) write to ALL open files to keep them self-contained
            if (category == LogCategory::Meta) {
                for (auto& pair : state.logFiles) {
                    writeTo(pair.second);
                }
                return;
            }

            // Normal events write to their specific file
            auto it = state.logFiles.find(category);
            if (it != state.logFiles.end()) {
                writeTo(it->second);
            }
        }

        // Helper to format the memory array into JSON
        inline void writeDeviceJson(std::ostringstream& oss, const std::vector<DeviceSnapshot>& snapshots) {
            if (snapshots.empty()) return;
            oss << ",\"devices\":[";
            for (size_t i = 0; i < snapshots.size(); ++i) {
                if (i > 0) oss << ",";
            const auto& s = snapshots[i];
                oss << "{\"id\":" << s.deviceId
                    << ",\"name\":\"" << escapeJson(s.name) << "\""
                    << ",\"uuid\":\"" << s.uuid << "\""
                    << ",\"pci_bus\":" << s.pciBusId
                    << ",\"used_mib\":" << s.usedMiB
                    << ",\"free_mib\":" << s.freeMiB
                    << ",\"total_mib\":" << s.totalMiB
                    << ",\"util_gpu\":" << s.gpuUtil
                    << ",\"util_mem\":" << s.memUtil
                    << ",\"temp_c\":" << s.tempC
                    << ",\"power_mw\":" << s.powermW
                    << ",\"clk_gfx\":" << s.clockGfx
                    << ",\"clk_sm\":" << s.clockSm
                    << ",\"clk_mem\":" << s.clockMem
                    << "}";
            }
            oss << "]";
        }

    } // namespace detail
} // namespace gfl

#endif // GFL_COMMON_HPP