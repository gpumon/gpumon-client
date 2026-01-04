#pragma once

#include <cstdint>
#include <iomanip>
#include <ios>
#include <random>
#include <sstream>
#include <string>
#include "scope_registry.hpp"

namespace gpufl {
    inline thread_local std::vector<std::string> g_threadScopeStack;

    namespace detail {

        static std::string uuidToString(const char* bytes) {
            std::stringstream ss;
            ss << "GPU-";
            ss << std::hex << std::setfill('0');
            for (int i = 0; i < 16; ++i) {
                if (i == 4 || i == 6 || i == 8 || i == 10) ss << "-";
                ss << std::setw(2) << (static_cast<unsigned int>(bytes[i]) & 0xFF);
            }
            return ss.str();
        }

        static std::string generateSessionId() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 15);
            std::uniform_int_distribution<> dis2(8, 11);

            std::stringstream ss;
            ss << std::hex;
            for (int i = 0; i < 8; i++) ss << dis(gen);
            ss << "-";
            for (int i = 0; i < 4; i++) ss << dis(gen);
            ss << "-4"; // UUID version 4
            for (int i = 0; i < 3; i++) ss << dis(gen);
            ss << "-";
            ss << dis2(gen); // UUID variant
            for (int i = 0; i < 3; i++) ss << dis(gen);
            ss << "-";
            for (int i = 0; i < 12; i++) ss << dis(gen);
            return ss.str();
        }

        int64_t getTimestampNs();
        int getPid();
        std::string toIso8601Utc();
    }
}
