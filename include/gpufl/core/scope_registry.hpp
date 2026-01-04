#pragma once
#include <string>
#include <vector>

namespace gpufl {
    extern thread_local std::vector<std::string> g_threadScopeStack;
}
