#ifndef GPUFL_HPP
#define GPUFL_HPP

// 1. Backend Auto-Detection
#if !defined(GFL_BACKEND_CUDA) && !defined(GFL_BACKEND_OPENCL)
    #if defined(__CUDACC__)
        #define GFL_BACKEND_CUDA
    #endif
#endif

// 2. Core (InitOptions, State, Logging)
#include "core/common.hpp"

// 3. Backend Selection
#if defined(GFL_BACKEND_CUDA)
    #include "backends/cuda.hpp"
#elif defined(GFL_BACKEND_OPENCL)
    #error "GPUFL: OpenCL backend not implemented."
#else
    #error "GPUFL Error: No backend selected. Define GPUFL_BACKEND_CUDA."
#endif

// 4. Monitor (ScopedMonitor, Init, Shutdown)
#include "core/monitor.hpp"

#endif
