#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "gpufl/gpufl.hpp"

#if !defined(__CUDACC__)
  #error "gpufl/cuda/launch.hpp must be compiled with NVCC (__CUDACC__)"
#endif
#define GFL_LAUNCH_TAGGED(tagArg, kernel, gridArg, blockArg, sharedMem, stream, ...) \
    do { \
        dim3 _gDim(gridArg); \
        dim3 _bDim(blockArg); \
        \
        auto& _attrs = gpufl::cuda::get_kernel_static_attrs(kernel); \
        std::string _gStr = gpufl::cuda::dim3ToString(_gDim); \
        std::string _bStr = gpufl::cuda::dim3ToString(_bDim); \
        \
        int _dev; cudaGetDevice(&_dev); \
        cudaDeviceProp _prop; cudaGetDeviceProperties(&_prop, _dev); \
        int _blockSize = _bDim.x * _bDim.y * _bDim.z; \
        int _maxActiveBlocks = 0; \
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&_maxActiveBlocks, kernel, _blockSize, sharedMem); \
        \
        float _occupancy = 0.0f; \
        if (_prop.maxThreadsPerMultiProcessor > 0) { \
            int _activeWarps = _maxActiveBlocks * (_blockSize / _prop.warpSize); \
            int _maxWarps = _prop.maxThreadsPerMultiProcessor / _prop.warpSize; \
            _occupancy = (float)_activeWarps / (float)_maxWarps; \
        } \
        \
        { \
            auto _events = gpufl::cuda::getEventPair(); \
            gpufl::cuda::PendingKernel _pk; \
            _pk.name = #kernel; \
            _pk.tag = tagArg;   \
            _pk.grid = _gStr;   \
            _pk.block = _bStr;  \
            _pk.cpuDispatchNs = gpufl::detail::getTimestampNs(); \
            _pk.startEvent = _events.first; \
            _pk.stopEvent = _events.second; \
            _pk.numRegs = _attrs.numRegs; \
            _pk.staticShared = _attrs.sharedSizeBytes; \
            _pk.dynShared = (int)sharedMem; \
            _pk.localBytes = _attrs.localSizeBytes; \
            _pk.constBytes = _attrs.constSizeBytes; \
            _pk.occupancy = _occupancy; \
            _pk.maxActiveBlocks = _maxActiveBlocks; \
            _pk.deviceId = _dev; \
            \
            cudaEventRecord(_pk.startEvent, stream); \
            kernel<<<gridArg, blockArg, sharedMem, stream>>>(__VA_ARGS__); \
            cudaEventRecord(_pk.stopEvent, stream); \
            \
            gpufl::cuda::pushPendingKernel(std::move(_pk)); \
        } \
    } while(0)

#define GFL_LAUNCH(kernel, grid, block, sharedMem, stream, ...) \
    GFL_LAUNCH_TAGGED("", kernel, grid, block, sharedMem, stream, __VA_ARGS__)