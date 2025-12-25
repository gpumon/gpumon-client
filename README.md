# GPUFlight Client Library (gpufl)

**The Flight Recorder for GPU Production Workloads.**

`gpufl` is a lightweight, high-performance C++ observability library designed for always-on monitoring of GPU applications. 
Unlike traditional profilers (Nsight) that stop the world, GPUFlight is designed to run in production with minimal overhead, capturing kernel telemetry and logical scopes into structured logs.

## 🚀 Key Features

- **Kernel Monitoring**: Automatically intercepts all CUDA kernel launches via CUPTI.
- **Production Grade**: Uses a **Lock-Free Ring Buffer** and a **Background Collector Thread** to decouple logging from your hot path.
- **Logical Scoping**: Group thousands of micro-kernels into meaningful phases (e.g., "Inference", "PhysicsStep") using `GFL_SCOPE`.
- **Rich Metadata**: Captures Kernel Names, Grid/Block dimensions, Register counts, and Shared Memory usage.
- **Sidecar Ready**: Outputs structured NDJSON logs designed to be tailed by a separate Agent/Crawler (e.g., Kafka/Elastic).
- **Vendor Agnostic Design**: Architecture ready for AMD (ROCm) support.

---

## 📦 Integration

`gpufl` is designed to be pulled in via CMake `FetchContent`.

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX CUDA)

include(FetchContent)
FetchContent_Declare(
    gpufl
    GIT_REPOSITORY [https://github.com/gpu-flight/gpufl-client.git](https://github.com/gpu-flight/gpufl-client.git)
    GIT_TAG        main 
)
FetchContent_MakeAvailable(gpufl)

add_executable(my_app main.cu)

# Link against gpufl and required CUDA libraries
target_link_libraries(my_app PRIVATE gpufl CUDA::cudart CUDA::cupti)

# Ensure DLLs are copied on Windows (Optional but recommended)
# See 'Troubleshooting' section for post-build commands.