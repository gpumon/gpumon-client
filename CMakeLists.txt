cmake_minimum_required(VERSION 3.18)
project(gpufl_client
    VERSION 0.1.0
    LANGUAGES CXX CUDA
    DESCRIPTION "Header-only GPU monitoring client library"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only library
add_library(gpufl INTERFACE)

add_compile_definitions(GFL_BACKEND_CUDA)

target_include_directories(gpufl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(gpufl INTERFACE cxx_std_17)

# Namespaced aliases (new and legacy)
add_library(gpufl::gpufl ALIAS gpufl)

# Python Bindings (pybind11)
option(BUILD_PYTHON "Build Python bindings" OFF)

if(BUILD_PYTHON)
    find_package(CUDAToolkit REQUIRED)

    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(_gpufl_client python/bindings.cpp)

    target_link_libraries(_gpufl_client PRIVATE
        CUDA::cudart gpufl)

    target_include_directories(_gpufl_client PRIVATE include)
    install(TARGETS _gpufl_client DESTINATION gpufl)
endif()
# Optional: Build example
option(BUILD_GPUFL_EXAMPLE "Build gpufl example application" ON)
if(BUILD_GPUFL_EXAMPLE AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_subdirectory(example/cuda)
endif()

# Installation support
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install header files
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install targets
install(TARGETS gpufl
    EXPORT gpufl_clientTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets
install(EXPORT gpufl_clientTargets
    FILE gpufl_clientTargets.cmake
    NAMESPACE gpufl::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpufl_client
)
