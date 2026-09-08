cmake_minimum_required(VERSION 3.16)
project(FTGSPPVulkan)

option(FTGSPP_VULKAN_VALIDATION "Enable VK_LAYER_KHRONOS_validation" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    set(CMAKE_CXX_FLAGS "/W4")
    # set(CMAKE_CXX_FLAGS_DEBUG "/Zi /fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    # set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address -fno-omit-frame-pointer -fno-common")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

# Find required packages
find_package(Vulkan REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Threads REQUIRED)
include(CTest)

# Find pybind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # If pybind11 is not found, download it
    message(STATUS "Downloading pybind11")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.11.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Find GLM
find_package(glm QUIET)
if(NOT glm_FOUND)
    # If GLM is not found, download it
    message(STATUS "Downloading GLM")
    include(FetchContent)
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm
        GIT_TAG        0.9.9.8
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(glm)
endif()

# Source files
set(SOURCES
    src/buffer.cpp
    src/scheduler.cpp
    src/gs_pipeline.cpp
    src/pipeline_ring.cpp
    src/gs_renderer.cpp
    src/gs_trainer.cpp
    src/perf_timer.cpp
    src/host_mailbox_owner.cpp
    src/hierarchical_host_collective.cpp
    src/sync_generation_state.cpp
    src/python_bindings.cpp
)

set(HEADERS
    src/config.h
    src/buffer.h
    src/scheduler.h
    src/gs_pipeline.h
    src/gs_renderer.h
    src/gs_trainer.h
    src/perf_timer.h
    src/host_mailbox_owner.h
    src/hierarchical_host_collective.h
    src/sync_generation_state.h
)

# Create the Python module
pybind11_add_module(_ftgspp_vk ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(_ftgspp_vk PRIVATE 
    Vulkan::Vulkan
    glm::glm
    Threads::Threads
    # asan
)

if(BUILD_TESTING)
    add_executable(hierarchical_host_collective_test
        tests/hierarchical_host_collective_test.cpp
        src/hierarchical_host_collective.cpp
        src/host_mailbox_owner.cpp
        src/sync_generation_state.cpp
    )
    target_include_directories(
        hierarchical_host_collective_test PRIVATE src)
    target_link_libraries(
        hierarchical_host_collective_test PRIVATE Threads::Threads)
    add_test(
        NAME hierarchical_host_collective_contracts
        COMMAND hierarchical_host_collective_test)
endif()

# Compiler-specific options
target_compile_definitions(_ftgspp_vk PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
if(FTGSPP_VULKAN_VALIDATION)
    target_compile_definitions(_ftgspp_vk PRIVATE ENABLE_VULKAN_VALIDATION_LAYER=1)
endif()

if(DEFINED OPTIMIZATION_LEVEL)
    target_compile_definitions(_ftgspp_vk PRIVATE
        "OPTIMIZATION_LEVEL=${OPTIMIZATION_LEVEL}")
endif()

# Set properties
set_target_properties(_ftgspp_vk PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

# This module is compute-only and creates no window-system surface.  Keeping
# platform surface macros disabled avoids unnecessary X11/Win32/Metal header
# dependencies on headless training servers.

# Install target
install(TARGETS _ftgspp_vk DESTINATION .)
