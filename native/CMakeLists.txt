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
set(PYBIND11_FINDPYTHON ON)
find_package(Python 3.12 EXACT COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Threads REQUIRED)
include(CTest)

# Find pybind11
find_package(pybind11 3.0.1 QUIET)
if(NOT pybind11_FOUND)
    # If pybind11 is not found, download it
    message(STATUS "Downloading pybind11")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v3.0.1
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
    src/device/buffer.cpp
    src/render/scheduler.cpp
    src/device/gs_pipeline.cpp
    src/device/pipeline_ring.cpp
    src/render/gs_renderer.cpp
    src/training/gs_trainer.cpp
    src/device/perf_timer.cpp
    src/collective/host_mailbox_owner.cpp
    src/collective/sync_generation_state.cpp
    src/bindings/python_bindings.cpp
)

# Create the Python module
pybind11_add_module(_ftgspp_vk ${SOURCES})
target_include_directories(_ftgspp_vk PRIVATE src)

# Link libraries
target_link_libraries(_ftgspp_vk PRIVATE 
    Vulkan::Vulkan
    glm::glm
    Threads::Threads
    # asan
)

if(BUILD_TESTING)
    add_executable(pair_collective_test
        tests/pair_collective_test.cpp
        src/render/scheduler.cpp
        src/collective/host_mailbox_owner.cpp
        src/collective/sync_generation_state.cpp
    )
    target_include_directories(
        pair_collective_test PRIVATE src)
    target_link_libraries(
        pair_collective_test PRIVATE Threads::Threads)
    add_test(
        NAME pair_collective_contracts
        COMMAND pair_collective_test)
endif()

# Compiler-specific options
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
install(TARGETS _ftgspp_vk DESTINATION ftgspp)
file(GLOB_RECURSE SHADER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/slang/*"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*")
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/manifest.json"
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/compile_shaders.py"
        --src "${CMAKE_CURRENT_SOURCE_DIR}/slang"
        --dst "${CMAKE_CURRENT_BINARY_DIR}/shaders" --force
    DEPENDS ${SHADER_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/compile_shaders.py"
    VERBATIM)
add_custom_target(ftgspp_shaders ALL
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/shaders/manifest.json")
add_dependencies(_ftgspp_vk ftgspp_shaders)
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders/"
    DESTINATION ftgspp/shaders FILES_MATCHING PATTERN "*.spv" PATTERN "manifest.json")
