# Learn a lot from the MLC - LLM Project
# https: // github.com/mlc-ai/mlc-llm/blob/main/CMakeLists.txt

cmake_minimum_required(VERSION 3.18)
project(TILE_LANG C CXX)

option(TILE_LANG_STATIC_STDCPP "Statically link libstdc++ for TileLang libraries" ON)
option(TILE_LANG_INSTALL_STATIC_LIB "Install the static library" ON)

if(TILE_LANG_STATIC_STDCPP)
  message(STATUS "Enabling static linking of C++ standard library")
  # Note: We'll apply static linking flags selectively to avoid Python extension conflicts
  # The flags will be applied per-target below rather than globally
endif()

# Set default build type to Release if not provided
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

# Enable compile command export
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT Python_EXECUTABLE)
  execute_process(
    COMMAND which python
    OUTPUT_VARIABLE Python_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(Python_EXECUTABLE "${Python_EXECUTABLE}" CACHE FILEPATH "Path to the Python executable")
endif()

# Define a custom macro for globbing files with conditional CONFIGURE_DEPENDS
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12.0")
  macro(tilelang_file_glob glob variable)
    file(${glob} ${variable} CONFIGURE_DEPENDS ${ARGN})
  endmacro()
else()
  macro(tilelang_file_glob glob variable)
    file(${glob} ${variable} ${ARGN})
  endmacro()
endif()

# Handle TVM prebuild path or use default configuration
if(DEFINED TVM_PREBUILD_PATH)
  message(STATUS "TVM_PREBUILD_PATH: ${TVM_PREBUILD_PATH}")

  if(EXISTS ${TVM_PREBUILD_PATH}/config.cmake)
    include(${TVM_PREBUILD_PATH}/config.cmake)
  endif()
else()
  if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
    include(${CMAKE_BINARY_DIR}/config.cmake)
  elseif(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()

  # Set default build type to RelWithDebInfo if not provided
  if(NOT CMAKE_BUILD_TYPE)
  # Set default build type to Release if not provided
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "Setting default build type to ${CMAKE_BUILD_TYPE}")
  endif()
endif()

# include cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CheckCXXCompilerFlag)
include(TTMetal)

# Enable static runtime build if required
if(TILE_LANG_INSTALL_STATIC_LIB)
  set(BUILD_STATIC_RUNTIME ON)
endif()

# Enforce CUDA standard
if(USE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
endif()

# Enforce HIP standard
if(USE_ROCM)
  set(CMAKE_HIP_STANDARD 17)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "-D__HIP_PLATFORM_AMD__ ${CMAKE_CXX_FLAGS}")
endif()

# Enforce C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate TVM prebuild path
if(NOT DEFINED TVM_PREBUILD_PATH)
  if(DEFINED ENV{TVM_PREBUILD_PATH})
    set(TVM_PREBUILD_PATH "$ENV{TVM_PREBUILD_PATH}")
  endif()
endif()

# Locate TVM source directory
if(NOT DEFINED TVM_SOURCE_DIR)
  if(DEFINED ENV{TVM_SOURCE_DIR})
    set(TVM_SOURCE_DIR "$ENV{TVM_SOURCE_DIR}")
  elseif(DEFINED TVM_PREBUILD_PATH)
    set(TVM_SOURCE_DIR "${TVM_PREBUILD_PATH}/..")
  else()
    set(TVM_SOURCE_DIR ${PROJECT_SOURCE_DIR}/3rdparty/tvm)
  endif()
endif()

# Handle TVM prebuild or build TVM from source
if(DEFINED TVM_PREBUILD_PATH)
  message(STATUS "Using prebuilt TVM from ${TVM_PREBUILD_PATH}")
  add_library(tvm SHARED IMPORTED)
  set_target_properties(tvm PROPERTIES
    IMPORTED_LOCATION "${TVM_PREBUILD_PATH}/libtvm.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TVM_PREBUILD_PATH}/../include"
  )
  add_library(tvm_runtime SHARED IMPORTED)
  set_target_properties(tvm_runtime PROPERTIES
    IMPORTED_LOCATION "${TVM_PREBUILD_PATH}/libtvm_runtime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${TVM_PREBUILD_PATH}/../include"
  )
else()
  message(STATUS "Building TVM from source at ${TVM_SOURCE_DIR}")
  add_subdirectory(${TVM_SOURCE_DIR} tvm EXCLUDE_FROM_ALL)
endif()

# Collect source files
tilelang_file_glob(GLOB TILE_LANG_SRCS
  src/*.cc
  src/layout/*.cc
  src/transform/*.cc
  src/transform/tt/*.cc
  src/op/*.cc
  src/target/utils.cc
  src/target/codegen_cpp.cc
  src/target/rt_mod_cpp.cc
  src/target/tt/*.cc
  # webgpu doesn't have system dependency
  src/target/codegen_webgpu.cc
  # intrin_rule doesn't have system dependency
  src/target/intrin_rule*.cc
)

# Include CUDA source files if CUDA is enabled
if(USE_CUDA)
  tilelang_file_glob(GLOB TILE_LANG_CUDA_SRCS
    src/runtime/*.cc
    src/target/ptx.cc
    src/target/codegen_cuda.cc
    src/target/rt_mod_cuda.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_CUDA_SRCS})
endif()

# Include ROCm source files if ROCm is enabled
if(USE_ROCM)
  tilelang_file_glob(GLOB TILE_LANG_HIP_SRCS
    src/target/codegen_hip.cc
    src/target/rt_mod_hip.cc
  )
  list(APPEND TILE_LANG_SRCS ${TILE_LANG_HIP_SRCS})
endif()

message(STATUS "Collected source files: ${TILE_LANG_SRCS}")

# Add TileLang object library
add_library(tilelang_objs OBJECT ${TILE_LANG_SRCS})

message(STATUS "TVM_SOURCE_DIR: ${TVM_SOURCE_DIR}")
# Include directories for TileLang
set(TILE_LANG_INCLUDES
  ${TVM_SOURCE_DIR}/include
  ${TVM_SOURCE_DIR}/ffi/include
  ${TVM_SOURCE_DIR}/src
  ${TVM_SOURCE_DIR}/3rdparty/dlpack/include
  ${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include
)

# Find CUDA Toolkit
if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)

  if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found. Please set CUDAToolkit_ROOT.")
  endif()

  message(STATUS "CUDA Toolkit includes: ${CUDAToolkit_INCLUDE_DIRS}")
  set(CUDA_MAJOR_VERSION ${CUDAToolkit_VERSION_MAJOR})
  message(STATUS "Setting CUDA_MAJOR_VERSION=${CUDA_MAJOR_VERSION}")
  add_compile_definitions(CUDA_MAJOR_VERSION=${CUDA_MAJOR_VERSION})
  
  list(APPEND TILE_LANG_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
endif(USE_CUDA)

# Find ROCM Toolkit
if(USE_ROCM)
  find_rocm(${USE_ROCM})
  message(STATUS "USE_ROCM: ${USE_ROCM}")

  if(ROCM_FOUND)
    # always set the includedir
    # avoid global retrigger of cmake
    include_directories(SYSTEM ${ROCM_INCLUDE_DIRS})
    add_definitions(-D__HIP_PLATFORM_HCC__=1)
  else()
    message(FATAL_ERROR "ROCM Toolkit not found. Please set HIP_ROOT.")
  endif(ROCM_FOUND)

  message(STATUS "ROCM Toolkit includes: ${ROCM_INCLUDE_DIRS}")
  list(APPEND TILE_LANG_INCLUDES ${ROCM_INCLUDE_DIRS})
endif(USE_ROCM)

# Find TT-Metalium SDK (optional, for Tenstorrent hardware support)
option(USE_REAL_METALIUM "Use real TT-Metalium APIs instead of mock APIs" OFF)

if(USE_REAL_METALIUM)
  # Try to find tt-metal package (uses their package config if installed)
  # Falls back to our custom FindMetalium.cmake if not
  find_package(TT-Metalium QUIET)

  if(NOT TT-Metalium_FOUND)
    # Try legacy FindMetalium.cmake (for custom SDK locations)
    find_package(Metalium QUIET)
  endif()

  if(TT-Metalium_FOUND OR TARGET TT::Metalium)
    message(STATUS "Building with REAL TT-Metalium APIs")
    if(TT-Metalium_DIR)
      message(STATUS "Found TT-Metalium package config: ${TT-Metalium_DIR}")
    endif()

    # Add compile definition to enable real Metalium code paths
    add_compile_definitions(TL_USE_REAL_METALIUM)

    # Set flag for linking (TT::Metalium target will be linked later)
    set(METALIUM_FOUND TRUE)
  else()
    message(WARNING "USE_REAL_METALIUM is ON but Metalium SDK not found.")
    message(WARNING "")
    message(WARNING "To install TT-Metalium SDK:")
    message(WARNING "  1. git clone https://github.com/tenstorrent/tt-metal.git ~/tt-metal")
    message(WARNING "  2. cd ~/tt-metal && cmake -B build && cmake --build build")
    message(WARNING "  3. export CMAKE_PREFIX_PATH=~/tt-metal/build")
    message(WARNING "")
    message(WARNING "Or set TT_METAL_HOME environment variable to SDK location.")
    message(WARNING "See docs/tenstorrent/METALIUM_SETUP_GUIDE.md for details.")
    message(WARNING "")
    message(WARNING "Falling back to MOCK Metalium APIs (dry-run only).")
    set(USE_REAL_METALIUM OFF CACHE BOOL "Use real TT-Metalium APIs" FORCE)
  endif()
else()
  message(STATUS "Building with MOCK TT-Metalium APIs (dry-run only)")
  message(STATUS "To enable real Metalium: cmake -DUSE_REAL_METALIUM=ON")
  message(STATUS "See docs/tenstorrent/METALIUM_SETUP_GUIDE.md for setup instructions.")
endif()

# Define compile-time macros
set(TILE_LANG_COMPILE_DEFS
  DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
  __STDC_FORMAT_MACROS=1
  PICOJSON_USE_INT64
)

# Set target properties for object library
target_include_directories(tilelang_objs PRIVATE ${TILE_LANG_INCLUDES})
target_compile_definitions(tilelang_objs PRIVATE ${TILE_LANG_COMPILE_DEFS})
target_compile_definitions(tilelang_objs PRIVATE -DTILE_LANG_EXPORTS)

# Shared library
add_library(tilelang SHARED $<TARGET_OBJECTS:tilelang_objs>)
target_link_libraries(tilelang PUBLIC tvm_runtime)

# Link Metalium libraries if available
if(METALIUM_FOUND AND TARGET TT::Metalium)
  target_link_libraries(tilelang PRIVATE TT::Metalium)
  target_link_libraries(tilelang_static PRIVATE TT::Metalium)
  message(STATUS "Linked TileLang against TT::Metalium target")
endif()

# Static library
add_library(tilelang_static STATIC $<TARGET_OBJECTS:tilelang_objs>)
add_dependencies(tilelang_static tvm_runtime)
set_target_properties(tilelang_static PROPERTIES OUTPUT_NAME tilelang)

# Apply static linking flags only to static library to avoid Python extension conflicts
if(TILE_LANG_STATIC_STDCPP AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_link_options(tilelang_static PRIVATE -static-libstdc++ -static-libgcc)
endif()

# Debug build type-specific definitions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(tilelang PRIVATE "TVM_LOG_DEBUG")
  target_compile_definitions(tilelang_objs PRIVATE "TVM_LOG_DEBUG")
  target_compile_definitions(tilelang_static PRIVATE "TVM_LOG_DEBUG")
endif()

# Building tvm_cython modules
# NOTE: tvm_cython target may not exist in all TVM versions (e.g., tilelang_codebase branch)
# Commenting out to allow build to proceed
# if(NOT DEFINED TVM_PREBUILD_PATH)
#   add_dependencies(tilelang tvm_cython)
# endif()

# Module shared library
add_library(tilelang_module SHARED $<TARGET_OBJECTS:tilelang_objs>)
target_link_libraries(tilelang_module PUBLIC tvm)

# Install targets
if(TILE_LANG_INSTALL_STATIC_LIB)
  install(TARGETS tilelang_static tvm_runtime
    LIBRARY DESTINATION lib${LIB_SUFFIX}
  )
else()
  if(DEFINED TVM_PREBUILD_PATH)
    install(TARGETS tilelang tilelang_module
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
  else()
    install(TARGETS tvm_runtime tilelang tilelang_module
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
  endif()
endif()
