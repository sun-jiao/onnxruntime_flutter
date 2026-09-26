# The Flutter tooling requires that developers have a version of Visual Studio
# installed that includes CMake 3.14 or later. You should not increase this
# version, as doing so will cause the plugin to fail to compile for some
# customers of the plugin.
cmake_minimum_required(VERSION 3.14)

# Project-level configuration.
set(PROJECT_NAME "onnxruntime")
project(${PROJECT_NAME} LANGUAGES CXX)

# Select the target architecture rather than the build host architecture.
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/onnxruntime_arch.cmake")
onnxruntime_target_arch(windows ORT_TARGET_ARCH)
set(ORT_LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
if(ORT_TARGET_ARCH STREQUAL "arm64")
  set(ORT_LIBRARY_DIR "${ORT_LIBRARY_DIR}/arm64")
endif()

set(onnxruntime_bundled_libraries
  "${ORT_LIBRARY_DIR}/onnxruntime.dll"
  PARENT_SCOPE
)
