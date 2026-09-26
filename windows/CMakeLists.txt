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
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/onnxruntime_download.cmake")
onnxruntime_download("windows-${ORT_TARGET_ARCH}" ORT_ROOT)
set(ORT_LIBRARY_DIR "${ORT_ROOT}/lib")

# Install the upstream license and notices with the application.
install(FILES "${ORT_ROOT}/LICENSE" "${ORT_ROOT}/ThirdPartyNotices.txt"
  DESTINATION "share/onnxruntime" COMPONENT Runtime)

set(onnxruntime_bundled_libraries
  "${ORT_LIBRARY_DIR}/onnxruntime.dll"
  "${ORT_LIBRARY_DIR}/onnxruntime_providers_shared.dll"
  PARENT_SCOPE
)
