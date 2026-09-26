cmake_minimum_required(VERSION 3.10)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/onnxruntime_download.cmake")
onnxruntime_download("${ORT_TARGET}" runtime)
file(WRITE "${ORT_OUTPUT}" "${runtime}/lib")
