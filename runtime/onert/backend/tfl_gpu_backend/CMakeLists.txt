# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(platform ${TARGET_ARCH}.${TARGET_OS})
if (NOT ${platform} STREQUAL "x86_64.linux" AND NOT ${platform} STREQUAL "aarch64.android")
    message(STATUS "TF Lite gpu backend supports x86_64-linux or arm64-android platforms. Build step will be skipped")
    return()
endif()

nnas_find_package(TensorFlowLite2 QUIET)
if(NOT TensorFlowLite2_FOUND)
    message(WARNING "TensorFlow Lite 2 not found. TF Lite GPU backend will be skipped from build process.")
    return()
endif(NOT TensorFlowLite2_FOUND)


list(APPEND tfl_gpu_srcs kernel/operation_specific_traits_provider.cc)
list(APPEND tfl_gpu_srcs kernel/operation_traits.cc)
list(APPEND tfl_gpu_srcs kernel/nnfw_operation_as_tflite_model.cc)
list(APPEND tfl_gpu_srcs TensorBuilder.cc)
list(APPEND tfl_gpu_srcs kernel/Kernel.cc)
list(APPEND tfl_gpu_srcs KernelGenerator.cc)
list(APPEND tfl_gpu_srcs tfl_gpu.cc)
add_library(backend_tfl_gpu SHARED ${tfl_gpu_srcs})
target_link_libraries(backend_tfl_gpu PUBLIC onert_core)
target_link_libraries(backend_tfl_gpu PRIVATE tensorflow-lite2)

add_subdirectory(tests)

install(TARGETS backend_tfl_gpu DESTINATION lib)
install(FILES ${EXT_OVERLAY_DIR}/lib/libtensorflowlite.so DESTINATION lib)
install(FILES ${EXT_OVERLAY_DIR}/lib/libtensorflowlite_gpu_delegate.so DESTINATION lib)
