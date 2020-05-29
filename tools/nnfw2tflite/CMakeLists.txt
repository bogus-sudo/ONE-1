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

nnas_find_package(TensorFlowLite2 QUIET)
if(NOT TensorFlowLite2_FOUND)
  message(WARNING "TensorFlow Lite 2 not found. nnfw2tflite will be skipped from build process.")
  return()
endif(NOT TensorFlowLite2_FOUND)

if (NOT ${TARGET_ARCH} STREQUAL "x86_64" AND NOT ${TARGET_OS} STREQUAL "linux" AND NOT ${TARGET_ARCH} STREQUAL "aarch64" AND NOT ${TARGET_OS} STREQUAL "android")
  message(WARNING "nnfw2tflite supports x86_64-linux or arm64-android platforms.")
  return()
endif()

add_subdirectory(tests)

list(APPEND nnfw2tflite_srcs nnfw2tflite.cc)
list(APPEND nnfw2tflite_srcs program_options.cc)
list(APPEND nnfw2tflite_srcs data_filling.cc)
list(APPEND nnfw2tflite_srcs data_comparison.cc)
list(APPEND nnfw2tflite_srcs nnfw_runtime.cc)
if (${TARGET_ARCH} STREQUAL "aarch64")
  list(APPEND nnfw2tflite_srcs nnfw_runtime_model_loading_on_aarch64_android.cc)
elseif (${TARGET_ARCH} STREQUAL "x86_64")
  list(APPEND nnfw2tflite_srcs nnfw_runtime_model_loading_on_x86_64.cc)
endif()
list(APPEND nnfw2tflite_srcs tflite_runtime.cc)
if (${TARGET_ARCH} STREQUAL "aarch64")
  list(APPEND nnfw2tflite_srcs tflite_runtime_model_loading_on_aarch64_android.cc)
elseif (${TARGET_ARCH} STREQUAL "x86_64")
  list(APPEND nnfw2tflite_srcs tflite_runtime_model_loading_on_x86_64.cc)
endif()
list(APPEND nnfw2tflite_srcs nnfw_operation_as_tflite_model.cc)
list(APPEND nnfw2tflite_srcs operation_traits.cc)
list(APPEND nnfw2tflite_srcs working_with_nnfw_ir.cc)
add_executable(nnfw2tflite ${nnfw2tflite_srcs})
target_link_libraries(nnfw2tflite onert_core tflite_loader)
target_link_libraries(nnfw2tflite tensorflow-lite2)


install(TARGETS nnfw2tflite DESTINATION bin)
install(FILES ${EXT_OVERLAY_DIR}/lib/libtensorflowlite.so DESTINATION lib)
install(FILES ${EXT_OVERLAY_DIR}/lib/libtensorflowlite_gpu_delegate.so DESTINATION lib)
if (${TARGET_ARCH} STREQUAL "aarch64" AND  ${TARGET_OS} STREQUAL "android")
  install(FILES ${EXT_OVERLAY_DIR}/lib/libhexagon_interface.so DESTINATION lib)
endif()