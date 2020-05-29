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


list(APPEND nnfw2tflite_tests_srcs test_models.cc)
list(APPEND nnfw2tflite_tests_srcs nnfw2tflite_tests.cc)
list(APPEND nnfw2tflite_tests_srcs program_options_tests.cc)
list(APPEND nnfw2tflite_tests_srcs ../program_options.cc)
list(APPEND nnfw2tflite_tests_srcs nnfw2tflite_converter_test.cc)
if (${TARGET_ARCH} STREQUAL "aarch64")
    list(APPEND nnfw2tflite_tests_srcs nnfw2tflite_converter_test_env_for_aarch64.cc)
elseif(${TARGET_ARCH} STREQUAL "x86_64")
    list(APPEND nnfw2tflite_tests_srcs nnfw2tflite_converter_test_env_for_x86_64.cc)
endif()
list(APPEND nnfw2tflite_tests_srcs ../data_filling.cc)
list(APPEND nnfw2tflite_tests_srcs ../data_comparison.cc)
list(APPEND nnfw2tflite_tests_srcs ../nnfw_runtime.cc)
if (${TARGET_ARCH} STREQUAL "aarch64")
    list(APPEND nnfw2tflite_tests_srcs ../nnfw_runtime_model_loading_on_aarch64_android.cc)
elseif (${TARGET_ARCH} STREQUAL "x86_64")
    list(APPEND nnfw2tflite_tests_srcs ../nnfw_runtime_model_loading_on_x86_64.cc)
endif()
list(APPEND nnfw2tflite_tests_srcs ../tflite_runtime.cc)
if (${TARGET_ARCH} STREQUAL "aarch64")
    list(APPEND nnfw2tflite_tests_srcs ../tflite_runtime_model_loading_on_aarch64_android.cc)
elseif (${TARGET_ARCH} STREQUAL "x86_64")
    list(APPEND nnfw2tflite_tests_srcs ../tflite_runtime_model_loading_on_x86_64.cc)
endif()
list(APPEND nnfw2tflite_tests_srcs ../working_with_nnfw_ir.cc)
list(APPEND nnfw2tflite_tests_srcs ../nnfw_operation_as_tflite_model.cc)
list(APPEND nnfw2tflite_tests_srcs ../operation_traits.cc)
list(APPEND nnfw2tflite_tests_srcs working_with_nnfw_ir_tests.cc)
add_executable(nnfw2tflite_tests ${nnfw2tflite_tests_srcs})
target_include_directories(nnfw2tflite_tests PRIVATE "../")
target_link_libraries(nnfw2tflite_tests onert_core tflite_loader)
target_link_libraries(nnfw2tflite_tests tensorflow-lite2)
target_link_libraries(nnfw2tflite_tests gtest)
target_link_libraries(nnfw2tflite_tests gtest_main)
target_link_libraries(nnfw2tflite_tests ${LIB_PTHREAD})
target_link_libraries(nnfw2tflite_tests ${CMAKE_DL_LIBS})

add_test(nnfw2tflite_tests nnfw2tflite_tests.cc)

install(TARGETS nnfw2tflite_tests DESTINATION unittest)
