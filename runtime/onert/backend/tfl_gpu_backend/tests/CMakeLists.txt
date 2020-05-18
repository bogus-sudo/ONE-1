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


list(APPEND tfl_gpu_backend_tests_srcs data_filling.cc)
list(APPEND tfl_gpu_backend_tests_srcs data_comparison.cc)
list(APPEND tfl_gpu_backend_tests_srcs test_models.cc)
list(APPEND tfl_gpu_backend_tests_srcs nnfw_runtime.cc)
if (${platform} STREQUAL "aarch64.android")
    list(APPEND tfl_gpu_backend_tests_srcs tfl_gpu_backend_tests_on_aarch64_android_platform.cc)
elseif(${platform} STREQUAL "x86_64.linux")
    list(APPEND tfl_gpu_backend_tests_srcs tfl_gpu_backend_tests_on_x86_64_linux_platform.cc)
endif()
list(APPEND tfl_gpu_backend_tests_srcs tfl_gpu_backend_tests.cc)
add_executable(tfl_gpu_backend_tests ${tfl_gpu_backend_tests_srcs})
target_link_libraries(tfl_gpu_backend_tests onert_core)
target_link_libraries(tfl_gpu_backend_tests gtest)
target_link_libraries(tfl_gpu_backend_tests gtest_main)
target_link_libraries(tfl_gpu_backend_tests ${LIB_PTHREAD})
target_link_libraries(tfl_gpu_backend_tests ${CMAKE_DL_LIBS})

add_test(tfl_gpu_backend_tests tfl_gpu_backend_tests.cc)

install(TARGETS tfl_gpu_backend_tests DESTINATION unittest)
