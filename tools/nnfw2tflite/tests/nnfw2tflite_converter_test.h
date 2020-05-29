/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NNFW_NNFW2TFLITE_CONVERTER_TEST_H
#define NNFW_NNFW2TFLITE_CONVERTER_TEST_H

#include "gtest/gtest.h"


struct NNFWOperationToTFLiteGraphConvertationTest: public testing::Test {
  void SetUp() override;

  void TearDown() override {
    if (old_value_of_BACKENDS_env_var) {
      setenv("BACKENDS", old_value_of_BACKENDS_env_var, 1);
    }
    else {
      unsetenv("BACKENDS");
    }

    if(old_value_of_EXECUTOR_env_var) {
      setenv("EXECUTOR", old_value_of_EXECUTOR_env_var, 1);
    }
    else {
      unsetenv("EXECUTOR");
    }
  }

private:
  const char* old_value_of_BACKENDS_env_var = nullptr;
  const char* old_value_of_EXECUTOR_env_var = nullptr;
};

#endif // NNFW_NNFW2TFLITE_CONVERTER_TEST_H
