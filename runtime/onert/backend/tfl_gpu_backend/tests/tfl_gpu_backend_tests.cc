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

#include "tfl_gpu_backend_tests.h"

#include "test_models.h"
#include "nnfw_runtime.h"
#include "data_filling.h"
#include "data_comparison.h"


TEST_F(TflGpuBackendTest, model_with_just_one_Conv2D_can_be_evalueated_on_tfl_gpu_bakend) {
  auto nnfw_ir = makeModelWithJustOneConvolution2DOperation();

  NNFWRuntime nnfw_runtime1;
  nnfw_runtime1.doCalculationsUsingCpuBackend();
  nnfw_runtime1.loadGraph(nnfw_ir);
  NNFWRuntime nnfw_runtime2;
  nnfw_runtime2.doCalculationsUsingTflGpuBackend();
  nnfw_runtime2.loadGraph(nnfw_ir);

  for (size_t i = 0; i < std::min(nnfw_runtime1.numberOfInputs(), nnfw_runtime2.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime1.sizeOfInput(i), nnfw_runtime2.sizeOfInput(i)));
    nnfw_runtime1.setDataForInput(i, data);
    nnfw_runtime2.setDataForInput(i, data);
  }

  nnfw_runtime1.evaluate();
  nnfw_runtime2.evaluate();

  ASSERT_TRUE(nnfw_runtime1.numberOfInputs() == nnfw_runtime2.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime1.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime1.sizeOfInput(i) == nnfw_runtime2.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime1.numberOfOutputs() == nnfw_runtime2.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime1.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime1.getDataOfOutput(i), nnfw_runtime2.getDataOfOutput(i), 10e-5));
  }
}



int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}