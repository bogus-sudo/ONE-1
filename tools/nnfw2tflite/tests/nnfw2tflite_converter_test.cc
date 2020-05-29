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

#include "nnfw2tflite_converter_test.h"

#include "test_models.h"
#include "working_with_nnfw_ir.h"
#include "nnfw_runtime.h"
#include "tflite_runtime.h"

#include "data_filling.h"
#include "data_comparison.h"


TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_relu_operation) {
  auto nnfw_ir = makeModelWithJustOneReluOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_and_executed_on_GPU_in_the_same_manner_relu_operation) {
  auto nnfw_ir = makeModelWithJustOneReluOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphOnGpuFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}


TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_relu6_operation) {
  auto nnfw_ir = makeModelWithJustOneRelu6Operation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_and_executed_on_GPU_in_the_same_manner_relu6_operation) {
  auto nnfw_ir = makeModelWithJustOneRelu6Operation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.runAllOperationsOnAclClBackend();
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphOnGpuFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_add_operation) {
  auto nnfw_ir = makeModelWithJustOneAddOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_and_executed_on_GPU_in_the_same_manner_add_operation) {
  auto nnfw_ir = makeModelWithJustOneAddOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.runAllOperationsOnAclClBackend();
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphOnGpuFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_convolution2d_operation) {
  auto nnfw_ir = makeModelWithJustOneConvolution2DOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_and_executed_on_GPU_in_the_same_manner_Conv2D_operation) {
  auto nnfw_ir = makeModelWithJustOneConvolution2DOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.runAllOperationsOnAclClBackend();
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphOnGpuFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
//    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    auto data = Ones(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_DepthwiseConv2D_operation) {
  auto nnfw_ir = makeModelWithJustOneDepthwiseConv2dOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_and_executed_on_GPU_in_the_same_manner_DepthwiseConv2D_operation) {
  auto nnfw_ir = makeModelWithJustOneDepthwiseConv2dOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.runAllOperationsOnAclClBackend();
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphOnGpuFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_AvgPool2D_operation) {
  auto nnfw_ir = makeModelWithJustOneAveragePool2dOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_and_executed_on_GPU_in_the_same_manner_AvgPool2D_operation) {
  auto nnfw_ir = makeModelWithJustOneAveragePool2dOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.runAllOperationsOnAclClBackend();
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphOnGpuFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_Squeeze_operation) {
  auto nnfw_ir = makeModelWithJustOneSqueezeOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_and_executed_on_GPU_in_the_same_manner_Squeeze_operation) {
  auto nnfw_ir = makeModelWithJustOneSqueezeOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.runAllOperationsOnAclClBackend();
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphOnGpuFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_Softmax_operation) {
  auto nnfw_ir = makeModelWithJustOneSoftmaxOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}

TEST_F(NNFWOperationToTFLiteGraphConvertationTest, can_be_converted_and_executed_on_GPU_in_the_same_manner_Softmax_operation) {
  auto nnfw_ir = makeModelWithJustOneSoftmaxOperation();
  auto traits_of_operations = generateTraitsOfOperationsFrom(nnfw_ir);

  NNFWRuntime nnfw_runtime;
  nnfw_runtime.runAllOperationsOnAclClBackend();
  nnfw_runtime.loadGraph(nnfw_ir);
  TFLiteRuntime tflite_runtime;
  tflite_runtime.makeGraphOnGpuFrom(traits_of_operations[0]);

  for (size_t i = 0; i < std::min(nnfw_runtime.numberOfInputs(), tflite_runtime.numberOfInputs()); ++i) {
    auto data = randomData(std::min(nnfw_runtime.sizeOfInput(i), tflite_runtime.sizeOfInput(i)));
    nnfw_runtime.setDataForInput(i, data);
    tflite_runtime.setDataForInput(i, data);
  }

  nnfw_runtime.evaluate();
  tflite_runtime.evaluate();

  ASSERT_TRUE(nnfw_runtime.numberOfInputs() == tflite_runtime.numberOfInputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
    ASSERT_TRUE(nnfw_runtime.sizeOfInput(i) == tflite_runtime.sizeOfInput(i));
  }
  ASSERT_TRUE(nnfw_runtime.numberOfOutputs() == tflite_runtime.numberOfOutputs());
  for (size_t i = 0; i < nnfw_runtime.numberOfOutputs(); ++i) {
    ASSERT_TRUE(isAlmostEqual(nnfw_runtime.getDataOfOutput(i), tflite_runtime.getDataOfOutput(i), 10e-5));
  }
}
