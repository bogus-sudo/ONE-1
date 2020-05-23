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

#include "test_models.h"

#include "data_filling.h"

#include "ir/Graph.h"
#include "ir/Shape.h"
#include "ir/operation/ReLU.h"
#include "ir/operation/ReLU6.h"
#include "ir/operation/Add.h"
#include "ir/operation/Conv2D.h"
#include "ir/operation/DepthwiseConv2D.h"
#include "ir/operation/AvgPool2D.h"
#include "ir/operation/Squeeze.h"
#include "ir/operation/Softmax.h"


std::shared_ptr<onert::ir::Graph> makeModelWithJustOneReluOperation() {
  using OIS = onert::ir::OperandIndexSequence;

  auto graph = std::make_shared<onert::ir::Graph>();
  const onert::ir::TypeInfo float_type(onert::ir::DataType::FLOAT32);
  auto in_a = graph->addOperand(onert::ir::Shape{1,2,2,2}, float_type);
  auto out_a = graph->addOperand(onert::ir::Shape{1,2,2,2}, float_type);
  graph->addOperation(std::make_unique<onert::ir::operation::ReLU>(OIS{in_a}, OIS{out_a}));
  graph->addInput(in_a);
  graph->addOutput(out_a);
  graph->finishBuilding();

  return graph;
}

std::shared_ptr<onert::ir::Graph> makeModelWithJustOneRelu6Operation() {
  using OIS = onert::ir::OperandIndexSequence;

  auto graph = std::make_shared<onert::ir::Graph>();
  const onert::ir::TypeInfo float_type(onert::ir::DataType::FLOAT32);

  auto in = graph->addOperand(onert::ir::Shape{1, 2, 2, 2}, float_type);
  auto out = graph->addOperand(onert::ir::Shape{1, 2, 2, 2}, float_type);

  graph->addOperation(std::make_unique<onert::ir::operation::ReLU6>(OIS{in}, OIS{out}));
  graph->addInput(in);
  graph->addOutput(out);

  graph->finishBuilding();

  return graph;
}

std::shared_ptr<onert::ir::Graph> makeModelWithJustOneAddOperation() {
  using OIS = onert::ir::OperandIndexSequence;

  auto graph = std::make_shared<onert::ir::Graph>();
  const onert::ir::TypeInfo float_type(onert::ir::DataType::FLOAT32);

  auto in1 = graph->addOperand(onert::ir::Shape{1, 2, 2, 2}, float_type);
  auto in2 = graph->addOperand(onert::ir::Shape{1, 2, 2, 2}, float_type);
  auto out = graph->addOperand(onert::ir::Shape{1, 2, 2, 2}, float_type);

  onert::ir::operation::Add::Param param;
  param.activation = onert::ir::Activation::NONE;
  graph->addOperation(std::make_unique<onert::ir::operation::Add>(OIS{in1,in2}, OIS{out}, param));
  graph->addInput(in1);
  graph->addInput(in2);
  graph->addOutput(out);

  graph->finishBuilding();

  return graph;
}


std::shared_ptr<onert::ir::Graph> makeModelWithJustOneConvolution2DOperation() {
  using OIS = onert::ir::OperandIndexSequence;

  auto graph = std::make_shared<onert::ir::Graph>();
  const onert::ir::TypeInfo float_type(onert::ir::DataType::FLOAT32);
  auto in = graph->addOperand(onert::ir::Shape{1, 56, 56, 128}, float_type);
  auto kernel = graph->addOperand(onert::ir::Shape{128, 1, 1, 128}, float_type);
  auto bias = graph->addOperand(onert::ir::Shape{128}, float_type);
  auto out = graph->addOperand(onert::ir::Shape{1, 56, 56, 128}, float_type);

  onert::ir::operation::Conv2D::Param param;
  param.padding.type = onert::ir::PaddingType::SAME;
  param.stride.vertical = 1;
  param.stride.horizontal = 1;
  param.activation = onert::ir::Activation::RELU6;

  graph->addOperation(std::make_unique<onert::ir::operation::Conv2D>(OIS{in, kernel, bias}, OIS{out}, param));
  graph->addInput(in);
  graph->addOutput(out);

  auto kernel_data = randomData(graph->operands().at(kernel).shape().num_elements());
  graph->setOperandValue(kernel, std::make_shared<onert::ir::CachedData>(reinterpret_cast<uint8_t*>(kernel_data.data()), kernel_data.size() * sizeof(float)));
  auto bias_data = randomData(graph->operands().at(bias).shape().num_elements());
  graph->setOperandValue(bias, std::make_shared<onert::ir::CachedData>(reinterpret_cast<uint8_t*>(bias_data.data()), bias_data.size() * sizeof(float)));
  graph->finishBuilding();

  return graph;
}

std::shared_ptr<onert::ir::Graph> makeModelWithJustOneDepthwiseConv2dOperation() {
  using OIS = onert::ir::OperandIndexSequence;

  auto graph = std::make_shared<onert::ir::Graph>();
  const onert::ir::TypeInfo float_type(onert::ir::DataType::FLOAT32);

  auto in = graph->addOperand(onert::ir::Shape{1, 112, 112, 32}, float_type);
  auto kernel = graph->addOperand(onert::ir::Shape{1, 3, 3, 32}, float_type);
  auto bias = graph->addOperand(onert::ir::Shape{32}, float_type);
  auto out = graph->addOperand(onert::ir::Shape{1, 112, 112, 32}, float_type);

  onert::ir::operation::DepthwiseConv2D::Param param;
  param.padding.type = onert::ir::PaddingType::SAME;
  param.stride.vertical = 1;
  param.stride.horizontal = 1;
  param.activation = onert::ir::Activation::RELU6;
  param.multiplier = 1;

  graph->addOperation(std::make_unique<onert::ir::operation::DepthwiseConv2D>(OIS{in, kernel, bias}, OIS{out}, param));
  graph->addInput(in);
  graph->addOutput(out);

  auto kernel_data = randomData(graph->operands().at(kernel).shape().num_elements());
  graph->setOperandValue(kernel, std::make_shared<onert::ir::CachedData>(reinterpret_cast<uint8_t*>(kernel_data.data()), kernel_data.size() * sizeof(float)));
  auto bias_data = randomData(graph->operands().at(bias).shape().num_elements());
  graph->setOperandValue(bias, std::make_shared<onert::ir::CachedData>(reinterpret_cast<uint8_t*>(bias_data.data()), bias_data.size() * sizeof(float)));
  graph->finishBuilding();

  return graph;
}

std::shared_ptr<onert::ir::Graph> makeModelWithJustOneAveragePool2dOperation() {
  using OIS = onert::ir::OperandIndexSequence;

  auto graph = std::make_shared<onert::ir::Graph>();
  const onert::ir::TypeInfo float_type(onert::ir::DataType::FLOAT32);

  auto in = graph->addOperand(onert::ir::Shape{1, 7, 7, 1024}, float_type);
  auto out = graph->addOperand(onert::ir::Shape{1, 1, 1, 1024}, float_type);

  onert::ir::operation::AvgPool2D::Param param;
  param.padding.type = onert::ir::PaddingType::VALID;
  param.stride.vertical = 2;
  param.stride.horizontal = 2;
  param.activation = onert::ir::Activation::NONE;
  param.kh = 7;
  param.kw = 7;

  graph->addOperation(std::make_unique<onert::ir::operation::AvgPool2D>(OIS{in}, OIS{out}, param));
  graph->addInput(in);
  graph->addOutput(out);

  graph->finishBuilding();

  return graph;
}

std::shared_ptr<onert::ir::Graph> makeModelWithJustOneSqueezeOperation() {
  using OIS = onert::ir::OperandIndexSequence;

  auto graph = std::make_shared<onert::ir::Graph>();
  const onert::ir::TypeInfo float_type(onert::ir::DataType::FLOAT32);

  auto in = graph->addOperand(onert::ir::Shape{1, 1, 1, 1024}, float_type);
  auto out = graph->addOperand(onert::ir::Shape{1, 1024}, float_type);

  onert::ir::operation::Squeeze::Param param;
  param.dims[0] = 1;
  param.dims[1] = 2;
  param.ndim = 2;

  graph->addOperation(std::make_unique<onert::ir::operation::Squeeze>(OIS{in}, OIS{out}, param));
  graph->addInput(in);
  graph->addOutput(out);

  graph->finishBuilding();

  return graph;
}

std::shared_ptr<onert::ir::Graph> makeModelWithJustOneSoftmaxOperation() {
  using OIS = onert::ir::OperandIndexSequence;

  auto graph = std::make_shared<onert::ir::Graph>();
  const onert::ir::TypeInfo float_type(onert::ir::DataType::FLOAT32);

  auto in = graph->addOperand(onert::ir::Shape{1, 1001}, float_type);
  auto out = graph->addOperand(onert::ir::Shape{1, 1001}, float_type);

  onert::ir::operation::Softmax::Param param;
  param.beta = 1.f;

  graph->addOperation(std::make_unique<onert::ir::operation::Softmax>(OIS{in}, OIS{out}, param));
  graph->addInput(in);
  graph->addOutput(out);

  graph->finishBuilding();

  return graph;
}

std::shared_ptr<onert::ir::Graph> makeModelWithTwoRhombsWithOneCommonEdge() {
  using OIS = onert::ir::OperandIndexSequence;

  auto graph = std::make_shared<onert::ir::Graph>();
  const onert::ir::TypeInfo float_type(onert::ir::DataType::FLOAT32);
  auto in_a = graph->addOperand(onert::ir::Shape{1,10,10,10}, float_type);
  auto out_a = graph->addOperand(onert::ir::Shape{1,10,10,10}, float_type);
  graph->addOperation(std::make_unique<onert::ir::operation::ReLU>(OIS{in_a}, OIS{out_a}));
  auto out_b = graph->addOperand(onert::ir::Shape{1,10,10,10}, float_type);
  graph->addOperation(std::make_unique<onert::ir::operation::ReLU6>(OIS{out_a}, OIS{out_b}));
  auto out_c = graph->addOperand(onert::ir::Shape{1,10,10,10}, float_type);
  graph->addOperation(std::make_unique<onert::ir::operation::ReLU6>(OIS{out_b}, OIS{out_c}));
  auto out_d = graph->addOperand(onert::ir::Shape{1,10,10,10}, float_type);
  graph->addOperation(std::make_unique<onert::ir::operation::ReLU>(OIS{out_a}, OIS{out_d}));
  auto out_e = graph->addOperand(onert::ir::Shape{1,10,10,10}, float_type);
  onert::ir::operation::Add::Param e_param{onert::ir::Activation::NONE};
  graph->addOperation(std::make_unique<onert::ir::operation::Add>(OIS{out_b, out_d}, OIS{out_e}, e_param));
  auto out_f = graph->addOperand(onert::ir::Shape{1,10,10,10}, float_type);
  onert::ir::operation::Add::Param f_param{onert::ir::Activation::NONE};
  graph->addOperation(std::make_unique<onert::ir::operation::Add>(OIS{out_c, out_e}, OIS{out_f}, f_param));

  graph->addInput(in_a);
  graph->addOutput(out_f);
  graph->finishBuilding();

  return graph;
}

std::shared_ptr<onert::ir::Graph> makeConv2dDepthwiseConv2dConv2dOperationsSequence() {
  using OIS = onert::ir::OperandIndexSequence;

  auto graph = std::make_shared<onert::ir::Graph>();
  const onert::ir::TypeInfo float_type(onert::ir::DataType::FLOAT32);

  auto in1 = graph->addOperand(onert::ir::Shape{1, 224, 224, 3}, float_type);
  auto kernel1 = graph->addOperand(onert::ir::Shape{32, 3, 3, 3}, float_type);
  auto bias1 = graph->addOperand(onert::ir::Shape{32}, float_type);
  auto out1 = graph->addOperand(onert::ir::Shape{1, 112, 112, 32}, float_type);
  onert::ir::operation::Conv2D::Param param1;
  param1.padding.type = onert::ir::PaddingType::SAME;
  param1.stride.vertical = 2;
  param1.stride.horizontal = 2;
  param1.activation = onert::ir::Activation::RELU6;
  graph->addOperation(std::make_unique<onert::ir::operation::Conv2D>(OIS{in1, kernel1, bias1}, OIS{out1}, param1));

  auto kernel2 = graph->addOperand(onert::ir::Shape{1, 3, 3, 32}, float_type);
  auto bias2 = graph->addOperand(onert::ir::Shape{32}, float_type);
  auto out2 = graph->addOperand(onert::ir::Shape{1, 112, 112, 32}, float_type);
  onert::ir::operation::DepthwiseConv2D::Param param2;
  param2.padding.type = onert::ir::PaddingType::SAME;
  param2.stride.vertical = 1;
  param2.stride.horizontal = 1;
  param2.activation = onert::ir::Activation::RELU6;
  param2.multiplier = 1;
  graph->addOperation(std::make_unique<onert::ir::operation::DepthwiseConv2D>(OIS{out1, kernel2, bias2}, OIS{out2}, param2));

  auto kernel3 = graph->addOperand(onert::ir::Shape{64, 1, 1, 32}, float_type);
  auto bias3 = graph->addOperand(onert::ir::Shape{64}, float_type);
  auto out3 = graph->addOperand(onert::ir::Shape{1, 112, 112, 64}, float_type);
  onert::ir::operation::Conv2D::Param param3;
  param3.padding.type = onert::ir::PaddingType::SAME;
  param3.stride.vertical = 1;
  param3.stride.horizontal = 1;
  param3.activation = onert::ir::Activation::RELU6;
  graph->addOperation(std::make_unique<onert::ir::operation::Conv2D>(OIS{out2, kernel3, bias3}, OIS{out3}, param3));

  graph->addInput(in1);
  graph->addOutput(out3);

  auto kernel1_data = randomData(graph->operands().at(kernel1).shape().num_elements());
  graph->setOperandValue(kernel1, std::make_shared<onert::ir::CachedData>(reinterpret_cast<uint8_t*>(kernel1_data.data()), kernel1_data.size() * sizeof(float)));
  auto bias1_data = randomData(graph->operands().at(bias1).shape().num_elements());
  graph->setOperandValue(bias1, std::make_shared<onert::ir::CachedData>(reinterpret_cast<uint8_t*>(bias1_data.data()), bias1_data.size() * sizeof(float)));
  auto kernel2_data = randomData(graph->operands().at(kernel2).shape().num_elements());
  graph->setOperandValue(kernel2, std::make_shared<onert::ir::CachedData>(reinterpret_cast<uint8_t*>(kernel2_data.data()), kernel2_data.size() * sizeof(float)));
  auto bias2_data = randomData(graph->operands().at(bias2).shape().num_elements());
  graph->setOperandValue(bias2, std::make_shared<onert::ir::CachedData>(reinterpret_cast<uint8_t*>(bias2_data.data()), bias2_data.size() * sizeof(float)));
  auto kernel3_data = randomData(graph->operands().at(kernel3).shape().num_elements());
  graph->setOperandValue(kernel3, std::make_shared<onert::ir::CachedData>(reinterpret_cast<uint8_t*>(kernel3_data.data()), kernel3_data.size() * sizeof(float)));
  auto bias3_data = randomData(graph->operands().at(bias3).shape().num_elements());
  graph->setOperandValue(bias3, std::make_shared<onert::ir::CachedData>(reinterpret_cast<uint8_t*>(bias3_data.data()), bias3_data.size() * sizeof(float)));
  graph->finishBuilding();

  return graph;
}

