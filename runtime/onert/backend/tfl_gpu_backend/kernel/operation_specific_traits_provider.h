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

#ifndef NNFW_TFL_GPU_OPERATION_SPECIFIC_TRAITS_PROVIDER_H
#define NNFW_TFL_GPU_OPERATION_SPECIFIC_TRAITS_PROVIDER_H

#include "tensorflow/lite/schema/schema_generated.h"
#include "ir/operation/Add.h"
#include "ir/operation/Conv2D.h"
#include "ir/operation/DepthwiseConv2D.h"
#include "ir/operation/AvgPool2D.h"
#include "ir/operation/Squeeze.h"
#include "ir/operation/Softmax.h"


class OperationSpecificTraitsProvider {
public:
  virtual tflite::BuiltinOperator operationCode() const = 0;
  virtual tflite::BuiltinOptions operationOptionsCode() const = 0;
  virtual flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder&) const = 0;

  virtual ~OperationSpecificTraitsProvider() = default;
};

class ReluSpecificTraitsProvider: public OperationSpecificTraitsProvider
{
public:
  tflite::BuiltinOperator operationCode() const final;
  tflite::BuiltinOptions operationOptionsCode() const final;
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder&) const final;
};

class Relu6SpecificTraitsProvider: public OperationSpecificTraitsProvider
{
public:
  tflite::BuiltinOperator operationCode() const final;
  tflite::BuiltinOptions operationOptionsCode() const final;
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder&) const final;
};

class AddSpecificTraitsProvider: public OperationSpecificTraitsProvider
{
public:
  AddSpecificTraitsProvider(const onert::ir::operation::Add::Param& nnfw_add_params);
  tflite::BuiltinOperator operationCode() const final;
  tflite::BuiltinOptions operationOptionsCode() const final;
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const final;
private:
  tflite::ActivationFunctionType fused_activation_type = tflite::ActivationFunctionType_NONE;
};

class Conv2dSpecificTraitsProvider: public OperationSpecificTraitsProvider
{
public:
  Conv2dSpecificTraitsProvider(const onert::ir::operation::Conv2D::Param& nnfw_conv2d_params);
  tflite::BuiltinOperator operationCode() const final;
  tflite::BuiltinOptions operationOptionsCode() const final;
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const final;

private:
  uint32_t vertical_stride = 1;
  uint32_t horisontal_stride = 1;
  tflite::Padding padding = tflite::Padding::Padding_SAME;
  tflite::ActivationFunctionType fused_activation_type = tflite::ActivationFunctionType_NONE;
};

class DepthwiseConv2dSpecificTraitsProvider: public OperationSpecificTraitsProvider
{
public:
  DepthwiseConv2dSpecificTraitsProvider(const onert::ir::operation::DepthwiseConv2D::Param& nnfw_depthwise_conv2d_params);
  tflite::BuiltinOperator operationCode() const final;
  tflite::BuiltinOptions operationOptionsCode() const final;
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const final;

private:
  uint32_t vertical_stride = 1;
  uint32_t horisontal_stride = 1;
  uint32_t multiplier = 1;
  tflite::Padding padding = tflite::Padding::Padding_SAME;
  tflite::ActivationFunctionType fused_activation_type = tflite::ActivationFunctionType_NONE;
};

class AvgPool2dSpecificTraitsProvider: public OperationSpecificTraitsProvider
{
public:
  AvgPool2dSpecificTraitsProvider(const onert::ir::operation::AvgPool2D::Param& nnfw_avg_pool2d_params);
  tflite::BuiltinOperator operationCode() const final;
  tflite::BuiltinOptions operationOptionsCode() const final;
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const final;

private:
  uint32_t vertical_stride = 1;
  uint32_t horisontal_stride = 1;
  uint32_t filter_height = 1;
  uint32_t filter_width = 1;
  tflite::Padding padding = tflite::Padding::Padding_SAME;
  tflite::ActivationFunctionType fused_activation_type = tflite::ActivationFunctionType_NONE;
};

class SqueezeSpecificTraitsProvider: public OperationSpecificTraitsProvider
{
public:
  SqueezeSpecificTraitsProvider(const onert::ir::operation::Squeeze::Param& nnfw_squeeze_params);
  tflite::BuiltinOperator operationCode() const final;
  tflite::BuiltinOptions operationOptionsCode() const final;
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const final;

private:
  std::vector<int32_t> dimensions;
};

class SoftmaxSpecificTraitsProvider: public OperationSpecificTraitsProvider
{
public:
  SoftmaxSpecificTraitsProvider(const onert::ir::operation::Softmax::Param& nnfw_softmax_params);
  tflite::BuiltinOperator operationCode() const final;
  tflite::BuiltinOptions operationOptionsCode() const final;
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const final;
private:
  float beta = 1.f;
};

#endif // NNFW_TFL_GPU_OPERATION_SPECIFIC_TRAITS_PROVIDER_H
