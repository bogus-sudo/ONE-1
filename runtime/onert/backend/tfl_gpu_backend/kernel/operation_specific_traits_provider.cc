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

#include "operation_specific_traits_provider.h"


tflite::BuiltinOperator ReluSpecificTraitsProvider::operationCode() const { return tflite::BuiltinOperator_RELU; }
tflite::BuiltinOptions ReluSpecificTraitsProvider::operationOptionsCode() const { return tflite::BuiltinOptions_NONE; }
flatbuffers::Offset<void> ReluSpecificTraitsProvider::serializedOperationOptions(flatbuffers::FlatBufferBuilder&) const { return 0; }


tflite::BuiltinOperator Relu6SpecificTraitsProvider::operationCode() const { return tflite::BuiltinOperator_RELU6; }
tflite::BuiltinOptions Relu6SpecificTraitsProvider::operationOptionsCode() const { return tflite::BuiltinOptions_NONE; }
flatbuffers::Offset<void> Relu6SpecificTraitsProvider::serializedOperationOptions(flatbuffers::FlatBufferBuilder&) const { return 0; }


AddSpecificTraitsProvider::AddSpecificTraitsProvider(const onert::ir::operation::Add::Param& nnfw_add_params) {
  if (nnfw_add_params.activation == onert::ir::Activation::NONE) {
    fused_activation_type = tflite::ActivationFunctionType_NONE;
  }
  else if (nnfw_add_params.activation == onert::ir::Activation::RELU) {
    fused_activation_type = tflite::ActivationFunctionType_RELU;
  }
  else if (nnfw_add_params.activation == onert::ir::Activation::RELU6) {
    fused_activation_type = tflite::ActivationFunctionType_RELU6;
  }
  else {
    throw std::logic_error("activation type for operation Add in NNFW IR cannot be converted to activation type in TF Lite");
  }
}

tflite::BuiltinOperator AddSpecificTraitsProvider::operationCode() const { return tflite::BuiltinOperator_ADD; }
tflite::BuiltinOptions AddSpecificTraitsProvider::operationOptionsCode() const { return tflite::BuiltinOptions_AddOptions; }
flatbuffers::Offset<void> AddSpecificTraitsProvider::serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const {
  return
      tflite::CreateAddOptions(
          flat_buffer_builder,
          fused_activation_type).Union();
}


Conv2dSpecificTraitsProvider::Conv2dSpecificTraitsProvider(const onert::ir::operation::Conv2D::Param& nnfw_conv2d_params) {
  vertical_stride = nnfw_conv2d_params.stride.vertical;
  horisontal_stride = nnfw_conv2d_params.stride.horizontal;
  if (nnfw_conv2d_params.padding.type == onert::ir::PaddingType::SAME) {
    padding = tflite::Padding_SAME;
  }
  else if (nnfw_conv2d_params.padding.type == onert::ir::PaddingType::VALID) {
    padding = tflite::Padding_VALID;
  }
  else {
    throw std::logic_error("padding type in NNFW IR cannot be converted to TF Lite padding type in Conv2D operation");
  }

  if (nnfw_conv2d_params.activation == onert::ir::Activation::NONE) {
    fused_activation_type = tflite::ActivationFunctionType_NONE;
  }
  else if (nnfw_conv2d_params.activation == onert::ir::Activation::RELU) {
    fused_activation_type = tflite::ActivationFunctionType_RELU;
  }
  else if (nnfw_conv2d_params.activation == onert::ir::Activation::RELU6) {
    fused_activation_type = tflite::ActivationFunctionType_RELU6;
  }
  else {
    throw std::logic_error("Activation function type in NNFW IR cannot be converted to TF Lite activation function type type in Conv2D operation");
  }
}

tflite::BuiltinOperator Conv2dSpecificTraitsProvider::operationCode() const { return tflite::BuiltinOperator_CONV_2D; }
tflite::BuiltinOptions Conv2dSpecificTraitsProvider::operationOptionsCode() const { return tflite::BuiltinOptions_Conv2DOptions; }

flatbuffers::Offset<void> Conv2dSpecificTraitsProvider::serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const {
  return
      tflite::CreateConv2DOptions(
          flat_buffer_builder,
          padding,
          vertical_stride,
          horisontal_stride,
          fused_activation_type).Union();
}

DepthwiseConv2dSpecificTraitsProvider::DepthwiseConv2dSpecificTraitsProvider(const onert::ir::operation::DepthwiseConv2D::Param& nnfw_depthwise_conv2d_params) {
  vertical_stride = nnfw_depthwise_conv2d_params.stride.vertical;
  horisontal_stride = nnfw_depthwise_conv2d_params.stride.horizontal;
  multiplier = nnfw_depthwise_conv2d_params.multiplier;
  if (nnfw_depthwise_conv2d_params.padding.type == onert::ir::PaddingType::SAME) {
    padding = tflite::Padding_SAME;
  }
  else if (nnfw_depthwise_conv2d_params.padding.type == onert::ir::PaddingType::VALID) {
    padding = tflite::Padding_VALID;
  }
  else {
    throw std::logic_error("padding type in NNFW IR cannot be converted to TF Lite padding type in DepthwiseConv2D operation");
  }

  if (nnfw_depthwise_conv2d_params.activation == onert::ir::Activation::NONE) {
    fused_activation_type = tflite::ActivationFunctionType_NONE;
  }
  else if (nnfw_depthwise_conv2d_params.activation == onert::ir::Activation::RELU) {
    fused_activation_type = tflite::ActivationFunctionType_RELU;
  }
  else if (nnfw_depthwise_conv2d_params.activation == onert::ir::Activation::RELU6) {
    fused_activation_type = tflite::ActivationFunctionType_RELU6;
  }
  else {
    throw std::logic_error("Activation function type in NNFW IR cannot be converted to TF Lite activation function type type in DepthwiseConv2D operation");
  }
}

tflite::BuiltinOperator DepthwiseConv2dSpecificTraitsProvider::operationCode() const { return tflite::BuiltinOperator_DEPTHWISE_CONV_2D; }
tflite::BuiltinOptions DepthwiseConv2dSpecificTraitsProvider::operationOptionsCode() const { return tflite::BuiltinOptions_DepthwiseConv2DOptions; }
flatbuffers::Offset<void> DepthwiseConv2dSpecificTraitsProvider::serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const {
  return
      tflite::CreateDepthwiseConv2DOptions(
          flat_buffer_builder,
          padding,
          vertical_stride,
          horisontal_stride,
          multiplier,
          fused_activation_type).Union();
}


AvgPool2dSpecificTraitsProvider::AvgPool2dSpecificTraitsProvider(const onert::ir::operation::AvgPool2D::Param& nnfw_avg_pool2d_params) {
  vertical_stride = nnfw_avg_pool2d_params.stride.vertical;
  horisontal_stride = nnfw_avg_pool2d_params.stride.horizontal;
  filter_height = nnfw_avg_pool2d_params.kh;
  filter_width = nnfw_avg_pool2d_params.kw;
  if (nnfw_avg_pool2d_params.padding.type == onert::ir::PaddingType::SAME) {
    padding = tflite::Padding_SAME;
  }
  else if (nnfw_avg_pool2d_params.padding.type == onert::ir::PaddingType::VALID) {
    padding = tflite::Padding_VALID;
  }
  else {
    throw std::logic_error("padding type in NNFW IR cannot be converted to TF Lite padding type in AvgPool2D operation");
  }

  if (nnfw_avg_pool2d_params.activation == onert::ir::Activation::NONE) {
    fused_activation_type = tflite::ActivationFunctionType_NONE;
  }
  else if (nnfw_avg_pool2d_params.activation == onert::ir::Activation::RELU) {
    fused_activation_type = tflite::ActivationFunctionType_RELU;
  }
  else if (nnfw_avg_pool2d_params.activation == onert::ir::Activation::RELU6) {
    fused_activation_type = tflite::ActivationFunctionType_RELU6;
  }
  else {
    throw std::logic_error("Activation function type in NNFW IR cannot be converted to TF Lite activation function type type in AvgPool2D operation");
  }
}

tflite::BuiltinOperator AvgPool2dSpecificTraitsProvider::operationCode() const { return tflite::BuiltinOperator_AVERAGE_POOL_2D; }
tflite::BuiltinOptions AvgPool2dSpecificTraitsProvider::operationOptionsCode() const { return tflite::BuiltinOptions_Pool2DOptions; }

flatbuffers::Offset<void> AvgPool2dSpecificTraitsProvider::serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const {
  return
      tflite::CreatePool2DOptions(
          flat_buffer_builder,
          padding,
          vertical_stride,
          horisontal_stride,
          filter_width,
          filter_height,
          fused_activation_type).Union();
}


SqueezeSpecificTraitsProvider::SqueezeSpecificTraitsProvider(const onert::ir::operation::Squeeze::Param& nnfw_squeeze_params) {
  for (size_t i = 0; i < nnfw_squeeze_params.ndim; ++i) {
    dimensions.push_back(nnfw_squeeze_params.dims[i]);
  }
}

tflite::BuiltinOperator SqueezeSpecificTraitsProvider::operationCode() const { return tflite::BuiltinOperator_SQUEEZE; }
tflite::BuiltinOptions SqueezeSpecificTraitsProvider::operationOptionsCode() const { return tflite::BuiltinOptions_SqueezeOptions; }
flatbuffers::Offset<void> SqueezeSpecificTraitsProvider::serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const {
  return
      tflite::CreateSqueezeOptions(
          flat_buffer_builder,
          flat_buffer_builder.CreateVector<int32_t>(dimensions)).Union();
}


SoftmaxSpecificTraitsProvider::SoftmaxSpecificTraitsProvider(const onert::ir::operation::Softmax::Param& nnfw_softmax_params) {
  beta = nnfw_softmax_params.beta;
}

tflite::BuiltinOperator SoftmaxSpecificTraitsProvider::operationCode() const { return tflite::BuiltinOperator_SOFTMAX; }
tflite::BuiltinOptions SoftmaxSpecificTraitsProvider::operationOptionsCode() const { return tflite::BuiltinOptions_SoftmaxOptions; }
flatbuffers::Offset<void> SoftmaxSpecificTraitsProvider::serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const {
  return tflite::CreateSoftmaxOptions(flat_buffer_builder, beta).Union();
}
