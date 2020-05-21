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

#include "operation_traits.h"

#include "ir/OperationVisitor.h"
#include "tensorflow/lite/schema/schema_generated.h"


class OperationSpecificTraitsProvider: public onert::ir::OperationVisitor {
public:
  static std::unique_ptr<OperationSpecificTraitsProvider> setUp(const onert::ir::Operation& operation) {
    auto provider = std::make_unique<OperationSpecificTraitsProvider>();
    operation.accept(*provider);

    return provider;
  }

  tflite::BuiltinOperator operationCode() const;
  tflite::BuiltinOptions operationOptionsCode() const;
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const;

private:
  friend std::unique_ptr<OperationSpecificTraitsProvider> std::make_unique<OperationSpecificTraitsProvider>();
  OperationSpecificTraitsProvider() = default;

  void visit(const onert::ir::operation::ReLU&) final;
  void visit(const onert::ir::operation::ReLU6&) final;
  void visit(const onert::ir::operation::Add&) final;
  void visit(const onert::ir::operation::Conv2D&) final;
  void visit(const onert::ir::operation::DepthwiseConv2D&) final;
  void visit(const onert::ir::operation::AvgPool2D&) final;
  void visit(const onert::ir::operation::Squeeze&) final;
  void visit(const onert::ir::operation::Softmax&) final;

private:
  std::unique_ptr<class OperationSpecificTraitsProviderImpl> provider_impl;
};

class OperationSpecificTraitsProviderImpl {
public:
  virtual tflite::BuiltinOperator operationCode() const = 0;
  virtual tflite::BuiltinOptions operationOptionsCode() const = 0;
  virtual flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder&) const = 0;

  virtual ~OperationSpecificTraitsProviderImpl() = default;
};

class ReluSpecificTraitsProvider: public OperationSpecificTraitsProviderImpl {
public:
  tflite::BuiltinOperator operationCode() const final { return tflite::BuiltinOperator_RELU; };
  tflite::BuiltinOptions operationOptionsCode() const final { return tflite::BuiltinOptions_NONE; };
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder&) const final { return 0; };
};

class Relu6SpecificTraitsProvider: public OperationSpecificTraitsProviderImpl {
public:
  tflite::BuiltinOperator operationCode() const final { return tflite::BuiltinOperator_RELU6; };
  tflite::BuiltinOptions operationOptionsCode() const final { return tflite::BuiltinOptions_NONE; };
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder&) const final { return 0; };
};

class AddSpecificTraitsProvider: public OperationSpecificTraitsProviderImpl {
public:
  AddSpecificTraitsProvider(const onert::ir::operation::Add::Param& nnfw_add_params) {
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
      /// ???
    }
  }

  tflite::BuiltinOperator operationCode() const final { return tflite::BuiltinOperator_ADD; };
  tflite::BuiltinOptions operationOptionsCode() const final { return tflite::BuiltinOptions_AddOptions; };
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const final {
    return
        tflite::CreateAddOptions(
            flat_buffer_builder,
            fused_activation_type).Union();
  }

private:
  tflite::ActivationFunctionType fused_activation_type = tflite::ActivationFunctionType_NONE;
};

class Conv2dSpecificTraitsProvider: public OperationSpecificTraitsProviderImpl {
public:
  Conv2dSpecificTraitsProvider(const onert::ir::operation::Conv2D::Param& nnfw_conv2d_params) {
    vertical_stride = nnfw_conv2d_params.stride.vertical;
    horisontal_stride = nnfw_conv2d_params.stride.horizontal;
    if (nnfw_conv2d_params.padding.type == onert::ir::PaddingType::SAME) {
      padding = tflite::Padding_SAME;
    }
    if (nnfw_conv2d_params.padding.type == onert::ir::PaddingType::VALID) {
      padding = tflite::Padding_VALID;
    }
    else {
      /// ????
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
      /// ???
    }
  }

  tflite::BuiltinOperator operationCode() const final { return tflite::BuiltinOperator_CONV_2D; };
  tflite::BuiltinOptions operationOptionsCode() const final { return tflite::BuiltinOptions_Conv2DOptions; };
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const final {
    return
      tflite::CreateConv2DOptions(
      flat_buffer_builder,
      padding,
      vertical_stride,
      horisontal_stride,
      fused_activation_type).Union();
  }

private:
  uint32_t vertical_stride = 1;
  uint32_t horisontal_stride = 1;
  tflite::Padding padding = tflite::Padding::Padding_SAME;
  tflite::ActivationFunctionType fused_activation_type = tflite::ActivationFunctionType_NONE;
};

class DepthwiseConv2dSpecificTraitsProvider: public OperationSpecificTraitsProviderImpl {
public:
  DepthwiseConv2dSpecificTraitsProvider(const onert::ir::operation::DepthwiseConv2D::Param& nnfw_depthwise_conv2d_params) {
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
      /// ????
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
      /// ???
    }
  }

  tflite::BuiltinOperator operationCode() const final { return tflite::BuiltinOperator_DEPTHWISE_CONV_2D; };
  tflite::BuiltinOptions operationOptionsCode() const final { return tflite::BuiltinOptions_DepthwiseConv2DOptions; };
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const final {
    return
        tflite::CreateDepthwiseConv2DOptions(
            flat_buffer_builder,
            padding,
            vertical_stride,
            horisontal_stride,
            multiplier,
            fused_activation_type).Union();
  }

private:
  uint32_t vertical_stride = 1;
  uint32_t horisontal_stride = 1;
  uint32_t multiplier = 1;
  tflite::Padding padding = tflite::Padding::Padding_SAME;
  tflite::ActivationFunctionType fused_activation_type = tflite::ActivationFunctionType_NONE;
};

class AvgPool2dSpecificTraitsProvider: public OperationSpecificTraitsProviderImpl {
public:
  AvgPool2dSpecificTraitsProvider(const onert::ir::operation::AvgPool2D::Param& nnfw_avg_pool2d_params) {
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
      /// ????
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
      /// ???
    }
  }

  tflite::BuiltinOperator operationCode() const final { return tflite::BuiltinOperator_AVERAGE_POOL_2D; };
  tflite::BuiltinOptions operationOptionsCode() const final { return tflite::BuiltinOptions_Pool2DOptions; };
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const final {
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

private:
  uint32_t vertical_stride = 1;
  uint32_t horisontal_stride = 1;
  uint32_t filter_height = 1;
  uint32_t filter_width = 1;
  tflite::Padding padding = tflite::Padding::Padding_SAME;
  tflite::ActivationFunctionType fused_activation_type = tflite::ActivationFunctionType_NONE;
};

class SqueezeSpecificTraitsProvider: public OperationSpecificTraitsProviderImpl {
public:
  SqueezeSpecificTraitsProvider(const onert::ir::operation::Squeeze::Param& nnfw_squeeze_params) {
    for (size_t i = 0; i < nnfw_squeeze_params.ndim; ++i) {
      dimensions.push_back(nnfw_squeeze_params.dims[i]);
    }
  }

  tflite::BuiltinOperator operationCode() const final { return tflite::BuiltinOperator_SQUEEZE; };
  tflite::BuiltinOptions operationOptionsCode() const final { return tflite::BuiltinOptions_SqueezeOptions; };
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const final {
    return
        tflite::CreateSqueezeOptions(
            flat_buffer_builder,
            flat_buffer_builder.CreateVector<int32_t>(dimensions)).Union();
  }

private:
  std::vector<int32_t> dimensions;
};

class SoftmaxSpecificTraitsProvider: public OperationSpecificTraitsProviderImpl {
public:
  SoftmaxSpecificTraitsProvider(const onert::ir::operation::Softmax::Param& nnfw_softmax_params) {
    beta = nnfw_softmax_params.beta;
  }

  tflite::BuiltinOperator operationCode() const final { return tflite::BuiltinOperator_SOFTMAX; };
  tflite::BuiltinOptions operationOptionsCode() const final { return tflite::BuiltinOptions_SoftmaxOptions; };
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const final {
    return tflite::CreateSoftmaxOptions(flat_buffer_builder, beta).Union();
  }

private:
  float beta = 1.f;
};


tflite::BuiltinOperator OperationSpecificTraitsProvider::operationCode() const { return provider_impl->operationCode(); };
tflite::BuiltinOptions OperationSpecificTraitsProvider::operationOptionsCode() const { return provider_impl->operationOptionsCode(); };
flatbuffers::Offset<void> OperationSpecificTraitsProvider::serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const {
  return provider_impl->serializedOperationOptions(flat_buffer_builder);
}


void OperationSpecificTraitsProvider::visit(const onert::ir::operation::ReLU& operation) {
  provider_impl = std::make_unique<ReluSpecificTraitsProvider>();
}

void OperationSpecificTraitsProvider::visit(const onert::ir::operation::ReLU6& operation) {
  provider_impl = std::make_unique<Relu6SpecificTraitsProvider>();
}

void OperationSpecificTraitsProvider::visit(const onert::ir::operation::Add& operation) {
  provider_impl = std::make_unique<AddSpecificTraitsProvider>(operation.param());
}

void OperationSpecificTraitsProvider::visit(const onert::ir::operation::Conv2D& conv2d_operation) {
  provider_impl = std::make_unique<Conv2dSpecificTraitsProvider>(conv2d_operation.param());
}

void OperationSpecificTraitsProvider::visit(const onert::ir::operation::DepthwiseConv2D& depthwise_conv2d_operation) {
  provider_impl = std::make_unique<DepthwiseConv2dSpecificTraitsProvider>(depthwise_conv2d_operation.param());
}

void OperationSpecificTraitsProvider::visit(const onert::ir::operation::AvgPool2D& avg_pool2d_operation) {
  provider_impl = std::make_unique<AvgPool2dSpecificTraitsProvider>(avg_pool2d_operation.param());
}

void OperationSpecificTraitsProvider::visit(const onert::ir::operation::Squeeze& squeeze_operation) {
  provider_impl = std::make_unique<SqueezeSpecificTraitsProvider>(squeeze_operation.param());
}

void OperationSpecificTraitsProvider::visit(const onert::ir::operation::Softmax& softmax_operation) {
  provider_impl = std::make_unique<SoftmaxSpecificTraitsProvider>(softmax_operation.param());
}


OperandTraits OperandTraits::ForConstantFrom(std::shared_ptr<onert::backend::tfl_gpu::operand::Tensor> tensor) {
  OperandTraits traits;
  traits.dimensions = tensor->dimensions();
  traits.index_in_nnfw_ir = tensor->external_index();
  traits.size_of_place_for_constant_data = tensor->total_size();
  return traits;
}

OperandTraits OperandTraits::ForNonConstantFrom(std::shared_ptr<onert::backend::tfl_gpu::operand::Tensor> tensor) {
  OperandTraits traits;
  traits.dimensions = tensor->dimensions();
  traits.index_in_nnfw_ir = tensor->external_index();
  return traits;
}

flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
OperandTraits::serializedPlaceForConstantData(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const {
  // On this step we just create a place for constant data. This place will be filled later by ConstantInitializer
  // TODO it seems like owning of raw pointer have been move to flat_buffer_builder, so we must not free allocated memory.
  // TODO It is need to check it
  // TODO check that it is actually traits of constant operand
  uint8_t* place_for_constant_data = new uint8_t[size_of_place_for_constant_data];
  return flat_buffer_builder.CreateVector(place_for_constant_data, size_of_place_for_constant_data);
}


OperationTraits::OperationTraits() = default;
OperationTraits::OperationTraits(OperationTraits&& rhs) = default;
OperationTraits::~OperationTraits() = default;

int64_t OperationTraits::operationCode() const {
  static_assert(sizeof(int64_t) >= sizeof(tflite::BuiltinOperator), "bad cast: tflite::BuiltinOperator type has bigger size then int32_t");
  return specific_traits_provider->operationCode();
}

int64_t OperationTraits::operationOptionsCode() const {
  static_assert(sizeof(int64_t) >= sizeof(tflite::BuiltinOptions), "bad cast: tflite::BuiltinOptions type has bigger size then int32_t");

  return specific_traits_provider->operationOptionsCode();
}

flatbuffers::Offset<void> OperationTraits::serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const {
  return specific_traits_provider->serializedOperationOptions(flat_buffer_builder);
}

void OperationTraits::setOperationSpecificTraits(const onert::ir::Operation &operation) {
  specific_traits_provider = OperationSpecificTraitsProvider::setUp(operation);
}
