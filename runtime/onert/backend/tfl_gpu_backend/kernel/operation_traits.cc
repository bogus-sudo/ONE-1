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
#include "operation_specific_traits_provider.h"
#include "tensorflow/lite/schema/schema_generated.h"


OperandTraits OperandTraits::ForConstantFrom(std::shared_ptr<onert::backend::tfl_gpu::operand::Tensor> tensor) {
  OperandTraits traits;
  traits.dimensions = tensor->dimensions();
  traits.index_in_nnfw_ir = tensor->external_index();
  traits.place_for_constant_data = tensor->buffer();
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
  return flat_buffer_builder.CreateVector(place_for_constant_data, size_of_place_for_constant_data);
}


OperationTraits::OperationTraits() = default;
OperationTraits::OperationTraits(OperationTraits&& rhs) = default;
OperationTraits::~OperationTraits() = default;

void OperationTraits::setOperationSpecificTraits(std::unique_ptr<class OperationSpecificTraitsProvider> specific_traits) {
  specific_traits_provider = std::move(specific_traits);
}

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
