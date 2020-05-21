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

#include "nnfw_operation_as_tflite_model.h"

#include "operation_traits.h"

std::unique_ptr<tflite::FlatBufferModel> NnfwOperationConverter::generateTFLiteModel(const OperationTraits& operation_traits) {
  serializeTensors(operation_traits);
  serializeOperation(operation_traits);
  constructSubgraphUsingSeralizedOperationAndTensors();
  serializeOperationCode(operation_traits);
  constructSerializedModelFromParts();

  return getModel();
}

void NnfwOperationConverter::serializeTensors(const OperationTraits& operation_traits) {
  // by the rules of serializing TF Lite model, first buffer of constant data should be empty,
  // so all tensors which are not need constant data can use reference to that empty buffer
  serializeEmptyBuffer();

  for (const auto& operator_traits: operation_traits.traits_of_inputs) {
    int32_t reference_to_serialized_tensor = serializeTensor(operator_traits.dimensions, NO_NEED_CONSTANT_DATA);
    treatAsOperationInput(reference_to_serialized_tensor);
    treatAsModelInput(reference_to_serialized_tensor);
    operand_index_in_ir_to_tensor_index_in_kernel[operator_traits.index_in_nnfw_ir] = reference_to_serialized_tensor;
  }

  for (const auto& operator_traits: operation_traits.traits_of_constant_inputs) {
    int32_t reference_to_serialized_data = serializeConstantData(operator_traits);
    int32_t reference_to_serialized_tensor = serializeTensor(operator_traits.dimensions, reference_to_serialized_data);
    treatAsOperationInput(reference_to_serialized_tensor);
    operand_index_in_ir_to_tensor_index_in_kernel[operator_traits.index_in_nnfw_ir] = reference_to_serialized_tensor;
  }

  for (const auto& operator_traits: operation_traits.traits_of_outputs) {
    int32_t reference_to_serialized_tensor = serializeTensor(operator_traits.dimensions, NO_NEED_CONSTANT_DATA);
    treatAsOperationOutput(reference_to_serialized_tensor);
    treatAsModelOutput(reference_to_serialized_tensor);
    operand_index_in_ir_to_tensor_index_in_kernel[operator_traits.index_in_nnfw_ir] = reference_to_serialized_tensor;
  }
}

void NnfwOperationConverter::serializeEmptyBuffer() {
  tflite::BufferBuilder bb(flat_buffer_builder);
  serialized_constant_data.push_back(bb.Finish());
}

int32_t NnfwOperationConverter::serializeConstantData(const OperandTraits& operand_traits) {
  serialized_constant_data.push_back(
    tflite::CreateBuffer(flat_buffer_builder, operand_traits.serializedPlaceForConstantData(flat_buffer_builder)));

  return serialized_constant_data.size() - 1;
}

int32_t NnfwOperationConverter::serializeTensor(std::vector<int32_t> its_dimensions, uint32_t reference_to_constant_data) {
  serialized_tensors.push_back(
      tflite::CreateTensor(
          flat_buffer_builder,
          flat_buffer_builder.CreateVector<int>(its_dimensions),
          tflite::TensorType_FLOAT32,
          reference_to_constant_data, serializedTensorName(serialized_tensors.size())));

  return serialized_tensors.size() - 1;
}

flatbuffers::Offset<flatbuffers::String> NnfwOperationConverter::serializedTensorName(size_t tensor_id) {
  std::string tn = "tensor_" + std::to_string(tensor_id);
  return flat_buffer_builder.CreateString(tn.c_str());
}

void NnfwOperationConverter::treatAsOperationInput(int32_t reference_to_serialized_tensor) {
  references_to_serialized_operation_inputs.push_back(reference_to_serialized_tensor);
}

void NnfwOperationConverter::treatAsModelInput(int32_t reference_to_serialized_tensor) {
  references_to_serialized_model_inputs.push_back(reference_to_serialized_tensor);
}

void NnfwOperationConverter::treatAsOperationOutput(int32_t reference_to_serialized_tensor) {
  references_to_serialized_operation_outputs.push_back(reference_to_serialized_tensor);
}

void NnfwOperationConverter::treatAsModelOutput(int32_t reference_to_serialized_tensor) {
  references_to_serialized_model_outputs.push_back(reference_to_serialized_tensor);
}

void NnfwOperationConverter::serializeOperation(const OperationTraits& operation_traits) {
  serialized_operation =
      tflite::CreateOperator(
          flat_buffer_builder,
          reference_to_serialized_operation_code,
          serializedIndexesOfOperationInputs(),
          serializedIndexesOfOperationOutputs(),
          static_cast<tflite::BuiltinOptions>(operation_traits.operationOptionsCode()),
          operation_traits.serializedOperationOptions(flat_buffer_builder));
}

flatbuffers::Offset<flatbuffers::Vector<int>>
NnfwOperationConverter::serializedIndexesOfOperationInputs() {
  return flat_buffer_builder.CreateVector<int>(references_to_serialized_operation_inputs);
}

flatbuffers::Offset<flatbuffers::Vector<int>>
NnfwOperationConverter::serializedIndexesOfOperationOutputs() {
  return flat_buffer_builder.CreateVector<int>(references_to_serialized_operation_outputs);
}

void NnfwOperationConverter::constructSubgraphUsingSeralizedOperationAndTensors() {
  serialized_subgraph =
      tflite::CreateSubGraph(
          flat_buffer_builder, serializedTensors(), serializedIndexesOfModelInputs(),
          serializedIndexesOfModelOutputs(), serializedOperation(), serializedSubgraphName());
}

NnfwOperationConverter::Serialized<tflite::Tensor> NnfwOperationConverter::serializedTensors() {
  return flat_buffer_builder.CreateVector(serialized_tensors);
}

flatbuffers::Offset<flatbuffers::Vector<int>>
NnfwOperationConverter::serializedIndexesOfModelInputs() {
  return flat_buffer_builder.CreateVector<int>(references_to_serialized_model_inputs);
}

flatbuffers::Offset<flatbuffers::Vector<int>>
NnfwOperationConverter::serializedIndexesOfModelOutputs() {
  return flat_buffer_builder.CreateVector<int>(references_to_serialized_model_outputs);
}

NnfwOperationConverter::Serialized<tflite::Operator> NnfwOperationConverter::serializedOperation() {
  return flat_buffer_builder.CreateVector(std::vector<flatbuffers::Offset<tflite::Operator>>{serialized_operation});
}

flatbuffers::Offset<flatbuffers::String> NnfwOperationConverter::serializedSubgraphName() {
  return flat_buffer_builder.CreateString("main");
}

void NnfwOperationConverter::serializeOperationCode(const OperationTraits& operation_traits) {
  serialized_operation_code = tflite::CreateOperatorCode(flat_buffer_builder, static_cast<tflite::BuiltinOperator>(operation_traits.operationCode()));

  // this because currently converter is currently used to convert only one operation to a model
  reference_to_serialized_operation_code = 0;
}

void NnfwOperationConverter::constructSerializedModelFromParts() {
  flatbuffers::Offset<tflite::Model> model = tflite::CreateModel(flat_buffer_builder,
                                                                 TFLITE_SCHEMA_VERSION, serializedOperationCode(), serializedSubgraph(),
                                                                 serializedModelDescription(), serializedConstantData());

  tflite::FinishModelBuffer(flat_buffer_builder, model);
}

NnfwOperationConverter::Serialized<tflite::OperatorCode>
NnfwOperationConverter::serializedOperationCode() {
  return flat_buffer_builder.CreateVector(std::vector<flatbuffers::Offset<tflite::OperatorCode>>{serialized_operation_code});
}

NnfwOperationConverter::Serialized<tflite::SubGraph> NnfwOperationConverter::serializedSubgraph() {
  return flat_buffer_builder.CreateVector(std::vector<flatbuffers::Offset<tflite::SubGraph>>{serialized_subgraph});
}

flatbuffers::Offset<flatbuffers::String> NnfwOperationConverter::serializedModelDescription() {
  return flat_buffer_builder.CreateString("generated by nnfw operation to tflite model converter");
}

NnfwOperationConverter::Serialized<tflite::Buffer> NnfwOperationConverter::serializedConstantData() {
  return flat_buffer_builder.CreateVector(serialized_constant_data);
}

std::unique_ptr<tflite::FlatBufferModel> NnfwOperationConverter::getModel() {
  return tflite::FlatBufferModel::BuildFromModel(tflite::GetModel(flat_buffer_builder.GetBufferPointer()));
}
