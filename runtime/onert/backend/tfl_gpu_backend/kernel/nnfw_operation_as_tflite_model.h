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

#ifndef NNFW_OPERATION_AS_TFLITE_MODEL_H
#define NNFW_OPERATION_AS_TFLITE_MODEL_H

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/version.h"

#include "ir/Index.h"


class NnfwOperationConverter
{
  template<typename PartOfModel> using Serialized = flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<PartOfModel>>>;

  static constexpr uint32_t NO_NEED_CONSTANT_DATA = 0;

public:
  NnfwOperationConverter() {}
  std::unique_ptr<tflite::FlatBufferModel> generateTFLiteModel(const class OperationTraits& operation_traits);
  int32_t tensorIndexByOperandIndex(onert::ir::OperandIndex idx) {
    // TODO add check that there is mapping between operand index in IR and tensor index in kernel
    return operand_index_in_ir_to_tensor_index_in_kernel[idx];
  }

private:
  void serializeTensors(const class OperationTraits& operation_traits);
  int32_t serializeTensor(std::vector<int32_t> its_dimensions, uint32_t reference_to_constant_data);
  int32_t serializeConstantData(const class OperandTraits& operand_traits);
  void serializeOperation(const class OperationTraits& operation_traits);
  void serializeOperationCode(const class OperationTraits& operand_traits);
  void serializeEmptyBuffer();

  void treatAsOperationInput(int32_t reference_to_serialized_tensor);
  void treatAsModelInput(int32_t reference_to_serialized_tensor);
  void treatAsOperationOutput(int32_t reference_to_serialized_tensor);
  void treatAsModelOutput(int32_t reference_to_serialized_tensor);

  Serialized<tflite::Tensor> serializedTensors();
  Serialized<tflite::Operator> serializedOperation();
  Serialized<tflite::OperatorCode> serializedOperationCode();
  Serialized<tflite::SubGraph> serializedSubgraph();
  Serialized<tflite::Buffer> serializedConstantData();

  flatbuffers::Offset<flatbuffers::String> serializedTensorName(size_t tensor_id);
  flatbuffers::Offset<flatbuffers::String> serializedSubgraphName();
  flatbuffers::Offset<flatbuffers::String> serializedModelDescription();

  flatbuffers::Offset<flatbuffers::Vector<int>> serializedIndexesOfModelInputs();
  flatbuffers::Offset<flatbuffers::Vector<int>> serializedIndexesOfModelOutputs();
  flatbuffers::Offset<flatbuffers::Vector<int>> serializedIndexesOfOperationInputs();
  flatbuffers::Offset<flatbuffers::Vector<int>> serializedIndexesOfOperationOutputs();

  void constructSubgraphUsingSeralizedOperationAndTensors();
  void constructSerializedModelFromParts();

  std::unique_ptr<tflite::FlatBufferModel> getModel();

private:
  flatbuffers::FlatBufferBuilder flat_buffer_builder;

  std::unordered_map<onert::ir::OperandIndex, int32_t> operand_index_in_ir_to_tensor_index_in_kernel;

  std::vector<flatbuffers::Offset<tflite::Tensor>> serialized_tensors;
  std::vector<flatbuffers::Offset<tflite::Buffer>> serialized_constant_data;
  flatbuffers::Offset<tflite::Operator> serialized_operation;
  flatbuffers::Offset<tflite::SubGraph> serialized_subgraph;
  flatbuffers::Offset<tflite::OperatorCode> serialized_operation_code;
  std::vector<int32_t> references_to_serialized_operation_inputs;
  std::vector<int32_t> references_to_serialized_model_inputs;
  std::vector<int32_t> references_to_serialized_operation_outputs;
  std::vector<int32_t> references_to_serialized_model_outputs;

  int32_t reference_to_serialized_operation_code = 0;
};

#endif //NNFW_OPERATION_AS_TFLITE_MODEL_H
