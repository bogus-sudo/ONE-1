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

#ifndef NNFW_OPERATION_TRAITS_H
#define NNFW_OPERATION_TRAITS_H

#include <vector>
#include <cstdint>
#include <memory>


namespace onert
{

namespace ir {

class Data;
class Operation;

}

}

namespace flatbuffers {

template<typename> class Offset;
class FlatBufferBuilder;

}

struct OperandTraits {
  using dimension_t = int32_t;
  using Dimensions = std::vector<dimension_t>;

  Dimensions dimensions;
  const uint8_t* constant_data = nullptr;
  size_t constant_data_size = 0;

  OperandTraits(const Dimensions& its_dimensions): dimensions(its_dimensions) {}
  OperandTraits(const Dimensions& its_dimensions, const onert::ir::Data& const_data);
};

class OperationTraits {
public:
  std::vector<OperandTraits> traits_of_inputs;
  std::vector<OperandTraits> traits_of_constant_inputs;
  std::vector<OperandTraits> traits_of_outputs;

public:
  OperationTraits();
  OperationTraits(const OperationTraits& rhs) = delete;
  OperationTraits(OperationTraits&& rhs);

  const OperationTraits& operator=(const OperationTraits& rhs) = delete;
  const OperationTraits& operator=(const OperationTraits&& rhs) = delete;

  void addInput(const OperandTraits& traits) { traits_of_inputs.push_back(traits); }
  void addConstantInput(const OperandTraits& traits) { traits_of_constant_inputs.push_back(traits); }
  void addOutput(const OperandTraits& traits) { traits_of_outputs.push_back(traits); }

  void setOperationSpecificTraits(const onert::ir::Operation& operation);

  int64_t operationCode() const;
  int64_t operationOptionsCode() const;
  flatbuffers::Offset<void> serializedOperationOptions(flatbuffers::FlatBufferBuilder& flat_buffer_builder) const;

  ~OperationTraits();

private:
  std::unique_ptr<class OperationSpecificTraitsProvider> specific_traits_provider;
};




#endif // NNFW_OPERATION_TRAITS_H
