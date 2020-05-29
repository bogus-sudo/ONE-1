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

#include "working_with_nnfw_ir.h"

#include "ir/Graph.h"

#include <stack>


static OperationTraits setUpOperationTraits(const onert::ir::Graph& nnfw_ir, const onert::ir::Operation& operation);
std::vector<OperationTraits> generateTraitsOfOperationsFrom(std::shared_ptr<onert::ir::Graph> nnfw_ir) {
  onert::ir::OperandIndexSequence already_processed_operands;
  std::stack<onert::ir::OperationIndex> operations_which_are_can_be_processed;
  for (auto it = nnfw_ir->getInputs().begin(); it != nnfw_ir->getInputs().end(); ++it) {
    const auto& input = nnfw_ir->operands().at(*it);
    for (auto operation_index: input.getUses().list()) {
      operations_which_are_can_be_processed.push(operation_index);
    }
    already_processed_operands.append(*it);
  }

  std::vector<OperationTraits> traits_of_operations;
  while (!operations_which_are_can_be_processed.empty()) {
    const auto& operation = nnfw_ir->operations().at(operations_which_are_can_be_processed.top());
    operations_which_are_can_be_processed.pop();
    for (auto it = operation.getOutputs().begin(); it != operation.getOutputs().end(); ++it) {
      already_processed_operands.append(*it);
      const auto& operand = nnfw_ir->operands().at(*it);
      for (auto operation_index: operand.getUses().list()) {
        const auto& next_operation = nnfw_ir->operations().at(operation_index);
        bool is_next_operation_ready_to_process = true;
        for (auto op = next_operation.getInputs().begin(); op != next_operation.getInputs().end(); ++op) {
          if (!already_processed_operands.contains(*op) && !nnfw_ir->operands().at(*op).isConstant()) {
            is_next_operation_ready_to_process = false;
          }
        }
        if (is_next_operation_ready_to_process) {
          operations_which_are_can_be_processed.push(operation_index);
        }
      }
    }

    traits_of_operations.emplace_back(setUpOperationTraits(*nnfw_ir, operation));
  }

  return traits_of_operations;
}

OperationTraits setUpOperationTraits(const onert::ir::Graph& nnfw_ir, const onert::ir::Operation& operation) {
  OperationTraits operation_traits;
  operation_traits.setOperationSpecificTraits(operation);
  for (size_t i = 0; i < operation.getInputs().size(); ++i) {
    auto nnfw_operand_index = operation.getInputs().at(i);
    const auto& nnfw_operand = nnfw_ir.operands().at(nnfw_operand_index);
    if (nnfw_operand.isConstant()) {
      operation_traits.addConstantInput({nnfw_operand.shape().dims(), *nnfw_operand.data()});
    }
    else {
      operation_traits.addInput({nnfw_operand.shape().dims()});
    }
  }
  for (size_t i = 0; i < operation.getOutputs().size(); ++i) {
    operation_traits.addOutput({nnfw_ir.operands().at(operation.getOutputs().at(i)).shape().dims()});
  }

  return operation_traits;
}
