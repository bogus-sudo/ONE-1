/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "KernelGenerator.h"

#include "kernel/Kernel.h"
#include "kernel/operation_traits.h"

namespace onert
{

namespace backend
{

namespace tfl_gpu
{


void KernelGenerator::visit(const ir::OpSequence &operations_sequence)
{
  if (!_return_fn_seq) { _return_fn_seq = std::make_unique<exec::FunctionSequence>(); }
  for (const auto &e : operations_sequence.operations())
  {
    const auto &node = *(e.node);
    node.accept(*this);
    _return_fn_seq->append(releaseFunction());
  }
}

void KernelGenerator::visit(const ir::operation::Conv2D& node)
{
  using ir::operation::Conv2D;

  OperationTraits operation_traits;
  operation_traits.setOperationSpecificTraits(node);


  for (size_t i = 0; i < node.getInputs().size(); ++i) {
    auto operand_index_in_ir = node.getInputs().at(i);
    if (!_tensor_builder->isRegistered(operand_index_in_ir)) {
      std::logic_error("unregistered tensor detected");
    }

    auto tensor = _tensor_builder->at(operand_index_in_ir);
    if (tensor->is_constant()) {
      operation_traits.addConstantInput(OperandTraits::ForConstantFrom(tensor));
    }
    else {
      operation_traits.addInput(OperandTraits::ForNonConstantFrom(tensor));
    }
  }

  for (size_t i = 0; i < node.getOutputs().size(); ++i) {
    auto operand_index_in_ir = node.getOutputs().at(i);
    if (!_tensor_builder->isRegistered(operand_index_in_ir)) {
      std::logic_error("unregistered tensor detected");
    }
    auto tensor = _tensor_builder->at(operand_index_in_ir);
    operation_traits.addOutput(OperandTraits::ForNonConstantFrom(tensor));
  }

  auto fn = std::make_unique<::onert::backend::tfl_gpu::kernel::Kernel>(operation_traits);
  for (size_t i = 0; i < node.getInputs().size(); ++i) {
    auto operand_index_in_ir = node.getInputs().at(i);
    fn->shareBufferBetween(_tensor_builder->at(operand_index_in_ir), operand_index_in_ir);
  }
  for (size_t i = 0; i < node.getOutputs().size(); ++i) {
    auto operand_index_in_ir = node.getOutputs().at(i);
    fn->shareBufferBetween(_tensor_builder->at(operand_index_in_ir), operand_index_in_ir);
  }
  _return_fn = std::move(fn);
}

} // namespace tfl_gpu

} // namespace backend

} // namespace onert
