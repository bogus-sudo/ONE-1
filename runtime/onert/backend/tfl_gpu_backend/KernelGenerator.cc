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

  const auto ofm_index{node.getOutputs().at(0)};
  if (_tensor_builder->isRegistered(ofm_index)) {
    auto tensor = _tensor_builder->at(ofm_index);
    operation_traits.addOutput({tensor->dimensions()});
  }
  else {
    std::logic_error("unregistered tensor detected");
  }
  const auto ifm_index{node.getInputs().at(Conv2D::Input::INPUT)};
  if (_tensor_builder->isRegistered(ifm_index)) {
    auto tensor = _tensor_builder->at(ifm_index);
    operation_traits.addInput({tensor->dimensions()});
  }
  else {
    std::logic_error("unregistered tensor detected");
  }
  const auto ker_index{node.getInputs().at(Conv2D::Input::KERNEL)};
  operation_traits.addConstantInput({_ctx.at(ker_index).shape().dims(), *_ctx.at(ker_index).data()});
  const auto bias_index{node.getInputs().at(Conv2D::Input::BIAS)};
  operation_traits.addConstantInput({_ctx.at(bias_index).shape().dims(), *_ctx.at(bias_index).data()});

  auto fn = std::make_unique<::onert::backend::tfl_gpu::kernel::Kernel>(operation_traits);
  fn->shareBufferBetween(_tensor_builder->at(ifm_index), 0);
  fn->shareBufferBetween(_tensor_builder->at(ker_index), 1);
  fn->shareBufferBetween(_tensor_builder->at(bias_index), 2);
  fn->shareBufferBetween(_tensor_builder->at(ofm_index), 3);
  _return_fn = std::move(fn);
}

} // namespace tfl_gpu

} // namespace backend

} // namespace onert
