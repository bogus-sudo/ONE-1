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
#include "kernel/operation_specific_traits_provider.h"

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
  OperationTraits operation_traits;
  operation_traits.setOperationSpecificTraits(std::make_unique<Conv2dSpecificTraitsProvider>(node.param()));

  configureInputsAndOutputs(node, operation_traits);

  auto fn = std::make_unique<::onert::backend::tfl_gpu::kernel::Kernel>(operation_traits);
  shareInternallyAllocatedBuffers(node, *fn);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::DepthwiseConv2D& node)
{
  OperationTraits operation_traits;
  operation_traits.setOperationSpecificTraits(std::make_unique<DepthwiseConv2dSpecificTraitsProvider>(node.param()));

  configureInputsAndOutputs(node, operation_traits);

  auto fn = std::make_unique<::onert::backend::tfl_gpu::kernel::Kernel>(operation_traits);
  shareInternallyAllocatedBuffers(node, *fn);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::AvgPool2D& node)
{
  OperationTraits operation_traits;
  operation_traits.setOperationSpecificTraits(std::make_unique<AvgPool2dSpecificTraitsProvider>(node.param()));

  configureInputsAndOutputs(node, operation_traits);

  auto fn = std::make_unique<::onert::backend::tfl_gpu::kernel::Kernel>(operation_traits);
  shareInternallyAllocatedBuffers(node, *fn);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::Softmax& node)
{
  OperationTraits operation_traits;
  operation_traits.setOperationSpecificTraits(std::make_unique<SoftmaxSpecificTraitsProvider>(node.param()));

  configureInputsAndOutputs(node, operation_traits);

  auto fn = std::make_unique<::onert::backend::tfl_gpu::kernel::Kernel>(operation_traits);
  shareInternallyAllocatedBuffers(node, *fn);

  _return_fn = std::move(fn);
}

void KernelGenerator::configureInputsAndOutputs(const onert::ir::Operation &node, OperationTraits &operation_traits) {
  for (size_t i = 0; i < node.getInputs().size(); ++i) {
    auto operand_index_in_ir = node.getInputs().at(i);
    if (!_tensor_builder->isRegistered(operand_index_in_ir)) {
      std::logic_error("unregistered tensor detected");
    }

    auto tensor = _tensor_builder->at(operand_index_in_ir);
    if (tensor->is_constant()) {
      // TODO Using constant data from IR directly is workaround
      // TODO actually constant tensors should be initialized by ConstantInitializer, but
      // TODO TF Lite GPU delegate loads constant data only once during memory allocation step
      // TODO (see implementation of constructor of class Kernel in file kernel_initialization_on_aarch64_android_platform.cc).
      // TODO It is bad idea because potentially this workaround breaks working with layers.
      tensor->setBuffer(_ctx.at(operand_index_in_ir).data()->base());
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
}

void KernelGenerator::shareInternallyAllocatedBuffers(const onert::ir::Operation& node, onert::backend::tfl_gpu::kernel::Kernel& fn) {
  for (size_t i = 0; i < node.getInputs().size(); ++i) {
    auto operand_index_in_ir = node.getInputs().at(i);
    auto tensor = _tensor_builder->at(operand_index_in_ir);
    if(!tensor->is_constant()) {
      fn.shareBufferBetween(_tensor_builder->at(operand_index_in_ir), operand_index_in_ir);
    }
  }
  for (size_t i = 0; i < node.getOutputs().size(); ++i) {
    auto operand_index_in_ir = node.getOutputs().at(i);
    fn.shareBufferBetween(_tensor_builder->at(operand_index_in_ir), operand_index_in_ir);
  }
}


} // namespace tfl_gpu

} // namespace backend

} // namespace onert
