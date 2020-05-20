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

#ifndef NNFW_TFL_GPU_KERNEL_GENERATOR_H
#define NNFW_TFL_GPU_KERNEL_GENERATOR_H

#include "TensorBuilder.h"
#include "operand/Tensor.h"

#include <backend/CustomKernelBuilder.h>
#include <backend/IKernelGenerator.h>
#include <ir/Operands.h>

namespace onert
{

namespace backend
{

namespace tfl_gpu
{

class KernelGenerator : public IKernelGenerator
{
public:
  KernelGenerator(const ir::Operands &ctx, const std::shared_ptr<TensorBuilder> &tensor_builder,
                  const std::shared_ptr<custom::IKernelBuilder> &kernel_builder)
    : _ctx(ctx)
    , _tensor_builder(tensor_builder)
    , _kernel_builder(kernel_builder)
    , _current_op_seq_layout(ir::Layout::UNKNOWN)
  {}

  using IKernelGenerator::visit;

  void visit(const ir::OpSequence &operations_sequence) final;

  void visit(const ir::operation::Conv2D& node) final;

private:
  const ir::Operands &_ctx;
  std::shared_ptr<TensorBuilder> _tensor_builder;
  std::shared_ptr<backend::custom::IKernelBuilder> _kernel_builder;
  ir::Layout _current_op_seq_layout;
};

} // namespace tfl_gpu

} // namespace backend

} // namespace onert

#endif // NNFW_TFL_GPU_KERNEL_GENERATOR_H
