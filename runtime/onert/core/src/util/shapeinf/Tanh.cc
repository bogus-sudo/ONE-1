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

#include "util/ShapeInference.h"

namespace onert
{
namespace shape_inference
{

void StaticInferer::visit(const ir::operation::Tanh &op)
{
  const auto input_idx{op.getInputs().at(ir::operation::Tanh::Input::INPUT)};
  const auto &input = _operands.at(input_idx);

  // get mutable output operand
  const auto output_idx = op.getOutputs().at(0);
  ir::Operand &output = _operands.at(output_idx);

  // if input is dynamic, output also becomes dynamic
  if (input.info().isDynamic())
  {
    output.info().setDynamic();
    return;
  }

  // re-sizing output shape
  ir::Shape new_shape = input.info().shape();
  output.info().shape(new_shape);
}

void DynamicInferer::visit(const ir::operation::Tanh &op)
{
  // check if output is not dynamic
  auto output_ind = op.getOutputs().at(0);
  auto output = _tensor_registry->getITensor(output_ind);
  if (!output->is_dynamic())
    return;

  // getting output shape
  auto input_ind = op.getInputs().at(ir::operation::Tanh::Input::INPUT);
  auto input = _tensor_registry->getITensor(input_ind);
  auto output_shape = getShape(input.get());

  // set output shape and output buffer
  setShape(output.get(), output_shape);

  _dynamic_tensor_manager->allocate(output_ind, output_shape);
  assert(output->buffer() != nullptr);
}

} // namespace shape_inference
} // namespace onert
