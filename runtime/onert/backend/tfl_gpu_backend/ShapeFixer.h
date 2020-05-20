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

#ifndef NNFW_TFL_GPU_SHAPE_FIXER_H
#define NNFW_TFL_GPU_SHAPE_FIXER_H

#include "TensorBuilder.h"
#include "operand/Tensor.h"

#include <backend/IShapeFixer.h>
#include <ir/Operands.h>

#include <ir/operation/Conv2D.h>

namespace onert
{
namespace backend
{
namespace tfl_gpu
{

class ShapeFixer : public IShapeFixer
{
public:
  ShapeFixer(const ir::Operands &ctx): _ctx(ctx) {}

  using IShapeFixer::visit;

  void visit(const ir::operation::Conv2D&) final {}


private:
  const ir::Operands &_ctx;
};

} // namespace tfl_gpu

} // namespace backend

} // namespace onert

#endif // NNFW_TFL_GPU_SHAPE_FIXER_H