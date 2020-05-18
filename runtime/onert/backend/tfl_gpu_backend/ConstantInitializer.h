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

#ifndef NNFW_TFL_GPU_CONSTANT_INITIALIZER_H
#define NNFW_TFL_GPU_CONSTANT_INITIALIZER_H

#include "TensorBuilder.h"

#include <backend/IConstantInitializer.h>
#include <ir/Operands.h>

namespace onert
{
namespace backend
{
namespace tfl_gpu
{

class ConstantInitializer : public IConstantInitializer
{
public:
  ConstantInitializer(const ir::Operands &operands, const std::shared_ptr<TensorBuilder> &tensor_builder)
    : IConstantInitializer{operands}
    , _tensor_builder{tensor_builder}
  {}

public:

private:
  std::shared_ptr<ITensorBuilder> tensor_builder() const override { return _tensor_builder; }

private:
  std::shared_ptr<TensorBuilder> _tensor_builder;
};

} // namespace tfl_gpu

} // namespace backend

} // namespace onert

#endif // NNFW_TFL_GPU_CONSTANT_INITIALIZER_H
