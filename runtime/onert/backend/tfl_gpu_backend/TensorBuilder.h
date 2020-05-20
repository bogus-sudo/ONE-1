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

#ifndef NNFW_TFL_GPU_TENSOR_BUILDER_H
#define NNFW_TFL_GPU_TENSOR_BUILDER_H

#include "operand/Tensor.h"

#include <backend/ITensorBuilder.h>
#include <ir/OperandIndexMap.h>

#include <unordered_map>

namespace onert
{

namespace backend
{

namespace tfl_gpu
{

class TensorBuilder : public ITensorBuilder
{
public:
  bool supportDynamicTensor() final { return false; }

  void registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info, ir::Layout backend_layout, bool as_const) final;

  void notifyFirstUse(const ir::OperandIndex &) final {}
  void notifyLastUse(const ir::OperandIndex &) final {}

  bool isRegistered(const ir::OperandIndex &index) const final { return _info_about_operands.find(index) != _info_about_operands.end(); }

  void prepare(void) final {}
  void allocate() final {}
  void postFunctionPrepare() final {}

  std::shared_ptr<ITensor> tensorAt(const ir::OperandIndex &ind) final { return _info_about_operands[ind]; }

  void iterate(const IterateFunction &fn) final {}

  std::unique_ptr<ITensorManager> releaseStaticTensorManager(void) final { return nullptr; }

  std::shared_ptr<operand::Tensor> at(const ir::OperandIndex &ind) { return _info_about_operands[ind]; }

private:
  std::unordered_map<ir::OperandIndex, std::shared_ptr<operand::Tensor>> _info_about_operands;
  std::unordered_map<ir::OperandIndex, ir::Layout> _info_about_backend_layout;
  std::unordered_map<ir::OperandIndex, bool> _info_about_constants;
};

} // namespace tfl_gpu

} // namespace backend

} // namespace onert

#endif // NNFW_TFL_GPU_TENSOR_BUILDER_H
