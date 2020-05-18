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

#ifndef NNFW_TFL_GPU_TENSOR_H
#define NNFW_TFL_GPU_TENSOR_H

#include <backend/ITensor.h>

namespace onert
{

namespace backend
{

namespace tfl_gpu
{

namespace operand
{

class Tensor : public ITensor
{
public:
  Tensor() = delete;

public:
  Tensor(const ir::OperandInfo& info) {}

public:
  uint8_t *buffer() const override { return nullptr; }
  size_t dimension(size_t index) const override { return 0; }
  size_t num_dimensions() const override { return 0; }
  size_t total_size() const override { return 0; }
  size_t calcOffset(const ir::Coordinates &coords) const override { return 0; }
  ir::Layout layout() const override { return ir::Layout::NHWC; }
  ir::DataType data_type() const override { return ir::DataType::FLOAT32; }
  bool has_padding() const override { return false; }
  void access(const std::function<void(ITensor &tensor)> &fn) final {}
  bool is_dynamic() const override { return false; }
  void set_dynamic() override {}

  void dimension(size_t index, size_t dim) override {}

  void num_dimensions(size_t rank) override {}
};

} // namespace operand

} // namespace tfl_gpu

} // namespace backend

} // namespace onert

#endif // NNFW_TFL_GPU_TENSOR_H
