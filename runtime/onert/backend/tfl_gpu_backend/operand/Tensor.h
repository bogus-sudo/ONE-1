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
#include "ir/OperandInfo.h"
#include "ir/Index.h"

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
  Tensor(onert::ir::OperandIndex external_idx, const onert::ir::OperandInfo& info, bool is_constant)
    : _dimensions(info.shape().dims())
    , _total_size(info.total_size())
    , _external_idx(external_idx)
    , _type(info.typeInfo().type())
    , _is_constant(is_constant)
  {}

public:
  uint8_t *buffer() const final { return _data; }
  size_t dimension(size_t index) const final { return _dimensions.at(index); }
  size_t num_dimensions() const final { return _dimensions.size(); }
  size_t total_size() const final { return _total_size; }
  size_t calcOffset(const ir::Coordinates &coords) const final {
    size_t rank = num_dimensions();
    size_t offset = 0;
    for (size_t i = 0; i < rank; ++i)
    {
      offset = offset * dimension(i) + coords[i];
    }
    offset *= sizeOfDataType(data_type());
    return offset;
  }
  ir::Layout layout() const final { return ir::Layout::NHWC; }
  ir::DataType data_type() const final { return _type; }
  bool has_padding() const final { return false; }
  void access(const std::function<void(ITensor &tensor)> &fn) final { fn(*this); }

  void setBuffer(uint8_t* p) { _data = p; }
  const std::vector<int32_t>& dimensions() const { return _dimensions; }
  onert::ir::OperandIndex external_index() const { return _external_idx; }
  bool is_constant() const { return _is_constant; }

private:
  std::vector<int32_t> _dimensions;
  size_t _total_size;
  uint8_t* _data = nullptr;
  ir::OperandIndex _external_idx;
  ir::DataType _type;
  bool _is_constant;
};

} // namespace operand

} // namespace tfl_gpu

} // namespace backend

} // namespace onert

#endif // NNFW_TFL_GPU_TENSOR_H
