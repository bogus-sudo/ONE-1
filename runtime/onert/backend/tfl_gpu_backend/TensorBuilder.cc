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

#include "TensorBuilder.h"


namespace onert
{

namespace backend
{

namespace tfl_gpu
{

void TensorBuilder::registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info, ir::Layout backend_layout, bool as_const) {
  _info_about_operands[ind] = std::make_shared<operand::Tensor>(info);
  _info_about_backend_layout[ind] = backend_layout;
  _info_about_constants[ind] = as_const;
}

} // namespace tfl_gpu

} // namespace backend

} // namespace onert
