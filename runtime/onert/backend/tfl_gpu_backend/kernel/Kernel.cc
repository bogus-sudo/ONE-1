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

#include "Kernel.h"
#include "nnfw_operation_as_tflite_model.h"


namespace onert
{

namespace backend
{

namespace tfl_gpu
{

namespace kernel
{

void Kernel::run() {
  for (auto& tensor_to_data_source: _internal_tensor_to_data_source_map) {
    memcpy(tensor_to_data_source.first->data.raw, tensor_to_data_source.second, tensor_to_data_source.first->bytes);
  }
  if (interpreter_->Invoke() != kTfLiteOk) {
    throw std::runtime_error("Failed to invoke tflite!\n");
  }
}

void Kernel::shareBufferBetween(std::shared_ptr<operand::Tensor> tensor, onert::ir::OperandIndex index_in_ir) {
  auto kernel_tensor = interpreter_->tensor(converter_->tensorIndexByOperandIndex(index_in_ir));

  // TODO Because we cannot manage memory inside TF Lite interpreter, we should give an access to input and output data
  // TODO for external environment
  // TODO Can be two cases:
  // TODO 1) buffer for data already allocated by external environment.
  // TODO    In that case we should keep pointer to those buffer and copy data from it before do calculations.
  // TODO 2) buffer for data have not been allocated by external environment.
  // TODO    In that case we should give an access to buffer allocated by TF Lite interpreter fot external environment
  if (tensor->buffer()) {
    _internal_tensor_to_data_source_map.push_back({kernel_tensor, tensor->buffer()});
  }
  else {
    tensor->setBuffer(reinterpret_cast<uint8_t *>(kernel_tensor->data.raw));
  }
}


} // namespace kernel

} // namespace tfl_gpu

} // namespace backend

} // namespace onert
