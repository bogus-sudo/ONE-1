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

#include "tensorflow/lite/kernels/register.h"

//#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/tools/evaluation/utils.h"


namespace onert
{

namespace backend
{

namespace tfl_gpu
{

namespace kernel
{

Kernel::Kernel(const OperationTraits& operation_traits)
  : operation_resolver_(std::make_unique<tflite::ops::builtin::BuiltinOpResolver>())
{
  converter_ = std::make_unique<::NnfwOperationConverter>();
  model_ = converter_->generateTFLiteModel(operation_traits);
  tflite::InterpreterBuilder(*model_, *operation_resolver_)(&interpreter_);
  if (!interpreter_) {
    throw std::runtime_error("failed to construct interpreter\n");
  }

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    throw std::runtime_error("failed to allocate tensors!\n");
  }
}

Kernel::~Kernel() = default;

void Kernel::run() {
  if (interpreter_->Invoke() != kTfLiteOk) {
    throw std::runtime_error("Failed to invoke tflite!\n");
  }
}

void Kernel::shareBufferBetween(std::shared_ptr<operand::Tensor> tensor, onert::ir::OperandIndex index_in_ir) {
  auto kernel_tensor = interpreter_->tensor(converter_->tensorIndexByOperandIndex(index_in_ir));
  tensor->setBuffer(reinterpret_cast<uint8_t*>(kernel_tensor->data.raw));
}


} // namespace kernel

} // namespace tfl_gpu

} // namespace backend

} // namespace onert
