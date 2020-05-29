/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tflite_runtime.h"
#include "nnfw_operation_as_tflite_model.h"

#include "tensorflow/lite/kernels/register.h"

#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/tools/evaluation/utils.h"


TFLiteRuntime::TFLiteRuntime(): operation_resolver_(std::make_unique<tflite::ops::builtin::BuiltinOpResolver>()) {}

TFLiteRuntime::~TFLiteRuntime() = default;

void TFLiteRuntime::evaluate() {
   if (interpreter_->Invoke() != kTfLiteOk) {
      throw std::runtime_error("Failed to invoke tflite!\n");
   }
}

size_t TFLiteRuntime::numberOfInputs() const {
  if (!interpreter_) { throw std::logic_error("Neural network must be loaded before getting its characteristics"); }

  return interpreter_->inputs().size();
}

size_t TFLiteRuntime::sizeOfInput(size_t input_idx) const {
  if (!interpreter_) { throw std::logic_error("Neural network must be loaded before getting its characteristics"); }

  return interpreter_->input_tensor(input_idx)->bytes / sizeof(float);
}

void TFLiteRuntime::setDataForInput(size_t input_idx, const std::vector<float>& data) {
    auto input_tensor = interpreter_->input_tensor(input_idx);
    memcpy(input_tensor->data.f, data.data(), data.size() * sizeof(float));
}

size_t TFLiteRuntime::numberOfOutputs() const {

  return interpreter_->outputs().size();
}

size_t TFLiteRuntime::sizeOfOutput(size_t output_idx) const {

  return interpreter_->output_tensor(output_idx)->bytes / sizeof(float);
}

std::vector<float> TFLiteRuntime::getDataOfOutput(size_t output_idx) const {
  const auto tflite_output_tensor = interpreter_->output_tensor(output_idx);
  std::vector<float> result;
  result.assign(tflite_output_tensor->data.f, tflite_output_tensor->data.f + sizeOfOutput(output_idx));
  return result;
}