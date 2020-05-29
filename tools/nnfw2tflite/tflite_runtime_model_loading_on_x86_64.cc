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

#include <iostream>


void TFLiteRuntime::loadModelFrom(const std::string& file) {
  model_ = tflite::FlatBufferModel::BuildFromFile(file.c_str());
  tflite::InterpreterBuilder(*model_, *operation_resolver_)(&interpreter_);
  if (!interpreter_) {
    throw std::runtime_error("failed to construct interpreter\n");
  }

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    throw std::runtime_error("failed to allocate tensors!\n");
  }
}

void TFLiteRuntime::loadModelOnGpuFrom(const std::string& file) {
  std::cout << "WARNING: gpu delegate is not available on x86_64 platform, so all operations will be executed on CPU\n";
  model_ = tflite::FlatBufferModel::BuildFromFile(file.c_str());
  tflite::InterpreterBuilder(*model_, *operation_resolver_)(&interpreter_);
  if (!interpreter_) {
    throw std::runtime_error("failed to construct interpreter\n");
  }

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    throw std::runtime_error("failed to allocate tensors!\n");
  }
}

void TFLiteRuntime::makeGraphFrom(const OperationTraits& operation_traits) {
  converter_ = std::make_unique<NnfwOperationConverter>(operation_traits);
  model_ = converter_->generateTFLiteModel();
  tflite::InterpreterBuilder(*model_, *operation_resolver_)(&interpreter_);
  if (!interpreter_) {
    throw std::runtime_error("failed to construct interpreter\n");
  }

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    throw std::runtime_error("failed to allocate tensors!\n");
  }
}

void TFLiteRuntime::makeGraphOnGpuFrom(const OperationTraits& operation_traits) {
  std::cout << "WARNING: gpu delegate is not available on x86_64 platform, so all operations will be executed on CPU\n";
  converter_ = std::make_unique<NnfwOperationConverter>(operation_traits);
  model_ = converter_->generateTFLiteModel();
  tflite::InterpreterBuilder(*model_, *operation_resolver_)(&interpreter_);
  if (!interpreter_) {
    throw std::runtime_error("failed to construct interpreter\n");
  }

  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    throw std::runtime_error("failed to allocate tensors!\n");
  }
}