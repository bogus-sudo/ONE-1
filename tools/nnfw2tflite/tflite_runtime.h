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

#ifndef NNFW_TFLITE_RUNTIME_H
#define NNFW_TFLITE_RUNTIME_H

#include <string>
#include <vector>
#include <memory>

#include "tensorflow/lite/c/common.h"


namespace tflite {

class Interpreter;
class FlatBufferModel;

namespace ops {

namespace builtin {

class BuiltinOpResolver;

}

}

}

class TFLiteRuntime {
public:
  TFLiteRuntime();

  void loadModelFrom(const std::string& file);
  void loadModelOnGpuFrom(const std::string& file);
  void makeGraphFrom(const class OperationTraits& operation_traits);
  void makeGraphOnGpuFrom(const class OperationTraits& operation_traits);

  size_t numberOfInputs() const;
  size_t sizeOfInput(size_t input_idx) const;
  void setDataForInput(size_t input_idx, const std::vector<float>& data);
  size_t numberOfOutputs() const;
  size_t sizeOfOutput(size_t output_idx) const;
  std::vector<float> getDataOfOutput(size_t output_idx) const;

  void evaluate();

  ~TFLiteRuntime();

private:
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<class NnfwOperationConverter> converter_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> operation_resolver_;
  std::unique_ptr<TfLiteDelegate> gpu_delegate_;
};

#endif //NNFW_TFLITE_RUNTIME_H
