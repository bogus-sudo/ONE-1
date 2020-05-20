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

#ifndef NNFW_TFL_GPU_BACKEND_KERNEL_H
#define NNFW_TFL_GPU_BACKEND_KERNEL_H

#include <exec/IFunction.h>

#include "../operand/Tensor.h"

#include <memory>

class OperationTraits;
class NnfwOperationConverter;

namespace tflite {

class Interpreter;
class FlatBufferModel;

namespace ops {

namespace builtin {

class BuiltinOpResolver;

}

}

}

namespace onert
{

namespace backend
{

namespace tfl_gpu
{

namespace kernel
{

class Kernel: public onert::exec::IFunction {
public:
  Kernel(const OperationTraits& operation_traits);

  void shareBufferBetween(std::shared_ptr<operand::Tensor> external_tensor_presentation, uint64_t kernel_tensor_idx);

  void run() final;
  void runSync() final { run(); }

  ~Kernel();

private:
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<NnfwOperationConverter> converter_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::ops::builtin::BuiltinOpResolver> operation_resolver_;
  std::shared_ptr<operand::Tensor> _input;
  std::shared_ptr<operand::Tensor> _output;
};

} // namespace kernel

} // namespace tfl_gpu

} // namespace backend

} // namespace onert

#endif // NNFW_TFL_GPU_BACKEND_KERNEL_H
