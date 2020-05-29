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

#ifndef NNFW_NNFW_RUNTIME_H
#define NNFW_NNFW_RUNTIME_H

#include <memory>
#include <vector>

namespace onert
{

namespace ir {

class Graph;

}

namespace exec {

class Execution;

}

}


class NNFWRuntime {
public:
  NNFWRuntime();

  void runAllOperationsOnCpuBackend();
  void runAllOperationsOnAclNeonBackend();
  void runAllOperationsOnAclClBackend();
  void runAllOperationsOnTflGpuBackend();
  void loadGraph(std::shared_ptr<onert::ir::Graph> new_graph);
  std::shared_ptr<onert::ir::Graph> getGraph() const;
  void loadModelFrom(const std::string& file);

  size_t numberOfInputs() const;
  size_t sizeOfInput(size_t input_idx) const;
  void setDataForInput(size_t input_idx, const std::vector<float>& data);
  size_t numberOfOutputs() const;
  size_t sizeOfOutput(size_t output_idx) const;
  std::vector<float> getDataOfOutput(size_t output_idx) const;

  void evaluate();

  ~NNFWRuntime();

private:
  void preparePlaceForOutput();

private:
  std::vector<std::vector<float>> input_data_;
  std::vector<std::vector<float>> output_data_;
  std::shared_ptr<onert::ir::Graph> operations_graph_;
  std::unique_ptr<onert::exec::Execution> execution_;
};

#endif //NNFW_NNFW_RUNTIME_H
