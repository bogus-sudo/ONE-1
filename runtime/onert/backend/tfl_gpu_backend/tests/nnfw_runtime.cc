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

#include "nnfw_runtime.h"

#include "ir/Graph.h"
#include "compiler/Compiler.h"
#include "exec/Execution.h"

#include <iostream>


NNFWRuntime::NNFWRuntime() = default;
NNFWRuntime::~NNFWRuntime() = default;

void NNFWRuntime::doCalculationsUsingCpuBackend() {
  setenv("OP_BACKEND_ALLOPS", "cpu", 1);
}

void NNFWRuntime::doCalculationsUsingAclNeonBackend() {
  setenv("OP_BACKEND_ALLOPS", "acl_neon", 1);
}

void NNFWRuntime::doCalculationsUsingTflGpuBackend() {
  std::cout << "set up tfl_gpu backend" << std::endl;
  setenv("OP_BACKEND_ALLOPS", "tfl_gpu", 1);
}

void NNFWRuntime::loadGraph(std::shared_ptr<onert::ir::Graph> new_graph) {
  std::cout << "OP_BACKEND_ALLOPS: " << getenv("OP_BACKEND_ALLOPS") << std::endl;
  operations_graph_ = new_graph;

  auto subgs = std::make_shared<onert::ir::Subgraphs>();
  subgs->push(onert::ir::SubgraphIndex{0}, operations_graph_);
  onert::compiler::Compiler compiler(subgs);
  compiler.compile();

  std::shared_ptr<onert::exec::ExecutorMap> executors;
  compiler.release(executors);
  execution_.reset(new onert::exec::Execution(executors));
}


void NNFWRuntime::evaluate() {
  preparePlaceForOutput();
  execution_->execute();
}

void NNFWRuntime::preparePlaceForOutput() {
  output_data_.resize(operations_graph_->getOutputs().size());
  for (size_t i = 0, number_of_outputs = output_data_.size(); i < number_of_outputs; i++) {
    output_data_[i].resize(operations_graph_->operands().at(operations_graph_->getOutputs().at(i)).shape().num_elements());
    execution_->setOutput(onert::ir::IOIndex(i), output_data_[i].data(), output_data_[i].size() * sizeof(float));
  }
}

size_t NNFWRuntime::numberOfInputs() const {
  if (!operations_graph_) { throw std::logic_error("Neural network must be loaded before getting its characteristics"); }

  return operations_graph_->getInputs().size();
}

size_t NNFWRuntime::sizeOfInput(size_t input_idx) const {
  if (!operations_graph_) { throw std::logic_error("Neural network must be loaded before getting its characteristics"); }

  return operations_graph_->operands().at(operations_graph_->getInputs().at(input_idx)).shape().num_elements();
}

void NNFWRuntime::setDataForInput(size_t input_idx, const std::vector<float>& data) {
  if (!execution_) { throw std::logic_error("Neural network must be loaded before getting its characteristics"); }

  input_data_.push_back(data);
  execution_->setInput(onert::ir::IOIndex(input_idx), input_data_[input_idx].data(), input_data_[input_idx].size() * sizeof(float));
}

size_t NNFWRuntime::numberOfOutputs() const {
  if (!operations_graph_) { throw std::logic_error("Neural network must be loaded before getting its characteristics"); }

  return operations_graph_->getOutputs().size();
}

size_t NNFWRuntime::sizeOfOutput(size_t output_idx) const {
  if (!operations_graph_) { throw std::logic_error("Neural network must be loaded before getting its characteristics"); }

  return operations_graph_->operands().at(operations_graph_->getOutputs().at(output_idx)).shape().num_elements();
}

std::vector<float> NNFWRuntime::getDataOfOutput(size_t output_idx) const {
  return output_data_.at(output_idx);
}