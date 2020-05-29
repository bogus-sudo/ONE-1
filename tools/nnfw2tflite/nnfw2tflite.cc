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

#include"program_options.h"
#include "data_filling.h"
#include "data_comparison.h"

#include "nnfw_runtime.h"
#include "tflite_runtime.h"

#include "working_with_nnfw_ir.h"

#include <iostream>
#include <chrono>


class Timer {
public:
  Timer(): start(std::chrono::high_resolution_clock::now()) {}

  long elapsed() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
  }

  void restart() {
    start = std::chrono::high_resolution_clock::now();
  }

private:
  std::chrono::high_resolution_clock::time_point start;
};


class Statistics {
public:
  void addMeasurment(long measurment) {
    if (min > measurment) {
      min = measurment;
    }
    if (max < measurment) {
      max = measurment;
    }

    sum += measurment;
    ++count;
  }

  long minimum() const { return min; }
  long maximum() const { return max; }
  float average() const { return sum / float(count); }

  void cleanup() {
    min = std::numeric_limits<long>::max();
    max = 0;
    sum = 0;
    count = 0;
  }

private:
  long min = std::numeric_limits<long>::max();
  long max = 0;
  long sum = 0;
  long count = 0;
};


int main(int argc, const char* argv[]) {
  setenv("BACKENDS", "cpu;acl_cl;acl_neon;tfl_gpu", 1);
  try {
    ProgramOptions options(argv, argc);
    options
      .define(Option("mobilenet").description("path to mobilenet network in TF Lite format"))
      .define(Option("repetitions").description("how many times rerun evaluations, can be from 1 to N"))
      .define(Option("use-cpu").description("can be 1 or 0"))
      .define(Option("use-acl-neon").description("can be 1 or 0"))
      .define(Option("use-acl-cl").description("can be 1 or 0"))
      .define(Option("use-tfl-gpu").description("can be 1 or 0"))
      .define(Option("use-gpu-for-tfl").description("can be 1 or 0"));

    std::string tflite_model_file = options.get_value_of_option("mobilenet");
    std::cout << "mobilenet model: " << tflite_model_file << std::endl;
    size_t repetitions = strtoul(options.get_value_of_option("repetitions").c_str(), nullptr, 10);
    bool use_cpu = options.get_value_of_option("use-cpu") == "1";
    bool use_acl_cl = options.get_value_of_option("use-acl-cl") == "1";
    bool use_acl_neon = options.get_value_of_option("use-acl-neon") == "1";
    bool use_tfl_gpu = options.get_value_of_option("use-tfl-gpu") == "1";
    bool use_gpu_for_tfl = options.get_value_of_option("use-gpu-for-tfl") == "1";

    Timer t;
    NNFWRuntime nnfw_runtime;
    if (use_cpu) {
      nnfw_runtime.runAllOperationsOnCpuBackend();
    }
    else if (use_acl_neon) {
      nnfw_runtime.runAllOperationsOnAclNeonBackend();
    }
    else if (use_acl_cl) {
      nnfw_runtime.runAllOperationsOnAclClBackend();
    }
    else if (use_tfl_gpu) {
      nnfw_runtime.runAllOperationsOnTflGpuBackend();
    }
    else {
      throw std::runtime_error("Backend for NNFW undefined");
    }
    nnfw_runtime.loadModelFrom(tflite_model_file);
    std::cout << "nnfw runtime prepare: " << t.elapsed() << " ms" << std::endl;

    t.restart();
    auto sequence_of_operation_traits = generateTraitsOfOperationsFrom(nnfw_runtime.getGraph());
    std::vector<std::unique_ptr<TFLiteRuntime>> converted_operations;
    for (const auto& operation_traits: sequence_of_operation_traits) {
      converted_operations.emplace_back(std::make_unique<TFLiteRuntime>());
      if (use_gpu_for_tfl) {
        converted_operations.back()->makeGraphOnGpuFrom(operation_traits);
      }
      else {
        converted_operations.back()->makeGraphFrom(operation_traits);
      }
    }
    std::cout << "conversion of " << converted_operations.size() << " nnfw operations to tflite models and prepare tflite runtimes for them: " << t.elapsed() << " ms" << std::endl;

    t.restart();
    TFLiteRuntime tflite_runtime;
    if (use_gpu_for_tfl) {
      tflite_runtime.loadModelOnGpuFrom(tflite_model_file);
    }
    else {
      tflite_runtime.loadModelFrom(tflite_model_file);
    }
    std::cout << "tflite runtime prepare: " << t.elapsed() << " ms" << std::endl;

    std::vector<std::vector<float>> input_data;
    for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
      input_data.emplace_back(randomData(nnfw_runtime.sizeOfInput(i)));
    }

    Statistics s;
    for (size_t i = 0; i < repetitions; ++i) {
      t.restart();
      for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
        nnfw_runtime.setDataForInput(i, input_data[i]);
      }
      nnfw_runtime.evaluate();
      s.addMeasurment(t.elapsed());
    }
    std::cout << "evaluate neural network " << repetitions << " times, using nnfw runtime - "
              << "max " << s.maximum() << " ms, min " << s.minimum() << " ms, avg " << s.average() << " ms" << std::endl;

    s.cleanup();
    for (size_t i = 0; i < repetitions; ++i) {
      t.restart();
      for (size_t i = 0; i < converted_operations[0]->numberOfInputs(); ++i) {
        converted_operations[0]->setDataForInput(i, input_data[i]);
      }
      for (size_t i = 0; i < converted_operations.size() - 1; ++i) {
        converted_operations[i]->evaluate();
        for (size_t j = 0; j < converted_operations[i + 1]->numberOfInputs(); ++j) {
          converted_operations[i + 1]->setDataForInput(j, converted_operations[i]->getDataOfOutput(j));
        }
      }
      converted_operations[converted_operations.size() - 1]->evaluate();
      s.addMeasurment(t.elapsed());
    }
    std::cout << "evaluate converted operations " << repetitions << " times, using tflite runtimes - "
        << "max " << s.maximum() << " ms, min " << s.minimum() << " ms, avg " << s.average() << " ms" << std::endl;


    s.cleanup();
    for (size_t i = 0; i < repetitions; ++i) {
      t.restart();
      for (size_t i = 0; i < tflite_runtime.numberOfInputs(); ++i) {
        tflite_runtime.setDataForInput(i, input_data[i]);
      }
      tflite_runtime.evaluate();
      s.addMeasurment(t.elapsed());
    }
    std::cout << "evaluate neural network " << repetitions << " times, using tflite runtime - "
        << "max " << s.maximum() << " ms, min " << s.minimum() << " ms, avg " << s.average() << " ms" << std::endl;



    std::cout << "compare outputs of NNFW runtime and sequence of converted operations...\n";
    float max_difference = 0.0f;
    for (size_t i = 0; i < nnfw_runtime.numberOfInputs(); ++i) {
      auto diff = maxDifferenceBetween(nnfw_runtime.getDataOfOutput(i), converted_operations[converted_operations.size()-1]->getDataOfOutput(i));
      if (diff > max_difference) {
        max_difference = diff;
      }
    }

    std::cout << "max diff: " << max_difference << std::endl;

    const float DIFFERENCE_THRESHOLD = 10e-5;
    if (max_difference > DIFFERENCE_THRESHOLD) {
      std::cout << " Outputs is not equal!" << std::endl;
    }
    else {
      std::cout << "Outputs is equal!" << std::endl;
    }

    std::cout << "compare outputs of sequence of converted operations and tflite runtime...\n";
    max_difference = 0.0f;
    for (size_t i = 0; i < tflite_runtime.numberOfInputs(); ++i) {
      auto diff = maxDifferenceBetween(tflite_runtime.getDataOfOutput(i), converted_operations[converted_operations.size()-1]->getDataOfOutput(i));
      if (diff > max_difference) {
        max_difference = diff;
      }
    }

    std::cout << "max diff: " << max_difference << std::endl;

    if (max_difference > DIFFERENCE_THRESHOLD) {
      std::cout << " Outputs is not equal!" << std::endl;
    }
    else {
      std::cout << "Outputs is equal!" << std::endl;
    }

    std::cout << "compare outputs of nnfw runtime and tflite runtime...\n";
    max_difference = 0.0f;
    for (size_t i = 0; i < tflite_runtime.numberOfInputs(); ++i) {
      auto diff = maxDifferenceBetween(tflite_runtime.getDataOfOutput(i), nnfw_runtime.getDataOfOutput(i));
      if (diff > max_difference) {
        max_difference = diff;
      }
    }

    std::cout << "max diff: " << max_difference << std::endl;

    if (max_difference > DIFFERENCE_THRESHOLD) {
      std::cout << " Outputs is not equal!" << std::endl;
    }
    else {
      std::cout << "Outputs is equal!" << std::endl;
    }
  }
  catch(const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << std::endl;
  }
  catch(...) {
    std::cerr << "Unexpected error" << std::endl;
  }

  return 0;
}