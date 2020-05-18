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

#include "data_filling.h"

#include <random>
#include <chrono>


class RandomGenerator {
public:
  RandomGenerator(float mean, float stddev)
      : engine_(std::chrono::system_clock::now().time_since_epoch().count())
      , distribution_(mean, stddev)
  {}

  float operator()() { return distribution_(engine_); }

private:
  std::minstd_rand engine_;
  std::normal_distribution<float> distribution_;
};

float randomValue() {
  static RandomGenerator random(0.0f, 2.0f);

  return random();
}

std::vector<float> randomData(size_t size) {
  std::vector<float> result;
  while (size-- > 0) {
    result.push_back(randomValue());
  }

  return result;
}

std::vector<float> Ones(size_t size) {
  std::vector<float> result;
  while (size-- > 0) {
    result.push_back(1.f);
  }

  return result;
}

std::vector<float> Zeros(size_t size) {
  std::vector<float> result;
  while (size-- > 0) {
    result.push_back(0.f);
  }

  return result;
}
