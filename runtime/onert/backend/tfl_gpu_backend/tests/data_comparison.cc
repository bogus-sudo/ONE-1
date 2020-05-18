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

#include "data_comparison.h"

#include <cmath>


bool isAlmostEqual(const std::vector<float>& array1, const std::vector<float>& array2, float threshold) {
  if (array1.size() != array2.size()) { return false; }

  for (size_t i = 0, sz = array1.size(); i < sz; ++i) {
    if (std::abs(array1[i] - array2[i]) > threshold) {
      return false;
    }
  }

  return true;
}

float maxDifferenceBetween(const std::vector<float>& array1, const std::vector<float>& array2) {
  float max_diff = 0.0;
  for (size_t i = 0, sz = std::min(array1.size(), array2.size()); i < sz; ++i) {
    auto diff = std::abs(array1[i] - array2[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }

  return max_diff;
}
