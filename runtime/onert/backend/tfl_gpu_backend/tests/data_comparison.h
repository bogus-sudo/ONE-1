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

#ifndef NNFW_DATA_COMPARISON_H
#define NNFW_DATA_COMPARISON_H

#include <vector>


bool isAlmostEqual(const std::vector<float>& array1, const std::vector<float>& array2, float threshold);

float maxDifferenceBetween(const std::vector<float>& array1, const std::vector<float>& array2);

#endif // NNFW_DATA_COMPARISON_H
