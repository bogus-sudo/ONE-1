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

#include "gtest/gtest.h"

#include "test_models.h"

#include "working_with_nnfw_ir.h"

#include "tensorflow/lite/schema/schema_generated.h"


struct WorkingWithNnfwIrTests: public testing::Test {};

TEST_F(WorkingWithNnfwIrTests, OperationTraits_sequence_should_be_generated_using_topological_sorting) {
  auto nnfw_ir = makeModelWithTwoRhombsWithOneCommonEdge();
  std::vector<OperationTraits> operation_traits_sequence = generateTraitsOfOperationsFrom(nnfw_ir);

  ASSERT_EQ(operation_traits_sequence.size(), 6);
  ASSERT_TRUE(operation_traits_sequence[0].operationCode() == tflite::BuiltinOperator_RELU);
  ASSERT_TRUE(operation_traits_sequence[1].operationCode() == tflite::BuiltinOperator_RELU6);
  ASSERT_TRUE(operation_traits_sequence[2].operationCode() == tflite::BuiltinOperator_RELU6);
  ASSERT_TRUE(operation_traits_sequence[3].operationCode() == tflite::BuiltinOperator_RELU);
  ASSERT_TRUE(operation_traits_sequence[4].operationCode() == tflite::BuiltinOperator_ADD);
  ASSERT_TRUE(operation_traits_sequence[5].operationCode() == tflite::BuiltinOperator_ADD);
}
