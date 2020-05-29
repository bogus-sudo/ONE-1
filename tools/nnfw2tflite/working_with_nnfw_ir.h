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

#ifndef NNFW_WORKING_WITH_NNFW_IR_H
#define NNFW_WORKING_WITH_NNFW_IR_H

#include "operation_traits.h"

#include <vector>
#include <memory>


namespace onert
{

namespace ir {

class Graph;

}

}

std::vector<OperationTraits> generateTraitsOfOperationsFrom(std::shared_ptr<onert::ir::Graph> nnfw_ir);

#endif // NNFW_WORKING_WITH_NNFW_IR_H
