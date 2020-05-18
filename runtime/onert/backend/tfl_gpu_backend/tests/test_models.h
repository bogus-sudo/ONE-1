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

#ifndef NNFW_TEST_MODELS_H
#define NNFW_TEST_MODELS_H

#include <memory>


namespace onert
{

namespace ir {

class Graph;

}

}

std::shared_ptr<onert::ir::Graph> makeModelWithJustOneReluOperation();
std::shared_ptr<onert::ir::Graph> makeModelWithJustOneRelu6Operation();
std::shared_ptr<onert::ir::Graph> makeModelWithJustOneAddOperation();
std::shared_ptr<onert::ir::Graph> makeModelWithJustOneConvolution2DOperation();
std::shared_ptr<onert::ir::Graph> makeModelWithJustOneDepthwiseConv2dOperation();
std::shared_ptr<onert::ir::Graph> makeModelWithJustOneAveragePool2dOperation();
std::shared_ptr<onert::ir::Graph> makeModelWithJustOneSqueezeOperation();
std::shared_ptr<onert::ir::Graph> makeModelWithJustOneSoftmaxOperation();
std::shared_ptr<onert::ir::Graph> makeModelWithTwoRhombsWithOneCommonEdge();

#endif // NNFW_TEST_MODELS_H
