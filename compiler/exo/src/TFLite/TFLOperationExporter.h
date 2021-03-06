/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __TFL_OPERATION_EXPORTER_H__
#define __TFL_OPERATION_EXPORTER_H__

#include "TFLExporterUtils.h"

#include <loco/IR/Graph.h>

namespace exo
{
namespace tflite_detail
{

/**
 * @brief create Operators corresponding to model nodes
 * @param nodes container with nodes
 * @param gd information about serializer parts of model
 */
void exportNodes(loco::Graph *g, flatbuffers::FlatBufferBuilder &builder, SerializedModelData &gd);

} // namespace tflite_detail
} // namespace exo

#endif // __TFL_OPERATION_EXPORTER_H__
