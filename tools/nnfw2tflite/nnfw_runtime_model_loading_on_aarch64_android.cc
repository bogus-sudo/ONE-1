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

#include "tflite_loader.h"
#include "ir/Graph.h"
#include "compiler/Compiler.h"
#include "exec/Execution.h"


void NNFWRuntime::runAllOperationsOnCpuBackend() {
  setenv("OP_BACKEND_ALLOPS", "cpu", 1);
}

void NNFWRuntime::runAllOperationsOnAclNeonBackend() {
  setenv("OP_BACKEND_ALLOPS", "acl_neon", 1);
}

void NNFWRuntime::runAllOperationsOnAclClBackend() {
  setenv("OP_BACKEND_ALLOPS", "acl_cl", 1);
}

void NNFWRuntime::runAllOperationsOnTflGpuBackend() {
  setenv("OP_BACKEND_ALLOPS", "tfl_gpu", 1);
}
