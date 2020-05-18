/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef NNFW_TFL_GPU_CONFIG_H
#define NNFW_TFL_GPU_CONFIG_H

#include <backend/IConfig.h>
#include <memory>
#include <util/ITimer.h>

namespace onert
{
namespace backend
{
namespace tfl_gpu
{

class Config : public IConfig
{
public:
  std::string id() override { return "tfl_gpu"; }
  bool initialize() override { return true; }
  ir::Layout supportLayout(const ir::Operation & /* node */, ir::Layout /* frontend_layout */) override { return ir::Layout::NHWC; }
  bool supportPermutation() override { return false; }
  bool supportDynamicTensor() override { return false; }
  bool supportFP16() override { return false; }

  std::unique_ptr<util::ITimer> timer() override { return std::make_unique<util::CPUTimer>(); }
};

} // namespace cpu

} // namespace tfl_gpu

} // namespace onert


#endif // NNFW_TFL_GPU_CONFIG_H
