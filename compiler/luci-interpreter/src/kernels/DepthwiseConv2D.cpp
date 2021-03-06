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

#include "kernels/DepthwiseConv2D.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h>
#include <tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h>

#include <stdexcept>

namespace luci_interpreter
{
namespace kernels
{

DepthwiseConv2D::DepthwiseConv2D(const Tensor *input, const Tensor *filter, const Tensor *bias,
                                 Tensor *output, const DepthwiseConv2DParams &params)
    : KernelWithParams<DepthwiseConv2DParams>(params), _input(input), _filter(filter), _bias(bias),
      _output(output)
{
}

void DepthwiseConv2D::configure()
{
  // TensorFlow Lite (as of v2.2.0) supports the following combinations of types:
  //     | input filter bias  output |
  // ----+---------------------------+
  // (1) | float float  float float  |
  // (2) | float int8   float float  | hybrid
  // (3) | uint8 uint8  int32 uint8  | quantized
  // (4) | int8  int8   int32 int8   | quantized per channel
  // (5) | int16 int8   int64 int16  | quantized per channel 16x8
  //
  // We only support (1) and (3) for now.
  if (_input->element_type() == DataType::FLOAT32 && _filter->element_type() == DataType::FLOAT32)
  {
    assert(_bias == nullptr || _bias->element_type() == DataType::FLOAT32);
  }
  else if (_input->element_type() == DataType::U8 && _filter->element_type() == DataType::U8)
  {
    assert(_bias == nullptr || _bias->element_type() == DataType::S32);
  }
  else
  {
    throw std::runtime_error("Unsupported type.");
  }
  assert(_output->element_type() == _input->element_type());

  const Shape &input_shape = _input->shape();
  const Shape &filter_shape = _filter->shape();
  assert(input_shape.num_dims() == 4 && filter_shape.num_dims() == 4);

  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  // Filter format: [1, H, W, O].
  assert(filter_shape.dim(0) == 1);
  const int32_t filter_height = filter_shape.dim(1);
  const int32_t filter_width = filter_shape.dim(2);
  const int32_t channels_out = filter_shape.dim(3);

  assert(_bias == nullptr ||
         (_bias->shape().num_dims() == 1 && _bias->shape().dim(0) == channels_out));

  const int32_t output_height =
      computeOutputSize(_params.padding, input_height, filter_height, _params.stride_height,
                        _params.dilation_height_factor);
  const int32_t output_width =
      computeOutputSize(_params.padding, input_width, filter_width, _params.stride_width,
                        _params.dilation_width_factor);

  _padding_height = computePadding(_params.stride_height, _params.dilation_height_factor,
                                   input_height, filter_height, output_height);
  _padding_width = computePadding(_params.stride_width, _params.dilation_width_factor, input_width,
                                  filter_width, output_width);

  _output->resize({batches, output_height, output_width, channels_out});
}

void DepthwiseConv2D::execute() const
{
  switch (_input->element_type())
  {
    case DataType::FLOAT32:
      if (_filter->element_type() == DataType::FLOAT32)
      {
        evalFloat();
        break;
      }
      throw std::runtime_error("Unsupported type.");
    case DataType::U8:
      evalQuantized();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void DepthwiseConv2D::evalFloat() const
{
  float activation_min{};
  float activation_max{};
  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  tflite::DepthwiseParams params{};
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = _params.stride_height;
  params.stride_width = _params.stride_width;
  params.dilation_height_factor = _params.dilation_height_factor;
  params.dilation_width_factor = _params.dilation_width_factor;
  params.depth_multiplier = _params.depth_multiplier;
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;

  tflite::reference_ops::DepthwiseConv(params, getTensorShape(_input), getTensorData<float>(_input),
                                       getTensorShape(_filter), getTensorData<float>(_filter),
                                       getTensorShape(_bias), getTensorData<float>(_bias),
                                       getTensorShape(_output), getTensorData<float>(_output));
}

void DepthwiseConv2D::evalQuantized() const
{
  const auto input_scale = static_cast<double>(_input->scale());
  const auto filter_scale = static_cast<double>(_filter->scale());
  const auto output_scale = static_cast<double>(_output->scale());

  const double real_multiplier = input_scale * filter_scale / output_scale;
  int32_t output_multiplier{};
  int output_shift{};
  quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);

  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(_params.activation, _output, &activation_min, &activation_max);

  tflite::DepthwiseParams params{};
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = _params.stride_height;
  params.stride_width = _params.stride_width;
  params.dilation_height_factor = _params.dilation_height_factor;
  params.dilation_width_factor = _params.dilation_width_factor;
  params.depth_multiplier = _params.depth_multiplier;
  // The kernel expects input and filter zero points to be negated.
  params.input_offset = -_input->zero_point();    // Note the '-'.
  params.weights_offset = -_filter->zero_point(); // Note the '-'.
  params.output_offset = _output->zero_point();
  params.output_multiplier = output_multiplier;
  params.output_shift = output_shift;
  params.quantized_activation_min = activation_min;
  params.quantized_activation_max = activation_max;

  tflite::reference_ops::DepthwiseConv(
      params, getTensorShape(_input), getTensorData<uint8_t>(_input), getTensorShape(_filter),
      getTensorData<uint8_t>(_filter), getTensorShape(_bias), getTensorData<int32_t>(_bias),
      getTensorShape(_output), getTensorData<uint8_t>(_output));
}

} // namespace kernels
} // namespace luci_interpreter
