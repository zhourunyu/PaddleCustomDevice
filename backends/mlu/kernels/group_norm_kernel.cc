// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernels/funcs/mlu_baseop.h"

namespace custom_kernel {

template <typename T, typename Context>
void GroupNormKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const paddle::optional<phi::DenseTensor>& scale_opt,
                     const paddle::optional<phi::DenseTensor>& bias_opt,
                     float epsilon,
                     int groups,
                     const std::string& data_layout,
                     phi::DenseTensor* out,
                     phi::DenseTensor* mean,
                     phi::DenseTensor* variance) {
  auto* scale = scale_opt.get_ptr();
  auto* bias = bias_opt.get_ptr();
  cnnlTensorLayout_t layout = (data_layout == "NHWC") ? CNNL_LAYOUT_NHWC
                                                  : CNNL_LAYOUT_NCHW;

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(mean);
  dev_ctx.template Alloc<T>(variance);

  MLUCnnlTensorDesc x_desc(x, layout, ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(*out, layout, ToCnnlDataType<T>());
  MLUCnnlTensorDesc mean_var_desc(*mean);

  const auto& x_dims = x.dims();
  int channel_axis = (data_layout == "NHWC") ? x_dims.size() - 1 : 1;
  std::vector<int> scale_bias_axes = {x_dims[channel_axis]};
  // cnnl only support both of scale and bias is NULL or not.
  if (!scale && !bias) {
    MLUCnnl::GroupNormForward(dev_ctx,
                              x_desc.get(),
                              GetBasePtr(&x),
                              nullptr /*scale_bias_desc*/,
                              nullptr /*scale*/,
                              nullptr /*bias*/,
                              epsilon,
                              groups,
                              y_desc.get(),
                              GetBasePtr(out),
                              mean_var_desc.get(),
                              GetBasePtr(mean),
                              GetBasePtr(variance));
  } else {
    Tensor tmp_scale;
    if (!scale) {
      tmp_scale.Resize(phi::make_ddim(scale_bias_axes));
      dev_ctx.template Alloc<T>(&tmp_scale);
      FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(1), &tmp_scale);
    } else {
      tmp_scale = *scale;
    }

    Tensor tmp_bias;
    if (!bias) {
      tmp_bias.Resize(phi::make_ddim(scale_bias_axes));
      dev_ctx.template Alloc<T>(&tmp_bias);
      FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0), &tmp_bias);
    } else {
      tmp_bias = *bias;
    }

    // scale and bias should have same type with x/out
    MLUCnnlTensorDesc float32_desc(
        scale_bias_axes.size(), scale_bias_axes.data(), CNNL_DTYPE_FLOAT);
    MLUCnnlTensorDesc float16_desc(
        scale_bias_axes.size(), scale_bias_axes.data(), CNNL_DTYPE_HALF);
    cnnlCastDataType_t cast_type =
        GetCastDataType(DataType::FLOAT32, DataType::FLOAT16);

    Tensor final_scale;
    if (x.dtype() == DataType::FLOAT16 &&
        tmp_scale.dtype() == DataType::FLOAT32) {
      final_scale.Resize(phi::make_ddim(scale_bias_axes));
      dev_ctx.template Alloc<T>(&final_scale);
      // cast scale to fp16
      MLUCnnl::Cast(dev_ctx,
                    cast_type,
                    float32_desc.get(),
                    GetBasePtr(&tmp_scale),
                    float16_desc.get(),
                    GetBasePtr(&final_scale));
    } else {
      final_scale = tmp_scale;
    }

    Tensor final_bias;
    if (x.dtype() == DataType::FLOAT16 &&
        tmp_bias.dtype() == DataType::FLOAT32) {
      final_bias.Resize(phi::make_ddim(scale_bias_axes));
      dev_ctx.template Alloc<T>(&final_bias);
      // cast bias to fp16
      MLUCnnl::Cast(dev_ctx,
                    cast_type,
                    float32_desc.get(),
                    GetBasePtr(&tmp_bias),
                    float16_desc.get(),
                    GetBasePtr(&final_bias));
    } else {
      final_bias = tmp_bias;
    }

    MLUCnnlTensorDesc scale_bias_desc(
        scale_bias_axes.size(), scale_bias_axes.data(), ToCnnlDataType<T>());
    MLUCnnl::GroupNormForward(dev_ctx,
                              x_desc.get(),
                              GetBasePtr(&x),
                              scale_bias_desc.get(),
                              GetBasePtr(&final_scale),
                              GetBasePtr(&final_bias),
                              epsilon,
                              groups,
                              y_desc.get(),
                              GetBasePtr(out),
                              mean_var_desc.get(),
                              GetBasePtr(mean),
                              GetBasePtr(variance));
  }
}

template <typename T, typename Context>
void GroupNormGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const paddle::optional<phi::DenseTensor>& scale_opt,
                         const paddle::optional<phi::DenseTensor>& bias_opt,
                         const phi::DenseTensor& out,
                         const phi::DenseTensor& mean,
                         const phi::DenseTensor& variance,
                         const phi::DenseTensor& out_grad,
                         float epsilon,
                         int groups,
                         const std::string& data_layout,
                         phi::DenseTensor* x_grad,
                         phi::DenseTensor* scale_grad,
                         phi::DenseTensor* bias_grad) {
  auto* scale = scale_opt.get_ptr();
  dev_ctx.template Alloc<T>(x_grad);

  cnnlTensorLayout_t layout = (data_layout == "NHWC") ? CNNL_LAYOUT_NHWC
                                                  : CNNL_LAYOUT_NCHW;
  const auto& x_dims = x.dims();
  int channel_axis = (data_layout == "NHWC") ? x_dims.size() - 1 : 1;
  int batch_size = x_dims[0], channels = x_dims[channel_axis];
  std::vector<int> scale_bias_axes = {channels};

  MLUCnnlTensorDesc x_desc(x, layout, ToCnnlDataType<T>());
  MLUCnnlTensorDesc dy_desc(out_grad, layout, ToCnnlDataType<T>());
  MLUCnnlTensorDesc mean_var_desc(mean, layout, ToCnnlDataType<T>());
  MLUCnnlTensorDesc dx_desc(*x_grad, layout, ToCnnlDataType<T>());

  Tensor tmp_scale;
  if (!scale) {
    tmp_scale.Resize(phi::make_ddim(scale_bias_axes));
    dev_ctx.template Alloc<T>(&tmp_scale);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(1), &tmp_scale);
  } else {
    tmp_scale = *scale;
  }

  MLUCnnlTensorDesc float32_desc(
      scale_bias_axes.size(), scale_bias_axes.data(), CNNL_DTYPE_FLOAT);
  MLUCnnlTensorDesc float16_desc(
      scale_bias_axes.size(), scale_bias_axes.data(), CNNL_DTYPE_HALF);
  cnnlCastDataType_t cast_fp32_to_fp16 =
      GetCastDataType(DataType::FLOAT32, DataType::FLOAT16);
  cnnlCastDataType_t cast_fp16_to_fp32 =
      GetCastDataType(DataType::FLOAT16, DataType::FLOAT32);

  Tensor final_scale;
  if (x.dtype() == DataType::FLOAT16 &&
      tmp_scale.dtype() == DataType::FLOAT32) {
    final_scale.Resize(phi::make_ddim(scale_bias_axes));
    dev_ctx.template Alloc<T>(&final_scale);
    // cast scale to fp16
    MLUCnnl::Cast(dev_ctx,
                  cast_fp32_to_fp16,
                  float32_desc.get(),
                  GetBasePtr(&tmp_scale),
                  float16_desc.get(),
                  GetBasePtr(&final_scale));
  } else {
    final_scale = tmp_scale;
  }

  Tensor tmp_dscale;
  if (scale_grad && (x.dtype() == scale_grad->dtype())) {
    dev_ctx.template Alloc<T>(scale_grad);
    tmp_dscale = *scale_grad;
  } else {
    tmp_dscale.Resize(phi::make_ddim(scale_bias_axes));
    dev_ctx.template Alloc<T>(&tmp_dscale);
  }
  Tensor tmp_dbias;
  if (bias_grad && (x.dtype() == bias_grad->dtype())) {
    dev_ctx.template Alloc<T>(bias_grad);
    tmp_dbias = *bias_grad;
  } else {
    tmp_dbias.Resize(phi::make_ddim(scale_bias_axes));
    dev_ctx.template Alloc<T>(&tmp_dbias);
  }

  MLUCnnlTensorDesc scale_desc(
      scale_bias_axes.size(), scale_bias_axes.data(), ToCnnlDataType<T>());
  MLUCnnl::GroupNormBackward(dev_ctx,
                             x_desc.get(),
                             GetBasePtr(&x),
                             dy_desc.get(),
                             GetBasePtr(&out_grad),
                             scale_desc.get(),
                             GetBasePtr(&final_scale),
                             mean_var_desc.get(),
                             GetBasePtr(&mean),
                             GetBasePtr(&variance),
                             groups,
                             batch_size * channels,
                             dx_desc.get(),
                             GetBasePtr(x_grad),
                             GetBasePtr(&tmp_dscale),
                             GetBasePtr(&tmp_dbias));

  if (scale_grad && (tmp_dscale.dtype() == DataType::FLOAT16 &&
                     scale_grad->dtype() == DataType::FLOAT32)) {
    dev_ctx.template Alloc<float>(scale_grad);
    MLUCnnl::Cast(dev_ctx,
                  cast_fp16_to_fp32,
                  float16_desc.get(),
                  GetBasePtr(&tmp_dscale),
                  float32_desc.get(),
                  GetBasePtr(scale_grad));
  }
  if (bias_grad && (tmp_dbias.dtype() == DataType::FLOAT16 &&
                    bias_grad->dtype() == DataType::FLOAT32)) {
    dev_ctx.template Alloc<float>(bias_grad);
    MLUCnnl::Cast(dev_ctx,
                  cast_fp16_to_fp32,
                  float16_desc.get(),
                  GetBasePtr(&tmp_dbias),
                  float32_desc.get(),
                  GetBasePtr(bias_grad));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(group_norm,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GroupNormKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(group_norm_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GroupNormGradKernel,
                          float,
                          phi::dtype::float16) {}
