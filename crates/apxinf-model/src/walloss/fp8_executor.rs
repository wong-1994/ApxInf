//! Static-FP8 linear execution within the WallOSS BF16 residual stream.

use apxinf_core::{Error, Result, Tensor};

use super::backend::{kernels, Context};
use crate::vla::Fp8LinearWeights;

pub fn linear_bf16(
    context: &Context,
    input: &Tensor,
    input_scale: f32,
    weights: &Fp8LinearWeights,
) -> Result<Tensor> {
    if !input_scale.is_finite() || input_scale <= 0.0 {
        return Err(Error::Other(format!(
            "walloss FP8 activation scale must be finite and positive, got {input_scale}"
        )));
    }
    let quantized = kernels::quantization::quantize_bf16_e4m3(context, input, input_scale)?;
    let output = kernels::gemm::fp8(context, &quantized, input_scale, weights.as_kernel_view())?;
    kernels::quantization::cast_f16_bf16(context, &output)
}
