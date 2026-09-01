//! Device-ready dynamic and calibrated FP8 linear weights.

#[cfg(test)]
use apxinf_core::DType;
use apxinf_core::{Backend, Error, Result, Tensor};

#[cfg(feature = "cuda")]
use super::backend::{kernels, RuntimeBackend};

use super::{encode_e4m3, quantize_e4m3_absmax, E4M3_MAX};

#[derive(Debug)]
pub struct DynamicFp8LinearWeights {
    /// Contiguous output-major physical `[output, input]` E4M3 matrix.
    pub weight: Tensor,
    /// FP32 scale vector with one element per output channel.
    pub channel_scales: Tensor,
    /// Logical input width before the resident matrix was aligned.
    pub input_features: usize,
    /// Logical output width before the resident matrix was aligned.
    pub output_features: usize,
}

impl DynamicFp8LinearWeights {
    #[cfg(feature = "cuda")]
    pub fn as_kernel_view(&self) -> kernels::gemm::DynamicFp8WeightView<'_> {
        kernels::gemm::DynamicFp8WeightView {
            values_e4m3: &self.weight,
            channel_scales: &self.channel_scales,
        }
    }

    pub fn from_host(weight: &Tensor, backend: &dyn Backend) -> Result<Self> {
        let shape = weight.shape().dims();
        if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
            return Err(Error::Other(format!(
                "dynamic FP8 weight must be a non-empty 2D matrix, got {shape:?}"
            )));
        }

        let input_features = shape[0];
        let output_features = shape[1];
        let padded_input_features = align_16(input_features);
        let padded_output_features = align_16(output_features);
        let padded_weight =
            pad_transpose_linear_weight(weight, padded_output_features, padded_input_features)?;

        #[cfg(feature = "cuda")]
        let (weight, channel_scales) = if let Some(cuda_backend) =
            backend.as_any().downcast_ref::<RuntimeBackend>()
        {
            let values = padded_weight
                .to_f32_vec()?
                .into_iter()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>();
            let host_bf16 =
                Tensor::from_bf16(vec![padded_output_features, padded_input_features], &values)?;
            let device_bf16 = backend.to_device(&host_bf16)?;
            let quantized = kernels::quantization::quantize_rows_bf16_e4m3(
                cuda_backend.context(),
                &device_bf16,
            )?;
            (quantized.values, quantized.scales)
        } else {
            quantize_rows_host(&padded_weight, backend)?
        };
        #[cfg(not(feature = "cuda"))]
        let (weight, channel_scales) = quantize_rows_host(&padded_weight, backend)?;

        Ok(Self {
            weight,
            channel_scales,
            input_features,
            output_features,
        })
    }
}
fn align_16(value: usize) -> usize {
    value.div_ceil(16) * 16
}

fn pad_transpose_linear_weight(
    weight: &Tensor,
    padded_rows: usize,
    padded_cols: usize,
) -> Result<Tensor> {
    let shape = weight.shape().dims();
    let (input_features, output_features) = (shape[0], shape[1]);
    let source = weight.to_f32_vec()?;
    let mut padded = vec![0.0f32; padded_rows * padded_cols];
    for input in 0..input_features {
        for output in 0..output_features {
            padded[output * padded_cols + input] = source[input * output_features + output];
        }
    }
    Tensor::from_f32(vec![padded_rows, padded_cols], &padded)
}

fn quantize_rows_host(weight: &Tensor, backend: &dyn Backend) -> Result<(Tensor, Tensor)> {
    let shape = weight.shape().dims();
    let (rows, cols) = (shape[0], shape[1]);
    let source = weight.to_f32_vec()?;
    let mut scales = vec![1.0e-12f32; rows];
    for row in 0..rows {
        for col in 0..cols {
            scales[row] = scales[row].max(source[row * cols + col].abs() / E4M3_MAX);
        }
    }
    let values = source
        .iter()
        .enumerate()
        .map(|(index, value)| encode_e4m3(*value / scales[index / cols]))
        .collect::<Vec<_>>();
    Ok((
        backend.to_device(&Tensor::from_f8_e4m3(shape.to_vec(), &values)?)?,
        backend.to_device(&Tensor::from_f32(vec![rows], &scales)?)?,
    ))
}

#[derive(Debug)]
pub struct Fp8LinearWeights {
    /// `[input, output]` CUDA E4M3 matrix.
    pub weight: Tensor,
    pub weight_scale: f32,
    /// Physical [gate256,up256] column order for exact dual-GeGLU paths.
    /// When true this tensor must never be sent to a plain GEMM.
    pub dual_geglu_interleaved: bool,
    /// Additional [gate256,up256] resident matrix used only by auto routing.
    /// routing. The primary `weight` remains plain so every other shape and
    /// backend keeps its original physical contract.
    pub dual_geglu_auto_interleaved: Option<Tensor>,
}

impl Fp8LinearWeights {
    #[cfg(feature = "cuda")]
    pub fn as_kernel_view(&self) -> kernels::gemm::Fp8WeightView<'_> {
        kernels::gemm::Fp8WeightView {
            values_e4m3: &self.weight,
            scale: self.weight_scale,
            dual_geglu_interleaved: self.dual_geglu_interleaved,
            dual_geglu_auto_interleaved: self.dual_geglu_auto_interleaved.as_ref(),
        }
    }

    pub fn from_host(weight: &Tensor, backend: &dyn Backend) -> Result<Self> {
        Self::from_host_parts(&[weight], backend)
    }

    /// Concatenate projections along their output dimension before applying
    /// one absmax quantization scale. This produces graph-ready QKV and
    /// gate/up matrices without runtime concatenation or mixed descales.
    pub fn from_host_parts(weights: &[&Tensor], backend: &dyn Backend) -> Result<Self> {
        Self::from_host_parts_with_dual_layout(weights, backend, true)
    }

    pub(crate) fn from_host_parts_with_dual_layout(
        weights: &[&Tensor],
        backend: &dyn Backend,
        allow_dual_layout: bool,
    ) -> Result<Self> {
        if weights.is_empty() {
            return Err(Error::Other("cannot pack an empty FP8 linear group".into()));
        }
        let fp8_dual_geglu_mode = fp8_dual_geglu_mode()?;
        let dual_geglu_exact = allow_dual_layout
            && weights.len() == 2
            && weights
                .iter()
                .all(|weight| weight.shape().dims() == [2048, 16384]);
        let dual_geglu_interleaved =
            dual_geglu_exact && fp8_dual_geglu_mode == Fp8DualGeGluMode::On;
        let plain_host = concat_host_2d(weights)?;
        let interleaved_host = if dual_geglu_exact && fp8_dual_geglu_mode != Fp8DualGeGluMode::Off {
            Some(interleave_gate_up_host(weights[0], weights[1], 256)?)
        } else {
            None
        };
        let weight_host = if dual_geglu_interleaved {
            interleaved_host.as_ref().unwrap()
        } else {
            &plain_host
        };
        #[cfg(feature = "cuda")]
        let (weight, weight_scale) =
            if let Some(cuda_backend) = backend.as_any().downcast_ref::<RuntimeBackend>() {
                // Quantizing billions of parameters with the scalar CPU E4M3
                // encoder is prohibitively slow on Jetson. Upload FP16 once and
                // let the CUDA conversion kernel produce the resident FP8 matrix.
                let (weight_f16, amax) = fp16_host_and_amax(weight_host)?;
                let weight_scale = if amax == 0.0 { 1.0 } else { amax / E4M3_MAX };
                let weight_f16 = backend.to_device(&weight_f16)?;
                let weight = kernels::quantization::quantize_f16_e4m3(
                    cuda_backend.context(),
                    &weight_f16,
                    weight_scale,
                )?;
                (weight, weight_scale)
            } else {
                let quantized = quantize_e4m3_absmax(weight_host)?;
                (backend.to_device(&quantized.values)?, quantized.scale)
            };
        #[cfg(not(feature = "cuda"))]
        let (weight, weight_scale) = {
            let quantized = quantize_e4m3_absmax(weight_host)?;
            (backend.to_device(&quantized.values)?, quantized.scale)
        };
        let dual_geglu_auto_interleaved = if dual_geglu_exact
            && fp8_dual_geglu_mode == Fp8DualGeGluMode::Auto
        {
            let interleaved_host = interleaved_host.as_ref().unwrap();
            #[cfg(feature = "cuda")]
            let (interleaved, interleaved_scale) =
                if let Some(cuda_backend) = backend.as_any().downcast_ref::<RuntimeBackend>() {
                    let (weight_f16, amax) = fp16_host_and_amax(interleaved_host)?;
                    let interleaved_scale = if amax == 0.0 { 1.0 } else { amax / E4M3_MAX };
                    let weight_f16 = backend.to_device(&weight_f16)?;
                    let interleaved = kernels::quantization::quantize_f16_e4m3(
                        cuda_backend.context(),
                        &weight_f16,
                        weight_scale,
                    )?;
                    (interleaved, interleaved_scale)
                } else {
                    let quantized = quantize_e4m3_absmax(interleaved_host)?;
                    (backend.to_device(&quantized.values)?, quantized.scale)
                };
            #[cfg(not(feature = "cuda"))]
            let (interleaved, interleaved_scale) = {
                let quantized = quantize_e4m3_absmax(interleaved_host)?;
                (backend.to_device(&quantized.values)?, quantized.scale)
            };
            if weight_scale.to_bits() != interleaved_scale.to_bits() {
                return Err(Error::Other(format!(
                    "FP8 dual GeGLU auto layouts changed joint scale bits: plain={:#010x}, interleaved={:#010x}",
                    weight_scale.to_bits(),
                    interleaved_scale.to_bits()
                )));
            }
            Some(interleaved)
        } else {
            None
        };
        Ok(Self {
            weight,
            weight_scale,
            dual_geglu_interleaved,
            dual_geglu_auto_interleaved,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fp8DualGeGluMode {
    Auto,
    Off,
    On,
}

fn parse_fp8_dual_geglu_mode(value: Option<&str>) -> Result<Fp8DualGeGluMode> {
    match value {
        None | Some("auto") => Ok(Fp8DualGeGluMode::Auto),
        Some("0" | "off") => Ok(Fp8DualGeGluMode::Off),
        Some("1" | "on") => Ok(Fp8DualGeGluMode::On),
        Some(value) => Err(Error::Other(format!(
            "APXINF_WALLOSS_FP8_DUAL_GEGLU must be auto, 0/off, or 1/on; got {value}"
        ))),
    }
}

fn fp8_dual_geglu_mode() -> Result<Fp8DualGeGluMode> {
    const NAME: &str = "APXINF_WALLOSS_FP8_DUAL_GEGLU";
    match std::env::var(NAME) {
        Err(std::env::VarError::NotPresent) => parse_fp8_dual_geglu_mode(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(Error::Other(format!("{NAME} must be valid Unicode")))
        }
        Ok(value) => parse_fp8_dual_geglu_mode(Some(&value)),
    }
}

fn interleave_gate_up_host(gate: &Tensor, up: &Tensor, tile: usize) -> Result<Tensor> {
    let gate_shape = gate.shape().dims();
    let up_shape = up.shape().dims();
    if gate_shape.len() != 2 || gate_shape != up_shape || tile == 0 || gate_shape[1] % tile != 0 {
        return Err(Error::Other(format!(
            "FP8 dual GeGLU requires equal 2D Gate/Up widths divisible by {tile}, got {gate_shape:?} and {up_shape:?}"
        )));
    }
    let rows = gate_shape[0];
    let width = gate_shape[1];
    let gate_values = gate.to_f32_vec()?;
    let up_values = up.to_f32_vec()?;
    let mut output = vec![0.0f32; rows * width * 2];
    for row in 0..rows {
        for tile_index in 0..width / tile {
            let src = row * width + tile_index * tile;
            let dst = row * width * 2 + tile_index * tile * 2;
            output[dst..dst + tile].copy_from_slice(&gate_values[src..src + tile]);
            output[dst + tile..dst + 2 * tile].copy_from_slice(&up_values[src..src + tile]);
        }
    }
    // Max is order-independent for finite model weights. Checking raw bits
    // here ensures the interleaved candidate keeps the exact joint scale.
    let source_amax = gate_values
        .iter()
        .chain(&up_values)
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    let interleaved_amax = output
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    if source_amax.to_bits() != interleaved_amax.to_bits() {
        return Err(Error::Other(
            "FP8 dual GeGLU interleaving changed joint amax raw bits".into(),
        ));
    }
    Tensor::from_f32(vec![rows, width * 2], &output)
}

#[cfg(feature = "cuda")]
fn fp16_host_and_amax(tensor: &Tensor) -> Result<(Tensor, f32)> {
    let values = tensor.to_f32_vec()?;
    let amax = values
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    let values = values
        .into_iter()
        .map(half::f16::from_f32)
        .collect::<Vec<_>>();
    Ok((
        Tensor::from_f16(tensor.shape().dims().to_vec(), &values)?,
        amax,
    ))
}

pub(crate) fn concat_host_2d(tensors: &[&Tensor]) -> Result<Tensor> {
    let first = tensors
        .first()
        .ok_or_else(|| Error::Other("empty tensor concatenation".into()))?;
    let dims = first.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!("expected 2D weight, got {dims:?}")));
    }
    let rows = dims[0];
    let widths = tensors
        .iter()
        .map(|tensor| {
            let dims = tensor.shape().dims();
            if dims.len() != 2 || dims[0] != rows {
                return Err(Error::Other("packed linear input dimensions differ".into()));
            }
            Ok(dims[1])
        })
        .collect::<Result<Vec<_>>>()?;
    let total_cols = widths.iter().sum::<usize>();
    let sources = tensors
        .iter()
        .map(|tensor| tensor.to_f32_vec())
        .collect::<Result<Vec<_>>>()?;
    let mut output = vec![0.0f32; rows * total_cols];
    for row in 0..rows {
        let mut output_col = 0;
        for (source, width) in sources.iter().zip(&widths) {
            output[row * total_cols + output_col..row * total_cols + output_col + width]
                .copy_from_slice(&source[row * width..(row + 1) * width]);
            output_col += width;
        }
    }
    Tensor::from_f32(vec![rows, total_cols], &output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::CpuBackend;

    #[test]
    fn packs_qkv_before_quantization() {
        let q = Tensor::from_f32(vec![2, 2], &[1., 2., 3., 4.]).unwrap();
        let k = Tensor::from_f32(vec![2, 1], &[5., 6.]).unwrap();
        let v = Tensor::from_f32(vec![2, 1], &[7., 8.]).unwrap();
        let packed = Fp8LinearWeights::from_host_parts(&[&q, &k, &v], &CpuBackend).unwrap();
        assert_eq!(packed.weight.shape().dims(), &[2, 4]);
        assert_eq!(packed.weight.dtype(), DType::F8E4M3);
    }

    #[test]
    fn dynamic_fp8_weights_use_independent_output_channel_scales() {
        let source = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 4.0, -3.0, -8.0, 12.0]).unwrap();
        let quantized = DynamicFp8LinearWeights::from_host(&source, &CpuBackend).unwrap();
        assert_eq!(quantized.weight.shape().dims(), &[16, 16]);
        assert_eq!(quantized.input_features, 2);
        assert_eq!(quantized.output_features, 3);
        assert_eq!(quantized.weight.dtype(), DType::F8E4M3);
        let scales = quantized.channel_scales.to_f32_vec().unwrap();
        assert_eq!(
            &scales[..3],
            &[3.0 / E4M3_MAX, 8.0 / E4M3_MAX, 12.0 / E4M3_MAX]
        );
        assert!(scales[3..].iter().all(|scale| *scale == 1.0e-12));
        let values = quantized.weight.as_f8_e4m3().unwrap();
        for (output, expected) in [[1.0, -3.0], [2.0, -8.0], [4.0, 12.0]]
            .into_iter()
            .enumerate()
        {
            for input in 0..2 {
                let decoded =
                    crate::walloss::decode_e4m3(values[output * 16 + input]) * scales[output];
                assert!((decoded - expected[input]).abs() < expected[input].abs() * 0.05);
            }
        }
    }

    #[test]
    fn fp8_dual_geglu_mode_parser_is_tri_state_and_defaults_auto() {
        assert_eq!(
            parse_fp8_dual_geglu_mode(None).unwrap(),
            Fp8DualGeGluMode::Auto
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("auto")).unwrap(),
            Fp8DualGeGluMode::Auto
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("0")).unwrap(),
            Fp8DualGeGluMode::Off
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("off")).unwrap(),
            Fp8DualGeGluMode::Off
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("1")).unwrap(),
            Fp8DualGeGluMode::On
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("on")).unwrap(),
            Fp8DualGeGluMode::On
        );
        assert!(parse_fp8_dual_geglu_mode(Some("invalid")).is_err());
    }

    #[test]
    fn fp8_dual_geglu_eighteen_layer_interleave_preserves_bytes_and_scale() {
        const ROWS: usize = 2;
        const WIDTH: usize = 1024;
        const TILE: usize = 256;
        for layer in 0..18usize {
            let gate = (0..ROWS * WIDTH)
                .map(|index| ((index * 17 + layer * 31) % 1009) as f32 / 127.0 - 4.0)
                .collect::<Vec<_>>();
            let up = (0..ROWS * WIDTH)
                .map(|index| ((index * 29 + layer * 43) % 1013) as f32 / 131.0 - 3.5)
                .collect::<Vec<_>>();
            let gate = Tensor::from_f32(vec![ROWS, WIDTH], &gate).unwrap();
            let up = Tensor::from_f32(vec![ROWS, WIDTH], &up).unwrap();
            let plain = concat_host_2d(&[&gate, &up]).unwrap();
            let interleaved = interleave_gate_up_host(&gate, &up, TILE).unwrap();
            let plain_q = quantize_e4m3_absmax(&plain).unwrap();
            let interleaved_q = quantize_e4m3_absmax(&interleaved).unwrap();
            assert_eq!(plain_q.scale.to_bits(), interleaved_q.scale.to_bits());
            let plain_bytes = plain_q.values.as_f8_e4m3().unwrap();
            let interleaved_bytes = interleaved_q.values.as_f8_e4m3().unwrap();
            for row in 0..ROWS {
                for tile_index in 0..WIDTH / TILE {
                    let plain_gate = row * 2 * WIDTH + tile_index * TILE;
                    let plain_up = row * 2 * WIDTH + WIDTH + tile_index * TILE;
                    let packed = row * 2 * WIDTH + tile_index * 2 * TILE;
                    assert_eq!(
                        &interleaved_bytes[packed..packed + TILE],
                        &plain_bytes[plain_gate..plain_gate + TILE]
                    );
                    assert_eq!(
                        &interleaved_bytes[packed + TILE..packed + 2 * TILE],
                        &plain_bytes[plain_up..plain_up + TILE]
                    );
                }
            }
        }
    }
}
