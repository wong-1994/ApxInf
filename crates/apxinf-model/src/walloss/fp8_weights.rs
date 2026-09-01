//! Device-resident FP8 matrices for WallOSS inference.

use apxinf_core::{Backend, Error, Result, Tensor};

use super::weights::bf16_to_device;
use super::{
    DynamicFp8LinearWeights, Fp8LinearWeights, StaticFp8Calibration, WallossActionWeights,
    WallossLayerWeights, WallossVisionBlockWeights, WallossVisionWeights, WallossWeights,
};

pub struct WallossFp8Weights {
    pub token_embedding: Tensor,
    pub language_layers: Vec<WallossFp8LayerWeights>,
    pub action_layers: Vec<WallossFp8LayerWeights>,
    pub action_norm: Tensor,
    pub vision: WallossFp8VisionWeights,
    pub action: WallossActionWeights,
}

pub struct WallossDynamicFp8Weights {
    pub token_embedding: Tensor,
    pub language_layers: Vec<WallossDynamicFp8LayerWeights>,
    pub action_layers: Vec<WallossDynamicFp8LayerWeights>,
    pub action_norm: Tensor,
    pub vision: WallossDynamicFp8VisionWeights,
    pub action: WallossActionWeights,
}

pub struct WallossDynamicFp8LayerWeights {
    pub input_norm: Tensor,
    pub post_attention_norm: Tensor,
    pub qkv: DynamicFp8LinearWeights,
    pub qkv_bias: Tensor,
    pub output: DynamicFp8LinearWeights,
    pub gate_up: DynamicFp8LinearWeights,
    pub down: DynamicFp8LinearWeights,
}

pub struct WallossDynamicFp8VisionWeights {
    pub patch_projection: DynamicFp8LinearWeights,
    pub blocks: Vec<WallossDynamicFp8VisionBlockWeights>,
    pub merger_norm: Tensor,
    pub merger_hidden: DynamicFp8LinearWeights,
    pub merger_hidden_bias: Tensor,
    pub merger_output: DynamicFp8LinearWeights,
    pub merger_output_bias: Tensor,
}

pub struct WallossDynamicFp8VisionBlockWeights {
    pub input_norm: Tensor,
    pub qkv: DynamicFp8LinearWeights,
    pub qkv_bias: Tensor,
    pub output: DynamicFp8LinearWeights,
    pub output_bias: Tensor,
    pub post_attention_norm: Tensor,
    pub gate_up: DynamicFp8LinearWeights,
    pub gate_up_bias: Tensor,
    pub down: DynamicFp8LinearWeights,
    pub down_bias: Tensor,
}

pub struct WallossFp8LayerWeights {
    pub qkv_scale: f32,
    pub output_scale: f32,
    pub gate_up_scale: f32,
    pub down_scale: f32,
    pub input_norm: Tensor,
    pub post_attention_norm: Tensor,
    pub qkv: Fp8LinearWeights,
    pub qkv_bias: Tensor,
    pub output: Fp8LinearWeights,
    pub gate_up: Fp8LinearWeights,
    pub down: Fp8LinearWeights,
}

pub struct WallossFp8VisionWeights {
    pub patch_scale: f32,
    pub merger_norm_scale: f32,
    pub merger_hidden_scale: f32,
    pub patch_projection: Fp8LinearWeights,
    pub blocks: Vec<WallossFp8VisionBlockWeights>,
    pub merger_norm: Tensor,
    pub merger_hidden: Fp8LinearWeights,
    pub merger_hidden_bias: Tensor,
    pub merger_output: Fp8LinearWeights,
    pub merger_output_bias: Tensor,
}

pub struct WallossFp8VisionBlockWeights {
    pub qkv_scale: f32,
    pub output_scale: f32,
    pub gate_up_scale: f32,
    pub down_scale: f32,
    pub input_norm: Tensor,
    pub qkv: Fp8LinearWeights,
    pub qkv_bias: Tensor,
    pub output: Fp8LinearWeights,
    pub output_bias: Tensor,
    pub post_attention_norm: Tensor,
    pub gate_up: Fp8LinearWeights,
    pub gate_up_bias: Tensor,
    pub down: Fp8LinearWeights,
    pub down_bias: Tensor,
}

#[derive(Clone, Copy, Debug)]
pub struct WallossLayerActivationScales {
    pub qkv: f32,
    pub output: f32,
    pub gate_up: f32,
    pub down: f32,
}

#[derive(Clone, Debug)]
pub struct WallossActivationScales {
    pub vision_patch: f32,
    pub vision_layers: Vec<WallossLayerActivationScales>,
    pub vision_merger_norm: f32,
    pub vision_merger_hidden: f32,
    pub language_layers: Vec<WallossLayerActivationScales>,
    pub action_layers: Vec<WallossLayerActivationScales>,
}

impl WallossActivationScales {
    pub fn uniform(weights: &WallossWeights, scale: f32) -> Result<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::Other(format!("invalid uniform FP8 scale {scale}")));
        }
        let layer = WallossLayerActivationScales {
            qkv: scale,
            output: scale,
            gate_up: scale,
            down: scale,
        };
        Ok(Self {
            vision_patch: scale,
            vision_layers: vec![layer; weights.vision.blocks.len()],
            vision_merger_norm: scale,
            vision_merger_hidden: scale,
            language_layers: vec![layer; weights.language_layers.len()],
            action_layers: vec![layer; weights.action_layers.len()],
        })
    }

    pub fn from_calibration(
        weights: &WallossWeights,
        calibration: &StaticFp8Calibration,
    ) -> Result<Self> {
        let layers = |prefix: &str, depth: usize| {
            (0..depth)
                .map(|index| {
                    let name = |suffix: &str| format!("{prefix}.layers.{index}.{suffix}");
                    Ok(WallossLayerActivationScales {
                        qkv: calibration.scale(&name("attention_norm"))?,
                        output: calibration.scale(&name("attention_output"))?,
                        gate_up: calibration.scale(&name("mlp_norm"))?,
                        down: calibration.scale(&name("mlp_activation"))?,
                    })
                })
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            vision_patch: calibration.scale("vision.patch_input")?,
            vision_layers: layers("vision", weights.vision.blocks.len())?,
            vision_merger_norm: calibration.scale("vision.merger_norm")?,
            vision_merger_hidden: calibration.scale("vision.merger_hidden")?,
            language_layers: layers("language", weights.language_layers.len())?,
            action_layers: layers("action", weights.action_layers.len())?,
        })
    }
}

impl WallossFp8Weights {
    pub fn from_host(
        weights: &WallossWeights,
        backend: &dyn Backend,
        scales: &WallossActivationScales,
    ) -> Result<Self> {
        if scales.language_layers.len() != weights.language_layers.len()
            || scales.action_layers.len() != weights.action_layers.len()
            || scales.vision_layers.len() != weights.vision.blocks.len()
        {
            return Err(Error::Other(
                "walloss FP8 activation calibration depth mismatch".into(),
            ));
        }
        Ok(Self {
            token_embedding: bf16_to_device(&weights.token_embedding, backend)?,
            language_layers: weights
                .language_layers
                .iter()
                .zip(&scales.language_layers)
                .map(|(layer, scale)| WallossFp8LayerWeights::from_host(layer, backend, *scale))
                .collect::<Result<_>>()?,
            action_layers: weights
                .action_layers
                .iter()
                .zip(&scales.action_layers)
                .map(|(layer, scale)| WallossFp8LayerWeights::from_host(layer, backend, *scale))
                .collect::<Result<_>>()?,
            action_norm: bf16_to_device(&weights.action_norm, backend)?,
            vision: WallossFp8VisionWeights::from_host(&weights.vision, backend, scales)?,
            action: weights.action.to_bf16_device(backend)?,
        })
    }
}

impl WallossDynamicFp8Weights {
    pub fn from_host(weights: &WallossWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            token_embedding: bf16_to_device(&weights.token_embedding, backend)?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| WallossDynamicFp8LayerWeights::from_host(layer, backend))
                .collect::<Result<_>>()?,
            action_layers: weights
                .action_layers
                .iter()
                .map(|layer| WallossDynamicFp8LayerWeights::from_host(layer, backend))
                .collect::<Result<_>>()?,
            action_norm: bf16_to_device(&weights.action_norm, backend)?,
            vision: WallossDynamicFp8VisionWeights::from_host(&weights.vision, backend)?,
            action: weights.action.to_bf16_device(backend)?,
        })
    }
}

impl WallossDynamicFp8LayerWeights {
    fn from_host(weights: &WallossLayerWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            input_norm: bf16_to_device(&weights.input_norm, backend)?,
            post_attention_norm: bf16_to_device(&weights.post_attention_norm, backend)?,
            qkv: dynamic_fp8_matrix(&weights.qkv, backend)?,
            qkv_bias: bf16_to_device(&weights.qkv_bias, backend)?,
            output: dynamic_fp8_matrix(&weights.output, backend)?,
            gate_up: dynamic_fp8_matrix(&weights.gate_up, backend)?,
            down: dynamic_fp8_matrix(&weights.down, backend)?,
        })
    }
}

impl WallossDynamicFp8VisionWeights {
    fn from_host(weights: &WallossVisionWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            patch_projection: dynamic_fp8_matrix(&weights.patch_projection, backend)?,
            blocks: weights
                .blocks
                .iter()
                .map(|block| WallossDynamicFp8VisionBlockWeights::from_host(block, backend))
                .collect::<Result<_>>()?,
            merger_norm: bf16_to_device(&weights.merger_norm, backend)?,
            merger_hidden: dynamic_fp8_matrix(&weights.merger_hidden, backend)?,
            merger_hidden_bias: bf16_to_device(&weights.merger_hidden_bias, backend)?,
            merger_output: dynamic_fp8_matrix(&weights.merger_output, backend)?,
            merger_output_bias: bf16_to_device(&weights.merger_output_bias, backend)?,
        })
    }
}

impl WallossDynamicFp8VisionBlockWeights {
    fn from_host(weights: &WallossVisionBlockWeights, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            input_norm: bf16_to_device(&weights.input_norm, backend)?,
            qkv: dynamic_fp8_matrix(&weights.qkv, backend)?,
            qkv_bias: bf16_to_device(&weights.qkv_bias, backend)?,
            output: dynamic_fp8_matrix(&weights.output, backend)?,
            output_bias: bf16_to_device(&weights.output_bias, backend)?,
            post_attention_norm: bf16_to_device(&weights.post_attention_norm, backend)?,
            gate_up: dynamic_fp8_matrix(&weights.gate_up, backend)?,
            gate_up_bias: bf16_to_device(&weights.gate_up_bias, backend)?,
            down: dynamic_fp8_matrix(&weights.down, backend)?,
            down_bias: bf16_to_device(&weights.down_bias, backend)?,
        })
    }
}

impl WallossFp8LayerWeights {
    fn from_host(
        weights: &WallossLayerWeights,
        backend: &dyn Backend,
        scales: WallossLayerActivationScales,
    ) -> Result<Self> {
        Ok(Self {
            qkv_scale: scales.qkv,
            output_scale: scales.output,
            gate_up_scale: scales.gate_up,
            down_scale: scales.down,
            input_norm: bf16_to_device(&weights.input_norm, backend)?,
            post_attention_norm: bf16_to_device(&weights.post_attention_norm, backend)?,
            qkv: fp8_matrix(&weights.qkv, backend)?,
            qkv_bias: bf16_to_device(&weights.qkv_bias, backend)?,
            output: fp8_matrix(&weights.output, backend)?,
            gate_up: fp8_matrix(&weights.gate_up, backend)?,
            down: fp8_matrix(&weights.down, backend)?,
        })
    }
}

impl WallossFp8VisionWeights {
    fn from_host(
        weights: &WallossVisionWeights,
        backend: &dyn Backend,
        scales: &WallossActivationScales,
    ) -> Result<Self> {
        Ok(Self {
            patch_scale: scales.vision_patch,
            patch_projection: fp8_matrix(&weights.patch_projection, backend)?,
            blocks: weights
                .blocks
                .iter()
                .zip(&scales.vision_layers)
                .map(|(block, scale)| {
                    WallossFp8VisionBlockWeights::from_host(block, backend, *scale)
                })
                .collect::<Result<_>>()?,
            merger_norm: bf16_to_device(&weights.merger_norm, backend)?,
            merger_norm_scale: scales.vision_merger_norm,
            merger_hidden: fp8_matrix(&weights.merger_hidden, backend)?,
            merger_hidden_bias: bf16_to_device(&weights.merger_hidden_bias, backend)?,
            merger_hidden_scale: scales.vision_merger_hidden,
            merger_output: fp8_matrix(&weights.merger_output, backend)?,
            merger_output_bias: bf16_to_device(&weights.merger_output_bias, backend)?,
        })
    }
}

impl WallossFp8VisionBlockWeights {
    fn from_host(
        weights: &WallossVisionBlockWeights,
        backend: &dyn Backend,
        scales: WallossLayerActivationScales,
    ) -> Result<Self> {
        Ok(Self {
            qkv_scale: scales.qkv,
            output_scale: scales.output,
            gate_up_scale: scales.gate_up,
            down_scale: scales.down,
            input_norm: bf16_to_device(&weights.input_norm, backend)?,
            qkv: fp8_matrix(&weights.qkv, backend)?,
            qkv_bias: bf16_to_device(&weights.qkv_bias, backend)?,
            output: fp8_matrix(&weights.output, backend)?,
            output_bias: bf16_to_device(&weights.output_bias, backend)?,
            post_attention_norm: bf16_to_device(&weights.post_attention_norm, backend)?,
            gate_up: fp8_matrix(&weights.gate_up, backend)?,
            gate_up_bias: bf16_to_device(&weights.gate_up_bias, backend)?,
            down: fp8_matrix(&weights.down, backend)?,
            down_bias: bf16_to_device(&weights.down_bias, backend)?,
        })
    }
}

fn fp8_matrix(weight: &Tensor, backend: &dyn Backend) -> Result<Fp8LinearWeights> {
    Fp8LinearWeights::from_host(weight, backend)
}

fn dynamic_fp8_matrix(weight: &Tensor, backend: &dyn Backend) -> Result<DynamicFp8LinearWeights> {
    DynamicFp8LinearWeights::from_host(weight, backend)
}
