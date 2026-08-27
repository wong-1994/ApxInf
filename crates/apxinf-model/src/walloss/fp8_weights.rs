//! Device-resident static-FP8 matrices for WallOSS inference.

use apxinf_core::{Backend, Result, Tensor};

use crate::pi05::{Fp8LinearWeights, LinearWeights};

use super::weights::bf16_to_device;
use super::{
    WallossActionWeights, WallossLayerWeights, WallossVisionBlockWeights, WallossVisionWeights,
    WallossWeights,
};

pub struct WallossFp8Weights {
    pub token_embedding: Tensor,
    pub language_layers: Vec<WallossFp8LayerWeights>,
    pub action_layers: Vec<WallossFp8LayerWeights>,
    pub language_norm: Tensor,
    pub action_norm: Tensor,
    pub vision: WallossFp8VisionWeights,
    pub action: WallossActionWeights,
}

pub struct WallossFp8LayerWeights {
    pub activation_scale: f32,
    pub input_norm: Tensor,
    pub post_attention_norm: Tensor,
    pub qkv: Fp8LinearWeights,
    pub qkv_bias: Tensor,
    pub output: Fp8LinearWeights,
    pub gate_up: Fp8LinearWeights,
    pub down: Fp8LinearWeights,
}

pub struct WallossFp8VisionWeights {
    pub activation_scale: f32,
    pub patch_projection: Fp8LinearWeights,
    pub blocks: Vec<WallossFp8VisionBlockWeights>,
    pub merger_norm: Tensor,
    pub merger_hidden: Fp8LinearWeights,
    pub merger_hidden_bias: Tensor,
    pub merger_output: Fp8LinearWeights,
    pub merger_output_bias: Tensor,
}

pub struct WallossFp8VisionBlockWeights {
    pub activation_scale: f32,
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

impl WallossFp8Weights {
    pub fn from_host(
        weights: &WallossWeights,
        backend: &dyn Backend,
        activation_scale: f32,
    ) -> Result<Self> {
        Ok(Self {
            token_embedding: bf16_to_device(&weights.token_embedding, backend)?,
            language_layers: weights
                .language_layers
                .iter()
                .map(|layer| WallossFp8LayerWeights::from_host(layer, backend, activation_scale))
                .collect::<Result<_>>()?,
            action_layers: weights
                .action_layers
                .iter()
                .map(|layer| WallossFp8LayerWeights::from_host(layer, backend, activation_scale))
                .collect::<Result<_>>()?,
            language_norm: bf16_to_device(&weights.language_norm, backend)?,
            action_norm: bf16_to_device(&weights.action_norm, backend)?,
            vision: WallossFp8VisionWeights::from_host(&weights.vision, backend, activation_scale)?,
            action: weights.action.to_bf16_device(backend)?,
        })
    }
}

impl WallossFp8LayerWeights {
    fn from_host(weights: &WallossLayerWeights, backend: &dyn Backend, activation_scale: f32) -> Result<Self> {
        Ok(Self {
            activation_scale,
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
    fn from_host(weights: &WallossVisionWeights, backend: &dyn Backend, activation_scale: f32) -> Result<Self> {
        Ok(Self {
            activation_scale,
            patch_projection: fp8_matrix(&weights.patch_projection, backend)?,
            blocks: weights
                .blocks
                .iter()
                .map(|block| WallossFp8VisionBlockWeights::from_host(block, backend, activation_scale))
                .collect::<Result<_>>()?,
            merger_norm: bf16_to_device(&weights.merger_norm, backend)?,
            merger_hidden: fp8_matrix(&weights.merger_hidden, backend)?,
            merger_hidden_bias: bf16_to_device(&weights.merger_hidden_bias, backend)?,
            merger_output: fp8_matrix(&weights.merger_output, backend)?,
            merger_output_bias: bf16_to_device(&weights.merger_output_bias, backend)?,
        })
    }
}

impl WallossFp8VisionBlockWeights {
    fn from_host(weights: &WallossVisionBlockWeights, backend: &dyn Backend, activation_scale: f32) -> Result<Self> {
        Ok(Self {
            activation_scale,
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
    Fp8LinearWeights::from_host(
        &LinearWeights {
            weight: weight.clone(),
            bias: None,
        },
        backend,
    )
}
