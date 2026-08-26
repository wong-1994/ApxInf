use std::collections::HashMap;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};
use half::bf16;

use crate::qwen3vl::{GeneralQwen3VL, Qwen3VLConfig};

use super::GrootN1d7Config;

pub struct GrootN1d7BackboneOutput {
    pub features: Tensor,
    pub image_mask: Vec<u8>,
    pub attention_mask: Vec<u8>,
}

/// N1.7's checkpoint-local, layer-truncated Cosmos-Reason2-2B backbone.
pub struct GrootN1d7Backbone {
    model: GeneralQwen3VL,
}

impl GrootN1d7Backbone {
    pub fn from_map(
        cfg: &GrootN1d7Config,
        cosmos_config: Qwen3VLConfig,
        tensors: HashMap<String, Tensor>,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let mut cosmos_config = cosmos_config;
        cosmos_config.text.n_layers = cfg.select_layer;
        if cosmos_config.text.hidden_size != cfg.backbone_embedding_dim {
            return Err(Error::Other(format!(
                "GR00T N1.7 backbone width {} != Cosmos width {}",
                cfg.backbone_embedding_dim, cosmos_config.text.hidden_size
            )));
        }
        let prefix = "backbone.model.";
        let backbone = tensors
            .into_iter()
            .filter_map(|(name, tensor)| {
                name.strip_prefix(prefix)
                    .map(|stripped| (stripped.to_owned(), tensor))
            })
            .collect::<HashMap<_, _>>();
        if backbone.is_empty() {
            return Err(Error::Other(
                "GR00T N1.7 checkpoint has no backbone.model weights".into(),
            ));
        }
        let model = GeneralQwen3VL::from_weights_with_backend(cosmos_config, backbone, backend)?;
        Ok(Self { model })
    }

    pub fn forward(
        &mut self,
        token_ids: &[u32],
        pixel_values: &Tensor,
        grid_thw: &[[u32; 3]],
    ) -> Result<GrootN1d7BackboneOutput> {
        use crate::llm_trait::LlmTrait;
        self.model.reset();
        let converted;
        let pixel_values = if pixel_values.dtype() == DType::BF16 {
            pixel_values
        } else {
            let cpu = if pixel_values.device() == Device::Cpu {
                pixel_values.clone()
            } else {
                return Err(Error::Other(
                    "GR00T processor patches must be CPU f32 or bf16".into(),
                ));
            };
            let values = cpu
                .to_f32_vec()?
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>();
            converted = Tensor::from_bf16(cpu.shape().dims().to_vec(), &values)?;
            &converted
        };
        let output = self
            .model
            .backbone_with_image(token_ids, pixel_values, grid_thw)?;
        Ok(GrootN1d7BackboneOutput {
            features: output.features,
            image_mask: output.image_mask,
            attention_mask: output.attention_mask,
        })
    }
}
