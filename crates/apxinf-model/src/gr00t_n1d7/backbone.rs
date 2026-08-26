use std::collections::HashMap;
use std::sync::Arc;

use apxinf_core::{Backend, Error, Result, Tensor};

use crate::qwen3vl::config::{Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig};
use crate::qwen3vl::GeneralQwen3VL;

pub(crate) fn load_backbone(
    tensors: &mut HashMap<String, Tensor>,
    backend: Arc<dyn Backend>,
) -> Result<GeneralQwen3VL> {
    let prefix = "backbone.model.";
    let keys = tensors
        .keys()
        .filter(|key| key.starts_with(prefix))
        .cloned()
        .collect::<Vec<_>>();
    let mut backbone = HashMap::with_capacity(keys.len());
    for key in keys {
        let value = tensors
            .remove(&key)
            .ok_or_else(|| Error::Other(format!("missing {key}")))?;
        backbone.insert(key[prefix.len()..].to_owned(), value);
    }
    // The tied LM head is not used by GR00T's continuous action path.
    backbone.remove("lm_head.weight");
    GeneralQwen3VL::from_weights_with_backend(cosmos_reason2_config(), backbone, backend)
}

fn cosmos_reason2_config() -> Qwen3VLConfig {
    Qwen3VLConfig {
        text: Qwen3VLTextConfig {
            hidden_size: 2048,
            intermediate_size: 6144,
            // The official GR00T N1.7 checkpoint stores the selected prefix of
            // Cosmos-Reason2 (layers 0..15), matching select_layer=16.
            n_layers: 16,
            n_heads: 16,
            n_kv_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            max_position_embeddings: 262144,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            mrope_section: [24, 20, 20],
            mrope_interleaved: true,
            tie_word_embeddings: true,
        },
        vision: Qwen3VLVisionConfig {
            depth: 24,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_heads: 16,
            head_dim: 64,
            patch_size: 16,
            temporal_patch_size: 2,
            in_channels: 3,
            spatial_merge_size: 2,
            num_position_embeddings: 2304,
            out_hidden_size: 2048,
            deepstack_visual_indexes: vec![5, 11, 17],
        },
        image_token_id: 151655,
        video_token_id: 151656,
        vision_start_token_id: 151652,
        vision_end_token_id: 151653,
    }
}
