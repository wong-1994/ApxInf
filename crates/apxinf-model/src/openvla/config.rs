use apxinf_core::{Error, Result};
use std::path::Path;
/// The original fused DINOv2/SigLIP OpenVLA-7B profile.
#[derive(Clone, Debug)]
pub struct Config {
    pub max_tokens: usize,
}
impl Config {
    pub fn load(root: &Path) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("config.json")).map_err(|e| Error::Other(e.to_string()))?,
        )
        .map_err(|e| Error::Other(e.to_string()))?;
        if v["model_type"] != "openvla"
            || v["vision_backbone_id"] != "dinosiglip-vit-so-224px"
            || v["arch_specifier"] != "no-align+fused-gelu-mlp"
            || v["llm_backbone_id"] != "llama2-7b-pure"
            || v["image_resize_strategy"] != "resize-naive"
            || v["image_sizes"] != serde_json::json!([224, 224])
            || v["n_action_bins"] != 256
            || v["use_fused_vision_backbone"] != true
        {
            return Err(Error::Other(
                "OpenVLA requires the original fused 224px Llama2-7B checkpoint".into(),
            ));
        }
        let text = &v["text_config"];
        for (name, expected) in [
            ("hidden_size", 4096),
            ("intermediate_size", 11008),
            ("num_hidden_layers", 32),
            ("num_attention_heads", 32),
            ("num_key_value_heads", 32),
            ("vocab_size", 32064),
        ] {
            if text.get(name).is_some_and(|x| x.as_u64() != Some(expected)) {
                return Err(Error::Other(format!(
                    "unsupported OpenVLA text setting {name}"
                )));
            }
        }
        for (name, expected) in [("rope_theta", 10000.), ("rms_norm_eps", 1e-6)] {
            if text.get(name).is_some_and(|x| x.as_f64() != Some(expected)) {
                return Err(Error::Other(format!(
                    "unsupported OpenVLA text setting {name}"
                )));
            }
        }
        if text.get("rope_scaling").is_some_and(|x| !x.is_null())
            || text.get("hidden_act").is_some_and(|x| x != "silu")
        {
            return Err(Error::Other(
                "unsupported OpenVLA rotary/activation configuration".into(),
            ));
        }
        Ok(Self {
            max_tokens: 2048 - 256 - 7,
        })
    }
}
