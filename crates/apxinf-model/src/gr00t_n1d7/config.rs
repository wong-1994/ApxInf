//! NVIDIA GR00T N1.7 checkpoint configuration.

use apxinf_core::{Error, Result};
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct Gr00tN1d7Config {
    pub action_horizon: usize,
    pub action_dim: usize,
    pub state_dim: usize,
    pub state_history: usize,
    pub embodiment_count: usize,
    pub backbone_dim: usize,
    pub action_embed_dim: usize,
    pub dit_hidden: usize,
    pub dit_layers: usize,
    pub dit_heads: usize,
    pub dit_head_dim: usize,
    pub vl_layers: usize,
    pub vl_heads: usize,
    pub vl_head_dim: usize,
    pub max_seq_len: usize,
    pub flow_steps: usize,
    pub timestep_buckets: usize,
    pub attend_text_every_n_blocks: usize,
    pub image_token_id: u32,
}

impl Gr00tN1d7Config {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(raw)
            .map_err(|e| Error::Other(format!("GR00T N1.7 config json: {e}")))?;
        let model_type = v["model_type"].as_str().unwrap_or_default();
        if model_type != "Gr00tN1d7" {
            return Err(Error::Other(format!(
                "GR00T N1.7 loader expected model_type Gr00tN1d7, got {model_type:?}"
            )));
        }
        let diffusion = &v["diffusion_model_cfg"];
        let vl = &v["vl_self_attention_cfg"];
        let get = |obj: &serde_json::Value, key: &str, default: u64| {
            obj[key].as_u64().unwrap_or(default) as usize
        };
        let cfg = Self {
            action_horizon: get(&v, "action_horizon", 40),
            action_dim: get(&v, "max_action_dim", 132),
            state_dim: get(&v, "max_state_dim", 132),
            state_history: get(&v, "state_history_length", 1),
            embodiment_count: get(&v, "max_num_embodiments", 32),
            backbone_dim: get(&v, "backbone_embedding_dim", 2048),
            action_embed_dim: get(&v, "input_embedding_dim", 1536),
            dit_hidden: get(&v, "hidden_size", 1024),
            dit_layers: get(diffusion, "num_layers", 16),
            dit_heads: get(diffusion, "num_attention_heads", 32),
            dit_head_dim: get(diffusion, "attention_head_dim", 48),
            vl_layers: get(vl, "num_layers", 0),
            vl_heads: get(vl, "num_attention_heads", 32),
            vl_head_dim: get(vl, "attention_head_dim", 64),
            max_seq_len: get(&v, "max_seq_len", 1024),
            flow_steps: get(&v, "num_inference_timesteps", 4),
            timestep_buckets: get(&v, "num_timestep_buckets", 1000),
            attend_text_every_n_blocks: get(&v, "attend_text_every_n_blocks", 2),
            image_token_id: 151655,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        if self.action_horizon == 0 || self.action_dim == 0 || self.state_dim == 0 {
            return Err(Error::Other(
                "GR00T N1.7 state/action dimensions must be non-zero".into(),
            ));
        }
        if self.flow_steps != 4 {
            return Err(Error::Other(format!(
                "GR00T N1.7 ApxInf runtime supports the validated 4-step Euler schedule, got {}",
                self.flow_steps
            )));
        }
        if self.action_embed_dim != self.dit_heads * self.dit_head_dim {
            return Err(Error::Other(format!(
                "GR00T N1.7 DiT width {} != heads {} * head_dim {}",
                self.action_embed_dim, self.dit_heads, self.dit_head_dim
            )));
        }
        if self.vl_layers > 0 && self.backbone_dim != self.vl_heads * self.vl_head_dim {
            return Err(Error::Other(
                "GR00T N1.7 VL attention dimensions are inconsistent".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_official_base_shape() {
        let cfg = Gr00tN1d7Config::from_json_str(
            r#"{
          "model_type":"Gr00tN1d7", "action_horizon":40,
          "max_action_dim":132, "max_state_dim":132,
          "backbone_embedding_dim":2048, "input_embedding_dim":1536,
          "hidden_size":1024, "max_num_embodiments":32,
          "max_seq_len":1024, "num_inference_timesteps":4,
          "num_timestep_buckets":1000,
          "diffusion_model_cfg":{"num_layers":32,"num_attention_heads":32,"attention_head_dim":48},
          "vl_self_attention_cfg":{"num_layers":4,"num_attention_heads":32,"attention_head_dim":64}
        }"#,
        )
        .unwrap();
        assert_eq!((cfg.action_horizon, cfg.action_dim), (40, 132));
        assert_eq!((cfg.dit_layers, cfg.vl_layers, cfg.flow_steps), (32, 4, 4));
    }
}
