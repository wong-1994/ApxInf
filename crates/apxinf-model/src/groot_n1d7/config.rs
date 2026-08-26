use std::path::Path;

use apxinf_core::{Error, Result};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct GrootN1d7DiffusionConfig {
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub output_dim: usize,
    #[serde(default)]
    pub interleave_self_attention: bool,
    #[serde(default)]
    pub norm_type: String,
}

impl GrootN1d7DiffusionConfig {
    pub fn inner_dim(&self) -> usize {
        self.attention_head_dim * self.num_attention_heads
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct GrootN1d7VlSelfAttentionConfig {
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
}

impl GrootN1d7VlSelfAttentionConfig {
    pub fn inner_dim(&self) -> usize {
        self.attention_head_dim * self.num_attention_heads
    }
}

/// Architecture fields consumed by the native N1.7 runtime.
///
/// Unknown training and augmentation fields are deliberately ignored. Required
/// inference fields have no guessed defaults: malformed or older checkpoints
/// fail at load time instead of silently selecting N1.5 semantics.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct GrootN1d7Config {
    pub model_type: String,
    pub action_horizon: usize,
    pub max_action_dim: usize,
    pub max_state_dim: usize,
    pub max_num_embodiments: usize,
    pub hidden_size: usize,
    pub backbone_embedding_dim: usize,
    pub max_seq_len: usize,
    pub num_inference_timesteps: usize,
    pub num_timestep_buckets: usize,
    pub select_layer: usize,
    pub image_target_size: [usize; 2],
    pub diffusion_model_cfg: GrootN1d7DiffusionConfig,
    pub vl_self_attention_cfg: GrootN1d7VlSelfAttentionConfig,
    #[serde(default = "default_true")]
    pub add_pos_embed: bool,
    #[serde(default)]
    pub use_alternate_vl_dit: bool,
    #[serde(default)]
    pub use_vlln: bool,
    #[serde(default)]
    pub use_vl_self_attention: bool,
    #[serde(default = "default_state_history_length")]
    pub state_history_length: usize,
    #[serde(default = "default_attend_text_every_n_blocks")]
    pub attend_text_every_n_blocks: usize,
}

const fn default_true() -> bool {
    true
}
const fn default_state_history_length() -> usize {
    1
}
const fn default_attend_text_every_n_blocks() -> usize {
    2
}

impl GrootN1d7Config {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("GR00T N1.7 config json: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn input_embedding_dim(&self) -> usize {
        self.diffusion_model_cfg.inner_dim()
    }

    pub fn validate(&self) -> Result<()> {
        if !self.model_type.eq_ignore_ascii_case("Gr00tN1d7") {
            return Err(Error::Other(format!(
                "GR00T N1.7 config requires model_type Gr00tN1d7, got {:?}",
                self.model_type
            )));
        }
        for (name, value) in [
            ("action_horizon", self.action_horizon),
            ("max_action_dim", self.max_action_dim),
            ("max_state_dim", self.max_state_dim),
            ("max_num_embodiments", self.max_num_embodiments),
            ("num_inference_timesteps", self.num_inference_timesteps),
            ("num_timestep_buckets", self.num_timestep_buckets),
            ("state_history_length", self.state_history_length),
            (
                "attend_text_every_n_blocks",
                self.attend_text_every_n_blocks,
            ),
        ] {
            if value == 0 {
                return Err(Error::Other(format!("GR00T N1.7 {name} must be non-zero")));
            }
        }
        let diffusion_width = self.diffusion_model_cfg.inner_dim();
        if diffusion_width != 1536 {
            return Err(Error::Other(format!(
                "GR00T N1.7 diffusion width must be 1536, got {diffusion_width}"
            )));
        }
        if self.diffusion_model_cfg.output_dim != self.hidden_size {
            return Err(Error::Other(format!(
                "GR00T N1.7 diffusion output_dim {} != hidden_size {}",
                self.diffusion_model_cfg.output_dim, self.hidden_size
            )));
        }
        if self.vl_self_attention_cfg.inner_dim() != self.backbone_embedding_dim {
            return Err(Error::Other(format!(
                "GR00T N1.7 VL self-attention width {} != backbone width {}",
                self.vl_self_attention_cfg.inner_dim(),
                self.backbone_embedding_dim
            )));
        }
        if !self.use_alternate_vl_dit || !self.diffusion_model_cfg.interleave_self_attention {
            return Err(Error::Other(
                "GR00T N1.7 requires AlternateVLDiT with interleaved self-attention".into(),
            ));
        }
        if self.image_target_size[0] == 0 || self.image_target_size[1] == 0 {
            return Err(Error::Other(
                "GR00T N1.7 image_target_size must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFFICIAL: &str = r#"{
      "model_type":"Gr00tN1d7", "action_horizon":40,
      "max_action_dim":132, "max_state_dim":132, "max_num_embodiments":32,
      "hidden_size":1024, "backbone_embedding_dim":2048, "max_seq_len":1024,
      "num_inference_timesteps":4, "num_timestep_buckets":1000,
      "select_layer":16, "image_target_size":[256,256],
      "add_pos_embed":true, "use_alternate_vl_dit":true,
      "use_vlln":true, "use_vl_self_attention":true,
      "diffusion_model_cfg":{"attention_head_dim":48,"num_attention_heads":32,
        "num_layers":32,"output_dim":1024,"interleave_self_attention":true,
        "norm_type":"ada_norm"},
      "vl_self_attention_cfg":{"attention_head_dim":64,"num_attention_heads":32,
        "num_layers":4}
    }"#;

    #[test]
    fn parses_official_n17_shape_contract() {
        let cfg = GrootN1d7Config::from_json_str(OFFICIAL).unwrap();
        assert_eq!(cfg.action_horizon, 40);
        assert_eq!(cfg.max_action_dim, 132);
        assert_eq!(cfg.input_embedding_dim(), 1536);
        assert_eq!(cfg.diffusion_model_cfg.num_layers, 32);
        assert_eq!(cfg.attend_text_every_n_blocks, 2);
    }

    #[test]
    fn rejects_n15_and_non_alternate_topologies() {
        let n15 = OFFICIAL.replace("Gr00tN1d7", "gr00t_n1_5");
        assert!(GrootN1d7Config::from_json_str(&n15).is_err());
        let ordinary = OFFICIAL.replace(
            "\"use_alternate_vl_dit\":true",
            "\"use_alternate_vl_dit\":false",
        );
        assert!(GrootN1d7Config::from_json_str(&ordinary).is_err());
    }
}
