use std::path::Path;

use apxinf_core::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct GrootConfig {
    pub model_type: String,
    pub model_name: String,
    pub backbone_embedding_dim: usize,
    pub select_layer: usize,
    pub max_state_dim: usize,
    pub max_action_dim: usize,
    pub action_horizon: usize,
    pub hidden_size: usize,
    pub input_embedding_dim: usize,
    pub state_history_length: usize,
    pub max_num_embodiments: usize,
    pub num_inference_timesteps: usize,
    pub num_timestep_buckets: usize,
    pub max_seq_len: usize,
    pub add_pos_embed: bool,
    pub use_alternate_vl_dit: bool,
}

impl GrootConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        let value: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|error| Error::Other(format!("parse {}: {error}", path.display())))?;
        let string = |name: &str| -> Result<String> {
            value.get(name).and_then(|item| item.as_str()).map(str::to_owned)
                .ok_or_else(|| Error::Other(format!("GR00T config requires string {name}")))
        };
        let usize_value = |name: &str| -> Result<usize> {
            value.get(name).and_then(|item| item.as_u64()).map(|item| item as usize)
                .ok_or_else(|| Error::Other(format!("GR00T config requires integer {name}")))
        };
        let boolean = |name: &str| -> Result<bool> {
            value.get(name).and_then(|item| item.as_bool())
                .ok_or_else(|| Error::Other(format!("GR00T config requires boolean {name}")))
        };
        let config = Self {
            model_type: string("model_type")?,
            model_name: string("model_name")?,
            backbone_embedding_dim: usize_value("backbone_embedding_dim")?,
            select_layer: usize_value("select_layer")?,
            max_state_dim: usize_value("max_state_dim")?,
            max_action_dim: usize_value("max_action_dim")?,
            action_horizon: usize_value("action_horizon")?,
            hidden_size: usize_value("hidden_size")?,
            // These two dimensions are implicit in NVIDIA's N1.7 config and
            // explicit in the action-head weight shapes.
            input_embedding_dim: value.get("input_embedding_dim").and_then(|x| x.as_u64()).unwrap_or(1536) as usize,
            state_history_length: value.get("state_history_length").and_then(|x| x.as_u64()).unwrap_or(1) as usize,
            max_num_embodiments: usize_value("max_num_embodiments")?,
            num_inference_timesteps: usize_value("num_inference_timesteps")?,
            num_timestep_buckets: usize_value("num_timestep_buckets")?,
            max_seq_len: usize_value("max_seq_len")?,
            add_pos_embed: boolean("add_pos_embed")?,
            use_alternate_vl_dit: boolean("use_alternate_vl_dit")?,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "Gr00tN1d7" {
            return Err(Error::Other(format!("unsupported GR00T model_type {}", self.model_type)));
        }
        if self.model_name != "nvidia/Cosmos-Reason2-2B" {
            return Err(Error::Other(format!("unsupported GR00T backbone {}", self.model_name)));
        }
        if self.num_inference_timesteps != 4 || self.num_timestep_buckets != 1000 {
            return Err(Error::Other("GR00T N1.7 requires the fixed four-step/1000-bucket flow schedule".into()));
        }
        if self.backbone_embedding_dim != 2048 || self.select_layer != 16
            || self.max_state_dim != 132 || self.max_action_dim != 132
            || self.action_horizon != 40 || self.hidden_size != 1024
            || self.input_embedding_dim != 1536 || self.state_history_length != 1
            || self.max_num_embodiments != 32 || self.max_seq_len != 1024
            || !self.add_pos_embed || !self.use_alternate_vl_dit
        {
            return Err(Error::Other(
                "checkpoint is not the supported GR00T N1.7 132D/40-step architecture".into(),
            ));
        }
        if self.action_horizon == 0 || self.max_action_dim == 0 || self.max_state_dim == 0
            || self.max_num_embodiments == 0 || self.state_history_length == 0 {
            return Err(Error::Other("GR00T dimensions must be positive".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fixed_n17_contract_and_rejects_schedule_drift() {
        let path = std::env::temp_dir().join(format!("apxinf-groot-config-{}.json", std::process::id()));
        std::fs::write(&path, r#"{
            "model_type":"Gr00tN1d7","model_name":"nvidia/Cosmos-Reason2-2B",
            "backbone_embedding_dim":2048,"select_layer":16,"max_state_dim":132,
            "max_action_dim":132,"action_horizon":40,"hidden_size":1024,
            "input_embedding_dim":1536,"state_history_length":1,
            "max_num_embodiments":32,"num_inference_timesteps":4,
            "num_timestep_buckets":1000,"max_seq_len":1024,
            "add_pos_embed":true,"use_alternate_vl_dit":true
        }"#).unwrap();
        let mut config = GrootConfig::from_json_file(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(config.action_horizon, 40);
        assert_eq!(config.max_action_dim, 132);
        config.num_inference_timesteps = 5;
        assert!(config.validate().is_err());
    }
}
