//! Validate the maintained Walloss checkpoint contract without running inference.

use std::path::PathBuf;

use apxinf_model::walloss::{WallossConfig, WallossWeights};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: walloss_checkpoint_probe MODEL_DIR")?;
    let mut config = WallossConfig::from_json_file(&model_dir.join("config.json"))?;
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(&model_dir)?;
    let weights = WallossWeights::from_map(&mut config, tensors)?;
    println!(
        "walloss checkpoint ok: vocab={} language_layers={} action_layers={} vision_blocks={}",
        config.text.vocab_size,
        weights.language_layers.len(),
        weights.action_layers.len(),
        weights.vision.blocks.len(),
    );
    Ok(())
}
