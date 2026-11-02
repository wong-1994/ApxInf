//! CPU asset selection and assembly; no CUDA context is created.

use std::path::Path;
use std::rc::Rc;

use apxinf_core::{Error, Result};

use super::config::{WallossConfig, WallossImageProcessorConfig};
use super::model::CpuWallossModel;
use super::runner::CpuWallossRunner;
use super::weights::WallossWeights;
use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};

pub(crate) fn load(path: &Path, options: &LoadOptions) -> Result<LoadedModel> {
    if options.precision != ModelPrecision::Auto {
        return Err(Error::Other(
            "WallOSS CPU uses model_variant=fp32_reference instead of precision".into(),
        ));
    }
    if !matches!(
        options.model_variant.as_deref(),
        None | Some("auto") | Some("fp32_reference")
    ) {
        return Err(Error::Other(
            "WallOSS CPU supports only auto or fp32_reference".into(),
        ));
    }
    if options.config.is_some() || options.num_views.is_some() {
        return Err(Error::Other(
            "walloss does not support action_horizon/num_views config overrides".into(),
        ));
    }
    if options.calibration_path.is_some()
        || options.uniform_fp8_scale.is_some()
        || options.tuning_path.is_some()
        || options.autotune
    {
        return Err(Error::Other(
            "WallOSS CPU reference does not support CUDA tuning or quantization calibration"
                .into(),
        ));
    }
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let mut config = WallossConfig::from_json_file(&root.join("config.json"))?;
    let processor_path = root.join("preprocessor_config.json");
    let image_processor = processor_path
        .is_file()
        .then(|| WallossImageProcessorConfig::from_json_file(&processor_path, &config.vision))
        .transpose()?;
    let weights = match options.synthetic {
        Some(synthetic) => {
            // Synthetic checkpoints have no projections to resolve the native
            // action width from; use a small fixed width for contract tests.
            if config.action.action_dim == 0 {
                config.action.action_dim = 4;
                config.action.proprio_dim = 4;
            }
            WallossWeights::synthetic(&config, synthetic.seed)?
        }
        None => WallossWeights::from_safetensors(&mut config, path)?,
    };
    config.validate()?;
    let model = Rc::new(CpuWallossModel::new(
        config.clone(),
        image_processor,
        weights,
    )?);
    Ok(LoadedModel::Vla(Box::new(CpuWallossRunner::new(
        config, model,
    )?)))
}
