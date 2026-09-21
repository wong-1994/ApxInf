use super::{model::Model, model_runner::OpenVlaModelRunner};
use crate::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Device, Error, Result};
use std::{path::Path, sync::Arc};
pub(crate) fn load_registered(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if backend.device() != device {
        return Err(Error::Other(
            "OpenVLA device does not match its backend".into(),
        ));
    }
    if !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) || options.config.is_some()
        || options.synthetic.is_some()
        || options.model_variant.is_some()
    {
        return Err(Error::Other(
            "OpenVLA supports precision=bf16/auto with original checkpoint configuration".into(),
        ));
    }
    let backend = crate::accelerator::cuda::downcast_arc(backend)
        .ok_or_else(|| Error::Other("OpenVLA requires CUDA".into()))?;
    let model = Model::from_checkpoint(path, &backend)?;
    Ok(LoadedModel::Vla(Box::new(OpenVlaModelRunner::new(
        backend, model,
    ))))
}
