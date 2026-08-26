use std::path::Path;
use std::sync::{Arc, Mutex};

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::qwen3vl::GeneralQwen3VL;
use crate::vla::{
    Action, InferenceSpec, InitialLatent, PreparedInference, VisionObservation, VlaRequest,
    VlaRuntime,
};

use super::backbone::load_backbone;
use super::executor::Gr00tExecutor;
use super::weights::ActionWeights;
use super::Gr00tN1d7Config;

struct Core {
    backend: Arc<dyn Backend>,
    config: Gr00tN1d7Config,
    backbone: Mutex<GeneralQwen3VL>,
    executor: Gr00tExecutor,
}

impl Core {
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        request.observation.validate()?;
        let conditioning = request.conditioning.ok_or_else(|| {
            Error::Other(
                "GR00T N1.7 requires state, embodiment, grids, attention mask, and image mask"
                    .into(),
            )
        })?;
        if conditioning.attention_mask.len() != request.observation.token_ids.len()
            || conditioning.image_mask.len() != request.observation.token_ids.len()
        {
            return Err(Error::Other("GR00T masks must align with token_ids".into()));
        }
        let patches = match &request.observation.vision {
            VisionObservation::Patches(value) => value,
            VisionObservation::RgbU8 { .. } => {
                return Err(Error::Other(
                    "GR00T N1.7 Rust runtime accepts canonical Qwen3-VL pixel_values patches"
                        .into(),
                ))
            }
        };
        let backbone = self
            .backbone
            .lock()
            .map_err(|_| Error::Other("GR00T backbone lock poisoned".into()))?
            .encode_multimodal(
                &request.observation.token_ids,
                patches,
                conditioning.image_grid_thw,
            )?;
        let noise = match request.initial_latent {
            InitialLatent::Provided(value) => value.clone(),
            InitialLatent::Generate { rng } => {
                let values = apxinf_core::standard_normal_f32(
                    self.config.action_horizon * self.config.action_dim,
                    rng,
                );
                let bf16 = values
                    .into_iter()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>();
                Tensor::from_bf16(
                    vec![self.config.action_horizon, self.config.action_dim],
                    &bf16,
                )?
            }
        };
        #[cfg(feature = "cuda")]
        {
            let cuda = crate::accelerator::cuda::downcast(&*self.backend)
                .ok_or_else(|| Error::Other("GR00T N1.7 requires CUDA".into()))?;
            return Ok(Action::new(self.executor.infer(
                cuda,
                &backbone,
                conditioning.state,
                conditioning.embodiment_id,
                conditioning.attention_mask,
                conditioning.image_mask,
                &noise,
            )?));
        }
        #[cfg(not(feature = "cuda"))]
        Err(Error::Other("GR00T N1.7 requires the cuda feature".into()))
    }
}

struct Prepared {
    spec: InferenceSpec,
    core: Arc<Core>,
}
impl PreparedInference for Prepared {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other("GR00T prepared shape mismatch".into()));
        }
        self.core.run(request)
    }
}

pub struct Gr00tN1d7VlaRuntime {
    core: Arc<Core>,
}
impl VlaRuntime for Gr00tN1d7VlaRuntime {
    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.core.run(request)
    }
    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        spec.validate()?;
        Ok(Box::new(Prepared {
            spec: *spec,
            core: Arc::clone(&self.core),
        }))
    }
    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let out = self.infer(request)?;
        self.core.backend.to_cpu(out.tensor())?.to_f32_vec()
    }
}

pub(super) fn load_registered(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if options.precision != ModelPrecision::Auto && options.precision != ModelPrecision::Bf16 {
        return Err(Error::Other("GR00T N1.7 supports BF16 only".into()));
    }
    #[cfg(feature = "cuda")]
    if crate::accelerator::cuda::downcast(&*backend).is_none() {
        return Err(Error::Other("GR00T N1.7 requires CUDA".into()));
    }
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let config = Gr00tN1d7Config::from_json_file(&root.join("config.json"))?;
    let (mut tensors, _) = apxinf_loader::safetensors::load_native_path(path)
        .map_err(|e| Error::Other(format!("load {}: {e}", path.display())))?;
    if tensors.values().any(|t| t.dtype() != DType::BF16) {
        return Err(Error::Other("GR00T N1.7 checkpoint must be BF16".into()));
    }
    let action = ActionWeights::from_map(&config, &mut tensors)?;
    let backbone = load_backbone(&mut tensors, Arc::clone(&backend))?;
    let executor = Gr00tExecutor::new(config.clone(), action, &*backend)?;
    Ok(LoadedModel::Vla(Box::new(Gr00tN1d7VlaRuntime {
        core: Arc::new(Core {
            backend,
            config,
            backbone: Mutex::new(backbone),
            executor,
        }),
    })))
}
