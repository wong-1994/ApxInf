use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, NormalGenerator, Result, Tensor};

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::qwen3vl::config::{Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig};
use crate::vla::{
    Action, InferenceSpec, InitialLatent, PreparedInference, VisionObservation, VlaRequest,
    VlaRuntime,
};

use super::device_weights::DeviceActionWeights;
use super::executor::GrootN1d7Executor;
use super::{GrootN1d7ActionWeights, GrootN1d7Backbone, GrootN1d7Config};

pub struct GrootN1d7Runtime {
    inner: Rc<GrootInner>,
}
struct GrootInner {
    backend: Arc<dyn Backend>,
    config: Arc<GrootN1d7Config>,
    backbone: RefCell<GrootN1d7Backbone>,
    source_weights: Arc<GrootN1d7ActionWeights>,
    executors: RefCell<HashMap<u32, Rc<GrootN1d7Executor>>>,
    normal_generator: RefCell<Box<dyn NormalGenerator>>,
}

impl GrootInner {
    fn executor(&self, embodiment: u32) -> Result<Rc<GrootN1d7Executor>> {
        if embodiment as usize >= self.config.max_num_embodiments {
            return Err(Error::Other(format!(
                "GR00T N1.7 embodiment {embodiment} out of range"
            )));
        }
        if let Some(executor) = self.executors.borrow().get(&embodiment) {
            return Ok(executor.clone());
        }
        let weights =
            DeviceActionWeights::upload(&self.source_weights, embodiment as usize, &self.backend)?;
        let executor = Rc::new(GrootN1d7Executor::new(
            self.backend.clone(),
            self.config.clone(),
            weights,
        )?);
        self.executors
            .borrow_mut()
            .insert(embodiment, executor.clone());
        Ok(executor)
    }

    fn infer_inner(&self, request: &VlaRequest<'_>) -> Result<Tensor> {
        request.observation.validate()?;
        let state = request
            .state
            .ok_or_else(|| Error::Other("GR00T N1.7 request requires state".into()))?;
        let embodiment = request
            .embodiment_id
            .ok_or_else(|| Error::Other("GR00T N1.7 request requires embodiment_id".into()))?;
        let patches = match &request.observation.vision {
            VisionObservation::Patches(patches) => patches,
            VisionObservation::RgbU8 { .. } => return Err(Error::Other(
                "GR00T N1.7 Rust runtime expects Qwen3-VL processor patches; use GrootPolicy for RGB".into())),
        };
        let dims = patches.shape().dims();
        if dims.len() != 2 || dims[1] != 1536 || dims[0] % 256 != 0 {
            return Err(Error::Other(format!(
                "GR00T N1.7 patches must be [views*256,1536], got {dims:?}"
            )));
        }
        let grids = vec![[1u32, 16, 16]; dims[0] / 256];
        let backbone =
            self.backbone
                .borrow_mut()
                .forward(&request.observation.token_ids, patches, &grids)?;
        let noise = match request.initial_latent {
            InitialLatent::Provided(noise) => noise.clone(),
            InitialLatent::Generate { rng } => {
                self.normal_generator.borrow_mut().generate(rng)?.clone()
            }
        };
        self.executor(embodiment)?.infer(backbone, state, &noise)
    }
}

impl VlaRuntime for GrootN1d7Runtime {
    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        Ok(Action::new(self.inner.infer_inner(request)?))
    }
    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        self.inner
            .backend
            .to_cpu(&self.inner.infer_inner(request)?)?
            .to_f32_vec()
    }
    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        spec.validate()?;
        Ok(Box::new(GrootPrepared {
            inner: self.inner.clone(),
            spec: *spec,
        }))
    }
}

struct GrootPrepared {
    inner: Rc<GrootInner>,
    spec: InferenceSpec,
}
impl PreparedInference for GrootPrepared {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other(
                "GR00T prepared inference shape mismatch".into(),
            ));
        }
        Ok(Action::new(self.inner.infer_inner(request)?))
    }
}

pub(crate) fn load_registered(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Cuda(_))
        || !matches!(
            options.precision,
            ModelPrecision::Auto | ModelPrecision::Bf16
        )
    {
        return Err(Error::Other(
            "GR00T N1.7 supports CUDA BF16 on the current runtime".into(),
        ));
    }
    let dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    let config = Arc::new(GrootN1d7Config::from_json_file(&dir.join("config.json"))?);
    let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
        .map_err(|error| Error::Other(format!("load GR00T N1.7 checkpoint: {error}")))?;
    let (backbone_map, action_map): (HashMap<_, _>, HashMap<_, _>) = tensors
        .into_iter()
        .partition(|(name, _)| name.starts_with("backbone.model."));
    let backbone = GrootN1d7Backbone::from_map(
        &config,
        cosmos_config(config.select_layer),
        backbone_map,
        backend.clone(),
    )?;
    let source_weights = Arc::new(GrootN1d7ActionWeights::from_map(&config, action_map)?);
    let noise = Tensor::zeros((config.action_horizon, config.max_action_dim), DType::BF16);
    let noise = backend.to_device(&noise)?;
    let normal_generator = backend.create_normal_generator(noise)?;
    Ok(LoadedModel::Vla(Box::new(GrootN1d7Runtime {
        inner: Rc::new(GrootInner {
            backend,
            config,
            backbone: RefCell::new(backbone),
            source_weights,
            executors: RefCell::new(HashMap::new()),
            normal_generator: RefCell::new(normal_generator),
        }),
    })))
}

fn cosmos_config(layers: usize) -> Qwen3VLConfig {
    Qwen3VLConfig {
        text: Qwen3VLTextConfig {
            hidden_size: 2048,
            intermediate_size: 6144,
            n_layers: layers,
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
