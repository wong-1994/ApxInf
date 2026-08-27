//! Owning BF16 VLA runtime for fixed-shape WallOSS inference.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, NormalGenerator, Result, SamplingBackend, Tensor};

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::vla::{
    Action, InferenceSpec, InitialLatent, PreparedInference, VisionObservation, VlaRequest,
    VlaRuntime,
};

use super::backend::{DeviceBuffer, RuntimeBackend};
use super::bf16_executor::{action_stack, language_prefix, solver_update, vision_tower};
use super::{
    multimodal_position_ids, sinusoidal_time_embedding, solver_times, DeviceVisionGeometry,
    VisionGeometry, WallossConfig, WallossWeights,
};

const DEFAULT_GRIDS: [[usize; 3]; 2] = [[1, 18, 18], [1, 18, 18]];

pub struct WallossBf16Runtime {
    backend: Arc<RuntimeBackend>,
    config: Arc<WallossConfig>,
    weights: Arc<WallossWeights>,
    grids: Arc<Vec<[usize; 3]>>,
    geometry: Arc<DeviceVisionGeometry>,
    prepared: RefCell<Option<(InferenceSpec, Rc<WallossPreparedInference>)>>,
}

pub struct WallossPreparedInference {
    spec: InferenceSpec,
    backend: Arc<RuntimeBackend>,
    config: Arc<WallossConfig>,
    weights: Arc<WallossWeights>,
    grids: Arc<Vec<[usize; 3]>>,
    geometry: Arc<DeviceVisionGeometry>,
    noise: Tensor,
    normal_generator: RefCell<Box<dyn NormalGenerator>>,
}

impl WallossPreparedInference {
    fn run_impl(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let observation = request.observation;
        observation.validate()?;
        if !self.spec.matches(observation) {
            return Err(Error::Other(format!(
                "prepared walloss spec {:?} does not match observation {:?}",
                self.spec,
                observation.inference_spec()
            )));
        }
        let patches = match &observation.vision {
            VisionObservation::Patches(value) => normalize_host_bf16(
                value,
                vec![self.geometry.patch_order.len() / 4, patch_width(&self.config)],
                "patches",
            )?,
            VisionObservation::RgbU8 { .. } => {
                return Err(Error::Other(
                    "walloss raw RGB preprocessing is not connected yet; pass preprocessed patches"
                        .into(),
                ))
            }
        };
        let patches = self.backend.to_device(&patches)?;
        let context = self.backend.context();
        let vision = vision_tower(
            context,
            &self.config.vision,
            &self.weights.vision,
            &self.geometry,
            &patches,
        )?;

        let action_tokens = self.config.action.action_horizon;
        let prefix_tokens = observation.token_ids.len() - action_tokens;
        let prefix_ids = &observation.token_ids[..prefix_tokens];
        let token_ids = upload_u32(context.device_id(), prefix_ids)?;
        let vision_rows = self.geometry.reverse_indices.len() / std::mem::size_of::<u32>();
        let mut vision_row_map = vec![u32::MAX; prefix_tokens];
        let mut vision_row = 0u32;
        for (row, &token) in prefix_ids.iter().enumerate() {
            if token == self.config.image_token_id {
                vision_row_map[row] = vision_row;
                vision_row += 1;
            }
        }
        if vision_row as usize != vision_rows {
            return Err(Error::Other(format!(
                "walloss prompt has {vision_row} image tokens, expected {vision_rows}"
            )));
        }
        let vision_row_map = upload_u32(context.device_id(), &vision_row_map)?;
        let position_ids = multimodal_position_ids(
            &observation.token_ids,
            &self.grids,
            self.config.image_token_id,
            self.config.vision.spatial_merge_size,
        )?;
        let prefix_position_ids = upload_u32(
            context.device_id(),
            &position_ids[..prefix_tokens * 3],
        )?;
        let action_position_ids = upload_u32(
            context.device_id(),
            &position_ids[prefix_tokens * 3..],
        )?;
        let prefix = language_prefix(
            context,
            &self.config.text,
            &self.weights.language_layers,
            &self.weights.token_embedding,
            &token_ids,
            &vision,
            &vision_row_map,
            &prefix_position_ids,
            prefix_tokens,
        )?;

        let mut state = match request.initial_latent {
            InitialLatent::Provided(value) => self.backend.to_device(&normalize_host_bf16(
                value,
                vec![action_tokens, self.config.action.action_dim],
                "initial latent",
            )?)?,
            InitialLatent::Generate { rng } => {
                self.normal_generator.borrow_mut().generate(rng)?;
                self.noise.clone()
            }
        };
        let mask_host = match observation.action_mask.as_ref() {
            Some(value) => normalize_host_bf16(
                value,
                vec![action_tokens, self.config.action.action_dim],
                "action mask",
            )?,
            None => Tensor::from_bf16(
                vec![action_tokens, self.config.action.action_dim],
                &vec![half::bf16::ONE; action_tokens * self.config.action.action_dim],
            )?,
        };
        let action_mask = self.backend.to_device(&mask_host)?;
        let times = solver_times(
            self.config.action.solver_steps,
            self.config.action.scheduler_s,
            1.0,
        )?;
        for step in 0..self.config.action.solver_steps {
            let embedding = sinusoidal_time_embedding(
                times[step],
                self.config.action.hidden_size,
            )?;
            let repeated = embedding
                .iter()
                .copied()
                .cycle()
                .take(action_tokens * embedding.len())
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>();
            let time_embedding = self.backend.to_device(&Tensor::from_bf16(
                vec![action_tokens, self.config.action.hidden_size],
                &repeated,
            )?)?;
            let velocity = action_stack(
                context,
                &self.config.text,
                &self.weights.action_layers,
                &self.weights.action,
                &self.weights.action_norm,
                &prefix,
                &state,
                &action_mask,
                &time_embedding,
                &action_position_ids,
            )?;
            state = solver_update(context, &state, &velocity, times[step + 1] - times[step])?;
        }
        Ok(Action::new(state))
    }
}

impl PreparedInference for WallossPreparedInference {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.run_impl(request)
    }
}

impl WallossBf16Runtime {
    fn build_prepared(&self, spec: InferenceSpec) -> Result<WallossPreparedInference> {
        spec.validate()?;
        if spec.token_count <= self.config.action.action_horizon {
            return Err(Error::Other(
                "walloss token sequence must contain a language prefix and action suffix".into(),
            ));
        }
        let noise_host = Tensor::zeros(
            [self.config.action.action_horizon, self.config.action.action_dim],
            DType::BF16,
        );
        let noise = self.backend.to_device(&noise_host)?;
        let normal_generator = self.backend.create_normal_generator(noise.clone())?;
        Ok(WallossPreparedInference {
            spec,
            backend: Arc::clone(&self.backend),
            config: Arc::clone(&self.config),
            weights: Arc::clone(&self.weights),
            grids: Arc::clone(&self.grids),
            geometry: Arc::clone(&self.geometry),
            noise,
            normal_generator: RefCell::new(normal_generator),
        })
    }
}

impl VlaRuntime for WallossBf16Runtime {
    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let spec = request.observation.inference_spec();
        let prepared = {
            let mut cache = self.prepared.borrow_mut();
            if cache.as_ref().is_none_or(|(cached, _)| *cached != spec) {
                *cache = Some((spec, Rc::new(self.build_prepared(spec)?)));
            }
            Rc::clone(&cache.as_ref().unwrap().1)
        };
        prepared.run(request)
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(self.build_prepared(*spec)?))
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let output = self.infer(request)?;
        self.backend.to_cpu(output.tensor())?.to_f32_vec()
    }
}

pub(super) fn load_registered(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(options.precision, ModelPrecision::Auto | ModelPrecision::Bf16) {
        return Err(Error::Other(
            "walloss currently accepts BF16 while the FP8 runtime is being connected".into(),
        ));
    }
    let backend = crate::accelerator::cuda::downcast_arc(backend)
        .ok_or_else(|| Error::Other("walloss is only registered for CUDA".into()))?;
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let mut config = WallossConfig::from_json_file(&root.join("config.json"))?;
    let host_weights = WallossWeights::from_safetensors(&mut config, path)?;
    let weights = Arc::new(host_weights.to_bf16_device(&*backend)?);
    let grids = Arc::new(DEFAULT_GRIDS.to_vec());
    let host_geometry = VisionGeometry::new(&config.vision, &grids)?;
    let geometry = Arc::new(host_geometry.upload(backend.context())?);
    Ok(LoadedModel::Vla(Box::new(WallossBf16Runtime {
        backend,
        config: Arc::new(config),
        weights,
        grids,
        geometry,
        prepared: RefCell::new(None),
    })))
}

fn patch_width(config: &WallossConfig) -> usize {
    3 * config.vision.temporal_patch_size * config.vision.patch_size * config.vision.patch_size
}

fn normalize_host_bf16(value: &Tensor, shape: Vec<usize>, name: &str) -> Result<Tensor> {
    if value.shape().dims() != shape {
        return Err(Error::Other(format!(
            "walloss {name} shape {:?}, expected {shape:?}",
            value.shape().dims()
        )));
    }
    let values = value
        .to_f32_vec()?
        .into_iter()
        .map(half::bf16::from_f32)
        .collect::<Vec<_>>();
    Tensor::from_bf16(shape, &values)
}

fn upload_u32(device_id: usize, values: &[u32]) -> Result<DeviceBuffer> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let buffer = DeviceBuffer::alloc_zeros(bytes.len(), device_id).map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(buffer)
}
