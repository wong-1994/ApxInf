//! Minimal maintained VLA used to prove the VLA Family Pack end to end.

use std::path::Path;
use std::sync::Arc;

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::vla::{
    Action, ImageLayout, InferenceSpec, Observation, PreparedInference, VisionObservation,
    VlaRuntime,
};
use apxinf_core::{Backend, Device, Error, Result, Tensor};

#[derive(Clone, Debug)]
pub struct MinimalVlaConfig {
    pub image_size: usize,
    pub num_views: usize,
    pub action_horizon: usize,
    pub action_dim: usize,
    pub vocab_size: usize,
    pub max_token_len: usize,
}

impl MinimalVlaConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        let value: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| Error::Other(format!("parse {}: {e}", path.display())))?;
        let get = |name: &str| {
            value
                .get(name)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| Error::Other(format!("minimal_vla config missing {name}")))
        };
        let cfg = Self {
            image_size: get("image_size")?,
            num_views: get("num_views")?,
            action_horizon: get("action_horizon")?,
            action_dim: get("action_dim")?,
            vocab_size: get("vocab_size")?,
            max_token_len: get("max_token_len")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.image_size != 1 || self.num_views != 1 || self.action_horizon != 1 {
            return Err(Error::Other(
                "minimal_vla requires one 1x1 RGB view and action_horizon=1".into(),
            ));
        }
        if self.action_dim == 0 || self.vocab_size == 0 || self.max_token_len == 0 {
            return Err(Error::Other(
                "minimal_vla dimensions must be positive".into(),
            ));
        }
        Ok(())
    }
}

struct Weights {
    vision_projection: Tensor,
    token_embedding: Tensor,
    action_projection: Tensor,
}

pub struct MinimalVlaRuntime {
    cfg: MinimalVlaConfig,
    weights: Arc<Weights>,
    backend: Arc<dyn Backend>,
}

struct MinimalPrepared {
    spec: InferenceSpec,
    cfg: MinimalVlaConfig,
    weights: Arc<Weights>,
    backend: Arc<dyn Backend>,
}

impl MinimalPrepared {
    fn upload_values(&self, shape: (usize, usize), values: &[f32]) -> Result<Tensor> {
        let cpu = if self.backend.device() == Device::Cpu {
            Tensor::from_f32(shape, values)?
        } else {
            let values = values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>();
            Tensor::from_bf16(shape, &values)?
        };
        self.backend.to_device(&cpu)
    }

    fn input_tensor(&self, observation: &Observation) -> Result<Tensor> {
        let values = match &observation.vision {
            VisionObservation::RgbU8 { bytes, layout } => {
                if bytes.len() != 3 || *layout != ImageLayout::Nhwc {
                    return Err(Error::Other(
                        "minimal_vla expects NHWC [1,1,1,3] RGB".into(),
                    ));
                }
                bytes
                    .iter()
                    .map(|v| *v as f32 / 255.0)
                    .collect::<Vec<_>>()
            }
            VisionObservation::Patches(patches) => {
                if patches.shape().dims() != [1, 3] {
                    return Err(Error::Other(
                        "minimal_vla patches must have shape [1,3]".into(),
                    ));
                }
                patches.to_f32_vec()?
            }
        };
        self.upload_values((1, 3), &values)
    }
}

impl PreparedInference for MinimalPrepared {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn run(&self, observation: &Observation) -> Result<Action> {
        observation.validate()?;
        if !self.spec.matches(observation) {
            return Err(Error::Other(
                "observation does not match prepared minimal_vla shape".into(),
            ));
        }
        if observation.token_ids.len() != 1
            || observation.token_ids[0] as usize >= self.cfg.vocab_size
        {
            return Err(Error::Other(
                "minimal_vla requires one in-range token".into(),
            ));
        }
        if observation.noise.shape().dims() != [1, self.cfg.action_dim] {
            return Err(Error::Other("minimal_vla noise has the wrong shape".into()));
        }
        let image = self.input_tensor(observation)?;
        let vision = self
            .backend
            .matmul(&image, &self.weights.vision_projection)?;
        let language = self
            .backend
            .embedding(&self.weights.token_embedding, &observation.token_ids)?;
        let noise_values = observation.noise.to_f32_vec()?;
        let noise = self.upload_values((1, self.cfg.action_dim), &noise_values)?;
        let conditioned = self
            .backend
            .add(&self.backend.add(&vision, &language)?, &noise)?;
        let action = self
            .backend
            .matmul(&conditioned, &self.weights.action_projection)?;
        self.backend.synchronize()?;
        Ok(Action::new(action))
    }
}

impl VlaRuntime for MinimalVlaRuntime {
    fn infer(&self, observation: &Observation) -> Result<Action> {
        self.prepare(&observation.inference_spec())?
            .run(observation)
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        spec.validate()?;
        if spec.token_count != 1 {
            return Err(Error::Other(
                "minimal_vla supports exactly one token".into(),
            ));
        }
        if spec.state_shape.is_some() || spec.image_grid_count != 0 || spec.has_attention_mask {
            return Err(Error::Other(
                "minimal_vla does not accept external conditioning".into(),
            ));
        }
        Ok(Box::new(MinimalPrepared {
            spec: *spec,
            cfg: self.cfg.clone(),
            weights: Arc::clone(&self.weights),
            backend: Arc::clone(&self.backend),
        }))
    }

    fn infer_host_f32(&self, observation: &Observation) -> Result<Vec<f32>> {
        let action = self.infer(observation)?;
        self.backend.to_cpu(action.tensor())?.to_f32_vec()
    }
}

fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other("minimal_vla projection must be 2-D".into()));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let src = tensor.to_f32_vec()?;
    let mut dst = vec![0.0; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    Tensor::from_f32((cols, rows), &dst)
}

pub(crate) fn load(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) {
        return Err(Error::Other(
            "minimal_vla implements only the requested BF16 precision".into(),
        ));
    }
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let cfg = MinimalVlaConfig::from_json_file(&root.join("config.json"))?;
    let (mut map, _) = apxinf_loader::safetensors::load_native_path(path)
        .map_err(|e| Error::Other(format!("load {}: {e}", path.display())))?;
    let mut take = |name: &str| {
        map.remove(name)
            .ok_or_else(|| Error::Other(format!("missing minimal_vla parameter {name}")))
    };
    let vision_projection = transpose_2d(&take("vision_projection.weight")?)?;
    let token_embedding = take("token_embedding.weight")?;
    let action_projection = transpose_2d(&take("action_projection.weight")?)?;
    if !map.is_empty() {
        return Err(Error::Other(format!(
            "unexpected minimal_vla parameters: {:?}",
            map.keys().collect::<Vec<_>>()
        )));
    }
    let upload = |t: Tensor| -> Result<Tensor> {
        let values = t.to_f32_vec()?;
        let t = if device == Device::Cpu {
            Tensor::from_f32(t.shape().dims().to_vec(), &values)?
        } else {
            Tensor::from_bf16(
                t.shape().dims().to_vec(),
                &values
                    .into_iter()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>(),
            )?
        };
        backend.to_device(&t)
    };
    let weights = Weights {
        vision_projection: upload(vision_projection)?,
        token_embedding: upload(token_embedding)?,
        action_projection: upload(action_projection)?,
    };
    Ok(LoadedModel::Vla(Box::new(MinimalVlaRuntime {
        cfg,
        weights: Arc::new(weights),
        backend,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_produce_known_normalized_action_through_prepared_seam() {
        let backend = crate::accelerator::create_backend(Device::Cpu).unwrap();
        let runtime = MinimalVlaRuntime {
            cfg: MinimalVlaConfig {
                image_size: 1,
                num_views: 1,
                action_horizon: 1,
                action_dim: 2,
                vocab_size: 4,
                max_token_len: 1,
            },
            weights: Arc::new(Weights {
                vision_projection: Tensor::from_f32((3, 2), &[0.5, 0.0, 0.0, 0.25, -0.5, 0.25])
                    .unwrap(),
                token_embedding: Tensor::from_f32(
                    (4, 2),
                    &[0.0, 0.0, 0.25, -0.25, 0.5, 0.5, -0.5, 0.25],
                )
                .unwrap(),
                action_projection: Tensor::from_f32((2, 2), &[1.0, 0.0, 0.0, 1.0]).unwrap(),
            }),
            backend,
        };
        let observation = Observation {
            vision: VisionObservation::RgbU8 {
                bytes: vec![255, 0, 0],
                layout: ImageLayout::Nhwc,
            },
            token_ids: vec![1],
            noise: Tensor::from_f32((1, 2), &[0.25, 0.5]).unwrap(),
            conditioning: crate::vla::VlaConditioning::default(),
        };
        let prepared = runtime.prepare(&observation.inference_spec()).unwrap();
        assert_eq!(
            prepared
                .run(&observation)
                .unwrap()
                .tensor()
                .to_f32_vec()
                .unwrap(),
            vec![1.0, 0.25]
        );
        assert_eq!(
            runtime.infer_host_f32(&observation).unwrap(),
            vec![1.0, 0.25]
        );
    }
}
