//! Model-neutral interfaces for vision-language-action runtimes.

use apxinf_core::{Error, Result, Tensor};

/// Memory layout for an RGB `u8` observation batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageLayout {
    Nhwc,
    Nchw,
}

/// Vision input accepted by a VLA runtime.
#[derive(Clone, Debug)]
pub enum VisionObservation {
    /// Preprocessed patch rows. The model defines the expected dtype and shape.
    Patches(Tensor),
    /// Resized RGB images. The byte buffer contains the complete view batch.
    RgbU8 { bytes: Vec<u8>, layout: ImageLayout },
}

/// Optional robot and multimodal conditioning carried alongside the common
/// image/language/noise inputs.
#[derive(Clone, Debug, Default)]
pub struct VlaConditioning {
    /// Normalized state history, laid out `[history, max_state_dim]`.
    pub state: Option<Tensor>,
    /// Checkpoint-defined embodiment/category selector.
    pub embodiment_id: Option<u32>,
    /// Qwen-style image grid entries, flattened `(temporal, height, width)`.
    pub image_grid_thw: Vec<u32>,
    /// One entry per token; non-zero entries participate in attention.
    pub attention_mask: Vec<u8>,
}

/// Complete input for one VLA inference.
#[derive(Clone, Debug)]
pub struct Observation {
    pub vision: VisionObservation,
    pub token_ids: Vec<u32>,
    pub noise: Tensor,
    pub conditioning: VlaConditioning,
}

impl Observation {
    pub fn validate(&self) -> Result<()> {
        if self.token_ids.is_empty() {
            return Err(Error::Other("VLA observation has no token IDs".into()));
        }
        if !self.conditioning.attention_mask.is_empty()
            && self.conditioning.attention_mask.len() != self.token_ids.len()
        {
            return Err(Error::Other(format!(
                "VLA attention mask has {} entries for {} token IDs",
                self.conditioning.attention_mask.len(),
                self.token_ids.len()
            )));
        }
        if self.conditioning.image_grid_thw.len() % 3 != 0 {
            return Err(Error::Other(
                "VLA image_grid_thw length must be divisible by three".into(),
            ));
        }
        if let Some(state) = &self.conditioning.state {
            let dims = state.shape().dims();
            if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
                return Err(Error::Other(format!(
                    "VLA state must have shape [history,state_dim], got {dims:?}"
                )));
            }
        }
        Ok(())
    }

    pub fn inference_spec(&self) -> InferenceSpec {
        InferenceSpec {
            token_count: self.token_ids.len(),
            image_layout: match self.vision {
                VisionObservation::Patches(_) => None,
                VisionObservation::RgbU8 { layout, .. } => Some(layout),
            },
            state_shape: self.conditioning.state.as_ref().and_then(|state| {
                let dims = state.shape().dims();
                (dims.len() == 2).then_some([dims[0], dims[1]])
            }),
            image_grid_count: self.conditioning.image_grid_thw.len() / 3,
            has_attention_mask: !self.conditioning.attention_mask.is_empty(),
        }
    }
}

/// Fixed-shape contract established during preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InferenceSpec {
    pub token_count: usize,
    /// `None` means preprocessed patches; `Some` means raw RGB input.
    pub image_layout: Option<ImageLayout>,
    pub state_shape: Option<[usize; 2]>,
    pub image_grid_count: usize,
    pub has_attention_mask: bool,
}

impl InferenceSpec {
    pub fn validate(&self) -> Result<()> {
        if self.token_count == 0 {
            return Err(Error::Other(
                "VLA inference spec requires at least one token".into(),
            ));
        }
        Ok(())
    }

    pub fn matches(&self, observation: &Observation) -> bool {
        *self == observation.inference_spec()
    }
}

/// Model action output. The tensor stays on the runtime device unless the
/// caller explicitly asks its backend-facing integration to transfer it, or
/// uses [`VlaRuntime::infer_host_f32`] to get host values directly.
#[derive(Clone, Debug)]
pub struct Action {
    tensor: Tensor,
}

impl Action {
    pub fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }
}

/// A prepared, fixed-shape inference plan. Implementations own every resource
/// referenced by eager execution or a captured graph.
pub trait PreparedInference {
    fn spec(&self) -> &InferenceSpec;
    fn run(&self, observation: &Observation) -> Result<Action>;
}

/// Unified VLA runtime interface.
///
/// The boxed return keeps this trait object-safe so `LoadedModel::Vla` can
/// directly hold heterogeneous model runtimes.
pub trait VlaRuntime {
    fn infer(&self, observation: &Observation) -> Result<Action>;
    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>>;

    /// Run inference and copy the resulting action to host as `f32`.
    ///
    /// [`infer`](Self::infer) returns an [`Action`] whose tensor lives on the
    /// runtime device. Consumers that need host values (servers writing actions
    /// back, benches checking outputs) would otherwise have to hold a backend
    /// handle and transfer it themselves — reaching around the abstraction.
    /// This convenience performs the device→host copy inside the runtime, which
    /// already owns the backend.
    fn infer_host_f32(&self, observation: &Observation) -> Result<Vec<f32>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::DType;

    fn observation() -> Observation {
        Observation {
            vision: VisionObservation::RgbU8 {
                bytes: vec![0; 3],
                layout: ImageLayout::Nhwc,
            },
            token_ids: vec![1, 2],
            noise: Tensor::zeros(vec![1, 2], DType::F32),
            conditioning: VlaConditioning::default(),
        }
    }

    #[test]
    fn prepared_spec_binds_conditioning_shapes() {
        let base = observation();
        let base_spec = base.inference_spec();
        let mut conditioned = observation();
        conditioned.conditioning.state = Some(Tensor::zeros(vec![1, 132], DType::F32));
        conditioned.conditioning.embodiment_id = Some(2);
        conditioned.conditioning.image_grid_thw = vec![1, 16, 16];
        conditioned.conditioning.attention_mask = vec![1, 1];
        conditioned.validate().unwrap();
        let spec = conditioned.inference_spec();
        assert_eq!(spec.state_shape, Some([1, 132]));
        assert_eq!(spec.image_grid_count, 1);
        assert!(spec.has_attention_mask);
        assert_ne!(base_spec, spec);
        assert!(spec.matches(&conditioned));
        assert!(!base_spec.matches(&conditioned));
    }

    #[test]
    fn invalid_conditioning_shapes_fail_closed() {
        let mut value = observation();
        value.conditioning.state = Some(Tensor::zeros(vec![132], DType::F32));
        assert!(value.validate().is_err());
        value.conditioning.state = None;
        value.conditioning.image_grid_thw = vec![1, 16];
        assert!(value.validate().is_err());
        value.conditioning.image_grid_thw.clear();
        value.conditioning.attention_mask = vec![1];
        assert!(value.validate().is_err());
    }
}
