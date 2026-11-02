//! Synchronous CPU execution for the WallOSS reference.
//!
//! Explicit plans retain the model and never claim graph readiness. Host
//! input preparation mirrors the CUDA runner: grids, image-token runs,
//! multimodal position IDs and action inputs are validated before any
//! allocation.

use std::rc::Rc;

use apxinf_core::{
    standard_normal_f32, DType, Device, Error, Result, RngKey, Tensor,
};

use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent,
    Observation, PreparationStatus, PreparedInference, VisionObservation, VlaContract,
    VlaRequest, VlaRuntime,
};

use super::geometry::{multimodal_position_ids, VisionGeometry};
use super::model::{CpuWallossModel, WallossHostInputs};
use super::WallossConfig;

/// Published LIBERO profile: two 252x252 views (18x18 patches).
const DEFAULT_GRIDS: [[u32; 3]; 2] = [[1, 18, 18], [1, 18, 18]];

#[derive(Clone)]
pub(in crate::cpu_walloss) struct CpuWallossRunner {
    config: WallossConfig,
    model: Rc<CpuWallossModel>,
}

struct PreparedShape {
    grids: Vec<[usize; 3]>,
}

struct CpuWallossPrepared {
    runner: CpuWallossRunner,
    spec: InferenceSpec,
    fallback: bool,
}

impl CpuWallossRunner {
    pub(in crate::cpu_walloss) fn new(
        config: WallossConfig,
        model: Rc<CpuWallossModel>,
    ) -> Result<Self> {
        Ok(Self { config, model })
    }

    fn action_shape(&self) -> [usize; 2] {
        [
            self.config.action.action_horizon,
            self.config.action.action_dim,
        ]
    }

    fn request_shape(&self, request: &VlaRequest<'_>) -> Result<PreparedShape> {
        let merge = self.config.vision.spatial_merge_size;
        let grids = match request.metadata.image_grid_thw {
            None => DEFAULT_GRIDS
                .iter()
                .map(|&[t, h, w]| [t as usize, h as usize, w as usize])
                .collect(),
            Some(grids) => {
                if grids.is_empty() {
                    return Err(Error::Other("walloss image_grid_thw cannot be empty".into()));
                }
                grids
                    .iter()
                    .map(|&[t, h, w]| {
                        let [t, h, w] = [t as usize, h as usize, w as usize];
                        if t == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
                            return Err(Error::Other(format!(
                                "walloss vision grid [{t}, {h}, {w}] must be positive and divisible by merge size {merge}"
                            )));
                        }
                        Ok([t, h, w])
                    })
                    .collect::<Result<Vec<_>>>()?
            }
        };
        Ok(PreparedShape { grids })
    }

    fn validate_spec(&self, spec: &InferenceSpec) -> Result<()> {
        spec.validate()?;
        if spec.token_count <= self.config.action.action_horizon {
            return Err(Error::Other(
                "walloss token sequence must contain a language prefix and action suffix".into(),
            ));
        }
        if spec.token_count > self.config.text.max_position_embeddings {
            return Err(Error::Other(format!(
                "walloss sequence of {} tokens exceeds max_position_embeddings {}",
                spec.token_count, self.config.text.max_position_embeddings
            )));
        }
        Ok(())
    }

    fn host_inputs(
        &self,
        spec: &InferenceSpec,
        shape: &PreparedShape,
        request: &VlaRequest<'_>,
    ) -> Result<WallossHostInputs> {
        let observation = request.observation;
        observation.validate()?;
        self.validate_spec(spec)?;
        if *spec != observation.inference_spec() {
            return Err(Error::Other(format!(
                "walloss CPU spec {spec:?} does not match observation {:?}",
                observation.inference_spec()
            )));
        }
        let requested = self.request_shape(request)?;
        if requested.grids != shape.grids {
            return Err(Error::Other(format!(
                "walloss CPU plan was prepared for grids {:?}, request declares {:?}",
                shape.grids, requested.grids
            )));
        }
        let geometry = VisionGeometry::new(&self.config.vision, &shape.grids)?;
        if spec.image_layout.is_some() && geometry.rgb_views.is_none() {
            return Err(Error::Other(format!(
                "walloss RGB input requires still views of one size, got grids {:?}; send patches instead",
                shape.grids
            )));
        }
        let token_ids = &observation.token_ids;
        let merge = self.config.vision.spatial_merge_size;
        let image_tokens: usize = shape
            .grids
            .iter()
            .map(|&[t, h, w]| t * (h / merge) * (w / merge))
            .sum();
        let action_tokens = self.config.action.action_horizon;
        let prefix_tokens = token_ids.len() - action_tokens;
        let prefix_ids = token_ids[..prefix_tokens].to_vec();
        if prefix_ids
            .iter()
            .any(|&id| id as usize >= self.config.text.vocab_size)
        {
            return Err(Error::Other(
                "walloss CPU token ID outside the embedding vocabulary".into(),
            ));
        }
        let mut vision_row_map = vec![u32::MAX; prefix_tokens];
        let mut vision_row = 0u32;
        for (row, &token) in prefix_ids.iter().enumerate() {
            if token == self.config.image_token_id {
                vision_row_map[row] = vision_row;
                vision_row += 1;
            }
        }
        if vision_row as usize != image_tokens {
            return Err(Error::Other(format!(
                "walloss prompt has {vision_row} image tokens, grids {:?} expect {image_tokens}",
                shape.grids
            )));
        }
        let positions =
            multimodal_position_ids(token_ids, &shape.grids, self.config.image_token_id, merge)?;
        let patches = match &observation.vision {
            VisionObservation::Patches(value) => {
                let expected = [geometry.patch_order.len(), self.model.patch_width()];
                host_f32(value, expected, "patches")?
            }
            VisionObservation::RgbU8 { bytes, layout } => {
                let views = geometry
                    .rgb_views
                    .ok_or_else(|| Error::Other("walloss CPU RGB input geometry is invalid".into()))?;
                let expected = views.views
                    * (views.grid_h * self.config.vision.patch_size)
                    * (views.grid_w * self.config.vision.patch_size)
                    * 3;
                if bytes.len() != expected {
                    return Err(Error::Other(format!(
                        "walloss expected {expected} RGB bytes for grids {:?}, got {}",
                        shape.grids,
                        bytes.len()
                    )));
                }
                let processor = self.model.image_processor().ok_or_else(|| {
                    Error::Other(
                        "walloss CPU RGB input requires preprocessor_config.json".into(),
                    )
                })?;
                patchify_rgb(bytes, *layout, views, &self.config, processor)?
            }
        };
        let initial_state = match request.initial_latent {
            InitialLatent::Provided(value) => {
                host_f32(value, self.action_shape(), "initial latent")?
            }
            InitialLatent::Generate { rng } => {
                let count = self.action_shape().iter().product();
                Tensor::from_f32(self.action_shape().to_vec(), &normal(count, rng))?
            }
        };
        let action_mask = match observation.action_mask.as_ref() {
            Some(value) => host_f32(value, self.action_shape(), "action mask")?,
            None => Tensor::from_f32(
                self.action_shape().to_vec(),
                &vec![1.0; self.action_shape().iter().product()],
            )?,
        };
        Ok(WallossHostInputs {
            patches,
            geometry,
            prefix_ids,
            vision_row_map,
            prefix_position_ids: positions[..prefix_tokens * 3].to_vec(),
            action_position_ids: positions[prefix_tokens * 3..].to_vec(),
            initial_state,
            action_mask,
        })
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let shape = self.request_shape(request)?;
        let spec = request.observation.inference_spec();
        let inputs = self.host_inputs(&spec, &shape, request)?;
        let output = self.model.infer(&inputs)?;
        if !output.to_f32_vec()?.iter().all(|value| value.is_finite()) {
            return Err(Error::Other(
                "walloss CPU inference produced non-finite actions".into(),
            ));
        }
        Ok(Action::new(output))
    }

    /// Placeholder request for `prepare`: default grids, all-zero inputs.
    fn placeholder_observation(&self, spec: &InferenceSpec) -> Result<Observation> {
        let action_tokens = self.config.action.action_horizon;
        let prefix_tokens = spec.token_count - action_tokens;
        let geometry = VisionGeometry::new(
            &self.config.vision,
            &DEFAULT_GRIDS
                .iter()
                .map(|&[t, h, w]| [t as usize, h as usize, w as usize])
                .collect::<Vec<_>>(),
        )?;
        if prefix_tokens < geometry.reverse_indices.len() {
            return Err(Error::Other(
                "walloss CPU token count leaves no room for the default image tokens".into(),
            ));
        }
        let mut token_ids = vec![self.config.image_token_id; geometry.reverse_indices.len()];
        token_ids.resize(spec.token_count, 0);
        let vision = match spec.image_layout {
            None => VisionObservation::Patches(Tensor::zeros(
                vec![geometry.patch_order.len(), self.model.patch_width()],
                DType::F32,
            )),
            Some(layout) => {
                let views = geometry
                    .rgb_views
                    .ok_or_else(|| Error::Other("walloss CPU default RGB geometry is invalid".into()))?;
                VisionObservation::RgbU8 {
                    bytes: vec![
                        0;
                        views.views
                            * (views.grid_h * self.config.vision.patch_size)
                            * (views.grid_w * self.config.vision.patch_size)
                            * 3
                    ],
                    layout,
                }
            }
        };
        Ok(Observation {
            vision,
            token_ids,
            state: None,
            action_mask: None,
        })
    }

    fn prepare_contract(
        &self,
        spec: InferenceSpec,
        policy: ExecutionPolicy,
        run: impl FnOnce() -> Result<()>,
    ) -> Result<Box<dyn PreparedInference>> {
        self.validate_spec(&spec)?;
        if policy == ExecutionPolicy::RequireGraph {
            return Err(Error::Other(
                "walloss CPU does not support RequireGraph".into(),
            ));
        }
        run()?;
        Ok(Box::new(CpuWallossPrepared {
            runner: self.clone(),
            spec,
            fallback: policy == ExecutionPolicy::PreferGraph,
        }))
    }
}

impl PreparedInference for CpuWallossPrepared {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }
    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: ExecutionMode::Eager,
            fallback_reason: self
                .fallback
                .then(|| "CPU reference does not support graph capture".into()),
        }
    }
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other(
                "walloss CPU prepared input specification mismatch".into(),
            ));
        }
        self.runner.run(request)
    }
}

impl VlaRuntime for CpuWallossRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some("fp32_reference")
    }

    fn contract(&self) -> VlaContract {
        let [_, grid_h, _] = DEFAULT_GRIDS[0];
        VlaContract {
            action_shape: self.action_shape(),
            patch_shape: [
                DEFAULT_GRIDS
                    .iter()
                    .map(|&[t, h, w]| (t * h * w) as usize)
                    .sum(),
                self.model.patch_width(),
            ],
            max_token_len: self.config.text.max_position_embeddings,
            num_views: DEFAULT_GRIDS.len(),
            image_size: grid_h as usize * self.config.vision.patch_size,
            patch_size: self.config.vision.patch_size,
            accepts_rgb_u8: self.model.image_processor().is_some(),
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.run(request)
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(spec, ExecutionPolicy::Eager)
    }

    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        self.prepare_contract(*spec, policy, || {
            let observation = self.placeholder_observation(spec)?;
            let noise = Tensor::zeros(self.action_shape().to_vec(), DType::F32);
            self.run(&VlaRequest::provided(&observation, &noise))?;
            Ok(())
        })
    }

    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        let spec = sample.observation.inference_spec();
        self.prepare_contract(spec, policy, || self.run(sample).map(|_| ()))
    }

    fn clear_prepared(&self) -> Result<()> {
        Ok(())
    }

    fn execution_mode(&self) -> &'static str {
        "eager"
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        self.infer(request)?.tensor().to_f32_vec()
    }
}

fn host_f32(tensor: &Tensor, shape: [usize; 2], name: &str) -> Result<Tensor> {
    if tensor.device() != Device::Cpu
        || tensor.shape().dims() != shape
        || !matches!(tensor.dtype(), DType::F32 | DType::F16 | DType::BF16)
    {
        return Err(Error::Other(format!(
            "walloss CPU {name} requires CPU floating point {shape:?}, got {:?} {}",
            tensor.shape().dims(),
            tensor.dtype()
        )));
    }
    let values = tensor.to_f32_vec()?;
    if !values.iter().all(|value| value.is_finite()) {
        return Err(Error::Other(format!(
            "walloss CPU {name} contains non-finite values"
        )));
    }
    if tensor.dtype() == DType::F32 {
        Ok(tensor.clone())
    } else {
        Tensor::from_f32(shape.to_vec(), &values)
    }
}

fn normal(count: usize, rng: RngKey) -> Vec<f32> {
    standard_normal_f32(count, rng)
}

/// Normalize and patchify resized RGB into merged-group rows.
fn patchify_rgb(
    bytes: &[u8],
    layout: ImageLayout,
    views: super::geometry::RgbViews,
    config: &WallossConfig,
    processor: &super::WallossImageProcessorConfig,
) -> Result<Tensor> {
    let patch = config.vision.patch_size;
    let temporal = config.vision.temporal_patch_size;
    let merge = config.vision.spatial_merge_size;
    let area = patch * patch;
    let patch_width = 3 * temporal * area;
    let height = views.grid_h * patch;
    let width = views.grid_w * patch;
    let patch_rows = views.views * views.grid_h * views.grid_w;
    let expected = patch_rows * patch_width;
    if bytes.len() != views.views * height * width * 3 {
        return Err(Error::Other("walloss CPU RGB byte count mismatch".into()));
    }
    let mut output = Vec::with_capacity(expected);
    let mutate = |channel: usize, y: usize, x: usize, view: usize| -> f32 {
        let index = match layout {
            ImageLayout::Nhwc => ((view * height + y) * width + x) * 3 + channel,
            ImageLayout::Nchw => ((view * 3 + channel) * height + y) * width + x,
        };
        let scaled = (bytes[index] as f64 * processor.rescale_factor) as f32;
        (scaled - processor.image_mean[channel]) / processor.image_std[channel]
    };
    for view in 0..views.views {
        for group_y in 0..(views.grid_h / merge) {
            for group_x in 0..(views.grid_w / merge) {
                for merge_y in 0..merge {
                    for merge_x in 0..merge {
                        for channel in 0..3 {
                            for _ in 0..temporal {
                                for dy in 0..patch {
                                    for dx in 0..patch {
                                        let y = (group_y * merge + merge_y) * patch + dy;
                                        let x = (group_x * merge + merge_x) * patch + dx;
                                        output.push(mutate(channel, y, x, view));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if output.len() != expected {
        return Err(Error::Other(
            "walloss CPU patchification produced the wrong row count".into(),
        ));
    }
    Tensor::from_f32(vec![patch_rows, patch_width], &output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_grids_match_the_public_contract() {
        let [_, h, w] = DEFAULT_GRIDS[0];
        assert_eq!(h, w);
        assert_eq!(DEFAULT_GRIDS.len(), 2);
    }

    #[test]
    fn patchify_layout_is_channel_temporal_patch_x() {
        // 1 view, 4x4 pixels, patch 2, temporal 2, merge 2.
        let config = WallossConfig::from_json_str(TEST_CONFIG).unwrap();
        let processor = super::super::WallossImageProcessorConfig {
            image_mean: [0.0, 0.0, 0.0],
            image_std: [1.0, 1.0, 1.0],
            rescale_factor: 1.0,
        };
        let views = super::super::geometry::RgbViews {
            views: 1,
            grid_h: 2,
            grid_w: 2,
        };
        let bytes: Vec<u8> = (0..16 * 3).map(|x| x as u8).collect();
        let patches = patchify_rgb(
            &bytes,
            ImageLayout::Nhwc,
            views,
            &config,
            &processor,
        )
        .unwrap();
        assert_eq!(patches.shape().dims(), [4, 24]);
        let values = patches.to_f32_vec().unwrap();
        // Row 0 is the first merged group (top-left): channel 0, both temporal
        // copies, then four patch elements in y/x order.
        assert_eq!(&values[0..8], &[0., 3., 12., 15., 0., 3., 12., 15.]);
    }

    #[test]
    fn normal_generation_is_reproducible() {
        let first = normal(8, RngKey::new(1, 2, 3));
        let second = normal(8, RngKey::new(1, 2, 3));
        assert_eq!(first, second);
        assert!(first.iter().all(|value| value.is_finite()));
    }

    const TEST_CONFIG: &str = r#"{
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "rms_norm_eps": 0.000001,
        "rope_theta": 10000.0,
        "rope_scaling": {"mrope_section": [1, 2, 1]},
        "action_hidden_size": 8,
        "experts": [
            {"hidden_size": 8, "intermediate_size": 16},
            {"hidden_size": 8, "intermediate_size": 16}
        ],
        "noise_scheduler": {"s": 0.999, "num_inference_timesteps": 2},
        "causal_action_attention_mask": true,
        "use_x_pred": false,
        "image_token_id": 7,
        "video_token_id": 8,
        "vision_start_token_id": 9,
        "vision_end_token_id": 10,
        "vision_config": {
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 8,
            "window_size": 4,
            "fullatt_block_indexes": []
        }
    }"#;
}
