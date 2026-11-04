//! CPU-only WallOSS public loading, request and preparation contracts.
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use apxinf_core::{DType, Device, RngKey, Tensor};
use apxinf_model::{
    AutoModel, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, LoadOptions,
    Observation, PreparationStatus, SyntheticWeights, VisionObservation, VlaRequest,
};

const CONFIG: &str = r#"{
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 64,
    "max_position_embeddings": 512,
    "rms_norm_eps": 0.000001,
    "rope_theta": 10000.0,
    "rope_scaling": {"mrope_section": [1, 1, 0]},
    "action_hidden_size": 8,
    "state_hidden_size": 8,
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
        "depth": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_heads": 2,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 8,
        "window_size": 4,
        "fullatt_block_indexes": [1]
    }
}"#;

const PROCESSOR: &str = r#"{
    "image_processor_type": "Qwen2VLImageProcessor",
    "patch_size": 2,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "rescale_factor": 0.00392156862745098
}"#;

const ACTION_TOKENS: usize = 10;
const ACTION_DIM: usize = 4;
const IMAGE_TOKENS: usize = 162;

struct Fixture {
    root: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "apxinf-cpu-walloss-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("config.json"), CONFIG).unwrap();
        std::fs::write(root.join("preprocessor_config.json"), PROCESSOR).unwrap();
        Self { root }
    }

    fn path(&self) -> &Path {
        &self.root
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn options() -> LoadOptions {
    LoadOptions {
        model_name: Some("walloss".into()),
        model_variant: Some("fp32_reference".into()),
        synthetic: Some(SyntheticWeights { seed: 17 }),
        ..LoadOptions::default()
    }
}

fn token_ids() -> Vec<u32> {
    let mut ids = vec![7u32; IMAGE_TOKENS];
    ids.extend([1, 2]);
    ids.extend([3u32; ACTION_TOKENS]);
    ids
}

fn latent() -> Tensor {
    let values = (0..ACTION_TOKENS * ACTION_DIM)
        .map(|index| (index as f32 * 0.37).sin() * 0.5)
        .collect::<Vec<_>>();
    Tensor::from_f32(vec![ACTION_TOKENS, ACTION_DIM], &values).unwrap()
}

fn rgb_bytes() -> Vec<u8> {
    (0..2 * 36 * 36 * 3)
        .map(|index| (index * 37 % 256) as u8)
        .collect()
}

fn observation(vision: VisionObservation) -> Observation {
    Observation {
        vision,
        token_ids: token_ids(),
        state: None,
        action_mask: None,
    }
}

fn load(fixture: &Fixture) -> apxinf_model::LoadedModel {
    AutoModel::load_model(Device::Cpu, fixture.path(), &options()).unwrap()
}

/// Patch rows produced by the CPU RGB path: merged-group order, channel,
/// temporal copies, patch y, patch x.
fn patches_from_rgb(bytes: &[u8]) -> Tensor {
    let (patch, temporal, merge) = (2usize, 2usize, 2usize);
    let (grid_h, grid_w) = (18usize, 18usize);
    let (height, width) = (grid_h * patch, grid_w * patch);
    let area = patch * patch;
    let row_width = 3 * temporal * area;
    let rows = 2 * grid_h * grid_w;
    let mut out = vec![0.0f32; rows * row_width];
    let mut row = 0;
    for view in 0..2 {
        for group_y in 0..(grid_h / merge) {
            for group_x in 0..(grid_w / merge) {
                for merge_y in 0..merge {
                    for merge_x in 0..merge {
                        let mut column = 0;
                        for channel in 0..3 {
                            for _ in 0..temporal {
                                for dy in 0..patch {
                                    for dx in 0..patch {
                                        let y = (group_y * merge + merge_y) * patch + dy;
                                        let x = (group_x * merge + merge_x) * patch + dx;
                                        let index =
                                            ((view * height + y) * width + x) * 3 + channel;
                                        let scaled =
                                            (bytes[index] as f64 * (1.0 / 255.0)) as f32;
                                        out[row * row_width + column] =
                                            (scaled - 0.5) / 0.5;
                                        column += 1;
                                    }
                                }
                            }
                        }
                        row += 1;
                    }
                }
            }
        }
    }
    Tensor::from_f32(vec![rows, row_width], &out).unwrap()
}

#[test]
fn eager_cpu_loads_and_executes_without_cuda() {
    let fixture = Fixture::new();
    let model = load(&fixture);
    let vla = model.vla().unwrap();
    assert_eq!(vla.model_variant(), Some("fp32_reference"));
    assert_eq!(vla.execution_mode(), "eager");
    assert!(vla.contract().accepts_rgb_u8);

    let obs = observation(VisionObservation::RgbU8 {
        bytes: rgb_bytes(),
        layout: ImageLayout::Nhwc,
    });
    let noise = latent();
    let request = VlaRequest::provided(&obs, &noise);
    let action = vla.infer(&request).unwrap();
    assert_eq!(action.tensor().device(), Device::Cpu);
    assert_eq!(action.tensor().dtype(), DType::F32);
    assert_eq!(action.tensor().shape().dims(), [ACTION_TOKENS, ACTION_DIM]);
    let values = action.tensor().to_f32_vec().unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
    assert_ne!(values, noise.to_f32_vec().unwrap());
    assert_eq!(values, vla.infer_host_f32(&request).unwrap());
    assert_eq!(values, vla.infer_host_f32(&request).unwrap());
}

#[test]
fn rgb_and_patch_inputs_agree() {
    let fixture = Fixture::new();
    let model = load(&fixture);
    let vla = model.vla().unwrap();
    let bytes = rgb_bytes();
    let noise = latent();

    let rgb = observation(VisionObservation::RgbU8 {
        bytes: bytes.clone(),
        layout: ImageLayout::Nhwc,
    });
    let patches = observation(VisionObservation::Patches(patches_from_rgb(&bytes)));
    let from_rgb = vla
        .infer_host_f32(&VlaRequest::provided(&rgb, &noise))
        .unwrap();
    let from_patches = vla
        .infer_host_f32(&VlaRequest::provided(&patches, &noise))
        .unwrap();
    for (left, right) in from_rgb.iter().zip(&from_patches) {
        assert!((left - right).abs() < 1e-6, "{left} vs {right}");
    }
}

#[test]
fn generated_latents_are_reproducible() {
    let fixture = Fixture::new();
    let model = load(&fixture);
    let vla = model.vla().unwrap();
    let obs = observation(VisionObservation::RgbU8 {
        bytes: rgb_bytes(),
        layout: ImageLayout::Nchw,
    });
    let request = VlaRequest::generated(&obs, RngKey::new(7, 2, 3));
    let first = vla.infer_host_f32(&request).unwrap();
    assert_eq!(first, vla.infer_host_f32(&request).unwrap());
    assert!(first.iter().all(|value| value.is_finite()));
}

#[test]
fn prepare_is_eager_and_rejects_graph_requirements() {
    let fixture = Fixture::new();
    let model = load(&fixture);
    let vla = model.vla().unwrap();
    let spec = InferenceSpec {
        token_count: IMAGE_TOKENS + 12,
        image_layout: Some(ImageLayout::Nhwc),
    };
    let prepared = vla.prepare(&spec).unwrap();
    assert_eq!(prepared.spec(), &spec);
    match prepared.status() {
        PreparationStatus::Ready { mode, .. } => assert_eq!(mode, ExecutionMode::Eager),
        status => panic!("unexpected preparation status: {status:?}"),
    }
    let obs = observation(VisionObservation::RgbU8 {
        bytes: rgb_bytes(),
        layout: ImageLayout::Nhwc,
    });
    let action = prepared
        .run(&VlaRequest::provided(&obs, &latent()))
        .unwrap();
    assert_eq!(action.tensor().shape().dims(), [ACTION_TOKENS, ACTION_DIM]);

    assert!(vla
        .prepare_with_policy(&spec, ExecutionPolicy::RequireGraph)
        .is_err());
    let fallback = vla
        .prepare_with_policy(&spec, ExecutionPolicy::PreferGraph)
        .unwrap();
    match fallback.status() {
        PreparationStatus::Ready {
            mode,
            fallback_reason,
        } => {
            assert_eq!(mode, ExecutionMode::Eager);
            assert!(fallback_reason.is_some());
        }
        status => panic!("unexpected preparation status: {status:?}"),
    }
}

#[test]
fn prepare_for_retains_the_sample_specification() {
    let fixture = Fixture::new();
    let model = load(&fixture);
    let vla = model.vla().unwrap();
    let obs = observation(VisionObservation::RgbU8 {
        bytes: rgb_bytes(),
        layout: ImageLayout::Nhwc,
    });
    let noise = latent();
    let sample = VlaRequest::provided(&obs, &noise);
    let prepared = vla
        .prepare_for(&sample, ExecutionPolicy::Eager)
        .unwrap();
    assert_eq!(
        prepared.spec(),
        &InferenceSpec {
            token_count: IMAGE_TOKENS + 12,
            image_layout: Some(ImageLayout::Nhwc),
        }
    );
    assert_eq!(
        prepared.run(&sample).unwrap().tensor().to_f32_vec().unwrap(),
        vla.infer_host_f32(&sample).unwrap()
    );
}

#[test]
fn rejects_invalid_requests() {
    let fixture = Fixture::new();
    let model = load(&fixture);
    let vla = model.vla().unwrap();
    let noise = latent();

    // Wrong patch width.
    let patches = Tensor::zeros(vec![648, 23], DType::F32);
    let obs = observation(VisionObservation::Patches(patches));
    assert!(vla
        .infer(&VlaRequest::provided(&obs, &noise))
        .is_err());

    // No image-token run.
    let mut text_only = observation(VisionObservation::RgbU8 {
        bytes: rgb_bytes(),
        layout: ImageLayout::Nhwc,
    });
    text_only.token_ids = vec![1; IMAGE_TOKENS + 4];
    assert!(vla
        .infer(&VlaRequest::provided(&text_only, &noise))
        .is_err());

    // Wrong action mask shape.
    let mut masked = observation(VisionObservation::RgbU8 {
        bytes: rgb_bytes(),
        layout: ImageLayout::Nhwc,
    });
    masked.action_mask = Some(Tensor::zeros(vec![1, ACTION_DIM], DType::F32));
    assert!(vla
        .infer(&VlaRequest::provided(&masked, &noise))
        .is_err());

    // Token count beyond max_position_embeddings.
    let mut long = observation(VisionObservation::RgbU8 {
        bytes: rgb_bytes(),
        layout: ImageLayout::Nhwc,
    });
    long.token_ids.resize(513, 1);
    assert!(vla
        .infer(&VlaRequest::provided(&long, &noise))
        .is_err());
}
