//! Full-checkpoint CPU FP32 vs CUDA BF16 comparison.
//!
//! This case is opt-in: set `APXINF_WALLOSS_TEST_MODEL` to a `wall-oss-0.5`
//! checkpoint and select it explicitly. A selected run without the variable
//! fails instead of silently skipping.

#![cfg(feature = "cuda")]

use std::path::Path;

use apxinf_core::{Device, Tensor};
use apxinf_model::{
    AutoModel, ImageLayout, LoadOptions, Observation, VisionObservation, VlaRequest, VlaRuntime,
};

const ACTION_TOKENS: usize = 10;

fn image_token(root: &Path) -> u32 {
    let raw = std::fs::read_to_string(root.join("config.json")).unwrap();
    let value: serde_json::Value = serde_json::from_str(&raw).unwrap();
    value["image_token_id"].as_u64().unwrap() as u32
}

fn observation(root: &Path) -> Observation {
    let image_token = image_token(root);
    let mut token_ids = vec![image_token; 162];
    token_ids.extend([1000, 1001, 1002, 1003]);
    token_ids.extend([1004u32; ACTION_TOKENS]);
    let bytes = (0..2 * 252 * 252 * 3)
        .map(|index| (index * 29 % 256) as u8)
        .collect();
    Observation {
        vision: VisionObservation::RgbU8 {
            bytes,
            layout: ImageLayout::Nhwc,
        },
        token_ids,
        state: None,
        action_mask: None,
    }
}

fn cpu_options() -> LoadOptions {
    LoadOptions {
        model_name: Some("walloss".into()),
        model_variant: Some("fp32_reference".into()),
        ..LoadOptions::default()
    }
}

fn cuda_options() -> LoadOptions {
    LoadOptions {
        model_name: Some("walloss".into()),
        ..LoadOptions::default()
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (&a, &b) in left.iter().zip(right) {
        dot += a as f64 * b as f64;
        left_norm += (a as f64) * (a as f64);
        right_norm += (b as f64) * (b as f64);
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

#[test]
#[ignore = "requires a wall-oss checkpoint and CUDA"]
fn cpu_reference_matches_cuda_bf16_on_checkpoint() {
    let root = std::env::var_os("APXINF_WALLOSS_TEST_MODEL")
        .map(std::path::PathBuf::from)
        .expect("APXINF_WALLOSS_TEST_MODEL must name the wall-oss checkpoint");
    let observation = observation(&root);

    let (cpu_actions, action_shape) = {
        let model = AutoModel::load_model(Device::Cpu, &root, &cpu_options()).unwrap();
        let vla = model.vla().unwrap();
        let action_shape = vla.contract().action_shape;
        let values = (0..action_shape[0] * action_shape[1])
            .map(|index| ((index as f32 * 0.37).sin() * 0.5) as f32)
            .collect::<Vec<_>>();
        let latent = Tensor::from_f32(action_shape.to_vec(), &values).unwrap();
        let request = VlaRequest::provided(&observation, &latent);
        let actions = vla.infer_host_f32(&request).unwrap();
        drop(model);
        (actions, action_shape)
    };
    assert_eq!(
        action_shape,
        [ACTION_TOKENS, 7],
        "unexpected LIBERO checkpoint action geometry"
    );

    let values = (0..action_shape[0] * action_shape[1])
        .map(|index| ((index as f32 * 0.37).sin() * 0.5) as f32)
        .collect::<Vec<_>>();
    let latent = Tensor::from_f32(action_shape.to_vec(), &values).unwrap();
    let request = VlaRequest::provided(&observation, &latent);

    let model = AutoModel::load_model(Device::Cuda(0), &root, &cuda_options()).unwrap();
    let vla = model.vla().unwrap();
    let cuda_actions = vla.infer_host_f32(&request).unwrap();
    assert_eq!(cpu_actions.len(), cuda_actions.len());

    let cosine = cosine(&cpu_actions, &cuda_actions);
    let mut l2 = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut cpu_norm = 0.0f64;
    for (&cpu, &cuda) in cpu_actions.iter().zip(&cuda_actions) {
        l2 += (cpu as f64 - cuda as f64).powi(2);
        cpu_norm += (cpu as f64) * (cpu as f64);
        max_abs = max_abs.max((cpu as f64 - cuda as f64).abs());
    }
    let relative_l2 = (l2 / cpu_norm).sqrt();
    eprintln!(
        "walloss cpu fp32 vs cuda bf16: cosine={cosine:.9} relative_l2={relative_l2:.9} max_abs={max_abs:.9}"
    );
    assert!(cosine > 0.99, "cosine {cosine} is below the reference floor");
}
