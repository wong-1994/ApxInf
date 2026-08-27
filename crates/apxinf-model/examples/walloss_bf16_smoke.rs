use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::{DType, Device, Tensor};
use apxinf_model::{
    AutoModel, LoadOptions, ModelPrecision, Observation, VisionObservation, VlaRequest,
};
use apxinf_model::walloss::WallossConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: walloss_bf16_smoke CHECKPOINT")?;
    let config = WallossConfig::from_json_file(&checkpoint.join("config.json"))?;
    let image_tokens_per_view = (18usize / config.vision.spatial_merge_size).pow(2);
    let mut token_ids = vec![1, config.vision_start_token_id];
    token_ids.extend(std::iter::repeat_n(
        config.image_token_id,
        image_tokens_per_view,
    ));
    token_ids.extend([config.vision_end_token_id, 2, config.vision_start_token_id]);
    token_ids.extend(std::iter::repeat_n(
        config.image_token_id,
        image_tokens_per_view,
    ));
    token_ids.extend([config.vision_end_token_id, 3]);
    token_ids.extend(std::iter::repeat_n(4, config.action.action_horizon));

    let patch_rows = 2 * 18 * 18;
    let patch_width = 3
        * config.vision.temporal_patch_size
        * config.vision.patch_size
        * config.vision.patch_size;
    let observation = Observation {
        vision: VisionObservation::Patches(Tensor::zeros(
            (patch_rows, patch_width),
            DType::F32,
        )),
        token_ids,
        state: None,
        action_mask: None,
    };
    let latent = Tensor::zeros(
        (config.action.action_horizon, config.action.action_dim),
        DType::F32,
    );
    let fp8_scale = std::env::var("APXINF_WALLOSS_FP8_SCALE")
        .ok()
        .map(|value| value.parse::<f32>())
        .transpose()?;
    let options = LoadOptions {
        model_name: Some("walloss".into()),
        precision: if fp8_scale.is_some() {
            ModelPrecision::Fp8
        } else {
            ModelPrecision::Bf16
        },
        uniform_fp8_scale: fp8_scale,
        ..LoadOptions::default()
    };
    let load_start = Instant::now();
    let model = AutoModel::load_model(Device::Cuda(0), &checkpoint, &options)?;
    eprintln!("load_ms={:.3}", load_start.elapsed().as_secs_f64() * 1e3);
    let request = VlaRequest::provided(&observation, &latent);
    let profile = std::env::var_os("APXINF_PROFILE_RUN").is_some();
    let mut reference = None::<Vec<f32>>;
    for run in 0..4 {
        if profile && run == 2 {
            apxinf_cuda::profiler::start().map_err(std::io::Error::other)?;
        }
        let infer_start = Instant::now();
        let action = model.infer_host_f32(&request)?;
        let max_abs_diff = reference
            .as_ref()
            .map(|expected| {
                expected
                    .iter()
                    .zip(&action)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            })
            .unwrap_or(0.0);
        eprintln!(
            "infer_run={} infer_ms={:.3} output={} finite={} max_abs_diff={:.6}",
            run + 1,
            infer_start.elapsed().as_secs_f64() * 1e3,
            action.len(),
            action.iter().all(|value| value.is_finite()),
            max_abs_diff,
        );
        reference.get_or_insert(action);
        if profile && run == 2 {
            apxinf_cuda::profiler::stop().map_err(std::io::Error::other)?;
        }
    }
    Ok(())
}
