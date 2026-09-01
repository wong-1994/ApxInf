use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::{DType, Device, Tensor};
use apxinf_model::walloss::WallossConfig;
use apxinf_model::{
    AutoModel, LoadOptions, ModelPrecision, Observation, VisionObservation, VlaRequest,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: walloss_bf16_smoke CHECKPOINT")?;
    let config = WallossConfig::from_json_file(&checkpoint.join("config.json"))?;
    let fp8_scale = std::env::var("APXINF_WALLOSS_FP8_SCALE")
        .ok()
        .map(|value| value.parse::<f32>())
        .transpose()?;
    let calibration_path = std::env::var_os("APXINF_WALLOSS_CALIBRATION").map(PathBuf::from);
    let tuning_path = std::env::var_os("APXINF_WALLOSS_TUNING").map(PathBuf::from);
    let dynamic_fp8 = std::env::var_os("APXINF_WALLOSS_FP8").is_some();
    let options = LoadOptions {
        model_name: Some("walloss".into()),
        precision: if dynamic_fp8 || fp8_scale.is_some() || calibration_path.is_some() {
            ModelPrecision::Fp8
        } else {
            ModelPrecision::Bf16
        },
        uniform_fp8_scale: fp8_scale,
        calibration_path,
        tuning_path,
        ..LoadOptions::default()
    };
    let load_start = Instant::now();
    let model = AutoModel::load_model(Device::Cuda(0), &checkpoint, &options)?;
    eprintln!("load_ms={:.3}", load_start.elapsed().as_secs_f64() * 1e3);
    let [action_horizon, action_dim] = model.vla()?.action_shape();
    let patch_rows = 2 * 18 * 18;
    let patch_width =
        3 * config.vision.temporal_patch_size * config.vision.patch_size * config.vision.patch_size;
    let fixture = std::env::var_os("APXINF_WALLOSS_FIXTURE_DIR").map(PathBuf::from);
    let (patches, token_ids, state, action_mask, latent) = if let Some(directory) = fixture {
        let read_f32 = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let bytes = std::fs::read(directory.join(name))?;
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect())
        };
        let input_bytes = std::fs::read(directory.join("input_ids.bin"))?;
        let token_ids = input_bytes
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()) as u32)
            .collect::<Vec<_>>();
        let patches = Tensor::from_f32((patch_rows, patch_width), &read_f32("pixel_values.bin")?)?;
        let state_values = read_f32("state.bin")?;
        let state = Some(Tensor::from_f32(
            (1, action_dim),
            &state_values,
        )?);
        let mask = read_f32("dof_mask.bin")?;
        let action_mask = Some(Tensor::from_f32(
            (action_horizon, action_dim),
            &mask
                .into_iter()
                .cycle()
                .take(action_horizon * action_dim)
                .collect::<Vec<_>>(),
        )?);
        let latent = Tensor::from_f32(
            (action_horizon, action_dim),
            &read_f32("initial_noise.bin")?,
        )?;
        (patches, token_ids, state, action_mask, latent)
    } else {
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
        token_ids.extend(std::iter::repeat_n(4, action_horizon));
        (
            Tensor::zeros((patch_rows, patch_width), DType::F32),
            token_ids,
            None,
            None,
            Tensor::zeros(
                (action_horizon, action_dim),
                DType::F32,
            ),
        )
    };
    let observation = Observation {
        vision: VisionObservation::Patches(patches),
        token_ids,
        state,
        action_mask,
    };
    let request = VlaRequest::provided(&observation, &latent);
    let profile_run = std::env::var("APXINF_PROFILE_RUN")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let runs = std::env::var("APXINF_WALLOSS_RUNS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(4);
    if runs == 0 {
        return Err("APXINF_WALLOSS_RUNS must be non-zero".into());
    }
    if profile_run.is_some_and(|run| run == 0 || run > runs) {
        return Err("APXINF_PROFILE_RUN must be between 1 and APXINF_WALLOSS_RUNS".into());
    }
    let mut reference = None::<Vec<f32>>;
    let mut elapsed_ms = Vec::with_capacity(runs);
    for run in 0..runs {
        if profile_run == Some(run + 1) {
            apxinf_cuda::profiler::start().map_err(std::io::Error::other)?;
        }
        let infer_start = Instant::now();
        let action = model.infer_host_f32(&request)?;
        let run_ms = infer_start.elapsed().as_secs_f64() * 1e3;
        elapsed_ms.push(run_ms);
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
            run_ms,
            action.len(),
            action.iter().all(|value| value.is_finite()),
            max_abs_diff,
        );
        reference.get_or_insert(action);
        if profile_run == Some(run + 1) {
            apxinf_cuda::profiler::stop().map_err(std::io::Error::other)?;
        }
    }
    if elapsed_ms.len() > 1 {
        let mut steady = elapsed_ms[1..].to_vec();
        steady.sort_by(f64::total_cmp);
        let percentile = |p: f64| {
            let index = ((steady.len() - 1) as f64 * p).round() as usize;
            steady[index]
        };
        let mean = steady.iter().sum::<f64>() / steady.len() as f64;
        eprintln!(
            "steady_runs={} mean_ms={mean:.3} p50_ms={:.3} p90_ms={:.3} p95_ms={:.3} min_ms={:.3} max_ms={:.3}",
            steady.len(),
            percentile(0.50),
            percentile(0.90),
            percentile(0.95),
            steady[0],
            steady[steady.len() - 1],
        );
    }
    if let (Some(path), Some(action)) = (
        std::env::var_os("APXINF_WALLOSS_OUTPUT_PATH"),
        reference.as_ref(),
    ) {
        let bytes = action
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(path, bytes)?;
    }
    Ok(())
}
