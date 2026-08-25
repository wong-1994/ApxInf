use std::path::{Path, PathBuf};

use apxinf_core::{Device, Tensor};
use apxinf_model::{DebugCapture, DebugConfig, GrootRuntime, Observation, VisionObservation, VlaConditioning};

fn load(root: &Path, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(root.join(format!("{name}.f32")))?;
    if bytes.len() % 4 != 0 { return Err(format!("{name}: invalid f32 data").into()); }
    Ok(bytes.chunks_exact(4).map(|item| f32::from_le_bytes(item.try_into().unwrap())).collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1).map(PathBuf::from);
    let model_dir = arguments.next().ok_or("usage: groot_infer_replay MODEL_DIR EXPORT_DIR")?;
    let root = arguments.next().ok_or("usage: groot_infer_replay MODEL_DIR EXPORT_DIR")?;
    let debug_path = arguments.next();
    let runtime = GrootRuntime::from_dir(&model_dir, Device::Cuda(0))?;
    let pixels = load(&root, "observation_pixel_values")?;
    let state = load(&root, "observation_state")?;
    let noise = load(&root, "observation_noise")?;
    let token_ids = load(&root, "observation_input_ids")?.into_iter().map(|value| value as u32).collect();
    let attention_mask = load(&root, "observation_attention_mask")?.into_iter().map(|value| value as u8).collect();
    let image_grid_thw = load(&root, "observation_image_grid_thw")?.into_iter().map(|value| value as u32).collect();
    let observation = Observation {
        vision: VisionObservation::Patches(Tensor::from_f32(vec![256, 1536], &pixels)?),
        token_ids,
        noise: Tensor::from_f32(vec![40, 132], &noise)?,
        conditioning: VlaConditioning {
            state: Some(Tensor::from_f32(vec![1, 132], &state)?),
            embodiment_id: Some(2), image_grid_thw, attention_mask,
        },
    };
    let actual = if let Some(path) = debug_path {
        let mut debug = DebugCapture::new(DebugConfig::new(path.clone()));
        let actual = runtime.infer_host_f32_with_debug(&observation, &mut debug)?;
        debug.save(&path)?;
        actual
    } else {
        runtime.infer_host_f32_with_debug(
            &observation,
            &mut DebugCapture::new(DebugConfig::default()),
        )?
    };
    let reference = load(&root, "observation_reference_actions")?;
    let mut max_abs = 0.0f32; let mut max_rel = 0.0f32; let mut max_excess = f32::NEG_INFINITY;
    for (&actual, &reference) in actual.iter().zip(&reference) {
        let difference = (actual - reference).abs();
        max_abs = max_abs.max(difference);
        max_rel = max_rel.max(difference / reference.abs().max(1e-12));
        max_excess = max_excess.max(difference - (0.02 + 0.02 * reference.abs()));
    }
    let passed = actual.len() == reference.len() && max_excess <= 0.0;
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "schema_version":"1.0","checkpoint":"normalized_actions",
        "shape":[40,132],"max_abs_error":max_abs,"max_rel_error":max_rel,
        "max_tolerance_excess":max_excess,"passed":passed}))?);
    if !passed { std::process::exit(1); }
    Ok(())
}
