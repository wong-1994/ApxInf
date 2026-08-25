use std::path::PathBuf;

use apxinf_core::{Device, Tensor};
use apxinf_model::{GrootRuntime, Observation, VisionObservation, VlaConditioning, VlaRuntime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args_os().nth(1)
        .map(PathBuf::from).ok_or("usage: groot_load_smoke MODEL_DIR")?;
    let runtime = GrootRuntime::from_dir(&model_dir, Device::Cuda(0))?;
    println!("GR00T checkpoint load: PASS");
    if let Some(fixture) = std::env::args_os().nth(2).map(PathBuf::from) {
        let f32s = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            Ok(std::fs::read(fixture.join(format!("{name}.f32")))?.chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect())
        };
        let i64s = |name: &str| -> Result<Vec<i64>, Box<dyn std::error::Error>> {
            Ok(std::fs::read(fixture.join(format!("{name}.i64")))?.chunks_exact(8)
                .map(|x| i64::from_le_bytes(x.try_into().unwrap())).collect())
        };
        let pixels = Tensor::from_f32(vec![512, 1536], &f32s("pixel_values")?)?;
        let state = Tensor::from_f32(vec![1, 132], &f32s("state")?)?;
        let noise = Tensor::from_f32(vec![40, 132], &f32s("noise")?)?;
        let observation = Observation {
            vision: VisionObservation::Patches(pixels),
            token_ids: i64s("input_ids")?.into_iter().map(|x| x as u32).collect(),
            noise,
            conditioning: VlaConditioning {
                state: Some(state), embodiment_id: Some(i64s("embodiment_id")?[0] as u32),
                image_grid_thw: i64s("image_grid_thw")?.into_iter().map(|x| x as u32).collect(),
                attention_mask: i64s("attention_mask")?.into_iter().map(|x| x as u8).collect(),
            },
        };
        let started = std::time::Instant::now();
        let actual = runtime.infer_host_f32(&observation)?;
        let elapsed = started.elapsed();
        let expected = f32s("actions")?;
        let max_abs = actual.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let error_l2 = actual.iter().zip(&expected).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        let reference_l2 = expected.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let relative_l2 = error_l2 / reference_l2.max(1e-12);
        println!("GR00T action parity: max_abs={max_abs:.8} relative_l2={relative_l2:.8} latency_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
        if max_abs > 0.02 || relative_l2 > 0.02 {
            return Err(format!("action parity exceeded tolerance: max_abs={max_abs}, relative_l2={relative_l2}").into());
        }
    }
    Ok(())
}
