use std::path::{Path, PathBuf};
use std::sync::Arc;

use apxinf_core::{Backend, Shape, Tensor};
use apxinf_cuda::CudaBackend;
use apxinf_model::groot::{CategorySpecificLinear, CategorySpecificMlp};
use half::bf16;

fn load_f32(root: &Path, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(root.join(format!("{name}.f32")))?;
    if bytes.len() % 4 != 0 { return Err(format!("{name}: invalid f32 byte count").into()); }
    Ok(bytes.chunks_exact(4).map(|item| f32::from_le_bytes(item.try_into().unwrap())).collect())
}

fn shape(manifest: &serde_json::Value, name: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    manifest["arrays"][name].as_array().ok_or_else(|| format!("missing shape {name}"))?
        .iter().map(|item| item.as_u64().map(|value| value as usize)
            .ok_or_else(|| format!("invalid shape {name}").into())).collect()
}

fn upload(backend: &dyn Backend, root: &Path, manifest: &serde_json::Value, name: &str)
    -> Result<Tensor, Box<dyn std::error::Error>> {
    let values = load_f32(root, name)?;
    let values = values.into_iter().map(bf16::from_f32).collect::<Vec<_>>();
    Ok(backend.to_device(&Tensor::from_bf16(Shape::new(shape(manifest, name)?), &values)?)?)
}

fn linear(backend: Arc<dyn Backend>, root: &Path, manifest: &serde_json::Value, alias: &str)
    -> Result<CategorySpecificLinear, Box<dyn std::error::Error>> {
    Ok(CategorySpecificLinear::new(
        vec![upload(&*backend, root, manifest, &format!("{alias}_weight"))?],
        vec![upload(&*backend, root, manifest, &format!("{alias}_bias"))?], backend)?)
}

fn comparison(backend: &dyn Backend, actual: &Tensor, reference: &[f32], name: &str) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let actual = backend.to_cpu(actual)?.to_f32_vec()?;
    if actual.len() != reference.len() { return Err(format!("{name}: length mismatch").into()); }
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut max_excess = f32::NEG_INFINITY;
    for (&actual, &reference) in actual.iter().zip(reference) {
        let difference = (actual - reference).abs();
        max_abs = max_abs.max(difference);
        max_rel = max_rel.max(difference / reference.abs().max(1e-12));
        max_excess = max_excess.max(difference - (0.02 + 0.02 * reference.abs()));
    }
    Ok(serde_json::json!({"name":name,"max_abs_error":max_abs,"max_rel_error":max_rel,
        "max_tolerance_excess":max_excess,"passed":max_excess <= 0.0}))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(std::env::args().nth(1).ok_or("usage: groot_operator_replay <export-dir>")?);
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    let cuda = Arc::new(CudaBackend::new(0)?);
    let backend: Arc<dyn Backend> = cuda;
    let mut comparisons = Vec::new();

    for alias in ["conv", "state1", "state2", "action1", "action2", "action3", "decoder1", "decoder2"] {
        let operation = linear(Arc::clone(&backend), &root, &manifest, alias)?;
        let input = upload(&*backend, &root, &manifest, &format!("{alias}_input"))?;
        let actual = operation.forward(&input, 0)?;
        comparisons.push(comparison(&*backend, &actual, &load_f32(&root, &format!("{alias}_output"))?, alias)?);
    }

    let state = CategorySpecificMlp::new(
        linear(Arc::clone(&backend), &root, &manifest, "state1")?,
        linear(Arc::clone(&backend), &root, &manifest, "state2")?, Arc::clone(&backend));
    let actual = state.forward(&upload(&*backend, &root, &manifest, "state1_input")?, 0)?;
    comparisons.push(comparison(&*backend, &actual, &load_f32(&root, "state_output")?, "state_mlp")?);

    let decoder = CategorySpecificMlp::new(
        linear(Arc::clone(&backend), &root, &manifest, "decoder1")?,
        linear(Arc::clone(&backend), &root, &manifest, "decoder2")?, Arc::clone(&backend));
    let actual = decoder.forward(&upload(&*backend, &root, &manifest, "decoder1_input")?, 0)?;
    comparisons.push(comparison(&*backend, &actual, &load_f32(&root, "decoder_output")?, "decoder_mlp")?);

    let passed = comparisons.iter().all(|item| item["passed"] == true);
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "schema_version":"1.0","comparison_rule":"abs(actual-reference) <= 0.02 + 0.02*abs(reference)",
        "comparisons":comparisons,"passed":passed}))?);
    if !passed { std::process::exit(1); }
    Ok(())
}
