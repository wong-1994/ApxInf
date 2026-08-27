use std::collections::BTreeMap;
use std::path::PathBuf;

use apxinf_core::{Backend, Tensor};
use apxinf_cuda::{
    kernels::gemm::{autotune_cublaslt_fp8, autotune_cutlass_fp8, cold_l2_tuning_metadata},
    tuning::{KERNEL_BUILD_ID, TUNING_SCHEMA_V1},
    CudaBackend,
};
use apxinf_model::walloss::{WallossConfig, WallossWeights};

#[derive(Default)]
struct ShapeUse {
    names: Vec<String>,
    repetitions: usize,
}

fn add_shape(
    shapes: &mut BTreeMap<(usize, usize, usize), ShapeUse>,
    m: usize,
    weight: &Tensor,
    name: impl Into<String>,
    repetitions: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let dims = weight.shape().dims();
    if dims.len() != 2 {
        return Err(format!("matrix has non-rank-two shape {dims:?}").into());
    }
    let usage = shapes.entry((m, dims[1], dims[0])).or_default();
    usage.names.push(name.into());
    usage.repetitions += repetitions;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: walloss_cutlass_tune CHECKPOINT [PREFIX_COUNTS] [WARMUP] [ITERATIONS]")?;
    let prefix_counts = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "169,217".into())
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    if prefix_counts.is_empty() || prefix_counts.contains(&0) {
        return Err("PREFIX_COUNTS must contain positive comma-separated integers".into());
    }
    let warmup = std::env::args()
        .nth(3)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(3usize);
    let iterations = std::env::args()
        .nth(4)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(12usize);
    if iterations == 0 {
        return Err("ITERATIONS must be non-zero".into());
    }

    let mut config = WallossConfig::from_json_file(&checkpoint.join("config.json"))?;
    let weights = WallossWeights::from_safetensors(&mut config, &checkpoint)?;
    let mut shapes = BTreeMap::<(usize, usize, usize), ShapeUse>::new();
    let patch_tokens = 2 * 18 * 18;
    let merged_tokens =
        patch_tokens / (config.vision.spatial_merge_size * config.vision.spatial_merge_size);

    add_shape(
        &mut shapes,
        patch_tokens,
        &weights.vision.patch_projection,
        "vision.patch_projection",
        1,
    )?;
    if let Some(block) = weights.vision.blocks.first() {
        add_shape(
            &mut shapes,
            patch_tokens,
            &block.qkv,
            "vision.qkv",
            config.vision.depth,
        )?;
        add_shape(
            &mut shapes,
            patch_tokens,
            &block.output,
            "vision.output",
            config.vision.depth,
        )?;
        add_shape(
            &mut shapes,
            patch_tokens,
            &block.gate_up,
            "vision.gate_up",
            config.vision.depth,
        )?;
        add_shape(
            &mut shapes,
            patch_tokens,
            &block.down,
            "vision.down",
            config.vision.depth,
        )?;
    }
    add_shape(
        &mut shapes,
        merged_tokens,
        &weights.vision.merger_hidden,
        "vision.merger_hidden",
        1,
    )?;
    add_shape(
        &mut shapes,
        merged_tokens,
        &weights.vision.merger_output,
        "vision.merger_output",
        1,
    )?;

    if let Some(layer) = weights.language_layers.first() {
        for &prefix_tokens in &prefix_counts {
            add_shape(
                &mut shapes,
                prefix_tokens,
                &layer.qkv,
                format!("language.qkv.m{prefix_tokens}"),
                config.text.num_layers,
            )?;
            add_shape(
                &mut shapes,
                prefix_tokens,
                &layer.output,
                format!("language.output.m{prefix_tokens}"),
                config.text.num_layers,
            )?;
            add_shape(
                &mut shapes,
                prefix_tokens,
                &layer.gate_up,
                format!("language.gate_up.m{prefix_tokens}"),
                config.text.num_layers,
            )?;
            add_shape(
                &mut shapes,
                prefix_tokens,
                &layer.down,
                format!("language.down.m{prefix_tokens}"),
                config.text.num_layers,
            )?;
        }
    }
    if let Some(layer) = weights.action_layers.first() {
        let action_tokens = config.action.action_horizon;
        let repetitions = config.text.num_layers * config.action.solver_steps;
        add_shape(
            &mut shapes,
            action_tokens,
            &layer.qkv,
            "action.qkv",
            repetitions,
        )?;
        add_shape(
            &mut shapes,
            action_tokens,
            &layer.output,
            "action.output",
            repetitions,
        )?;
        add_shape(
            &mut shapes,
            action_tokens,
            &layer.gate_up,
            "action.gate_up",
            repetitions,
        )?;
        add_shape(
            &mut shapes,
            action_tokens,
            &layer.down,
            "action.down",
            repetitions,
        )?;
    }

    let backend = CudaBackend::new(0)?;
    let cold_l2 = cold_l2_tuning_metadata(backend.context())?;
    let mut tactics = serde_json::Map::new();
    for ((m, n, k), usage) in shapes {
        eprintln!(
            "cold-L2 exact tune M={m} N={n} K={k} uses={}",
            usage.repetitions
        );
        let activation = backend.to_device(&Tensor::from_f8_e4m3(vec![m, k], &vec![0; m * k])?)?;
        let weight = backend.to_device(&Tensor::from_f8_e4m3(vec![k, n], &vec![0; k * n])?)?;
        let cutlass = if n % 16 == 0 && k % 16 == 0 {
            autotune_cutlass_fp8(
                backend.context(),
                &activation,
                &weight,
                1.0,
                1.0,
                warmup,
                iterations,
            )?
        } else {
            Vec::new()
        };
        let cublaslt = autotune_cublaslt_fp8(
            backend.context(),
            &activation,
            &weight,
            1.0,
            1.0,
            32,
            warmup,
            iterations,
        )?;
        let best_cutlass = cutlass
            .iter()
            .min_by(|a, b| a.milliseconds.total_cmp(&b.milliseconds));
        let best_cublaslt = cublaslt
            .iter()
            .min_by(|a, b| a.milliseconds.total_cmp(&b.milliseconds));
        let (backend_name, tactic, milliseconds) = match (best_cutlass, best_cublaslt) {
            (Some(a), Some(b)) if b.milliseconds < a.milliseconds => {
                ("cublaslt", b.heuristic_rank, b.milliseconds)
            }
            (Some(a), _) => ("cutlass", a.tactic, a.milliseconds),
            (None, Some(b)) => ("cublaslt", b.heuristic_rank, b.milliseconds),
            (None, None) => return Err(format!("no FP8 tactic accepted [{m},{n},{k}]").into()),
        };
        tactics.insert(
            format!("fp8_f16_m{m}_n{n}_k{k}"),
            serde_json::json!({
                "backend": backend_name,
                "tactic": tactic,
                "milliseconds": milliseconds,
                "workloads": usage.names,
                "repetitions": usage.repetitions,
                "cutlass_candidates": cutlass.iter().map(|x| serde_json::json!({"tactic": x.tactic, "milliseconds": x.milliseconds})).collect::<Vec<_>>(),
                "cublaslt_candidates": cublaslt.iter().map(|x| serde_json::json!({"heuristic_rank": x.heuristic_rank, "milliseconds": x.milliseconds})).collect::<Vec<_>>(),
            }),
        );
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": TUNING_SCHEMA_V1,
            "kernel_build_id": KERNEL_BUILD_ID,
            "device_name": backend.context().caps().device_name.as_str(),
            "sm": backend.context().caps().sm,
            "cuda_version": backend.context().library_versions().cuda.as_str(),
            "cublas_version": backend.context().library_versions().cublas.as_str(),
            "generator": {"name": "walloss_cutlass_tune", "method": "cold_l2_exact_shape"},
            "measurement": {
                "warmup_iterations": warmup,
                "benchmark_iterations": iterations,
                "l2_cache_bytes": cold_l2.l2_cache_bytes,
                "eviction_buffer_bytes": cold_l2.eviction_buffer_bytes,
                "selection": "minimum candidate mean; exact physical M/N/K"
            },
            "prefix_counts": prefix_counts,
            "tactics": tactics
        }))?
    );
    Ok(())
}
