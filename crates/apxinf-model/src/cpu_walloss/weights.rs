//! Checkpoint mapping for the CPU WallOSS reference.
//!
//! A copy owned by `cpu_walloss`: it reads the same checkpoint names and
//! transposes every linear weight into the BLAS-ready `[input, output]`
//! layout, but keeps all values in FP32. No device packing, FP8 decoding or
//! tile-alignment padding is applied.

use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{DType, Error, Result, Tensor};

use super::WallossConfig;

pub struct WallossWeights {
    pub token_embedding: Tensor,
    pub language_layers: Vec<WallossLayerWeights>,
    pub action_layers: Vec<WallossLayerWeights>,
    pub action_norm: Tensor,
    pub vision: WallossVisionWeights,
    pub action: WallossActionWeights,
}

pub struct WallossLayerWeights {
    pub input_norm: Tensor,
    pub post_attention_norm: Tensor,
    /// `[input, q + k + v]` projection.
    pub qkv: Tensor,
    pub qkv_bias: Tensor,
    /// `[attention_width, output]` projection.
    pub output: Tensor,
    /// `[hidden, 2 * intermediate]` fused gate/up projection.
    pub gate_up: Tensor,
    /// `[intermediate, hidden]` projection.
    pub down: Tensor,
}

pub struct WallossVisionWeights {
    /// Flattened and transposed patch projection.
    pub patch_projection: Tensor,
    pub blocks: Vec<WallossVisionBlockWeights>,
    pub merger_norm: Tensor,
    pub merger_hidden: Tensor,
    pub merger_hidden_bias: Tensor,
    pub merger_output: Tensor,
    pub merger_output_bias: Tensor,
}

pub struct WallossVisionBlockWeights {
    pub input_norm: Tensor,
    pub qkv: Tensor,
    pub qkv_bias: Tensor,
    pub output: Tensor,
    pub output_bias: Tensor,
    pub post_attention_norm: Tensor,
    pub gate_up: Tensor,
    pub gate_up_bias: Tensor,
    pub down: Tensor,
    pub down_bias: Tensor,
}

pub struct WallossActionWeights {
    pub noisy_action_projection: Tensor,
    pub dof_projection: Tensor,
    pub action_projection: Tensor,
    pub time_projection: Tensor,
    pub action_embedding_projection: Tensor,
    pub velocity_projection: Tensor,
}

impl WallossWeights {
    pub fn from_safetensors(config: &mut WallossConfig, path: &Path) -> Result<Self> {
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(path)
            .map_err(|error| Error::Other(format!("load walloss checkpoint: {error}")))?;
        Self::from_map(config, tensors)
    }

    pub fn from_map(
        config: &mut WallossConfig,
        mut tensors: HashMap<String, Tensor>,
    ) -> Result<Self> {
        let token_embedding = to_f32(&take(&mut tensors, "model.embed_tokens.weight")?)?;
        expect_rank(&token_embedding, 2, "model.embed_tokens.weight")?;
        let embedding_shape = token_embedding.shape().dims();
        if embedding_shape[1] != config.text.hidden_size {
            return Err(Error::Other(format!(
                "walloss checkpoint: embedding width {} != configured {}",
                embedding_shape[1], config.text.hidden_size
            )));
        }
        // Added control tokens make the tensor shape authoritative.
        config.text.vocab_size = embedding_shape[0];
        let action_dim = checkpoint_action_dim(&tensors, config.action.hidden_size)?;
        config.action.action_dim = action_dim;
        config.action.proprio_dim = action_dim;

        let q_width = config.text.num_attention_heads * config.text.head_dim;
        let kv_width = config.text.num_kv_heads * config.text.head_dim;
        let qkv_width = q_width + 2 * kv_width;
        let mut language_layers = Vec::with_capacity(config.text.num_layers);
        let mut action_layers = Vec::with_capacity(config.text.num_layers);
        for layer in 0..config.text.num_layers {
            language_layers.push(load_layer(
                &mut tensors,
                layer,
                0,
                config.text.hidden_size,
                config.text.intermediate_size,
                qkv_width,
                q_width,
            )?);
            action_layers.push(load_layer(
                &mut tensors,
                layer,
                1,
                config.action.hidden_size,
                config.action.intermediate_size,
                qkv_width,
                q_width,
            )?);
        }

        let language_norm = take(&mut tensors, "model.norms.0.weight")?;
        let action_norm = to_f32(&take(&mut tensors, "model.norms.1.weight")?)?;
        expect_shape(
            &language_norm,
            &[config.text.hidden_size],
            "language final norm",
        )?;
        expect_shape(
            &action_norm,
            &[config.action.hidden_size],
            "action final norm",
        )?;
        // The language final norm has no numerical consumer because only the
        // per-layer KV cache reaches denoising; require it to reject
        // incomplete checkpoints, matching the CUDA loader.
        drop(language_norm);

        Ok(Self {
            token_embedding,
            language_layers,
            action_layers,
            action_norm,
            vision: load_vision(config, &mut tensors)?,
            action: load_action(config, &mut tensors)?,
        })
    }

    /// Deterministic small weights with checkpoint shapes, for contract tests.
    pub fn synthetic(config: &WallossConfig, seed: u64) -> Result<Self> {
        let text = &config.text;
        let q_width = text.num_attention_heads * text.head_dim;
        let kv_width = text.num_kv_heads * text.head_dim;
        let qkv_width = q_width + 2 * kv_width;
        let mut rng = Rng::new(seed);
        let mut language_layers = Vec::with_capacity(text.num_layers);
        let mut action_layers = Vec::with_capacity(text.num_layers);
        for _ in 0..text.num_layers {
            language_layers.push(synthetic_layer(
                &mut rng,
                text.hidden_size,
                text.intermediate_size,
                qkv_width,
                q_width,
            )?);
            action_layers.push(synthetic_layer(
                &mut rng,
                config.action.hidden_size,
                config.action.intermediate_size,
                qkv_width,
                q_width,
            )?);
        }
        Ok(Self {
            token_embedding: synthetic_tensor(
                &mut rng,
                &[text.vocab_size, text.hidden_size],
                0.05,
            )?,
            language_layers,
            action_layers,
            action_norm: ones(&[config.action.hidden_size])?,
            vision: synthetic_vision(config, &mut rng)?,
            action: synthetic_action(config, &mut rng)?,
        })
    }
}

fn synthetic_layer(
    rng: &mut Rng,
    hidden: usize,
    intermediate: usize,
    qkv_width: usize,
    attention_width: usize,
) -> Result<WallossLayerWeights> {
    Ok(WallossLayerWeights {
        input_norm: ones(&[hidden])?,
        post_attention_norm: ones(&[hidden])?,
        qkv: synthetic_tensor(rng, &[hidden, qkv_width], 0.05)?,
        qkv_bias: zeros(&[qkv_width])?,
        output: synthetic_tensor(rng, &[attention_width, hidden], 0.05)?,
        gate_up: synthetic_tensor(rng, &[hidden, 2 * intermediate], 0.05)?,
        down: synthetic_tensor(rng, &[intermediate, hidden], 0.05)?,
    })
}

fn synthetic_vision(config: &WallossConfig, rng: &mut Rng) -> Result<WallossVisionWeights> {
    let v = &config.vision;
    let out = v.out_hidden_size;
    let merge_unit = v.spatial_merge_size * v.spatial_merge_size;
    let mut blocks = Vec::with_capacity(v.depth);
    for _ in 0..v.depth {
        blocks.push(WallossVisionBlockWeights {
            input_norm: ones(&[v.hidden_size])?,
            qkv: synthetic_tensor(rng, &[v.hidden_size, 3 * v.hidden_size], 0.05)?,
            qkv_bias: zeros(&[3 * v.hidden_size])?,
            output: synthetic_tensor(rng, &[v.hidden_size, v.hidden_size], 0.05)?,
            output_bias: zeros(&[v.hidden_size])?,
            post_attention_norm: ones(&[v.hidden_size])?,
            gate_up: synthetic_tensor(rng, &[v.hidden_size, 2 * v.intermediate_size], 0.05)?,
            gate_up_bias: zeros(&[2 * v.intermediate_size])?,
            down: synthetic_tensor(rng, &[v.intermediate_size, v.hidden_size], 0.05)?,
            down_bias: zeros(&[v.hidden_size])?,
        });
    }
    let patch_columns = 3 * v.temporal_patch_size * v.patch_size * v.patch_size;
    Ok(WallossVisionWeights {
        patch_projection: synthetic_tensor(rng, &[patch_columns, v.hidden_size], 0.05)?,
        blocks,
        merger_norm: ones(&[v.hidden_size])?,
        merger_hidden: synthetic_tensor(rng, &[merge_unit * v.hidden_size, out], 0.05)?,
        merger_hidden_bias: zeros(&[out])?,
        merger_output: synthetic_tensor(rng, &[out, out], 0.05)?,
        merger_output_bias: zeros(&[out])?,
    })
}

fn synthetic_action(config: &WallossConfig, rng: &mut Rng) -> Result<WallossActionWeights> {
    let width = config.action.hidden_size;
    let dim = config.action.action_dim;
    Ok(WallossActionWeights {
        noisy_action_projection: synthetic_tensor(rng, &[dim, width], 0.05)?,
        dof_projection: synthetic_tensor(rng, &[dim, width], 0.05)?,
        action_projection: synthetic_tensor(rng, &[width, width], 0.05)?,
        time_projection: synthetic_tensor(rng, &[width, width], 0.05)?,
        action_embedding_projection: synthetic_tensor(rng, &[width, width], 0.05)?,
        velocity_projection: synthetic_tensor(rng, &[width, dim], 0.05)?,
    })
}

fn load_layer(
    tensors: &mut HashMap<String, Tensor>,
    layer: usize,
    branch: usize,
    hidden: usize,
    intermediate: usize,
    qkv_width: usize,
    attention_width: usize,
) -> Result<WallossLayerWeights> {
    let prefix = format!("model.layers.{layer}");
    let input_norm = to_f32(&take(
        tensors,
        &format!("{prefix}.input_layernorms.{branch}.weight"),
    )?)?;
    let post_attention_norm = to_f32(&take(
        tensors,
        &format!("{prefix}.post_attention_layernorms.{branch}.weight"),
    )?)?;
    let qkv_source = take(
        tensors,
        &format!("{prefix}.self_attn.qkv_proj_experts.{branch}.weight"),
    )?;
    let qkv_bias = to_f32(&take(
        tensors,
        &format!("{prefix}.self_attn.qkv_proj_experts.{branch}.bias"),
    )?)?;
    let output_source = take(
        tensors,
        &format!("{prefix}.self_attn.o_proj_experts.{branch}.weight"),
    )?;
    let gate_up_source = take(
        tensors,
        &format!("{prefix}.moe.experts.{branch}.gate_up_proj.weight"),
    )?;
    let down_source = take(
        tensors,
        &format!("{prefix}.moe.experts.{branch}.down_proj.weight"),
    )?;

    expect_shape(&input_norm, &[hidden], "input norm")?;
    expect_shape(&post_attention_norm, &[hidden], "post-attention norm")?;
    expect_shape(&qkv_source, &[qkv_width, hidden], "QKV projection")?;
    expect_shape(&qkv_bias, &[qkv_width], "QKV bias")?;
    expect_shape(
        &output_source,
        &[hidden, attention_width],
        "output projection",
    )?;
    expect_shape(
        &gate_up_source,
        &[2 * intermediate, hidden],
        "gate/up projection",
    )?;
    expect_shape(&down_source, &[hidden, intermediate], "down projection")?;

    Ok(WallossLayerWeights {
        input_norm,
        post_attention_norm,
        qkv: transpose_2d(&qkv_source)?,
        qkv_bias,
        output: transpose_2d(&output_source)?,
        gate_up: transpose_2d(&gate_up_source)?,
        down: transpose_2d(&down_source)?,
    })
}

fn load_vision(
    config: &WallossConfig,
    tensors: &mut HashMap<String, Tensor>,
) -> Result<WallossVisionWeights> {
    let vision = &config.vision;
    if vision.hidden_size % vision.num_heads != 0 {
        return Err(Error::Other(
            "walloss checkpoint: invalid vision head geometry".into(),
        ));
    }
    let mut blocks = Vec::with_capacity(vision.depth);
    for layer in 0..vision.depth {
        let prefix = format!("visual.blocks.{layer}");
        blocks.push(WallossVisionBlockWeights {
            input_norm: to_f32(&take(tensors, &format!("{prefix}.norm1.weight"))?)?,
            qkv: transpose_2d(&take(tensors, &format!("{prefix}.attn.qkv.weight"))?)?,
            qkv_bias: to_f32(&take(tensors, &format!("{prefix}.attn.qkv.bias"))?)?,
            output: transpose_2d(&take(tensors, &format!("{prefix}.attn.proj.weight"))?)?,
            output_bias: to_f32(&take(tensors, &format!("{prefix}.attn.proj.bias"))?)?,
            post_attention_norm: to_f32(&take(tensors, &format!("{prefix}.norm2.weight"))?)?,
            gate_up: transpose_2d(&take(
                tensors,
                &format!("{prefix}.mlp.gate_up_proj.weight"),
            )?)?,
            gate_up_bias: to_f32(&take(
                tensors,
                &format!("{prefix}.mlp.gate_up_proj.bias"),
            )?)?,
            down: transpose_2d(&take(tensors, &format!("{prefix}.mlp.down_proj.weight"))?)?,
            down_bias: to_f32(&take(tensors, &format!("{prefix}.mlp.down_proj.bias"))?)?,
        });
    }

    let patch = take(tensors, "visual.patch_embed.proj.weight")?;
    let patch_columns = 3 * vision.temporal_patch_size * vision.patch_size * vision.patch_size;
    let patch_projection = flatten_and_transpose(&patch, vision.hidden_size, patch_columns)?;

    Ok(WallossVisionWeights {
        patch_projection,
        blocks,
        merger_norm: to_f32(&take(tensors, "visual.merger.ln_q.weight")?)?,
        merger_hidden: transpose_2d(&take(tensors, "visual.merger.mlp.0.weight")?)?,
        merger_hidden_bias: to_f32(&take(tensors, "visual.merger.mlp.0.bias")?)?,
        merger_output: transpose_2d(&take(tensors, "visual.merger.mlp.2.weight")?)?,
        merger_output_bias: to_f32(&take(tensors, "visual.merger.mlp.2.bias")?)?,
    })
}

fn load_action(
    config: &WallossConfig,
    tensors: &mut HashMap<String, Tensor>,
) -> Result<WallossActionWeights> {
    let width = config.action.hidden_size;
    let dim = config.action.action_dim;
    let noisy_action_projection = take(tensors, "action_preprocessor.w1.weight")?;
    let action_time_projection = take(tensors, "action_preprocessor.w2.weight")?;
    let action_embedding_projection = take(tensors, "action_preprocessor.w3.weight")?;
    let velocity_projection = take(tensors, "action_preprocessor.action_proj_back.weight")?;
    expect_shape(
        &noisy_action_projection,
        &[width, 2 * dim],
        "noisy action projection",
    )?;
    expect_shape(
        &action_time_projection,
        &[width, 2 * width],
        "action/time projection",
    )?;
    expect_shape(
        &action_embedding_projection,
        &[width, width],
        "action embedding projection",
    )?;
    expect_shape(&velocity_projection, &[dim, width], "velocity projection")?;
    let (noisy_action_projection, dof_projection) = split_columns(&noisy_action_projection, dim)?;
    let (action_projection, time_projection) = split_columns(&action_time_projection, width)?;
    Ok(WallossActionWeights {
        noisy_action_projection: transpose_2d(&noisy_action_projection)?,
        dof_projection: transpose_2d(&dof_projection)?,
        action_projection: transpose_2d(&action_projection)?,
        time_projection: transpose_2d(&time_projection)?,
        action_embedding_projection: transpose_2d(&action_embedding_projection)?,
        velocity_projection: transpose_2d(&velocity_projection)?,
    })
}

fn checkpoint_action_dim(
    tensors: &HashMap<String, Tensor>,
    action_hidden_size: usize,
) -> Result<usize> {
    let output = tensors
        .get("action_preprocessor.action_proj_back.weight")
        .ok_or_else(|| {
            Error::Other(
                "walloss checkpoint: missing action_preprocessor.action_proj_back.weight".into(),
            )
        })?;
    expect_rank(output, 2, "velocity projection")?;
    let output_shape = output.shape().dims();
    let action_dim = output_shape[0];
    if action_dim == 0 || output_shape[1] != action_hidden_size {
        return Err(Error::Other(format!(
            "walloss checkpoint: velocity projection has shape {output_shape:?}, expected [action_dim, {action_hidden_size}]"
        )));
    }
    let input = tensors
        .get("action_preprocessor.w1.weight")
        .ok_or_else(|| {
            Error::Other("walloss checkpoint: missing action_preprocessor.w1.weight".into())
        })?;
    expect_shape(
        input,
        &[action_hidden_size, 2 * action_dim],
        "noisy action projection",
    )?;
    Ok(action_dim)
}

fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| Error::Other(format!("walloss checkpoint: missing {name}")))
}

fn expect_rank(tensor: &Tensor, rank: usize, name: &str) -> Result<()> {
    if tensor.shape().dims().len() != rank {
        return Err(Error::Other(format!(
            "walloss checkpoint: {name} has shape {:?}, expected rank {rank}",
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn expect_shape(tensor: &Tensor, expected: &[usize], name: &str) -> Result<()> {
    if tensor.shape().dims() != expected {
        return Err(Error::Other(format!(
            "walloss checkpoint: {name} has shape {:?}, expected {expected:?}",
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn to_f32(tensor: &Tensor) -> Result<Tensor> {
    if tensor.dtype() == DType::F32 {
        return Ok(tensor.clone());
    }
    if matches!(tensor.dtype(), DType::F16 | DType::BF16) {
        return Tensor::from_f32(tensor.shape().dims().to_vec(), &tensor.to_f32_vec()?);
    }
    Err(Error::Other(format!(
        "walloss checkpoint: unsupported weight dtype {}",
        tensor.dtype()
    )))
}

fn flatten_and_transpose(tensor: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    if tensor.shape().numel() != rows * cols {
        return Err(Error::Other(format!(
            "walloss checkpoint: patch projection has {} elements, expected {}",
            tensor.shape().numel(),
            rows * cols
        )));
    }
    let flat = Tensor::from_f32(vec![rows, cols], &tensor.to_f32_vec()?)?;
    transpose_2d(&flat)
}

fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    expect_rank(tensor, 2, "linear weight")?;
    let rows = tensor.shape().dims()[0];
    let cols = tensor.shape().dims()[1];
    let source = tensor.to_f32_vec()?;
    let mut output = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = source[row * cols + col];
        }
    }
    Tensor::from_f32(vec![cols, rows], &output)
}

fn split_columns(tensor: &Tensor, first_columns: usize) -> Result<(Tensor, Tensor)> {
    expect_rank(tensor, 2, "split weight")?;
    let rows = tensor.shape().dims()[0];
    let columns = tensor.shape().dims()[1];
    if first_columns == 0 || first_columns >= columns {
        return Err(Error::Other(format!(
            "walloss checkpoint: invalid column split {first_columns} for {columns} columns"
        )));
    }
    let second_columns = columns - first_columns;
    let source = tensor.to_f32_vec()?;
    let mut first = vec![0.0; rows * first_columns];
    let mut second = vec![0.0; rows * second_columns];
    for row in 0..rows {
        first[row * first_columns..(row + 1) * first_columns]
            .copy_from_slice(&source[row * columns..row * columns + first_columns]);
        second[row * second_columns..(row + 1) * second_columns]
            .copy_from_slice(&source[row * columns + first_columns..(row + 1) * columns]);
    }
    Ok((
        Tensor::from_f32(vec![rows, first_columns], &first)?,
        Tensor::from_f32(vec![rows, second_columns], &second)?,
    ))
}

fn zeros(shape: &[usize]) -> Result<Tensor> {
    Ok(Tensor::zeros(shape, DType::F32))
}

fn ones(shape: &[usize]) -> Result<Tensor> {
    Tensor::from_f32(shape.to_vec(), &vec![1.0; shape.iter().product()])
}

fn synthetic_tensor(rng: &mut Rng, shape: &[usize], scale: f32) -> Result<Tensor> {
    let count = shape.iter().product();
    Tensor::from_f32(shape.to_vec(), &rng.values(count, scale))
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
        }
    }

    fn values(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|_| {
                self.state = self
                    .state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (self.state >> 40) as f32 / (1u32 << 24) as f32;
                (unit * 2.0 - 1.0) * scale
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action_tensors(action_dim: usize, hidden_size: usize) -> HashMap<String, Tensor> {
        HashMap::from([
            (
                "action_preprocessor.action_proj_back.weight".to_owned(),
                Tensor::zeros((action_dim, hidden_size), DType::BF16),
            ),
            (
                "action_preprocessor.w1.weight".to_owned(),
                Tensor::zeros((hidden_size, 2 * action_dim), DType::BF16),
            ),
        ])
    }

    #[test]
    fn derives_native_action_width_from_checkpoint_projections() {
        assert_eq!(
            checkpoint_action_dim(&action_tensors(7, 1024), 1024).unwrap(),
            7
        );
        assert_eq!(
            checkpoint_action_dim(&action_tensors(26, 1024), 1024).unwrap(),
            26
        );
    }

    #[test]
    fn rejects_inconsistent_action_and_mask_projection_width() {
        let mut tensors = action_tensors(7, 1024);
        tensors.insert(
            "action_preprocessor.w1.weight".to_owned(),
            Tensor::zeros((1024, 15), DType::BF16),
        );
        assert!(checkpoint_action_dim(&tensors, 1024).is_err());
    }

    #[test]
    fn split_columns_preserves_row_stride() {
        let tensor =
            Tensor::from_f32(vec![2, 5], &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]).unwrap();
        let (first, second) = split_columns(&tensor, 2).unwrap();
        assert_eq!(first.shape().dims(), [2, 2]);
        assert_eq!(second.shape().dims(), [2, 3]);
        assert_eq!(first.to_f32_vec().unwrap(), [1., 2., 6., 7.]);
        assert_eq!(second.to_f32_vec().unwrap(), [3., 4., 5., 8., 9., 10.]);
    }
}
