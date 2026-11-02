//! Unfused FP32 WallOSS reference forward.
//!
//! BLAS owns projection and attention parallelism. The forward order mirrors
//! the CUDA family (vision tower -> language prefix KV -> fixed-step action
//! denoise) without device packing, fused kernels or BF16 rounding.

use apxinf_core::{Backend, Error, Result, Tensor};
use apxinf_cpu::{CpuAttentionMask, CpuBackend};

use super::geometry::VisionGeometry;
use super::schedule::{sinusoidal_time_embedding, solver_times};
use super::weights::{
    WallossActionWeights, WallossLayerWeights, WallossVisionBlockWeights, WallossWeights,
};
use super::{WallossConfig, WallossImageProcessorConfig};

/// Host inputs for one fixed-shape forward pass.
pub(in crate::cpu_walloss) struct WallossHostInputs {
    /// Canonical patch rows in merged-group order, `[rows, patch_width]`.
    pub patches: Tensor,
    pub geometry: VisionGeometry,
    pub prefix_ids: Vec<u32>,
    pub vision_row_map: Vec<u32>,
    pub prefix_position_ids: Vec<u32>,
    pub action_position_ids: Vec<u32>,
    /// Initial action latent, `[action_horizon, action_dim]`.
    pub initial_state: Tensor,
    /// Degrees-of-freedom mask, `[action_horizon, action_dim]`.
    pub action_mask: Tensor,
}

pub(in crate::cpu_walloss) struct CpuWallossModel {
    config: WallossConfig,
    image_processor: Option<WallossImageProcessorConfig>,
    weights: WallossWeights,
    time_embeddings: Vec<Tensor>,
    step_deltas: Vec<f32>,
}

pub(in crate::cpu_walloss) struct PrefixCache {
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    prefix_tokens: usize,
    cache_tokens: usize,
}

impl CpuWallossModel {
    pub(in crate::cpu_walloss) fn new(
        config: WallossConfig,
        image_processor: Option<WallossImageProcessorConfig>,
        weights: WallossWeights,
    ) -> Result<Self> {
        config.validate()?;
        let steps = config.action.solver_steps;
        let times = solver_times(steps, config.action.scheduler_s, 1.0)?;
        let mut step_deltas = Vec::with_capacity(steps);
        let mut time_embeddings = Vec::with_capacity(steps);
        for step in 0..steps {
            step_deltas.push(times[step + 1] - times[step]);
            let embedding = sinusoidal_time_embedding(times[step], config.action.hidden_size)?;
            let repeated = embedding
                .iter()
                .copied()
                .cycle()
                .take(config.action.action_horizon * embedding.len())
                .collect::<Vec<_>>();
            time_embeddings.push(Tensor::from_f32(
                vec![config.action.action_horizon, config.action.hidden_size],
                &repeated,
            )?);
        }
        Ok(Self {
            config,
            image_processor,
            weights,
            time_embeddings,
            step_deltas,
        })
    }

    pub(in crate::cpu_walloss) fn image_processor(&self) -> Option<&WallossImageProcessorConfig> {
        self.image_processor.as_ref()
    }

    pub(in crate::cpu_walloss) fn patch_width(&self) -> usize {
        3 * self.config.vision.temporal_patch_size
            * self.config.vision.patch_size
            * self.config.vision.patch_size
    }

    pub(in crate::cpu_walloss) fn infer(
        &self,
        inputs: &WallossHostInputs,
    ) -> Result<Tensor> {
        let vision = self.vision_tower(&inputs.patches, &inputs.geometry)?;
        let prefix = self.language_prefix(
            &inputs.prefix_ids,
            &inputs.vision_row_map,
            &vision,
            &inputs.prefix_position_ids,
            self.config.action.action_horizon,
        )?;
        self.denoise(&prefix, inputs)
    }

    fn vision_tower(&self, patches: &Tensor, geometry: &VisionGeometry) -> Result<Tensor> {
        let config = &self.config.vision;
        let rows = geometry.patch_order.len();
        if patches.shape().dims() != [rows, self.patch_width()] {
            return Err(Error::Other(format!(
                "walloss CPU patches must be [{rows}, {}], got {:?}",
                self.patch_width(),
                patches.shape().dims()
            )));
        }
        let ordered = gather_rows(patches, &geometry.patch_order)?;
        let mut hidden = linear(&ordered, &self.weights.vision.patch_projection)?;
        for (index, block) in self.weights.vision.blocks.iter().enumerate() {
            let full = config.full_attention_blocks.contains(&index);
            hidden = self.vision_layer(block, &hidden, geometry, full)?;
        }
        self.vision_merger(&hidden, geometry)
    }

    fn vision_layer(
        &self,
        block: &WallossVisionBlockWeights,
        hidden: &Tensor,
        geometry: &VisionGeometry,
        full: bool,
    ) -> Result<Tensor> {
        let config = &self.config.vision;
        let head_dim = config.hidden_size / config.num_heads;
        let tokens = hidden.shape().dims()[0];
        let normalized = rms_norm(hidden, &block.input_norm, config.rms_norm_eps)?;
        let qkv = linear(&normalized, &block.qkv)?;
        let (q, k, v) = split_qkv(
            &qkv,
            Some(&block.qkv_bias),
            config.num_heads,
            config.num_heads,
            head_dim,
        )?;
        let q = CpuBackend.rope_vision_2d(
            &q,
            config.num_heads,
            head_dim,
            config.rope_theta,
            &geometry.position_ids,
        )?;
        let k = CpuBackend.rope_vision_2d(
            &k,
            config.num_heads,
            head_dim,
            config.rope_theta,
            &geometry.position_ids,
        )?;
        let offsets = if full {
            &geometry.full_offsets
        } else {
            &geometry.window_offsets
        };
        let attention = segmented_attention(&q, &k, &v, offsets)?;
        let attention = attention.reshape(vec![tokens, config.hidden_size])?;
        let projected = linear_bias(
            &attention,
            &block.output,
            Some(&block.output_bias),
        )?;
        let hidden = CpuBackend.add(hidden, &projected)?;
        let normalized = rms_norm(&hidden, &block.post_attention_norm, config.rms_norm_eps)?;
        let gate_up = linear(&normalized, &block.gate_up)?;
        let activated = swiglu(&gate_up, Some(&block.gate_up_bias))?;
        let down = linear_bias(&activated, &block.down, Some(&block.down_bias))?;
        CpuBackend.add(&hidden, &down)
    }

    fn vision_merger(
        &self,
        hidden: &Tensor,
        geometry: &VisionGeometry,
    ) -> Result<Tensor> {
        let config = &self.config.vision;
        let merge_unit = config.spatial_merge_size * config.spatial_merge_size;
        let tokens = hidden.shape().dims()[0];
        if tokens % merge_unit != 0 {
            return Err(Error::Other(format!(
                "walloss CPU vision tokens {tokens} are not divisible by merge unit {merge_unit}"
            )));
        }
        let merged = tokens / merge_unit;
        let normalized = rms_norm(hidden, &self.weights.vision.merger_norm, config.rms_norm_eps)?;
        let normalized = normalized.reshape(vec![merged, merge_unit * config.hidden_size])?;
        let hidden = linear_bias(
            &normalized,
            &self.weights.vision.merger_hidden,
            Some(&self.weights.vision.merger_hidden_bias),
        )?;
        let hidden = CpuBackend.gelu_tanh(&hidden)?;
        let output = linear_bias(
            &hidden,
            &self.weights.vision.merger_output,
            Some(&self.weights.vision.merger_output_bias),
        )?;
        gather_rows(&output, &geometry.reverse_indices)
    }

    fn language_prefix(
        &self,
        prefix_ids: &[u32],
        vision_row_map: &[u32],
        vision: &Tensor,
        position_ids: &[u32],
        action_tokens: usize,
    ) -> Result<PrefixCache> {
        let config = &self.config.text;
        let tokens = prefix_ids.len();
        if tokens == 0
            || vision_row_map.len() != tokens
            || position_ids.len() != tokens * 3
            || vision.shape().dims() != [vision_rows(vision_row_map), config.hidden_size]
        {
            return Err(Error::Other(
                "walloss CPU prefix inputs do not match the language geometry".into(),
            ));
        }
        let hidden = CpuBackend.embedding(&self.weights.token_embedding, prefix_ids)?;
        let vision_data = vision.to_f32_vec()?;
        let mut data = hidden.to_f32_vec()?;
        for (row, &mapped) in vision_row_map.iter().enumerate() {
            if mapped != u32::MAX {
                let source = mapped as usize * config.hidden_size;
                if source + config.hidden_size > vision_data.len() {
                    return Err(Error::Other(
                        "walloss CPU vision row map exceeds the vision output".into(),
                    ));
                }
                data[row * config.hidden_size..(row + 1) * config.hidden_size]
                    .copy_from_slice(&vision_data[source..source + config.hidden_size]);
            }
        }
        let mut hidden = Tensor::from_f32(vec![tokens, config.hidden_size], &data)?;
        let cache_tokens = tokens + action_tokens;
        let q_width = config.num_attention_heads * config.head_dim;
        let mut keys = Vec::with_capacity(self.weights.language_layers.len());
        let mut values = Vec::with_capacity(self.weights.language_layers.len());
        for layer in &self.weights.language_layers {
            let normalized = rms_norm(&hidden, &layer.input_norm, config.rms_norm_eps)?;
            let qkv = linear(&normalized, &layer.qkv)?;
            let (q, k, v) = split_qkv(
                &qkv,
                Some(&layer.qkv_bias),
                config.num_attention_heads,
                config.num_kv_heads,
                config.head_dim,
            )?;
            let q = CpuBackend.rope_mrope(
                &q,
                config.num_attention_heads,
                config.head_dim,
                config.rope_theta,
                config.mrope_section,
                position_ids,
            )?;
            let k = CpuBackend.rope_mrope(
                &k,
                config.num_kv_heads,
                config.head_dim,
                config.rope_theta,
                config.mrope_section,
                position_ids,
            )?;
            let attention =
                CpuBackend.attention_masked(&q, &k, &v, CpuAttentionMask::Causal)?;
            let attention = attention.reshape(vec![tokens, q_width])?;
            let projected = linear(&attention, &layer.output)?;
            hidden = CpuBackend.add(&hidden, &projected)?;
            hidden = self.layer_mlp(&hidden, layer, config.rms_norm_eps)?;
            keys.push(pad_cache(&k, cache_tokens)?);
            values.push(pad_cache(&v, cache_tokens)?);
        }
        Ok(PrefixCache {
            keys,
            values,
            prefix_tokens: tokens,
            cache_tokens,
        })
    }

    fn layer_mlp(
        &self,
        hidden: &Tensor,
        layer: &WallossLayerWeights,
        eps: f32,
    ) -> Result<Tensor> {
        let normalized = rms_norm(hidden, &layer.post_attention_norm, eps)?;
        let gate_up = linear(&normalized, &layer.gate_up)?;
        let activated = swiglu(&gate_up, None)?;
        let down = linear(&activated, &layer.down)?;
        CpuBackend.add(hidden, &down)
    }

    fn denoise(&self, prefix: &PrefixCache, inputs: &WallossHostInputs) -> Result<Tensor> {
        let action_tokens = self.config.action.action_horizon;
        let action_dim = self.config.action.action_dim;
        if inputs.initial_state.shape().dims() != [action_tokens, action_dim]
            || inputs.action_mask.shape().dims() != [action_tokens, action_dim]
            || inputs.action_position_ids.len() != action_tokens * 3
        {
            return Err(Error::Other(
                "walloss CPU action inputs do not match the action geometry".into(),
            ));
        }
        let mut state = inputs.initial_state.clone();
        for (step, time_embedding) in self.time_embeddings.iter().enumerate() {
            let velocity = self.action_stack(
                prefix,
                &state,
                &inputs.action_mask,
                time_embedding,
                &inputs.action_position_ids,
            )?;
            state = euler_update(&state, &velocity, self.step_deltas[step])?;
        }
        Ok(state)
    }

    fn action_stack(
        &self,
        prefix: &PrefixCache,
        noisy_action: &Tensor,
        dof_mask: &Tensor,
        time_embedding: &Tensor,
        position_ids: &[u32],
    ) -> Result<Tensor> {
        let config = &self.config.text;
        let action = &self.config.action;
        let q_width = config.num_attention_heads * config.head_dim;
        let mut hidden = self.action_embedding(noisy_action, dof_mask, time_embedding)?;
        for (index, layer) in self.weights.action_layers.iter().enumerate() {
            let normalized = rms_norm(&hidden, &layer.input_norm, config.rms_norm_eps)?;
            let qkv = linear(&normalized, &layer.qkv)?;
            let (q, k, v) = split_qkv(
                &qkv,
                Some(&layer.qkv_bias),
                config.num_attention_heads,
                config.num_kv_heads,
                config.head_dim,
            )?;
            let q = CpuBackend.rope_mrope(
                &q,
                config.num_attention_heads,
                config.head_dim,
                config.rope_theta,
                config.mrope_section,
                position_ids,
            )?;
            let k = CpuBackend.rope_mrope(
                &k,
                config.num_kv_heads,
                config.head_dim,
                config.rope_theta,
                config.mrope_section,
                position_ids,
            )?;
            let key_cache =
                append_cache(&prefix.keys[index], &k, prefix.prefix_tokens, prefix.cache_tokens)?;
            let value_cache = append_cache(
                &prefix.values[index],
                &v,
                prefix.prefix_tokens,
                prefix.cache_tokens,
            )?;
            let attention =
                CpuBackend.attention_masked(&q, &key_cache, &value_cache, CpuAttentionMask::None)?;
            let attention = attention.reshape(vec![action.action_horizon, q_width])?;
            let projected = linear(&attention, &layer.output)?;
            hidden = CpuBackend.add(&hidden, &projected)?;
            hidden = self.layer_mlp(&hidden, layer, config.rms_norm_eps)?;
        }
        let normalized = rms_norm(&hidden, &self.weights.action_norm, config.rms_norm_eps)?;
        linear(&normalized, &self.weights.action.velocity_projection)
    }

    fn action_embedding(
        &self,
        noisy_action: &Tensor,
        dof_mask: &Tensor,
        time_embedding: &Tensor,
    ) -> Result<Tensor> {
        let weights: &WallossActionWeights = &self.weights.action;
        let action = linear(noisy_action, &weights.noisy_action_projection)?;
        let mask = linear(dof_mask, &weights.dof_projection)?;
        let sum = CpuBackend.add(&action, &mask)?;
        let action_part = linear(&sum, &weights.action_projection)?;
        let time_part = linear(time_embedding, &weights.time_projection)?;
        let fused = CpuBackend.add(&action_part, &time_part)?;
        let activated = CpuBackend.silu(&fused)?;
        linear(&activated, &weights.action_embedding_projection)
    }
}

fn vision_rows(map: &[u32]) -> usize {
    map.iter().filter(|&&row| row != u32::MAX).count()
}

fn rms_norm(input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    CpuBackend.rms_norm(input, weight, eps)
}

fn linear(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    CpuBackend.matmul(input, weight)
}

fn linear_bias(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let output = CpuBackend.matmul(input, weight)?;
    match bias {
        Some(bias) => CpuBackend.add_bias(&output, bias),
        None => Ok(output),
    }
}

fn swiglu(gate_up: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let gate_up = match bias {
        Some(bias) => CpuBackend.add_bias(gate_up, bias)?,
        None => gate_up.clone(),
    };
    let dims = gate_up.shape().dims();
    if dims.len() != 2 || dims[1] % 2 != 0 {
        return Err(Error::Other("walloss CPU SwiGLU width must be even".into()));
    }
    let (rows, columns) = (dims[0], dims[1]);
    let inner = columns / 2;
    let data = gate_up.to_f32_vec()?;
    let mut gate = vec![0.0; rows * inner];
    let mut up = vec![0.0; rows * inner];
    for row in 0..rows {
        gate[row * inner..(row + 1) * inner]
            .copy_from_slice(&data[row * columns..row * columns + inner]);
        up[row * inner..(row + 1) * inner]
            .copy_from_slice(&data[row * columns + inner..(row + 1) * columns]);
    }
    let gate = CpuBackend.silu(&Tensor::from_f32(vec![rows, inner], &gate)?)?;
    let up = Tensor::from_f32(vec![rows, inner], &up)?;
    CpuBackend.mul(&gate, &up)
}

fn split_qkv(
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = qkv.shape().dims();
    let q_width = q_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let fused = q_width + 2 * kv_width;
    if dims.len() != 2 || dims[1] != fused || q_heads % kv_heads != 0 {
        return Err(Error::Other(
            "walloss CPU packed QKV geometry is invalid".into(),
        ));
    }
    let tokens = dims[0];
    let data = qkv.to_f32_vec()?;
    let bias = bias.map(|bias| bias.to_f32_vec()).transpose()?;
    if let Some(bias) = &bias {
        if bias.len() != fused {
            return Err(Error::Other("walloss CPU QKV bias width mismatch".into()));
        }
    }
    let mut query = vec![0.0; tokens * q_width];
    let mut key = vec![0.0; tokens * kv_width];
    let mut value = vec![0.0; tokens * kv_width];
    for token in 0..tokens {
        for column in 0..q_width {
            let offset = bias.as_ref().map_or(0.0, |bias| bias[column]);
            query[token * q_width + column] = data[token * fused + column] + offset;
        }
        for column in 0..kv_width {
            let offset = bias
                .as_ref()
                .map_or(0.0, |bias| bias[q_width + column]);
            key[token * kv_width + column] = data[token * fused + q_width + column] + offset;
            let offset = bias
                .as_ref()
                .map_or(0.0, |bias| bias[q_width + kv_width + column]);
            value[token * kv_width + column] =
                data[token * fused + q_width + kv_width + column] + offset;
        }
    }
    Ok((
        Tensor::from_f32(vec![tokens, q_heads, head_dim], &query)?,
        Tensor::from_f32(vec![tokens, kv_heads, head_dim], &key)?,
        Tensor::from_f32(vec![tokens, kv_heads, head_dim], &value)?,
    ))
}

fn pad_cache(rows: &Tensor, cache_tokens: usize) -> Result<Tensor> {
    let dims = rows.shape().dims();
    let row_width = dims[1] * dims[2];
    let data = rows.to_f32_vec()?;
    let mut output = vec![0.0; cache_tokens * row_width];
    output[..dims[0] * row_width].copy_from_slice(&data);
    Tensor::from_f32(vec![cache_tokens, dims[1], dims[2]], &output)
}

fn append_cache(
    cache: &Tensor,
    rows: &Tensor,
    offset: usize,
    cache_tokens: usize,
) -> Result<Tensor> {
    let dims = cache.shape().dims();
    let row_dims = rows.shape().dims();
    if dims != [cache_tokens, row_dims[1], row_dims[2]]
        || offset + row_dims[0] > cache_tokens
    {
        return Err(Error::Other("walloss CPU KV cache append mismatch".into()));
    }
    let row_width = row_dims[1] * row_dims[2];
    let mut output = cache.to_f32_vec()?;
    output[offset * row_width..(offset + row_dims[0]) * row_width]
        .copy_from_slice(&rows.to_f32_vec()?);
    Tensor::from_f32(dims.to_vec(), &output)
}

fn euler_update(state: &Tensor, velocity: &Tensor, dt: f32) -> Result<Tensor> {
    if state.shape() != velocity.shape() {
        return Err(Error::Other(
            "walloss CPU Euler update shape mismatch".into(),
        ));
    }
    let step = CpuBackend.scale(velocity, dt)?;
    CpuBackend.add(state, &step)
}

fn gather_rows(input: &Tensor, indices: &[u32]) -> Result<Tensor> {
    let dims = input.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other("walloss CPU gather expects rank 2".into()));
    }
    let width = dims[1];
    let data = input.to_f32_vec()?;
    let mut output = Vec::with_capacity(indices.len() * width);
    for &index in indices {
        let start = index as usize * width;
        let end = start + width;
        if end > data.len() {
            return Err(Error::Other("walloss CPU gather index out of range".into()));
        }
        output.extend_from_slice(&data[start..end]);
    }
    Tensor::from_f32(vec![indices.len(), width], &output)
}

fn segmented_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    offsets: &[u32],
) -> Result<Tensor> {
    let dims = query.shape().dims();
    let (tokens, heads, head_dim) = (dims[0], dims[1], dims[2]);
    let mut output = vec![0.0; tokens * heads * head_dim];
    let width = heads * head_dim;
    for bounds in offsets.windows(2) {
        let (start, end) = (bounds[0] as usize, bounds[1] as usize);
        if start == end {
            continue;
        }
        if end > tokens {
            return Err(Error::Other(
                "walloss CPU attention segment exceeds the token count".into(),
            ));
        }
        let segment = end - start;
        let query_segment = slice_rows(query, start, end)?;
        let key_segment = slice_rows(key, start, end)?;
        let value_segment = slice_rows(value, start, end)?;
        let attended = CpuBackend.attention(&query_segment, &key_segment, &value_segment)?;
        let attended = attended.to_f32_vec()?;
        output[start * width..end * width].copy_from_slice(&attended[..segment * width]);
    }
    Tensor::from_f32(vec![tokens, heads, head_dim], &output)
}

fn slice_rows(input: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    let dims = input.shape().dims();
    let width = dims[1] * dims[2];
    let data = input.to_f32_vec()?;
    if end * width > data.len() {
        return Err(Error::Other("walloss CPU row slice out of range".into()));
    }
    Tensor::from_f32(
        vec![end - start, dims[1], dims[2]],
        &data[start * width..end * width],
    )
}
