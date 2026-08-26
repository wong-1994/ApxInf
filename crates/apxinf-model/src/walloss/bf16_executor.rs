//! Native-BF16 action entry, projection, and solver update.

use apxinf_core::{Error, Result, Tensor};

use super::backend::{kernels, Context, DeviceBuffer};
use super::WallossActionWeights;
use super::{
    DeviceVisionGeometry, WallossLayerWeights, WallossTextConfig, WallossVisionBlockWeights,
    WallossVisionConfig, WallossVisionWeights,
};

pub struct LanguageLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

pub struct ActionLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

pub struct PrefixCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
}

pub fn language_prefix(
    context: &Context,
    config: &WallossTextConfig,
    weights: &[WallossLayerWeights],
    token_embedding: &Tensor,
    token_ids: &DeviceBuffer,
    vision_tokens: &Tensor,
    vision_row_map: &DeviceBuffer,
    position_ids: &DeviceBuffer,
    tokens: usize,
) -> Result<PrefixCache> {
    let embedded = kernels::embedding::lookup_bf16(context, token_embedding, token_ids, tokens)?;
    let mut hidden = kernels::elementwise::replace_rows_bf16(
        context,
        &embedded,
        vision_tokens,
        vision_row_map,
    )?;
    let mut keys = Vec::with_capacity(weights.len());
    let mut values = Vec::with_capacity(weights.len());
    for layer in weights {
        let output = language_layer(context, config, layer, &hidden, position_ids)?;
        hidden = output.hidden;
        keys.push(output.key);
        values.push(output.value);
    }
    Ok(PrefixCache { keys, values })
}

pub fn language_layer(
    context: &Context,
    config: &WallossTextConfig,
    weights: &WallossLayerWeights,
    input: &Tensor,
    position_ids: &DeviceBuffer,
) -> Result<LanguageLayerOutput> {
    let tokens = input.shape().dims()[0];
    let normalized = kernels::norm::rms_bf16(
        context,
        input,
        &weights.input_norm,
        config.rms_norm_eps,
    )?;
    let qkv = kernels::gemm::bf16(context, &normalized, &weights.qkv)?;
    let qkv = kernels::attention::split_gqa_qkv_bias_bf16(
        context,
        &qkv,
        Some(&weights.qkv_bias),
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
    )?;
    let q = kernels::rope::apply_mrope(
        context,
        &qkv.q,
        config.num_attention_heads,
        config.head_dim,
        config.rope_theta,
        config.mrope_section,
        position_ids,
    )?;
    let k = kernels::rope::apply_mrope(
        context,
        &qkv.k,
        config.num_kv_heads,
        config.head_dim,
        config.rope_theta,
        config.mrope_section,
        position_ids,
    )?;
    let attention = kernels::attention::gqa_bf16(context, &q, &k, &qkv.v, tokens)?
        .reshape(vec![tokens, config.num_attention_heads * config.head_dim])?;
    let projected = kernels::gemm::bf16(context, &attention, &weights.output)?;
    let fused = kernels::fused::bias_residual_rms_bf16(
        context,
        &projected,
        None,
        input,
        &weights.post_attention_norm,
        config.rms_norm_eps,
    )?;
    let gate_up = kernels::gemm::bf16(context, &fused.normalized, &weights.gate_up)?;
    let activated = kernels::activation::geglu_bf16(context, &gate_up)?;
    let down = kernels::gemm::bf16(context, &activated, &weights.down)?;
    let hidden = kernels::fused::bias_residual_bf16(context, &down, None, &fused.hidden)?;
    Ok(LanguageLayerOutput {
        hidden,
        key: k,
        value: qkv.v,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn action_layer(
    context: &Context,
    config: &WallossTextConfig,
    weights: &WallossLayerWeights,
    input: &Tensor,
    prefix_key: &Tensor,
    prefix_value: &Tensor,
    action_position_ids: &DeviceBuffer,
) -> Result<ActionLayerOutput> {
    let action_tokens = input.shape().dims()[0];
    let prefix_tokens = prefix_key.shape().dims()[0];
    let kv_width = config.num_kv_heads * config.head_dim;
    let normalized = kernels::norm::rms_bf16(
        context,
        input,
        &weights.input_norm,
        config.rms_norm_eps,
    )?;
    let qkv = kernels::gemm::bf16(context, &normalized, &weights.qkv)?;
    let qkv = kernels::attention::split_gqa_qkv_bias_bf16(
        context,
        &qkv,
        Some(&weights.qkv_bias),
        config.num_attention_heads,
        config.num_kv_heads,
        config.head_dim,
    )?;
    let q = kernels::rope::apply_mrope(
        context,
        &qkv.q,
        config.num_attention_heads,
        config.head_dim,
        config.rope_theta,
        config.mrope_section,
        action_position_ids,
    )?;
    let action_key = kernels::rope::apply_mrope(
        context,
        &qkv.k,
        config.num_kv_heads,
        config.head_dim,
        config.rope_theta,
        config.mrope_section,
        action_position_ids,
    )?;
    let key = kernels::elementwise::concat_rows_bf16(
        context,
        &prefix_key.reshape(vec![prefix_tokens, kv_width])?,
        &action_key.reshape(vec![action_tokens, kv_width])?,
    )?
    .reshape(vec![
        prefix_tokens + action_tokens,
        config.num_kv_heads,
        config.head_dim,
    ])?;
    let value = kernels::elementwise::concat_rows_bf16(
        context,
        &prefix_value.reshape(vec![prefix_tokens, kv_width])?,
        &qkv.v.reshape(vec![action_tokens, kv_width])?,
    )?
    .reshape(vec![
        prefix_tokens + action_tokens,
        config.num_kv_heads,
        config.head_dim,
    ])?;
    let attention = kernels::attention::causal_gqa_bf16(
        context,
        &q,
        &key,
        &value,
        prefix_tokens + action_tokens,
    )?
    .reshape(vec![
        action_tokens,
        config.num_attention_heads * config.head_dim,
    ])?;
    let projected = kernels::gemm::bf16(context, &attention, &weights.output)?;
    let fused = kernels::fused::bias_residual_rms_bf16(
        context,
        &projected,
        None,
        input,
        &weights.post_attention_norm,
        config.rms_norm_eps,
    )?;
    let gate_up = kernels::gemm::bf16(context, &fused.normalized, &weights.gate_up)?;
    let activated = kernels::activation::geglu_bf16(context, &gate_up)?;
    let down = kernels::gemm::bf16(context, &activated, &weights.down)?;
    let hidden = kernels::fused::bias_residual_bf16(context, &down, None, &fused.hidden)?;
    Ok(ActionLayerOutput {
        hidden,
        key: action_key,
        value: qkv.v,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn action_stack(
    context: &Context,
    config: &WallossTextConfig,
    transformer_weights: &[WallossLayerWeights],
    action_weights: &WallossActionWeights,
    final_norm: &Tensor,
    prefix: &PrefixCache,
    noisy_action: &Tensor,
    dof_mask: &Tensor,
    time_embedding: &Tensor,
    action_position_ids: &DeviceBuffer,
) -> Result<Tensor> {
    if transformer_weights.len() != prefix.keys.len()
        || transformer_weights.len() != prefix.values.len()
    {
        return Err(Error::Other(
            "walloss action stack and prefix cache depth differ".into(),
        ));
    }
    let mut hidden = action_embedding(
        context,
        action_weights,
        noisy_action,
        dof_mask,
        time_embedding,
    )?;
    for (index, layer) in transformer_weights.iter().enumerate() {
        hidden = action_layer(
            context,
            config,
            layer,
            &hidden,
            &prefix.keys[index],
            &prefix.values[index],
            action_position_ids,
        )?
        .hidden;
    }
    let hidden = kernels::norm::rms_bf16(
        context,
        &hidden,
        final_norm,
        config.rms_norm_eps,
    )?;
    velocity(context, action_weights, &hidden)
}

#[allow(clippy::too_many_arguments)]
pub fn vision_layer(
    context: &Context,
    config: &WallossVisionConfig,
    weights: &WallossVisionBlockWeights,
    input: &Tensor,
    position_ids: &DeviceBuffer,
    attention_offsets: &DeviceBuffer,
    segments: usize,
    max_segment_tokens: usize,
) -> Result<Tensor> {
    let tokens = input.shape().dims()[0];
    let head_dim = config.hidden_size / config.num_heads;
    let normalized = kernels::norm::rms_bf16(
        context,
        input,
        &weights.input_norm,
        config.rms_norm_eps,
    )?;
    let qkv = kernels::gemm::bf16(context, &normalized, &weights.qkv)?;
    let qkv = kernels::attention::split_qkv_bias_bf16(
        context,
        &qkv,
        Some(&weights.qkv_bias),
        config.num_heads,
        head_dim,
    )?;
    let q = kernels::rope::apply_vision_2d(
        context,
        &qkv.q,
        config.num_heads,
        head_dim,
        config.rope_theta,
        position_ids,
    )?;
    let k = kernels::rope::apply_vision_2d(
        context,
        &qkv.k,
        config.num_heads,
        head_dim,
        config.rope_theta,
        position_ids,
    )?;
    let attention = kernels::attention::segmented_mha_bf16(
        context,
        &q,
        &k,
        &qkv.v,
        attention_offsets,
        segments,
        max_segment_tokens,
    )?
    .reshape(vec![tokens, config.hidden_size])?;
    let projected = kernels::gemm::bf16(context, &attention, &weights.output)?;
    let fused = kernels::fused::bias_residual_rms_bf16(
        context,
        &projected,
        Some(&weights.output_bias),
        input,
        &weights.post_attention_norm,
        config.rms_norm_eps,
    )?;
    let gate_up = kernels::gemm::bf16(context, &fused.normalized, &weights.gate_up)?;
    let gate_up = kernels::elementwise::bias_bf16(context, &gate_up, Some(&weights.gate_up_bias))?;
    let activated = kernels::activation::geglu_bf16(context, &gate_up)?;
    let down = kernels::gemm::bf16(context, &activated, &weights.down)?;
    kernels::fused::bias_residual_bf16(
        context,
        &down,
        Some(&weights.down_bias),
        &fused.hidden,
    )
}

pub fn vision_tower(
    context: &Context,
    config: &WallossVisionConfig,
    weights: &WallossVisionWeights,
    geometry: &DeviceVisionGeometry,
    patches: &Tensor,
) -> Result<Tensor> {
    let ordered_patches = kernels::elementwise::gather_rows_bf16(
        context,
        patches,
        &geometry.patch_order,
        geometry.patch_order.len() / std::mem::size_of::<u32>(),
    )?;
    let mut hidden = kernels::gemm::bf16(context, &ordered_patches, &weights.patch_projection)?;
    for (layer_index, block) in weights.blocks.iter().enumerate() {
        let full_attention = config.full_attention_blocks.contains(&layer_index);
        let (offsets, segments, max_tokens) = if full_attention {
            (
                &geometry.full_offsets,
                geometry.full_segments,
                geometry.max_full_tokens,
            )
        } else {
            (
                &geometry.window_offsets,
                geometry.window_segments,
                geometry.max_window_tokens,
            )
        };
        hidden = vision_layer(
            context,
            config,
            block,
            &hidden,
            &geometry.position_ids,
            offsets,
            segments,
            max_tokens,
        )?;
    }
    vision_merger(
        context,
        config,
        weights,
        &hidden,
        &geometry.reverse_indices,
    )
}

pub fn vision_merger(
    context: &Context,
    config: &WallossVisionConfig,
    weights: &WallossVisionWeights,
    input: &Tensor,
    reverse_indices: &DeviceBuffer,
) -> Result<Tensor> {
    let merge_unit = config.spatial_merge_size * config.spatial_merge_size;
    let tokens = input.shape().dims()[0];
    if tokens % merge_unit != 0 {
        return Err(Error::Other(format!(
            "walloss vision tokens {tokens} are not divisible by merge unit {merge_unit}"
        )));
    }
    let merged_tokens = tokens / merge_unit;
    let normalized = kernels::norm::rms_bf16(
        context,
        input,
        &weights.merger_norm,
        config.rms_norm_eps,
    )?
    .reshape(vec![merged_tokens, merge_unit * config.hidden_size])?;
    let hidden = kernels::gemm::bf16(context, &normalized, &weights.merger_hidden)?;
    let hidden = kernels::activation::bias_activation(
        context,
        &hidden,
        Some(&weights.merger_hidden_bias),
        1,
    )?;
    let output = kernels::gemm::bf16(context, &hidden, &weights.merger_output)?;
    let output = kernels::elementwise::bias_bf16(
        context,
        &output,
        Some(&weights.merger_output_bias),
    )?;
    kernels::elementwise::gather_rows_bf16(
        context,
        &output,
        reverse_indices,
        merged_tokens,
    )
}

pub fn action_embedding(
    context: &Context,
    weights: &WallossActionWeights,
    noisy_action: &Tensor,
    dof_mask: &Tensor,
    time_embedding: &Tensor,
) -> Result<Tensor> {
    if noisy_action.shape() != dof_mask.shape() {
        return Err(Error::Other(format!(
            "walloss action and degree-of-freedom mask shapes differ: {:?} vs {:?}",
            noisy_action.shape().dims(),
            dof_mask.shape().dims()
        )));
    }
    let action = kernels::gemm::bf16(context, noisy_action, &weights.noisy_action_projection)?;
    let mask = kernels::gemm::bf16(context, dof_mask, &weights.dof_projection)?;
    let action = kernels::elementwise::add(context, &action, &mask)?;

    let action_part = kernels::gemm::bf16(context, &action, &weights.action_projection)?;
    let time_part = kernels::gemm::bf16(context, time_embedding, &weights.time_projection)?;
    let fused = kernels::elementwise::add(context, &action_part, &time_part)?;
    let activated = kernels::activation::silu(context, &fused)?;
    kernels::gemm::bf16(context, &activated, &weights.action_embedding_projection)
}

pub fn velocity(
    context: &Context,
    weights: &WallossActionWeights,
    action_hidden: &Tensor,
) -> Result<Tensor> {
    kernels::gemm::bf16(context, action_hidden, &weights.velocity_projection)
}

pub fn solver_update(
    context: &Context,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    kernels::elementwise::euler_update_bf16(context, state, velocity, dt)
}
