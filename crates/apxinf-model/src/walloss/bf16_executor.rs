//! Native-BF16 action entry, projection, and solver update.

use apxinf_core::{Error, Result, Tensor};

use super::backend::{kernels, Context, DeviceBuffer};
use super::WallossActionWeights;
use super::{WallossLayerWeights, WallossTextConfig};

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
    let attention = kernels::attention::gqa_bf16(
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
