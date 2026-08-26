//! Native-BF16 action entry, projection, and solver update.

use apxinf_core::{Error, Result, Tensor};

use super::backend::{kernels, Context};
use super::WallossActionWeights;

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
