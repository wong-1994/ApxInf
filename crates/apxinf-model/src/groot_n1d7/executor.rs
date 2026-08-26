use std::sync::Arc;

use apxinf_core::{Backend, DType, Error, Result, Tensor};
use half::bf16;

use super::backend::{kernels, linear, relu_roundtrip, row_view, RuntimeBackend};
use super::device_weights::{DeviceActionWeights, DeviceMlp};
use super::{GrootN1d7BackboneOutput, GrootN1d7Config};

pub struct GrootN1d7Executor {
    backend: Arc<dyn Backend>,
    cuda: Arc<RuntimeBackend>,
    config: Arc<GrootN1d7Config>,
    weights: DeviceActionWeights,
    zero_dit_style: Tensor,
}

impl GrootN1d7Executor {
    pub fn new(
        backend: Arc<dyn Backend>,
        config: Arc<GrootN1d7Config>,
        weights: DeviceActionWeights,
    ) -> Result<Self> {
        let cuda = crate::accelerator::cuda::downcast_arc(backend.clone())
            .ok_or_else(|| Error::Other("GR00T N1.7 BF16 runtime requires CUDA".into()))?;
        let zero = Tensor::from_bf16(
            vec![2 * config.input_embedding_dim()],
            &vec![bf16::from_f32(0.0); 2 * config.input_embedding_dim()],
        )?;
        let zero_dit_style = backend.to_device(&zero)?;
        Ok(Self {
            backend,
            cuda,
            config,
            weights,
            zero_dit_style,
        })
    }

    pub fn infer(
        &self,
        backbone: GrootN1d7BackboneOutput,
        state: &Tensor,
        noise: &Tensor,
    ) -> Result<Tensor> {
        let b = &*self.backend;
        let state = upload_bf16(b, state)?.reshape(vec![
            1,
            self.config.max_state_dim * self.config.state_history_length,
        ])?;
        let state_features = mlp_relu(b, &state, &self.weights.state_encoder)?;
        let mut vl = b.layer_norm(
            &backbone.features,
            &self.weights.vl_norm_weight,
            &self.weights.vl_norm_bias,
            1e-5,
        )?;
        vl = self.vl_self_attention(vl)?;
        let mut actions = upload_bf16(b, noise)?
            .reshape(vec![self.config.action_horizon, self.config.max_action_dim])?;
        for step in 0..self.config.num_inference_timesteps {
            let timestep =
                step * self.config.num_timestep_buckets / self.config.num_inference_timesteps;
            let action_features = self.action_encode(&actions, timestep)?;
            let hidden = kernels::elementwise::concat_rows_bf16(
                self.cuda.context(),
                &state_features,
                &action_features,
            )?;
            let output = self.dit(hidden, &vl, &backbone, timestep)?;
            let decoded = mlp_relu(b, &output, &self.weights.action_decoder)?;
            let velocity = row_view(
                &decoded,
                decoded.shape().dims()[0] - self.config.action_horizon,
                self.config.action_horizon,
            )?;
            actions = kernels::elementwise::euler_update_bf16(
                self.cuda.context(),
                &actions,
                &velocity,
                1.0 / self.config.num_inference_timesteps as f32,
            )?;
        }
        Ok(actions)
    }

    fn action_encode(&self, actions: &Tensor, timestep: usize) -> Result<Tensor> {
        let b = &*self.backend;
        let action = linear(b, actions, &self.weights.action_w1)?;
        let tau = action_time_embedding(
            timestep as f32,
            self.config.input_embedding_dim(),
            self.config.action_horizon,
        )?;
        let tau = b.to_device(&tau)?;
        let joined = b.concat_2d(&[&action, &tau])?;
        let hidden = b.silu(&linear(b, &joined, &self.weights.action_w2)?)?;
        let encoded = linear(b, &hidden, &self.weights.action_w3)?;
        let position = row_view(
            &self.weights.position_embedding,
            0,
            self.config.action_horizon,
        )?;
        b.add(&encoded, &position)
    }

    fn vl_self_attention(&self, mut hidden: Tensor) -> Result<Tensor> {
        let b = &*self.backend;
        let rows = hidden.shape().dims()[0];
        let heads = self.config.vl_self_attention_cfg.num_attention_heads;
        let dim = self.config.vl_self_attention_cfg.attention_head_dim;
        let mask = vec![1u8; rows];
        for block in &self.weights.vl_blocks {
            let norm = b.layer_norm(&hidden, &block.norm1_weight, &block.norm1_bias, 1e-5)?;
            let q = linear(b, &norm, &block.q)?.reshape(vec![rows, heads, dim])?;
            let k = linear(b, &norm, &block.k)?.reshape(vec![rows, heads, dim])?;
            let v = linear(b, &norm, &block.v)?.reshape(vec![rows, heads, dim])?;
            let attention = b.masked_cross_sdpa(&q, &k, &v, &mask, heads, dim)?;
            hidden = b.add(&hidden, &linear(b, &attention, &block.output)?)?;
            let norm = b.layer_norm(&hidden, &block.norm3_weight, &block.norm3_bias, 1e-5)?;
            // N1.7's VL refinement stack uses diffusers' `gelu-approximate`,
            // unlike the GEGLU feed-forward in AlternateVLDiT.
            let ff = b.gelu_tanh(&linear(b, &norm, &block.ff_in)?)?;
            hidden = b.add(&hidden, &linear(b, &ff, &block.ff_out)?)?;
        }
        Ok(hidden)
    }

    fn dit(
        &self,
        mut hidden: Tensor,
        vl: &Tensor,
        backbone: &GrootN1d7BackboneOutput,
        timestep: usize,
    ) -> Result<Tensor> {
        let b = &*self.backend;
        let rows = hidden.shape().dims()[0];
        let heads = self.config.diffusion_model_cfg.num_attention_heads;
        let dim = self.config.diffusion_model_cfg.attention_head_dim;
        let time = dit_time_embedding(timestep as f32);
        let time = b.to_device(&time)?;
        let temb = linear(
            b,
            &b.silu(&linear(b, &time, &self.weights.timestep_in)?)?,
            &self.weights.timestep_out,
        )?;
        for (index, block) in self.weights.dit_blocks.iter().enumerate() {
            let style = linear(b, &b.silu(&temb)?, &block.ada_norm)?
                .reshape(vec![2 * self.config.input_embedding_dim()])?;
            let norm = kernels::norm::adaptive_layer_bf16(
                self.cuda.context(),
                &hidden,
                &style,
                1e-5,
                false,
            )?;
            let q = linear(b, &norm, &block.q)?.reshape(vec![rows, heads, dim])?;
            let (keys, values, mask) = if index % 2 == 1 {
                (norm.clone(), norm.clone(), vec![1u8; rows])
            } else {
                let use_text = index % (2 * self.config.attend_text_every_n_blocks) == 0;
                let mask = backbone
                    .image_mask
                    .iter()
                    .zip(&backbone.attention_mask)
                    .map(|(&image, &valid)| valid & if use_text { 1 - image } else { image })
                    .collect();
                (vl.clone(), vl.clone(), mask)
            };
            let key_rows = keys.shape().dims()[0];
            let k = linear(b, &keys, &block.k)?.reshape(vec![key_rows, heads, dim])?;
            let v = linear(b, &values, &block.v)?.reshape(vec![key_rows, heads, dim])?;
            let attention = b.masked_cross_sdpa(&q, &k, &v, &mask, heads, dim)?;
            hidden = b.add(&hidden, &linear(b, &attention, &block.output)?)?;
            let norm = kernels::norm::adaptive_layer_bf16(
                self.cuda.context(),
                &hidden,
                &self.zero_dit_style,
                1e-5,
                false,
            )?;
            let ff = b.gelu_tanh(&linear(b, &norm, &block.ff_in)?)?;
            hidden = b.add(&hidden, &linear(b, &ff, &block.ff_out)?)?;
        }
        let style = linear(b, &b.silu(&temb)?, &self.weights.final_condition)?
            .reshape(vec![2 * self.config.input_embedding_dim()])?;
        let hidden =
            kernels::norm::adaptive_layer_bf16(self.cuda.context(), &hidden, &style, 1e-6, true)?;
        linear(b, &hidden, &self.weights.final_output)
    }
}

fn mlp_relu(b: &dyn Backend, x: &Tensor, mlp: &DeviceMlp) -> Result<Tensor> {
    linear(
        b,
        &relu_roundtrip(b, &linear(b, x, &mlp.layer1)?)?,
        &mlp.layer2,
    )
}
fn upload_bf16(b: &dyn Backend, x: &Tensor) -> Result<Tensor> {
    if x.device() == b.device() && x.dtype() == DType::BF16 {
        return Ok(x.clone());
    }
    let cpu = if x.device() == apxinf_core::Device::Cpu {
        x.clone()
    } else {
        b.to_cpu(x)?
    };
    let values = cpu
        .to_f32_vec()?
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    b.to_device(&Tensor::from_bf16(cpu.shape().dims().to_vec(), &values)?)
}
fn action_time_embedding(timestep: f32, width: usize, rows: usize) -> Result<Tensor> {
    let half = width / 2;
    let mut row = Vec::with_capacity(width);
    for i in 0..half {
        row.push((timestep * (-((i as f32) * 10000.0f32.ln() / half as f32)).exp()).sin());
    }
    for i in 0..half {
        row.push((timestep * (-((i as f32) * 10000.0f32.ln() / half as f32)).exp()).cos());
    }
    let values = (0..rows)
        .flat_map(|_| row.iter().copied())
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    Tensor::from_bf16(vec![rows, width], &values)
}
fn dit_time_embedding(timestep: f32) -> Tensor {
    let half = 128usize;
    let mut values = Vec::with_capacity(256);
    for i in 0..half {
        values.push((timestep * (-(10000.0f32.ln()) * i as f32 / (half - 1) as f32).exp()).cos());
    }
    for i in 0..half {
        values.push((timestep * (-(10000.0f32.ln()) * i as f32 / (half - 1) as f32).exp()).sin());
    }
    Tensor::from_bf16(
        vec![1, 256],
        &values.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
    )
    .unwrap()
}
