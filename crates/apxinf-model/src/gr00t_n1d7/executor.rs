#[cfg(feature = "cuda")]
use super::backend::{kernels, RuntimeBackend};
use super::math::{action_time_embedding, euler_timesteps, timestep_embedding};
use super::weights::{ActionWeights, CategoryLinear, Linear, TransformerBlockWeights};
use super::Gr00tN1d7Config;
use apxinf_core::{Backend, Error, Result, Tensor};

struct Selected {
    state_1: Linear,
    state_2: Linear,
    action_1: Linear,
    action_2: Linear,
    action_3: Linear,
    decoder_1: Linear,
    decoder_2: Linear,
}
pub(crate) struct Gr00tExecutor {
    config: Gr00tN1d7Config,
    weights: ActionWeights,
}

impl Gr00tExecutor {
    pub fn new(config: Gr00tN1d7Config, weights: ActionWeights, b: &dyn Backend) -> Result<Self> {
        Ok(Self {
            config,
            weights: weights.to_device(b)?,
        })
    }
    #[cfg(feature = "cuda")]
    pub fn infer(
        &self,
        cuda: &RuntimeBackend,
        backbone: &Tensor,
        state: &Tensor,
        embodiment: u32,
        attention_mask: &[u8],
        image_mask: &[u8],
        noise: &Tensor,
    ) -> Result<Tensor> {
        let b: &dyn Backend = cuda;
        let selected = self.select(embodiment as usize, b)?;
        let mut vl = b.layer_norm(
            backbone,
            &self.weights.vlln_weight,
            &self.weights.vlln_bias,
            1e-5,
        )?;
        for block in &self.weights.vl {
            vl = self.block(cuda, block, &vl, None, None)?;
        }
        let text = select_rows(b, &vl, attention_mask, image_mask, false)?;
        let image = select_rows(b, &vl, attention_mask, image_mask, true)?;
        let uploaded_state = if state.device() == b.device() {
            None
        } else {
            Some(b.to_device(state)?)
        };
        let state = uploaded_state.as_ref().unwrap_or(state);
        let flat = state.reshape(vec![1, self.config.state_dim * self.config.state_history])?;
        let sf = relu(b, &linear(b, &flat, &selected.state_1)?)?;
        let sf = linear(b, &sf, &selected.state_2)?;
        let positions = (0..self.config.action_horizon as u32).collect::<Vec<_>>();
        let pos = b.embedding(&self.weights.position_embedding, &positions)?;
        let mut actions = if noise.device() == b.device() {
            noise.clone()
        } else {
            b.to_device(noise)?
        };
        for timestep in euler_timesteps(self.config.flow_steps, self.config.timestep_buckets)? {
            let a1 = linear(b, &actions, &selected.action_1)?;
            let tau = upload(
                b,
                vec![self.config.action_horizon, self.config.action_embed_dim],
                &action_time_embedding(
                    timestep,
                    self.config.action_horizon,
                    self.config.action_embed_dim,
                )?,
            )?;
            let joined = b.concat_2d(&[&a1, &tau])?;
            let a2 = b.silu(&linear(b, &joined, &selected.action_2)?)?;
            let af = b.add(&linear(b, &a2, &selected.action_3)?, &pos)?;
            let mut hidden = kernels::elementwise::concat_rows_bf16(cuda.context(), &sf, &af)?;
            let t0 = upload(b, vec![1, 256], &timestep_embedding(timestep, 256)?)?;
            let temb = linear(
                b,
                &b.silu(&linear(b, &t0, &self.weights.timestep_1)?)?,
                &self.weights.timestep_2,
            )?;
            for (index, block) in self.weights.dit.iter().enumerate() {
                let context = if index % 2 == 1 {
                    None
                } else if index % (2 * self.config.attend_text_every_n_blocks) == 0 {
                    Some(&text)
                } else {
                    Some(&image)
                };
                hidden = self.block(cuda, block, &hidden, context, Some(&temb))?;
            }
            let cond = linear(b, &b.silu(&temb)?, &self.weights.output_1)?;
            let out = linear(
                b,
                &adaptive_norm_shift_scale(b, &hidden, &cond, 1e-6)?,
                &self.weights.output_2,
            )?;
            let decoded = linear(
                b,
                &relu(b, &linear(b, &out, &selected.decoder_1)?)?,
                &selected.decoder_2,
            )?;
            let velocity = tail_rows(b, &decoded, self.config.action_horizon)?;
            actions =
                kernels::elementwise::euler_update_bf16(cuda.context(), &actions, &velocity, 0.25)?;
        }
        Ok(actions)
    }
    #[cfg(feature = "cuda")]
    fn block(
        &self,
        cuda: &RuntimeBackend,
        w: &TransformerBlockWeights,
        x: &Tensor,
        context: Option<&Tensor>,
        temb: Option<&Tensor>,
    ) -> Result<Tensor> {
        let b: &dyn Backend = cuda;
        let norm = match (&w.ada, temb) {
            (Some(ada), Some(t)) => adaptive_norm(b, x, &linear(b, &b.silu(t)?, ada)?, 1e-5)?,
            (None, None) => b.layer_norm(
                x,
                w.norm1_weight.as_ref().unwrap(),
                w.norm1_bias.as_ref().unwrap(),
                1e-5,
            )?,
            _ => return Err(Error::Other("GR00T normalization contract mismatch".into())),
        };
        let kv = context.unwrap_or(&norm);
        let rows = norm.shape().dims()[0];
        let kv_rows = kv.shape().dims()[0];
        let (heads, dim) = if temb.is_some() {
            (self.config.dit_heads, self.config.dit_head_dim)
        } else {
            (self.config.vl_heads, self.config.vl_head_dim)
        };
        let q = linear(b, &norm, &w.q)?.reshape(vec![rows, heads, dim])?;
        let k = linear(b, kv, &w.k)?.reshape(vec![kv_rows, heads, dim])?;
        let v = linear(b, kv, &w.v)?.reshape(vec![kv_rows, heads, dim])?;
        let attn =
            kernels::attention::cross_bf16(cuda.context(), &q, &k, &v, rows, kv_rows, heads, dim)?;
        let residual = b.add(x, &linear(b, &attn, &w.out)?)?;
        let ff = b.layer_norm(&residual, &w.norm3_weight, &w.norm3_bias, 1e-5)?;
        let activated = b.gelu_tanh(&linear(b, &ff, &w.ff_in)?)?;
        b.add(&residual, &linear(b, &activated, &w.ff_out)?)
    }
    fn select(&self, id: usize, b: &dyn Backend) -> Result<Selected> {
        Ok(Selected {
            state_1: category(&self.weights.state_1, id, b)?,
            state_2: category(&self.weights.state_2, id, b)?,
            action_1: category(&self.weights.action_1, id, b)?,
            action_2: category(&self.weights.action_2, id, b)?,
            action_3: category(&self.weights.action_3, id, b)?,
            decoder_1: category(&self.weights.decoder_1, id, b)?,
            decoder_2: category(&self.weights.decoder_2, id, b)?,
        })
    }
}
fn linear(b: &dyn Backend, x: &Tensor, w: &Linear) -> Result<Tensor> {
    b.add_bias(&b.matmul(x, &w.weight)?, &w.bias)
}
fn category(bank: &CategoryLinear, id: usize, b: &dyn Backend) -> Result<Linear> {
    let d = bank.weight.shape().dims();
    if d.len() != 3 || id >= d[0] {
        return Err(Error::Other(format!("invalid embodiment {id}")));
    }
    let (i, o) = (d[1], d[2]);
    let w = bank.weight.as_bf16()?[id * i * o..(id + 1) * i * o].to_vec();
    let bias = bank.bias.as_bf16()?[id * o..(id + 1) * o].to_vec();
    Ok(Linear {
        weight: b.to_device(&Tensor::from_bf16(vec![i, o], &w)?)?,
        bias: b.to_device(&Tensor::from_bf16(vec![o], &bias)?)?,
    })
}
fn values(b: &dyn Backend, x: &Tensor) -> Result<Vec<f32>> {
    b.to_cpu(x)?.to_f32_vec()
}
fn upload(b: &dyn Backend, shape: Vec<usize>, data: &[f32]) -> Result<Tensor> {
    let v = data
        .iter()
        .map(|&x| half::bf16::from_f32(x))
        .collect::<Vec<_>>();
    b.to_device(&Tensor::from_bf16(shape, &v)?)
}
fn relu(b: &dyn Backend, x: &Tensor) -> Result<Tensor> {
    let mut v = values(b, x)?;
    for x in &mut v {
        *x = x.max(0.0)
    }
    upload(b, x.shape().dims().to_vec(), &v)
}
fn adaptive_norm(b: &dyn Backend, x: &Tensor, ss: &Tensor, eps: f32) -> Result<Tensor> {
    let d = x.shape().dims();
    let (rows, cols) = (d[0], *d.last().unwrap());
    let input = values(b, x)?;
    let ss = values(b, ss)?;
    if ss.len() != 2 * cols {
        return Err(Error::Other("AdaLayerNorm width mismatch".into()));
    }
    let mut out = vec![0.; input.len()];
    for r in 0..rows {
        let row = &input[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = (var + eps).sqrt().recip();
        for c in 0..cols {
            out[r * cols + c] = (row[c] - mean) * inv * (1. + ss[c]) + ss[cols + c]
        }
    }
    upload(b, d.to_vec(), &out)
}
fn adaptive_norm_shift_scale(b: &dyn Backend, x: &Tensor, ss: &Tensor, eps: f32) -> Result<Tensor> {
    let d = x.shape().dims();
    let (rows, cols) = (d[0], *d.last().unwrap());
    let input = values(b, x)?;
    let ss = values(b, ss)?;
    if ss.len() != 2 * cols {
        return Err(Error::Other("AdaLayerNorm width mismatch".into()));
    }
    let mut out = vec![0.; input.len()];
    for r in 0..rows {
        let row = &input[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = (var + eps).sqrt().recip();
        for c in 0..cols {
            out[r * cols + c] = (row[c] - mean) * inv * (1. + ss[cols + c]) + ss[c]
        }
    }
    upload(b, d.to_vec(), &out)
}
fn select_rows(
    b: &dyn Backend,
    x: &Tensor,
    valid: &[u8],
    image: &[u8],
    want: bool,
) -> Result<Tensor> {
    let d = x.shape().dims();
    if valid.len() != d[0] || image.len() != d[0] {
        return Err(Error::Other("mask length mismatch".into()));
    }
    let cols = d[1];
    let input = values(b, x)?;
    let ids = (0..d[0])
        .filter(|&i| valid[i] != 0 && (image[i] != 0) == want)
        .collect::<Vec<_>>();
    if ids.is_empty() {
        return Err(Error::Other("attention mask selected no rows".into()));
    }
    let mut out = Vec::with_capacity(ids.len() * cols);
    for i in ids {
        out.extend_from_slice(&input[i * cols..(i + 1) * cols])
    }
    upload(b, vec![out.len() / cols, cols], &out)
}
fn tail_rows(b: &dyn Backend, x: &Tensor, rows: usize) -> Result<Tensor> {
    let d = x.shape().dims();
    let cols = d[1];
    let v = values(b, x)?;
    upload(b, vec![rows, cols], &v[(d[0] - rows) * cols..])
}
