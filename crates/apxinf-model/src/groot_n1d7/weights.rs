use std::collections::HashMap;

use apxinf_core::{DType, Error, Result, Tensor};

use super::GrootN1d7Config;

pub struct LinearWeights {
    /// Row-major `[input, output]` projection weight.
    pub weight: Tensor,
    pub bias: Tensor,
}

pub struct CategoryLinearWeights {
    /// Reference-native `[category, input, output]` weight.
    pub weight: Tensor,
    pub bias: Tensor,
}

pub struct CategoryMlpWeights {
    pub layer1: CategoryLinearWeights,
    pub layer2: CategoryLinearWeights,
}

pub struct GrootN1d7DiTBlockWeights {
    pub q: LinearWeights,
    pub k: LinearWeights,
    pub v: LinearWeights,
    pub output: LinearWeights,
    /// AdaLayerNorm conditioning projection; ordinary self-attention blocks
    /// use the same checkpoint slot in N1.7.
    pub ada_norm: LinearWeights,
    /// GEGLU input `[1536, 6144]`, split evenly after projection.
    pub ff_in: LinearWeights,
    pub ff_out: LinearWeights,
}

pub struct GrootN1d7SelfAttentionBlockWeights {
    pub norm1_weight: Tensor,
    pub norm1_bias: Tensor,
    pub q: LinearWeights,
    pub k: LinearWeights,
    pub v: LinearWeights,
    pub output: LinearWeights,
    pub norm3_weight: Tensor,
    pub norm3_bias: Tensor,
    pub ff_in: LinearWeights,
    pub ff_out: LinearWeights,
}

pub struct GrootN1d7ActionWeights {
    pub state_encoder: CategoryMlpWeights,
    pub action_w1: CategoryLinearWeights,
    pub action_w2: CategoryLinearWeights,
    pub action_w3: CategoryLinearWeights,
    pub action_decoder: CategoryMlpWeights,
    pub position_embedding: Tensor,
    pub vl_norm_weight: Tensor,
    pub vl_norm_bias: Tensor,
    pub vl_blocks: Vec<GrootN1d7SelfAttentionBlockWeights>,
    pub timestep_in: LinearWeights,
    pub timestep_out: LinearWeights,
    pub dit_blocks: Vec<GrootN1d7DiTBlockWeights>,
    pub final_condition: LinearWeights,
    pub final_output: LinearWeights,
}

impl GrootN1d7ActionWeights {
    /// Load the native BF16 action head from a complete Hugging Face snapshot.
    /// A refs-only cache directory fails in the loader instead of being
    /// mistaken for a usable checkpoint.
    pub fn from_safetensors(cfg: &GrootN1d7Config, path: &std::path::Path) -> Result<Self> {
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(path).map_err(|error| {
            Error::Other(format!(
                "load GR00T N1.7 checkpoint {}: {error}",
                path.display()
            ))
        })?;
        Self::from_map(cfg, tensors)
    }

    pub fn from_map(cfg: &GrootN1d7Config, mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        let root = "action_head";
        let state_encoder = take_category_mlp(&mut tensors, &format!("{root}.state_encoder"))?;
        let action_w1 = take_category_linear(&mut tensors, &format!("{root}.action_encoder.W1"))?;
        let action_w2 = take_category_linear(&mut tensors, &format!("{root}.action_encoder.W2"))?;
        let action_w3 = take_category_linear(&mut tensors, &format!("{root}.action_encoder.W3"))?;
        let action_decoder = take_category_mlp(&mut tensors, &format!("{root}.action_decoder"))?;
        let position_embedding = take(&mut tensors, &format!("{root}.position_embedding.weight"))?;
        let vl_norm_weight = take(&mut tensors, &format!("{root}.vlln.weight"))?;
        let vl_norm_bias = take(&mut tensors, &format!("{root}.vlln.bias"))?;

        let mut vl_blocks = Vec::with_capacity(cfg.vl_self_attention_cfg.num_layers);
        for index in 0..cfg.vl_self_attention_cfg.num_layers {
            let p = format!("{root}.vl_self_attention.transformer_blocks.{index}");
            vl_blocks.push(GrootN1d7SelfAttentionBlockWeights {
                norm1_weight: take(&mut tensors, &format!("{p}.norm1.weight"))?,
                norm1_bias: take(&mut tensors, &format!("{p}.norm1.bias"))?,
                q: take_linear(&mut tensors, &format!("{p}.attn1.to_q"))?,
                k: take_linear(&mut tensors, &format!("{p}.attn1.to_k"))?,
                v: take_linear(&mut tensors, &format!("{p}.attn1.to_v"))?,
                output: take_linear(&mut tensors, &format!("{p}.attn1.to_out.0"))?,
                norm3_weight: take(&mut tensors, &format!("{p}.norm3.weight"))?,
                norm3_bias: take(&mut tensors, &format!("{p}.norm3.bias"))?,
                ff_in: take_linear(&mut tensors, &format!("{p}.ff.net.0.proj"))?,
                ff_out: take_linear(&mut tensors, &format!("{p}.ff.net.2"))?,
            });
        }

        let model = format!("{root}.model");
        let timestep_in = take_linear(
            &mut tensors,
            &format!("{model}.timestep_encoder.timestep_embedder.linear_1"),
        )?;
        let timestep_out = take_linear(
            &mut tensors,
            &format!("{model}.timestep_encoder.timestep_embedder.linear_2"),
        )?;
        let mut dit_blocks = Vec::with_capacity(cfg.diffusion_model_cfg.num_layers);
        for index in 0..cfg.diffusion_model_cfg.num_layers {
            let p = format!("{model}.transformer_blocks.{index}");
            dit_blocks.push(GrootN1d7DiTBlockWeights {
                q: take_linear(&mut tensors, &format!("{p}.attn1.to_q"))?,
                k: take_linear(&mut tensors, &format!("{p}.attn1.to_k"))?,
                v: take_linear(&mut tensors, &format!("{p}.attn1.to_v"))?,
                output: take_linear(&mut tensors, &format!("{p}.attn1.to_out.0"))?,
                ada_norm: take_linear(&mut tensors, &format!("{p}.norm1.linear"))?,
                ff_in: take_linear(&mut tensors, &format!("{p}.ff.net.0.proj"))?,
                ff_out: take_linear(&mut tensors, &format!("{p}.ff.net.2"))?,
            });
        }
        let final_condition = take_linear(&mut tensors, &format!("{model}.proj_out_1"))?;
        let final_output = take_linear(&mut tensors, &format!("{model}.proj_out_2"))?;

        let weights = Self {
            state_encoder,
            action_w1,
            action_w2,
            action_w3,
            action_decoder,
            position_embedding,
            vl_norm_weight,
            vl_norm_bias,
            vl_blocks,
            timestep_in,
            timestep_out,
            dit_blocks,
            final_condition,
            final_output,
        };
        weights.validate(cfg)?;
        Ok(weights)
    }

    fn validate(&self, cfg: &GrootN1d7Config) -> Result<()> {
        let cats = cfg.max_num_embodiments;
        let action = cfg.max_action_dim;
        let state = cfg.max_state_dim * cfg.state_history_length;
        let hidden = cfg.hidden_size;
        let width = cfg.input_embedding_dim();
        let backbone = cfg.backbone_embedding_dim;
        expect_category(
            "state_encoder.layer1",
            &self.state_encoder.layer1,
            cats,
            state,
            hidden,
        )?;
        expect_category(
            "state_encoder.layer2",
            &self.state_encoder.layer2,
            cats,
            hidden,
            width,
        )?;
        expect_category("action_encoder.W1", &self.action_w1, cats, action, width)?;
        expect_category("action_encoder.W2", &self.action_w2, cats, 2 * width, width)?;
        expect_category("action_encoder.W3", &self.action_w3, cats, width, width)?;
        expect_category(
            "action_decoder.layer1",
            &self.action_decoder.layer1,
            cats,
            hidden,
            hidden,
        )?;
        expect_category(
            "action_decoder.layer2",
            &self.action_decoder.layer2,
            cats,
            hidden,
            action,
        )?;
        expect_shape(
            "position_embedding",
            &self.position_embedding,
            &[cfg.max_seq_len, width],
        )?;
        expect_shape("vlln.weight", &self.vl_norm_weight, &[backbone])?;
        expect_shape("vlln.bias", &self.vl_norm_bias, &[backbone])?;
        if self.vl_blocks.len() != cfg.vl_self_attention_cfg.num_layers
            || self.dit_blocks.len() != cfg.diffusion_model_cfg.num_layers
        {
            return Err(Error::Other(
                "GR00T N1.7 action-head layer count mismatch".into(),
            ));
        }
        Ok(())
    }
}

fn take_category_mlp(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
) -> Result<CategoryMlpWeights> {
    Ok(CategoryMlpWeights {
        layer1: take_category_linear(tensors, &format!("{prefix}.layer1"))?,
        layer2: take_category_linear(tensors, &format!("{prefix}.layer2"))?,
    })
}

fn take_category_linear(
    tensors: &mut HashMap<String, Tensor>,
    prefix: &str,
) -> Result<CategoryLinearWeights> {
    Ok(CategoryLinearWeights {
        weight: take(tensors, &format!("{prefix}.W"))?,
        bias: take(tensors, &format!("{prefix}.b"))?,
    })
}

fn take_linear(tensors: &mut HashMap<String, Tensor>, prefix: &str) -> Result<LinearWeights> {
    Ok(LinearWeights {
        weight: transpose_2d(&take(tensors, &format!("{prefix}.weight"))?)?,
        bias: take(tensors, &format!("{prefix}.bias"))?,
    })
}

fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    tensors
        .remove(name)
        .ok_or_else(|| Error::Other(format!("missing GR00T N1.7 tensor {name}")))
}

fn expect_category(
    name: &str,
    linear: &CategoryLinearWeights,
    categories: usize,
    input: usize,
    output: usize,
) -> Result<()> {
    expect_shape(
        &format!("{name}.W"),
        &linear.weight,
        &[categories, input, output],
    )?;
    expect_shape(&format!("{name}.b"), &linear.bias, &[categories, output])
}

fn expect_shape(name: &str, tensor: &Tensor, expected: &[usize]) -> Result<()> {
    if tensor.shape().dims() != expected {
        return Err(Error::Other(format!(
            "GR00T N1.7 {name} shape {:?}, expected {expected:?}",
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "GR00T N1.7 linear weight must be 2D, got {dims:?}"
        )));
    }
    let (rows, cols) = (dims[0], dims[1]);
    match tensor.dtype() {
        DType::F32 => {
            let data = tensor.as_f32()?;
            let mut out = vec![0.0f32; rows * cols];
            for row in 0..rows {
                for col in 0..cols {
                    out[col * rows + row] = data[row * cols + col];
                }
            }
            Tensor::from_f32(vec![cols, rows], &out)
        }
        DType::BF16 => {
            let data = tensor.as_bf16()?;
            let mut out = vec![half::bf16::from_f32(0.0); rows * cols];
            for row in 0..rows {
                for col in 0..cols {
                    out[col * rows + row] = data[row * cols + col];
                }
            }
            Tensor::from_bf16(vec![cols, rows], &out)
        }
        dtype => Err(Error::Other(format!(
            "GR00T N1.7 weight transpose does not support {dtype}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transposes_hugging_face_linear_weight_physically() {
        let input = Tensor::from_f32(vec![2, 3], &[1., 2., 3., 4., 5., 6.]).unwrap();
        let output = transpose_2d(&input).unwrap();
        assert_eq!(output.shape().dims(), &[3, 2]);
        assert_eq!(output.as_f32().unwrap(), &[1., 4., 2., 5., 3., 6.]);
    }
}
