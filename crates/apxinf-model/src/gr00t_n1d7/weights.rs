use std::collections::HashMap;

use apxinf_core::{Backend, DType, Error, Result, Tensor};

use super::Gr00tN1d7Config;

pub(crate) struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

pub(crate) struct TransformerBlockWeights {
    pub ada: Option<Linear>,
    pub norm1_weight: Option<Tensor>,
    pub norm1_bias: Option<Tensor>,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub out: Linear,
    pub norm3_weight: Tensor,
    pub norm3_bias: Tensor,
    pub ff_in: Linear,
    pub ff_out: Linear,
}

pub(crate) struct CategoryLinear {
    pub weight: Tensor,
    pub bias: Tensor,
}

pub(crate) struct ActionWeights {
    pub timestep_1: Linear,
    pub timestep_2: Linear,
    pub dit: Vec<TransformerBlockWeights>,
    pub output_1: Linear,
    pub output_2: Linear,
    pub position_embedding: Tensor,
    pub vlln_weight: Tensor,
    pub vlln_bias: Tensor,
    pub vl: Vec<TransformerBlockWeights>,
    pub state_1: CategoryLinear,
    pub state_2: CategoryLinear,
    pub action_1: CategoryLinear,
    pub action_2: CategoryLinear,
    pub action_3: CategoryLinear,
    pub decoder_1: CategoryLinear,
    pub decoder_2: CategoryLinear,
}

impl ActionWeights {
    pub fn from_map(cfg: &Gr00tN1d7Config, tensors: &mut HashMap<String, Tensor>) -> Result<Self> {
        let mut dit = Vec::with_capacity(cfg.dit_layers);
        for index in 0..cfg.dit_layers {
            dit.push(take_block(
                tensors,
                &format!("action_head.model.transformer_blocks.{index}"),
                true,
            )?);
        }
        let mut vl = Vec::with_capacity(cfg.vl_layers);
        for index in 0..cfg.vl_layers {
            vl.push(take_block(
                tensors,
                &format!("action_head.vl_self_attention.transformer_blocks.{index}"),
                false,
            )?);
        }
        Ok(Self {
            timestep_1: take_linear(
                tensors,
                "action_head.model.timestep_encoder.timestep_embedder.linear_1",
            )?,
            timestep_2: take_linear(
                tensors,
                "action_head.model.timestep_encoder.timestep_embedder.linear_2",
            )?,
            dit,
            output_1: take_linear(tensors, "action_head.model.proj_out_1")?,
            output_2: take_linear(tensors, "action_head.model.proj_out_2")?,
            position_embedding: take(tensors, "action_head.position_embedding.weight")?,
            vlln_weight: take(tensors, "action_head.vlln.weight")?,
            vlln_bias: take(tensors, "action_head.vlln.bias")?,
            vl,
            state_1: take_category(tensors, "action_head.state_encoder.layer1")?,
            state_2: take_category(tensors, "action_head.state_encoder.layer2")?,
            action_1: take_category(tensors, "action_head.action_encoder.W1")?,
            action_2: take_category(tensors, "action_head.action_encoder.W2")?,
            action_3: take_category(tensors, "action_head.action_encoder.W3")?,
            decoder_1: take_category(tensors, "action_head.action_decoder.layer1")?,
            decoder_2: take_category(tensors, "action_head.action_decoder.layer2")?,
        })
    }

    pub fn to_device(self, backend: &dyn Backend) -> Result<Self> {
        fn linear(backend: &dyn Backend, value: Linear) -> Result<Linear> {
            Ok(Linear {
                weight: backend.to_device(&value.weight)?,
                bias: backend.to_device(&value.bias)?,
            })
        }
        fn block(
            backend: &dyn Backend,
            value: TransformerBlockWeights,
        ) -> Result<TransformerBlockWeights> {
            Ok(TransformerBlockWeights {
                ada: value.ada.map(|v| linear(backend, v)).transpose()?,
                norm1_weight: value
                    .norm1_weight
                    .map(|v| backend.to_device(&v))
                    .transpose()?,
                norm1_bias: value
                    .norm1_bias
                    .map(|v| backend.to_device(&v))
                    .transpose()?,
                q: linear(backend, value.q)?,
                k: linear(backend, value.k)?,
                v: linear(backend, value.v)?,
                out: linear(backend, value.out)?,
                norm3_weight: backend.to_device(&value.norm3_weight)?,
                norm3_bias: backend.to_device(&value.norm3_bias)?,
                ff_in: linear(backend, value.ff_in)?,
                ff_out: linear(backend, value.ff_out)?,
            })
        }
        Ok(Self {
            timestep_1: linear(backend, self.timestep_1)?,
            timestep_2: linear(backend, self.timestep_2)?,
            dit: self
                .dit
                .into_iter()
                .map(|v| block(backend, v))
                .collect::<Result<_>>()?,
            output_1: linear(backend, self.output_1)?,
            output_2: linear(backend, self.output_2)?,
            position_embedding: backend.to_device(&self.position_embedding)?,
            vlln_weight: backend.to_device(&self.vlln_weight)?,
            vlln_bias: backend.to_device(&self.vlln_bias)?,
            vl: self
                .vl
                .into_iter()
                .map(|v| block(backend, v))
                .collect::<Result<_>>()?,
            // Category banks stay on the host. A request selects one category
            // once and uploads only its seven compact matrices.
            state_1: self.state_1,
            state_2: self.state_2,
            action_1: self.action_1,
            action_2: self.action_2,
            action_3: self.action_3,
            decoder_1: self.decoder_1,
            decoder_2: self.decoder_2,
        })
    }
}

fn take_block(
    map: &mut HashMap<String, Tensor>,
    prefix: &str,
    ada: bool,
) -> Result<TransformerBlockWeights> {
    Ok(TransformerBlockWeights {
        ada: ada
            .then(|| take_linear(map, &format!("{prefix}.norm1.linear")))
            .transpose()?,
        norm1_weight: (!ada)
            .then(|| take(map, &format!("{prefix}.norm1.weight")))
            .transpose()?,
        norm1_bias: (!ada)
            .then(|| take(map, &format!("{prefix}.norm1.bias")))
            .transpose()?,
        q: take_linear(map, &format!("{prefix}.attn1.to_q"))?,
        k: take_linear(map, &format!("{prefix}.attn1.to_k"))?,
        v: take_linear(map, &format!("{prefix}.attn1.to_v"))?,
        out: take_linear(map, &format!("{prefix}.attn1.to_out.0"))?,
        // DiT uses elementwise_affine=false; the VL self-attention stack uses
        // ordinary affine LayerNorm. Materialize the identity affine tensors
        // for the former so both paths share the safe LayerNorm operator.
        norm3_weight: if ada {
            Tensor::from_bf16(vec![1536], &vec![half::bf16::ONE; 1536])?
        } else {
            take(map, &format!("{prefix}.norm3.weight"))?
        },
        norm3_bias: if ada {
            Tensor::from_bf16(vec![1536], &vec![half::bf16::ZERO; 1536])?
        } else {
            take(map, &format!("{prefix}.norm3.bias"))?
        },
        ff_in: take_linear(map, &format!("{prefix}.ff.net.0.proj"))?,
        ff_out: take_linear(map, &format!("{prefix}.ff.net.2"))?,
    })
}

fn take_linear(map: &mut HashMap<String, Tensor>, prefix: &str) -> Result<Linear> {
    Ok(Linear {
        weight: transpose_2d(&take(map, &format!("{prefix}.weight"))?)?,
        bias: take(map, &format!("{prefix}.bias"))?,
    })
}

fn take_category(map: &mut HashMap<String, Tensor>, prefix: &str) -> Result<CategoryLinear> {
    Ok(CategoryLinear {
        // Official CategorySpecificLinear stores [category, in, out], already
        // in ApxInf row-major GEMM orientation.
        weight: take(map, &format!("{prefix}.W"))?,
        bias: take(map, &format!("{prefix}.b"))?,
    })
}

fn take(map: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    map.remove(name)
        .ok_or_else(|| Error::Other(format!("missing GR00T N1.7 weight {name}")))
}

fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "expected 2D linear weight, got {dims:?}"
        )));
    }
    let (rows, cols) = (dims[0], dims[1]);
    match tensor.dtype() {
        DType::BF16 => {
            let input = tensor.as_bf16()?;
            let mut output = vec![half::bf16::ZERO; rows * cols];
            for row in 0..rows {
                for col in 0..cols {
                    output[col * rows + row] = input[row * cols + col];
                }
            }
            Tensor::from_bf16(vec![cols, rows], &output)
        }
        DType::F32 => {
            let input = tensor.as_f32()?;
            let mut output = vec![0.0; rows * cols];
            for row in 0..rows {
                for col in 0..cols {
                    output[col * rows + row] = input[row * cols + col];
                }
            }
            Tensor::from_f32(vec![cols, rows], &output)
        }
        dtype => Err(Error::Other(format!(
            "GR00T linear weight dtype {dtype} is unsupported"
        ))),
    }
}
