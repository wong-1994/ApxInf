use std::sync::Arc;

use apxinf_core::{Backend, DType, Error, Result, Tensor};

use super::{
    CategoryLinearWeights, CategoryMlpWeights, GrootN1d7ActionWeights, GrootN1d7DiTBlockWeights,
    GrootN1d7SelfAttentionBlockWeights, LinearWeights,
};

pub struct DeviceLinear {
    pub weight: Tensor,
    pub bias: Tensor,
}
pub struct DeviceMlp {
    pub layer1: DeviceLinear,
    pub layer2: DeviceLinear,
}
pub struct DeviceDiTBlock {
    pub q: DeviceLinear,
    pub k: DeviceLinear,
    pub v: DeviceLinear,
    pub output: DeviceLinear,
    pub ada_norm: DeviceLinear,
    pub ff_in: DeviceLinear,
    pub ff_out: DeviceLinear,
}
pub struct DeviceSelfAttentionBlock {
    pub norm1_weight: Tensor,
    pub norm1_bias: Tensor,
    pub q: DeviceLinear,
    pub k: DeviceLinear,
    pub v: DeviceLinear,
    pub output: DeviceLinear,
    pub norm3_weight: Tensor,
    pub norm3_bias: Tensor,
    pub ff_in: DeviceLinear,
    pub ff_out: DeviceLinear,
}
pub struct DeviceActionWeights {
    pub state_encoder: DeviceMlp,
    pub action_w1: DeviceLinear,
    pub action_w2: DeviceLinear,
    pub action_w3: DeviceLinear,
    pub action_decoder: DeviceMlp,
    pub position_embedding: Tensor,
    pub vl_norm_weight: Tensor,
    pub vl_norm_bias: Tensor,
    pub vl_blocks: Vec<DeviceSelfAttentionBlock>,
    pub timestep_in: DeviceLinear,
    pub timestep_out: DeviceLinear,
    pub dit_blocks: Vec<DeviceDiTBlock>,
    pub final_condition: DeviceLinear,
    pub final_output: DeviceLinear,
}

impl DeviceActionWeights {
    pub fn upload(
        source: &GrootN1d7ActionWeights,
        category: usize,
        b: &Arc<dyn Backend>,
    ) -> Result<Self> {
        let linear = |w: &LinearWeights| upload_linear(w, &**b);
        let category_linear = |w: &CategoryLinearWeights| upload_category(w, category, &**b);
        let mlp = |w: &CategoryMlpWeights| -> Result<DeviceMlp> {
            Ok(DeviceMlp {
                layer1: category_linear(&w.layer1)?,
                layer2: category_linear(&w.layer2)?,
            })
        };
        let mut vl_blocks = Vec::with_capacity(source.vl_blocks.len());
        for w in &source.vl_blocks {
            vl_blocks.push(upload_vl(w, &**b)?);
        }
        let mut dit_blocks = Vec::with_capacity(source.dit_blocks.len());
        for w in &source.dit_blocks {
            dit_blocks.push(upload_dit(w, &**b)?);
        }
        Ok(Self {
            state_encoder: mlp(&source.state_encoder)?,
            action_w1: category_linear(&source.action_w1)?,
            action_w2: category_linear(&source.action_w2)?,
            action_w3: category_linear(&source.action_w3)?,
            action_decoder: mlp(&source.action_decoder)?,
            position_embedding: b.to_device(&source.position_embedding)?,
            vl_norm_weight: b.to_device(&source.vl_norm_weight)?,
            vl_norm_bias: b.to_device(&source.vl_norm_bias)?,
            vl_blocks,
            timestep_in: linear(&source.timestep_in)?,
            timestep_out: linear(&source.timestep_out)?,
            dit_blocks,
            final_condition: linear(&source.final_condition)?,
            final_output: linear(&source.final_output)?,
        })
    }
}

fn upload_linear(w: &LinearWeights, b: &dyn Backend) -> Result<DeviceLinear> {
    Ok(DeviceLinear {
        weight: b.to_device(&w.weight)?,
        bias: b.to_device(&w.bias)?,
    })
}
fn upload_category(
    w: &CategoryLinearWeights,
    category: usize,
    b: &dyn Backend,
) -> Result<DeviceLinear> {
    let dims = w.weight.shape().dims();
    if dims.len() != 3 || category >= dims[0] {
        return Err(Error::Other(format!("invalid GR00T embodiment {category}")));
    }
    let weight = category_slice(&w.weight, category, vec![dims[1], dims[2]])?;
    let bias_dims = w.bias.shape().dims();
    let bias = category_slice(&w.bias, category, vec![bias_dims[1]])?;
    Ok(DeviceLinear {
        weight: b.to_device(&weight)?,
        bias: b.to_device(&bias)?,
    })
}
fn category_slice(t: &Tensor, category: usize, shape: Vec<usize>) -> Result<Tensor> {
    let width: usize = shape.iter().product();
    let start = category * width;
    match t.dtype() {
        DType::BF16 => Tensor::from_bf16(shape, &t.as_bf16()?[start..start + width]),
        DType::F32 => Tensor::from_f32(shape, &t.as_f32()?[start..start + width]),
        dtype => Err(Error::Other(format!(
            "GR00T category tensor dtype {dtype} unsupported"
        ))),
    }
}
fn upload_vl(
    w: &GrootN1d7SelfAttentionBlockWeights,
    b: &dyn Backend,
) -> Result<DeviceSelfAttentionBlock> {
    Ok(DeviceSelfAttentionBlock {
        norm1_weight: b.to_device(&w.norm1_weight)?,
        norm1_bias: b.to_device(&w.norm1_bias)?,
        q: upload_linear(&w.q, b)?,
        k: upload_linear(&w.k, b)?,
        v: upload_linear(&w.v, b)?,
        output: upload_linear(&w.output, b)?,
        norm3_weight: b.to_device(&w.norm3_weight)?,
        norm3_bias: b.to_device(&w.norm3_bias)?,
        ff_in: upload_linear(&w.ff_in, b)?,
        ff_out: upload_linear(&w.ff_out, b)?,
    })
}
fn upload_dit(w: &GrootN1d7DiTBlockWeights, b: &dyn Backend) -> Result<DeviceDiTBlock> {
    Ok(DeviceDiTBlock {
        q: upload_linear(&w.q, b)?,
        k: upload_linear(&w.k, b)?,
        v: upload_linear(&w.v, b)?,
        output: upload_linear(&w.output, b)?,
        ada_norm: upload_linear(&w.ada_norm, b)?,
        ff_in: upload_linear(&w.ff_in, b)?,
        ff_out: upload_linear(&w.ff_out, b)?,
    })
}
