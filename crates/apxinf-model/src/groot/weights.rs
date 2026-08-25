use std::{collections::HashMap, sync::Arc};

use apxinf_core::{Backend, DType, Error, Result, Tensor};

use super::{CategorySpecificLinear, CategorySpecificMlp};

pub(super) struct Linear { pub weight: Tensor, pub bias: Tensor }
pub(super) struct Attention { pub q: Linear, pub k: Linear, pub v: Linear, pub out: Linear }
pub(super) struct VlBlock {
    pub norm1_w: Tensor, pub norm1_b: Tensor, pub attention: Attention,
    pub norm3_w: Tensor, pub norm3_b: Tensor, pub ff_in: Linear, pub ff_out: Linear,
}
pub(super) struct DitBlock { pub ada: Linear, pub attention: Attention, pub ff_in: Linear, pub ff_out: Linear }
pub(super) struct GrootActionWeights {
    pub state: CategorySpecificMlp,
    pub action_w1: CategorySpecificLinear,
    pub action_w2: CategorySpecificLinear,
    pub action_w3: CategorySpecificLinear,
    pub decoder: CategorySpecificMlp,
    pub position: Tensor,
    pub vlln_w: Tensor, pub vlln_b: Tensor,
    pub vl_blocks: Vec<VlBlock>,
    pub timestep_1: Linear, pub timestep_2: Linear,
    pub dit_blocks: Vec<DitBlock>,
    pub out_style: Linear, pub out: Linear,
}

impl Linear {
    fn take(prefix: &str, map: &mut HashMap<String, Tensor>, backend: &dyn Backend) -> Result<Self> {
        let weight = transpose(&take(map, &format!("{prefix}.weight"))?)?;
        let bias = take(map, &format!("{prefix}.bias"))?;
        Ok(Self { weight: backend.to_device(&weight)?, bias: backend.to_device(&bias)? })
    }
}

impl Attention {
    fn take(prefix: &str, map: &mut HashMap<String, Tensor>, backend: &dyn Backend) -> Result<Self> {
        Ok(Self {
            q: Linear::take(&format!("{prefix}.to_q"), map, backend)?,
            k: Linear::take(&format!("{prefix}.to_k"), map, backend)?,
            v: Linear::take(&format!("{prefix}.to_v"), map, backend)?,
            out: Linear::take(&format!("{prefix}.to_out.0"), map, backend)?,
        })
    }
}

impl GrootActionWeights {
    pub fn take(map: &mut HashMap<String, Tensor>, backend: Arc<dyn Backend>) -> Result<Self> {
        let category = |prefix: &str, map: &mut HashMap<String, Tensor>| -> Result<CategorySpecificLinear> {
            let weights = split_categories(take(map, &format!("{prefix}.W"))?, &*backend)?;
            let biases = split_categories(take(map, &format!("{prefix}.b"))?, &*backend)?;
            CategorySpecificLinear::new(weights, biases, Arc::clone(&backend))
        };
        let state1 = category("action_head.state_encoder.layer1", map)?;
        let state2 = category("action_head.state_encoder.layer2", map)?;
        let state = CategorySpecificMlp::new(state1, state2, Arc::clone(&backend));
        let action_w1 = category("action_head.action_encoder.W1", map)?;
        let action_w2 = category("action_head.action_encoder.W2", map)?;
        let action_w3 = category("action_head.action_encoder.W3", map)?;
        let decoder1 = category("action_head.action_decoder.layer1", map)?;
        let decoder2 = category("action_head.action_decoder.layer2", map)?;
        let decoder = CategorySpecificMlp::new(decoder1, decoder2, Arc::clone(&backend));

        let position = backend.to_device(&take(map, "action_head.position_embedding.weight")?)?;
        let vlln_w = backend.to_device(&take(map, "action_head.vlln.weight")?)?;
        let vlln_b = backend.to_device(&take(map, "action_head.vlln.bias")?)?;
        let mut vl_blocks = Vec::with_capacity(4);
        for index in 0..4 {
            let prefix = format!("action_head.vl_self_attention.transformer_blocks.{index}");
            vl_blocks.push(VlBlock {
                norm1_w: backend.to_device(&take(map, &format!("{prefix}.norm1.weight"))?)?,
                norm1_b: backend.to_device(&take(map, &format!("{prefix}.norm1.bias"))?)?,
                attention: Attention::take(&format!("{prefix}.attn1"), map, &*backend)?,
                norm3_w: backend.to_device(&take(map, &format!("{prefix}.norm3.weight"))?)?,
                norm3_b: backend.to_device(&take(map, &format!("{prefix}.norm3.bias"))?)?,
                ff_in: Linear::take(&format!("{prefix}.ff.net.0.proj"), map, &*backend)?,
                ff_out: Linear::take(&format!("{prefix}.ff.net.2"), map, &*backend)?,
            });
        }
        let root = "action_head.model";
        let timestep_1 = Linear::take(&format!("{root}.timestep_encoder.timestep_embedder.linear_1"), map, &*backend)?;
        let timestep_2 = Linear::take(&format!("{root}.timestep_encoder.timestep_embedder.linear_2"), map, &*backend)?;
        let mut dit_blocks = Vec::with_capacity(32);
        for index in 0..32 {
            let prefix = format!("{root}.transformer_blocks.{index}");
            dit_blocks.push(DitBlock {
                ada: Linear::take(&format!("{prefix}.norm1.linear"), map, &*backend)?,
                attention: Attention::take(&format!("{prefix}.attn1"), map, &*backend)?,
                ff_in: Linear::take(&format!("{prefix}.ff.net.0.proj"), map, &*backend)?,
                ff_out: Linear::take(&format!("{prefix}.ff.net.2"), map, &*backend)?,
            });
        }
        Ok(Self {
            state, action_w1, action_w2, action_w3, decoder, position, vlln_w, vlln_b,
            vl_blocks, timestep_1, timestep_2, dit_blocks,
            out_style: Linear::take(&format!("{root}.proj_out_1"), map, &*backend)?,
            out: Linear::take(&format!("{root}.proj_out_2"), map, &*backend)?,
        })
    }
}

fn take(map: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    map.remove(name).ok_or_else(|| Error::Other(format!("missing GR00T weight {name}")))
}

fn split_categories(tensor: Tensor, backend: &dyn Backend) -> Result<Vec<Tensor>> {
    let dims = tensor.shape().dims();
    if dims.len() < 2 || dims[0] == 0 { return Err(Error::Other("category tensor must begin with category axis".into())); }
    let width: usize = dims[1..].iter().product();
    let values = tensor.to_f32_vec()?;
    (0..dims[0]).map(|category| {
        let shape = dims[1..].to_vec();
        let host = match tensor.dtype() {
            DType::BF16 => Tensor::from_bf16(shape, &values[category * width..(category + 1) * width]
                .iter().map(|x| half::bf16::from_f32(*x)).collect::<Vec<_>>())?,
            DType::F32 => Tensor::from_f32(shape, &values[category * width..(category + 1) * width])?,
            dtype => return Err(Error::Other(format!("unsupported category dtype {dtype}"))),
        };
        backend.to_device(&host)
    }).collect()
}

fn transpose(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 { return Err(Error::Other(format!("linear weight must be 2-D, got {dims:?}"))); }
    let (rows, cols) = (dims[0], dims[1]);
    let src = tensor.to_f32_vec()?;
    let mut dst = vec![0.0f32; src.len()];
    for row in 0..rows { for col in 0..cols { dst[col * rows + row] = src[row * cols + col]; } }
    match tensor.dtype() {
        DType::BF16 => Tensor::from_bf16(vec![cols, rows], &dst.into_iter().map(half::bf16::from_f32).collect::<Vec<_>>()),
        DType::F32 => Tensor::from_f32(vec![cols, rows], &dst),
        dtype => Err(Error::Other(format!("unsupported linear dtype {dtype}"))),
    }
}
