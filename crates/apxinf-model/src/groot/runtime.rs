use std::{cell::RefCell, collections::HashMap, path::Path, sync::Arc};

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};

use crate::{accelerator::create_backend, qwen3vl::{GeneralQwen3VL, Qwen3VLConfig}, vla::{Action, InferenceSpec, Observation, PreparedInference, VisionObservation, VlaRuntime}};

use super::{four_step_schedule, weights::{Attention, DitBlock, GrootActionWeights, Linear, VlBlock}, GrootConfig};

pub struct GrootRuntime {
    config: GrootConfig,
    qwen: RefCell<GeneralQwen3VL>,
    weights: GrootActionWeights,
    backend: Arc<dyn Backend>,
    norm_w: Tensor,
    norm_b: Tensor,
}

impl GrootRuntime {
    pub fn from_dir(model_dir: &Path, device: Device) -> Result<Self> {
        Self::from_dir_with_backend(model_dir, create_backend(device)?)
    }

    pub(crate) fn from_dir_with_backend(model_dir: &Path, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = GrootConfig::from_json_file(&model_dir.join("config.json"))?;
        let (mut tensors, _) = apxinf_loader::safetensors::load_native_path(model_dir)
            .map_err(|error| Error::Other(format!("load GR00T safetensors: {error}")))?;
        let qwen_tensors = tensors.iter().filter_map(|(name, tensor)| {
            name.strip_prefix("backbone.model.").map(|name| (name.to_owned(), tensor.clone()))
        }).collect::<HashMap<_, _>>();
        let qwen_config = cosmos_reason2_config(config.select_layer)?;
        let qwen = GeneralQwen3VL::from_weights_with_backend(qwen_config, qwen_tensors, Arc::clone(&backend))?;
        let weights = GrootActionWeights::take(&mut tensors, Arc::clone(&backend))?;
        let ones = vec![half::bf16::from_f32(1.0); config.input_embedding_dim];
        let zeros = vec![half::bf16::ZERO; config.input_embedding_dim];
        let norm_w = backend.to_device(&Tensor::from_bf16(vec![config.input_embedding_dim], &ones)?)?;
        let norm_b = backend.to_device(&Tensor::from_bf16(vec![config.input_embedding_dim], &zeros)?)?;
        Ok(Self { config, qwen: RefCell::new(qwen), weights, backend, norm_w, norm_b })
    }

    fn run(&self, observation: &Observation) -> Result<Tensor> {
        observation.validate()?;
        let VisionObservation::Patches(pixels) = &observation.vision else {
            return Err(Error::Other("GR00T expects Qwen-preprocessed image patches".into()));
        };
        let state = observation.conditioning.state.as_ref()
            .ok_or_else(|| Error::Other("GR00T requires normalized state conditioning".into()))?;
        if state.shape().dims() != [self.config.state_history_length, self.config.max_state_dim] {
            return Err(Error::Other(format!("GR00T state shape must be [{},{}]", self.config.state_history_length, self.config.max_state_dim)));
        }
        if observation.noise.shape().dims() != [self.config.action_horizon, self.config.max_action_dim] {
            return Err(Error::Other(format!("GR00T noise shape must be [{},{}]", self.config.action_horizon, self.config.max_action_dim)));
        }
        let embodiment = observation.conditioning.embodiment_id
            .ok_or_else(|| Error::Other("GR00T requires embodiment_id".into()))? as usize;
        let grid = observation.conditioning.image_grid_thw.chunks_exact(3)
            .map(|x| [x[0], x[1], x[2]]).collect::<Vec<_>>();
        if grid.is_empty() { return Err(Error::Other("GR00T requires image_grid_thw".into())); }
        let pixels = self.upload_bf16(pixels)?;
        let mut qwen = self.qwen.borrow_mut();
        qwen.reset_state()?;
        let mut vl = qwen.encode_multimodal_to_layer(&observation.token_ids, &pixels, &grid, self.config.select_layer)?;
        drop(qwen);
        vl = self.backend.layer_norm(&vl, &self.weights.vlln_w, &self.weights.vlln_b, 1e-5)?;
        for block in &self.weights.vl_blocks { vl = self.vl_block(&vl, block)?; }

        let state = self.upload_bf16(state)?;
        let state_features = self.weights.state.forward(&state, embodiment)?;
        let mut actions = self.upload_bf16(&observation.noise)?;
        let full_mask = if observation.conditioning.attention_mask.is_empty() {
            vec![1; observation.token_ids.len()]
        } else { observation.conditioning.attention_mask.clone() };
        let image_mask = observation.token_ids.iter().zip(&full_mask)
            .map(|(&id, &valid)| u8::from(valid != 0 && id == 151655)).collect::<Vec<_>>();
        let text_mask = observation.token_ids.iter().zip(&full_mask)
            .map(|(&id, &valid)| u8::from(valid != 0 && id != 151655)).collect::<Vec<_>>();
        for step in four_step_schedule(self.config.num_timestep_buckets)? {
            let encoded = self.action_encode(&actions, step.bucket, embodiment)?;
            let position = self.backend.slice_2d(&self.weights.position, 0, self.config.action_horizon, 0, self.config.input_embedding_dim)?;
            let encoded = self.backend.add(&encoded, &position)?;
            let mut hidden = self.backend.concat_rows(&state_features, &encoded)?;
            let temb = self.timestep_embedding(step.bucket)?;
            for (index, block) in self.weights.dit_blocks.iter().enumerate() {
                let mask = if index % 4 == 0 { Some(text_mask.as_slice()) }
                    else if index % 2 == 0 { Some(image_mask.as_slice()) } else { None };
                hidden = self.dit_block(&hidden, &vl, &temb, block, mask, index % 2 == 1)?;
            }
            let style = linear(&*self.backend, &self.backend.silu(&temb)?, &self.weights.out_style)?;
            hidden = self.adaptive_norm(&hidden, &style, 1e-6)?;
            let output = linear(&*self.backend, &hidden, &self.weights.out)?;
            let velocity = self.weights.decoder.forward(&output, embodiment)?;
            let velocity = self.backend.slice_2d(&velocity, 1, self.config.action_horizon, 0, self.config.max_action_dim)?;
            actions = self.backend.add(&actions, &self.backend.scale(&velocity, step.dt)?)?;
        }
        self.backend.synchronize()?;
        Ok(actions)
    }

    fn upload_bf16(&self, tensor: &Tensor) -> Result<Tensor> {
        let host = if tensor.device() == Device::Cpu { tensor.clone() } else { self.backend.to_cpu(tensor)? };
        let host = if host.dtype() == DType::BF16 { host } else {
            Tensor::from_bf16(host.shape().dims().to_vec(), &host.to_f32_vec()?.into_iter().map(half::bf16::from_f32).collect::<Vec<_>>())?
        };
        self.backend.to_device(&host)
    }

    fn vl_block(&self, x: &Tensor, block: &VlBlock) -> Result<Tensor> {
        let norm = self.backend.layer_norm(x, &block.norm1_w, &block.norm1_b, 1e-5)?;
        let attended = self.attention(&norm, &norm, &block.attention, None, false, 32, 64)?;
        let x = self.backend.add(x, &attended)?;
        let norm = self.backend.layer_norm(&x, &block.norm3_w, &block.norm3_b, 1e-5)?;
        let ff = linear(&*self.backend, &self.backend.gelu_tanh(&linear(&*self.backend, &norm, &block.ff_in)?)?, &block.ff_out)?;
        self.backend.add(&x, &ff)
    }

    fn action_encode(&self, actions: &Tensor, bucket: u32, embodiment: usize) -> Result<Tensor> {
        let action = self.weights.action_w1.forward(actions, embodiment)?;
        let time = sinusoid_rows(self.config.action_horizon, self.config.input_embedding_dim, bucket as f32, false, 0.0)?;
        let time = self.backend.to_device(&time)?;
        let joined = self.backend.concat_2d(&[&action, &time])?;
        let hidden = self.backend.silu(&self.weights.action_w2.forward(&joined, embodiment)?)?;
        self.weights.action_w3.forward(&hidden, embodiment)
    }

    fn timestep_embedding(&self, bucket: u32) -> Result<Tensor> {
        let time = sinusoid_rows(1, 256, bucket as f32, true, 1.0)?;
        let time = self.backend.to_device(&time)?;
        let hidden = self.backend.silu(&linear(&*self.backend, &time, &self.weights.timestep_1)?)?;
        linear(&*self.backend, &hidden, &self.weights.timestep_2)
    }

    fn adaptive_norm(&self, input: &Tensor, style: &Tensor, eps: f32) -> Result<Tensor> {
        let rows = input.shape().dims()[0];
        let width = input.shape().dims()[1];
        let shift = self.backend.slice_2d(style, 0, 1, 0, width)?;
        let scale = self.backend.slice_2d(style, 0, 1, width, width)?;
        let shift = repeat_rows(&*self.backend, &shift, rows)?;
        let scale = repeat_rows(&*self.backend, &scale, rows)?;
        let normalized = self.backend.layer_norm(input, &self.norm_w, &self.norm_b, eps)?;
        let ones = self.backend.to_device(&Tensor::from_bf16(vec![rows, width], &vec![half::bf16::from_f32(1.0); rows * width])?)?;
        self.backend.add(&self.backend.mul(&normalized, &self.backend.add(&ones, &scale)?)?, &shift)
    }

    fn dit_block(&self, hidden: &Tensor, vl: &Tensor, temb: &Tensor, block: &DitBlock,
                 mask: Option<&[u8]>, self_attention: bool) -> Result<Tensor> {
        let style = linear(&*self.backend, &self.backend.silu(temb)?, &block.ada)?;
        let norm = self.adaptive_norm(hidden, &style, 1e-5)?;
        let attended = if self_attention {
            self.attention(&norm, &norm, &block.attention, None, false, 32, 48)?
        } else { self.attention(&norm, vl, &block.attention, mask, false, 32, 48)? };
        let hidden = self.backend.add(hidden, &attended)?;
        let norm = self.backend.layer_norm(&hidden, &self.norm_w, &self.norm_b, 1e-5)?;
        let ff = linear(&*self.backend, &self.backend.gelu_tanh(&linear(&*self.backend, &norm, &block.ff_in)?)?, &block.ff_out)?;
        self.backend.add(&hidden, &ff)
    }

    fn attention(&self, query: &Tensor, context: &Tensor, weights: &Attention,
                 mask: Option<&[u8]>, causal: bool, heads: usize, head_dim: usize) -> Result<Tensor> {
        let q_len = query.shape().dims()[0]; let kv_len = context.shape().dims()[0];
        let q = linear(&*self.backend, query, &weights.q)?.reshape(vec![q_len, heads, head_dim])?;
        let k = linear(&*self.backend, context, &weights.k)?.reshape(vec![kv_len, heads, head_dim])?;
        let v = linear(&*self.backend, context, &weights.v)?.reshape(vec![kv_len, heads, head_dim])?;
        let output = self.backend.cross_sdpa(&q, &k, &v, q_len, kv_len, heads, head_dim, mask, causal)?;
        linear(&*self.backend, &output, &weights.out)
    }
}

impl VlaRuntime for GrootRuntime {
    fn infer(&self, observation: &Observation) -> Result<Action> { Ok(Action::new(self.run(observation)?)) }
    fn prepare(&self, _spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Err(Error::Other("GR00T prepared graph capture is not implemented yet".into()))
    }
    fn infer_host_f32(&self, observation: &Observation) -> Result<Vec<f32>> {
        self.backend.to_cpu(&self.run(observation)?)?.to_f32_vec()
    }
}

fn linear(backend: &dyn Backend, input: &Tensor, weights: &Linear) -> Result<Tensor> {
    backend.add_bias(&backend.matmul(input, &weights.weight)?, &weights.bias)
}

fn repeat_rows(backend: &dyn Backend, row: &Tensor, count: usize) -> Result<Tensor> {
    if count == 0 { return Err(Error::Other("cannot repeat zero rows".into())); }
    let mut output = row.clone();
    for _ in 1..count { output = backend.concat_rows(&output, row)?; }
    Ok(output)
}

fn sinusoid_rows(rows: usize, dim: usize, time: f32, flip: bool, downscale_shift: f32) -> Result<Tensor> {
    let half = dim / 2;
    let denominator = half as f32 - downscale_shift;
    let mut one = Vec::with_capacity(dim);
    let angles = (0..half).map(|i| time * (-10000.0f32.ln() * i as f32 / denominator).exp()).collect::<Vec<_>>();
    if flip { one.extend(angles.iter().map(|x| x.cos())); one.extend(angles.iter().map(|x| x.sin())); }
    else { one.extend(angles.iter().map(|x| x.sin())); one.extend(angles.iter().map(|x| x.cos())); }
    let values = (0..rows).flat_map(|_| one.iter().copied()).map(half::bf16::from_f32).collect::<Vec<_>>();
    Tensor::from_bf16(vec![rows, dim], &values)
}

fn cosmos_reason2_config(layer_count: usize) -> Result<Qwen3VLConfig> {
    let json = format!(r#"{{"image_token_id":151655,"video_token_id":151656,"vision_start_token_id":151652,"vision_end_token_id":151653,
      "text_config":{{"hidden_size":2048,"intermediate_size":6144,"num_hidden_layers":{layer_count},"num_attention_heads":16,
      "num_key_value_heads":8,"head_dim":128,"vocab_size":151936,"max_position_embeddings":262144,"rms_norm_eps":1e-6,
      "rope_theta":5000000,"tie_word_embeddings":true,"rope_scaling":{{"mrope_interleaved":true,"mrope_section":[24,20,20]}}}},
      "vision_config":{{"depth":24,"hidden_size":1024,"intermediate_size":4096,"num_heads":16,"patch_size":16,
      "temporal_patch_size":2,"in_channels":3,"spatial_merge_size":2,"num_position_embeddings":2304,"out_hidden_size":2048,
      "deepstack_visual_indexes":[5,11,17]}}}}"#);
    Qwen3VLConfig::from_json_str(&json)
}
