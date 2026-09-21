//! GPU-only forward topology. Request binding and graph ownership live in ModelRunner.
use super::{
    backend::{kernels, Context, DeviceBuffer, RuntimeBackend},
    config::Config,
    weights::{Linear, Weights},
};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use kernels::{
    activation, attention, cache, elementwise, embedding, gemm, linear_attention, norm, rope,
    sampling,
};
use std::path::Path;
pub struct Model {
    pub config: Config,
    weights: Weights,
}
fn buffer(t: &Tensor) -> Result<DeviceBuffer> {
    DeviceBuffer::from_tensor(t).map_err(Error::Cuda)
}
fn binary(ctx: &Context, a: &Tensor, b: &Tensor, multiply: bool) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(Error::Other("OpenVLA residual/scale shape mismatch".into()));
    }
    let out = kernels::scratch_buffer(ctx, a.size_in_bytes())?;
    if multiply {
        elementwise::mul_into(ctx, DType::BF16, &buffer(a)?, &buffer(b)?, &out, a.numel())?;
    } else {
        elementwise::add_into(ctx, DType::BF16, &buffer(a)?, &buffer(b)?, &out, a.numel())?;
    }
    out.as_tensor(a.shape().clone(), DType::BF16)
        .map_err(Error::Cuda)
}
fn linear(ctx: &Context, x: &Tensor, w: &Linear) -> Result<Tensor> {
    match &w.bias {
        Some(b) => gemm::bf16_bias_with_workspace_limit(ctx, x, &w.weight, b, 1024 * 1024),
        None => w.plan.run(ctx, x, &w.weight),
    }
}
impl Model {
    pub fn from_checkpoint(root: &Path, backend: &RuntimeBackend) -> Result<Self> {
        Ok(Self {
            config: Config::load(root)?,
            weights: Weights::load(root, backend)?,
        })
    }
    /// Conservative bound for the monotonic per-traversal arena (including all KV).
    pub fn workspace_bytes(&self, tokens: usize) -> Result<usize> {
        if tokens < 2 || tokens > self.config.max_tokens {
            return Err(Error::Other("invalid OpenVLA token count".into()));
        }
        let rows = tokens + 256;
        let conv = self
            .weights
            .towers
            .iter()
            .map(|t| t.plan.workspace_bytes())
            .sum::<usize>();
        Ok(conv + 768 * 1024 * 1024 + rows * 32 * 400_000)
    }
    /// Preprocessed BF16 pixels `[6,224*224]`, DINO channels first then SigLIP.
    pub fn encode(&self, ctx: &Context, pixels: &Tensor) -> Result<Tensor> {
        if pixels.dtype() != DType::BF16 || pixels.shape().dims() != [6, 50176] {
            return Err(Error::Other(
                "OpenVLA expects BF16 normalized pixels [6,50176]".into(),
            ));
        }
        let mut features = Vec::new();
        for (i, t) in self.weights.towers.iter().enumerate() {
            let rgb = elementwise::contiguous_rows(ctx, pixels, i * 3, 3)?
                .reshape(vec![1, 3, 224, 224])?;
            let patch = t
                .plan
                .execute(ctx, &rgb, &t.conv)?
                .reshape(vec![256, t.width])?;
            let patch = elementwise::bias_bf16(ctx, &patch, Some(&t.bias))?;
            let mut h = binary(ctx, &patch, &t.position, false)?;
            if let Some(prefix) = &t.prefix {
                h = elementwise::concat_rows_bf16(ctx, prefix, &h)?;
            }
            for w in &t.blocks {
                let x = norm::layer_bf16_welford(ctx, &h, &w.norm1.weight, &w.norm1.bias, 1e-6)?;
                let qkv = linear(ctx, &x, &w.qkv)?;
                let parts = attention::split_qkv_bias_bf16(ctx, &qkv, None, 16, t.width / 16)?;
                let x = attention::mha_bf16_precise(
                    ctx,
                    &parts.q,
                    &parts.k,
                    &parts.v,
                    h.shape().dims()[0],
                )?
                .reshape(h.shape().dims().to_vec())?;
                let mut x = linear(ctx, &x, &w.proj)?;
                if let Some(scale) = &w.scale1 {
                    x = binary(ctx, &x, scale, true)?;
                }
                h = binary(ctx, &h, &x, false)?;
                let x = norm::layer_bf16_welford(ctx, &h, &w.norm2.weight, &w.norm2.bias, 1e-6)?;
                let x = linear_attention::gelu_exact(ctx, &linear(ctx, &x, &w.fc1)?)?;
                let mut x = linear(ctx, &x, &w.fc2)?;
                if let Some(scale) = &w.scale2 {
                    x = binary(ctx, &x, scale, true)?;
                }
                h = binary(ctx, &h, &x, false)?;
            }
            // timm get_intermediate_layers(..., norm=False): omit final block and norm.
            features.push(elementwise::contiguous_rows(
                ctx,
                &h,
                if t.prefix.is_some() { 5 } else { 0 },
                256,
            )?);
        }
        let mut h = elementwise::concat_columns_bf16(ctx, &features.iter().collect::<Vec<_>>())?;
        for (i, w) in self.weights.projector.iter().enumerate() {
            h = linear(ctx, &h, w)?;
            if i < 2 {
                h = linear_attention::gelu_exact(ctx, &h)?;
            }
        }
        Ok(h)
    }
    fn embed(&self, ctx: &Context, ids: &DeviceBuffer, count: usize) -> Result<Tensor> {
        let output = kernels::scratch_buffer(ctx, count * 4096 * 2)?;
        embedding::lookup_into(
            ctx,
            DType::BF16,
            &buffer(&self.weights.embed)?,
            ids.address(),
            &output,
            4096,
            count,
        )?;
        output
            .as_tensor(Shape::new(vec![count, 4096]), DType::BF16)
            .map_err(Error::Cuda)
    }
    /// Fixed seven-step greedy decode. IDs stay on device through the entire loop.
    pub fn forward(
        &self,
        ctx: &Context,
        pixels: &Tensor,
        ids: &DeviceBuffer,
        count: usize,
    ) -> Result<Tensor> {
        self.workspace_bytes(count)?;
        let image = self.encode(ctx, pixels)?;
        let text = self.embed(ctx, ids, count)?;
        let bos = elementwise::contiguous_rows(ctx, &text, 0, 1)?;
        let tail = elementwise::contiguous_rows(ctx, &text, 1, count - 1)?;
        let mut h = elementwise::concat_rows_bf16(
            ctx,
            &elementwise::concat_rows_bf16(ctx, &bos, &image)?,
            &tail,
        )?;
        let prefill = count + 256;
        let mut caches = Vec::new();
        let output_ids = kernels::scratch_buffer(ctx, 7 * 4)?;
        for step in 0..7 {
            let offset = if step == 0 { 0 } else { prefill + step - 1 };
            for (i, w) in self.weights.language.iter().enumerate() {
                let x = norm::rms_bf16_rounded(ctx, &h, &w.norm1, 1e-6)?;
                let qkv = linear(ctx, &x, &w.qkv)?;
                let (q, k, v) = if step == 0 {
                    let p = rope::split_qkv_apply_bf16_rounded(
                        ctx, &qkv, None, 32, 32, 128, &self.weights.rope_frequencies, 0,
                    )?;
                    let k = cache::reserve_prefix_bf16(
                        ctx,
                        &p.k.reshape(vec![prefill, 4096])?,
                        prefill + 6,
                    )?
                    .reshape(vec![prefill + 6, 32, 128])?;
                    let v = cache::reserve_prefix_bf16(
                        ctx,
                        &p.v.reshape(vec![prefill, 4096])?,
                        prefill + 6,
                    )?
                    .reshape(vec![prefill + 6, 32, 128])?;
                    caches.push((k.clone(), v.clone()));
                    (p.q, k, v)
                } else {
                    let (k, v) = &caches[i];
                    let q = rope::apply_q_write_kv_bf16_rounded(
                        ctx, &qkv, None, 32, 32, 128, &self.weights.rope_frequencies, offset, k, v, offset,
                    )?;
                    (q, k.clone(), v.clone())
                };
                let x = attention::causal_mha_bf16_rounded(ctx, &q, &k, &v, prefill + step)?
                    .reshape(h.shape().dims().to_vec())?;
                h = binary(ctx, &h, &linear(ctx, &x, &w.proj)?, false)?;
                let x = norm::rms_bf16_rounded(ctx, &h, &w.norm2, 1e-6)?;
                let x = activation::swiglu_bf16_rounded(ctx, &linear(ctx, &x, &w.gate_up)?)?;
                h = binary(ctx, &h, &linear(ctx, &x, &w.down)?, false)?;
            }
            // Preserve prefill's matrix shape: reducing/projecting only the last
            // row changes cuBLAS and RMS reduction choices near BF16 ties.
            let normalized = norm::rms_bf16_rounded(ctx, &h, &self.weights.norm, 1e-6)?;
            let logits = linear(ctx, &normalized, &self.weights.head)?;
            let logits = elementwise::contiguous_rows(ctx, &logits, h.shape().dims()[0] - 1, 1)?;
            let id = output_ids.view(step * 4, 4).map_err(Error::Cuda)?;
            sampling::argmax_bf16_into(ctx, &logits, &id)?;
            if step < 6 {
                h = self.embed(ctx, &id, 1)?;
            }
        }
        let output = kernels::scratch_buffer(ctx, 7 * 4)?;
        embedding::lookup_into(
            ctx,
            DType::F32,
            &buffer(&self.weights.token_values)?,
            output_ids.address(),
            &output,
            1,
            7,
        )?;
        output
            .as_tensor(Shape::new(vec![1, 7]), DType::F32)
            .map_err(Error::Cuda)
    }
}
