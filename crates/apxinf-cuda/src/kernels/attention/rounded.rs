//! Materialized causal attention with BF16 score and probability boundaries.
use crate::workspace::output_buffer;
use crate::{ffi, CublasTranspose, CudaBuffer, CudaContext};
use apxinf_core::{DType, Device, Error, Result, Tensor};
/// `[tokens,heads,dim]` equal-head attention. QK, scaled QK, softmax, PV retain
/// the rounding boundaries of an unfused BF16 attention computation.
pub fn causal_mha_bf16_rounded(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let a = q.shape().dims();
    let b = k.shape().dims();
    if a.len() != 3
        || b.len() != 3
        || v.shape() != k.shape()
        || a.contains(&0)
        || b.contains(&0)
        || a[1..] != b[1..]
        || key_tokens < a[0]
        || key_tokens > b[0]
        || [q, k, v]
            .iter()
            .any(|t| t.dtype() != DType::BF16 || t.device() != Device::Cuda(ctx.device_id()))
    {
        return Err(Error::Other(
            "rounded causal MHA shape/dtype/device mismatch".into(),
        ));
    }
    let (rows, heads, dim) = (a[0], a[1], a[2]);
    let count = rows
        .checked_mul(heads)
        .and_then(|x| x.checked_mul(key_tokens))
        .ok_or_else(|| Error::Other("rounded attention size overflow".into()))?;
    if key_tokens > 2048 || count > i32::MAX as usize || heads * dim > i32::MAX as usize {
        return Err(Error::Other(
            "rounded attention dimensions exceed provider limits".into(),
        ));
    }
    let scores = output_buffer(ctx, count * 2)?;
    let prob = output_buffer(ctx, count * 2)?;
    let out = output_buffer(ctx, q.size_in_bytes())?;
    let qb = CudaBuffer::from_tensor(q).map_err(Error::Cuda)?;
    let kb = CudaBuffer::from_tensor(k).map_err(Error::Cuda)?;
    let vb = CudaBuffer::from_tensor(v).map_err(Error::Cuda)?;
    ctx.cublas()
        .batched_gemm_bf16_ex(
            CublasTranspose::None,
            CublasTranspose::Transpose,
            rows,
            key_tokens,
            dim,
            &qb,
            (heads * dim) as _,
            dim as _,
            &kb,
            (heads * dim) as _,
            dim as _,
            &scores,
            (heads * key_tokens) as _,
            key_tokens as _,
            heads as _,
        )
        .map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_scale_bf16(
            scores.ptr(),
            scores.ptr(),
            count as _,
            1. / (dim as f32).sqrt(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
        ffi::check_cuda(ffi::apxinf_attention_softmax_warp_bf16(
            scores.ptr(),
            prob.ptr(),
            key_tokens as _,
            (rows * heads) as _,
            (key_tokens - rows) as _,
            heads as _,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    ctx.cublas()
        .batched_gemm_bf16_ex(
            CublasTranspose::None,
            CublasTranspose::None,
            rows,
            dim,
            key_tokens,
            &prob,
            (heads * key_tokens) as _,
            key_tokens as _,
            &vb,
            (heads * dim) as _,
            dim as _,
            &out,
            (heads * dim) as _,
            dim as _,
            heads as _,
        )
        .map_err(Error::Cuda)?;
    Ok(out.into_tensor(q.shape().clone(), DType::BF16))
}
