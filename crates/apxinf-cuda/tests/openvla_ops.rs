//! Numerical boundary regressions found by the native OpenVLA port.
use apxinf_core::{Backend, Result, Tensor};
use apxinf_cuda::{kernels::sampling, CudaBackend, CudaBuffer};
use half::bf16;

#[test]
#[ignore = "requires a CUDA device"]
fn bf16_argmax_uses_first_index_for_ties_and_nans() -> Result<()> {
    let backend = CudaBackend::new(0)?;
    let output = CudaBuffer::alloc(4, 0).map_err(apxinf_core::Error::Cuda)?;
    let mut cases = Vec::new();
    let mut across_warps = vec![bf16::from_f32(-1.0); 1024];
    across_warps[258] = bf16::from_f32(12.5);
    across_warps[293] = bf16::from_f32(12.5);
    cases.push((across_warps, 258));
    cases.push((vec![bf16::NEG_ZERO, bf16::ZERO], 0));
    cases.push((vec![bf16::NEG_INFINITY; 513], 0));
    cases.push((vec![bf16::ONE, bf16::NAN, bf16::NAN], 1));
    for (values, expected) in cases {
        let logits = backend.to_device(&Tensor::from_bf16(vec![values.len()], &values)?)?;
        sampling::argmax_bf16_into(backend.context(), &logits, &output)?;
        let mut bytes = [0; 4];
        output
            .copy_to_host(&mut bytes)
            .map_err(apxinf_core::Error::Cuda)?;
        assert_eq!(u32::from_ne_bytes(bytes), expected);
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn bf16_rope_preserves_reference_frequency_near_full_rotation() -> Result<()> {
    let backend = CudaBackend::new(0)?;
    // Real LIBERO observation: position 149, pair 22 is near 2*pi.
    // A one-ULP inverse-frequency change alters the BF16 query and action tokens.
    let mut values = vec![bf16::ZERO; 384];
    values[22] = bf16::from_f32(0.0283203125);
    values[86] = bf16::from_f32(-0.6640625);
    values[150] = values[22];
    values[214] = values[86];
    let qkv = backend.to_device(&Tensor::from_bf16(vec![1, 384], &values)?)?;
    let frequencies = (0..64)
        .map(|i| (10000_f64.powf(i as f64 / 64.0) as f32).recip())
        .collect::<Vec<_>>();
    let frequencies = backend.to_device(&Tensor::from_f32(vec![64], &frequencies)?)?;
    let output = apxinf_cuda::kernels::rope::split_qkv_apply_bf16_rounded(
        backend.context(), &qkv, None, 1, 1, 128, &frequencies, 149,
    )?;
    assert_eq!(backend.to_cpu(&output.q)?.to_f32_vec()?[22], 0.0283203125);
    assert_eq!(backend.to_cpu(&output.k)?.to_f32_vec()?[22], 0.0283203125);
    let k = backend.to_device(&Tensor::from_bf16(vec![3, 1, 128], &vec![bf16::ZERO; 384])?)?;
    let v = backend.to_device(&Tensor::from_bf16(vec![3, 1, 128], &vec![bf16::ZERO; 384])?)?;
    let query = apxinf_cuda::kernels::rope::apply_q_write_kv_bf16_rounded(
        backend.context(), &qkv, None, 1, 1, 128, &frequencies, 149, &k, &v, 1,
    )?;
    assert_eq!(backend.to_cpu(&query)?.to_f32_vec()?, backend.to_cpu(&output.q)?.to_f32_vec()?);
    assert_eq!(backend.to_cpu(&k)?.to_f32_vec()?[150], 0.0283203125);
    let invalid = backend.to_device(&Tensor::from_f32(vec![63], &vec![1.; 63])?)?;
    assert!(apxinf_cuda::kernels::rope::split_qkv_apply_bf16_rounded(
        backend.context(), &qkv, None, 1, 1, 128, &invalid, 149,
    ).is_err());
    Ok(())
}
