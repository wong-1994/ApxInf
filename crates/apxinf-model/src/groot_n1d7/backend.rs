use apxinf_core::{Backend, Error, Result, Shape, Storage, Tensor};

use super::device_weights::DeviceLinear;

pub(crate) use crate::accelerator::cuda::{kernels, RuntimeBackend};

pub fn linear(b: &dyn Backend, x: &Tensor, w: &DeviceLinear) -> Result<Tensor> {
    let projected = b.matmul(x, &w.weight)?;
    b.add_bias(&projected, &w.bias)
}

pub fn row_view(tensor: &Tensor, start: usize, rows: usize) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 || start.checked_add(rows).is_none_or(|end| end > dims[0]) {
        return Err(Error::Other(format!(
            "row view [{start}..{}] out of {:?}",
            start + rows,
            dims
        )));
    }
    let cols = dims[1];
    match tensor.storage() {
        Storage::Gpu { device, handle } => {
            let byte_offset = start * cols * tensor.dtype().size_in_bytes();
            let byte_len = rows * cols * tensor.dtype().size_in_bytes();
            Ok(Tensor::from_raw_parts(
                Shape::new(vec![rows, cols]),
                tensor.dtype(),
                *device,
                Storage::Gpu {
                    device: *device,
                    handle: apxinf_core::storage::GpuStorageHandle {
                        ptr: handle.ptr + byte_offset,
                        len: byte_len,
                        _prevent_leak: handle._prevent_leak.clone(),
                    },
                },
            ))
        }
        Storage::Cpu(_) => Err(Error::Other("GR00T row_view requires device tensor".into())),
    }
}

pub fn relu_roundtrip(b: &dyn Backend, x: &Tensor) -> Result<Tensor> {
    let cpu = b.to_cpu(x)?;
    let shape = cpu.shape().dims().to_vec();
    let values = cpu
        .to_f32_vec()?
        .into_iter()
        .map(|v| v.max(0.0))
        .collect::<Vec<_>>();
    let relu = Tensor::from_bf16(
        shape,
        &values
            .into_iter()
            .map(half::bf16::from_f32)
            .collect::<Vec<_>>(),
    )?;
    b.to_device(&relu)
}
