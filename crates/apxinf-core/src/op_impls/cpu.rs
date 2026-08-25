//! CPU backend implementation.

use crate::{Backend, Device, Error, Graph, Result, Tensor};
use crate::kv_cache::{CpuKVCache, KvCache};

/// CPU backend — all ops execute synchronously on the host.
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn concat_rows(&self, first: &Tensor, second: &Tensor) -> Result<Tensor> {
        let a = first.shape().dims();
        let b = second.shape().dims();
        if a.len() != 2 || b.len() != 2 || a[1] != b[1] || first.dtype() != second.dtype() {
            return Err(Error::Other("concat_rows requires matching 2-D tensors".into()));
        }
        let mut values = first.to_f32_vec()?;
        values.extend(second.to_f32_vec()?);
        Tensor::from_f32(vec![a[0] + b[0], a[1]], &values)
    }

    fn slice_2d(&self, input: &Tensor, row_start: usize, row_count: usize,
                col_start: usize, col_count: usize) -> Result<Tensor> {
        let dims = input.shape().dims();
        if dims.len() != 2 || row_start + row_count > dims[0] || col_start + col_count > dims[1] {
            return Err(Error::Other("slice_2d range is outside input".into()));
        }
        let source = input.to_f32_vec()?;
        let mut values = Vec::with_capacity(row_count * col_count);
        for row in row_start..row_start + row_count {
            values.extend_from_slice(&source[row * dims[1] + col_start..row * dims[1] + col_start + col_count]);
        }
        Tensor::from_f32(vec![row_count, col_count], &values)
    }

    fn relu(&self, input: &Tensor) -> Result<Tensor> {
        Tensor::from_f32(
            input.shape().clone(),
            &input.as_f32()?.iter().map(|value| value.max(0.0)).collect::<Vec<_>>(),
        )
    }

    fn add_bias(&self, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let dims = input.shape().dims();
        if dims.len() != 2 || bias.shape().dims() != [dims[1]] {
            return Err(Error::Other("add_bias requires input [rows,cols] and bias [cols]".into()));
        }
        let bias = bias.as_f32()?;
        let output = input.as_f32()?.iter().enumerate()
            .map(|(index, value)| value + bias[index % dims[1]])
            .collect::<Vec<_>>();
        Tensor::from_f32(input.shape().clone(), &output)
    }

    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let data = input.as_f32()?;
        let w = weight.as_f32()?;
        let dims = input.shape().dims();
        let seq_len = dims[0];
        let hidden = dims[1];

        let mut out = vec![0.0f32; data.len()];
        for s in 0..seq_len {
            let row_start = s * hidden;
            let row = &data[row_start..row_start + hidden];
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let rms = (mean_sq + eps).sqrt();
            for (i, &val) in row.iter().enumerate() {
                out[row_start + i] = (val / rms) * w[i];
            }
        }
        Tensor::from_f32(dims.to_vec(), &out)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        let data = x.as_f32()?;
        let out: Vec<f32> = data.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        Tensor::from_f32(x.shape().dims().to_vec(), &out)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let out: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a + b).collect();
        Tensor::from_f32(a.shape().dims().to_vec(), &out)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let out: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).collect();
        Tensor::from_f32(a.shape().dims().to_vec(), &out)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        let data = input.as_f32()?;
        let out: Vec<f32> = data.iter().map(|&v| v * factor).collect();
        Tensor::from_f32(input.shape().dims().to_vec(), &out)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.matmul_cpu(b)
    }

    fn rope(&self, input: &Tensor, n_heads: usize, head_dim: usize,
            theta: f32, pos_offset: u32) -> Result<Tensor> {
        let data = input.as_f32()?;
        let dims = input.shape().dims();
        let seq_len = if dims.len() == 2 { 1 } else { dims[0] };
        let half_dim = head_dim / 2;

        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        let mut out = vec![0.0f32; data.len()];
        for s in 0..seq_len {
            let pos = pos_offset as usize + s;
            for h in 0..n_heads {
                let base = s * n_heads * head_dim + h * head_dim;
                for i in 0..half_dim {
                    let angle = pos as f32 * freqs[i];
                    let cos_v = angle.cos();
                    let sin_v = angle.sin();
                    let x1 = data[base + i];
                    let x2 = data[base + half_dim + i];
                    out[base + i] = x1 * cos_v - x2 * sin_v;
                    out[base + half_dim + i] = x1 * sin_v + x2 * cos_v;
                }
            }
        }
        Tensor::from_f32(dims.to_vec(), &out)
    }

    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        let table_data = table.as_f32()?;
        let embed_dim = table.shape().dims()[1];
        let seq_len = ids.len();

        let mut out = vec![0.0f32; seq_len * embed_dim];
        for (i, &tid) in ids.iter().enumerate() {
            let src_offset = tid as usize * embed_dim;
            let dst_offset = i * embed_dim;
            out[dst_offset..dst_offset + embed_dim]
                .copy_from_slice(&table_data[src_offset..src_offset + embed_dim]);
        }
        Tensor::from_f32(vec![seq_len, embed_dim], &out)
    }

    fn cross_sdpa(&self, q: &Tensor, k: &Tensor, v: &Tensor,
                  q_len: usize, kv_len: usize, n_heads: usize,
                  head_dim: usize, key_mask: Option<&[u8]>, causal: bool) -> Result<Tensor> {
        let expected_q = [q_len, n_heads, head_dim];
        let k_dims = k.shape().dims();
        if k_dims.len() != 3 || k_dims[0] != kv_len || k_dims[2] != head_dim
            || n_heads % k_dims[1] != 0 {
            return Err(Error::Other("cross_sdpa requires Q heads divisible by KV heads".into()));
        }
        let n_kv_heads = k_dims[1];
        let expected_kv = [kv_len, n_kv_heads, head_dim];
        if q.shape().dims() != expected_q || v.shape().dims() != expected_kv {
            return Err(Error::Other("cross_sdpa received inconsistent shapes".into()));
        }
        if key_mask.is_some_and(|mask| mask.len() != kv_len) {
            return Err(Error::Other("cross_sdpa key mask has the wrong length".into()));
        }
        let (q_data, k_data, v_data) = (q.as_f32()?, k.as_f32()?, v.as_f32()?);
        let mut output = vec![0.0f32; q_len * n_heads * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();
        for qi in 0..q_len {
            for head in 0..n_heads {
                let kv_head = head / (n_heads / n_kv_heads);
                let mut scores = vec![f32::NEG_INFINITY; kv_len];
                for ki in 0..kv_len {
                    if key_mask.is_some_and(|mask| mask[ki] == 0) || (causal && ki > qi) { continue; }
                    let mut dot = 0.0;
                    for dim in 0..head_dim {
                        dot += q_data[(qi * n_heads + head) * head_dim + dim]
                            * k_data[(ki * n_kv_heads + kv_head) * head_dim + dim];
                    }
                    scores[ki] = dot * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if !max.is_finite() {
                    return Err(Error::Other("cross_sdpa mask excludes every key".into()));
                }
                let sum = scores.iter().map(|score| (*score - max).exp()).sum::<f32>();
                for ki in 0..kv_len {
                    let probability = (scores[ki] - max).exp() / sum;
                    for dim in 0..head_dim {
                        output[(qi * n_heads + head) * head_dim + dim] += probability
                            * v_data[(ki * n_kv_heads + kv_head) * head_dim + dim];
                    }
                }
            }
        }
        Tensor::from_f32(vec![q_len, n_heads * head_dim], &output)
    }

    fn sdpa_decode(&self, q: &Tensor, kv: &mut dyn KvCache,
                   layer_idx: usize, n_heads: usize, n_kv_heads: usize,
                   head_dim: usize, kv_len: usize, max_seq_len: usize) -> Result<Tensor> {
        let _ = max_seq_len;
        let q_data = q.as_f32()?;
        let cache = kv.as_any_mut().downcast_mut::<CpuKVCache>()
            .ok_or_else(|| Error::Other("expected CpuKVCache".into()))?;
        let (k_cached, v_cached) = cache.get_kv(layer_idx);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; n_heads * head_dim];

        for h in 0..n_heads {
            let kv_h = h * n_kv_heads / n_heads;
            let mut scores = vec![0.0f32; kv_len];
            for t in 0..kv_len {
                for d in 0..head_dim {
                    scores[t] += q_data[h * head_dim + d] * k_cached[kv_h][t][d];
                }
                scores[t] *= scale;
            }
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores.iter().map(|&s| (s - max_score).exp()).sum();
            for t in 0..kv_len {
                scores[t] = (scores[t] - max_score).exp() / exp_sum;
            }
            for t in 0..kv_len {
                for d in 0..head_dim {
                    output[h * head_dim + d] += scores[t] * v_cached[kv_h][t][d];
                }
            }
        }
        Tensor::from_f32(vec![1, n_heads * head_dim], &output)
    }

    fn sdpa_prefill(&self, q: &Tensor, kv: &mut dyn KvCache,
                    layer_idx: usize, n_heads: usize, n_kv_heads: usize,
                    head_dim: usize, kv_len: usize, max_seq_len: usize) -> Result<Tensor> {
        let _ = max_seq_len;
        let q_data = q.as_f32()?;
        let seq_len = q.shape().dims()[0];
        let cache = kv.as_any_mut().downcast_mut::<CpuKVCache>()
            .ok_or_else(|| Error::Other("expected CpuKVCache".into()))?;
        let (k_cached, v_cached) = cache.get_kv(layer_idx);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; seq_len * n_heads * head_dim];

        for s in 0..seq_len {
            for h in 0..n_heads {
                let kv_h = h * n_kv_heads / n_heads;
                let valid_len = kv_len.min(s + 1 + kv_len - seq_len);
                let mut scores = vec![0.0f32; kv_len];
                for t in 0..valid_len {
                    for d in 0..head_dim {
                        scores[t] += q_data[s * n_heads * head_dim + h * head_dim + d]
                            * k_cached[kv_h][t][d];
                    }
                    scores[t] *= scale;
                }
                let max_score = scores[..valid_len].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores[..valid_len].iter().map(|&s| (s - max_score).exp()).sum();
                for t in 0..valid_len {
                    scores[t] = (scores[t] - max_score).exp() / exp_sum;
                }
                for t in 0..valid_len {
                    for d in 0..head_dim {
                        output[s * n_heads * head_dim + h * head_dim + d]
                            += scores[t] * v_cached[kv_h][t][d];
                    }
                }
            }
        }
        Tensor::from_f32(vec![seq_len, n_heads * head_dim], &output)
    }

    fn create_kv_cache(&self, n_layers: usize, n_kv_heads: usize,
                       head_dim: usize, max_seq_len: usize) -> Box<dyn KvCache> {
        Box::new(CpuKVCache::new(n_layers, n_kv_heads, head_dim, max_seq_len))
    }

    fn kv_append(&self, kv: &mut dyn KvCache, layer_idx: usize,
                 k: &Tensor, v: &Tensor, append_len: usize) -> Result<()> {
        kv.append(layer_idx, k, v, append_len)
    }

    fn synchronize(&self) -> Result<()> { Ok(()) }

    fn begin_capture(&self) -> Result<()> { Ok(()) }

    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        Ok(Box::new(NoopGraph))
    }

    fn device(&self) -> Device { Device::Cpu }

    fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
}

/// No-op graph (CPU has nothing to capture).
struct NoopGraph;

impl Graph for NoopGraph {
    fn replay(&self) -> Result<()> { Ok(()) }
}
