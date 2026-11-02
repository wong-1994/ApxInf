//! Multimodal rotary embeddings on host tensors.
//!
//! Frequency layout follows the maintained CUDA operators: half-split pairs,
//! striped `[t, h, w]` axes for language mRoPE, and height/width halves for
//! the packed vision tower.

use apxinf_core::{Error, Result, Tensor};

fn validate(input: &Tensor, n_heads: usize, head_dim: usize) -> Result<(usize, usize)> {
    let dims = input.shape().dims();
    if dims.len() != 3
        || dims[0] == 0
        || dims[1] != n_heads
        || dims[2] != head_dim
        || n_heads == 0
        || head_dim == 0
        || head_dim % 2 != 0
    {
        return Err(Error::Other(
            "CPU RoPE: input must be [seq, n_heads, even head_dim]".into(),
        ));
    }
    Ok((dims[0], dims[1]))
}

fn rotate(out: &mut [f32], base: usize, pair: usize, half: usize, sin: f32, cos: f32) {
    let first = out[base + pair];
    let second = out[base + half + pair];
    out[base + pair] = first * cos - second * sin;
    out[base + half + pair] = second * cos + first * sin;
}

/// Striped temporal/height/width RoPE. `pos_ids` holds `(t, h, w)` per token.
pub(crate) fn mrope(
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    sections: [usize; 3],
    pos_ids: &[u32],
) -> Result<Tensor> {
    let (seq, heads) = validate(input, n_heads, head_dim)?;
    let half = head_dim / 2;
    if pos_ids.len() != seq * 3
        || sections.iter().sum::<usize>() != half
        || !(theta.is_finite() && theta > 0.0)
    {
        return Err(Error::Other(
            "CPU mRoPE: position IDs or sections do not match the head geometry".into(),
        ));
    }
    let mut out = input.to_f32_vec()?;
    for token in 0..seq {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for pair in 0..half {
                let remainder = pair % 3;
                let axis = if remainder == 1 && pair < sections[1] * 3 {
                    1
                } else if remainder == 2 && pair < sections[2] * 3 {
                    2
                } else {
                    0
                };
                let position = pos_ids[token * 3 + axis] as f32;
                let frequency = theta.powf(-2.0 * pair as f32 / head_dim as f32);
                let (sin, cos) = (position * frequency).sin_cos();
                rotate(&mut out, base, pair, half, sin, cos);
            }
        }
    }
    Tensor::from_f32(input.shape().dims().to_vec(), &out)
}

/// Height/width axial RoPE. `pos_ids` holds `(h, w)` per token.
pub(crate) fn vision_2d(
    input: &Tensor,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    pos_ids: &[u32],
) -> Result<Tensor> {
    let (seq, heads) = validate(input, n_heads, head_dim)?;
    let half = head_dim / 2;
    let quarter = half / 2;
    if head_dim % 4 != 0
        || pos_ids.len() != seq * 2
        || !(theta.is_finite() && theta > 0.0)
    {
        return Err(Error::Other(
            "CPU vision RoPE: position IDs do not match the axial geometry".into(),
        ));
    }
    let mut out = input.to_f32_vec()?;
    for token in 0..seq {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for pair in 0..half {
                let (axis, pair_in_axis) = if pair < quarter {
                    (0, pair)
                } else {
                    (1, pair - quarter)
                };
                let position = pos_ids[token * 2 + axis] as f32;
                let frequency =
                    theta.powf(-2.0 * pair_in_axis as f32 / half as f32);
                let (sin, cos) = (position * frequency).sin_cos();
                rotate(&mut out, base, pair, half, sin, cos);
            }
        }
    }
    Tensor::from_f32(input.shape().dims().to_vec(), &out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_positions_leave_values_unchanged() {
        let input = Tensor::from_f32(vec![2, 2, 4], &(0..16).map(|x| x as f32).collect::<Vec<_>>())
            .unwrap();
        let mrope_out = mrope(&input, 2, 4, 10_000.0, [1, 1, 0], &[0; 6]).unwrap();
        let vision_out = vision_2d(&input, 2, 4, 10_000.0, &[0; 4]).unwrap();
        assert_eq!(mrope_out.to_f32_vec().unwrap(), input.to_f32_vec().unwrap());
        assert_eq!(vision_out.to_f32_vec().unwrap(), input.to_f32_vec().unwrap());
    }

    #[test]
    fn mrope_matches_scalar_angles_per_axis() {
        let (seq, heads, dim) = (2usize, 1usize, 8usize);
        let half = dim / 2;
        let data: Vec<f32> = (0..seq * heads * dim).map(|x| (x as f32 * 0.31).sin()).collect();
        let positions = [0u32, 1, 2, 3, 4, 5];
        let sections = [1usize, 1, 2];
        let out = mrope(
            &Tensor::from_f32(vec![seq, heads, dim], &data).unwrap(),
            heads,
            dim,
            10_000.0,
            sections,
            &positions,
        )
        .unwrap()
        .to_f32_vec()
        .unwrap();
        for token in 0..seq {
            for pair in 0..half {
                let remainder = pair % 3;
                let axis = if remainder == 1 && pair < sections[1] * 3 {
                    1
                } else if remainder == 2 && pair < sections[2] * 3 {
                    2
                } else {
                    0
                };
                let angle = positions[token * 3 + axis] as f64
                    * (10_000f64).powf(-2.0 * pair as f64 / dim as f64);
                let (sin, cos) = (angle as f32).sin_cos();
                let base = token * dim;
                let expected_first = data[base + pair] * cos - data[base + half + pair] * sin;
                let expected_second = data[base + half + pair] * cos + data[base + pair] * sin;
                assert!((out[base + pair] - expected_first).abs() < 2e-6);
                assert!((out[base + half + pair] - expected_second).abs() < 2e-6);
            }
        }
    }

    #[test]
    fn vision_rope_uses_half_dim_frequencies_and_restarts_per_axis() {
        let (seq, heads, dim) = (1usize, 1usize, 8usize);
        let half = dim / 2;
        let quarter = half / 2;
        let data: Vec<f32> = (0..dim).map(|x| x as f32 * 0.1).collect();
        let positions = [3u32, 5];
        let out = vision_2d(
            &Tensor::from_f32(vec![seq, heads, dim], &data).unwrap(),
            heads,
            dim,
            10_000.0,
            &positions,
        )
        .unwrap()
        .to_f32_vec()
        .unwrap();
        for pair in 0..half {
            let (axis, pair_in_axis) = if pair < quarter { (0, pair) } else { (1, pair - quarter) };
            let angle = positions[axis] as f64
                * (10_000f64).powf(-2.0 * pair_in_axis as f64 / half as f64);
            let (sin, cos) = (angle as f32).sin_cos();
            let expected_first = data[pair] * cos - data[half + pair] * sin;
            let expected_second = data[half + pair] * cos + data[pair] * sin;
            assert!((out[pair] - expected_first).abs() < 2e-6);
            assert!((out[half + pair] - expected_second).abs() < 2e-6);
        }
    }

    #[test]
    fn rejects_mismatched_positions() {
        let input = Tensor::zeros(vec![2, 1, 4], apxinf_core::DType::F32);
        assert!(mrope(&input, 1, 4, 10_000.0, [2, 0, 0], &[0; 5]).is_err());
        assert!(vision_2d(&input, 1, 4, 10_000.0, &[0; 3]).is_err());
    }
}
