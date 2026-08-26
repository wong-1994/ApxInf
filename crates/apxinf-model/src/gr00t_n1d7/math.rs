//! Framework-independent GR00T N1.7 schedule and sinusoidal embeddings.

use apxinf_core::{Error, Result};

/// Diffusers `Timesteps(256, flip_sin_to_cos=true, downscale_freq_shift=1)`.
pub(crate) fn timestep_embedding(timestep: u32, channels: usize) -> Result<Vec<f32>> {
    if channels == 0 || channels % 2 != 0 {
        return Err(Error::Other(
            "timestep embedding channels must be positive and even".into(),
        ));
    }
    let half = channels / 2;
    let denominator = (half - 1).max(1) as f32;
    let mut sin = Vec::with_capacity(half);
    let mut cos = Vec::with_capacity(half);
    for index in 0..half {
        let exponent = -(10_000.0_f32.ln()) * index as f32 / denominator;
        let value = timestep as f32 * exponent.exp();
        sin.push(value.sin());
        cos.push(value.cos());
    }
    // flip_sin_to_cos=True
    cos.extend(sin);
    Ok(cos)
}

/// Action-encoder sinusoid. The reference divides by `half_dim` rather than
/// `half_dim - 1` and concatenates sin before cos.
pub(crate) fn action_time_embedding(
    timestep: u32,
    horizon: usize,
    width: usize,
) -> Result<Vec<f32>> {
    if width == 0 || width % 2 != 0 {
        return Err(Error::Other(
            "action time embedding width must be positive and even".into(),
        ));
    }
    let half = width / 2;
    let mut one = Vec::with_capacity(width);
    for index in 0..half {
        let frequency = (-(10_000.0_f32.ln()) * index as f32 / half as f32).exp();
        one.push((timestep as f32 * frequency).sin());
    }
    for index in 0..half {
        let frequency = (-(10_000.0_f32.ln()) * index as f32 / half as f32).exp();
        one.push((timestep as f32 * frequency).cos());
    }
    let mut output = Vec::with_capacity(horizon * width);
    for _ in 0..horizon {
        output.extend_from_slice(&one);
    }
    Ok(output)
}

pub(crate) fn euler_timesteps(steps: usize, buckets: usize) -> Result<Vec<u32>> {
    if steps == 0 || buckets == 0 {
        return Err(Error::Other(
            "Euler steps and timestep buckets must be non-zero".into(),
        ));
    }
    Ok((0..steps)
        .map(|step| ((step as f32 / steps as f32) * buckets as f32) as u32)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_four_step_schedule() {
        assert_eq!(euler_timesteps(4, 1000).unwrap(), [0, 250, 500, 750]);
    }
    #[test]
    fn zero_timestep_layout_matches_reference() {
        let t = timestep_embedding(0, 8).unwrap();
        assert_eq!(&t[..4], &[1.0; 4]);
        assert_eq!(&t[4..], &[0.0; 4]);
        let a = action_time_embedding(0, 2, 8).unwrap();
        assert_eq!(&a[..4], &[0.0; 4]);
        assert_eq!(&a[4..8], &[1.0; 4]);
        assert_eq!(&a[..8], &a[8..]);
    }
}
