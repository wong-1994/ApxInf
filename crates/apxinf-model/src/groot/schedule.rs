use apxinf_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowStep {
    pub index: usize,
    pub continuous_time: f32,
    pub bucket: u32,
    pub dt: f32,
}

pub fn four_step_schedule(num_buckets: usize) -> Result<[FlowStep; 4]> {
    if num_buckets != 1000 {
        return Err(Error::Other(format!("GR00T schedule requires 1000 buckets, got {num_buckets}")));
    }
    Ok(std::array::from_fn(|index| {
        let continuous_time = index as f32 / 4.0;
        FlowStep { index, continuous_time, bucket: (continuous_time * num_buckets as f32) as u32, dt: 0.25 }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_schedule_matches_reference_timesteps() {
        let steps = four_step_schedule(1000).unwrap();
        assert_eq!(steps.map(|step| step.bucket), [0, 250, 500, 750]);
        assert_eq!(steps.map(|step| step.dt), [0.25; 4]);
    }
}
