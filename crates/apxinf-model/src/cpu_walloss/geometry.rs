//! Host-side WallOSS vision layout and multimodal position IDs.
//!
//! A host-only copy owned by `cpu_walloss`. Row orders match the CUDA
//! preprocessing output and checkpoint grid contract.

use apxinf_core::{Error, Result};

use super::WallossVisionConfig;

#[derive(Clone, Debug, PartialEq)]
pub struct VisionGeometry {
    /// Raw patch rows in window-major order.
    pub patch_order: Vec<u32>,
    /// `(height, width)` position IDs matching `patch_order`.
    pub position_ids: Vec<u32>,
    /// Packed local-window boundaries in raw patch tokens.
    pub window_offsets: Vec<u32>,
    /// Packed full-attention boundaries, one segment per temporal frame.
    pub full_offsets: Vec<u32>,
    /// Restores merged vision tokens to image-major order.
    pub reverse_indices: Vec<u32>,
    pub max_window_tokens: usize,
    pub max_full_tokens: usize,
    /// Packed RGB layout when every view is one still frame of the same patch
    /// grid; `None` means the grids can only be served as patches.
    pub rgb_views: Option<RgbViews>,
}

/// Uniform still-frame RGB layout: `views` images of `grid_h x grid_w` patches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RgbViews {
    pub views: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl VisionGeometry {
    pub fn new(config: &WallossVisionConfig, grids: &[[usize; 3]]) -> Result<Self> {
        if grids.is_empty() {
            return Err(Error::Other("walloss vision grid cannot be empty".into()));
        }
        let merge = config.spatial_merge_size;
        let merge_unit = merge * merge;
        let merger_window = config.window_size / merge / config.patch_size;
        if merge == 0 || merger_window == 0 {
            return Err(Error::Other(
                "walloss vision merge geometry is invalid".into(),
            ));
        }

        let mut merged_order = Vec::new();
        let mut base_positions = Vec::new();
        let mut window_offsets = vec![0u32];
        let mut full_offsets = vec![0u32];
        let mut merged_base = 0usize;

        for &[t, h, w] in grids {
            if t == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
                return Err(Error::Other(format!(
                    "walloss vision grid [{t}, {h}, {w}] is incompatible with merge size {merge}"
                )));
            }
            let llm_h = h / merge;
            let llm_w = w / merge;
            for _ in 0..t {
                for group_h in 0..llm_h {
                    for group_w in 0..llm_w {
                        for inner_h in 0..merge {
                            for inner_w in 0..merge {
                                base_positions.push((
                                    (group_h * merge + inner_h) as u32,
                                    (group_w * merge + inner_w) as u32,
                                ));
                            }
                        }
                    }
                }
                let next = full_offsets.last().copied().unwrap() as usize + h * w;
                full_offsets.push(
                    u32::try_from(next)
                        .map_err(|_| Error::Other("walloss full-attention offsets overflow u32".into()))?,
                );
            }

            for frame in 0..t {
                for window_h in (0..llm_h).step_by(merger_window) {
                    for window_w in (0..llm_w).step_by(merger_window) {
                        let before = merged_order.len();
                        for local_h in 0..merger_window.min(llm_h - window_h) {
                            for local_w in 0..merger_window.min(llm_w - window_w) {
                                let index = merged_base
                                    + frame * llm_h * llm_w
                                    + (window_h + local_h) * llm_w
                                    + window_w
                                    + local_w;
                                merged_order.push(index);
                            }
                        }
                        let tokens = (merged_order.len() - before) * merge_unit;
                        let next = window_offsets.last().copied().unwrap() as usize + tokens;
                        window_offsets.push(u32::try_from(next).map_err(|_| {
                            Error::Other("walloss window offsets overflow u32".into())
                        })?);
                    }
                }
            }
            merged_base += t * llm_h * llm_w;
        }

        let mut patch_order = Vec::with_capacity(merged_order.len() * merge_unit);
        let mut position_ids = Vec::with_capacity(merged_order.len() * merge_unit * 2);
        for &group in &merged_order {
            for inner in 0..merge_unit {
                let patch = group * merge_unit + inner;
                patch_order.push(
                    u32::try_from(patch)
                        .map_err(|_| Error::Other("walloss patch index overflow u32".into()))?,
                );
                let (height, width) = base_positions[patch];
                position_ids.extend([height, width]);
            }
        }
        let mut reverse_indices = vec![0u32; merged_order.len()];
        for (window_position, &original_position) in merged_order.iter().enumerate() {
            reverse_indices[original_position] = u32::try_from(window_position)
                .map_err(|_| Error::Other("walloss reverse index overflow u32".into()))?;
        }
        let max_window_tokens = offset_max_delta(&window_offsets);
        let max_full_tokens = offset_max_delta(&full_offsets);
        let [_, grid_h, grid_w] = grids[0];
        let rgb_views = grids
            .iter()
            .all(|&[t, h, w]| t == 1 && h == grid_h && w == grid_w)
            .then_some(RgbViews {
                views: grids.len(),
                grid_h,
                grid_w,
            });
        Ok(Self {
            patch_order,
            position_ids,
            window_offsets,
            full_offsets,
            reverse_indices,
            max_window_tokens,
            max_full_tokens,
            rgb_views,
        })
    }
}

fn offset_max_delta(offsets: &[u32]) -> usize {
    offsets
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as usize)
        .max()
        .unwrap_or(0)
}

/// Build token-major multimodal position IDs for one unpadded sequence.
pub fn multimodal_position_ids(
    token_ids: &[u32],
    grids: &[[usize; 3]],
    image_token_id: u32,
    spatial_merge: usize,
) -> Result<Vec<u32>> {
    let mut axes = [Vec::new(), Vec::new(), Vec::new()];
    let mut token_offset = 0usize;
    let mut next_position = 0u32;
    for &[t, h, w] in grids {
        let image_start = token_ids[token_offset..]
            .iter()
            .position(|&token| token == image_token_id)
            .map(|index| token_offset + index)
            .ok_or_else(|| {
                Error::Other(
                    "walloss prompt has fewer image-token runs than configured grids".into(),
                )
            })?;
        for position in token_offset..image_start {
            let value = next_position + (position - token_offset) as u32;
            for axis in &mut axes {
                axis.push(value);
            }
        }
        next_position += (image_start - token_offset) as u32;
        let llm_h = h / spatial_merge;
        let llm_w = w / spatial_merge;
        let image_tokens = t * llm_h * llm_w;
        if token_ids.get(image_start..image_start + image_tokens).is_none()
            || token_ids[image_start..image_start + image_tokens]
                .iter()
                .any(|&token| token != image_token_id)
        {
            return Err(Error::Other(format!(
                "walloss prompt image run does not contain {image_tokens} tokens"
            )));
        }
        for time in 0..t {
            for height in 0..llm_h {
                for width in 0..llm_w {
                    axes[0].push(next_position + time as u32);
                    axes[1].push(next_position + height as u32);
                    axes[2].push(next_position + width as u32);
                }
            }
        }
        next_position += t.max(llm_h).max(llm_w) as u32;
        token_offset = image_start + image_tokens;
    }
    for position in token_offset..token_ids.len() {
        let value = next_position + (position - token_offset) as u32;
        for axis in &mut axes {
            axis.push(value);
        }
    }
    let mut output = Vec::with_capacity(token_ids.len() * 3);
    for token in 0..token_ids.len() {
        output.extend([axes[0][token], axes[1][token], axes[2][token]]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> WallossVisionConfig {
        WallossVisionConfig {
            depth: 4,
            hidden_size: 16,
            intermediate_size: 32,
            num_heads: 2,
            patch_size: 2,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            out_hidden_size: 8,
            window_size: 8,
            full_attention_blocks: vec![3],
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        }
    }

    #[test]
    fn geometry_restores_image_major_order() {
        let geometry = VisionGeometry::new(&config(), &[[1, 4, 6]]).unwrap();
        assert_eq!(geometry.patch_order.len(), 24);
        assert_eq!(geometry.reverse_indices.len(), 6);
        assert_eq!(geometry.max_full_tokens, 24);
        assert_eq!(geometry.window_offsets.len(), 3);
        // Reverse indices are a permutation of the merged positions.
        let mut sorted = geometry.reverse_indices.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..6).collect::<Vec<_>>());
    }

    #[test]
    fn rgb_views_require_uniform_still_grids() {
        let geometry = VisionGeometry::new(&config(), &[[1, 4, 6], [1, 4, 6]]).unwrap();
        assert_eq!(
            geometry.rgb_views,
            Some(RgbViews {
                views: 2,
                grid_h: 4,
                grid_w: 6
            })
        );
        let mixed = VisionGeometry::new(&config(), &[[1, 4, 6], [2, 4, 6]]).unwrap();
        assert_eq!(mixed.rgb_views, None);
    }

    #[test]
    fn multimodal_positions_are_token_major_and_axis_specific() {
        let image_token = 9;
        let mut tokens = vec![1u32];
        tokens.extend(std::iter::repeat(image_token).take(4));
        tokens.extend([2u32, 3]);
        let positions = multimodal_position_ids(&tokens, &[[1, 4, 4]], image_token, 2).unwrap();
        assert_eq!(positions.len(), tokens.len() * 3);
        // Leading text: all axes share the running position.
        assert_eq!(&positions[0..3], &[0, 0, 0]);
        // Image tokens: temporal stays fixed, height/width offset the image
        // base position while walking the 2x2 grid.
        assert_eq!(&positions[3..9], &[1, 1, 1, 1, 1, 2]);
        assert_eq!(&positions[9..15], &[1, 2, 1, 1, 2, 2]);
        // Trailing text resumes after max(t, h, w).
        assert_eq!(&positions[15..21], &[3, 3, 3, 4, 4, 4]);
    }

    #[test]
    fn rejects_mismatched_image_runs() {
        let tokens = [1u32, 9, 2];
        assert!(multimodal_position_ids(&tokens, &[[1, 4, 4]], 9, 2).is_err());
    }
}
