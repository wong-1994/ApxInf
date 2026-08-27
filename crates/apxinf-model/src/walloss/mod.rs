//! Walloss vision-language-action runtime.

#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod bf16_executor;
#[cfg(feature = "cuda")]
mod bf16_runtime;
mod config;
mod geometry;
mod schedule;
mod weights;

pub use config::{WallossActionConfig, WallossConfig, WallossTextConfig, WallossVisionConfig};
pub use geometry::{multimodal_position_ids, VisionGeometry};
#[cfg(feature = "cuda")]
pub use geometry::DeviceVisionGeometry;
pub use schedule::{sinusoidal_time_embedding, solver_times};
pub use weights::{
    WallossActionWeights, WallossLayerWeights, WallossVisionBlockWeights, WallossVisionWeights,
    WallossWeights,
};

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("walloss-cuda", bf16_runtime::load_registered);
}
