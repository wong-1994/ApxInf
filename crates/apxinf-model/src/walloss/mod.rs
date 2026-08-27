//! Walloss vision-language-action runtime.

#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod bf16_executor;
#[cfg(feature = "cuda")]
mod bf16_runtime;
mod config;
#[cfg(feature = "cuda")]
mod fp8_executor;
#[cfg(feature = "cuda")]
mod fp8_weights;
mod geometry;
mod schedule;
mod weights;

pub use config::{WallossActionConfig, WallossConfig, WallossTextConfig, WallossVisionConfig};
#[cfg(feature = "cuda")]
pub use fp8_weights::{
    WallossActivationScales, WallossFp8LayerWeights, WallossFp8VisionBlockWeights,
    WallossFp8VisionWeights, WallossFp8Weights,
};
#[cfg(feature = "cuda")]
pub use geometry::DeviceVisionGeometry;
pub use geometry::{multimodal_position_ids, VisionGeometry};
pub use schedule::{sinusoidal_time_embedding, solver_times};
pub use weights::{
    WallossActionWeights, WallossLayerWeights, WallossVisionBlockWeights, WallossVisionWeights,
    WallossWeights,
};

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("walloss-cuda", bf16_runtime::load_registered);
}
