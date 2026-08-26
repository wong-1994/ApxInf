//! NVIDIA GR00T N1.7 vision-language-action model.
//!
//! GR00T owns its Qwen3-VL backbone adaptation, multi-embodiment projectors,
//! AlternateVLDiT schedule, and checkpoint mapping in this isolated module.

mod backbone;
#[cfg(feature = "cuda")]
mod backend;
mod config;
mod device_weights;
#[cfg(feature = "cuda")]
mod executor;
#[cfg(feature = "cuda")]
mod vla_runtime;
mod weights;

pub use backbone::{GrootN1d7Backbone, GrootN1d7BackboneOutput};
pub use config::{GrootN1d7Config, GrootN1d7DiffusionConfig, GrootN1d7VlSelfAttentionConfig};
pub use weights::{
    CategoryLinearWeights, CategoryMlpWeights, GrootN1d7ActionWeights, GrootN1d7DiTBlockWeights,
    GrootN1d7SelfAttentionBlockWeights, LinearWeights,
};

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("Gr00tN1d7", vla_runtime::load_registered);
    crate::registry::register("gr00t_n1d7", vla_runtime::load_registered);
}
