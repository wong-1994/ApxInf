//! Vanilla OpenVLA: dual vision encoder, projector and autoregressive Llama action tokens.
mod backend;
pub mod config;
mod load;
pub mod model;
mod model_runner;
mod weights;
pub(crate) fn register_builtin() {
    crate::registry::register("openvla_cuda", load::load_registered);
    crate::registry::register("openvla", load::load_registered);
}
