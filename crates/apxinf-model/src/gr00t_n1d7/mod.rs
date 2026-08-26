//! NVIDIA GR00T N1.7 VLA runtime.

mod backbone;
mod backend;
mod config;
mod executor;
mod math;
mod vla_runtime;
mod weights;

pub use vla_runtime::Gr00tN1d7VlaRuntime;

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("Gr00tN1d7-cuda", vla_runtime::load_registered);
    crate::registry::register("gr00t_n1d7-cuda", vla_runtime::load_registered);
    crate::registry::register("groot-cuda", vla_runtime::load_registered);
}
pub use config::Gr00tN1d7Config;
