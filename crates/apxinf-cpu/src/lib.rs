//! Synchronous CPU backend with optional OpenBLAS matrix multiplication.
mod backend;
mod dense;
mod gemm;
mod kv_cache;
mod rope;
mod sampling;

pub use backend::CpuBackend;
pub use dense::CpuAttentionMask;
pub use kv_cache::CpuKVCache;
