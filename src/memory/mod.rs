//! Memory Estimation
//!
//! Unified memory accounting for models, KV caches, and activations.
//! Supports single models, quantized models (AWQ/GPTQ), and speculative decoding pairs.

pub mod estimate;
pub mod speculative;
pub mod utils;

pub use estimate::{ActivationMemory, KvCacheMemory, MemoryEstimate, WeightMemory};
pub use speculative::SpeculativeMemoryEstimate;
pub use utils::{estimate_embedding_size, estimate_parameters, format_bytes};
