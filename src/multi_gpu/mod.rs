pub mod config;
/// Multi-GPU inference support (M3.6)
///
/// Provides tensor parallelism and pipeline parallelism for running large models
/// (70B+ parameters) across multiple CUDA devices.
///
/// `distributed_cache`/`pipeline_parallel`/`tensor_parallel` removed 2026-10-22
/// (CireSnave: "Remove all dependencies on Candle and Candlelight..."): all
/// three were candlelight-typed (`Tensor`/`Device`) with ZERO external
/// callers even before removal — never wired into `ParallelModelManager`'s
/// multi-GPU bail sites or anything else. See
/// `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`.
pub mod topology;

pub use config::{CommunicationBackend, LoadBalancingStrategy, MultiGPUConfig, ParallelismMode};
pub use topology::{DeviceTopology, InterconnectTopology, LinkType};
