pub mod config;
pub mod distributed_cache;
pub mod pipeline_parallel;
pub mod tensor_parallel;
/// Multi-GPU inference support (M3.6)
///
/// Provides tensor parallelism and pipeline parallelism for running large models
/// (70B+ parameters) across multiple CUDA devices.
pub mod topology;

pub use config::{CommunicationBackend, LoadBalancingStrategy, MultiGPUConfig, ParallelismMode};
pub use distributed_cache::{CacheSyncStrategy, DistributedCacheManager};
pub use pipeline_parallel::{PipelineScheduler, PipelineStage, PipelineStrategy};
pub use tensor_parallel::{ShardedLinear, ShardingStrategy, TensorShard};
pub use topology::{DeviceTopology, InterconnectTopology, LinkType};
