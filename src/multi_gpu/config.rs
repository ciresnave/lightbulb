use crate::multi_gpu::topology::DeviceTopology;
use anyhow::Result;

/// Parallelism strategy for multi-GPU inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismMode {
    /// Single GPU (no parallelism)
    Single,

    /// Tensor parallelism: split weights column/row-wise across GPUs
    TensorParallel {
        /// Number of GPUs (world size)
        world_size: usize,
    },

    /// Pipeline parallelism: split layers across GPUs
    PipelineParallel {
        /// Number of pipeline stages (GPUs)
        num_stages: usize,

        /// Micro-batch size for pipeline scheduling
        micro_batch_size: usize,
    },

    /// Hybrid: tensor parallelism within stages, pipeline between stages
    Hybrid {
        /// Tensor parallel world size per stage
        tensor_world_size: usize,

        /// Number of pipeline stages
        pipeline_stages: usize,

        /// Micro-batch size
        micro_batch_size: usize,
    },
}

impl ParallelismMode {
    /// Total number of GPUs required
    pub fn num_gpus(&self) -> usize {
        match self {
            Self::Single => 1,
            Self::TensorParallel { world_size } => *world_size,
            Self::PipelineParallel { num_stages, .. } => *num_stages,
            Self::Hybrid {
                tensor_world_size,
                pipeline_stages,
                ..
            } => tensor_world_size * pipeline_stages,
        }
    }

    /// Communication pattern description
    pub fn communication_pattern(&self) -> &str {
        match self {
            Self::Single => "None",
            Self::TensorParallel { .. } => "All-reduce per layer (high frequency)",
            Self::PipelineParallel { .. } => "Point-to-point between stages (low frequency)",
            Self::Hybrid { .. } => "All-reduce within stages + P2P between stages",
        }
    }
}

/// Multi-GPU inference configuration
#[derive(Debug, Clone)]
pub struct MultiGPUConfig {
    /// Parallelism mode
    pub mode: ParallelismMode,

    /// Device topology (discovered automatically)
    pub topology: DeviceTopology,

    /// Sharding strategy for tensor parallelism
    pub sharding_strategy: crate::multi_gpu::tensor_parallel::ShardingStrategy,

    /// Communication backend (NCCL, custom)
    pub communication_backend: CommunicationBackend,

    /// Enable KV cache distribution across GPUs
    pub distributed_kv_cache: bool,

    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

impl MultiGPUConfig {
    /// Create automatic configuration based on model size
    ///
    /// Discovers available GPUs and selects appropriate parallelism strategy.
    ///
    /// # Example
    /// ```rust,ignore
    /// // 70B model (~140GB in fp16)
    /// let config = MultiGPUConfig::auto(140 * 1024 * 1024 * 1024)?;
    /// // Automatically selects tensor or pipeline parallelism
    /// ```
    pub fn auto(model_size_bytes: usize) -> Result<Self> {
        let topology = DeviceTopology::discover()?;
        let mode = topology.recommend_strategy(model_size_bytes);

        Ok(Self {
            mode,
            topology,
            sharding_strategy: crate::multi_gpu::tensor_parallel::ShardingStrategy::ColumnWise,
            communication_backend: CommunicationBackend::Candle,
            distributed_kv_cache: true,
            load_balancing: LoadBalancingStrategy::MemoryAware,
        })
    }

    /// Create manual configuration with specific parallelism mode
    ///
    /// # Example
    /// ```rust,ignore
    /// let mode = ParallelismMode::TensorParallel { world_size: 2 };
    /// let config = MultiGPUConfig::manual(mode, 2)?;
    /// ```
    pub fn manual(mode: ParallelismMode, num_devices: usize) -> Result<Self> {
        let topology = DeviceTopology::discover()?;

        if topology.num_gpus() < num_devices {
            anyhow::bail!(
                "Requested {} GPUs but only {} available",
                num_devices,
                topology.num_gpus()
            );
        }

        if mode.num_gpus() != num_devices {
            anyhow::bail!(
                "Parallelism mode requires {} GPUs but {} requested",
                mode.num_gpus(),
                num_devices
            );
        }

        Ok(Self {
            mode,
            topology,
            sharding_strategy: crate::multi_gpu::tensor_parallel::ShardingStrategy::ColumnWise,
            communication_backend: CommunicationBackend::Candle,
            distributed_kv_cache: true,
            load_balancing: LoadBalancingStrategy::MemoryAware,
        })
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.mode.num_gpus() > self.topology.num_gpus() {
            anyhow::bail!(
                "Configuration requires {} GPUs but only {} available",
                self.mode.num_gpus(),
                self.topology.num_gpus()
            );
        }

        Ok(())
    }
}

/// Communication backend for cross-GPU operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationBackend {
    /// Use Candle's built-in multi-device support
    Candle,

    /// NCCL (NVIDIA Collective Communications Library) - if available
    #[allow(dead_code)]
    NCCL,
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Balance by memory usage
    MemoryAware,

    /// Balance by compute time
    ComputeAware,

    /// Static round-robin
    RoundRobin,
}
