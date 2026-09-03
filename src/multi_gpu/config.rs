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
        let mode = topology.recommend_strategy(model_size_bytes)?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_gpu::topology::InterconnectTopology;

    /// A topology of plain CPU devices.
    ///
    /// `DeviceTopology`'s fields are public, so nothing here needs a GPU or
    /// `discover()`. This module had ZERO tests before — not zero running, zero
    /// written — while being the code that decides whether a caller's
    /// parallelism configuration is runnable at all.
    fn cpu_topology(n: usize) -> DeviceTopology {
        DeviceTopology {
            devices: vec![candlelight::core::Device::Cpu; n],
            memory_capacity: vec![1 << 30; n],
            memory_available: vec![1 << 30; n],
            interconnect: InterconnectTopology::PCIe {
                bandwidth_gbps: 16.0,
            },
            p2p_access: vec![vec![false; n]; n],
        }
    }

    fn config_over(mode: ParallelismMode, gpus: usize) -> MultiGPUConfig {
        MultiGPUConfig {
            mode,
            topology: cpu_topology(gpus),
            sharding_strategy: crate::multi_gpu::tensor_parallel::ShardingStrategy::ColumnWise,
            communication_backend: CommunicationBackend::Candle,
            distributed_kv_cache: true,
            load_balancing: LoadBalancingStrategy::MemoryAware,
        }
    }

    /// **`Hybrid` costs the PRODUCT of its two dimensions, not either one.**
    ///
    /// This is the number `manual` and `validate` both gate on, so getting it
    /// wrong would either reject a runnable configuration or admit one that
    /// needs more GPUs than exist. A 2x2 hybrid needs 4.
    #[test]
    fn a_modes_gpu_count_is_the_product_for_hybrid() {
        assert_eq!(ParallelismMode::Single.num_gpus(), 1);
        assert_eq!(
            ParallelismMode::TensorParallel { world_size: 8 }.num_gpus(),
            8
        );
        assert_eq!(
            ParallelismMode::PipelineParallel {
                num_stages: 4,
                micro_batch_size: 2
            }
            .num_gpus(),
            4
        );
        assert_eq!(
            ParallelismMode::Hybrid {
                tensor_world_size: 2,
                pipeline_stages: 2,
                micro_batch_size: 4
            }
            .num_gpus(),
            4,
            "a 2x2 hybrid needs 4 GPUs, not 2"
        );
        assert_eq!(
            ParallelismMode::Hybrid {
                tensor_world_size: 4,
                pipeline_stages: 2,
                micro_batch_size: 4
            }
            .num_gpus(),
            8,
            "asymmetric dimensions must still multiply"
        );
    }

    /// `validate` accepts a configuration the topology can run.
    #[test]
    fn validate_accepts_a_configuration_that_fits() {
        let cfg = config_over(ParallelismMode::TensorParallel { world_size: 2 }, 2);
        assert!(cfg.validate().is_ok());
    }

    /// **And rejects one that needs more GPUs than exist** — the safety
    /// property, and the reason the function exists.
    ///
    /// Its necessary pair: without it, `validate` returning `Ok(())`
    /// unconditionally would satisfy the test above.
    #[test]
    fn validate_rejects_a_configuration_that_does_not_fit() {
        let cfg = config_over(ParallelismMode::TensorParallel { world_size: 4 }, 2);
        let err = cfg
            .validate()
            .expect_err("a 4-way config on a 2-device topology must not validate");
        let msg = format!("{err:#}");
        assert!(
            msg.contains('4') && msg.contains('2'),
            "the error should say what was needed and what exists: {msg}"
        );
    }

    /// The boundary: exactly as many GPUs as the mode needs is ACCEPTED.
    ///
    /// `validate` compares with `>`, so an off-by-one to `>=` would reject
    /// every exactly-sized configuration — the most common kind — while both
    /// tests above still passed.
    #[test]
    fn validate_accepts_an_exactly_sized_configuration() {
        let cfg = config_over(
            ParallelismMode::Hybrid {
                tensor_world_size: 2,
                pipeline_stages: 2,
                micro_batch_size: 4,
            },
            4,
        );
        assert!(
            cfg.validate().is_ok(),
            "4 GPUs required and 4 available must validate"
        );
    }

    /// Each mode describes a distinct communication pattern.
    #[test]
    fn each_mode_reports_a_distinct_communication_pattern() {
        let modes = [
            ParallelismMode::Single,
            ParallelismMode::TensorParallel { world_size: 2 },
            ParallelismMode::PipelineParallel {
                num_stages: 2,
                micro_batch_size: 1,
            },
            ParallelismMode::Hybrid {
                tensor_world_size: 2,
                pipeline_stages: 2,
                micro_batch_size: 1,
            },
        ];
        let patterns: std::collections::BTreeSet<&str> =
            modes.iter().map(|m| m.communication_pattern()).collect();
        assert_eq!(
            patterns.len(),
            modes.len(),
            "two modes share a description, so the string cannot identify the mode"
        );
    }
}
