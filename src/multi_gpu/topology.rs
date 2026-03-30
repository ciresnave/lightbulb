use anyhow::Result;
use candlelight::core::Device;

/// Multi-GPU device topology and capabilities
#[derive(Debug, Clone)]
pub struct DeviceTopology {
    /// All available CUDA devices
    pub devices: Vec<Device>,

    /// Memory capacity per device (bytes)
    pub memory_capacity: Vec<usize>,

    /// Available memory per device (bytes)
    pub memory_available: Vec<usize>,

    /// Interconnect topology (NVLink, PCIe bandwidth)
    pub interconnect: InterconnectTopology,

    /// Peer-to-peer access matrix (can GPU i access GPU j directly?)
    pub p2p_access: Vec<Vec<bool>>,
}

impl DeviceTopology {
    /// Discover all available CUDA devices
    pub fn discover() -> Result<Self> {
        let mut devices = Vec::new();
        let mut device_id = 0;

        // Probe CUDA devices until we hit an error
        loop {
            match Device::cuda_if_available(device_id) {
                Ok(device) => {
                    devices.push(device);
                    device_id += 1;
                }
                Err(_) => break,
            }
        }

        if devices.is_empty() {
            anyhow::bail!("No CUDA devices available for multi-GPU inference");
        }

        // Query memory capacity (TODO: Candle API for memory info)
        // For now, assume 80GB per GPU (A100/H100 common size)
        let memory_capacity = vec![80 * 1024 * 1024 * 1024; devices.len()];
        let memory_available = memory_capacity.clone();

        // Detect interconnect topology
        let interconnect = InterconnectTopology::detect(&devices)?;

        // Query peer-to-peer access
        let p2p_access = Self::query_p2p_access(&devices)?;

        Ok(Self {
            devices,
            memory_capacity,
            memory_available,
            interconnect,
            p2p_access,
        })
    }

    /// Check if peer-to-peer access is available between two devices
    fn query_p2p_access(devices: &[Device]) -> Result<Vec<Vec<bool>>> {
        // TODO: Query actual P2P capabilities via CUDA
        // For now, assume all GPUs can access each other
        let n = devices.len();
        Ok(vec![vec![true; n]; n])
    }

    /// Get recommended parallelism strategy based on topology
    pub fn recommend_strategy(
        &self,
        model_size_bytes: usize,
    ) -> crate::multi_gpu::config::ParallelismMode {
        use crate::multi_gpu::config::ParallelismMode;

        let num_gpus = self.devices.len();
        let total_memory = self.memory_available.iter().sum::<usize>();

        if model_size_bytes > total_memory {
            panic!(
                "Model too large for available GPU memory: {} bytes required, {} available",
                model_size_bytes, total_memory
            );
        }

        // If model fits on single GPU, no parallelism needed
        if model_size_bytes < self.memory_available[0] {
            return ParallelismMode::Single;
        }

        // If model fits on 2 GPUs with tensor parallelism, prefer that
        if num_gpus >= 2 && model_size_bytes < (self.memory_available[0] + self.memory_available[1])
        {
            return ParallelismMode::TensorParallel { world_size: 2 };
        }

        // Otherwise, use pipeline parallelism with more stages
        ParallelismMode::PipelineParallel {
            num_stages: num_gpus.min(4),
            micro_batch_size: 1,
        }
    }

    /// Number of discovered GPUs
    pub fn num_gpus(&self) -> usize {
        self.devices.len()
    }

    /// Get device by index
    pub fn device(&self, idx: usize) -> Option<&Device> {
        self.devices.get(idx)
    }
}

/// Interconnect topology between GPUs
#[derive(Debug, Clone)]
pub enum InterconnectTopology {
    /// NVLink (high bandwidth, low latency)
    NVLink { bandwidth_gbps: f32 },

    /// PCIe (lower bandwidth, higher latency)
    PCIe { bandwidth_gbps: f32 },

    /// Mixed (some NVLink, some PCIe)
    Mixed { links: Vec<InterconnectLink> },
}

#[derive(Debug, Clone)]
pub struct InterconnectLink {
    pub from_device: usize,
    pub to_device: usize,
    pub link_type: LinkType,
    pub bandwidth_gbps: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkType {
    NVLink,
    PCIe,
}

impl InterconnectTopology {
    pub(crate) fn detect(devices: &[Device]) -> Result<Self> {
        // TODO: Query actual interconnect via CUDA
        // For now, assume NVLink for 2-GPU, PCIe for 4+
        if devices.len() == 2 {
            Ok(Self::NVLink {
                bandwidth_gbps: 600.0, // NVLink 4.0
            })
        } else {
            Ok(Self::PCIe {
                bandwidth_gbps: 32.0, // PCIe 4.0 x16
            })
        }
    }

    /// Get bandwidth description
    pub fn description(&self) -> String {
        match self {
            Self::NVLink { bandwidth_gbps } => {
                format!("NVLink ({:.1} GB/s)", bandwidth_gbps)
            }
            Self::PCIe { bandwidth_gbps } => {
                format!("PCIe ({:.1} GB/s)", bandwidth_gbps)
            }
            Self::Mixed { links } => {
                format!("Mixed ({} links)", links.len())
            }
        }
    }
}
