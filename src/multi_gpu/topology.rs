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

        // Probe CUDA devices until one is not a CUDA device.
        //
        // ⚠️ AN `Ok` FROM `cuda_if_available` IS NOT EVIDENCE THE ORDINAL
        // EXISTS. candle-core 0.10.2 `device.rs:323` is
        // `if cuda_is_available() { new_cuda(ordinal) } else { Ok(Self::Cpu) }`
        // — the fallback arm returns `Ok` unconditionally and NEVER CONSULTS
        // THE ORDINAL.
        //
        // This loop previously broke only on `Err`, so on any build without the
        // `cuda` feature every iteration returned `Ok(Cpu)`, pushed, and
        // incremented: it did not terminate. And because it PUSHES each time it
        // was unbounded allocation rather than a quiet spin. `Cargo.toml`
        // records that `candlelight/cuda` does not build on this toolchain, so
        // the non-terminating configuration is the ordinary one here.
        //
        // Measured before the fix: the termination test below did not fail, it
        // hung — a bounded subprocess exited 124 after 25s.
        loop {
            match Device::cuda_if_available(device_id) {
                Ok(device) if device.is_cuda() => {
                    devices.push(device);
                    device_id += 1;
                }
                _ => break,
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
    ) -> Result<crate::multi_gpu::config::ParallelismMode> {
        use crate::multi_gpu::config::ParallelismMode;

        let num_gpus = self.devices.len();
        let total_memory = self.memory_available.iter().sum::<usize>();

        // A topology with no devices is a caller error, not a crash. The fields
        // of this struct are public, so one can be constructed directly, and
        // `self.memory_available[0]` below would panic on the index.
        if self.memory_available.is_empty() {
            anyhow::bail!("Cannot recommend a parallelism strategy for a topology with no devices");
        }

        // A MODEL THAT DOES NOT FIT IS A RECOVERABLE CONDITION, NOT A CRASH.
        // This was `panic!`, in a public method reachable through
        // `MultiGPUConfig::auto`, for the entirely ordinary case of asking
        // about a model larger than the machine. A caller sizing a deployment
        // has every reason to ask that question and get an answer.
        if model_size_bytes > total_memory {
            anyhow::bail!(
                "Model too large for available GPU memory: {} bytes required, {} available",
                model_size_bytes,
                total_memory
            );
        }

        // If model fits on single GPU, no parallelism needed
        if model_size_bytes < self.memory_available[0] {
            return Ok(ParallelismMode::Single);
        }

        // If model fits on 2 GPUs with tensor parallelism, prefer that
        if num_gpus >= 2 && model_size_bytes < (self.memory_available[0] + self.memory_available[1])
        {
            return Ok(ParallelismMode::TensorParallel { world_size: 2 });
        }

        // Otherwise, use pipeline parallelism with more stages
        Ok(ParallelismMode::PipelineParallel {
            num_stages: num_gpus.min(4),
            micro_batch_size: 1,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A topology of CPU devices with a stated per-device memory budget.
    fn topology_with(memory_per_device: usize, n: usize) -> DeviceTopology {
        DeviceTopology {
            devices: vec![Device::Cpu; n],
            memory_capacity: vec![memory_per_device; n],
            memory_available: vec![memory_per_device; n],
            interconnect: InterconnectTopology::PCIe {
                bandwidth_gbps: 16.0,
            },
            p2p_access: vec![vec![false; n]; n],
        }
    }

    /// **A model that does not fit is an ERROR, not a crash.**
    ///
    /// This was `panic!`, in a public method reachable through
    /// `MultiGPUConfig::auto`, for the ordinary case of asking about a model
    /// larger than the machine. Sizing a deployment is exactly the question a
    /// caller asks, and it deserves an answer rather than an abort.
    #[test]
    fn a_model_larger_than_the_machine_is_an_error() {
        let t = topology_with(1000, 2);
        let err = t
            .recommend_strategy(5000)
            .expect_err("a model exceeding total memory must not be recommended a strategy");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("5000") && msg.contains("2000"),
            "the error should state what was needed and what exists: {msg}"
        );
    }

    /// And a topology with no devices is an error rather than an index panic.
    ///
    /// `DeviceTopology`'s fields are public, so an empty one is constructible,
    /// and `self.memory_available[0]` would have panicked on the index.
    #[test]
    fn an_empty_topology_is_an_error_not_an_index_panic() {
        let t = topology_with(1000, 0);
        assert!(t.recommend_strategy(1).is_err());
    }

    /// Fits on one device -> no parallelism.
    #[test]
    fn a_model_that_fits_on_one_device_needs_no_parallelism() {
        use crate::multi_gpu::config::ParallelismMode;
        let t = topology_with(1000, 4);
        assert_eq!(t.recommend_strategy(500).unwrap(), ParallelismMode::Single);
    }

    /// Fits across two -> tensor parallel over two.
    #[test]
    fn a_model_that_fits_on_two_devices_uses_tensor_parallelism() {
        use crate::multi_gpu::config::ParallelismMode;
        let t = topology_with(1000, 4);
        assert_eq!(
            t.recommend_strategy(1500).unwrap(),
            ParallelismMode::TensorParallel { world_size: 2 }
        );
    }

    /// Needs more than two -> pipeline, capped at four stages.
    ///
    /// The cap is the part worth pinning: `num_gpus.min(4)` on an 8-device
    /// topology must give 4, and a test on a 2- or 4-device topology cannot
    /// tell `min(4)` from the identity.
    #[test]
    fn a_model_spanning_many_devices_uses_pipeline_capped_at_four_stages() {
        use crate::multi_gpu::config::ParallelismMode;
        let t = topology_with(1000, 8);
        assert_eq!(
            t.recommend_strategy(7500).unwrap(),
            ParallelismMode::PipelineParallel {
                num_stages: 4,
                micro_batch_size: 1
            }
        );
    }

    /// **`discover()` must TERMINATE on a machine with no CUDA device.**
    ///
    /// # The defect this pins
    ///
    /// The probe loop broke only on `Err`:
    ///
    /// ```text
    /// loop {
    ///     match Device::cuda_if_available(device_id) {
    ///         Ok(device) => { devices.push(device); device_id += 1; }
    ///         Err(_) => break,
    ///     }
    /// }
    /// ```
    ///
    /// `Device::cuda_if_available` returns `Ok(Device::Cpu)` when CUDA is
    /// unavailable and **does not consult the ordinal in that path**
    /// (candle-core 0.10.2, `device.rs:323`). So on a build without the `cuda`
    /// feature every iteration returns `Ok`, pushes, and increments — the loop
    /// never terminates, and because it PUSHES each time it is unbounded
    /// allocation rather than a quiet spin.
    ///
    /// `Cargo.toml` records that `candlelight/cuda` does not build on this
    /// toolchain, so the non-CUDA build is the ordinary one here.
    ///
    /// **An `Ok` from that function is not evidence the ordinal exists**, which
    /// is why the fix tests `is_cuda()` rather than trusting the `Result`.
    ///
    /// This test is bounded on both kinds of machine: without CUDA it must be
    /// the documented error, and with CUDA every discovered device must be a
    /// CUDA device. Before the fix it does not fail — it hangs.
    #[test]
    fn discover_terminates_when_no_cuda_device_is_present() {
        match DeviceTopology::discover() {
            Err(e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("No CUDA devices"),
                    "without CUDA, discovery must report that, got: {msg}"
                );
            }
            Ok(topology) => {
                assert!(
                    !topology.devices.is_empty(),
                    "a successful discovery must find at least one device"
                );
                assert!(
                    topology.devices.iter().all(|d| d.is_cuda()),
                    "discovery must not report a CPU fallback as a discovered GPU"
                );
                assert_eq!(
                    topology.memory_capacity.len(),
                    topology.devices.len(),
                    "per-device vectors must match the device count"
                );
            }
        }
    }
}
