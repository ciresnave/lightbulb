use anyhow::Result;
use fuel::probe::ProbeReport;
use fuel::{Device, DeviceLocation};
use fuel_ir::backend::TransferPath;
use fuel_ir::probe::BackendId;

/// Multi-GPU device topology and capabilities.
///
/// Sourced from real Fuel measurements (board item 106, PR 1), not guesses:
/// `fuel::probe::ProbeReport` for per-device memory, `fuel::topology::
/// SystemTopology::transfer_path`/`transfer_estimate` for real P2P status and
/// measured bandwidth. The prior version of this file hardcoded an 80GB
/// per-GPU guess, assumed all GPUs can reach each other, and guessed NVLink
/// vs PCIe purely from GPU count — none of that reflected real hardware.
#[derive(Debug, Clone)]
pub struct DeviceTopology {
    /// All available CUDA devices
    pub devices: Vec<Device>,

    /// Memory capacity per device (bytes)
    pub memory_capacity: Vec<usize>,

    /// Available memory per device (bytes)
    ///
    /// Fuel's probe reports total device memory, not live free memory — this
    /// is currently identical to `memory_capacity` (same as the pre-fuel
    /// version of this struct, which also treated the two as equal). A real
    /// live-utilization query would need a separate fuel primitive this
    /// session did not find; not fabricated here.
    pub memory_available: Vec<usize>,

    /// Interconnect topology (NVLink, PCIe bandwidth)
    pub interconnect: InterconnectTopology,

    /// Peer-to-peer access matrix (can GPU i access GPU j directly?)
    pub p2p_access: Vec<Vec<bool>>,
}

impl DeviceTopology {
    /// Discover all available CUDA devices via Fuel's real hardware probe.
    pub fn discover() -> Result<Self> {
        let report = ProbeReport::probe_all();
        let sys = fuel::topology::SystemTopology::current();
        Self::from_probe(
            &report,
            |loc| match loc {
                DeviceLocation::Cuda { gpu_id } => {
                    Ok(fuel::cuda_backend::device_if_available(gpu_id)?)
                }
                other => anyhow::bail!(
                    "DeviceTopology::discover: asked to open a non-CUDA location {other:?} — \
                     this is a bug in the CUDA-only filter above, not a hardware condition"
                ),
            },
            |a, b| sys.transfer_path(a, b),
            |a, b| transfer_bandwidth_gbps(&sys, a, b),
        )
    }

    /// The pure decision core of [`Self::discover`]: given a probe report and
    /// two injected queries (open a device by location; the interconnect
    /// facts between two locations), build a `DeviceTopology`. Separated out
    /// so the mapping logic is testable without real GPU hardware — this
    /// machine has none — the same pattern as `model_fuel::policies::
    /// extract_prefix_match_for_kv_cache_seed`'s injected `seed` callback.
    ///
    /// Filters to `BackendId::Cuda` only, preserving this struct's original
    /// CUDA-only scope — broadening to Vulkan/Metal multi-GPU is a separate
    /// decision, not a side effect of this fix.
    fn from_probe(
        report: &ProbeReport,
        open_device: impl Fn(DeviceLocation) -> Result<Device>,
        transfer_path: impl Fn(DeviceLocation, DeviceLocation) -> TransferPath,
        transfer_bandwidth_gbps: impl Fn(DeviceLocation, DeviceLocation) -> f32,
    ) -> Result<Self> {
        let cuda: Vec<_> = report
            .devices
            .iter()
            .filter(|d| d.backend == BackendId::Cuda)
            .collect();
        if cuda.is_empty() {
            anyhow::bail!("No CUDA devices available for multi-GPU inference");
        }

        let mut devices = Vec::with_capacity(cuda.len());
        let mut memory_capacity = Vec::with_capacity(cuda.len());
        let mut locations = Vec::with_capacity(cuda.len());
        for d in &cuda {
            devices.push(open_device(d.location)?);
            memory_capacity.push(d.total_memory_bytes as usize);
            locations.push(d.location);
        }
        let memory_available = memory_capacity.clone();

        let n = locations.len();
        let mut p2p_access = vec![vec![false; n]; n];
        for i in 0..n {
            for j in 0..n {
                p2p_access[i][j] =
                    i == j || transfer_path(locations[i], locations[j]) == TransferPath::Peer;
            }
        }

        let interconnect = InterconnectTopology::from_measurements(
            &locations,
            &transfer_path,
            &transfer_bandwidth_gbps,
        );

        Ok(Self {
            devices,
            memory_capacity,
            memory_available,
            interconnect,
            p2p_access,
        })
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

/// Real measured GB/s between two locations, derived from Fuel's calibrated
/// transfer estimate for a 1 GiB transfer. `fuel::transfer_cost::
/// TransferEstimate::estimate_ns` is the only numeric cost primitive Fuel
/// exposes (no direct "GB/s" field) — see `InterconnectTopology`'s doc for
/// why this is an inference, not a value Fuel asserts directly.
fn transfer_bandwidth_gbps(
    sys: &fuel::topology::SystemTopology,
    a: DeviceLocation,
    b: DeviceLocation,
) -> f32 {
    const ONE_GIB: u64 = 1 << 30;
    let ns = sys.transfer_estimate(a, b).estimate_ns(ONE_GIB);
    if ns == 0 {
        return 0.0;
    }
    (ONE_GIB as f64 / ns as f64) as f32
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
    /// Classify every device pair's measured link, then collapse to the
    /// simplest shape that describes them all.
    ///
    /// ⚠️ Fuel has no type that names a link "NVLink" or "PCIe" — only
    /// `TransferPath` (`Peer`/`DeviceCopy`/`HostStaging`/...) and a numeric
    /// bandwidth estimate. `LinkType::NVLink` here means "Fuel reports a
    /// direct peer path" and `LinkType::PCIe` means "it does not" — an
    /// INFERENCE from the path classification, not a literal hardware query.
    /// Fewer than 2 devices produces `PCIe { bandwidth_gbps: 0.0 }` — there is
    /// no interconnect to describe.
    fn from_measurements(
        locations: &[DeviceLocation],
        transfer_path: &impl Fn(DeviceLocation, DeviceLocation) -> TransferPath,
        transfer_bandwidth_gbps: &impl Fn(DeviceLocation, DeviceLocation) -> f32,
    ) -> Self {
        if locations.len() < 2 {
            return Self::PCIe {
                bandwidth_gbps: 0.0,
            };
        }

        let mut links = Vec::new();
        for i in 0..locations.len() {
            for j in (i + 1)..locations.len() {
                let is_peer = transfer_path(locations[i], locations[j]) == TransferPath::Peer;
                links.push(InterconnectLink {
                    from_device: i,
                    to_device: j,
                    link_type: if is_peer {
                        LinkType::NVLink
                    } else {
                        LinkType::PCIe
                    },
                    bandwidth_gbps: transfer_bandwidth_gbps(locations[i], locations[j]),
                });
            }
        }

        if links.iter().all(|l| l.link_type == LinkType::NVLink) {
            let avg = links.iter().map(|l| l.bandwidth_gbps).sum::<f32>() / links.len() as f32;
            return Self::NVLink {
                bandwidth_gbps: avg,
            };
        }
        if links.iter().all(|l| l.link_type == LinkType::PCIe) {
            let avg = links.iter().map(|l| l.bandwidth_gbps).sum::<f32>() / links.len() as f32;
            return Self::PCIe {
                bandwidth_gbps: avg,
            };
        }
        Self::Mixed { links }
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

    fn descriptor(
        backend: BackendId,
        device_index: u32,
        location: DeviceLocation,
        total_memory_bytes: u64,
    ) -> fuel_ir::probe::DeviceDescriptor {
        fuel_ir::probe::DeviceDescriptor {
            backend,
            device_index,
            hardware_sku: "test-gpu".to_string(),
            vendor_id: 0x10DE,
            device_id: 0,
            compute_capability: Some((8, 9)),
            subgroup_width: Some(32),
            driver_version: "test".to_string(),
            total_memory_bytes,
            location,
        }
    }

    fn report(devices: Vec<fuel_ir::probe::DeviceDescriptor>) -> ProbeReport {
        ProbeReport {
            version: fuel::probe::PROBE_REPORT_VERSION,
            devices,
        }
    }

    fn open_cpu_stand_in(_loc: DeviceLocation) -> Result<Device> {
        // No GPU on this machine — a CPU `Device` stands in for "the open
        // succeeded", since these tests assert on DeviceTopology's own fields
        // (memory/p2p/interconnect), not on what kind of Device was opened.
        Ok(Device::cpu())
    }

    #[test]
    fn from_probe_errors_when_no_cuda_descriptor_is_present() {
        let r = report(vec![descriptor(BackendId::Cpu, 0, DeviceLocation::Cpu, 0)]);
        let err = DeviceTopology::from_probe(
            &r,
            open_cpu_stand_in,
            |_, _| TransferPath::HostStaging,
            |_, _| 0.0,
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("No CUDA devices"));
    }

    #[test]
    fn from_probe_reads_real_memory_per_device_not_a_guess() {
        let r = report(vec![
            descriptor(
                BackendId::Cuda,
                0,
                DeviceLocation::Cuda { gpu_id: 0 },
                24 * (1 << 30),
            ),
            descriptor(
                BackendId::Cuda,
                1,
                DeviceLocation::Cuda { gpu_id: 1 },
                80 * (1 << 30),
            ),
        ]);
        let t = DeviceTopology::from_probe(
            &r,
            open_cpu_stand_in,
            |_, _| TransferPath::Peer,
            |_, _| 100.0,
        )
        .unwrap();
        assert_eq!(t.num_gpus(), 2);
        // The two descriptors have DIFFERENT memory sizes — if this read a
        // hardcoded constant instead of the descriptor, both entries would be
        // identical.
        assert_eq!(t.memory_capacity, vec![24 * (1 << 30), 80 * (1 << 30)]);
        assert_eq!(t.memory_available, t.memory_capacity);
    }

    #[test]
    fn from_probe_ignores_non_cuda_descriptors_in_the_report() {
        let r = report(vec![
            descriptor(BackendId::Cpu, 0, DeviceLocation::Cpu, 1 << 30),
            descriptor(
                BackendId::Cuda,
                0,
                DeviceLocation::Cuda { gpu_id: 0 },
                16 * (1 << 30),
            ),
        ]);
        let t = DeviceTopology::from_probe(
            &r,
            open_cpu_stand_in,
            |_, _| TransferPath::Peer,
            |_, _| 50.0,
        )
        .unwrap();
        assert_eq!(
            t.num_gpus(),
            1,
            "the CPU descriptor must not count as a GPU"
        );
    }

    #[test]
    fn from_probe_p2p_matrix_reflects_real_transfer_path_not_a_blanket_true() {
        let r = report(vec![
            descriptor(
                BackendId::Cuda,
                0,
                DeviceLocation::Cuda { gpu_id: 0 },
                1 << 30,
            ),
            descriptor(
                BackendId::Cuda,
                1,
                DeviceLocation::Cuda { gpu_id: 1 },
                1 << 30,
            ),
        ]);
        // Device 0<->1 requires host staging: NOT direct P2P.
        let t = DeviceTopology::from_probe(
            &r,
            open_cpu_stand_in,
            |_, _| TransferPath::HostStaging,
            |_, _| 10.0,
        )
        .unwrap();
        assert_eq!(t.p2p_access, vec![vec![true, false], vec![false, true]]);
    }

    #[test]
    fn from_probe_p2p_matrix_is_true_for_a_real_peer_path() {
        let r = report(vec![
            descriptor(
                BackendId::Cuda,
                0,
                DeviceLocation::Cuda { gpu_id: 0 },
                1 << 30,
            ),
            descriptor(
                BackendId::Cuda,
                1,
                DeviceLocation::Cuda { gpu_id: 1 },
                1 << 30,
            ),
        ]);
        let t = DeviceTopology::from_probe(
            &r,
            open_cpu_stand_in,
            |_, _| TransferPath::Peer,
            |_, _| 300.0,
        )
        .unwrap();
        assert_eq!(t.p2p_access, vec![vec![true, true], vec![true, true]]);
    }

    #[test]
    fn interconnect_classifies_all_peer_links_as_nvlink() {
        let locs = vec![
            DeviceLocation::Cuda { gpu_id: 0 },
            DeviceLocation::Cuda { gpu_id: 1 },
        ];
        let t =
            InterconnectTopology::from_measurements(&locs, &|_, _| TransferPath::Peer, &|_, _| {
                600.0
            });
        match t {
            InterconnectTopology::NVLink { bandwidth_gbps } => assert_eq!(bandwidth_gbps, 600.0),
            other => panic!("expected NVLink, got {other:?}"),
        }
    }

    #[test]
    fn interconnect_classifies_all_non_peer_links_as_pcie() {
        let locs = vec![
            DeviceLocation::Cuda { gpu_id: 0 },
            DeviceLocation::Cuda { gpu_id: 1 },
        ];
        let t = InterconnectTopology::from_measurements(
            &locs,
            &|_, _| TransferPath::DeviceCopy,
            &|_, _| 32.0,
        );
        match t {
            InterconnectTopology::PCIe { bandwidth_gbps } => assert_eq!(bandwidth_gbps, 32.0),
            other => panic!("expected PCIe, got {other:?}"),
        }
    }

    #[test]
    fn interconnect_classifies_a_mix_of_peer_and_non_peer_links_as_mixed() {
        let locs = vec![
            DeviceLocation::Cuda { gpu_id: 0 },
            DeviceLocation::Cuda { gpu_id: 1 },
            DeviceLocation::Cuda { gpu_id: 2 },
        ];
        // 0<->1 peer, everything else not — a real heterogeneous topology.
        let t = InterconnectTopology::from_measurements(
            &locs,
            &|a, b| {
                if matches!(
                    (a, b),
                    (
                        DeviceLocation::Cuda { gpu_id: 0 },
                        DeviceLocation::Cuda { gpu_id: 1 }
                    )
                ) {
                    TransferPath::Peer
                } else {
                    TransferPath::DeviceCopy
                }
            },
            &|_, _| 50.0,
        );
        match t {
            InterconnectTopology::Mixed { links } => assert_eq!(links.len(), 3),
            other => panic!("expected Mixed, got {other:?}"),
        }
    }

    /// A topology of CPU devices with a stated per-device memory budget, for
    /// `recommend_strategy`'s own logic — unrelated to `from_probe`'s
    /// descriptor mapping above, so these build `DeviceTopology` directly via
    /// its public fields, same as before this PR.
    fn topology_with(memory_per_device: usize, n: usize) -> DeviceTopology {
        DeviceTopology {
            devices: vec![Device::cpu(); n],
            memory_capacity: vec![memory_per_device; n],
            memory_available: vec![memory_per_device; n],
            interconnect: InterconnectTopology::PCIe {
                bandwidth_gbps: 16.0,
            },
            p2p_access: vec![vec![false; n]; n],
        }
    }

    /// **A model that does not fit is an ERROR, not a crash.**
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

    /// **`discover()` must still report the documented error on a machine
    /// with no CUDA device** (this machine). Unlike the pre-fuel version,
    /// there is no non-terminating-loop hazard to pin here — `ProbeReport::
    /// probe_all()` is bounded by construction (it walks a fixed backend
    /// registry once), so that regression class cannot recur through this
    /// path. Kept as an integration smoke test of the real `discover()`
    /// wrapper, not just `from_probe`.
    #[test]
    fn discover_reports_the_documented_error_without_cuda_hardware() {
        match DeviceTopology::discover() {
            Err(e) => {
                let msg = format!("{e:#}");
                assert!(
                    msg.contains("No CUDA devices"),
                    "without CUDA, discovery must report that, got: {msg}"
                );
            }
            Ok(topology) => {
                // This CI machine has no GPU; a stray `Ok` would mean the
                // CUDA filter is broken, not that hardware appeared.
                panic!(
                    "discover() unexpectedly succeeded with {} device(s) on a CPU-only machine",
                    topology.num_gpus()
                );
            }
        }
    }
}
