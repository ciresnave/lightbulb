// SPDX-License-Identifier: MIT OR Apache-2.0
//! Where the Fuel path's device is chosen — and the only place it is.
//!
//! `generate.rs` originally hardcoded `Device::cpu()`. Spreading that choice
//! across the loader, the session and the runner is how a model and its KV
//! cache end up on different devices, which fails deep inside `realize` as a
//! confusing byte-count error rather than at the point of the mistake.
//!
//! # Selecting a device (ROADMAP F10 L2)
//!
//! `LIGHTBULB_DEVICE` is `auto` (the default), `cpu`, `cuda:<n>`, or a
//! case-insensitive substring of a device name (`p40`). The list of devices
//! comes from Fuel's own inventory (`fuel_hardware::probe::ProbeReport`), never
//! from an OS probe of ours. `LIGHTBULB_CUDA_DEVICE_INDEX` still works and means
//! `cuda:<n>`; setting both is an error.
//!
//! * `auto` with no accelerator is the CPU; with one it is that one; with
//!   several it is the one with the most **total** memory (lowest index on a
//!   tie), the runners-up are logged. Total, not free: Fuel's inventory has no
//!   free-bytes figure yet, and a choice that depended on free memory would
//!   differ between the several calls a model load makes. The preflight still
//!   refuses a model that does not fit the chosen card.
//! * An explicit selector that names a device that is not there is an error,
//!   never a CPU fallback: "you asked for the P40 and it is missing" must not
//!   quietly become a slow CPU run. Only `auto` falls back.
//! * Only CUDA devices are listed. Vulkan and ROCm wait on Fuel exposing them.

use anyhow::{Result, bail};
use fuel::Device;

/// One accelerator Fuel reports, as far as selection needs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Backend name as Fuel spells it (`"cuda"`).
    pub backend: String,
    /// Ordinal within the backend.
    pub index: u32,
    /// Driver-reported name, for example `NVIDIA Tesla P40`.
    pub name: String,
    /// Total memory in bytes; 0 when Fuel could not say.
    pub total_bytes: u64,
}

impl DeviceInfo {
    fn label(&self) -> String {
        format!(
            "{}:{} {} ({} MiB)",
            self.backend,
            self.index,
            self.name,
            self.total_bytes >> 20
        )
    }
}

/// What `LIGHTBULB_DEVICE` asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selector {
    Auto,
    Cpu,
    /// `<backend>:<index>`.
    Index {
        backend: String,
        index: u32,
    },
    /// A case-insensitive substring of a device name.
    Name(String),
}

impl Selector {
    /// Parse one selector string. Empty means `auto`.
    pub fn parse(raw: &str) -> Selector {
        let s = raw.trim();
        if s.is_empty() || s.eq_ignore_ascii_case("auto") {
            return Selector::Auto;
        }
        if s.eq_ignore_ascii_case("cpu") {
            return Selector::Cpu;
        }
        if let Some((backend, index)) = s.split_once(':')
            && ["cuda", "vulkan"].contains(&backend.to_ascii_lowercase().as_str())
            && let Ok(index) = index.trim().parse::<u32>()
        {
            return Selector::Index {
                backend: backend.to_ascii_lowercase(),
                index,
            };
        }
        Selector::Name(s.to_string())
    }

    /// Combine the two environment variables. Both set is an error; the legacy
    /// `LIGHTBULB_CUDA_DEVICE_INDEX` must be a whole number and means
    /// `cuda:<n>` (it used to read garbage as 0, which chose a different card
    /// from the one asked for).
    pub fn from_env_values(device: Option<&str>, cuda_index: Option<&str>) -> Result<Selector> {
        match (device, cuda_index) {
            (Some(_), Some(_)) => bail!(
                "both LIGHTBULB_DEVICE and LIGHTBULB_CUDA_DEVICE_INDEX are set; \
                 use LIGHTBULB_DEVICE alone (LIGHTBULB_CUDA_DEVICE_INDEX=n means \
                 LIGHTBULB_DEVICE=cuda:n)"
            ),
            (Some(d), None) => Ok(Selector::parse(d)),
            (None, Some(i)) => match i.trim().parse::<u32>() {
                Ok(index) => Ok(Selector::Index {
                    backend: "cuda".into(),
                    index,
                }),
                Err(_) => bail!(
                    "LIGHTBULB_CUDA_DEVICE_INDEX={i:?} is not a whole number; \
                     refusing to guess a different card"
                ),
            },
            (None, None) => Ok(Selector::Auto),
        }
    }

    /// Read the two environment variables, fresh on each call (a model load
    /// reads them; nothing caches the answer).
    pub fn from_env() -> Result<Selector> {
        Self::from_env_values(
            std::env::var("LIGHTBULB_DEVICE").ok().as_deref(),
            std::env::var("LIGHTBULB_CUDA_DEVICE_INDEX").ok().as_deref(),
        )
    }
}

/// The outcome of [`choose`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Choice {
    Cpu,
    Accelerator { backend: String, index: u32 },
}

fn list(infos: &[DeviceInfo]) -> String {
    if infos.is_empty() {
        "none (this build lists CUDA devices only, and only with the `fuel-cuda` feature)".into()
    } else {
        infos
            .iter()
            .map(DeviceInfo::label)
            .collect::<Vec<_>>()
            .join("; ")
    }
}

fn accelerator(d: &DeviceInfo) -> Choice {
    Choice::Accelerator {
        backend: d.backend.clone(),
        index: d.index,
    }
}

/// `auto`: the card with the most total memory, the lowest index on a tie.
fn choose_auto(infos: &[DeviceInfo]) -> Choice {
    let best = infos
        .iter()
        .max_by_key(|d| (d.total_bytes, std::cmp::Reverse(d.index)));
    let Some(d) = best else {
        return Choice::Cpu;
    };
    if infos.len() > 1 {
        let others: Vec<_> = infos
            .iter()
            .filter(|o| *o != d)
            .map(DeviceInfo::label)
            .collect();
        tracing::info!(
            "model_fuel: auto chose {} by total memory over {}",
            d.label(),
            others.join("; ")
        );
    }
    accelerator(d)
}

/// A device named by a substring of its name; ambiguity is an error.
fn choose_by_name(infos: &[DeviceInfo], needle: &str) -> Result<Choice> {
    let needle_lc = needle.to_lowercase();
    let hits: Vec<&DeviceInfo> = infos
        .iter()
        .filter(|d| d.name.to_lowercase().contains(&needle_lc))
        .collect();
    match hits.as_slice() {
        [d] => Ok(accelerator(d)),
        [] => bail!(
            "LIGHTBULB_DEVICE={needle:?} matches no device name. Devices: {}",
            list(infos)
        ),
        many => bail!(
            "LIGHTBULB_DEVICE={needle:?} matches {} devices; name one with \
             `<backend>:<index>`: {}",
            many.len(),
            many.iter()
                .map(|d| d.label())
                .collect::<Vec<_>>()
                .join("; ")
        ),
    }
}

/// Pick a device from `infos` for `selector`. Pure: no hardware is touched.
///
/// `auto` is the only selector that falls back to the CPU.
pub fn choose(infos: &[DeviceInfo], selector: &Selector) -> Result<Choice> {
    match selector {
        Selector::Cpu => Ok(Choice::Cpu),
        Selector::Auto => Ok(choose_auto(infos)),
        Selector::Index { backend, index } => {
            match infos
                .iter()
                .find(|d| d.backend == *backend && d.index == *index)
            {
                Some(d) => Ok(accelerator(d)),
                None => bail!(
                    "LIGHTBULB_DEVICE asked for {backend}:{index}, which is not present. \
                     Devices: {}",
                    list(infos)
                ),
            }
        }
        Selector::Name(needle) => choose_by_name(infos, needle),
    }
}

/// The accelerators this build can open, from Fuel's inventory.
#[cfg(feature = "fuel-cuda")]
pub fn enumerate() -> Vec<DeviceInfo> {
    fuel_hardware::probe::ProbeReport::probe_all()
        .devices
        .into_iter()
        .filter(|d| d.backend.as_str() == "cuda")
        .map(|d| DeviceInfo {
            backend: d.backend.as_str().to_string(),
            index: d.device_index,
            name: d.hardware_sku,
            total_bytes: d.total_memory_bytes,
        })
        .collect()
}

/// Without `fuel-cuda` there is nothing but the CPU to open.
#[cfg(not(feature = "fuel-cuda"))]
pub fn enumerate() -> Vec<DeviceInfo> {
    Vec::new()
}

/// The device this build serves on, or why the requested one cannot be used.
///
/// An invalid or absent explicit selector is an error here; `auto` (the
/// default) falls back to the CPU when no accelerator is usable. The engine
/// calls this before a load, so a bad `LIGHTBULB_DEVICE` stops the load with a
/// message rather than serving on the wrong device.
pub fn try_select() -> Result<Device> {
    let selector = Selector::from_env()?;
    let infos = enumerate();
    let choice = choose(&infos, &selector)?;
    open(choice, &selector)
}

#[cfg(feature = "fuel-cuda")]
fn open(choice: Choice, selector: &Selector) -> Result<Device> {
    if let Choice::Accelerator { index, .. } = choice {
        match fuel_cuda_backend::CudaDevice::new(index as usize) {
            Ok(d) => {
                tracing::info!("model_fuel: CUDA device {index} selected");
                return Ok(Device::from(d));
            }
            Err(e) if *selector == Selector::Auto => {
                tracing::warn!(
                    "model_fuel: `fuel-cuda` is enabled but CUDA device {index} is \
                     unavailable ({e:?}); falling back to CPU"
                );
            }
            Err(e) => bail!("CUDA device {index} could not be opened: {e:?}"),
        }
    }
    tracing::info!("model_fuel: CPU selected");
    Ok(Device::cpu())
}

#[cfg(not(feature = "fuel-cuda"))]
fn open(_choice: Choice, _selector: &Selector) -> Result<Device> {
    tracing::info!("model_fuel: CPU selected");
    Ok(Device::cpu())
}

/// The device this build serves on.
///
/// Infallible for the call sites deep in a load; they run after
/// [`try_select`] has already accepted the same environment at the start of the
/// load (`BatchedServeConfig::new`), so the error arm is unreachable in
/// practice. If it is reached anyway it is logged as an error and the CPU is
/// returned, which is the one place a bad explicit selector is not fatal.
///
/// **Falling back to CPU for `auto` is deliberate.** A server that refuses to
/// start on a CPU-only host is worse than one that starts slow, and the log
/// line names which was chosen so a surprising benchmark has somewhere to look.
///
/// The measurement harness (`tests/gpu_paged_vs_contiguous.rs`) makes the
/// OPPOSITE choice and fails outright, because a benchmark that silently
/// measures CPU is an artifact rather than a result.
pub fn select() -> Device {
    try_select().unwrap_or_else(|e| {
        tracing::error!("model_fuel: device selection failed ({e:#}); using the CPU");
        Device::cpu()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(backend: &str, index: u32, name: &str, mib: u64) -> DeviceInfo {
        DeviceInfo {
            backend: backend.into(),
            index,
            name: name.into(),
            total_bytes: mib << 20,
        }
    }

    /// CireSnave's CUDA cards as the driver names them.
    fn rig() -> Vec<DeviceInfo> {
        vec![
            info("cuda", 0, "NVIDIA GeForce RTX 4060", 8188),
            info("cuda", 1, "Tesla P40", 24576),
        ]
    }

    fn acc(index: u32) -> Choice {
        Choice::Accelerator {
            backend: "cuda".into(),
            index,
        }
    }

    #[test]
    fn selector_spellings_parse() {
        assert_eq!(Selector::parse(""), Selector::Auto);
        assert_eq!(Selector::parse(" AUTO "), Selector::Auto);
        assert_eq!(Selector::parse("CPU"), Selector::Cpu);
        assert_eq!(
            Selector::parse("cuda:1"),
            Selector::Index {
                backend: "cuda".into(),
                index: 1
            }
        );
        assert_eq!(Selector::parse("p40"), Selector::Name("p40".into()));
        // Not an index: the backend is unknown or the ordinal is not a number.
        assert_eq!(Selector::parse("foo:1"), Selector::Name("foo:1".into()));
        assert_eq!(Selector::parse("cuda:x"), Selector::Name("cuda:x".into()));
    }

    #[test]
    fn both_variables_set_is_an_error_and_the_legacy_one_means_cuda_n() {
        assert!(Selector::from_env_values(Some("cpu"), Some("1")).is_err());
        assert_eq!(
            Selector::from_env_values(None, Some(" 1 ")).unwrap(),
            Selector::Index {
                backend: "cuda".into(),
                index: 1
            }
        );
        assert_eq!(
            Selector::from_env_values(Some("p40"), None).unwrap(),
            Selector::Name("p40".into())
        );
        assert_eq!(
            Selector::from_env_values(None, None).unwrap(),
            Selector::Auto
        );
        // Garbage used to read as 0 and choose a different card.
        assert!(Selector::from_env_values(None, Some("one")).is_err());
    }

    #[test]
    fn auto_takes_the_card_with_the_most_total_memory_not_the_first() {
        // The P40 is index 1; "first" would pick the 4060.
        assert_eq!(choose(&rig(), &Selector::Auto).unwrap(), acc(1));
        let mut rev = rig();
        rev.reverse();
        assert_eq!(choose(&rev, &Selector::Auto).unwrap(), acc(1));
    }

    #[test]
    fn auto_breaks_a_tie_on_the_lowest_index_and_with_no_card_is_the_cpu() {
        let twins = vec![
            info("cuda", 3, "Tesla P40", 24576),
            info("cuda", 2, "Tesla P40", 24576),
        ];
        assert_eq!(choose(&twins, &Selector::Auto).unwrap(), acc(2));
        assert_eq!(choose(&[], &Selector::Auto).unwrap(), Choice::Cpu);
        assert_eq!(
            choose(&[info("cuda", 0, "x", 0)], &Selector::Auto).unwrap(),
            acc(0)
        );
    }

    #[test]
    fn cpu_is_always_available() {
        assert_eq!(choose(&rig(), &Selector::Cpu).unwrap(), Choice::Cpu);
        assert_eq!(choose(&[], &Selector::Cpu).unwrap(), Choice::Cpu);
    }

    #[test]
    fn a_name_substring_picks_one_card_ignoring_case() {
        assert_eq!(choose(&rig(), &Selector::parse("p40")).unwrap(), acc(1));
        assert_eq!(choose(&rig(), &Selector::parse("4060")).unwrap(), acc(0));
    }

    #[test]
    fn an_ambiguous_name_names_the_candidates_and_never_takes_the_first() {
        // Control: on the rig only the 4060's name says "NVIDIA", so one card matches.
        assert_eq!(choose(&rig(), &Selector::parse("nvidia")).unwrap(), acc(0));
        let both = vec![
            info("cuda", 0, "NVIDIA Tesla P40", 24576),
            info("cuda", 1, "NVIDIA GeForce RTX 4060", 8188),
        ];
        let e = choose(&both, &Selector::parse("nvidia"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("matches 2 devices"), "{e}");
        assert!(e.contains("cuda:0") && e.contains("cuda:1"), "{e}");
    }

    #[test]
    fn an_absent_explicit_device_is_an_error_listing_what_exists() {
        let e = choose(&rig(), &Selector::parse("cuda:5"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("cuda:5") && e.contains("Tesla P40"), "{e}");
        let e = choose(&rig(), &Selector::parse("mi50"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("matches no device name"), "{e}");
        // No CPU fallback for an explicit request, even with no devices at all.
        assert!(choose(&[], &Selector::parse("cuda:0")).is_err());
        assert!(choose(&[], &Selector::parse("p40")).is_err());
    }

    #[test]
    fn index_selection_matches_the_backend_as_well_as_the_ordinal() {
        let mixed = vec![
            info("cuda", 0, "Tesla P40", 24576),
            info("vulkan", 0, "Radeon VII", 16384),
        ];
        assert_eq!(choose(&mixed, &Selector::parse("cuda:0")).unwrap(), acc(0));
        assert_eq!(
            choose(&mixed, &Selector::parse("vulkan:0")).unwrap(),
            Choice::Accelerator {
                backend: "vulkan".into(),
                index: 0
            }
        );
    }

    /// `select()` returns a usable device and does not panic.
    ///
    /// Deliberately weak on WHICH device — that depends on features and
    /// hardware. The claim under test is that selection is total: there is no
    /// build configuration in which it fails to produce one.
    #[test]
    fn select_returns_a_usable_device() {
        let dev = super::select();
        // Realizing a trivial graph proves the device is not merely
        // constructed but actually drivable.
        let a = fuel::lazy::Tensor::from_f32(vec![2.0, 3.0], (1usize, 2usize), &dev)
            .expect("from_f32: graph build");
        let w = a
            .const_f32_like(vec![1.0, 0.0, 0.0, 1.0], (2usize, 2usize))
            .expect("const_f32_like: graph build");
        let y = a.matmul(&w).expect("matmul on the selected device");
        assert_eq!(y.realize_f32(), vec![2.0, 3.0]);
    }
}
