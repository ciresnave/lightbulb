// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pre-load memory check for the Fuel serving path (ROADMAP B9).
//!
//! Before a loader allocates, ask the serving device (through Fuel's
//! `BackendRuntime`) whether the weights can fit. The figure used is the
//! model's on-disk size, a LOWER bound: the f32 loader upcasts, and the KV
//! cache and activations come on top. A refusal is therefore certain; a pass
//! is not a promise.

use std::path::Path;

use anyhow::bail;
use fuel::DeviceLocation;
use fuel_backend_contract::backend::BackendRuntime;

use crate::memory::format_bytes;
use crate::memory::preflight::{Preflight, check_with};

/// On-disk weight bytes of the model at `path`: the file itself for a
/// `.gguf`, the sum of the shard files for a SafeTensors directory. `None`
/// when it cannot be read; the loader then reports its own error.
pub(crate) fn weights_lower_bound(path: &Path) -> Option<u64> {
    if path.is_file() {
        return std::fs::metadata(path).ok().map(|m| m.len());
    }
    let files = super::loader::safetensors_files(path).ok()?;
    files.iter().try_fold(0u64, |sum, f| {
        std::fs::metadata(f)
            .ok()
            .map(|m| sum.saturating_add(m.len()))
    })
}

/// Where the scheduler's memory budget came from, for the startup log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetSource {
    /// `LIGHTBULB_MAX_MEMORY_BYTES` was set and parsed.
    Env,
    /// The serving device's `available_bytes` at startup.
    Device,
    /// Neither: the built-in default.
    BuiltIn,
}

impl BudgetSource {
    /// Human-readable name for the log line.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Env => "LIGHTBULB_MAX_MEMORY_BYTES",
            Self::Device => "serving device available_bytes",
            Self::BuiltIn => "built-in default",
        }
    }
}

/// The scheduler's `max_memory_bytes`: the explicit setting always wins, else
/// what the serving device reports available, else `built_in`. A device
/// reading of 0 is treated as no usable figure: `MemoryAwareConfig::validate`
/// refuses a zero budget, so honouring it would turn "the GPU is momentarily
/// full" into a server that will not start.
pub fn resolve_max_memory(
    explicit: Option<u64>,
    device_available: Option<u64>,
    built_in: u64,
) -> (u64, BudgetSource) {
    match (explicit, device_available) {
        (Some(v), _) => (v, BudgetSource::Env),
        (None, Some(v)) if v > 0 => (v, BudgetSource::Device),
        (None, _) => (built_in, BudgetSource::BuiltIn),
    }
}

/// Bytes the serving device reports available, `None` when it cannot say.
pub fn device_available_bytes(device: &fuel::Device) -> Option<u64> {
    device
        .as_dyn()
        .as_backend_runtime()
        .and_then(|r| r.available_bytes())
}

/// Whether a `WontFit` refuses the load (PM ruling 2026-10-30). VRAM is a hard
/// limit, so on a GPU it refuses unless `force` is set; free RAM is soft (page
/// cache, swap, mmap'd weights), so on the CPU it only warns.
pub(crate) fn refuses_on_wont_fit(location: DeviceLocation, force: bool) -> bool {
    !force && !matches!(location, DeviceLocation::Cpu)
}

/// [`enforce`] with the runtime and the refusal rule supplied, so tests need
/// no real device.
pub(crate) fn enforce_with(
    model_path: &Path,
    weights_bytes: u64,
    runtime: Option<&dyn BackendRuntime>,
    refuse: bool,
) -> anyhow::Result<()> {
    match check_with(weights_bytes, runtime) {
        Preflight::Fits => tracing::info!(
            "memory preflight: {} of weights fit ({})",
            format_bytes(weights_bytes as usize),
            model_path.display()
        ),
        Preflight::Tight => tracing::warn!(
            "memory preflight: {} of weights fit but leave the device tight ({})",
            format_bytes(weights_bytes as usize),
            model_path.display()
        ),
        Preflight::Unknown => tracing::debug!(
            "memory preflight: backend reports no memory figure; not checking ({})",
            model_path.display()
        ),
        Preflight::WontFit { needed, available } if refuse => bail!(
            "model weights need at least {} but the device reports {} available \
             (set LIGHTBULB_ALLOW_OVERSIZE_MODEL=1 to load anyway)",
            format_bytes(needed as usize),
            format_bytes(available as usize)
        ),
        Preflight::WontFit { needed, available } => tracing::warn!(
            "memory preflight: weights need at least {} but the device reports {} available; \
             loading anyway ({})",
            format_bytes(needed as usize),
            format_bytes(available as usize),
            model_path.display()
        ),
    }
    Ok(())
}

/// Refuse the load when the device says the weights cannot fit, per
/// [`refuses_on_wont_fit`]. Never refuses on "no signal".
pub(crate) fn enforce(model_path: &Path, device: &fuel::Device, force: bool) -> anyhow::Result<()> {
    let Some(bytes) = weights_lower_bound(model_path) else {
        return Ok(());
    };
    enforce_with(
        model_path,
        bytes,
        device.as_dyn().as_backend_runtime(),
        refuses_on_wont_fit(device.as_dyn().location_dyn(), force),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mock(Option<u64>, Option<u64>);
    impl BackendRuntime for Mock {
        fn available_bytes(&self) -> Option<u64> {
            self.0
        }
        fn total_bytes(&self) -> Option<u64> {
            self.1
        }
    }

    #[test]
    fn an_explicit_budget_beats_the_device_and_unset_differs_from_set() {
        let built_in = 8 << 30;
        let (set, set_src) = resolve_max_memory(Some(5), Some(9), built_in);
        let (unset, unset_src) = resolve_max_memory(None, Some(9), built_in);
        assert_eq!((set, set_src), (5, BudgetSource::Env));
        assert_eq!((unset, unset_src), (9, BudgetSource::Device));
        assert_ne!(set, unset, "positive control: env must change the value");
    }

    #[test]
    fn with_no_env_and_no_device_figure_the_built_in_default_is_kept() {
        assert_eq!(
            resolve_max_memory(None, None, 123),
            (123, BudgetSource::BuiltIn)
        );
    }

    #[test]
    fn a_device_answer_of_zero_falls_back_rather_than_yielding_a_budget_validate_refuses() {
        assert_eq!(
            resolve_max_memory(None, Some(0), 123),
            (123, BudgetSource::BuiltIn)
        );
        // An explicit 0 is the operator's word and is passed through unchanged
        // (validate then reports it by name).
        assert_eq!(
            resolve_max_memory(Some(0), Some(9), 123),
            (0, BudgetSource::Env)
        );
    }

    #[test]
    fn a_gguf_file_counts_its_own_size() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("m.gguf");
        std::fs::write(&f, vec![0u8; 123]).unwrap();
        assert_eq!(weights_lower_bound(&f), Some(123));
    }

    #[test]
    fn a_safetensors_dir_sums_its_shards() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), vec![0u8; 40]).unwrap();
        assert_eq!(weights_lower_bound(dir.path()), Some(40));

        let sharded = tempfile::tempdir().unwrap();
        std::fs::write(sharded.path().join("a.safetensors"), vec![0u8; 10]).unwrap();
        std::fs::write(sharded.path().join("b.safetensors"), vec![0u8; 25]).unwrap();
        std::fs::write(
            sharded.path().join("model.safetensors.index.json"),
            r#"{"metadata":{},"weight_map":{"x":"a.safetensors","y":"b.safetensors"}}"#,
        )
        .unwrap();
        assert_eq!(weights_lower_bound(sharded.path()), Some(35));
    }

    #[test]
    fn an_unreadable_path_is_none_so_the_loader_reports_it() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(weights_lower_bound(&dir.path().join("missing.gguf")), None);
        assert_eq!(weights_lower_bound(dir.path()), None);
    }

    #[test]
    fn wont_fit_refuses_with_both_figures_in_the_message() {
        let rt = Mock(Some(1_000), Some(10_000));
        let err = enforce_with(Path::new("m.gguf"), 5_000, Some(&rt), true).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("need at least")
                && msg.contains("available")
                && msg.contains("LIGHTBULB_ALLOW_OVERSIZE_MODEL"),
            "{msg}"
        );
    }

    #[test]
    fn wont_fit_without_the_refuse_flag_warns_and_proceeds() {
        let rt = Mock(Some(1_000), Some(10_000));
        assert!(enforce_with(Path::new("m.gguf"), 5_000, Some(&rt), false).is_ok());
    }

    #[test]
    fn only_a_gpu_without_the_override_refuses() {
        assert!(!refuses_on_wont_fit(DeviceLocation::Cpu, false));
        assert!(!refuses_on_wont_fit(DeviceLocation::Cpu, true));
        for gpu in [
            DeviceLocation::Cuda { gpu_id: 0 },
            DeviceLocation::Metal { gpu_id: 0 },
            DeviceLocation::Vulkan { gpu_id: 0 },
        ] {
            assert!(refuses_on_wont_fit(gpu, false), "{gpu:?} must refuse");
            assert!(
                !refuses_on_wont_fit(gpu, true),
                "{gpu:?} + override must proceed"
            );
        }
    }

    #[test]
    fn tight_and_unknown_and_no_runtime_proceed() {
        let tight = Mock(Some(80), Some(100));
        assert!(enforce_with(Path::new("m"), 70, Some(&tight), true).is_ok());
        let blind = Mock(None, None);
        assert!(enforce_with(Path::new("m"), u64::MAX, Some(&blind), true).is_ok());
        assert!(enforce_with(Path::new("m"), u64::MAX, None, true).is_ok());
    }

    /// The real CPU device answers (or honestly declines) without panicking,
    /// and 1 byte of weights is never refused.
    #[test]
    fn the_real_cpu_device_never_refuses_one_byte() {
        let dev = fuel::Device::cpu();
        assert!(enforce_with(Path::new("m"), 1, dev.as_dyn().as_backend_runtime(), true).is_ok());
    }
}
