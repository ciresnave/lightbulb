// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pre-load memory check for the Fuel serving path (ROADMAP B9).
//!
//! Before a loader allocates, ask the serving device (through Fuel's
//! `BackendRuntime`) whether the weights can fit. The figure used is the
//! bytes the weights occupy once Fuel has loaded them (B17): for a GGUF,
//! computed from the tensor directory, because Fuel keeps only `Q4_0`
//! quantized and dequantizes every other dtype to f32 (Fuel PR #530 audit,
//! 2026-10-30), so a Q4_K_M file is about 4x its size on disk. Otherwise the
//! on-disk size, a LOWER bound. The KV cache and activations come on top, so
//! a pass is not a promise. **Fuel row 124 (K-quants stay quantized) makes
//! the GGUF estimate too high; delete `gguf_resident_bytes` then.**

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

/// GGML type code of `Q4_0`, the one dtype Fuel's GGUF loader keeps quantized.
const GGML_Q4_0: u32 = 2;

/// Bytes of one `Q4_0` block (32 weights: an f16 scale and 16 nibble bytes).
const Q4_0_BLOCK_BYTES: u64 = 18;

/// Bytes a GGUF's tensors occupy after Fuel's loader has run, from
/// `(element count, GGML type code)` pairs: `Q4_0` as stored, every other
/// dtype as f32 (4 bytes per element). Remove when Fuel keeps K-quants
/// quantized (Fuel ROADMAP row 124).
pub(crate) fn gguf_resident_bytes(tensors: impl IntoIterator<Item = (u64, u32)>) -> u64 {
    tensors.into_iter().fold(0u64, |sum, (elements, ty)| {
        let bytes = if ty == GGML_Q4_0 {
            elements.div_ceil(32).saturating_mul(Q4_0_BLOCK_BYTES)
        } else {
            elements.saturating_mul(4)
        };
        sum.saturating_add(bytes)
    })
}

/// [`gguf_resident_bytes`] over the tensor directory of the GGUF at `path`;
/// `None` when the directory cannot be read.
fn gguf_resident_bytes_of(path: &Path) -> Option<u64> {
    let content = crate::gguf::Content::read(path).ok()?;
    let infos = content.lightning_tensor_infos().ok()?;
    Some(gguf_resident_bytes(infos.iter().map(|t| {
        (
            t.dimensions.iter().fold(1u64, |n, d| n.saturating_mul(*d)),
            t.tensor_type,
        )
    })))
}

/// Bytes the weights at `path` will occupy on the serving device: for a
/// `.gguf` the larger of the on-disk size and the resident estimate (never
/// below the lower bound), else [`weights_lower_bound`].
pub(crate) fn weights_resident_estimate(path: &Path) -> Option<u64> {
    let on_disk = weights_lower_bound(path)?;
    if path.is_file() {
        if let Some(resident) = gguf_resident_bytes_of(path) {
            return Some(resident.max(on_disk));
        }
    }
    Some(on_disk)
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
    let Some(bytes) = weights_resident_estimate(model_path) else {
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

    /// A GGUF with one tensor of `dims` and GGML type `ty`, `data_len` data bytes.
    fn tiny_gguf(dir: &Path, dims: &[u64], ty: u32, data_len: usize) -> std::path::PathBuf {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // tensors
        b.extend_from_slice(&0u64.to_le_bytes()); // metadata kvs
        let name = b"w";
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name);
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // offset
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend(std::iter::repeat_n(0u8, data_len));
        let f = dir.join("m.gguf");
        std::fs::write(&f, b).unwrap();
        f
    }

    #[test]
    fn q4_0_stays_quantized_and_every_other_dtype_counts_as_f32() {
        // 4096 weights: Q4_0 = 128 blocks x 18 B; Q4_K (12), Q8_0 (8), f16 (1), f32 (0) = 4 B each.
        assert_eq!(gguf_resident_bytes([(4096, 2)]), 128 * 18);
        for ty in [12, 8, 1, 0, 14] {
            assert_eq!(gguf_resident_bytes([(4096, ty)]), 4096 * 4, "type {ty}");
        }
        assert_eq!(
            gguf_resident_bytes([(4096, 2), (4096, 12)]),
            128 * 18 + 4096 * 4
        );
    }

    #[test]
    fn a_q4_k_m_8b_is_refused_by_resident_bytes_not_on_disk_size() {
        // ~8B weights: 4.9 GB on disk as Q4_K_M, ~32 GB once Fuel dequantizes to f32.
        let elements = 8_000_000_000u64;
        let on_disk = 4_900_000_000u64;
        let resident = gguf_resident_bytes([(elements, 12)]);
        let rt = Mock(Some(24 << 30), Some(24 << 30)); // a 24 GiB GPU
        assert!(
            enforce_with(Path::new("m.gguf"), on_disk, Some(&rt), true).is_ok(),
            "positive control: the on-disk figure passes, which is the B17 defect"
        );
        let err = enforce_with(Path::new("m.gguf"), resident, Some(&rt), true).unwrap_err();
        assert!(format!("{err:#}").contains("need at least"), "{err:#}");
    }

    #[test]
    fn the_estimate_reads_the_tensor_directory_of_a_real_gguf() {
        let dir = tempfile::tempdir().unwrap();
        // 256x2 Q4_K: 2 super-blocks of 144 B on disk, 512 x 4 B resident.
        let f = tiny_gguf(dir.path(), &[256, 2], 12, 288);
        let on_disk = std::fs::metadata(&f).unwrap().len();
        let est = weights_resident_estimate(&f).unwrap();
        assert_eq!(est, 2048.max(on_disk));
        assert!(
            est > on_disk,
            "resident {est} must exceed on-disk {on_disk}"
        );
        // Q4_0 stays as stored, so the figure falls back to the on-disk floor.
        let g = tiny_gguf(dir.path(), &[64, 1], 2, 36);
        let on_disk_q = std::fs::metadata(&g).unwrap().len();
        assert_eq!(weights_resident_estimate(&g), Some(on_disk_q));
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
