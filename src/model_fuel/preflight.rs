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
//!
//! The KV pool is added to the figure for non-hybrid models (see
//! `kv_pool_bytes`): the serving path allocates it as soon as the model is
//! loaded, so weights alone under-count what the device must hold.

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

/// Bytes per KV element: the serving path runs f32 activations and keeps its
/// KV cache in the same dtype (`check_dtype` refuses anything else).
const KV_ELEMENT_BYTES: u64 = 4;

/// Per-token KV geometry of a non-hybrid model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvShape {
    pub layers: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
}

impl KvShape {
    /// Bytes of K and V for one token across every layer.
    pub(crate) fn bytes_per_token(&self) -> u64 {
        self.layers
            .saturating_mul(2)
            .saturating_mul(self.kv_heads)
            .saturating_mul(self.head_dim)
            .saturating_mul(KV_ELEMENT_BYTES)
    }
}

/// Bytes of the KV pool the engine allocates for `max_batch_size` sessions of
/// up to `context_length` tokens: `num_blocks * KV_BLOCK_SIZE` token slots
/// (`BatchedServeConfig::new`), each holding K and V for every layer.
pub(crate) fn kv_pool_bytes(shape: KvShape, max_batch_size: usize, context_length: usize) -> u64 {
    let block = super::scheduled_llama_engine::KV_BLOCK_SIZE as u64;
    let blocks =
        (max_batch_size.max(1) as u64).saturating_mul((context_length as u64).div_ceil(block));
    let tokens = blocks.saturating_mul(block);
    tokens.saturating_mul(shape.bytes_per_token())
}

/// KV geometry from a GGUF's own metadata, for the architectures whose KV
/// exists in every layer (`llama`, `qwen3`). `None` for anything else
/// (including the hybrid `qwen35`, whose KV budget is L4) and for any key that
/// is missing or not a scalar: no figure means no KV term, never a refusal.
fn kv_shape_of_gguf(content: &crate::gguf::Content) -> Option<KvShape> {
    use crate::gguf::{GgufArchitecture, metadata_u64};
    let md = content.metadata();
    let arch = match crate::gguf::detect_architecture(md).ok()? {
        GgufArchitecture::Llama => "llama",
        GgufArchitecture::Qwen3 => "qwen3",
        GgufArchitecture::Qwen35 => return None,
    };
    let key = |k: &str| metadata_u64(md, &format!("{arch}.{k}"));
    let heads = key("attention.head_count").ok().filter(|h| *h > 0)?;
    Some(KvShape {
        layers: key("block_count").ok()?,
        kv_heads: key("attention.head_count_kv").unwrap_or(heads),
        head_dim: gguf_head_dim(arch, heads, key)?,
    })
}

/// Head dimension as the loader for `arch` derives it: the Llama loader
/// divides, the Qwen3 loader reads `key_length`.
fn gguf_head_dim(arch: &str, heads: u64, key: impl Fn(&str) -> anyhow::Result<u64>) -> Option<u64> {
    let key_length = (arch == "qwen3")
        .then(|| key("attention.key_length").ok())
        .flatten();
    key_length.or_else(|| key("embedding_length").ok().map(|e| e / heads))
}

/// KV geometry from a SafeTensors directory's `config.json` (the Llama-shape
/// loader's fields); `None` when unreadable or incomplete.
fn kv_shape_of_config_json(dir: &Path) -> Option<KvShape> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let get = |k: &str| v.get(k).and_then(serde_json::Value::as_u64);
    let heads = get("num_attention_heads").filter(|h| *h > 0)?;
    Some(KvShape {
        layers: get("num_hidden_layers")?,
        kv_heads: get("num_key_value_heads").unwrap_or(heads),
        head_dim: get("head_dim").or_else(|| get("hidden_size").map(|h| h / heads))?,
    })
}

/// [`KvShape`] of the model at `path`, `None` when it has no figure.
pub(crate) fn kv_shape(path: &Path) -> Option<KvShape> {
    if path.is_file() {
        let content = crate::gguf::Content::read(path).ok()?;
        kv_shape_of_gguf(&content)
    } else {
        kv_shape_of_config_json(path)
    }
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

/// Total memory Fuel's inventory lists for the device at `location`, from
/// `infos` (`model_fuel::device::enumerate`). `None` for the CPU, for a device
/// the list does not name, and for a listed 0 ("could not say").
pub(crate) fn listed_total_bytes(
    location: DeviceLocation,
    infos: &[super::device::DeviceInfo],
) -> Option<u64> {
    let DeviceLocation::Cuda { gpu_id } = location else {
        return None;
    };
    infos
        .iter()
        .find(|d| d.backend == "cuda" && u64::from(d.index) == gpu_id as u64)
        .map(|d| d.total_bytes)
        .filter(|t| *t > 0)
}

/// Classify `needed` bytes on a device. The runtime's own verdict wins. When it
/// has no free-bytes figure (`Unknown`) but the device's total is known, a need
/// above the total still cannot fit, so that is `WontFit`; the second value
/// names which figure the verdict compared against.
pub(crate) fn classify(
    needed: u64,
    runtime: Option<&dyn BackendRuntime>,
    device_total: Option<u64>,
) -> (Preflight, &'static str) {
    match (check_with(needed, runtime), device_total) {
        (Preflight::Unknown, Some(total)) if needed > total => (
            Preflight::WontFit {
                needed,
                available: total,
            },
            "in total",
        ),
        (verdict, _) => (verdict, "available"),
    }
}

/// [`enforce`] with the runtime and the refusal rule supplied, so tests need
/// no real device. Tests only: production goes through [`enforce_with_total`].
#[cfg(test)]
pub(crate) fn enforce_with(
    model_path: &Path,
    weights_bytes: u64,
    kv_bytes: u64,
    runtime: Option<&dyn BackendRuntime>,
    refuse: bool,
) -> anyhow::Result<()> {
    enforce_with_total(model_path, weights_bytes, kv_bytes, runtime, None, refuse)
}

/// [`enforce_with`] plus the device's listed total, the fallback when the
/// runtime reports no free-bytes figure.
pub(crate) fn enforce_with_total(
    model_path: &Path,
    weights_bytes: u64,
    kv_bytes: u64,
    runtime: Option<&dyn BackendRuntime>,
    device_total: Option<u64>,
    refuse: bool,
) -> anyhow::Result<()> {
    let total = weights_bytes.saturating_add(kv_bytes);
    let what = format!(
        "{} (weights {} + KV cache {})",
        format_bytes(total as usize),
        format_bytes(weights_bytes as usize),
        format_bytes(kv_bytes as usize)
    );
    let (verdict, basis) = classify(total, runtime, device_total);
    report(model_path, &what, verdict, basis, refuse)
}

/// Log the verdict, or refuse when it is `WontFit` and `refuse` is set.
fn report(
    model_path: &Path,
    what: &str,
    verdict: Preflight,
    basis: &str,
    refuse: bool,
) -> anyhow::Result<()> {
    match verdict {
        Preflight::Fits => {
            tracing::info!("memory preflight: {what} fit ({})", model_path.display())
        }
        Preflight::Tight => tracing::warn!(
            "memory preflight: {what} fit but leave the device tight ({})",
            model_path.display()
        ),
        Preflight::Unknown => tracing::debug!(
            "memory preflight: backend reports no memory figure; not checking ({})",
            model_path.display()
        ),
        Preflight::WontFit { available, .. } if refuse => bail!(
            "model needs at least {what} but the device reports {} {basis}              (set LIGHTBULB_ALLOW_OVERSIZE_MODEL=1 to load anyway)",
            format_bytes(available as usize)
        ),
        Preflight::WontFit { available, .. } => tracing::warn!(
            "memory preflight: model needs at least {what} but the device reports {} {basis};              loading anyway ({})",
            format_bytes(available as usize),
            model_path.display()
        ),
    }
    Ok(())
}

/// Refuse the load when the device says the weights cannot fit, per
/// [`refuses_on_wont_fit`]. Never refuses on "no signal".
pub(crate) fn enforce(
    model_path: &Path,
    device: &fuel::Device,
    force: bool,
    max_batch_size: usize,
    context_length: usize,
) -> anyhow::Result<()> {
    let Some(bytes) = weights_resident_estimate(model_path) else {
        return Ok(());
    };
    let kv_bytes = kv_shape(model_path).map_or(0, |shape| {
        kv_pool_bytes(shape, max_batch_size, context_length)
    });
    let location = device.as_dyn().location_dyn();
    let total = listed_total_bytes(location, &super::device::enumerate());
    enforce_with_total(
        model_path,
        bytes,
        kv_bytes,
        device.as_dyn().as_backend_runtime(),
        total,
        refuses_on_wont_fit(location, force),
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
            enforce_with(Path::new("m.gguf"), on_disk, 0, Some(&rt), true).is_ok(),
            "positive control: the on-disk figure passes, which is the B17 defect"
        );
        let err = enforce_with(Path::new("m.gguf"), resident, 0, Some(&rt), true).unwrap_err();
        assert!(format!("{err:#}").contains("needs at least"), "{err:#}");
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

    /// A GGUF with `arch` metadata (`u32` keys under `<arch>.`) and one tiny tensor.
    fn gguf_with_meta(dir: &Path, arch: &str, keys: &[(&str, u32)]) -> std::path::PathBuf {
        fn s(b: &mut Vec<u8>, v: &str) {
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            b.extend_from_slice(v.as_bytes());
        }
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // tensors
        b.extend_from_slice(&(keys.len() as u64 + 1).to_le_bytes()); // metadata kvs
        s(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        s(&mut b, arch);
        for (k, v) in keys {
            s(&mut b, &format!("{arch}.{k}"));
            b.extend_from_slice(&4u32.to_le_bytes());
            b.extend_from_slice(&v.to_le_bytes());
        }
        s(&mut b, "w");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&32u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend(std::iter::repeat_n(0u8, 128));
        let f = dir.join(format!("{arch}.gguf"));
        std::fs::write(&f, b).unwrap();
        f
    }

    const LLAMA_8B: &[(&str, u32)] = &[
        ("block_count", 32),
        ("attention.head_count", 32),
        ("attention.head_count_kv", 8),
        ("embedding_length", 4096),
    ];

    #[test]
    fn the_kv_pool_is_the_engines_blocks_times_k_and_v_for_every_layer() {
        let shape = KvShape {
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
        };
        // 32 x 2 x 8 x 128 x 4 B = 256 KiB per token; 2 sessions x 4096 tokens.
        assert_eq!(kv_pool_bytes(shape, 2, 4096), 2 * 4096 * 256 * 1024);
        // The pool rounds each session up to whole 16-token blocks.
        assert_eq!(kv_pool_bytes(shape, 1, 17), 32 * 256 * 1024);
        // The engine clamps a batch of 0 to 1, so the estimate does too.
        assert_eq!(kv_pool_bytes(shape, 0, 16), 16 * 256 * 1024);
    }

    #[test]
    fn the_kv_shape_is_read_from_gguf_metadata_for_llama_and_qwen3() {
        let dir = tempfile::tempdir().unwrap();
        let llama = gguf_with_meta(dir.path(), "llama", LLAMA_8B);
        assert_eq!(
            kv_shape(&llama),
            Some(KvShape {
                layers: 32,
                kv_heads: 8,
                head_dim: 128
            })
        );
        // Qwen3 declares `key_length`, which wins over embedding / heads.
        let qwen3 = gguf_with_meta(
            dir.path(),
            "qwen3",
            &[
                ("block_count", 36),
                ("attention.head_count", 32),
                ("attention.head_count_kv", 8),
                ("attention.key_length", 128),
                ("embedding_length", 2560),
            ],
        );
        assert_eq!(
            kv_shape(&qwen3),
            Some(KvShape {
                layers: 36,
                kv_heads: 8,
                head_dim: 128
            })
        );
    }

    #[test]
    fn no_kv_figure_for_a_hybrid_an_unknown_architecture_or_a_missing_key() {
        let dir = tempfile::tempdir().unwrap();
        // Hybrid: only 16 of 64 layers hold KV (L4), so no per-layer figure is claimed.
        let hybrid = gguf_with_meta(dir.path(), "qwen35", LLAMA_8B);
        assert_eq!(kv_shape(&hybrid), None);
        let other = gguf_with_meta(dir.path(), "mamba", LLAMA_8B);
        assert_eq!(kv_shape(&other), None);
        // Control: the same keys under `llama` do give a figure.
        assert!(kv_shape(&gguf_with_meta(dir.path(), "llama", LLAMA_8B)).is_some());
        let no_layers = gguf_with_meta(dir.path(), "llama", &LLAMA_8B[1..]);
        assert_eq!(kv_shape(&no_layers), None);
        assert_eq!(kv_shape(&dir.path().join("absent.gguf")), None);
    }

    #[test]
    fn the_kv_shape_is_read_from_a_safetensors_config_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"num_hidden_layers": 22, "num_attention_heads": 32,
                "num_key_value_heads": 4, "hidden_size": 2048}"#,
        )
        .unwrap();
        assert_eq!(
            kv_shape(dir.path()),
            Some(KvShape {
                layers: 22,
                kv_heads: 4,
                head_dim: 64
            })
        );
        // GQA key absent: every head has its own KV.
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"num_hidden_layers": 2, "num_attention_heads": 8, "hidden_size": 64}"#,
        )
        .unwrap();
        assert_eq!(kv_shape(dir.path()).map(|s| s.kv_heads), Some(8));
    }

    /// Both sides of the new rule: the same weights on the same device load
    /// without a KV term and are refused with one, and a small context
    /// brings the KV term back under the line.
    #[test]
    fn the_kv_pool_can_turn_a_pass_into_a_refusal_and_a_small_context_turns_it_back() {
        let shape = KvShape {
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
        };
        let weights = 10u64 << 30;
        let rt = Mock(Some(12 << 30), Some(12 << 30)); // a 12 GiB GPU
        let big = kv_pool_bytes(shape, 1, 32_768); // 8 GiB
        let small = kv_pool_bytes(shape, 1, 1_024); // 256 MiB
        let m = Path::new("m.gguf");
        assert!(
            enforce_with(m, weights, 0, Some(&rt), true).is_ok(),
            "weights alone fit: this is what used to pass"
        );
        let err = enforce_with(m, weights, big, Some(&rt), true).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("KV cache 8.00 GB") && msg.contains("weights 10.00 GB"),
            "{msg}"
        );
        assert!(
            enforce_with(m, weights, small, Some(&rt), true).is_ok(),
            "a short context keeps the same model loadable"
        );
        // CPU rule unchanged: the same overage only warns.
        assert!(enforce_with(m, weights, big, Some(&rt), false).is_ok());
    }

    #[test]
    fn wont_fit_refuses_with_both_figures_in_the_message() {
        let rt = Mock(Some(1_000), Some(10_000));
        let err = enforce_with(Path::new("m.gguf"), 5_000, 0, Some(&rt), true).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("needs at least")
                && msg.contains("available")
                && msg.contains("LIGHTBULB_ALLOW_OVERSIZE_MODEL"),
            "{msg}"
        );
    }

    #[test]
    fn wont_fit_without_the_refuse_flag_warns_and_proceeds() {
        let rt = Mock(Some(1_000), Some(10_000));
        assert!(enforce_with(Path::new("m.gguf"), 5_000, 0, Some(&rt), false).is_ok());
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
        assert!(enforce_with(Path::new("m"), 70, 0, Some(&tight), true).is_ok());
        let blind = Mock(None, None);
        assert!(enforce_with(Path::new("m"), u64::MAX, 0, Some(&blind), true).is_ok());
        assert!(enforce_with(Path::new("m"), u64::MAX, 0, None, true).is_ok());
    }

    /// The real CPU device answers (or honestly declines) without panicking,
    /// and 1 byte of weights is never refused.
    #[test]
    fn the_real_cpu_device_never_refuses_one_byte() {
        let dev = fuel::Device::cpu();
        assert!(
            enforce_with(
                Path::new("m"),
                1,
                0,
                dev.as_dyn().as_backend_runtime(),
                true
            )
            .is_ok()
        );
    }

    fn listed(index: u32, mib: u64) -> super::super::device::DeviceInfo {
        super::super::device::DeviceInfo {
            backend: "cuda".into(),
            index,
            name: "P40".into(),
            total_bytes: mib << 20,
        }
    }

    #[test]
    fn the_listed_total_is_found_by_the_device_actually_chosen() {
        let infos = [listed(0, 8_192), listed(1, 24_576)];
        let at = |gpu_id| listed_total_bytes(DeviceLocation::Cuda { gpu_id }, &infos);
        assert_eq!(at(1), Some(24_576 << 20));
        assert_eq!(at(0), Some(8_192 << 20));
        assert_eq!(at(2), None, "a device the list does not name");
        assert_eq!(listed_total_bytes(DeviceLocation::Cpu, &infos), None);
        let unknown = [listed(0, 0)];
        assert_eq!(
            listed_total_bytes(DeviceLocation::Cuda { gpu_id: 0 }, &unknown),
            None
        );
    }

    #[test]
    fn with_no_free_figure_a_need_above_the_total_is_refused_and_one_below_is_not() {
        let blind = Mock(None, None);
        let total = Some(24u64 << 30);
        let m = Path::new("m.gguf");
        let err = enforce_with_total(m, 30 << 30, 0, Some(&blind), total, true).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("24.00 GB in total"), "{msg}");
        assert!(
            enforce_with_total(m, 20 << 30, 0, Some(&blind), total, true).is_ok(),
            "control: below the total there is no evidence of a misfit"
        );
        assert!(
            enforce_with_total(m, 30 << 30, 0, Some(&blind), None, true).is_ok(),
            "control: with no listed total the old no-signal rule holds"
        );
        assert!(enforce_with_total(m, 30 << 30, 0, Some(&blind), total, false).is_ok());
    }

    #[test]
    fn a_runtime_verdict_beats_the_listed_total() {
        let rt = Mock(Some(1_000), Some(10_000));
        let (v, basis) = classify(5_000, Some(&rt), Some(1 << 40));
        assert_eq!(
            v,
            Preflight::WontFit {
                needed: 5_000,
                available: 1_000
            }
        );
        assert_eq!(basis, "available");
    }
}
