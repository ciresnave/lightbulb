//! Model loaders and helpers for local, offline operation

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

/// Discover all .safetensors files under a directory (non-recursive), sorted by name.
pub fn find_safetensors_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = vec![];
    for entry in fs::read_dir(dir).with_context(|| format!("read_dir {dir:?}"))? {
        let entry = entry?;
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "safetensors" {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

/// Map string dtype to Candle DType
fn parse_dtype(dtype: Option<&str>) -> Result<DType> {
    match dtype {
        None => Ok(DType::F32),
        Some("f32") => Ok(DType::F32),
        Some("bf16") => Ok(DType::BF16),
        Some("f16") => Ok(DType::F16),
        Some(x) => bail!("Unsupported dtype: {x}"),
    }
}

/// Load a local LLaMA family model from a directory containing:
/// - config.json
/// - one or more model.safetensors files
/// Returns the model, its cache, config, and the selected device (CPU by default).
pub fn load_local_llama(
    model_dir: &str,
    dtype: Option<&str>,
    use_kv_cache: bool,
    use_flash_attn: bool,
) -> Result<(
    candle_transformers::models::llama::Llama,
    candle_transformers::models::llama::Cache,
    candle_transformers::models::llama::Config,
    Device,
)> {
    use candle_transformers::models::llama::{Llama, LlamaConfig};

    let dir = Path::new(model_dir);
    if !dir.is_dir() {
        bail!("model_dir is not a directory: {model_dir}");
    }
    let config_path = dir.join("config.json");
    let config_bytes = fs::read(&config_path)
        .with_context(|| format!("reading config.json at {config_path:?}"))?;
    let raw_cfg: LlamaConfig =
        serde_json::from_slice(&config_bytes).context("parsing LLaMA config.json")?;
    let cfg = raw_cfg.into_config(use_flash_attn);

    let files = find_safetensors_files(dir)?;
    if files.is_empty() {
        bail!("No .safetensors files found in {model_dir}");
    }

    let device = Device::Cpu;
    let dtype = parse_dtype(dtype)?;
    let cache = candle_transformers::models::llama::Cache::new(use_kv_cache, dtype, &cfg, &device)?;
    // SAFETY: from_mmaped_safetensors uses memory-mapped immutable files; paths come from local dir.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let model = Llama::load(vb, &cfg)?;
    Ok((model, cache, cfg, device))
}
