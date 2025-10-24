//! Model loaders and helpers for local, offline operation

use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::quantized_llama::ModelWeights as QuantizedLlamaWeights;

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

    // Use CUDA if available, otherwise fall back to CPU
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Loading model on device: {:?}", device);

    let dtype = parse_dtype(dtype)?;
    let cache = candle_transformers::models::llama::Cache::new(use_kv_cache, dtype, &cfg, &device)?;
    // SAFETY: from_mmaped_safetensors uses memory-mapped immutable files; paths come from local dir.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let model = Llama::load(vb, &cfg)?;
    Ok((model, cache, cfg, device))
}

/// Load a quantized LLaMA model from a GGUF file
///
/// This loader supports quantized GGUF files (Q4_0, Q4_1, Q4_K, Q8_0, etc.)
/// and provides 2-4x memory savings compared to fp16/fp32 models.
///
/// # Arguments
/// * `gguf_path` - Path to the .gguf file
///
/// # Returns
/// Tuple of (quantized_model, config, tokenizer, device)
///
/// Note: Quantized models don't use a separate KV cache object - the cache
/// is built into the model's forward pass.
pub fn load_gguf_llama(
    gguf_path: &str,
) -> Result<(
    QuantizedLlamaWeights,
    candle_transformers::models::llama::Config,
    tokenizers::Tokenizer,
    Device,
)> {
    use crate::gguf;

    let path = Path::new(gguf_path);
    if !path.exists() {
        bail!("GGUF file not found: {}", gguf_path);
    }

    println!("Loading quantized model from: {}", gguf_path);

    // Use CUDA if available, otherwise CPU
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Loading model on device: {:?}", device);

    // Load GGUF using our memory-mapped loader for metadata and tokenizer
    let gguf_content =
        gguf::Content::read(gguf_path).context("Failed to load GGUF with memory-mapped loader")?;

    // Extract tokenizer from GGUF metadata
    let tokenizer = gguf_content
        .extract_tokenizer()
        .context("Failed to extract tokenizer from GGUF")?;

    // Extract config from GGUF metadata
    let metadata = gguf_content.metadata();
    let config = extract_llama_config_from_metadata(metadata)?;

    // Load quantized model weights using Candle's loader
    // (We still need Candle's loader for the actual weight tensors)
    let mut file =
        File::open(path).with_context(|| format!("Failed to open GGUF file: {}", gguf_path))?;

    let candle_content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .context("Failed to read GGUF file content with Candle")?;

    let model = QuantizedLlamaWeights::from_gguf(candle_content, &mut file, &device)
        .context("Failed to load quantized model from GGUF")?;

    println!("✓ Quantized model loaded successfully");
    println!("  - Hidden size: {}", config.hidden_size);
    println!("  - Num layers: {}", config.num_hidden_layers);
    println!("  - Num heads: {}", config.num_attention_heads);
    println!("  - Vocab size: {}", config.vocab_size);

    Ok((model, config, tokenizer, device))
}

/// Extract LLaMA config from GGUF metadata
fn extract_llama_config_from_metadata(
    metadata: &std::collections::HashMap<String, candle_core::quantized::gguf_file::Value>,
) -> Result<candle_transformers::models::llama::Config> {
    use candle_core::quantized::gguf_file::Value;

    // Helper to get u64 from metadata
    let get_u64 = |key: &str| -> Result<u64> {
        match metadata.get(key) {
            Some(Value::U64(v)) => Ok(*v),
            Some(Value::U32(v)) => Ok(*v as u64),
            _ => bail!("Missing or invalid metadata key: {}", key),
        }
    };

    // Helper to get f32 from metadata
    let get_f32 = |key: &str| -> Result<f32> {
        match metadata.get(key) {
            Some(Value::F32(v)) => Ok(*v),
            _ => bail!("Missing or invalid metadata key: {}", key),
        }
    };

    // Extract standard LLaMA config fields from GGUF metadata
    // GGUF uses different key naming than HuggingFace config.json
    let hidden_size = get_u64("llama.embedding_length")? as usize;
    let intermediate_size = get_u64("llama.feed_forward_length")? as usize;
    let num_hidden_layers = get_u64("llama.block_count")? as usize;
    let num_attention_heads = get_u64("llama.attention.head_count")? as usize;
    let num_key_value_heads = get_u64("llama.attention.head_count_kv")? as usize;
    let vocab_size = get_u64("llama.vocab_size")? as usize;

    let rms_norm_eps = get_f32("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);

    let rope_theta = get_f32("llama.rope.freq_base").unwrap_or(10000.0);

    // Use max context length if available, otherwise default
    let max_position_embeddings = get_u64("llama.context_length").unwrap_or(2048) as usize;

    // Extract token IDs (optional fields)
    let bos_token_id = metadata
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| match v {
            Value::U32(id) => Some(*id),
            _ => None,
        });

    let eos_token_id = metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| match v {
            Value::U32(id) => Some(*id),
            _ => None,
        });

    Ok(candle_transformers::models::llama::Config {
        hidden_size,
        intermediate_size,
        vocab_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        use_flash_attn: false,
        rms_norm_eps: rms_norm_eps as f64,
        rope_theta,
        max_position_embeddings,
        bos_token_id,
        eos_token_id: eos_token_id
            .map(|id| candle_transformers::models::llama::LlamaEosToks::Single(id)),
        rope_scaling: None,
        tie_word_embeddings: false,
    })
}
