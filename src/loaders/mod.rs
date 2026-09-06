//! Model loaders and helpers for local, offline operation

pub mod awq;

use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use candlelight::core::{DType, Device};
use candlelight::nn::VarBuilder;
use candlelight::transformers::models::quantized_llama::ModelWeights as QuantizedLlamaWeights;

// Re-export AWQ types
pub use awq::{AwqConfig, AwqLinear, load_awq_metadata, should_quantize};

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
///
/// **Every arm is pinned individually by the tests at the bottom of this file,
/// and must stay that way.** `f16` and `bf16` are both two bytes wide with
/// different exponent/mantissa splits, so swapping those two arms parses
/// cleanly, allocates every buffer at the correct size, computes identical
/// tensor offsets, and errors nowhere — the model simply emits garbage that
/// reads as a *quality* problem. The loader is never a suspect because the
/// loader never complained. Width, byte-size and "it parsed" are all invariant
/// to that swap; only the identity of the returned variant discriminates it.
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
/// Returns the model, its cache, config, the selected device (CPU by default), and optional name mapper.
pub fn load_local_llama(
    model_dir: &str,
    dtype: Option<&str>,
    use_kv_cache: bool,
    use_flash_attn: bool,
) -> Result<(
    candlelight::transformers::models::llama::Llama,
    candlelight::transformers::models::llama::Cache,
    candlelight::transformers::models::llama::Config,
    Device,
    Option<crate::pruning::name_mapping::TensorNameMapper>,
)> {
    use candlelight::transformers::models::llama::{Llama, LlamaConfig};

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

    // Load weights through candlelight's own mmap VarBuilder — the same
    // pattern `load_awq_llama` uses below.
    //
    // This replaced MLMF, which could never have worked here: MLMF (all
    // published versions, including the v0.2.1 tag the old comment named as
    // "compatible") depends on candle-core/candle-nn 0.9.2 *directly*, while
    // candlelight is on 0.10.2. `loaded.var_builder` was therefore a
    // different `VarBuilder` type than `Llama::load` accepts. MLMF also
    // requires `protoc` at build time, which broke the build outright.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };

    // TODO: Add architecture detection using TensorNameMapper
    let name_mapper = None;

    let cache =
        candlelight::transformers::models::llama::Cache::new(use_kv_cache, dtype, &cfg, &device)?;

    let model = Llama::load(vb, &cfg)?;
    Ok((model, cache, cfg, device, name_mapper))
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
/// Tuple of (quantized_model, config, tokenizer, device, name_mapper)
///
/// Note: Quantized models don't use a separate KV cache object - the cache
/// is built into the model's forward pass.
///
/// # UNREACHABLE FROM THIS CRATE, AND THAT NEARLY COST A DAY
///
/// **Nothing calls this.** The GGUF serving path runs through
/// [`crate::model::parallel_model_manager::ParallelModelManager::load_gguf`],
/// reached from `engine::model_runner`. This function is kept only because it
/// is `pub` in a `pub mod` and removing it would break any external consumer.
///
/// It matters because **both functions call `Content::extract_tokenizer`, and
/// nothing in either name says which one is live.** While fixing a defect in
/// that tokenizer, applying the fix here instead of to the live path would
/// have produced an unchanged end-to-end failure — which reads as evidence
/// against a correct diagnosis. **A correct fix applied to the wrong caller is
/// indistinguishable from a wrong fix**, and it arrives disguised as a
/// refutation.
///
/// So: if you change tokenizer or config handling, change it in
/// `parallel_model_manager`, and treat any edit here as documentation of a
/// path no test exercises. Verify by call site, not by name.
pub fn load_gguf_llama(
    gguf_path: &str,
) -> Result<(
    QuantizedLlamaWeights,
    candlelight::transformers::models::llama::Config,
    tokenizers::Tokenizer,
    Device,
    Option<crate::pruning::name_mapping::TensorNameMapper>,
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
    let gguf_content = crate::gguf::Content::read(gguf_path)
        .context("Failed to load GGUF with memory-mapped loader")?;

    // Extract tokenizer from GGUF metadata
    let tokenizer = gguf_content
        .extract_tokenizer()
        .context("Failed to extract tokenizer from GGUF")?;

    // Extract config from GGUF metadata
    let metadata = gguf_content.metadata();
    let config = extract_llama_config_from_metadata(metadata)?;

    // Build name mapper for architecture-agnostic tensor loading
    use crate::pruning::name_mapping::TensorNameMapper;
    let tensor_names: Vec<String> = gguf_content.tensor_infos().keys().cloned().collect();
    let name_mapper = match TensorNameMapper::from_tensor_names(&tensor_names) {
        Ok(mapper) => {
            println!("🔍 Detected model architecture: {:?}", mapper.architecture);
            println!("  - Layer count: {}", mapper.layer_indices.len());
            Some(mapper)
        }
        Err(e) => {
            println!("⚠️  Warning: Failed to build tensor name mapper: {}", e);
            println!("  Proceeding without architecture awareness");
            None
        }
    };

    // Load quantized model weights using Candle's loader
    // (MLMF doesn't yet support loading into Candle's quantized format)
    let mut file =
        File::open(path).with_context(|| format!("Failed to open GGUF file: {}", gguf_path))?;

    let candle_content = candlelight::core::quantized::gguf_file::Content::read(&mut file)
        .context("Failed to read GGUF file content with Candle")?;

    let model = QuantizedLlamaWeights::from_gguf(candle_content, &mut file, &device)
        .context("Failed to load quantized model from GGUF")?;

    println!("✓ Quantized model loaded successfully");
    println!("  - Hidden size: {}", config.hidden_size);
    println!("  - Num layers: {}", config.num_hidden_layers);
    println!("  - Num heads: {}", config.num_attention_heads);
    println!("  - Vocab size: {}", config.vocab_size);

    Ok((model, config, tokenizer, device, name_mapper))
}

/// Extract LLaMA config from GGUF metadata
fn extract_llama_config_from_metadata(
    metadata: &std::collections::HashMap<String, candlelight::core::quantized::gguf_file::Value>,
) -> Result<candlelight::transformers::models::llama::Config> {
    use candlelight::core::quantized::gguf_file::Value;

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

    // Refuse on the DECLARED architecture rather than on a missing key. See
    // `crate::gguf::require_llama_architecture` for why this is not a prefix
    // substitution and why the check has one implementation.
    //
    // ⚠️ THIS FUNCTION IS ON THE UNREACHABLE PATH — `load_gguf_llama`'s own doc
    // comment says nothing calls it, and the live GGUF config read is in
    // `parallel_model_manager::load_gguf`. Both call the same helper, so a
    // future edit cannot fix one and silently leave the other.
    crate::gguf::require_llama_architecture(metadata)?;

    // Extract standard LLaMA config fields from GGUF metadata
    // GGUF uses different key naming than HuggingFace config.json
    let hidden_size = get_u64("llama.embedding_length")? as usize;
    let intermediate_size = get_u64("llama.feed_forward_length")? as usize;
    let num_hidden_layers = get_u64("llama.block_count")? as usize;
    let num_attention_heads = get_u64("llama.attention.head_count")? as usize;
    let num_key_value_heads = get_u64("llama.attention.head_count_kv")? as usize;

    // ⚠️ `llama.vocab_size` IS ABSENT ON TINYLLAMA, this repo's primary
    // checkpoint, and reading it with `?` made this function fail on the one
    // GGUF the project actually serves. Measured 2026-09-05 over the local
    // corpus: absent on tinyllama-1.1b-chat-v1.0.Q4_0 and
    // tinyllamas-stories-260k, present on all nine SmolLM2 builds. It is also
    // absent under EVERY non-llama prefix in a 13-architecture sample, so a
    // prefix substitution alone would have relocated the failure rather than
    // ending it.
    //
    // The fallback chain mirrors the working one in
    // `src/model/parallel_model_manager.rs`, which is why serving TinyLlama
    // succeeds while this path did not: same job, two implementations, one of
    // them correct.
    let vocab_size = get_u64("llama.vocab_size")
        .or_else(|_| get_u64("llama.n_vocab"))
        .map(|v| v as usize)
        .or_else(|_| match metadata.get("tokenizer.ggml.tokens") {
            Some(Value::Array(tokens)) => Ok(tokens.len()),
            _ => bail!(
                "could not determine vocab_size: tried llama.vocab_size, llama.n_vocab, \
                 and counting tokenizer.ggml.tokens"
            ),
        })?;

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

    Ok(candlelight::transformers::models::llama::Config {
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
            .map(|id| candlelight::transformers::models::llama::LlamaEosToks::Single(id)),
        rope_scaling: None,
        tie_word_embeddings: false,
    })
}

/// Load an AWQ-quantized model from safetensors files
///
/// This loader supports AWQ 4-bit quantized models (like Qwen3-32B-AWQ) and uses
/// Marlin CUDA kernels for efficient inference.
///
/// # Arguments
/// * `model_dir` - Path to directory containing config.json and model safetensors files
/// * `dtype` - Data type for activations (f16 or bf16, weights are always 4-bit)
/// * `use_flash_attn` - Whether to use flash attention (not yet supported for AWQ)
///
/// # Returns
/// Tuple of (awq_model, config, device, name_mapper)
///
/// Note: AWQ models use KV cache built into the model's forward pass
pub fn load_awq_llama(
    model_dir: &str,
    dtype: Option<&str>,
    use_flash_attn: bool,
) -> Result<(
    crate::model::AwqQwen3,
    crate::model::Qwen3Config,
    Device,
    Option<crate::pruning::name_mapping::TensorNameMapper>,
)> {
    use crate::model::{AwqQwen3, Qwen3Config};

    let dir = Path::new(model_dir);
    if !dir.is_dir() {
        bail!("model_dir is not a directory: {model_dir}");
    }

    println!("Loading AWQ model from: {}", model_dir);

    // Load and validate AWQ metadata
    let (config_json, awq_config) = awq::load_awq_metadata(dir)?;

    // Parse Qwen3 config
    let config: Qwen3Config = serde_json::from_value(config_json)
        .context("Failed to parse Qwen3Config from config.json")?;

    println!("\nModel configuration:");
    println!("  - Architecture: Qwen3");
    println!("  - Hidden size: {}", config.hidden_size);
    println!("  - Num layers: {}", config.num_hidden_layers);
    println!("  - Num heads: {}", config.num_attention_heads);
    println!("  - Num KV heads: {}", config.num_key_value_heads);
    println!("  - Vocab size: {}", config.vocab_size);
    println!("  - Intermediate size: {}", config.intermediate_size);

    // Find safetensors files
    let files = find_safetensors_files(dir)?;
    if files.is_empty() {
        bail!("No .safetensors files found in {model_dir}");
    }
    println!("\nFound {} safetensors files", files.len());

    // Use CUDA if available (AWQ requires CUDA for Marlin kernels)
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    if !matches!(device, Device::Cuda(_)) {
        bail!("AWQ inference requires CUDA. CPU inference not yet implemented.");
    }
    println!("Loading model on device: {:?}", device);

    // Parse dtype (AWQ uses f16/bf16 for activations, 4-bit for weights)
    let dtype = parse_dtype(dtype)?;
    if !matches!(dtype, DType::F16 | DType::BF16) {
        bail!("AWQ only supports F16 or BF16 activations, got {:?}", dtype);
    }

    // Build name mapper for architecture detection
    use crate::pruning::name_mapping::TensorNameMapper;
    let name_mapper = {
        let mut all_tensor_names = Vec::new();
        for safetensors_path in &files {
            let tensors = candlelight::core::safetensors::load(safetensors_path, &device)
                .with_context(|| format!("Failed to load safetensors: {:?}", safetensors_path))?;
            all_tensor_names.extend(tensors.keys().cloned());
        }

        match TensorNameMapper::from_tensor_names(&all_tensor_names) {
            Ok(mapper) => {
                println!("🔍 Detected model architecture: {:?}", mapper.architecture);
                println!("  - Layer count: {}", mapper.layer_indices.len());
                Some(mapper)
            }
            Err(e) => {
                println!("⚠️  Warning: Failed to build tensor name mapper: {}", e);
                println!("  Proceeding without architecture awareness");
                None
            }
        }
    };

    // Create VarBuilder from safetensors files
    println!("\nLoading model weights...");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };

    // Load AWQ model
    let model = AwqQwen3::new(&config, vb)
        .map_err(|e| anyhow::anyhow!("Failed to create AWQ Qwen3 model: {}", e))?;

    println!("✓ AWQ model loaded successfully");
    println!(
        "  - Quantization: 4-bit AWQ (group_size={})",
        awq_config.group_size
    );
    println!("  - Activation dtype: {:?}", dtype);
    println!(
        "  - Memory: ~{} GB (estimated)",
        (config.hidden_size * config.num_hidden_layers * 4) / 1024 / 1024 / 1024
    );

    Ok((model, config, device, name_mapper))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `parse_dtype` arm, pinned to its exact variant.
    ///
    /// Asserts the IDENTITY of the returned `DType`, never its width. The
    /// defect this guards against is a swap between two equal-width variants,
    /// which every size-based, allocation-based or "it returned Ok" assertion
    /// is invariant to. Cases are listed one per line rather than looped so a
    /// failure names the offending string.
    #[test]
    fn parse_dtype_pins_every_arm_to_its_exact_variant() {
        assert_eq!(parse_dtype(Some("f32")).unwrap(), DType::F32);
        assert_eq!(parse_dtype(Some("f16")).unwrap(), DType::F16);
        assert_eq!(parse_dtype(Some("bf16")).unwrap(), DType::BF16);
    }

    /// `f16` and `bf16` are the equal-width pair, so they get their own guard.
    ///
    /// Kept separate from the arm sweep above because this is the assertion
    /// with a known failure mode rather than a completeness check: these two
    /// are mutually substitutable everywhere except in the bits they mean.
    #[test]
    fn f16_and_bf16_are_not_interchangeable() {
        let f16 = parse_dtype(Some("f16")).unwrap();
        let bf16 = parse_dtype(Some("bf16")).unwrap();
        assert_ne!(f16, bf16, "f16 and bf16 must not map to the same DType");
        assert_eq!(
            f16,
            DType::F16,
            "f16 must map to F16, not merely to some 2-byte type"
        );
        assert_eq!(
            bf16,
            DType::BF16,
            "bf16 must map to BF16, not merely to some 2-byte type"
        );
    }

    /// An absent dtype defaults to F32.
    ///
    /// Separate from the arm sweep because `None` is a different input class,
    /// not another string: it is the path taken by every caller that does not
    /// specify one, so a change here silently re-types every default load.
    #[test]
    fn an_absent_dtype_defaults_to_f32() {
        assert_eq!(parse_dtype(None).unwrap(), DType::F32);
    }

    /// An unrecognised dtype is an error naming the offending string, not a
    /// silent fallback to F32.
    ///
    /// A fallback here would be the same shape as the defect above: a typo
    /// (`"fp16"`, `"float16"`) would load at the wrong precision with no
    /// complaint.
    #[test]
    fn an_unrecognised_dtype_is_an_error_naming_the_string() {
        let err = parse_dtype(Some("fp16")).expect_err("fp16 is not a supported dtype");
        assert!(
            err.to_string().contains("fp16"),
            "error must name the rejected string, got: {err}"
        );
    }
}

#[cfg(test)]
mod gguf_config_tests {
    use super::extract_llama_config_from_metadata;
    use candlelight::core::quantized::gguf_file::Value;
    use std::collections::HashMap;

    /// A minimal llama header: every key the extractor requires, and nothing
    /// else. `llama.vocab_size` is deliberately ABSENT, because that is the
    /// shape TinyLlama actually has.
    fn llama_header() -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert("general.architecture".into(), Value::String("llama".into()));
        m.insert("llama.embedding_length".into(), Value::U32(2048));
        m.insert("llama.feed_forward_length".into(), Value::U32(5632));
        m.insert("llama.block_count".into(), Value::U32(22));
        m.insert("llama.attention.head_count".into(), Value::U32(32));
        m.insert("llama.attention.head_count_kv".into(), Value::U32(4));
        m
    }

    /// ⚠️ BORN-RED AGAINST THE REPO'S OWN PRIMARY CHECKPOINT.
    ///
    /// `llama.vocab_size` is absent on `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` --
    /// measured over the local corpus, absent there and on
    /// tinyllamas-stories-260k, present on all nine SmolLM2 builds. Reading it
    /// with `?` made this PUBLIC loader fail on the one GGUF the project
    /// actually serves, and nothing noticed because `load_gguf_llama` has no
    /// caller anywhere in the repo.
    #[test]
    fn vocab_size_falls_back_to_the_token_count() {
        let mut m = llama_header();
        m.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array(vec![Value::String("a".into()); 32000]),
        );
        let config = extract_llama_config_from_metadata(&m)
            .expect("a llama header without llama.vocab_size must still load");
        assert_eq!(
            config.vocab_size, 32000,
            "vocab_size must fall back to the length of tokenizer.ggml.tokens"
        );
    }

    /// And when there is no token list either, the failure names all three
    /// things it tried rather than one key.
    #[test]
    fn an_underivable_vocab_size_names_every_source_it_tried() {
        let err = extract_llama_config_from_metadata(&llama_header())
            .expect_err("no vocab_size and no token list cannot succeed")
            .to_string();
        for expected in ["llama.vocab_size", "llama.n_vocab", "tokenizer.ggml.tokens"] {
            assert!(
                err.contains(expected),
                "the error must name {expected}, so a reader knows what was tried: {err}"
            );
        }
    }

    /// ⚠️ THE REFUSAL MUST NAME THE ARCHITECTURE, NOT A MISSING KEY.
    ///
    /// Before this change a qwen2 header died with
    /// `Missing or invalid metadata key: llama.embedding_length`, which sends a
    /// reader hunting for a corrupt GGUF. The old message is asserted ABSENT so
    /// a regression to key-shaped reporting reddens here.
    #[test]
    fn a_non_llama_architecture_is_refused_by_name() {
        let mut m = HashMap::new();
        m.insert("general.architecture".into(), Value::String("qwen2".into()));
        m.insert("qwen2.embedding_length".into(), Value::U32(896));
        m.insert("qwen2.block_count".into(), Value::U32(24));
        let err = extract_llama_config_from_metadata(&m)
            .expect_err("a qwen2 header must be refused")
            .to_string();
        assert!(
            err.contains("qwen2"),
            "the refusal must name the declared architecture: {err}"
        );
        assert!(
            !err.contains("Missing or invalid metadata key"),
            "the refusal must not report a missing key -- the key is not missing, it is \
             under a different prefix, and the key-shaped message is the defect: {err}"
        );
    }

    /// An absent declaration is a fact about the FILE and is reported as one.
    #[test]
    fn a_header_with_no_architecture_says_so() {
        let mut m = llama_header();
        m.remove("general.architecture");
        let err = extract_llama_config_from_metadata(&m)
            .expect_err("no architecture must be refused")
            .to_string();
        assert!(
            err.contains("general.architecture"),
            "the error must name the key that is genuinely absent: {err}"
        );
    }

    /// The control: a complete llama header still reads exactly as before, so
    /// the architecture gate is not rejecting the case it must accept.
    #[test]
    fn a_llama_header_still_loads() {
        let mut m = llama_header();
        m.insert("llama.vocab_size".into(), Value::U32(32000));
        let config = extract_llama_config_from_metadata(&m).expect("llama must still load");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 22);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.vocab_size, 32000);
    }
}
