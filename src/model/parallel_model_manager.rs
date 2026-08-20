//! Parallel model manager with true batched inference
//!
//! This is the production-ready implementation using:
//! - BatchedTransformer for parallel forward passes (safetensors AND GGUF)
//! - QuantizableLinear for transparent quantization support
//! - ParallelCacheBuilder + ParallelKvCache for efficient per-slot cache management  
//! - ChunkedPrefillScheduler for optimal prefill batching with padding
//!
//! Expected performance improvements over sequential model_manager:
//! - CPU: 5-10x faster
//! - GPU: 10-50x faster

use anyhow::{Context, Result};
use candlelight::core::{DType, Device, IndexOp, Tensor};
use candlelight::transformers::models::llama::{Config, LlamaEosToks};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

use crate::cache::{PrefixCacheConfig, PrefixKvCache};
use crate::engine::{ParallelCacheBuilder, ParallelKvCache, RequestContext, RequestState};
use crate::hardware::{BatchSizeConfig, RuntimeBatchAdjuster};
use crate::loaders::load_local_llama;
use crate::model::{
    BatchMetadata, BatchedTransformer, BatchedTransformerConfig, ChunkedPrefillConfig,
    ChunkedPrefillScheduler, PrefillRequest,
};

/// Performance statistics for parallel batched inference
#[derive(Debug, Clone, Default)]
pub struct ParallelBatchStats {
    pub total_batches: usize,
    pub total_requests_processed: usize,
    pub total_tokens_generated: usize,
    pub total_forward_time_ms: f64,
    pub prefill_batches: usize,
    pub decode_batches: usize,
    pub chunked_prefill_batches: usize,
    pub max_batch_size: usize,
    pub total_padding_tokens: usize,
}

impl ParallelBatchStats {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn average_batch_size(&self) -> f64 {
        if self.total_batches == 0 {
            0.0
        } else {
            self.total_requests_processed as f64 / self.total_batches as f64
        }
    }

    pub fn tokens_per_second(&self) -> f64 {
        if self.total_forward_time_ms == 0.0 {
            0.0
        } else {
            self.total_tokens_generated as f64 / (self.total_forward_time_ms / 1000.0)
        }
    }

    pub fn padding_efficiency(&self) -> f64 {
        let total_tokens = self.total_tokens_generated + self.total_padding_tokens;
        if total_tokens == 0 {
            1.0
        } else {
            self.total_tokens_generated as f64 / total_tokens as f64
        }
    }
}

/// Parallel model manager using BatchedTransformer (supports both safetensors and GGUF)
pub struct ParallelModelManager {
    model: BatchedTransformer,
    cache_builder: ParallelCacheBuilder,
    caches: Vec<ParallelKvCache>, // Per-layer KV caches
    tokenizer: Tokenizer,
    config: Config,
    chunked_prefill_config: ChunkedPrefillConfig,
    device: Device,
    stats: ParallelBatchStats,
    cache_index_pool: Vec<bool>, // true = in use, false = available
    runtime_adjuster: Option<RuntimeBatchAdjuster>, // Dynamic batch size adjustment
    prefix_cache: PrefixKvCache, // Prefix KV cache for shared prompt prefixes
    // Segmented KV cache fields
    segmented_config: Option<crate::cache::SegmentedCacheConfig>,
    /// Maps cache_idx → open ModelGeneration span for active decode requests
    active_generation_spans: std::collections::HashMap<usize, crate::cache::SpanId>,
    /// Tiered storage for demoted KV cache segments (VRAM → RAM → Disk)
    tiered_storage: Option<crate::cache::TieredStorageManager>,
    /// Per-token tracker for hybrid eviction (Phase 4). Only one active at a time.
    per_token_tracker: Option<crate::cache::PerTokenTracker>,
    /// Tool call detector for mid-reasoning context injection (CR.1).
    tool_call_detector: Option<crate::engine::tool_call::ToolCallDetector>,
}

impl ParallelModelManager {
    /// Load a model for parallel batched inference
    ///
    /// # Arguments
    /// * `model_dir` - Path to model directory
    /// * `max_batch_size` - Maximum concurrent requests
    /// * `context_length` - Context window size
    /// * `dtype_str` - Data type ("f32", "f16", "bf16")
    /// * `chunked_prefill_config` - Optional config for chunked prefill (uses default if None)
    pub fn load(
        model_dir: impl AsRef<Path>,
        max_batch_size: usize,
        context_length: usize,
        dtype_str: Option<&str>,
        chunked_prefill_config: Option<ChunkedPrefillConfig>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Load config and setup using existing loader (but we'll ignore the Llama model)
        let (_llama_model, _cache, config, device, _name_mapper) = load_local_llama(
            model_dir.to_str().unwrap_or(""),
            dtype_str,
            true,  // use_kv_cache
            false, // use_flash_attn
        )?;

        // Parse dtype
        let dtype = match dtype_str {
            Some("f32") | None => DType::F32,
            Some("f16") => DType::F16,
            Some("bf16") => DType::BF16,
            _ => DType::F32,
        }; // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Create ParallelCacheBuilder and caches
        let num_layers = config.num_hidden_layers;
        let _num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads; // For GQA
        let head_dim = config.hidden_size / config.num_attention_heads;

        // Create cache builder - tracks position for each request slot
        // Note: Cache should be created with num_kv_heads, not num_heads
        // because K/V tensors have num_kv_heads dimension in GQA models
        // The head expansion from num_kv_heads -> num_heads happens in attention layer
        let cache_builder =
            ParallelCacheBuilder::new(max_batch_size, context_length, dtype, &device)?;

        // Create per-layer KV caches
        let mut caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            caches.push(cache_builder.make_cache(num_kv_heads, head_dim)?);
        }

        // Create BatchedTransformer configuration
        let transformer_config = BatchedTransformerConfig {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            rope_scaling: None, // TODO: Convert Llama3RopeConfig to HashMap if needed
            sliding_window: None,
            tie_word_embeddings: config.tie_word_embeddings,
            use_flash_attn: false,
            multi_gpu: None, // M3.6: Can be configured via separate API
        };

        // Load model weights
        let vb = unsafe {
            candlelight::nn::VarBuilder::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                dtype,
                &device,
            )?
        };

        let model = BatchedTransformer::new(transformer_config, vb)?;

        let chunked_prefill_config = chunked_prefill_config.unwrap_or_default();

        // Enable runtime monitoring by default with reasonable defaults
        let batch_config = BatchSizeConfig::default();
        let runtime_adjuster = Some(RuntimeBatchAdjuster::new(max_batch_size, batch_config));

        // Initialize prefix KV cache with default config
        // For benchmarks with short prompts, temporarily lower min_prefix_len to 1
        // so we can observe cache hit/miss behavior in tests. This is a safe,
        // local change for diagnostics and can be reverted or made configurable.
        let mut prefix_cache_config = PrefixCacheConfig::default();
        prefix_cache_config.min_prefix_len = 1;
        let prefix_cache = PrefixKvCache::new(prefix_cache_config);

        Ok(Self {
            model,
            cache_builder,
            caches,
            tokenizer,
            config,
            chunked_prefill_config,
            device,
            stats: ParallelBatchStats::new(),
            cache_index_pool: vec![false; max_batch_size],
            runtime_adjuster,
            prefix_cache,
            segmented_config: None,
            active_generation_spans: std::collections::HashMap::new(),
            tiered_storage: None,
            per_token_tracker: None,
            tool_call_detector: None,
        })
    }

    /// Load a quantized GGUF model for parallel batched inference
    ///
    /// # Arguments
    /// * `gguf_path` - Path to GGUF model file
    /// * `max_batch_size` - Maximum concurrent requests
    /// * `context_length` - Context window size
    /// * `device` - Device to load model on (None = auto-detect: CUDA if available, else CPU)
    /// * `chunked_prefill_config` - Optional config for chunked prefill (uses default if None)
    pub fn load_gguf(
        gguf_path: impl AsRef<Path>,
        max_batch_size: usize,
        context_length: usize,
        device: Option<Device>,
        chunked_prefill_config: Option<ChunkedPrefillConfig>,
    ) -> Result<Self> {
        use candlelight::core::quantized::gguf_file::Value;
        use std::fs::File;
        use std::io::BufReader;

        let gguf_path = gguf_path.as_ref();

        // Auto-detect device: prefer CUDA, fallback to CPU
        let device = device.unwrap_or_else(|| {
            if Device::cuda_if_available(0).is_ok() {
                println!("  🎮 Using CUDA GPU (device 0)");
                Device::cuda_if_available(0).unwrap()
            } else {
                println!("  💻 Using CPU (CUDA not available)");
                Device::Cpu
            }
        });

        // Load GGUF file with mmap
        let content = crate::gguf::Content::read(gguf_path)?;

        // Extract config from GGUF metadata
        let metadata = content.metadata();

        // Helper to get u64 from metadata
        let get_u64 = |key: &str| -> Result<u64> {
            match metadata.get(key) {
                Some(Value::U64(v)) => Ok(*v),
                Some(Value::U32(v)) => Ok(*v as u64),
                _ => anyhow::bail!("Missing or invalid metadata key: {}", key),
            }
        };

        // Helper to get f32 from metadata
        let get_f32 = |key: &str| -> Result<f32> {
            match metadata.get(key) {
                Some(Value::F32(v)) => Ok(*v),
                _ => anyhow::bail!("Missing or invalid metadata key: {}", key),
            }
        };

        // Extract LLaMA config from GGUF metadata with intelligent fallbacks

        // Hidden size (required)
        let hidden_size = get_u64("llama.embedding_length")
            .or_else(|_| get_u64("llama.n_embd"))
            .map_err(|_| {
                anyhow::anyhow!(
                    "Could not determine hidden_size (tried: llama.embedding_length, llama.n_embd)"
                )
            })? as usize;

        // Number of layers (required)
        let num_hidden_layers = get_u64("llama.block_count")
            .or_else(|_| get_u64("llama.n_layer"))
            .map_err(|_| {
                anyhow::anyhow!(
                    "Could not determine num_hidden_layers (tried: llama.block_count, llama.n_layer)"
                )
            })? as usize;

        // Attention heads (required)
        let num_attention_heads = get_u64("llama.attention.head_count")
            .or_else(|_| get_u64("llama.n_head"))
            .map_err(|_| {
                anyhow::anyhow!(
                    "Could not determine num_attention_heads (tried: llama.attention.head_count, llama.n_head)"
                )
            })? as usize;

        // KV heads (fallback to num_attention_heads for non-GQA models)
        let num_key_value_heads = get_u64("llama.attention.head_count_kv")
            .or_else(|_| get_u64("llama.n_head_kv"))
            .unwrap_or(num_attention_heads as u64) as usize;

        // Intermediate/FFN size (fallback to 4x hidden_size, standard for LLaMA)
        let intermediate_size = get_u64("llama.feed_forward_length")
            .or_else(|_| get_u64("llama.n_ff"))
            .unwrap_or((hidden_size * 4) as u64) as usize;

        // Vocab size (try metadata, fallback to counting tokens)
        let vocab_size = get_u64("llama.vocab_size")
            .or_else(|_| get_u64("llama.n_vocab"))
            .ok()
            .or_else(|| {
                // Fallback: count tokens in tokenizer metadata
                metadata.get("tokenizer.ggml.tokens").and_then(|v| match v {
                    Value::Array(arr) => Some(arr.len() as u64),
                    _ => None,
                })
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Could not determine vocab_size (tried: llama.vocab_size, llama.n_vocab, tokenizer.ggml.tokens count)"
                )
            })? as usize;

        // RMS norm epsilon (common defaults: 1e-5 for LLaMA, 1e-6 for some models)
        let rms_norm_eps = get_f32("llama.attention.layer_norm_rms_epsilon")
            .or_else(|_| get_f32("llama.attention.layer_norm_epsilon"))
            .or_else(|_| get_f32("llama.norm_eps"))
            .unwrap_or(1e-5);

        // RoPE theta (10000.0 for LLaMA 1/2, 500000.0 for LLaMA 3+)
        let rope_theta = get_f32("llama.rope.freq_base")
            .or_else(|_| get_f32("llama.rope_freq_base"))
            .unwrap_or(10000.0);

        // Max position embeddings / context length
        let max_position_embeddings = get_u64("llama.context_length")
            .or_else(|_| get_u64("llama.n_ctx"))
            .or_else(|_| get_u64("llama.n_ctx_train"))
            .unwrap_or(2048) as usize;

        // Token IDs (optional)
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

        // Log detected configuration
        println!("  ✓ Detected model config:");
        println!("    - hidden_size: {}", hidden_size);
        println!("    - num_layers: {}", num_hidden_layers);
        println!(
            "    - num_heads: {} (kv_heads: {})",
            num_attention_heads, num_key_value_heads
        );
        println!("    - vocab_size: {}", vocab_size);
        println!("    - intermediate_size: {}", intermediate_size);
        println!("    - context_length: {}", max_position_embeddings);
        println!("    - rope_theta: {}", rope_theta);
        println!();

        // Build Config struct
        let config = candlelight::transformers::models::llama::Config {
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
        };

        // Extract tokenizer
        let tokenizer = content.extract_tokenizer()?;

        // Build name mapper for architecture-agnostic tensor loading
        use crate::pruning::name_mapping::TensorNameMapper;
        let tensor_names: Vec<String> = content.tensor_infos().keys().cloned().collect();
        let name_mapper = TensorNameMapper::from_tensor_names(&tensor_names)
            .context("Failed to build tensor name mapper")?;

        println!(
            "🔍 Detected model architecture: {:?}",
            name_mapper.architecture
        );
        println!();

        // Wrap name mapper in Arc for sharing
        let name_mapper_arc = std::sync::Arc::new(name_mapper.clone());

        // Create cache builder and per-layer KV caches
        let head_dim = hidden_size / num_attention_heads;

        let mut cache_builder = ParallelCacheBuilder::new(
            max_batch_size,
            context_length,
            DType::F32, // GGUF quantized weights will be dequantized for cache
            &device,
        )?;

        // Enable architecture-aware cache management (M5.1)
        cache_builder.set_name_mapper(Some(name_mapper_arc.clone()));

        // Verify layer count detection
        if let Some(detected_layers) = cache_builder.detected_layer_count() {
            if detected_layers != num_hidden_layers {
                println!(
                    "⚠️  Layer count mismatch: config={}, detected={}",
                    num_hidden_layers, detected_layers
                );
            }
        }

        let mut caches = Vec::with_capacity(num_hidden_layers);
        for _ in 0..num_hidden_layers {
            caches.push(cache_builder.make_cache(num_key_value_heads, head_dim)?);
        }

        // Build BatchedTransformer config
        let transformer_config = BatchedTransformerConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rms_norm_eps: rms_norm_eps as f64,
            rope_theta,
            rope_scaling: None,
            sliding_window: None,
            tie_word_embeddings: false,
            use_flash_attn: false,
            multi_gpu: None, // M3.6: Can be configured via separate API
        };

        // Open file for tensor loading
        let mut file = BufReader::new(File::open(gguf_path)?);

        // Load model from GGUF with quantization
        let model = BatchedTransformer::from_gguf(
            transformer_config,
            &content,
            &mut file,
            &device,
            &name_mapper,
        )?;

        let chunked_prefill_config = chunked_prefill_config.unwrap_or_default();

        // Enable runtime monitoring by default
        let batch_config = BatchSizeConfig::default();
        let runtime_adjuster = Some(RuntimeBatchAdjuster::new(max_batch_size, batch_config));

        // Initialize prefix KV cache
        let prefix_cache_config = PrefixCacheConfig {
            enabled: true,
            min_prefix_len: 1,
            max_size_mb: 512,
            max_prefix_len: context_length,
        };
        let prefix_cache = PrefixKvCache::new(prefix_cache_config);

        Ok(Self {
            model,
            cache_builder,
            caches,
            tokenizer,
            config,
            chunked_prefill_config,
            device,
            stats: ParallelBatchStats::new(),
            cache_index_pool: vec![false; max_batch_size],
            runtime_adjuster,
            prefix_cache,
            segmented_config: None,
            active_generation_spans: std::collections::HashMap::new(),
            tiered_storage: None,
            per_token_tracker: None,
            tool_call_detector: None,
        })
    }
    /// Load a model with hardware-adaptive configuration
    ///
    /// Automatically detects hardware capabilities and selects optimal batch size.
    /// Falls back to load() defaults if hardware detection fails.
    ///
    /// # Arguments
    /// * `model_dir` - Path to model directory
    /// * `context_length` - Context window size
    /// * `dtype_str` - Data type ("f32", "f16", "bf16", "q8", "q4")
    /// * `chunked_prefill_config` - Optional config for chunked prefill
    pub fn load_adaptive(
        model_dir: impl AsRef<Path>,
        context_length: usize,
        dtype_str: Option<&str>,
        chunked_prefill_config: Option<ChunkedPrefillConfig>,
    ) -> Result<Self> {
        use crate::hardware::{
            HardwareProfile,
            batch_sizing::{BatchSizeConfig, ModelMemoryProfile, calculate_optimal_batch_size},
        };

        let model_dir_ref = model_dir.as_ref();

        // Detect hardware
        let profile = HardwareProfile::detect().unwrap_or_else(|e| {
            eprintln!("Warning: Hardware detection failed: {}", e);
            eprintln!("Using fallback profile with conservative defaults");
            // Minimal fallback profile
            HardwareProfile {
                cpu: crate::hardware::CpuInfo {
                    physical_cores: 4,
                    logical_cores: 8,
                    architecture: "unknown".to_string(),
                    model_name: "Unknown CPU".to_string(),
                },
                memory: crate::hardware::MemoryInfo {
                    total_bytes: 8 * 1024 * 1024 * 1024, // 8GB fallback
                    available_bytes: 6 * 1024 * 1024 * 1024,
                    bandwidth_gbs: Some(25.0),
                },
                gpu: None,
                ml_score: 3.0,
            }
        });

        println!("Hardware detected: {}", profile.summary());

        // Load config using existing loader to get model configuration
        let (_llama_model, _cache, config, _device, _name_mapper) = load_local_llama(
            model_dir_ref.to_str().unwrap_or(""),
            dtype_str,
            true,  // use_kv_cache
            false, // use_flash_attn
        )?;

        // Estimate model memory profile
        let dtype = match dtype_str {
            Some("f32") => crate::hardware::model_selection::DataType::F32,
            Some("f16") => crate::hardware::model_selection::DataType::F16,
            Some("bf16") => crate::hardware::model_selection::DataType::BF16,
            Some("q8") => crate::hardware::model_selection::DataType::Q8,
            Some("q4") => crate::hardware::model_selection::DataType::Q4,
            _ => crate::hardware::model_selection::DataType::F16, // Default
        };

        let bytes_per_param = dtype.bytes_per_param();
        let total_params = config.hidden_size * config.hidden_size * config.num_hidden_layers * 12;
        let weights_bytes = (total_params as f64 * bytes_per_param) as u64;

        let model_profile = ModelMemoryProfile {
            weights_bytes,
            num_layers: config.num_hidden_layers,
            hidden_size: config.hidden_size,
            num_kv_heads: config.num_key_value_heads,
            context_window: context_length,
        };

        // Calculate optimal batch size
        let batch_config = BatchSizeConfig::default();
        let optimal_batch_size = calculate_optimal_batch_size(
            &profile,
            &model_profile,
            2, // Minimum batch size
            Some(batch_config),
        )?;

        // Print adaptive batch size with relevant hardware info
        if let Some(gpu) = &profile.gpu {
            println!(
                "Adaptive batch size: {} (GPU: {}, {} GB VRAM)",
                optimal_batch_size,
                gpu.name,
                gpu.vram_bytes / (1024 * 1024 * 1024)
            );
        } else {
            println!(
                "Adaptive batch size: {} (CPU: {} cores, {} GB RAM)",
                optimal_batch_size,
                profile.cpu.physical_cores,
                profile.memory.total_bytes / (1024 * 1024 * 1024)
            );
        }

        // Use regular load with the adaptive batch size
        Self::load(
            model_dir,
            optimal_batch_size,
            context_length,
            dtype_str,
            chunked_prefill_config,
        )
    }

    // === Segmented KV Cache Configuration ===

    /// Enable the segmented KV cache system.
    ///
    /// This activates:
    /// - Automatic CacheSpan creation for prefill (UserInput) and decode (ModelGeneration)
    /// - Attention weight capture on the last transformer layer
    /// - Segment-level attention logging (if `config.log_span_attention` is true)
    ///
    /// Call this after loading the model but before processing any requests.
    pub fn enable_segmented_cache(&mut self, config: crate::cache::SegmentedCacheConfig) {
        if let Err(e) = config.validate() {
            eprintln!("Invalid SegmentedCacheConfig: {}. Using defaults.", e);
            let config = crate::cache::SegmentedCacheConfig::default();
            self.segmented_config = Some(config);
            return;
        }

        if config.capture_attention {
            // Only capture on last layer to minimize memory overhead
            self.model.set_capture_attention(true, true);
        }

        // Enable H2O + segmented eviction on the cache builder
        self.cache_builder.enable_segmented_eviction(config.clone());

        // Create tiered storage manager if demotion is enabled
        if config.tiered_demotion {
            let max_ram = config.max_ram_demoted_segments;
            let mut storage = crate::cache::TieredStorageManager::new(self.device.clone(), max_ram);

            // Set up disk backend if configured
            if let Some(ref path) = config.disk_storage_path {
                match crate::cache::tiered_storage::FileDiskStore::new(path) {
                    Ok(store) => {
                        storage.set_disk_store(Box::new(store));
                        println!("Disk tier enabled at: {}", path);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to create disk store at {}: {}", path, e);
                    }
                }
            }

            self.tiered_storage = Some(storage);
        }

        self.segmented_config = Some(config);
    }

    /// Generate a summary for a segment being evicted.
    ///
    /// Decodes the token IDs in the evicted range back to text and creates
    /// a truncated summary. For a full model-powered summary, use
    /// `generate_model_summary()` (requires an additional forward pass).
    fn generate_eviction_summary_text(
        &self,
        slot: usize,
        position_range: &std::ops::Range<usize>,
    ) -> String {
        // Try to reconstruct text from the request context
        // Since we don't store per-position token IDs, use a positional summary
        let span_len = position_range.end - position_range.start;
        format!(
            "Context from positions {}..{} ({} tokens, slot {})",
            position_range.start, position_range.end, span_len, slot
        )
    }

    /// Handle eviction pressure: per-token eviction or full segment demotion.
    ///
    /// Called when context utilization exceeds the pressure threshold.
    /// If per-token eviction is enabled and a tracker is active, it processes
    /// individual token evictions. Otherwise, it demotes the lowest-scoring
    /// full segment.
    fn handle_eviction_pressure(
        &mut self,
        config: &crate::cache::SegmentedCacheConfig,
        max_util: f32,
    ) {
        // Phase 4: Update active per-token tracker
        if self.per_token_tracker.is_some() {
            if let Some(h2o) = self.cache_builder.h2o_policy().cloned() {
                let pressure_mod = 1.0 + (max_util - config.eviction_pressure_threshold) * 2.0;

                let tracker = self.per_token_tracker.as_mut().unwrap();
                tracker.update_scores(&h2o, pressure_mod);

                let evictions = tracker.select_evictions(5, config.base_decay_rate * 2.0);

                if !evictions.is_empty() {
                    tracker.mark_evicted(&evictions);
                    let span_id = tracker.target_span;
                    let remaining_after = tracker.remaining_ranges();

                    if let Err(e) = self.cache_builder.partially_evict_span(
                        span_id,
                        &evictions,
                        remaining_after,
                    ) {
                        eprintln!("Warning: Partial eviction failed: {}", e);
                    }
                }

                // If span is >75% evicted, fully evict and drop tracker
                let total = tracker.token_scores.len();
                let evicted = tracker.evicted_count();
                if total > 0 && evicted * 4 >= total * 3 {
                    let span_id = tracker.target_span;
                    let _ = self.cache_builder.evict_span(span_id);
                    self.per_token_tracker = None;
                    println!(
                        "Per-token tracker: span {} fully evicted ({}% tokens removed)",
                        span_id,
                        evicted * 100 / total
                    );
                }
            }
            return; // Tracker is active — don't do full-segment eviction
        }

        // Full segment eviction path
        let candidates = self
            .cache_builder
            .rank_eviction_candidates(config, max_util);

        let (span_id, _score) = match candidates.first() {
            Some(c) => *c,
            None => return,
        };

        // Phase 4: Activate per-token tracking instead of immediate eviction
        if config.per_token_eviction {
            let span_meta = self
                .cache_builder
                .get_span(span_id)
                .map(|s| (s.start_pos, s.end_pos - s.start_pos));

            if let Some((start_pos, span_len)) = span_meta {
                self.per_token_tracker = Some(crate::cache::PerTokenTracker::new(
                    span_id,
                    start_pos,
                    span_len,
                    config.attention_power,
                    config.base_decay_rate,
                ));
                println!(
                    "Per-token tracker activated on span {} ({} tokens)",
                    span_id, span_len
                );
            }
            return;
        }

        // Standard path: demote entire segment
        let span_meta = self
            .cache_builder
            .get_span(span_id)
            .map(|s| (s.slot, s.start_pos..s.end_pos));

        if let Some((slot, position_range)) = span_meta {
            if config.tiered_demotion {
                // Generate summary before borrowing tiered_storage mutably
                let summary = self.generate_eviction_summary_text(slot, &position_range);

                if let Some(ref mut storage) = self.tiered_storage {
                    let span_len = position_range.end - position_range.start;

                    // Extract per-layer KV tensors
                    let mut kv_layers = Vec::new();
                    let mut extraction_ok = true;

                    for cache in self.caches.iter() {
                        match (|| -> candlelight::core::Result<(Tensor, Tensor)> {
                            let k = cache
                                .k()
                                .narrow(0, slot, 1)?
                                .narrow(2, position_range.start, span_len)?
                                .to_device(&Device::Cpu)?;
                            let v = cache
                                .v()
                                .narrow(0, slot, 1)?
                                .narrow(2, position_range.start, span_len)?
                                .to_device(&Device::Cpu)?;
                            Ok((k, v))
                        })() {
                            Ok((k, v)) => kv_layers.push((k, v)),
                            Err(e) => {
                                eprintln!(
                                    "Warning: Failed to extract KV for span {}: {}",
                                    span_id, e
                                );
                                extraction_ok = false;
                                break;
                            }
                        }
                    }

                    if extraction_ok {
                        // Auto-demote oldest RAM segment to disk if full
                        if storage.is_ram_full() {
                            match storage.auto_demote_oldest_to_disk() {
                                Ok(Some(old_id)) => {
                                    println!("Auto-demoted span {} from RAM to disk", old_id);
                                }
                                Ok(None) => {}
                                Err(e) => {
                                    eprintln!("Warning: RAM->disk demotion failed: {}", e);
                                }
                            }
                        }

                        match storage.demote_to_ram(
                            span_id,
                            slot,
                            position_range,
                            kv_layers,
                            summary.clone(),
                            summary,
                            None,
                        ) {
                            Ok(fact_key) => {
                                println!("Demoted span {} to RAM (KB key: {})", span_id, fact_key);
                            }
                            Err(e) => {
                                eprintln!("Warning: Failed to demote span {}: {}", span_id, e);
                            }
                        }
                    }
                }
            }

            // Evict from KV cache (marks metadata + masks positions)
            if let Err(e) = self.cache_builder.evict_span(span_id) {
                eprintln!("Warning: Failed to evict span {}: {}", span_id, e);
            }
        }
    }

    // === Mid-Reasoning Context Injection (CR.1) ===

    /// Enable tool call detection with default patterns.
    ///
    /// Registers common tool call patterns (<tool_call>, `[TOOL_CALL]`, <|tool_call|>)
    /// and enables detection in the decode loop.
    pub fn enable_tool_call_detection(&mut self) {
        self.tool_call_detector = Some(crate::engine::tool_call::ToolCallDetector::with_defaults());
        println!("Tool call detection enabled (3 default patterns)");
    }

    /// Enable tool call detection with a custom detector.
    pub fn set_tool_call_detector(&mut self, detector: crate::engine::tool_call::ToolCallDetector) {
        self.tool_call_detector = Some(detector);
    }

    /// Disable tool call detection.
    pub fn disable_tool_call_detection(&mut self) {
        self.tool_call_detector = None;
    }

    /// Inject a tool result into an active request's KV cache.
    ///
    /// The request must be in `AwaitingToolResult` state (paused after detecting
    /// a tool call, KV cache preserved). The tool result is tokenized and processed
    /// as a mini-prefill against the preserved cache, then the request transitions
    /// back to `Decoding` state.
    ///
    /// This is the core mechanism of CR.1 — the model processes the tool result
    /// in the full attentional context of the reasoning that requested it.
    ///
    /// # Arguments
    ///
    /// * `batch` - The request batch (must contain the paused request)
    /// * `request_idx` - Index of the request in the batch
    /// * `tool_result` - The formatted tool result text to inject
    ///
    /// # How It Works
    ///
    /// 1. Tokenizes the tool result text
    /// 2. Creates a ToolOutput CacheSpan
    /// 3. Runs a mini-prefill: forward pass of result tokens against preserved cache
    /// 4. Advances cache positions by the result token count
    /// 5. Transitions request back to Decoding
    ///
    /// The model's next generated token will attend to both its prior reasoning
    /// AND the injected tool result — as if the result arrived mid-thought.
    pub fn inject_tool_result(
        &mut self,
        batch: &mut [RequestContext],
        request_idx: usize,
        tool_result: &str,
    ) -> Result<()> {
        let ctx = &batch[request_idx];

        // Verify request is in the right state
        if !ctx.state.is_awaiting_tool_result() {
            anyhow::bail!(
                "Request {} is not awaiting tool result (state: {:?})",
                ctx.request.id,
                ctx.state
            );
        }

        let cache_idx = ctx
            .cache_index
            .ok_or_else(|| anyhow::anyhow!("Request has no cache index assigned"))?;

        // Tokenize the tool result
        let result_tokens = self.tokenize(tool_result, false)?;
        if result_tokens.is_empty() {
            // Empty result — just resume decoding
            batch[request_idx].resume_decoding();
            return Ok(());
        }

        let result_len = result_tokens.len();

        // Create ToolOutput span
        if self.is_segmented_cache_enabled() {
            let _ =
                self.cache_builder
                    .begin_span(cache_idx, crate::cache::CacheTag::ToolOutput, None);
        }

        // Create input tensor for mini-prefill
        let input_ids = Tensor::new(result_tokens.as_slice(), &self.device)?;

        // Create metadata for this mini-prefill
        // We're processing result_len tokens for a single request at the current position
        let metadata = crate::model::BatchMetadata::from_chunked_prefill_batch(
            vec![cache_idx],
            vec![result_len],
            vec![result_len],
        );

        // Forward pass: process result tokens against preserved cache
        // This is identical to chunked prefill — tokens processed, KV cache updated,
        // no token generation (we just want the cache populated)
        let _logits = self.model.forward(
            &input_ids,
            &mut self.cache_builder,
            &mut self.caches,
            &metadata,
        )?;

        // Advance cache position by the number of injected tokens
        let old_pos = self.cache_builder.get_position(cache_idx);
        let new_pos = old_pos + result_len;
        self.cache_builder.set_position(cache_idx, new_pos);

        // End ToolOutput span
        if self.is_segmented_cache_enabled() {
            // Find the most recent open span for this slot and close it
            let spans = self.cache_builder.spans_for_slot(cache_idx);
            if let Some(last_span) = spans.last() {
                let span_id = last_span.id;
                let _ = self.cache_builder.end_span(span_id);
            }
        }

        // Update request context
        let ctx = &mut batch[request_idx];
        ctx.position += result_len;
        ctx.resume_decoding();

        println!(
            "Injected {} tool result tokens at position {}..{} for request {}",
            result_len, old_pos, new_pos, ctx.request.id
        );

        Ok(())
    }

    /// Disable the segmented KV cache system.
    pub fn disable_segmented_cache(&mut self) {
        self.model.set_capture_attention(false, false);
        self.segmented_config = None;
        self.active_generation_spans.clear();
    }

    /// Check if segmented cache is enabled.
    pub fn is_segmented_cache_enabled(&self) -> bool {
        self.segmented_config
            .as_ref()
            .map(|c| c.enabled)
            .unwrap_or(false)
    }

    /// Tokenize a prompt
    pub fn tokenize(&self, prompt: &str, add_bos: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, add_bos)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode tokens to text
    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }

    /// Allocate a cache index from the pool
    fn allocate_cache_index(&mut self) -> Option<usize> {
        self.cache_index_pool
            .iter()
            .position(|&used| !used)
            .map(|idx| {
                debug_assert!(
                    idx < self.cache_index_pool.len(),
                    "Allocated cache index {} >= pool size {}",
                    idx,
                    self.cache_index_pool.len()
                );
                debug_assert!(
                    !self.cache_index_pool[idx],
                    "Attempting to allocate already-occupied slot {}",
                    idx
                );
                self.cache_index_pool[idx] = true;
                idx
            })
    }

    /// Release a cache index back to the pool
    fn release_cache_index(&mut self, index: usize) {
        debug_assert!(
            index < self.cache_index_pool.len(),
            "Attempted to release cache index {} >= pool size {}",
            index,
            self.cache_index_pool.len()
        );
        debug_assert!(
            self.cache_index_pool[index],
            "Attempting to release already-free slot {}",
            index
        );
        if index < self.cache_index_pool.len() {
            self.cache_index_pool[index] = false;
        }
    }

    /// Restore KV cache to a specific slot (for prefix caching)
    fn restore_kv_to_slot(
        &mut self,
        slot_idx: usize,
        kv_by_layer: &[(Tensor, Tensor)],
    ) -> Result<usize> {
        if kv_by_layer.is_empty() {
            return Ok(0);
        }

        // Get prefix length from first layer
        let (k_first, _v_first) = &kv_by_layer[0];
        let prefix_len = k_first.dim(2)?;

        // Restore each layer
        for (layer_idx, (k, v)) in kv_by_layer.iter().enumerate() {
            if layer_idx >= self.caches.len() {
                break;
            }

            self.caches[layer_idx].set_slot_kv(slot_idx, k, v)?;
        }

        Ok(prefix_len)
    }

    /// Extract KV cache for a specific slot (for prefix caching)
    fn extract_kv_for_slot(
        &self,
        slot_idx: usize,
        prefix_len: usize,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let mut kv_by_layer = Vec::with_capacity(self.caches.len());

        for cache in &self.caches {
            // Extract K/V for this slot
            let k_full = cache.k().i(slot_idx)?; // [num_heads, context, head_dim]
            let v_full = cache.v().i(slot_idx)?;

            // Narrow to prefix length
            let k_prefix = k_full.narrow(1, 0, prefix_len)?; // [num_heads, prefix_len, head_dim]
            let v_prefix = v_full.narrow(1, 0, prefix_len)?;

            // Expand batch dimension for set_slot_kv compatibility
            let k_with_batch = k_prefix.unsqueeze(0)?; // [1, num_heads, prefix_len, head_dim]
            let v_with_batch = v_prefix.unsqueeze(0)?;

            kv_by_layer.push((k_with_batch, v_with_batch));
        }

        Ok(kv_by_layer)
    }

    /// Override runtime batch size monitoring configuration
    ///
    /// Runtime monitoring is enabled by default with reasonable settings.
    /// Use this method to customize the adjustment thresholds or initial batch size.
    ///
    /// # Arguments
    ///
    /// * `initial_batch_size` - Starting batch size
    /// * `config` - Configuration for adjustment thresholds
    pub fn configure_runtime_monitoring(
        &mut self,
        initial_batch_size: usize,
        config: BatchSizeConfig,
    ) {
        self.runtime_adjuster = Some(RuntimeBatchAdjuster::new(initial_batch_size, config));
    }

    /// Disable runtime batch size monitoring
    ///
    /// Use this if you want fixed batch sizes without dynamic adjustment.
    pub fn disable_runtime_monitoring(&mut self) {
        self.runtime_adjuster = None;
    }

    /// Get current adaptive batch size
    ///
    /// Returns the current batch size recommendation from the runtime adjuster,
    /// or None if runtime monitoring is not enabled.
    pub fn get_adaptive_batch_size(&self) -> Option<usize> {
        self.runtime_adjuster
            .as_ref()
            .map(|adjuster| adjuster.current_batch_size())
    }

    /// Check and apply batch size adjustment
    ///
    /// Call this periodically (e.g., once per batch) to allow the runtime
    /// adjuster to monitor system state and adjust batch size if needed.
    ///
    /// # Arguments
    ///
    /// * `queue_length` - Number of pending requests waiting to be processed
    ///
    /// # Returns
    ///
    /// New batch size if adjustment was made, None otherwise
    pub fn check_batch_adjustment(&mut self, queue_length: usize) -> Option<usize> {
        if self.runtime_adjuster.is_some() {
            // Get current memory usage (simplified - in production would query actual memory)
            let current_memory = self.estimate_current_memory_usage();
            let available_memory = self.estimate_available_memory();

            self.runtime_adjuster.as_mut().unwrap().check_adjustment(
                current_memory,
                available_memory,
                queue_length,
            )
        } else {
            None
        }
    }

    /// Estimate current memory usage (simplified)
    ///
    /// In production, this would query actual memory usage from the system
    fn estimate_current_memory_usage(&self) -> u64 {
        // Rough estimate: model weights + active cache entries
        let active_entries = self.cache_index_pool.iter().filter(|&&used| used).count();
        let per_entry_memory = 1024 * 1024 * 10; // Assume 10MB per active entry
        (active_entries * per_entry_memory) as u64
    }

    /// Estimate available memory (simplified)
    ///
    /// In production, this would query actual available memory from the system
    fn estimate_available_memory(&self) -> u64 {
        // Placeholder: return a reasonable default
        16 * 1024 * 1024 * 1024 // 16GB
    }

    /// Process a batch of requests with parallel batched inference
    ///
    /// This method implements:
    /// 1. Chunked prefill with padding for optimal batching
    /// 2. True parallel forward passes (all requests in single batch)
    /// 3. Efficient scattered KV cache management
    ///
    /// Returns generated tokens for each request (None if request not ready or completed)
    pub fn forward_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>> {
        debug_assert!(!batch.is_empty(), "forward_batch called with empty batch");
        debug_assert!(
            batch.len() <= self.cache_builder.batch_size(),
            "Batch size {} exceeds slot pool size {}",
            batch.len(),
            self.cache_builder.batch_size()
        );

        let _batch_start = Instant::now();

        // Check for runtime batch size adjustment (if enabled)
        if let Some(new_batch_size) = self.check_batch_adjustment(batch.len()) {
            println!(
                "⚡ Runtime batch size adjusted: {} -> {} (queue: {})",
                self.runtime_adjuster.as_ref().unwrap().current_batch_size(),
                new_batch_size,
                batch.len()
            );
            // Note: The new batch size will be used for future batches
            // Current batch continues with its original size
        }

        // Separate requests by state
        let mut prefill_requests = Vec::new();
        let mut decode_requests = Vec::new();

        for (idx, ctx) in batch.iter_mut().enumerate() {
            match ctx.state {
                RequestState::Pending => {
                    // Tokenize the prompt. `ctx.add_special_tokens`, never a
                    // literal `true`: a prompt rendered through a chat template
                    // already carries the checkpoint's BOS and this tokenizer's
                    // `TemplateProcessing` post_processor would prepend a
                    // second one. See `RequestContext::add_special_tokens`.
                    let tokens = self.tokenize(&ctx.request.prompt, ctx.add_special_tokens)?;

                    // Record the real count before any prefix-cache shortcut:
                    // `usage.prompt_tokens` must describe the whole prompt, not
                    // the uncached remainder.
                    ctx.prompt_tokens = tokens.len();

                    // Try to get cached prefix KV (find best matching prefix)
                    if let Some(cached_entry) = self.prefix_cache.get_best_prefix(&tokens) {
                        // Cache hit! Restore KV and treat as if we just finished prefill

                        // Allocate cache slot for this request
                        let cache_idx = self
                            .allocate_cache_index()
                            .expect("No available cache indices in pool");
                        ctx.assign_cache_index(cache_idx);

                        // Restore cached KV into this slot
                        let prefix_len =
                            self.restore_kv_to_slot(cache_idx, &cached_entry.kv_by_layer)?;

                        // Set cache position to prefix length
                        self.cache_builder.set_position(cache_idx, prefix_len);

                        // Update context position to reflect cached tokens
                        ctx.position = prefix_len;

                        // Get the last token from the cached sequence to use as "first generated token"
                        // This allows decode logic to work correctly
                        let last_cached_token = *tokens.last().expect("tokens should not be empty");
                        ctx.generated_tokens.push(last_cached_token);

                        // Transition directly to decoding state
                        ctx.start_decoding();

                        // Add to decode queue (will generate next token beyond cached prefix)
                        decode_requests.push(idx);

                        crate::debug_prefill!(
                            "Cache HIT for req_id={}, restored {} tokens, last_token={}, starting decode",
                            ctx.request.id,
                            prefix_len,
                            last_cached_token
                        );
                    } else {
                        // Cache miss - do normal prefill
                        prefill_requests.push((
                            idx,
                            PrefillRequest::new(ctx.request.id.clone(), tokens.clone()),
                        ));

                        crate::debug_prefill!(
                            "Cache MISS for req_id={}, {} tokens, doing prefill",
                            ctx.request.id,
                            tokens.len()
                        );
                    }
                }
                RequestState::Decoding => {
                    if !ctx.generated_tokens.is_empty() {
                        decode_requests.push(idx);
                    }
                }
                RequestState::Completed => {
                    // In continuous batching, we need to include Completed slots too
                    // They'll be masked out, but we need the full batch size
                    decode_requests.push(idx);
                }
                RequestState::AwaitingToolResult { .. } => {
                    // CR.1: Request is paused waiting for tool result.
                    // KV cache is preserved. Include in decode batch as inactive
                    // (will be masked out, but slot must be present for batch sizing).
                    decode_requests.push(idx);
                }
            }
        }

        let mut results = vec![None; batch.len()];

        // === PREFILL PHASE: Chunked with padding ===
        if !prefill_requests.is_empty() {
            let (indices, mut prefill_reqs): (Vec<_>, Vec<_>) =
                prefill_requests.into_iter().unzip();

            // Use chunked prefill scheduler
            let scheduler = ChunkedPrefillScheduler::new(self.chunked_prefill_config.clone());
            let chunked_batches = scheduler.schedule_all(&mut prefill_reqs);

            self.stats.chunked_prefill_batches += chunked_batches.len();

            for chunked_batch in chunked_batches {
                // Track padding
                let padding: usize = chunked_batch.padding.iter().sum();
                self.stats.total_padding_tokens += padding;

                // Create input tensor (flattened 1D for BatchedTransformer)
                let input_ids = chunked_batch.to_flat_tensor(DType::U32, &self.device)?;
                let batch_size = chunked_batch.batch_size();

                // Build request_ids array using assigned cache indices
                let mut request_ids = Vec::with_capacity(batch_size);
                for (_i, req_id) in chunked_batch.request_ids.iter().enumerate() {
                    // Find the request context
                    let idx = indices
                        .iter()
                        .position(|&batch_idx| batch[batch_idx].request.id == *req_id)
                        .unwrap();
                    let ctx = &mut batch[indices[idx]];

                    // Assign stable cache index from pool if not already assigned
                    let is_first_chunk = ctx.cache_index.is_none();
                    if is_first_chunk {
                        // Allocate a stable cache index from the pool
                        let cache_idx = self
                            .allocate_cache_index()
                            .expect("No available cache indices in pool");
                        debug_assert!(
                            cache_idx < self.cache_builder.batch_size(),
                            "Allocated cache index {} >= batch size {}",
                            cache_idx,
                            self.cache_builder.batch_size()
                        );
                        ctx.assign_cache_index(cache_idx);
                        // ONLY reset cache builder position for the FIRST chunk of a new request
                        // This ensures RoPE uses correct positions (0, 1, 2, ...) from the start
                        self.cache_builder.reset_batch_index(cache_idx);
                    }

                    // Use the assigned cache index (stable across chunks and into decode)
                    let cache_idx = ctx.cache_index.unwrap();
                    debug_assert!(
                        cache_idx < self.cache_builder.batch_size(),
                        "Cache index {} >= batch size {}",
                        cache_idx,
                        self.cache_builder.batch_size()
                    );
                    debug_assert!(
                        self.cache_index_pool[cache_idx],
                        "Using cache index {} that's marked as free!",
                        cache_idx
                    );
                    request_ids.push(cache_idx);
                }

                // Create metadata for prefill
                let actual_lengths = chunked_batch.actual_lengths.clone();
                // All sequences are padded to the same length
                let padded_length = chunked_batch.token_sequences[0].len();
                let padded_lengths = vec![padded_length; batch_size];
                let metadata = BatchMetadata::from_chunked_prefill_batch(
                    request_ids.clone(),
                    actual_lengths,
                    padded_lengths,
                );

                // Segmented cache: begin UserInput spans before prefill forward pass
                let mut prefill_span_ids: Vec<Option<crate::cache::SpanId>> =
                    vec![None; request_ids.len()];
                if self.is_segmented_cache_enabled() {
                    for (i, &cache_idx) in request_ids.iter().enumerate() {
                        match self.cache_builder.begin_span(
                            cache_idx,
                            crate::cache::CacheTag::UserInput,
                            None,
                        ) {
                            Ok(span_id) => prefill_span_ids[i] = Some(span_id),
                            Err(e) => {
                                eprintln!(
                                    "Warning: Failed to begin UserInput span for slot {}: {}",
                                    cache_idx, e
                                );
                            }
                        }
                    }
                }

                // Forward pass
                let forward_start = Instant::now();
                crate::debug_prefill!(
                    "input_ids shape={:?}, batch_size={}, metadata={:?}",
                    input_ids.dims(),
                    batch_size,
                    metadata
                );
                let logits = self.model.forward(
                    &input_ids,
                    &mut self.cache_builder,
                    &mut self.caches,
                    &metadata,
                )?;
                crate::debug_prefill!("logits shape={:?}", logits.dims());
                self.stats.total_forward_time_ms += forward_start.elapsed().as_secs_f64() * 1000.0;

                // Advance positions by actual lengths after forward pass
                // indices_and_mask is now pure and doesn't mutate state, so we must
                // explicitly advance positions here by the actual token count (not padded)
                for (i, actual_len) in chunked_batch.actual_lengths.iter().enumerate() {
                    let cache_idx = request_ids[i];
                    let old_pos = self.cache_builder.get_position(cache_idx);
                    let new_pos = old_pos + actual_len;
                    self.cache_builder.set_position(cache_idx, new_pos);
                    crate::debug_prefill!(
                        "Slot {} position: {} -> {} (advanced by {} actual tokens)",
                        cache_idx,
                        old_pos,
                        new_pos,
                        actual_len
                    );
                }

                // Segmented cache: end UserInput spans after position advance
                if self.is_segmented_cache_enabled() {
                    for span_id in prefill_span_ids.into_iter().flatten() {
                        if let Err(e) = self.cache_builder.end_span(span_id) {
                            eprintln!("Warning: Failed to end UserInput span {}: {}", span_id, e);
                        }
                    }
                }

                self.stats.prefill_batches += 1;
                self.stats.total_batches += 1;
                self.stats.max_batch_size = self.stats.max_batch_size.max(batch_size);

                // Process results
                // BatchedTransformer already extracts last token, so logits shape: [batch_size, vocab_size]
                for (i, req_id) in chunked_batch.request_ids.iter().enumerate() {
                    crate::debug_prefill!("Processing result for req_id={}", req_id);
                    let idx = indices
                        .iter()
                        .position(|&batch_idx| batch[batch_idx].request.id == *req_id)
                        .unwrap();
                    let ctx = &mut batch[indices[idx]];

                    crate::debug_prefill!("req_id={}, ctx={:?}", req_id, ctx);

                    // Get logits for this sequence (already last token)
                    let seq_len = chunked_batch.actual_lengths[i];
                    crate::debug_prefill!("req_id={}, seq_len={}", req_id, seq_len);
                    let logits_slice = logits.i(i)?;
                    crate::debug_prefill!(
                        "req_id={}, seq_len={}, logits_slice shape={:?}",
                        req_id,
                        seq_len,
                        logits_slice.dims()
                    );
                    let next_token = logits_slice.argmax(0)?.to_scalar::<u32>()?;

                    crate::debug_prefill!(
                        "req_id={}, next_token={}, seq_len={} before updating context",
                        req_id,
                        next_token,
                        seq_len
                    );

                    ctx.generated_tokens.push(next_token);
                    ctx.position += seq_len;
                    ctx.tokens_generated += 1; // Count the generated token

                    crate::debug_prefill!(
                        "req_id={}, next_token={}, position={}",
                        req_id,
                        next_token,
                        ctx.position
                    );

                    let hit_eos = self.is_eos_token(next_token);
                    if hit_eos || ctx.tokens_generated >= ctx.request.max_new_tokens {
                        ctx.stopped_on_eos = hit_eos;
                        ctx.complete();
                        // Automatically release cache index when request completes
                        if let Some(cache_idx) = ctx.cache_index {
                            self.release_cache_index(cache_idx);
                        }
                        results[indices[idx]] = None;
                    } else {
                        ctx.start_decoding();

                        // Cache the prefix KV after successful prefill
                        // Only cache if this is the first time we've seen this prompt
                        if let Some(cache_idx) = ctx.cache_index {
                            // Extract KV for the full prompt length
                            let prompt_len = ctx.position;

                            // Get the original prompt tokens for cache key
                            // We need to re-tokenize to get the exact tokens
                            // (alternative: store tokens in ctx during prefill)
                            //
                            // The flag MUST match the prefill call above. This
                            // re-tokenization produces a prefix-cache KEY for KV
                            // that was computed from the prefill tokens; a
                            // different `add_special_tokens` here would key the
                            // entry under a token sequence that was never the
                            // one prefilled, so a later request with that
                            // sequence would be handed another prompt's KV.
                            let tokens =
                                self.tokenize(&ctx.request.prompt, ctx.add_special_tokens)?;

                            // IMPORTANT: Use only the tokens we actually processed (prefix_len)
                            // In case of chunked prefill, we may have only processed part of the prompt
                            let tokens_to_cache = &tokens[..prompt_len.min(tokens.len())];

                            // Only insert if not already cached (avoid redundant work)
                            if !self.prefix_cache.check_would_hit(tokens_to_cache) {
                                match self.extract_kv_for_slot(cache_idx, prompt_len) {
                                    Ok(kv_by_layer) => {
                                        // Insert into cache
                                        if let Err(e) = self.prefix_cache.insert(
                                            tokens_to_cache,
                                            kv_by_layer,
                                            &self.device,
                                        ) {
                                            // Log error but don't fail the request
                                            eprintln!(
                                                "Warning: Failed to cache prefix for req_id={}: {}",
                                                ctx.request.id, e
                                            );
                                        } else {
                                            crate::debug_prefill!(
                                                "Cached prefix for req_id={}, {} tokens",
                                                ctx.request.id,
                                                prompt_len
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "Warning: Failed to extract KV for caching req_id={}: {}",
                                            ctx.request.id, e
                                        );
                                    }
                                }
                            }
                        }

                        // Don't call record_token() here - we already incremented position and tokens_generated
                        results[indices[idx]] = Some(next_token);
                        self.stats.total_tokens_generated += 1;
                    }
                }
            }
        }

        // === DECODE PHASE: True parallel batching ===
        if !decode_requests.is_empty() {
            // For continuous batching, batch_size MUST equal slot pool size
            let batch_size = self.cache_builder.batch_size();

            debug_assert!(
                !decode_requests.is_empty(),
                "Decode phase entered with empty decode_requests"
            );
            debug_assert!(batch_size > 0, "Slot pool size is 0");

            crate::debug_decode!(
                "slot_pool_size={}, decode_requests={:?}",
                batch_size,
                decode_requests
            );

            // Collect tokens and positions for ALL slots in the pool
            // Active slots get real values, inactive slots get dummy values + will be masked
            let mut token_ids = Vec::with_capacity(batch_size);
            let mut positions = Vec::with_capacity(batch_size);
            let mut request_ids = Vec::with_capacity(batch_size);

            // Process ALL slots in the pool (batch.len() should equal batch_size for continuous batching)
            for idx in 0..batch_size {
                if idx < batch.len() {
                    let ctx = &batch[idx];

                    // Check if this slot is actually active (Decoding with tokens)
                    let is_active =
                        ctx.state == RequestState::Decoding && !ctx.generated_tokens.is_empty();

                    if is_active {
                        let last_token = *ctx.generated_tokens.last().unwrap();
                        token_ids.push(last_token);
                        positions.push(ctx.position);
                        // Use the cache index assigned during prefill
                        let cache_idx = ctx
                            .cache_index
                            .expect("cache_index should be assigned during prefill");
                        debug_assert!(
                            cache_idx < batch_size,
                            "Cache index {} >= batch size {}",
                            cache_idx,
                            batch_size
                        );
                        debug_assert!(
                            self.cache_index_pool[cache_idx],
                            "Using cache index {} that's marked as free in decode phase!",
                            cache_idx
                        );
                        request_ids.push(cache_idx);

                        crate::debug_decode!(
                            "PREP: idx={}, last_token={}, position={}, cache_idx={}, generated_tokens={:?}",
                            idx,
                            last_token,
                            ctx.position,
                            cache_idx,
                            ctx.generated_tokens
                        );
                    } else {
                        // Inactive slot (Completed or Pending) - use dummy values
                        token_ids.push(0);
                        positions.push(0);
                        // For inactive slots, use the slot index as a placeholder cache index
                        request_ids.push(idx);

                        crate::debug_decode!(
                            "PREP: idx={} INACTIVE (state={:?}, masked out)",
                            idx,
                            ctx.state
                        );
                    }
                } else {
                    // Slot beyond batch array - use dummy values
                    token_ids.push(0);
                    positions.push(0);
                    request_ids.push(idx);
                    crate::debug_decode!("PREP: idx={} EMPTY (beyond batch array)", idx);
                }
            }

            // Create batched tensor [batch_size] (1D for BatchedTransformer)
            let input_ids = Tensor::new(&token_ids[..], &self.device)?;

            // Create metadata (positions are used for both cache lookup and context tracking)
            let metadata = BatchMetadata::from_decode_batch(
                request_ids.clone(),
                positions.clone(),
                positions.clone(), // context_lens = positions for decode
            );

            crate::debug_decode!(
                "batch_size={}, request_ids={:?}, positions={:?}",
                batch_size,
                request_ids,
                positions
            );

            // CRITICAL: Advance cache positions BEFORE forward pass (only for active slots with valid cache_index)
            // The pure indices_and_mask uses current position to determine WHERE to write KV
            // So we must set position to where we WANT to write, not where we just read from
            for (idx, &cache_idx) in request_ids.iter().enumerate() {
                // Only advance if this is an active request with a valid cache index
                // (inactive slots use idx as placeholder, which doesn't match their actual cache_index)
                if idx < batch.len() {
                    let ctx = &batch[idx];
                    if ctx.state == RequestState::Decoding
                        && !ctx.generated_tokens.is_empty()
                        && ctx.cache_index == Some(cache_idx)
                    {
                        let new_position = positions[idx] + 1; // Advance to next position for KV write
                        self.cache_builder.set_position(cache_idx, new_position);
                    }
                }
            }

            // Single parallel forward pass for all decode requests!
            let forward_start = Instant::now();
            let logits = self.model.forward(
                &input_ids,
                &mut self.cache_builder,
                &mut self.caches,
                &metadata,
            )?;
            self.stats.total_forward_time_ms += forward_start.elapsed().as_secs_f64() * 1000.0;

            crate::debug_decode!("logits shape={:?}", logits.dims());

            self.stats.decode_batches += 1;
            self.stats.total_batches += 1;
            self.stats.max_batch_size = self.stats.max_batch_size.max(batch_size);

            // Segmented cache: periodic logging and eviction after decode forward pass
            if let Some(ref config) = self.segmented_config {
                if config.enabled {
                    // Throttle to H2O update interval
                    if self.stats.decode_batches % config.h2o_update_interval == 0 {
                        if config.log_span_attention {
                            self.cache_builder.log_span_attention_summary();
                        }

                        let max_util = self.cache_builder.get_max_utilization();

                        if config.shadow_mode {
                            self.cache_builder.log_eviction_candidates(config, 5);
                        } else if max_util > config.eviction_pressure_threshold {
                            let config_clone = config.clone();
                            self.handle_eviction_pressure(&config_clone, max_util);
                        }
                    }
                }
            }

            // Process results (only for decode_requests, which are the requests that went through decode)
            for (_i, &idx) in decode_requests.iter().enumerate() {
                let ctx = &mut batch[idx];

                // Skip if not actually decoding (Completed or Pending)
                if ctx.state != RequestState::Decoding || ctx.generated_tokens.is_empty() {
                    continue;
                }

                let logits_slice = logits.i(idx)?; // Use idx directly since logits shape matches batch_size
                let next_token = logits_slice.argmax(0)?.to_scalar::<u32>()?;

                crate::debug_decode!("RESULT: idx={}, i={}, next_token={}", idx, i, next_token);

                ctx.generated_tokens.push(next_token);
                ctx.record_token();
                self.stats.total_tokens_generated += 1;

                // CR.1: Tool call detection — check for tool call patterns in output
                if let Some(ref mut detector) = self.tool_call_detector {
                    if let Some(cache_idx) = ctx.cache_index {
                        detector.push_token(cache_idx, next_token);

                        if let Some(detected) = detector.check_for_tool_call(
                            cache_idx,
                            &self.tokenizer,
                            ctx.generated_tokens.len(),
                        ) {
                            println!(
                                "Tool call detected: {}({}) at tokens {}..{}",
                                detected.tool_name,
                                detected.tool_args,
                                detected.token_start,
                                detected.token_end
                            );

                            // Transition to AwaitingToolResult — KV cache preserved
                            ctx.await_tool_result(
                                detected.tool_name,
                                detected.tool_args,
                                None, // TODO: capture attention snapshot
                            );

                            // Clear the token buffer for this slot
                            detector.clear_slot(cache_idx);

                            // Signal caller (skip EOS check, don't generate more)
                            results[idx] = Some(next_token);
                            continue;
                        }
                    }
                }

                // Segmented cache: check for <RETRIEVE:key> token pattern
                if self.is_segmented_cache_enabled() {
                    if let Some(ref config) = self.segmented_config {
                        if config.tiered_demotion && !config.shadow_mode {
                            // Decode last ~15 tokens to check for RETRIEVE pattern
                            let check_len = ctx.generated_tokens.len().min(15);
                            let tail =
                                &ctx.generated_tokens[ctx.generated_tokens.len() - check_len..];
                            if let Ok(decoded) = self.tokenizer.decode(tail, false) {
                                if let Some(start) = decoded.find("<RETRIEVE:") {
                                    if let Some(end) = decoded[start..].find('>') {
                                        let key = &decoded[start + 10..start + end];
                                        // Look up the demoted segment
                                        if let Some(ref mut storage) = self.tiered_storage {
                                            if let Some(span_id) = storage.find_by_fact_key(key) {
                                                // Promote back to GPU
                                                match storage.promote_with_disk(span_id) {
                                                    Ok((slot, range, kv_layers)) => {
                                                        // Re-inject KV tensors into cache
                                                        let mut inject_ok = true;
                                                        for (layer_idx, (k, v)) in
                                                            kv_layers.iter().enumerate()
                                                        {
                                                            if layer_idx < self.caches.len() {
                                                                if let Err(e) = self.caches
                                                                    [layer_idx]
                                                                    .restore_kv_range(
                                                                        slot,
                                                                        range.start,
                                                                        k,
                                                                        v,
                                                                    )
                                                                {
                                                                    eprintln!(
                                                                        "Warning: KV re-injection failed at layer {}: {}",
                                                                        layer_idx, e
                                                                    );
                                                                    inject_ok = false;
                                                                    break;
                                                                }
                                                            }
                                                        }

                                                        if inject_ok {
                                                            // Clear evicted positions so mask allows attention again
                                                            self.cache_builder.clear_evicted_range(
                                                                slot,
                                                                range.start,
                                                                range.end,
                                                            );
                                                            println!(
                                                                "Promoted span {} back to GPU, re-injected at positions {}..{} (key: {})",
                                                                span_id,
                                                                range.start,
                                                                range.end,
                                                                key
                                                            );
                                                        }
                                                    }
                                                    Err(e) => {
                                                        eprintln!(
                                                            "Warning: Failed to promote span {}: {}",
                                                            span_id, e
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let hit_eos = self.is_eos_token(next_token);
                if hit_eos || ctx.tokens_generated >= ctx.request.max_new_tokens {
                    // Segmented cache: end ModelGeneration span on completion
                    if self.is_segmented_cache_enabled() {
                        if let Some(cache_idx) = ctx.cache_index {
                            if let Some(span_id) = self.active_generation_spans.remove(&cache_idx) {
                                if let Err(e) = self.cache_builder.end_span(span_id) {
                                    eprintln!(
                                        "Warning: Failed to end ModelGeneration span {}: {}",
                                        span_id, e
                                    );
                                }
                            }
                        }
                    }

                    ctx.stopped_on_eos = hit_eos;
                    ctx.complete();
                    // Automatically release cache index when request completes
                    if let Some(cache_idx) = ctx.cache_index {
                        self.release_cache_index(cache_idx);
                    }
                    results[idx] = None;
                } else {
                    // Segmented cache: begin ModelGeneration span on first decode token
                    if self.is_segmented_cache_enabled() {
                        if let Some(cache_idx) = ctx.cache_index {
                            if !self.active_generation_spans.contains_key(&cache_idx) {
                                match self.cache_builder.begin_span(
                                    cache_idx,
                                    crate::cache::CacheTag::ModelGeneration,
                                    None,
                                ) {
                                    Ok(span_id) => {
                                        self.active_generation_spans.insert(cache_idx, span_id);
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "Warning: Failed to begin ModelGeneration span for slot {}: {}",
                                            cache_idx, e
                                        );
                                    }
                                }
                            }
                        }
                    }

                    results[idx] = Some(next_token);
                }
            }
        }

        // Count only completed requests (not every batch processing)
        let completed_count = batch
            .iter()
            .filter(|ctx| ctx.state == RequestState::Completed)
            .count();
        self.stats.total_requests_processed += completed_count;

        // Record memory usage for runtime adjuster (if enabled)
        if self.runtime_adjuster.is_some() {
            let memory_used = self.estimate_current_memory_usage();
            let active_requests = batch
                .iter()
                .filter(|ctx| matches!(ctx.state, RequestState::Decoding))
                .count();
            if active_requests > 0 {
                self.runtime_adjuster
                    .as_mut()
                    .unwrap()
                    .record_batch_memory(memory_used, active_requests);
            }
        }

        Ok(results)
    }

    /// Get performance statistics
    pub fn stats(&self) -> &ParallelBatchStats {
        &self.stats
    }

    /// Get prefix cache statistics
    pub fn prefix_cache_stats(&self) -> crate::cache::PrefixCacheStats {
        self.prefix_cache.stats()
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ParallelBatchStats::new();
    }

    /// Check if token is EOS
    fn is_eos_token(&self, token: u32) -> bool {
        match &self.config.eos_token_id {
            Some(LlamaEosToks::Single(eos)) => token == *eos,
            Some(LlamaEosToks::Multiple(eos_tokens)) => eos_tokens.contains(&token),
            None => false,
        }
    }

    /// Get the model configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get a reference to the cache builder (for inspection/diagnostics)
    pub fn cache_builder(&self) -> &crate::engine::ParallelCacheBuilder {
        &self.cache_builder
    }

    /// Get a mutable reference to the cache builder
    pub fn cache_builder_mut(&mut self) -> &mut crate::engine::ParallelCacheBuilder {
        &mut self.cache_builder
    }
}
