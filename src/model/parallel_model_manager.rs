//! Parallel model manager with true batched inference
//!
//! This is the production-ready implementation using:
//! - BatchedTransformer for parallel forward passes
//! - BatchExecutor with ScatteredKvCache for efficient memory management  
//! - ChunkedPrefillScheduler for optimal prefill batching with padding
//!
//! Expected performance improvements over sequential model_manager:
//! - CPU: 5-10x faster
//! - GPU: 10-50x faster

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::llama::{Config, LlamaEosToks};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

use crate::engine::{BatchExecutor, RequestContext, RequestState};
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

/// Parallel model manager using BatchedTransformer
pub struct ParallelModelManager {
    model: BatchedTransformer,
    batch_executor: BatchExecutor,
    tokenizer: Tokenizer,
    config: Config,
    chunked_prefill_config: ChunkedPrefillConfig,
    device: Device,
    stats: ParallelBatchStats,
    cache_index_pool: Vec<bool>, // true = in use, false = available
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
        let (_llama_model, _cache, config, device) = load_local_llama(
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

        // Create BatchExecutor
        let num_layers = config.num_hidden_layers;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads; // For GQA
        let head_dim = config.hidden_size / config.num_attention_heads;

        // Note: BatchExecutor cache should be created with num_kv_heads, not num_heads
        // because K/V tensors have num_kv_heads dimension in GQA models
        // The head expansion from num_kv_heads -> num_heads happens in attention layer
        let batch_executor = BatchExecutor::new(
            max_batch_size,
            context_length,
            num_layers,
            num_kv_heads, // Use num_kv_heads for cache, not num_heads
            head_dim,
            dtype,
            &device,
        )?;

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
        };

        // Load model weights
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                dtype,
                &device,
            )?
        };

        let model = BatchedTransformer::new(transformer_config, vb)?;

        let chunked_prefill_config = chunked_prefill_config.unwrap_or_default();

        Ok(Self {
            model,
            batch_executor,
            tokenizer,
            config,
            chunked_prefill_config,
            device,
            stats: ParallelBatchStats::new(),
            cache_index_pool: vec![false; max_batch_size],
        })
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
                self.cache_index_pool[idx] = true;
                idx
            })
    }

    /// Release a cache index back to the pool
    fn release_cache_index(&mut self, index: usize) {
        if index < self.cache_index_pool.len() {
            self.cache_index_pool[index] = false;
        }
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
        let batch_start = Instant::now();

        // Separate requests by state
        let mut prefill_requests = Vec::new();
        let mut decode_requests = Vec::new();

        for (idx, ctx) in batch.iter_mut().enumerate() {
            match ctx.state {
                RequestState::Pending => {
                    // Tokenize the prompt
                    let tokens = self.tokenize(&ctx.request.prompt, true)?;
                    prefill_requests
                        .push((idx, PrefillRequest::new(ctx.request.id.clone(), tokens)));
                }
                RequestState::Decoding => {
                    if !ctx.generated_tokens.is_empty() {
                        decode_requests.push(idx);
                    }
                }
                RequestState::Completed => {}
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
                        ctx.assign_cache_index(cache_idx);
                        // ONLY reset cache builder position for the FIRST chunk of a new request
                        // This ensures RoPE uses correct positions (0, 1, 2, ...) from the start
                        self.batch_executor.reset_request_state(cache_idx);
                    }

                    // Use the assigned cache index (stable across chunks and into decode)
                    request_ids.push(ctx.cache_index.unwrap());
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

                // Forward pass
                let forward_start = Instant::now();
                crate::debug_prefill!(
                    "input_ids shape={:?}, batch_size={}, metadata={:?}",
                    input_ids.dims(),
                    batch_size,
                    metadata
                );
                let logits = self
                    .model
                    .forward(&input_ids, &mut self.batch_executor, &metadata)?;
                crate::debug_prefill!("logits shape={:?}", logits.dims());
                self.stats.total_forward_time_ms += forward_start.elapsed().as_secs_f64() * 1000.0;

                // Advance positions by actual lengths after forward pass
                // indices_and_mask is now pure and doesn't mutate state, so we must
                // explicitly advance positions here by the actual token count (not padded)
                for (i, actual_len) in chunked_batch.actual_lengths.iter().enumerate() {
                    let cache_idx = request_ids[i];
                    let old_pos = self.batch_executor.get_cache_position(cache_idx);
                    let new_pos = old_pos + actual_len;
                    self.batch_executor.set_cache_position(cache_idx, new_pos);
                    crate::debug_prefill!(
                        "Slot {} position: {} -> {} (advanced by {} actual tokens)",
                        cache_idx,
                        old_pos,
                        new_pos,
                        actual_len
                    );
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

                    if self.is_eos_token(next_token)
                        || ctx.tokens_generated >= ctx.request.max_new_tokens
                    {
                        ctx.complete();
                        results[indices[idx]] = None;
                    } else {
                        ctx.start_decoding();
                        // Don't call record_token() here - we already incremented position and tokens_generated
                        results[indices[idx]] = Some(next_token);
                        self.stats.total_tokens_generated += 1;
                    }
                }
            }
        }

        // === DECODE PHASE: True parallel batching ===
        if !decode_requests.is_empty() {
            let batch_size = decode_requests.len();

            crate::debug_decode!(
                "batch_size={}, decode_requests={:?}",
                batch_size,
                decode_requests
            );

            // Collect tokens and positions
            let mut token_ids = Vec::with_capacity(batch_size);
            let mut positions = Vec::with_capacity(batch_size);
            let mut request_ids = Vec::with_capacity(batch_size);

            for &idx in &decode_requests {
                let ctx = &batch[idx];
                let last_token = *ctx.generated_tokens.last().unwrap();
                token_ids.push(last_token);
                positions.push(ctx.position);
                // Use the stable cache index assigned during prefill
                request_ids.push(
                    ctx.cache_index
                        .expect("cache_index should be assigned during prefill"),
                );

                crate::debug_decode!(
                    "PREP: idx={}, last_token={}, position={}, generated_tokens={:?}",
                    idx,
                    last_token,
                    ctx.position,
                    ctx.generated_tokens
                );
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

            // CRITICAL: Advance cache positions BEFORE forward pass
            // The pure indices_and_mask uses current position to determine WHERE to write KV
            // So we must set position to where we WANT to write, not where we just read from
            for (i, &cache_idx) in request_ids.iter().enumerate() {
                let new_position = positions[i] + 1; // Advance to next position for KV write
                self.batch_executor
                    .set_cache_position(cache_idx, new_position);
            }

            // Single parallel forward pass for all decode requests!
            let forward_start = Instant::now();
            let logits = self
                .model
                .forward(&input_ids, &mut self.batch_executor, &metadata)?;
            self.stats.total_forward_time_ms += forward_start.elapsed().as_secs_f64() * 1000.0;

            crate::debug_decode!("logits shape={:?}", logits.dims());

            self.stats.decode_batches += 1;
            self.stats.total_batches += 1;
            self.stats.max_batch_size = self.stats.max_batch_size.max(batch_size);

            // Process results
            for (i, &idx) in decode_requests.iter().enumerate() {
                let ctx = &mut batch[idx];

                let logits_slice = logits.i(i)?;
                let next_token = logits_slice.argmax(0)?.to_scalar::<u32>()?;

                crate::debug_decode!("RESULT: idx={}, i={}, next_token={}", idx, i, next_token);

                ctx.generated_tokens.push(next_token);
                ctx.record_token();
                self.stats.total_tokens_generated += 1;

                if self.is_eos_token(next_token)
                    || ctx.tokens_generated >= ctx.request.max_new_tokens
                {
                    ctx.complete();
                    results[idx] = None;
                } else {
                    results[idx] = Some(next_token);
                }
            }
        }

        self.stats.total_requests_processed += batch.len();

        Ok(results)
    }

    /// Get performance statistics
    pub fn stats(&self) -> &ParallelBatchStats {
        &self.stats
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
}
