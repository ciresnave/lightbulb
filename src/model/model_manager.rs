//! Model integration for batched inference
//!
//! This module bridges our BatchExecutor with actual transformer models,
//! handling model loading, tokenization, and batched forward passes.
//!
//! # Batching Architecture (Phase 2D Status)
//!
//! ## Current Implementation
//!
//! **Prefill Phase** (Processing prompts):
//! - Sequential processing due to variable prompt lengths
//! - Each request processed independently with its own cache
//! - ~10% of total operations (1 per request)
//!
//! **Decode Phase** (Generating tokens):
//! - Batch-structured but still sequential processing
//! - Grouped together for future batching optimization
//! - ~90% of total operations (max_tokens per request)
//!
//! ## Batching Constraints
//!
//! The Candle Llama model API presents fundamental batching challenges:
//!
//! ```rust,ignore
//! fn forward(&mut self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor>
//! ```
//!
//! **Limitations**:
//! - `index_pos: usize` - Single position, not `Vec<usize>` for batched positions
//! - `cache: &mut Cache` - Single cache, not batched/scattered cache
//! - Internal KV cache management not exposed for external batching
//!
//! ## Path to True Batching
//!
//! ### Option A: Custom Layer Wrapper (Recommended)
//! 1. Bypass `model.forward()`, call transformer layers directly
//! 2. Extract K/V tensors after each attention layer
//! 3. Update `BatchExecutor::ScatteredKvCache` with K/V values
//! 4. Retrieve scattered K/V for next layer
//! 5. Single batched forward pass for all decode requests
//!
//! **Pros**: Works with existing Candle, full control
//! **Cons**: Complex, requires understanding Llama internals
//! **Expected speedup**: 5-10x CPU, 10-50x GPU
//!
//! ### Option B: Modified Llama Model
//! Fork or PR to Candle to add:
//! ```rust,ignore
//! fn forward_batched(
//!     &mut self,
//!     x: &Tensor,  // Shape: (batch_size, seq_len)
//!     positions: &[usize],  // Per-request positions
//!     scattered_cache: &mut ScatteredKvCache
//! ) -> Result<Tensor>  // Shape: (batch_size, vocab_size)
//! ```
//!
//! **Pros**: Clean API, upstream contribution
//! **Cons**: Requires Candle changes, longer timeline
//!
//! ### Option C: Wait for Candle Native Support
//! Candle may add batched KV cache support in future
//!
//! ## Current Performance
//!
//! **Baseline** (CPU, f32, llama-3b):
//! - Throughput: ~0.4-0.6 tokens/sec
//! - 90% time in decode operations
//! - Sequential processing limits GPU utilization
//!
//! **After True Batching** (estimated):
//! - CPU: 2-5 tokens/sec (5-10x improvement)
//! - GPU: 20-100 tokens/sec (50-200x improvement)
//! - Enables continuous batching at scale

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::llama::{Cache, Config, LlamaEosToks};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

use crate::engine::{BatchExecutor, RequestContext, RequestState};
use crate::loaders::load_local_llama;
use crate::model::{BatchManager, BatchMetadata};

/// Performance statistics for batched inference
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    pub total_batches: usize,
    pub total_requests_processed: usize,
    pub total_tokens_generated: usize,
    pub total_forward_time_ms: f64,
    pub prefill_requests: usize,
    pub decode_requests: usize,
    // Phase 2D: Batching opportunity tracking
    pub decode_batch_opportunities: usize, // Times we had multiple decode requests
    pub max_concurrent_decodes: usize,     // Largest decode batch seen
}

impl BatchStats {
    pub fn new() -> Self {
        Self::default()
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
            (self.total_tokens_generated as f64 / self.total_forward_time_ms) * 1000.0
        }
    }

    pub fn average_forward_time_ms(&self) -> f64 {
        if self.total_batches == 0 {
            0.0
        } else {
            self.total_forward_time_ms / self.total_batches as f64
        }
    }

    /// Calculate potential speedup from true batched decoding
    pub fn potential_batching_speedup(&self) -> f64 {
        if self.decode_batch_opportunities == 0 {
            1.0 // No batching opportunity
        } else {
            // If we could batch all concurrent decodes into single forward calls
            // Speedup = total decode ops / batching opportunities
            self.decode_requests as f64 / self.decode_batch_opportunities as f64
        }
    }

    /// Batching efficiency: how often we have batching opportunities
    pub fn batching_efficiency(&self) -> f64 {
        if self.total_batches == 0 {
            0.0
        } else {
            self.decode_batch_opportunities as f64 / self.total_batches as f64
        }
    }
}

/// Wrapper for a loaded model with tokenizer and batch manager
pub struct ModelManager {
    batch_manager: BatchManager<candle_transformers::models::llama::Llama>, // Model-agnostic batch manager
    caches: HashMap<String, Cache>, // Per-request caches (keyed by request ID)
    config: Config,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,      // Store dtype for creating new caches
    stats: BatchStats, // Performance statistics
}

impl ModelManager {
    /// Load a model from a local directory
    ///
    /// # Arguments
    /// * `model_dir` - Path to model directory (contains config.json, model.safetensors, tokenizer.json)
    /// * `max_batch_size` - Maximum number of concurrent requests
    /// * `context_length` - Context window size
    /// * `dtype` - Data type for model ("f32", "f16", "bf16")
    pub fn load(
        model_dir: impl AsRef<Path>,
        max_batch_size: usize,
        context_length: usize,
        dtype: Option<&str>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        // Load model and cache
        let (model, cache, config, device) = load_local_llama(
            model_dir.to_str().context("Invalid model path")?,
            dtype,
            true,  // use_kv_cache
            false, // use_flash_attn (can enable later)
        )?;

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Create batch executor
        let num_layers = config.num_hidden_layers;
        let num_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / config.num_attention_heads;

        let dtype_enum = match dtype {
            Some("f32") | None => DType::F32,
            Some("f16") => DType::F16,
            Some("bf16") => DType::BF16,
            _ => DType::F32,
        };

        let batch_executor = BatchExecutor::new(
            max_batch_size,
            context_length,
            num_layers,
            num_heads,
            head_dim,
            dtype_enum,
            &device,
        )?;

        // Create model-agnostic batch manager
        let batch_manager = BatchManager::new(model, batch_executor, device.clone());

        // We'll create caches on-demand per request instead of pre-allocating
        let caches = HashMap::new();

        Ok(Self {
            batch_manager,
            caches,
            config,
            tokenizer,
            device,
            dtype: dtype_enum,
            stats: BatchStats::new(),
        })
    }

    /// Tokenize a prompt string to token IDs
    pub fn tokenize(&self, prompt: &str, add_bos: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, add_bos)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))
    }

    /// Process a batch of requests through the model (single forward pass)
    ///
    /// This is the core batched inference method that:
    /// 1. Prepares the batch (assigns cache indices)
    /// 2. Tokenizes prompts (for Pending requests) or uses last token (for Decoding)
    /// 3. Runs forward pass through model
    /// 4. Updates KV caches via BatchExecutor
    /// 5. Returns generated tokens per request
    ///
    /// Note: This implementation processes requests sequentially with per-request caches.
    /// Full integration with BatchExecutor's ScatteredKvCache for true batched inference is TODO.
    pub fn forward_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>> {
        let batch_start = Instant::now();

        let mut results = Vec::with_capacity(batch.len());

        // Track batch statistics
        self.stats.total_batches += 1;
        self.stats.total_requests_processed += batch.len();

        // Ensure all requests have caches created
        for ctx in batch.iter() {
            if !self.caches.contains_key(&ctx.request.id) {
                let new_cache = Cache::new(true, self.dtype, &self.config, &self.device)?;
                self.caches.insert(ctx.request.id.clone(), new_cache);
            }
        }

        // Pre-tokenize prompts for Pending requests to avoid borrow conflicts
        let mut tokenized_prompts: HashMap<String, Vec<u32>> = HashMap::new();
        for ctx in batch.iter() {
            if ctx.state == RequestState::Pending {
                let tokens = self.tokenize(&ctx.request.prompt, true)?;
                tokenized_prompts.insert(ctx.request.id.clone(), tokens);
            }
        }

        // **PHASE 2D: Group requests by state for batched processing**
        let mut prefill_indices = Vec::new();
        let mut decode_indices = Vec::new();

        for (i, ctx) in batch.iter().enumerate() {
            match ctx.state {
                RequestState::Pending => prefill_indices.push(i),
                RequestState::Decoding => decode_indices.push(i),
                RequestState::Completed => {}
            }
        }

        // Track statistics
        self.stats.prefill_requests += prefill_indices.len();
        self.stats.decode_requests += decode_indices.len();

        let mut tokens_generated_this_batch = 0;

        // === PREFILL Phase: Process sequentially (variable prompt lengths) ===
        for &idx in &prefill_indices {
            let ctx = &mut batch[idx];

            let tokens = tokenized_prompts
                .get(&ctx.request.id)
                .expect("Tokenized prompt should exist");

            if tokens.is_empty() {
                continue;
            }

            // Process prompt tokens through model (using BatchManager with batch_size=1)
            let input_ids = Tensor::new(tokens.as_slice(), &self.device)?;

            let forward_start = Instant::now();
            // Create metadata for single prefill batch
            let metadata = BatchMetadata::from_prefill_batch(vec![0], vec![tokens.len()]);

            // Temporarily take ownership of cache from HashMap for batch processing
            let mut temp_cache = self
                .caches
                .remove(&ctx.request.id)
                .expect("Cache should exist");

            // Process through batch manager (batch_size=1) and restore cache
            let logits = {
                let mut cache_slice = [temp_cache];
                let result = self.batch_manager.forward_prefill_batch(
                    &input_ids,
                    &metadata,
                    &mut cache_slice,
                )?;
                // Take cache back from slice
                temp_cache = cache_slice.into_iter().next().unwrap();
                result
            };

            // Restore cache to HashMap
            self.caches.insert(ctx.request.id.clone(), temp_cache);

            self.stats.total_forward_time_ms += forward_start.elapsed().as_secs_f64() * 1000.0;

            let logits = logits.i(0)?; // Extract logits for single request
            let next_token = logits.argmax(0)?.to_scalar::<u32>()?;
            ctx.generated_tokens.push(next_token);
            tokens_generated_this_batch += 1;

            if self.is_eos_token(next_token) || ctx.tokens_generated >= ctx.request.max_new_tokens {
                ctx.complete();
            } else {
                ctx.start_decoding();
                ctx.record_token();
            }
        }

        // === DECODE Phase: BATCH-STRUCTURED PROCESSING ===
        // Phase 2D: Group decode requests for batch-oriented processing
        //
        // CURRENT LIMITATION: Candle's Llama model API constraints prevent true batching:
        // - forward(&mut self, x: &Tensor, index_pos: usize, cache: &mut Cache)
        // - Takes single index_pos (not Vec<usize>)
        // - Takes single cache (not batched cache)
        //
        // CURRENT APPROACH: Structural batching with per-request caches
        // - Groups all decode requests together
        // - Processes sequentially but in batch-oriented structure
        // - Prepares for future ScatteredKvCache integration
        //
        // FUTURE (Phase 2D+): True batched decoding
        // - Option A: Custom Llama layer wrapper extracting K/V tensors
        //   - Call model layers directly, bypassing forward()
        //   - Feed K/V into ScatteredKvCache after each layer
        //   - Single batched forward pass for all decode requests
        //
        // - Option B: Fork/modify Candle Llama model
        //   - Add batched_forward(Vec<index_pos>, ScatteredKvCache)
        //   - Direct integration with our cache infrastructure
        //
        // - Option C: Wait for Candle to add native batched cache support
        //
        // Expected improvement with true batching: 5-10x on CPU, 10-50x on GPU

        if !decode_indices.is_empty() {
            // Group decode requests for batched processing (Phase 2D - Approach 1)
            let decode_batch_size = decode_indices.len();

            // Track decode batch for statistics
            if decode_batch_size > 1 {
                // Multiple decode requests - using batched processing!
                self.stats.decode_batch_opportunities += 1;
                self.stats.max_concurrent_decodes =
                    self.stats.max_concurrent_decodes.max(decode_batch_size);
            }

            // Prepare batched tensors
            let mut token_ids = Vec::with_capacity(decode_batch_size);
            let mut positions = Vec::with_capacity(decode_batch_size);
            let mut request_ids = Vec::with_capacity(decode_batch_size);

            for &idx in &decode_indices {
                let ctx = &batch[idx];
                if ctx.generated_tokens.is_empty() {
                    continue;
                }
                let last_token = *ctx.generated_tokens.last().unwrap();
                token_ids.push(last_token);
                positions.push(ctx.position);
                request_ids.push(ctx.request.id.clone());
            }

            if !token_ids.is_empty() {
                // Create batched token tensor [batch_size, 1]
                let tokens_tensor =
                    Tensor::new(&token_ids[..], &self.device)?.reshape(&[token_ids.len(), 1])?;

                // Create batch metadata
                // Temporary: Convert string IDs to numeric for new BatchMetadata API
                let numeric_ids: Vec<usize> = (0..request_ids.len()).collect();
                // Debug: log request ids and positions before creating metadata
                eprintln!(
                    "DEBUG ModelManager: decode batch numeric_ids={:?}, positions={:?}, request_ids={:?}",
                    numeric_ids, positions, request_ids
                );
                let metadata = BatchMetadata::from_decode_batch(numeric_ids.clone(), positions.clone(), positions.clone());

                // Debug: inspect metadata sequences and context_lens
                eprintln!(
                    "DEBUG ModelManager: metadata.is_prefill={} batch_size={} context_lens={:?}",
                    metadata.is_prefill, metadata.batch_size, metadata.context_lens
                );
                for (i, seq) in metadata.sequences.iter().enumerate() {
                    eprintln!(
                        "DEBUG ModelManager: seq {} -> start_pos={}, actual_length={}, padded_length={}, cache_offset={}",
                        i, seq.start_pos, seq.actual_length, seq.padded_length, seq.cache_offset
                    );
                }

                // Collect caches for batch (preserve order)
                let mut batch_caches: Vec<Cache> = Vec::with_capacity(request_ids.len());
                for req_id in &request_ids {
                    let cache = self.caches.remove(req_id).expect("Cache should exist");
                    batch_caches.push(cache);
                }

                // **BATCHED FORWARD PASS** (Model-agnostic BatchManager)
                let forward_start = Instant::now();
                // Debug: log BatchExecutor cache positions before forward
                eprintln!(
                    "DEBUG ModelManager: BatchExecutor cache positions before forward: {:?}",
                    self.batch_manager.batch_executor().device() // placeholder to avoid borrow issues
                );

                let logits_batch = self.batch_manager.forward_decode_batch(
                    &tokens_tensor,
                    &metadata,
                    &mut batch_caches,
                )?;
                self.stats.total_forward_time_ms += forward_start.elapsed().as_secs_f64() * 1000.0;

                // CRITICAL: Advance cache positions after forward pass
                // The BatchExecutor's indices_and_mask is now pure (doesn't mutate positions),
                // so we must explicitly advance positions by 1 for each decode token
                for (i, &cache_idx) in numeric_ids.iter().enumerate() {
                    let old_pos = self.batch_manager.batch_executor_mut().get_cache_position(cache_idx);
                    let new_pos = old_pos + 1;
                    self.batch_manager.batch_executor_mut().set_cache_position(cache_idx, new_pos);
                    eprintln!(
                        "DEBUG: Advanced cache position for slot {}: {} -> {}",
                        cache_idx, old_pos, new_pos
                    );
                }

                // Restore caches (maintain correct request<->cache mapping)
                for (i, req_id) in request_ids.iter().enumerate() {
                    self.caches.insert(req_id.clone(), batch_caches[i].clone());
                }

                // Process results
                for (i, &idx) in decode_indices.iter().enumerate() {
                    let ctx = &mut batch[idx];

                    if ctx.generated_tokens.is_empty() {
                        continue;
                    }

                    // Extract logits for this request [vocab_size]
                    let logits = logits_batch.i(i)?;
                    let next_token = logits.argmax(0)?.to_scalar::<u32>()?;

                    ctx.generated_tokens.push(next_token);
                    ctx.record_token();
                    tokens_generated_this_batch += 1;

                    if self.is_eos_token(next_token)
                        || ctx.tokens_generated >= ctx.request.max_new_tokens
                    {
                        ctx.complete();
                    }
                }
            }
        }

        // Build results in correct order
        for ctx in batch.iter() {
            match ctx.state {
                RequestState::Pending | RequestState::Decoding => {
                    if let Some(&token) = ctx.generated_tokens.last() {
                        results.push(Some(token));
                    } else {
                        results.push(None);
                    }
                }
                RequestState::Completed => results.push(None),
            }
        }

        // Update total token count
        self.stats.total_tokens_generated += tokens_generated_this_batch;

        Ok(results)
    }

    /// Get performance statistics
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = BatchStats::new();
    }

    /// Get the model configuration
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get reference to the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get reference to the tokenizer
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Check if a token is an EOS (end-of-sequence) token
    fn is_eos_token(&self, token: u32) -> bool {
        match &self.config.eos_token_id {
            Some(LlamaEosToks::Single(eos)) => token == *eos,
            Some(LlamaEosToks::Multiple(eos_tokens)) => eos_tokens.contains(&token),
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Only run with actual model present
    fn test_model_loading() {
        let model_path = "../models/llama-3b";
        let result = ModelManager::load(model_path, 4, 512, Some("f32"));

        if Path::new(model_path).exists() {
            assert!(result.is_ok(), "Model loading failed: {:?}", result.err());
            let model = result.unwrap();

            // Verify config loaded
            assert!(model.config().hidden_size > 0);
            assert!(model.config().num_hidden_layers > 0);

            println!("Model loaded successfully!");
            println!("Hidden size: {}", model.config().hidden_size);
            println!("Layers: {}", model.config().num_hidden_layers);
            println!("Heads: {}", model.config().num_attention_heads);
        }
    }

    #[test]
    #[ignore] // Only run with actual model present
    fn test_tokenization() {
        let model_path = "../models/llama-3b";
        if !Path::new(model_path).exists() {
            return;
        }

        let model = ModelManager::load(model_path, 4, 512, Some("f32")).unwrap();

        // Test encoding
        let prompt = "Hello, world!";
        let tokens = model.tokenize(prompt, true).unwrap();
        assert!(!tokens.is_empty(), "Tokenization produced no tokens");

        println!("Prompt: {}", prompt);
        println!("Tokens: {:?}", tokens);

        // Test decoding
        let decoded = model.decode(&tokens, false).unwrap();
        println!("Decoded: {}", decoded);

        // May not match exactly due to BOS token, but should be similar
        assert!(decoded.contains("Hello") || decoded.contains("world"));
    }

    #[test]
    #[ignore] // Only run with actual model present
    fn test_single_request_generation() {
        use crate::engine::Request;

        let model_path = "../models/llama-3b";
        if !Path::new(model_path).exists() {
            return;
        }

        let mut model = ModelManager::load(model_path, 4, 512, Some("f32")).unwrap();

        // Create a single request
        let req = Request {
            id: "test-1".to_string(),
            prompt: "The capital of France is".to_string(),
            max_new_tokens: 5,
        };

        let mut ctx = RequestContext::new(req);
        let mut batch = vec![ctx];

        // Generate tokens
        let mut generated_tokens = Vec::new();
        for step in 0..5 {
            println!("\n=== Step {} ===", step);
            println!("State: {:?}", batch[0].state);

            let tokens = model.forward_batch(&mut batch).unwrap();
            println!("Generated tokens: {:?}", tokens);

            if let Some(Some(token)) = tokens.get(0) {
                generated_tokens.push(*token);
                println!("Token: {} ", token);

                // Decode to see what we're generating
                let decoded = model.decode(&generated_tokens, false).unwrap();
                println!("So far: {}", decoded);
            }

            // Check if should continue
            if !batch[0].should_continue() {
                println!("Generation complete!");
                break;
            }
        }

        assert!(
            !generated_tokens.is_empty(),
            "Should have generated some tokens"
        );
        println!("\n=== Final Result ===");
        println!("Generated {} tokens", generated_tokens.len());

        let full_text = model.decode(&generated_tokens, false).unwrap();
        println!("Generated text: {}", full_text);
    }

    #[test]
    #[ignore] // Only run with actual model present
    fn test_multi_request_generation() {
        use crate::engine::Request;
        use crate::model::ParallelModelManager;

        let model_path = "../models/llama-3b";
        if !Path::new(model_path).exists() {
            return;
        }

        let mut model = ParallelModelManager::load(model_path, 4, 512, Some("f32"), None).unwrap();

        // Create multiple requests
        let requests = vec![
            Request {
                id: "test-1".to_string(),
                prompt: "The capital of France is".to_string(),
                max_new_tokens: 8,
            },
            Request {
                id: "test-2".to_string(),
                prompt: "Rust is a programming".to_string(),
                max_new_tokens: 8,
            },
            Request {
                id: "test-3".to_string(),
                prompt: "Machine learning is".to_string(),
                max_new_tokens: 8,
            },
        ];

        let mut batch: Vec<RequestContext> =
            requests.into_iter().map(RequestContext::new).collect();

        println!("\n=== Starting Multi-Request Generation ===");
        println!("Batch size: {}", batch.len());

        // Generate tokens for all requests
        let max_steps = 15; // Allow enough steps for all to complete
        for step in 0..max_steps {
            println!("\n--- Step {} ---", step);

            // Show status
            for (i, ctx) in batch.iter().enumerate() {
                println!(
                    "Request {}: {:?} (tokens: {})",
                    i, ctx.state, ctx.tokens_generated
                );
            }

            // Forward pass for all requests
            let tokens = model.forward_batch(&mut batch).unwrap();

            // Update generated tokens tracking
            for (i, (ctx, token_opt)) in batch.iter().zip(tokens.iter()).enumerate() {
                if let Some(token) = token_opt {
                    let decoded = model.decode(&ctx.generated_tokens, false).unwrap();
                    println!("Request {}: token={} text=\"{}\"", i, token, decoded);
                }
            }

            // Check if all completed
            if batch.iter().all(|ctx| ctx.state == RequestState::Completed) {
                println!("\n=== All Requests Completed at Step {} ===", step);
                break;
            }
        }

        // Print final results
        println!("\n=== Final Results ===");
        for (i, ctx) in batch.iter().enumerate() {
            let text = model.decode(&ctx.generated_tokens, false).unwrap();
            println!("\nRequest {}: \"{}\"", i, ctx.request.prompt);
            println!("Generated: \"{}\"", text);
            println!("Tokens: {}", ctx.tokens_generated);
            println!("State: {:?}", ctx.state);
        }

        // Verify all requests generated something
        for ctx in &batch {
            assert!(
                !ctx.generated_tokens.is_empty(),
                "Request {} should have generated tokens",
                ctx.request.id
            );
        }

        // Print performance statistics
        println!("\n=== Performance Statistics ===");
        let stats = model.stats();
        println!("Total batches: {}", stats.total_batches);
        println!(
            "Total requests processed: {}",
            stats.total_requests_processed
        );
        println!("Total tokens generated: {}", stats.total_tokens_generated);
        println!("Average batch size: {:.2}", stats.average_batch_size());
        println!("Total forward time: {:.2} ms", stats.total_forward_time_ms);
        println!("Tokens per second: {:.2}", stats.tokens_per_second());
        println!("Prefill batches: {}", stats.prefill_batches);
        println!("Decode batches: {}", stats.decode_batches);
        println!(
            "Padding efficiency: {:.1}%",
            stats.padding_efficiency() * 100.0
        );
    }

    #[test]
    #[ignore] // Only run with actual model present
    fn test_batch_performance() {
        use crate::engine::Request;

        let model_path = "../models/llama-3b";
        if !Path::new(model_path).exists() {
            return;
        }

        let mut model = ModelManager::load(model_path, 8, 512, Some("f32")).unwrap();

        // Create a larger batch to test performance
        let requests: Vec<Request> = (0..6)
            .map(|i| Request {
                id: format!("perf-test-{}", i),
                prompt: match i % 3 {
                    0 => "The capital of France is".to_string(),
                    1 => "Rust programming language".to_string(),
                    _ => "Machine learning".to_string(),
                },
                max_new_tokens: 10,
            })
            .collect();

        let mut batch: Vec<RequestContext> =
            requests.into_iter().map(RequestContext::new).collect();

        println!("\n=== Performance Test ===");
        println!("Batch size: {}", batch.len());
        println!("Max tokens per request: 10");

        // Generate until all complete
        let max_steps = 20;
        for step in 0..max_steps {
            let active_count = batch
                .iter()
                .filter(|ctx| ctx.state != RequestState::Completed)
                .count();
            if active_count == 0 {
                println!("\nAll requests completed at step {}", step);
                break;
            }

            model.forward_batch(&mut batch).unwrap();
        }

        // Print comprehensive statistics
        println!("\n=== Performance Metrics ===");
        let stats = model.stats();
        println!("Total batches processed: {}", stats.total_batches);
        println!("Total requests: {}", stats.total_requests_processed);
        println!("Total tokens generated: {}", stats.total_tokens_generated);
        println!("Average batch size: {:.2}", stats.average_batch_size());
        println!("Total forward time: {:.2} ms", stats.total_forward_time_ms);
        println!(
            "Average forward time per batch: {:.2} ms",
            stats.average_forward_time_ms()
        );
        println!("Throughput: {:.2} tokens/sec", stats.tokens_per_second());
        println!("Prefill operations: {}", stats.prefill_requests);
        println!("Decode operations: {}", stats.decode_requests);
        println!(
            "Prefill/Decode ratio: {:.2}",
            stats.prefill_requests as f64 / stats.decode_requests.max(1) as f64
        );

        // Phase 2D metrics
        println!("\n=== Batching Analysis (Phase 2D) ===");
        println!(
            "Decode batch opportunities: {}",
            stats.decode_batch_opportunities
        );
        println!("Max concurrent decodes: {}", stats.max_concurrent_decodes);
        println!(
            "Batching efficiency: {:.1}%",
            stats.batching_efficiency() * 100.0
        );
        println!(
            "Potential speedup from true batching: {:.2}x",
            stats.potential_batching_speedup()
        );
        println!("\nNote: Current implementation groups decode requests but");
        println!("processes sequentially due to Candle API constraints.");
        println!(
            "True batching would achieve ~{:.2}x improvement.",
            stats.potential_batching_speedup()
        );

        // Verify all completed
        for ctx in &batch {
            assert_eq!(ctx.state, RequestState::Completed);
        }
    }
}
