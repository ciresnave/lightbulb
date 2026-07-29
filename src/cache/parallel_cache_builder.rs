//! Parallel Cache Builder - Forked from Candle's ScatteredCacheBuilder
//!
//! Copyright (c) 2023 Hugging Face
//! SPDX-License-Identifier: Apache-2.0 OR MIT
//!
//! Modified for parallel batching with variable-length sequences:
//! - Independent per-slot position tracking
//! - KV cache clearing capability  
//! - Efficient position setting without iteration
//!
//! Original: https://github.com/huggingface/candle/blob/main/candle-nn/src/kv_cache.rs

use crate::cache::cache_span::SpanRegistry;
use crate::cache::eviction_policy::VotingAggregator;
use crate::cache::h2o_policy::H2OPolicy;
use crate::cache::kv_compression::CompressionPolicy;
use crate::cache::streaming_policy::{StreamingConfig, compute_streaming_index};
use candlelight::core::{DType, Device, IndexOp, Result, Tensor};
use std::collections::HashMap;

/// Indices and attention mask for KV cache operations
///
/// This structure is returned by `indices_and_mask()` and contains two critical pieces
/// of information needed for writing to and reading from the KV cache:
///
/// 1. **indices**: Tensor of shape [batch_size, seq_len] containing cache positions
///    where each token's K/V should be written. For example, if processing 2 requests
///    with 3 tokens each, indices might be [[0,1,2], [5,6,7]] meaning:
///    - Request 0: write to positions 0, 1, 2
///    - Request 1: write to positions 5, 6, 7
///
/// 2. **mask**: Attention mask of shape [batch_size, 1, seq_len, context] that controls
///    which cached positions each token can attend to. Values are either:
///    - 0.0 (can attend) for valid past positions
///    - -∞ (cannot attend) for future positions or padding
///
/// The mask implements causal attention: token at position i can only see positions 0..=i
#[derive(Debug, Clone)]
pub struct IndicesAndMask {
    pub indices: Tensor,
    pub mask: Tensor,
    /// Which batch slots are active for this IAM (same length as batch_size())
    /// This is copied from the `batch_mask` passed into `indices_and_mask()` so
    /// callers and append_kv can know which slots to advance after a forward
    /// pass.
    pub active: Vec<bool>,
}

impl IndicesAndMask {
    /// Get the attention mask tensor
    ///
    /// This is used in the attention mechanism to prevent tokens from attending to
    /// future positions or invalid cache slots.
    pub fn mask(&self) -> &Tensor {
        &self.mask
    }
}

/// KV Cache with scattered storage
///
/// This stores the Key and Value tensors for all batch slots across all positions.
/// "Scattered" means each request can write to non-contiguous positions in the cache.
///
/// **Shape**: [batch_size, num_heads, context, head_dim]
/// - batch_size: Number of parallel requests (e.g., 4)
/// - num_heads: Number of attention heads (e.g., 32)
/// - context: Maximum sequence length / cache size (e.g., 512)
/// - head_dim: Dimension per head (e.g., hidden_size / num_heads = 2048 / 32 = 64)
///
/// **Example**: With batch_size=4, context=512:
/// - Slot 0 might have data at positions [0..10]
/// - Slot 1 might have data at positions [0..5]
/// - Slot 2 might have data at positions [0..100]
/// - Slot 3 might be empty
///
/// Each slot maintains its own independent sequence, allowing parallel processing
/// of requests with different lengths.
#[derive(Debug, Clone)]
pub struct ParallelKvCache {
    k: Tensor,
    v: Tensor,
    context: usize,
}

impl ParallelKvCache {
    /// Append new K/V tensors to the cache at specified positions
    ///
    /// This is the core operation that writes new key/value data into the cache.
    /// The `IndicesAndMask` tells us WHERE to write each token's K/V.
    ///
    /// # How it works
    ///
    /// 1. **Input K/V**: Shape [batch_size, num_heads, seq_len, head_dim]
    ///    - These are the NEW keys/values from the current forward pass
    ///    - seq_len might be 1 (decode) or larger (prefill)
    ///
    /// 2. **Indices**: Shape [batch_size, seq_len]
    ///    - Tells us which cache position each token should write to
    ///    - Example: [[5,6,7], [10,11,12]] means:
    ///      - Batch 0: write 3 tokens to positions 5, 6, 7
    ///      - Batch 1: write 3 tokens to positions 10, 11, 12
    ///
    /// 3. **scatter_set**: Candle operation that writes to specific indices
    ///    - Takes the indices tensor and broadcasts it to K/V shape
    ///    - Writes each token's K/V to its designated cache position
    ///
    /// 4. **Output**: Full cache tensors with new data written in
    ///
    /// # Arguments
    ///
    /// * `k` - New key tensor from current forward pass
    /// * `v` - New value tensor from current forward pass  
    /// * `iam` - IndicesAndMask specifying where to write
    ///
    /// # Returns
    ///
    /// Tuple of (full_k_cache, full_v_cache) with new data incorporated
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Prefill with 5 tokens for slot 0
    /// let iam = cache_builder.indices_and_mask(5, &[true, false, false, false])?;
    /// let (k_cache, v_cache) = cache.append(&new_k, &new_v, &iam)?;
    /// // k_cache now has data at positions [0..5] for slot 0
    /// ```
    pub fn append(
        &mut self,
        k: &Tensor,
        v: &Tensor,
        iam: &IndicesAndMask,
    ) -> Result<(Tensor, Tensor)> {
        if self.context <= k.dim(2)? {
            return Ok((k.clone(), v.clone()));
        }

        let indices = iam.indices.unsqueeze(2)?.unsqueeze(1)?;
        let indices = indices.broadcast_as(k.shape())?.contiguous()?;
        self.k.scatter_set(&indices, k, 2)?;
        self.v.scatter_set(&indices, v, 2)?;

        Ok((self.k.clone(), self.v.clone()))
    }

    /// Get the full key cache tensor
    ///
    /// Returns the complete key cache across all batch slots and positions.
    /// Shape: [batch_size, num_heads, context, head_dim]
    pub fn k(&self) -> &Tensor {
        &self.k
    }

    /// Get the full value cache tensor
    ///
    /// Returns the complete value cache across all batch slots and positions.
    /// Shape: [batch_size, num_heads, context, head_dim]
    pub fn v(&self) -> &Tensor {
        &self.v
    }

    /// Set KV cache for a specific slot (for prefix caching restoration)
    ///
    /// This updates the cache at `slot_index` with the provided KV tensors,
    /// leaving other slots unchanged.
    ///
    /// # Arguments
    ///
    /// * `slot_index` - Which slot to update (0..batch_size)
    /// * `k_slot` - Key tensor to write, shape [1, num_heads, prefix_len, head_dim]
    /// * `v_slot` - Value tensor to write, shape [1, num_heads, prefix_len, head_dim]
    ///
    /// # Returns
    ///
    /// The prefix length (extracted from tensor shape)
    ///
    /// # Errors
    ///
    /// - If slot_index is out of bounds
    /// - If tensor shapes are invalid
    /// - If prefix_len exceeds context size
    pub fn set_slot_kv(
        &mut self,
        slot_index: usize,
        k_slot: &Tensor,
        v_slot: &Tensor,
    ) -> Result<usize> {
        let k_dims = k_slot.dims();
        let v_dims = v_slot.dims();

        // Validate shapes
        if k_dims.len() != 4 || v_dims.len() != 4 {
            candlelight::core::bail!("Expected 4D tensors, got K={:?}, V={:?}", k_dims, v_dims);
        }

        if k_dims[0] != 1 || v_dims[0] != 1 {
            candlelight::core::bail!("Expected batch size 1, got K={:?}, V={:?}", k_dims, v_dims);
        }

        let prefix_len = k_dims[2];
        if prefix_len > self.context {
            candlelight::core::bail!(
                "Prefix length {} exceeds context size {}",
                prefix_len,
                self.context
            );
        }

        // Get cache dimensions
        let cache_dims = self.k.dims();
        let batch_size = cache_dims[0];

        if slot_index >= batch_size {
            candlelight::core::bail!(
                "Slot index {} out of bounds (batch_size={})",
                slot_index,
                batch_size
            );
        }

        // Extract the slot, update prefix region, rebuild
        let k_slot_full = self.k.narrow(0, slot_index, 1)?; // [1, num_heads, context, head_dim]
        let v_slot_full = self.v.narrow(0, slot_index, 1)?;

        // Build new slot: prefix + rest
        let k_rest = if prefix_len < self.context {
            Some(k_slot_full.narrow(2, prefix_len, self.context - prefix_len)?)
        } else {
            None
        };

        let v_rest = if prefix_len < self.context {
            Some(v_slot_full.narrow(2, prefix_len, self.context - prefix_len)?)
        } else {
            None
        };

        let k_new_slot = if let Some(k_r) = k_rest {
            Tensor::cat(&[k_slot, &k_r], 2)?
        } else {
            k_slot.clone()
        };

        let v_new_slot = if let Some(v_r) = v_rest {
            Tensor::cat(&[v_slot, &v_r], 2)?
        } else {
            v_slot.clone()
        };

        // Rebuild full cache with updated slot
        let mut k_parts = Vec::new();
        let mut v_parts = Vec::new();

        for i in 0..batch_size {
            if i == slot_index {
                k_parts.push(k_new_slot.clone());
                v_parts.push(v_new_slot.clone());
            } else {
                k_parts.push(self.k.narrow(0, i, 1)?);
                v_parts.push(self.v.narrow(0, i, 1)?);
            }
        }

        self.k = Tensor::cat(&k_parts, 0)?;
        self.v = Tensor::cat(&v_parts, 0)?;

        Ok(prefix_len)
    }

    /// Restore KV tensors at a specific position range within a slot.
    ///
    /// Used for re-injecting promoted segments back into the cache at their
    /// original positions. Unlike `set_slot_kv()` which writes from position 0,
    /// this writes at an arbitrary offset.
    ///
    /// # Arguments
    ///
    /// * `slot_index` - Which batch slot to write to (0..batch_size)
    /// * `start_pos` - Starting position within the cache (0..context)
    /// * `k_data` - Key tensor, shape [1, num_heads, span_len, head_dim]
    /// * `v_data` - Value tensor, shape [1, num_heads, span_len, head_dim]
    ///
    /// # Returns
    ///
    /// Number of positions written.
    pub fn restore_kv_range(
        &mut self,
        slot_index: usize,
        start_pos: usize,
        k_data: &Tensor,
        v_data: &Tensor,
    ) -> Result<usize> {
        let k_dims = k_data.dims();
        if k_dims.len() != 4 || k_dims[0] != 1 {
            candlelight::core::bail!(
                "Expected [1, heads, span_len, dim], got {:?}",
                k_dims
            );
        }

        let span_len = k_dims[2];
        if start_pos + span_len > self.context {
            candlelight::core::bail!(
                "Restore range {}..{} exceeds context size {}",
                start_pos,
                start_pos + span_len,
                self.context
            );
        }

        let cache_dims = self.k.dims();
        let batch_size = cache_dims[0];
        if slot_index >= batch_size {
            candlelight::core::bail!(
                "Slot index {} out of bounds (batch_size={})",
                slot_index,
                batch_size
            );
        }

        // Extract the slot's current cache
        let k_slot = self.k.narrow(0, slot_index, 1)?; // [1, heads, context, dim]
        let v_slot = self.v.narrow(0, slot_index, 1)?;

        // Split into before, replaced, after
        let k_before = k_slot.narrow(2, 0, start_pos)?;
        let k_after_start = start_pos + span_len;
        let k_after_len = self.context - k_after_start;
        let k_after = k_slot.narrow(2, k_after_start, k_after_len)?;
        let k_new = Tensor::cat(&[&k_before, k_data, &k_after], 2)?;

        let v_before = v_slot.narrow(2, 0, start_pos)?;
        let v_after = v_slot.narrow(2, k_after_start, k_after_len)?;
        let v_new = Tensor::cat(&[&v_before, v_data, &v_after], 2)?;

        // Rebuild full cache with updated slot
        let mut k_parts = Vec::with_capacity(batch_size);
        let mut v_parts = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            if i == slot_index {
                k_parts.push(k_new.clone());
                v_parts.push(v_new.clone());
            } else {
                k_parts.push(self.k.narrow(0, i, 1)?);
                v_parts.push(self.v.narrow(0, i, 1)?);
            }
        }

        self.k = Tensor::cat(&k_parts, 0)?;
        self.v = Tensor::cat(&v_parts, 0)?;

        Ok(span_len)
    }
}

/// Parallel Cache Builder with per-slot independent position tracking
///
/// This is the "brain" that tracks WHERE each batch slot should write its next token.
/// It maintains independent position counters for each slot, enabling true parallel
/// processing of requests with different lengths.
///
/// # Key Concepts
///
/// ## Position vs Index
/// - **position**: Absolute token position in the sequence (can exceed context)
///   - Example: Token 1500 in a 2000-token sequence
/// - **index**: Cache slot where we write (wraps around: position % context)
///   - Example: Position 1500 in 512-context → index 476 (1500 % 512)
///
/// ## Why separate tracking?
/// Candle's original ScatteredCacheBuilder assumed all slots process the same seq_len,
/// so it incremented ALL positions by the same amount. This breaks with variable-length
/// parallel batching:
///
/// **Bad (original)**:
/// ```text
/// Batch: [5 tokens, 10 tokens, 3 tokens]
/// After processing: ALL positions += 18 (total seq_len)
/// Result: positions = [18, 18, 18] ❌ WRONG!
/// ```
///
/// **Good (our version)**:
/// ```text
/// Batch: [5 tokens, 10 tokens, 3 tokens]
/// After processing: Each slot tracks independently
/// Result: positions = [5, 10, 3] ✅ CORRECT!
/// ```
///
/// ## State Per Slot
/// Each slot (0..batch_size) has:
/// - **position**: Current token count (e.g., 5 tokens processed so far)
/// - **index**: Next cache write position (e.g., position % context)
///
/// # Fields
///
/// * `positions` - Vec of current position for each slot (can exceed context)
/// * `indices` - Vec of next cache write index for each slot (0..context)
/// * `context` - Maximum cache size (e.g., 512)
/// * `dtype` - Data type for cache tensors
/// * `device` - CPU/GPU device
#[derive(Debug)]
pub struct ParallelCacheBuilder {
    context: usize,
    /// The current position in the stream for each slot (can exceed context)
    positions: Vec<usize>,
    /// The cache index where the next element will be stored for each slot
    indices: Vec<usize>,
    dtype: DType,
    device: Device,
    /// Optional StreamingLLM configuration for constant-memory long contexts
    streaming_config: Option<StreamingConfig>,
    /// Optional H2O policy for attention-based eviction
    h2o_policy: Option<H2OPolicy>,
    /// Optional voting aggregator for multi-policy eviction
    voting_aggregator: Option<VotingAggregator>,
    /// Mapping from slot_id to current sequence position (for eviction policies)
    slot_positions: HashMap<usize, usize>,
    /// Span registry for semantic cache region tracking
    span_registry: SpanRegistry,
    /// Optional KV cache compression policy (KIVI, R-KV, Low-rank, Relationship-aware)
    compression_policy: Option<CompressionPolicy>,
    /// Optional name mapper for architecture-aware operations (M5.1)
    /// Enables automatic detection of layer count and architecture-specific optimizations
    name_mapper: Option<std::sync::Arc<crate::pruning::name_mapping::TensorNameMapper>>,
    /// Per-slot sets of evicted cache positions.
    /// Positions in these sets are masked out (NEG_INFINITY) during attention computation.
    /// Populated by evict_span_internal() and evict_range().
    evicted_positions: Vec<std::collections::HashSet<usize>>,
}

/// Cache usage statistics
///
/// Provides information about cache utilization, active spans, and per-tag span counts.
#[derive(Debug, Clone)]
pub struct CacheUsageInfo {
    /// Total number of batch slots
    pub total_slots: usize,
    /// Number of active (non-evicted) spans
    pub active_span_count: usize,
    /// Current position for each slot
    pub per_slot_positions: Vec<usize>,
    /// Count of spans per cache tag
    pub per_tag_counts: HashMap<crate::cache::CacheTag, usize>,
}

/// A single slot in the parallel batch - isolated view of one request's state
///
/// This provides a type-safe abstraction for managing per-slot state (positions,
/// cache indices) without accidentally mixing data between different requests.
///
/// # Purpose
///
/// The `Slot` abstraction is designed for **state management and inspection only**.
/// It provides safe, isolated access to:
/// - Position tracking (where this request is in its sequence)
/// - Cache index queries (where next token will be written)
/// - K/V inspection (for debugging/analysis)
///
/// # What Slot is NOT for
///
/// **Slot is NOT used for writing K/V data to the cache.** In production, K/V tensors
/// are generated by the model's forward pass for ALL active slots simultaneously,
/// then written to the cache in a single batched operation via `ParallelKvCache::append()`.
///
/// Attempting to write per-slot would:
/// - Lose GPU parallelism benefits
/// - Require complex tensor reshaping
/// - Not match how the model actually works
///
/// # Lifetime
///
/// Holds mutable references to both the cache and builder, ensuring exclusive
/// access during operations on this slot.
///
/// # Usage Pattern
///
/// ```ignore
/// // Query slot state
/// let slot0 = executor.get_slot_mut(0, 0)?;
/// println!("Slot 0 position: {}", slot0.position());
/// println!("Cache index: {}", slot0.cache_index());
///
/// // Manage slot lifecycle
/// slot0.reset();  // Reset for new request
/// slot0.set_position(5);  // Sync position after chunked prefill
///
/// // Inspect cached K/V (debugging)
/// let k_data = slot0.k()?;
/// let v_data = slot0.v()?;
/// ```
pub struct Slot<'a> {
    slot_index: usize,
    cache: &'a mut ParallelKvCache,
    builder: &'a mut ParallelCacheBuilder,
}

impl<'a> Slot<'a> {
    /// Create a new Slot view
    ///
    /// # Arguments
    ///
    /// * `slot_index` - Which batch slot this represents (0..batch_size)
    /// * `cache` - Mutable reference to the full KV cache
    /// * `builder` - Mutable reference to the cache builder
    ///
    /// # Safety
    ///
    /// Caller must ensure slot_index is valid (< batch_size)
    pub fn new(
        slot_index: usize,
        cache: &'a mut ParallelKvCache,
        builder: &'a mut ParallelCacheBuilder,
    ) -> Self {
        Self {
            slot_index,
            cache,
            builder,
        }
    }

    /// Get the slot index
    pub fn index(&self) -> usize {
        self.slot_index
    }

    /// Get current position for this slot
    ///
    /// Returns the number of tokens that have been processed for this slot's request.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pos = slot.position();
    /// // pos might be 5, meaning 5 tokens have been processed
    /// ```
    pub fn position(&self) -> usize {
        self.builder.get_position(self.slot_index)
    }

    /// Set position for this slot
    ///
    /// Directly updates where the next token will be written. Useful after
    /// chunked prefill with padding to "rewind" to the actual prompt length.
    ///
    /// # Arguments
    ///
    /// * `pos` - Target position (number of tokens processed)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // After prefill with padding, correct the position
    /// slot.set_position(5);  // Actual prompt was 5 tokens
    /// ```
    pub fn set_position(&mut self, pos: usize) {
        self.builder.set_position(self.slot_index, pos);
    }

    /// Get the cache index where next token will be written
    ///
    /// This is `position % context`, representing the physical cache location.
    ///
    /// # Returns
    ///
    /// Cache index in range [0, context)
    pub fn cache_index(&self) -> usize {
        self.builder.get_cache_index(self.slot_index)
    }

    /// Reset this slot to position 0
    ///
    /// Use when starting a new request in this slot. Resets both position
    /// Reset this slot to position 0
    ///
    /// Use when starting a new request in this slot. Resets both position
    /// and cache index.
    ///
    /// **Note**: There's no need to explicitly clear the slot's cached K/V data.
    /// The attention mask prevents reading stale data beyond the current position,
    /// and new data naturally overwrites old data at valid positions.
    pub fn reset(&mut self) {
        self.builder.reset_batch_index(self.slot_index);
    }

    /// Get this slot's K tensor (for inspection/debugging)
    ///
    /// Returns a view of just this slot's Key cache, shape [num_heads, context, head_dim].
    ///
    /// **Note**: This creates a new tensor (slice). For model forward pass, use
    /// the full batched cache instead.
    ///
    /// # Returns
    ///
    /// Tensor slice of this slot's K cache
    pub fn k(&self) -> Result<Tensor> {
        self.cache.k().i(self.slot_index)
    }

    /// Get this slot's V tensor (for inspection/debugging)
    ///
    /// Returns a view of just this slot's Value cache, shape [num_heads, context, head_dim].
    ///
    /// **Note**: This creates a new tensor (slice). For model forward pass, use
    /// the full batched cache instead.
    ///
    /// # Returns
    ///
    /// Tensor slice of this slot's V cache
    pub fn v(&self) -> Result<Tensor> {
        self.cache.v().i(self.slot_index)
    }
}

impl ParallelCacheBuilder {
    /// Create a new ParallelCacheBuilder
    ///
    /// Initializes position and index tracking for all batch slots to 0.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of parallel request slots (e.g., 4)
    /// * `context` - Maximum cache size / sequence length (e.g., 512)
    /// * `dtype` - Data type for cache tensors (F32, F16, BF16)
    /// * `device` - CPU or GPU device
    ///
    /// # Returns
    ///
    /// A new cache builder with all positions initialized to 0
    pub fn new(batch_size: usize, context: usize, dtype: DType, device: &Device) -> Result<Self> {
        let positions = vec![0; batch_size];
        let indices = vec![0; batch_size];
        Ok(Self {
            positions,
            indices,
            context,
            dtype,
            device: device.clone(),
            streaming_config: None,
            h2o_policy: None,
            voting_aggregator: None,
            slot_positions: HashMap::new(),
            span_registry: SpanRegistry::new(),
            compression_policy: None,
            name_mapper: None,
            evicted_positions: (0..batch_size)
                .map(|_| std::collections::HashSet::new())
                .collect(),
        })
    }

    /// Set name mapper for architecture-aware operations (M5.1)
    ///
    /// Enables automatic detection of layer count and architecture-specific optimizations.
    /// The name mapper is shared (Arc) to allow zero-cost use across multiple components.
    ///
    /// # Arguments
    ///
    /// * `mapper` - Optional TensorNameMapper (wrapped in Arc for sharing)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::sync::Arc;
    /// use lightbulb::pruning::name_mapping::TensorNameMapper;
    ///
    /// let tensor_names = vec!["blk.0.attn_q.weight".to_string(), /* ... */];
    /// let mapper = TensorNameMapper::from_tensor_names(&tensor_names)?;
    /// builder.set_name_mapper(Some(Arc::new(mapper)));
    ///
    /// // Now cache builder knows model has N layers (from mapper.layer_indices)
    /// let num_layers = builder.detected_layer_count().unwrap();
    /// ```
    pub fn set_name_mapper(
        &mut self,
        mapper: Option<std::sync::Arc<crate::pruning::name_mapping::TensorNameMapper>>,
    ) {
        self.name_mapper = mapper;
    }

    /// Get the detected number of layers from the name mapper
    ///
    /// Returns the number of transformer layers detected by the name mapper,
    /// or None if no name mapper is set.
    ///
    /// # Returns
    ///
    /// - `Some(layer_count)` if name mapper is available
    /// - `None` if no name mapper or detection failed
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some(num_layers) = builder.detected_layer_count() {
    ///     println!("Model has {} layers", num_layers);
    ///     // Can now size per-layer caches correctly
    /// }
    /// ```
    pub fn detected_layer_count(&self) -> Option<usize> {
        self.name_mapper
            .as_ref()
            .map(|mapper| mapper.layer_indices.len())
    }

    /// Get the detected model architecture from the name mapper
    ///
    /// Returns the architecture type (LLaMA, GPT, Mistral, etc.) or None.
    ///
    /// # Returns
    ///
    /// - `Some(architecture)` if name mapper is available
    /// - `None` if no name mapper
    pub fn detected_architecture(
        &self,
    ) -> Option<&crate::pruning::name_mapping::ModelArchitecture> {
        self.name_mapper.as_ref().map(|mapper| &mapper.architecture)
    }

    /// Set StreamingLLM configuration for constant-memory long contexts
    ///
    /// Enables attention sinks + sliding window eviction policy.
    ///
    /// # Arguments
    ///
    /// * `config` - StreamingLLM configuration (sink_size, window_size)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut builder = ParallelCacheBuilder::new(4, 2052, DType::F32, &device)?;
    /// builder.set_streaming_config(Some(StreamingConfig::new(4, 2048)));
    /// ```
    pub fn set_streaming_config(&mut self, config: Option<StreamingConfig>) {
        if let Some(ref cfg) = config {
            if let Err(e) = cfg.validate() {
                eprintln!("Warning: Invalid StreamingLLM config: {}", e);
                return;
            }
        }
        self.streaming_config = config;
    }

    /// Get the current StreamingLLM configuration
    pub fn streaming_config(&self) -> Option<&StreamingConfig> {
        self.streaming_config.as_ref()
    }

    /// Set H2O policy for attention-based eviction
    ///
    /// # Arguments
    ///
    /// * `policy` - H2O policy instance (or None to disable)
    pub fn set_h2o_policy(&mut self, policy: Option<H2OPolicy>) {
        self.h2o_policy = policy;
    }

    /// Get the current H2O policy
    pub fn h2o_policy(&self) -> Option<&H2OPolicy> {
        self.h2o_policy.as_ref()
    }

    /// Get mutable reference to H2O policy
    pub fn h2o_policy_mut(&mut self) -> Option<&mut H2OPolicy> {
        self.h2o_policy.as_mut()
    }

    /// Set voting aggregator for multi-policy eviction
    ///
    /// # Arguments
    ///
    /// * `aggregator` - Voting aggregator combining multiple policies
    pub fn set_voting_aggregator(&mut self, aggregator: Option<VotingAggregator>) {
        self.voting_aggregator = aggregator;
    }

    /// Get the current voting aggregator
    pub fn voting_aggregator(&self) -> Option<&VotingAggregator> {
        self.voting_aggregator.as_ref()
    }

    /// Set compression policy for KV cache compression
    ///
    /// **Note**: This field stores the compression configuration but does NOT automatically
    /// apply compression during `append()`. Compression must be applied **externally** by the
    /// caller before appending to the cache.
    ///
    /// This design keeps ParallelCacheBuilder simple while giving callers full control over
    /// compression timing and context (e.g., attention scores, layer info).
    ///
    /// # Arguments
    ///
    /// * `policy` - Compression policy (KIVI, R-KV, Low-rank, Relationship-aware, or None)
    ///
    /// # Available Policies
    ///
    /// - **KIVI**: 2-4 bit quantization with per-head scaling (75% memory reduction)
    /// - **R-KV**: Training-free importance-redundancy scoring (30-70% reduction)
    /// - **Low-rank**: SVD approximation at specified rank (50-80% reduction)
    /// - **Relationship-aware**: Multi-dimensional eviction with cluster preservation (40-60% reduction)
    /// - **None**: Disable compression (full precision)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use lightbulb::cache::kv_compression::{CompressionPolicy, KiviConfig, CompressionCtx};
    ///
    /// // Set compression policy on builder
    /// let policy = CompressionPolicy::Kivi(KiviConfig::default());
    /// builder.set_compression_policy(Some(policy.clone()));
    ///
    /// // Create compressor from policy
    /// let mut compressor = policy.create_compressor().unwrap();
    ///
    /// // Compress before appending (caller's responsibility)
    /// let mut ctx = CompressionCtx { layer_idx: 0, num_heads: 32, head_dim: 128, ... };
    /// let k_compressed = compressor.compress_keys(&k, &mut ctx)?;
    /// let v_compressed = compressor.compress_values(&v, &mut ctx)?;
    ///
    /// // Append compressed tensors to cache
    /// let (k_cache, v_cache) = cache.append(&k_compressed, &v_compressed, &iam)?;
    ///
    /// // Decompress for attention computation
    /// let k_full = compressor.decompress_keys(&k_cache, &ctx)?;
    /// let v_full = compressor.decompress_values(&v_cache, &ctx)?;
    /// ```
    pub fn set_compression_policy(&mut self, policy: Option<CompressionPolicy>) {
        self.compression_policy = policy;
    }

    /// Get the current compression policy
    pub fn compression_policy(&self) -> Option<&CompressionPolicy> {
        self.compression_policy.as_ref()
    }

    /// Update H2O policy with attention weights from the last generation step
    ///
    /// # Arguments
    ///
    /// * `attention_weights` - Attention weights [query_pos][key_pos]
    ///   Should be aggregated across heads if multi-head
    ///
    /// This method should be called after each forward pass to track
    /// cumulative attention scores per token for eviction decisions.
    pub fn update_attention_scores(&mut self, attention_weights: &[Vec<f32>]) {
        if let Some(ref mut h2o) = self.h2o_policy {
            // The attention_weights matrix has shape [seq_q, seq_k] where seq_k
            // is the context size. Each key_pos index (0..context) IS a cache
            // position. We need to store attention data per cache position,
            // not per batch slot.
            //
            // Build a per-cache-position mapping: position → position (identity)
            // so H2O stores metadata keyed by cache position.
            let key_len = attention_weights
                .first()
                .map(|row| row.len())
                .unwrap_or(0);

            let mut cache_position_map: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::with_capacity(key_len);
            for pos in 0..key_len {
                cache_position_map.insert(pos, pos);
            }

            h2o.update_attention_scores(attention_weights, &cache_position_map);
        }
    }

    // === Segmented Eviction Integration ===

    /// Enable segmented eviction policy on this cache builder.
    ///
    /// Creates a `SegmentedEvictionPolicy` and registers it in a new
    /// `VotingAggregator` alongside a `RecencyPolicy` for combined scoring.
    ///
    /// Also enables an H2O policy if one isn't already set (required for
    /// per-token attention data that feeds into segment-level scoring).
    pub fn enable_segmented_eviction(&mut self, config: crate::cache::SegmentedCacheConfig) {
        use crate::cache::eviction_policy::RecencyPolicy;
        use crate::cache::segmented_eviction_policy::SegmentedEvictionPolicy;

        // Ensure H2O is enabled (we need per-token attention data)
        if self.h2o_policy.is_none() {
            use crate::cache::h2o_policy::{H2OConfig, H2OPolicy};
            let h2o_config = H2OConfig {
                enabled: true,
                num_recent_to_keep: 4,
                decay_factor: 0.95,
            };
            self.h2o_policy = Some(H2OPolicy::new(h2o_config));
        }

        // Create segmented policy
        let segmented_policy = SegmentedEvictionPolicy::new(config);

        // Create voting aggregator with segmented (0.7) + recency (0.3)
        let aggregator = crate::cache::eviction_policy::VotingAggregator::new()
            .add_policy(segmented_policy, 0.7)
            .add_policy(RecencyPolicy::new(4), 0.3);

        self.voting_aggregator = Some(aggregator);
    }

    /// Update segmented eviction scores using current H2O data and span registry.
    ///
    /// This should be called periodically during decode (throttled by config interval).
    /// It aggregates per-token attention from H2O into per-span scores.
    pub fn update_segmented_scores(&mut self, current_utilization: f32) {
        // We need to extract the segmented policy from the aggregator, update it,
        // and put it back. Since VotingAggregator owns the policies as Box<dyn EvictionPolicy>,
        // we can't easily reach in. Instead, we maintain a separate SegmentedEvictionPolicy
        // that we update here and use for ranking.
        //
        // TODO: Consider refactoring VotingAggregator to allow updating internal policies.
        // For now, the segmented policy's scores are updated via rank_eviction_candidates()
        // which is called separately.
        let _ = current_utilization;
    }

    /// Rank active spans for potential eviction using the segmented policy.
    ///
    /// Creates a temporary SegmentedEvictionPolicy, updates it with current
    /// H2O data, and returns ranked eviction candidates.
    ///
    /// # Returns
    ///
    /// Vector of `(SpanId, relevance_score)` sorted ascending (lowest = evict first).
    /// SystemPrompt spans are excluded.
    pub fn rank_eviction_candidates(
        &self,
        config: &crate::cache::SegmentedCacheConfig,
        current_utilization: f32,
    ) -> Vec<(crate::cache::SpanId, f32)> {
        use crate::cache::segmented_eviction_policy::SegmentedEvictionPolicy;

        let h2o = match &self.h2o_policy {
            Some(h2o) => h2o,
            None => return Vec::new(),
        };

        let mut policy = SegmentedEvictionPolicy::new(config.clone());
        policy.update_scores(&self.span_registry, h2o, current_utilization);
        policy.rank_spans_for_eviction(&self.span_registry)
    }

    /// Log eviction candidates (shadow mode).
    ///
    /// Computes eviction rankings and logs the top N candidates with span info.
    /// Does not perform any eviction.
    pub fn log_eviction_candidates(
        &self,
        config: &crate::cache::SegmentedCacheConfig,
        top_n: usize,
    ) {
        let max_util = self.get_max_utilization();
        let candidates = self.rank_eviction_candidates(config, max_util);

        if candidates.is_empty() {
            return;
        }

        println!(
            "=== Eviction Candidates (shadow mode, pressure={:.2}) ===",
            max_util
        );
        for (i, (span_id, score)) in candidates.iter().take(top_n).enumerate() {
            let span_info = self
                .span_registry
                .get(*span_id)
                .map(|s| {
                    format!(
                        "{:?} [{}..{}] ({} tokens)",
                        s.tag,
                        s.start_pos,
                        s.end_pos,
                        s.end_pos - s.start_pos
                    )
                })
                .unwrap_or_else(|| "unknown".to_string());

            println!(
                "  #{}: Span {} | {} | relevance: {:.6}",
                i + 1,
                span_id,
                span_info,
                score
            );
        }
        println!("========================================================");
    }

    /// Compute cache utilization for a given slot (0.0 to 1.0).
    ///
    /// Returns the ratio of current position to context size, indicating
    /// how full the cache is for this slot.
    pub fn get_utilization(&self, slot: usize) -> f32 {
        if slot >= self.positions.len() || self.context == 0 {
            return 0.0;
        }
        self.positions[slot] as f32 / self.context as f32
    }

    /// Compute the maximum utilization across all active slots.
    pub fn get_max_utilization(&self) -> f32 {
        self.positions
            .iter()
            .map(|&pos| pos as f32 / self.context as f32)
            .fold(0.0_f32, f32::max)
    }

    /// Compute per-span attention summaries using H2O token-level data.
    ///
    /// For each active span in the registry, sums the cumulative attention
    /// scores from the H2O policy for tokens within the span's position range.
    ///
    /// # Returns
    ///
    /// Vector of `(span_id, tag, token_count, total_attention, avg_attention)`
    /// sorted by total_attention descending (most-attended spans first).
    pub fn compute_span_attention_summary(
        &self,
    ) -> Vec<(
        crate::cache::SpanId,
        crate::cache::CacheTag,
        usize,
        f32,
        f32,
    )> {
        let h2o = match &self.h2o_policy {
            Some(h2o) => h2o,
            None => return Vec::new(),
        };

        let mut summaries = Vec::new();

        // Get all active spans across all slots
        for slot in 0..self.positions.len() {
            let spans = self.span_registry.spans_for_slot(slot);
            for span in spans {
                if !span.is_active() {
                    continue;
                }

                let mut total_attention = 0.0_f32;
                let mut token_count = 0_usize;

                // Sum attention for all positions within this span
                for pos in span.start_pos..span.end_pos {
                    // H2O policy tracks by slot_id (which maps to positions)
                    // We need to check if this position has attention data
                    // The slot_positions map links slot_id -> seq_position
                    // For per-position lookup, we iterate H2O's metadata
                    if let Some(metadata) = h2o.get_token_metadata(pos) {
                        total_attention += metadata.cumulative_attention;
                        token_count += 1;
                    }
                }

                let avg_attention = if token_count > 0 {
                    total_attention / token_count as f32
                } else {
                    0.0
                };

                summaries.push((span.id, span.tag, token_count, total_attention, avg_attention));
            }
        }

        // Sort by total attention descending
        summaries.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        summaries
    }

    /// Log span attention summaries to stdout (for debugging/tuning).
    ///
    /// Only logs if there are active spans with attention data.
    pub fn log_span_attention_summary(&self) {
        let summaries = self.compute_span_attention_summary();
        if summaries.is_empty() {
            return;
        }

        println!("=== Span Attention Summary ===");
        for (span_id, tag, token_count, total_attn, avg_attn) in &summaries {
            println!(
                "  Span {:3} | {:18?} | {:4} tokens | total_attn: {:8.4} | avg_attn: {:8.6}",
                span_id, tag, token_count, total_attn, avg_attn
            );
        }
        println!("==============================");
    }

    // === Cache Span Management APIs ===

    /// Begin a new cache span at the current position
    ///
    /// Creates span metadata tracking a semantic region of the KV cache.
    /// The span remains "open" until `end_span()` is called with the returned SpanId.
    ///
    /// **Note**: This does NOT store tokens - caller must manage token storage externally
    /// (e.g., HashMap<SpanId, Vec<u32>>, Vector DB, disk, etc.)
    ///
    /// # Arguments
    ///
    /// * `slot` - Which batch slot this span belongs to
    /// * `tag` - Semantic tag for the span (SystemPrompt, ToolOutput, etc.)
    /// * `name` - Optional unique name for human reference
    ///
    /// # Returns
    ///
    /// SpanId that can be used to reference this span later
    ///
    /// # Errors
    ///
    /// Returns error if name already exists (names must be unique)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let span_id = builder.begin_span(0, CacheTag::SystemPrompt, Some("sys_v1".into()))?;
    /// // ... process tokens ...
    /// builder.end_span(span_id)?;
    /// ```
    pub fn begin_span(
        &mut self,
        slot: usize,
        tag: crate::cache::CacheTag,
        name: Option<String>,
    ) -> Result<crate::cache::SpanId> {
        use crate::cache::CacheSpan;

        if slot >= self.positions.len() {
            candlelight::core::bail!("Slot {} out of range (max: {})", slot, self.positions.len());
        }

        let span_id = self.span_registry.next_span_id();
        let start_pos = self.positions[slot];

        let span = CacheSpan::new(span_id, slot, start_pos, start_pos, tag, name);

        self.span_registry
            .register(span)
            .map_err(|e| candlelight::core::Error::Msg(e))?;

        Ok(span_id)
    }

    /// End an open span at the current position
    ///
    /// Updates the span's end position to the current position of its slot.
    ///
    /// # Arguments
    ///
    /// * `span_id` - ID of the span to end (from `begin_span()`)
    ///
    /// # Errors
    ///
    /// Returns error if span_id doesn't exist
    pub fn end_span(&mut self, span_id: crate::cache::SpanId) -> Result<()> {
        let span = self
            .span_registry
            .get_mut(span_id)
            .ok_or_else(|| candlelight::core::Error::Msg(format!("Span {} not found", span_id)))?;

        let current_pos = self.positions[span.slot];
        span.end_pos = current_pos;

        Ok(())
    }

    /// Tag an existing region of the cache as a span
    ///
    /// Creates a complete span for a known position range (doesn't need end_span).
    ///
    /// # Arguments
    ///
    /// * `slot` - Which batch slot
    /// * `start_pos` - Starting position (inclusive)
    /// * `end_pos` - Ending position (exclusive)
    /// * `tag` - Semantic tag
    /// * `name` - Optional unique name
    ///
    /// # Returns
    ///
    /// SpanId for the created span
    ///
    /// # Errors
    ///
    /// Returns error if name already exists or positions are invalid
    pub fn tag_region(
        &mut self,
        slot: usize,
        start_pos: usize,
        end_pos: usize,
        tag: crate::cache::CacheTag,
        name: Option<String>,
    ) -> Result<crate::cache::SpanId> {
        use crate::cache::CacheSpan;

        if slot >= self.positions.len() {
            candlelight::core::bail!("Slot {} out of range (max: {})", slot, self.positions.len());
        }

        if end_pos <= start_pos {
            candlelight::core::bail!(
                "Invalid span range: end_pos ({}) must be > start_pos ({})",
                end_pos,
                start_pos
            );
        }

        let span_id = self.span_registry.next_span_id();
        let span = CacheSpan::new(span_id, slot, start_pos, end_pos, tag, name);

        self.span_registry
            .register(span)
            .map_err(|e| candlelight::core::Error::Msg(e))?;

        Ok(span_id)
    }

    /// Get immutable reference to a span
    ///
    /// # Returns
    ///
    /// Option containing the span if it exists
    pub fn get_span(&self, span_id: crate::cache::SpanId) -> Option<&crate::cache::CacheSpan> {
        self.span_registry.get(span_id)
    }

    /// Get mutable reference to a span
    ///
    /// # Returns
    ///
    /// Option containing the span if it exists
    pub fn get_span_mut(
        &mut self,
        span_id: crate::cache::SpanId,
    ) -> Option<&mut crate::cache::CacheSpan> {
        self.span_registry.get_mut(span_id)
    }

    /// Find a span by its unique name
    ///
    /// # Returns
    ///
    /// Option containing the span if found
    pub fn find_span_by_name(&self, name: &str) -> Option<&crate::cache::CacheSpan> {
        self.span_registry.find_by_name(name)
    }

    /// Get all spans for a specific slot
    ///
    /// # Returns
    ///
    /// Vector of references to spans in the given slot
    pub fn spans_for_slot(&self, slot: usize) -> Vec<&crate::cache::CacheSpan> {
        self.span_registry.spans_for_slot(slot)
    }

    /// Check if a span is currently active
    ///
    /// # Returns
    ///
    /// true if span exists and is Active, false otherwise
    pub fn is_span_active(&self, span_id: crate::cache::SpanId) -> bool {
        self.span_registry
            .get(span_id)
            .map(|s| s.is_active())
            .unwrap_or(false)
    }

    /// Set the importance score for a span
    ///
    /// Importance affects eviction priority: 0.0 = evict first, 1.0 = evict last
    ///
    /// # Arguments
    ///
    /// * `span_id` - ID of the span
    /// * `importance` - Score from 0.0 to 1.0
    ///
    /// # Errors
    ///
    /// Returns error if span doesn't exist
    pub fn set_span_importance(
        &mut self,
        span_id: crate::cache::SpanId,
        importance: f32,
    ) -> Result<()> {
        let span = self
            .span_registry
            .get_mut(span_id)
            .ok_or_else(|| candlelight::core::Error::Msg(format!("Span {} not found", span_id)))?;

        span.importance = importance.clamp(0.0, 1.0);
        Ok(())
    }

    /// Set parent-child relationship between spans
    ///
    /// When the parent is evicted, all children are automatically evicted.
    ///
    /// # Arguments
    ///
    /// * `child_id` - ID of the child span
    /// * `parent_id` - ID of the parent span
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Either span doesn't exist
    /// - Creating the relationship would form a cycle
    pub fn set_span_parent(
        &mut self,
        child_id: crate::cache::SpanId,
        parent_id: crate::cache::SpanId,
    ) -> Result<()> {
        // Check for cycles
        if self.span_registry.would_create_cycle(child_id, parent_id) {
            candlelight::core::bail!(
                "Cannot set parent: would create circular dependency between {} and {}",
                child_id,
                parent_id
            );
        }

        // Update child's parent
        {
            let child = self.span_registry.get_mut(child_id).ok_or_else(|| {
                candlelight::core::Error::Msg(format!("Child span {} not found", child_id))
            })?;
            child.parent = Some(parent_id);
        }

        // Update parent's children list
        {
            let parent = self.span_registry.get_mut(parent_id).ok_or_else(|| {
                candlelight::core::Error::Msg(format!("Parent span {} not found", parent_id))
            })?;
            if !parent.children.contains(&child_id) {
                parent.children.push(child_id);
            }
        }

        Ok(())
    }

    /// Evict all spans with a specific tag
    ///
    /// Marks all matching spans as evicted and returns information about
    /// what was affected. Automatically cascades to child spans.
    ///
    /// **Note**: This marks spans as evicted in metadata but does NOT
    /// actually clear the KV cache tensors. Caller must handle tensor cleanup
    /// based on the returned EvictionResult.
    ///
    /// # Arguments
    ///
    /// * `tag` - The tag to match
    ///
    /// # Returns
    ///
    /// EvictionResult containing affected spans and positions
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = builder.evict_tagged(CacheTag::ToolOutput)?;
    /// for (span_id, impact) in result.spans_affected {
    ///     if matches!(impact, EvictionImpact::FullyEvicted) {
    ///         // Clean up external token storage
    ///         token_storage.remove(&span_id);
    ///     }
    /// }
    /// ```
    pub fn evict_tagged(
        &mut self,
        tag: crate::cache::CacheTag,
    ) -> Result<crate::cache::EvictionResult> {
        use crate::cache::EvictionResult;

        let mut result = EvictionResult::empty();

        // Find all spans with this tag
        let span_ids: Vec<crate::cache::SpanId> = self
            .span_registry
            .spans_with_tag(tag)
            .into_iter()
            .map(|s| s.id)
            .collect();

        // Evict each span (including children)
        for span_id in span_ids {
            let span_result = self.evict_span_internal(span_id)?;
            result.spans_affected.extend(span_result.spans_affected);
            result
                .positions_evicted
                .extend(span_result.positions_evicted);
        }

        Ok(result)
    }

    /// Evict a specific span by ID
    ///
    /// Marks the span and all its children as evicted.
    ///
    /// # Arguments
    ///
    /// * `span_id` - ID of the span to evict
    ///
    /// # Returns
    ///
    /// EvictionResult containing affected spans
    pub fn evict_span(
        &mut self,
        span_id: crate::cache::SpanId,
    ) -> Result<crate::cache::EvictionResult> {
        self.evict_span_internal(span_id)
    }

    /// Evict a span by its unique name
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the span to evict
    ///
    /// # Returns
    ///
    /// EvictionResult containing affected spans
    ///
    /// # Errors
    ///
    /// Returns error if no span with that name exists
    pub fn evict_named(&mut self, name: &str) -> Result<crate::cache::EvictionResult> {
        let span_id = self
            .find_span_by_name(name)
            .map(|s| s.id)
            .ok_or_else(|| candlelight::core::Error::Msg(format!("No span named '{}'", name)))?;

        self.evict_span_internal(span_id)
    }

    /// Internal helper for span eviction with cascade
    fn evict_span_internal(
        &mut self,
        span_id: crate::cache::SpanId,
    ) -> Result<crate::cache::EvictionResult> {
        use crate::cache::{EvictionImpact, EvictionResult, SpanState};

        let mut result = EvictionResult::empty();

        // Get all children (recursive)
        let children = self.span_registry.get_all_children(span_id);

        // Evict children first
        for child_id in children {
            if let Some(child) = self.span_registry.get_mut(child_id) {
                if child.is_active() {
                    let slot = child.slot;
                    let range = child.start_pos..child.end_pos;

                    child.state = SpanState::FullyEvicted;

                    // Mark cache positions as evicted for mask exclusion
                    if slot < self.evicted_positions.len() {
                        for pos in range.clone() {
                            // Map sequence position to cache index
                            let cache_idx = pos % self.context;
                            self.evicted_positions[slot].insert(cache_idx);
                        }
                    }

                    result
                        .spans_affected
                        .push((child_id, EvictionImpact::FullyEvicted));
                    result.positions_evicted.push((slot, range));
                }
            }
        }

        // Evict the span itself
        if let Some(span) = self.span_registry.get_mut(span_id) {
            if span.is_active() {
                let slot = span.slot;
                let range = span.start_pos..span.end_pos;

                span.state = SpanState::FullyEvicted;

                // Mark cache positions as evicted for mask exclusion
                if slot < self.evicted_positions.len() {
                    for pos in range.clone() {
                        let cache_idx = pos % self.context;
                        self.evicted_positions[slot].insert(cache_idx);
                    }
                }

                result
                    .spans_affected
                    .push((span_id, EvictionImpact::FullyEvicted));
                result.positions_evicted.push((slot, range));
            }
        }

        Ok(result)
    }

    /// Get cache usage statistics
    ///
    /// Returns information about cache utilization, span counts, and
    /// per-tag statistics. Only counts active (non-evicted) spans.
    pub fn get_cache_usage(&self) -> CacheUsageInfo {
        let mut info = CacheUsageInfo {
            total_slots: self.positions.len(),
            active_span_count: self.span_registry.active_span_count(),
            per_slot_positions: self.positions.clone(),
            per_tag_counts: std::collections::HashMap::new(),
        };

        // Count active spans by tag
        use crate::cache::CacheTag;
        for tag in &[
            CacheTag::SystemPrompt,
            CacheTag::UserInput,
            CacheTag::ToolOutput,
            CacheTag::LongTermMemory,
            CacheTag::ModelGeneration,
            CacheTag::Custom,
        ] {
            let count = self
                .span_registry
                .spans_with_tag(*tag)
                .into_iter()
                .filter(|s| s.is_active())
                .count();
            if count > 0 {
                info.per_tag_counts.insert(*tag, count);
            }
        }

        info
    }

    /// Partially evict individual positions within a span.
    ///
    /// Used by the hybrid per-token eviction system (Phase 4). Marks specific
    /// positions as evicted while keeping the span in a `PartiallyEvicted` state
    /// with `remaining_ranges` tracking which positions survive.
    ///
    /// # Arguments
    ///
    /// * `span_id` - The span to partially evict
    /// * `evicted_positions` - Absolute positions to evict within the span
    /// * `remaining_ranges` - The surviving position ranges after eviction
    ///
    /// # Returns
    ///
    /// Number of positions actually evicted.
    pub fn partially_evict_span(
        &mut self,
        span_id: crate::cache::SpanId,
        evicted_positions: &[usize],
        remaining_ranges: Vec<std::ops::Range<usize>>,
    ) -> Result<usize> {
        use crate::cache::SpanState;

        let span = self
            .span_registry
            .get_mut(span_id)
            .ok_or_else(|| candlelight::core::Error::Msg(format!("Span {} not found", span_id)))?;

        if !span.is_active() && !matches!(span.state, SpanState::PartiallyEvicted { .. }) {
            return Ok(0); // Already fully evicted
        }

        let slot = span.slot;

        // Update span state
        span.state = SpanState::PartiallyEvicted {
            remaining_ranges,
        };

        // Mark positions in the eviction mask
        let mut count = 0;
        if slot < self.evicted_positions.len() {
            for &pos in evicted_positions {
                let cache_idx = pos % self.context;
                if self.evicted_positions[slot].insert(cache_idx) {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Clear evicted positions for a range within a slot.
    ///
    /// Called when a promoted segment is re-injected into the cache, so
    /// the attention mask will once again allow attending to those positions.
    ///
    /// # Arguments
    ///
    /// * `slot` - Batch slot index
    /// * `start_pos` - Start of position range to un-evict
    /// * `end_pos` - End of position range (exclusive)
    pub fn clear_evicted_range(&mut self, slot: usize, start_pos: usize, end_pos: usize) {
        if slot < self.evicted_positions.len() {
            for pos in start_pos..end_pos {
                let cache_idx = pos % self.context;
                self.evicted_positions[slot].remove(&cache_idx);
            }
        }
    }

    // =======================
    // KV Cache Insertion APIs
    // =======================

    /// Evict a range of cache positions to make room for insertion
    ///
    /// This clears cache slots in the range [start_pos, end_pos) for the given slot,
    /// marking any affected spans as evicted and returning information about what was
    /// removed. This is typically used before inserting new context mid-conversation.
    ///
    /// # Arguments
    ///
    /// * `slot` - Batch slot index to evict from
    /// * `start_pos` - Starting position (inclusive) to evict
    /// * `end_pos` - Ending position (exclusive) to evict
    ///
    /// # Returns
    ///
    /// EvictionResult containing:
    /// - `spans_affected`: List of spans that were fully or partially evicted
    /// - `positions_evicted`: List of (slot, range) tuples for evicted positions
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Evict positions 50-100 in slot 0 to make room for RAG context
    /// let result = cache_builder.evict_range(0, 50, 100)?;
    /// println!("Evicted {} spans", result.spans_affected.len());
    /// ```
    pub fn evict_range(
        &mut self,
        slot: usize,
        start_pos: usize,
        end_pos: usize,
    ) -> Result<crate::cache::EvictionResult> {
        use crate::cache::{EvictionImpact, EvictionResult, SpanState};

        let mut result = EvictionResult::empty();

        // Find all spans that overlap with this range and collect their info
        let overlapping_spans: Vec<_> = self
            .span_registry
            .spans_overlapping_range(slot, start_pos, end_pos)
            .into_iter()
            .map(|span| (span.id, span.start_pos, span.end_pos))
            .collect();

        for (span_id, span_start, span_end) in overlapping_spans {
            // Calculate overlap
            let overlap_start = start_pos.max(span_start);
            let overlap_end = end_pos.min(span_end);

            if overlap_start >= overlap_end {
                continue; // No actual overlap
            }

            // Get mutable reference to update span state
            if let Some(span_mut) = self.span_registry.get_mut(span_id) {
                if overlap_start == span_start && overlap_end == span_end {
                    // Fully evicted
                    span_mut.state = SpanState::FullyEvicted;
                    result
                        .spans_affected
                        .push((span_id, EvictionImpact::FullyEvicted));
                } else {
                    // Partially evicted - track remaining ranges
                    let mut remaining = Vec::new();
                    if overlap_start > span_start {
                        remaining.push(span_start..overlap_start);
                    }
                    if overlap_end < span_end {
                        remaining.push(overlap_end..span_end);
                    }
                    span_mut.state = SpanState::PartiallyEvicted {
                        remaining_ranges: remaining.clone(),
                    };
                    result
                        .spans_affected
                        .push((span_id, EvictionImpact::PartiallyEvicted { remaining }));
                }

                result
                    .positions_evicted
                    .push((slot, overlap_start..overlap_end));
            }
        }

        // Note: Actual KV tensor zeroing would happen in the model's forward pass
        // This method only tracks the metadata for what should be evicted

        Ok(result)
    }

    /// Insert new context at a specific position, evicting everything after it
    ///
    /// This is the main API for mid-conversation context injection (e.g., RAG retrieval).
    /// It evicts all cache content after the insertion point and returns information
    /// about what was evicted so it can be re-processed.
    ///
    /// # Process
    ///
    /// 1. Evict everything from `insertion_pos` onwards
    /// 2. Caller inserts new context tokens at `insertion_pos`
    /// 3. Caller re-processes evicted content to rebuild KV cache
    ///
    /// # Arguments
    ///
    /// * `slot` - Batch slot index for insertion
    /// * `insertion_pos` - Position where new context will be inserted
    ///
    /// # Returns
    ///
    /// EvictionResult with information about evicted spans and positions.
    /// The caller should use `positions_evicted` to determine what content
    /// needs to be re-processed after insertion.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Insert RAG context at position 50
    /// let evicted = cache_builder.insert_context_at(slot_id, 50)?;
    ///
    /// // Process new context
    /// model.forward(&new_context_tokens, &metadata)?;
    ///
    /// // Re-process evicted content to rebuild cache
    /// if !evicted.positions_evicted.is_empty() {
    ///     let evicted_tokens = get_tokens_for_positions(&evicted);
    ///     model.forward_kv_only(&evicted_tokens, &metadata)?;
    /// }
    /// ```
    pub fn insert_context_at(
        &mut self,
        slot: usize,
        insertion_pos: usize,
    ) -> Result<crate::cache::EvictionResult> {
        // Evict from insertion_pos to current position
        let current_pos = self.positions.get(slot).copied().unwrap_or(0);

        if insertion_pos >= current_pos {
            // Nothing to evict - insertion is at or beyond current position
            return Ok(crate::cache::EvictionResult::empty());
        }

        // Evict everything from insertion_pos onwards
        let result = self.evict_range(slot, insertion_pos, current_pos)?;

        // Reset the position to insertion_pos so new content can be written there
        if let Some(pos) = self.positions.get_mut(slot) {
            *pos = insertion_pos;
        }
        if let Some(idx) = self.indices.get_mut(slot) {
            *idx = insertion_pos % self.context;
        }

        Ok(result)
    }

    /// Helper to reconstruct token sequence after insertion
    ///
    /// When inserting context mid-conversation, you need to re-process evicted content
    /// to rebuild the KV cache. This helper constructs the proper token sequence:
    /// [cached_prefix] + [inserted_context] + [evicted_suffix]
    ///
    /// # Arguments
    ///
    /// * `cached_tokens` - All tokens that were cached before insertion
    /// * `insertion_pos` - Position where new context was inserted
    /// * `inserted_tokens` - New tokens inserted at insertion_pos
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `full_sequence`: Complete token sequence for re-processing
    /// - `reprocess_start`: Index in full_sequence where re-processing should start
    ///   (everything before this is cached and can be skipped)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Original: [A, B, C, D, E] cached
    /// // Insert [X, Y] at position 2
    /// let (seq, start) = ParallelCacheBuilder::reconstruct_after_insertion(
    ///     &[A, B, C, D, E],
    ///     2,
    ///     &[X, Y]
    /// );
    /// // Result: seq = [A, B, X, Y, C, D, E], start = 2
    /// // Only process [X, Y, C, D, E] with KV-only forward pass
    /// ```
    pub fn reconstruct_after_insertion(
        cached_tokens: &[u32],
        insertion_pos: usize,
        inserted_tokens: &[u32],
    ) -> (Vec<u32>, usize) {
        let mut full_sequence = Vec::with_capacity(cached_tokens.len() + inserted_tokens.len());

        // Copy prefix that's still cached (before insertion point)
        full_sequence.extend_from_slice(&cached_tokens[..insertion_pos]);

        // Add new inserted context
        full_sequence.extend_from_slice(inserted_tokens);

        // Add evicted suffix that needs re-processing
        if insertion_pos < cached_tokens.len() {
            full_sequence.extend_from_slice(&cached_tokens[insertion_pos..]);
        }

        // Everything from insertion_pos onwards needs re-processing
        let reprocess_start = insertion_pos;

        (full_sequence, reprocess_start)
    }

    /// Create a new ParallelKvCache with this builder's configuration
    ///
    /// Allocates zero-initialized K/V tensors for all batch slots.
    ///
    /// # Arguments
    ///
    /// * `num_heads` - Number of attention heads (e.g., 32 for Llama)
    /// * `head_dim` - Dimension per head (e.g., hidden_size / num_heads)
    ///
    /// # Returns
    ///
    /// A new KV cache with shape [batch_size, num_heads, context, head_dim]
    pub fn make_cache(&self, num_heads: usize, head_dim: usize) -> Result<ParallelKvCache> {
        let batch_size = self.batch_size();
        let shape = (batch_size, num_heads, self.context, head_dim);
        let k = Tensor::zeros(shape, self.dtype, self.device())?;
        let v = Tensor::zeros(shape, self.dtype, self.device())?;
        Ok(ParallelKvCache {
            k,
            v,
            context: self.context,
        })
    }

    /// Get current positions for all slots
    ///
    /// Returns a slice containing the current token position for each batch slot.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let positions = builder.positions();
    /// // positions might be [5, 10, 3, 0]
    /// // Slot 0: 5 tokens processed
    /// // Slot 1: 10 tokens processed
    /// // Slot 2: 3 tokens processed
    /// // Slot 3: empty/unused
    /// ```
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }

    /// Reset all slots to position 0
    ///
    /// Clears position and index for ALL batch slots. Use sparingly - typically
    /// you want `reset_batch_index(slot)` to reset just one slot.
    pub fn reset(&mut self) {
        self.positions.fill(0);
        self.indices.fill(0);
    }

    /// Get the number of batch slots
    pub fn batch_size(&self) -> usize {
        self.positions.len()
    }

    /// Reset a specific slot to position 0
    ///
    /// Use this when a slot finishes one request and is about to start a new one.
    /// Resets both position and cache index for the slot.
    ///
    /// # Arguments
    ///
    /// * `batch_index` - The slot to reset (0..batch_size)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Request in slot 0 finished, starting new request
    /// builder.reset_batch_index(0);
    /// // Now slot 0 is at position 0, ready for prefill
    /// ```
    pub fn reset_batch_index(&mut self, batch_index: usize) {
        let _old_pos = self.positions[batch_index];
        let _old_idx = self.indices[batch_index];
        self.positions[batch_index] = 0;
        self.indices[batch_index] = 0;

        // Clear from eviction policy tracking
        self.slot_positions.remove(&batch_index);
        if let Some(ref mut h2o) = self.h2o_policy {
            h2o.clear_slot(batch_index);
        }

        // Clear evicted positions for this slot
        if batch_index < self.evicted_positions.len() {
            self.evicted_positions[batch_index].clear();
        }

        crate::debug_cache!(
            "reset_batch_index: Slot {} position {} -> 0, index {} -> 0",
            batch_index,
            _old_pos,
            _old_idx
        );
    }

    /// **NEW**: Set position for a specific slot (O(1) operation)
    ///
    /// Directly sets a slot's position without iteration. This is the EFFICIENT
    /// way to sync positions after chunked prefill with padding.
    ///
    /// **Why we need this**: When we prefill with padding, positions advance by
    /// the padded length (e.g., 64 tokens), but the actual prompt was only 5 tokens.
    /// We need to "rewind" position back to 5.
    ///
    /// **Old approach (O(n))**:
    /// ```ignore
    /// reset_batch_index(slot);  // Position = 0
    /// for _ in 0..5 {
    ///     indices_and_mask(1, ...);  // Advance by 1, 5 times
    /// }
    /// // Position = 5, but took 5 function calls!
    /// ```
    ///
    /// **New approach (O(1))**:
    /// ```ignore
    /// set_position(slot, 5);  // Position = 5, done in one call!
    /// ```
    ///
    /// # Arguments
    ///
    /// * `slot` - The batch slot to update (0..batch_size)
    /// * `position` - The target position (can exceed context)
    ///
    /// # Implementation
    ///
    /// Sets both position and calculates the cache index via modulo:
    /// - position[slot] = position
    /// - indices[slot] = position % context
    ///
    /// This ensures the cache index wraps correctly for long sequences.
    pub fn set_position(&mut self, slot: usize, position: usize) {
        debug_assert!(
            slot < self.positions.len(),
            "Slot index {} >= batch size {}",
            slot,
            self.positions.len()
        );
        // Allow positions up to context * 10 before warning (handles long sequences)
        debug_assert!(
            position < self.context * 10,
            "Position {} unusually large (context size: {}). Possible overflow?",
            position,
            self.context
        );

        if slot < self.positions.len() {
            let _old_pos = self.positions[slot];
            self.positions[slot] = position;

            // Calculate cache index: use StreamingLLM if enabled, else simple wraparound
            self.indices[slot] = if let Some(ref cfg) = self.streaming_config {
                compute_streaming_index(position, cfg, self.context)
            } else {
                position % self.context
            };

            // Update slot_positions mapping for eviction policies
            self.slot_positions.insert(slot, position);

            crate::debug_cache!(
                "set_position: Slot {} position {} -> {}",
                slot,
                _old_pos,
                position
            );
        }
    }

    /// **NEW**: Get current position for a slot
    ///
    /// Returns the current token position for a specific slot.
    ///
    /// # Arguments
    ///
    /// * `slot` - The batch slot to query (0..batch_size)
    ///
    /// # Returns
    ///
    /// Current position, or 0 if slot index is invalid
    pub fn get_position(&self, slot: usize) -> usize {
        debug_assert!(
            slot < self.positions.len(),
            "Slot index {} >= batch size {}",
            slot,
            self.positions.len()
        );
        self.positions.get(slot).copied().unwrap_or(0)
    }

    /// **NEW**: Get current cache index for a slot
    ///
    /// Returns where the NEXT token will be written in the cache for this slot.
    ///
    /// # Arguments
    ///
    /// * `slot` - The batch slot to query (0..batch_size)
    ///
    /// # Returns
    ///
    /// Cache index (0..context), or 0 if slot index is invalid
    pub fn get_cache_index(&self, slot: usize) -> usize {
        debug_assert!(
            slot < self.indices.len(),
            "Slot index {} >= batch size {}",
            slot,
            self.indices.len()
        );
        self.indices.get(slot).copied().unwrap_or(0)
    }

    /// **NEW**: Check if a slot's cache should be cleared
    ///
    /// Returns true if the slot is at position 0, indicating it was recently reset
    /// and might need its KV cache cleared to prevent contamination.
    ///
    /// **INTENDED USE**: Call this after `reset_batch_index()` to determine if
    /// `clear_slot()` should be called on the KV cache.
    ///
    /// **CURRENT STATUS**: Not actually used yet since `clear_slot()` is a stub.
    ///
    /// # Arguments
    ///
    /// * `slot` - The batch slot to check (0..batch_size)
    ///
    /// # Returns
    ///
    /// true if slot is at position 0, false otherwise
    pub fn should_clear_slot(&self, slot: usize) -> bool {
        debug_assert!(
            slot < self.positions.len(),
            "Slot index {} >= batch size {}",
            slot,
            self.positions.len()
        );
        // Return true if this slot has been reset
        self.positions.get(slot).copied().unwrap_or(0) == 0
    }

    /// Create a Slot view for a specific batch slot
    ///
    /// This provides slot-scoped operations that prevent accidentally mixing data
    /// between different requests. The returned `Slot` has exclusive mutable access
    /// to both the cache and builder.
    ///
    /// # Arguments
    ///
    /// * `slot_index` - Which slot to access (0..batch_size)
    /// * `cache` - Mutable reference to the KV cache
    ///
    /// # Returns
    ///
    /// A `Slot` instance providing isolated access to the specified slot
    ///
    /// # Panics
    ///
    /// Panics if slot_index >= batch_size
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut slot0 = builder.get_slot_mut(0, &mut cache)?;
    /// slot0.set_position(5);
    /// slot0.append(&k, &v)?;
    /// ```
    pub fn get_slot_mut<'a>(
        &'a mut self,
        slot_index: usize,
        cache: &'a mut ParallelKvCache,
    ) -> Slot<'a> {
        assert!(
            slot_index < self.batch_size(),
            "Slot index {} out of bounds (batch_size={})",
            slot_index,
            self.batch_size()
        );
        Slot::new(slot_index, cache, self)
    }

    /// **CRITICAL**: Generate indices and attention mask for the next tokens
    ///
    /// This is the heart of the cache builder - it determines WHERE to write
    /// new K/V tensors and HOW attention should work across cached tokens.
    ///
    /// # Purpose
    ///
    /// When processing new tokens (prefill or decode), we need to:
    /// 1. Decide where to write K/V in the cache (which positions)
    /// 2. Create attention masks that control what each token can attend to
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Number of NEW tokens being processed (e.g., 1 for decode, 64 for chunked prefill)
    /// * `batch_mask` - Which slots are active: [true, true, false, false] means slots 0-1 active, 2-3 inactive
    ///
    /// # What it does for EACH active slot
    ///
    /// 1. **Calculate write positions**: For slot with position P, processing seq_len=S tokens:
    ///    - Will write to cache indices [P % context, (P+1) % context, ..., (P+S-1) % context]
    ///    - Example: position=5, seq_len=3, context=512 → indices=[5, 6, 7]
    ///
    /// 2. **Build attention mask**: For each new token, decide what it can attend to:
    ///    - Token i can attend to all PREVIOUS tokens (i=0..i-1) [causal mask]
    ///    - Token i can attend to ALL cached tokens from this slot (position 0..P-1)
    ///    - Token i CANNOT attend to other slots' cached tokens (isolation!)
    ///    - Creates lower-triangular pattern for new tokens + full access to cache
    ///
    /// 3. **Update position**: Advance slot's position by seq_len
    ///    - position[slot] += seq_len
    ///    - indices[slot] = new_position % context
    ///
    /// # Returns
    ///
    /// IndicesAndMask containing:
    /// - `indices`: Tensor of shape [batch_size, seq_len] with cache write positions
    /// - `mask`: Tensor of shape [batch_size, seq_len, total_len] where total_len = cache_pos + seq_len
    ///   - mask[b, i, j] = 1.0 means token i in batch b CAN attend to position j
    ///   - mask[b, i, j] = 0.0 means token i in batch b CANNOT attend to position j
    ///
    /// # **THIS IS WHERE THE BUG LIKELY IS**
    ///
    /// When multiple slots are active simultaneously (parallel batch), the mask
    /// generation needs to ensure slots DON'T see each other's cache. Current
    /// hypothesis: masks might be allowing cross-slot attention.
    ///
    /// # Example - Single slot (WORKING)
    ///
    /// ```ignore
    /// batch_mask = [true, false, false, false]  // Only slot 0 active
    /// position[0] = 5, seq_len = 3
    ///
    /// Indices for slot 0: [5, 6, 7]  // Write to cache positions 5, 6, 7
    /// Mask for slot 0:
    ///   Token 0 (writing to pos 5): can attend to [0,1,2,3,4] (cache) + nothing new
    ///   Token 1 (writing to pos 6): can attend to [0,1,2,3,4] (cache) + [5] (token 0)
    ///   Token 2 (writing to pos 7): can attend to [0,1,2,3,4] (cache) + [5,6] (tokens 0-1)
    ///
    /// Result: Coherent generation ✅
    /// ```
    ///
    /// # Example - Multiple slots (BROKEN)
    ///
    /// ```ignore
    /// batch_mask = [true, true, false, false]  // Slots 0 and 1 active
    /// position[0] = 5, position[1] = 10, seq_len = 3
    ///
    /// Indices for slot 0: [5, 6, 7]
    /// Indices for slot 1: [10, 11, 12]
    ///
    /// EXPECTED behavior:
    ///   Slot 0 token 0: can attend to slot 0's cache [0..5] only
    ///   Slot 1 token 0: can attend to slot 1's cache [0..10] only
    ///
    /// SUSPECTED bug:
    ///   Slot 0 token 0: attending to BOTH slots' caches [0..5] AND [0..10]
    ///   Slot 1 token 0: attending to BOTH slots' caches [0..5] AND [0..10]
    ///
    /// Result: Gibberish generation ❌
    /// ```
    ///
    /// # Debugging focus
    ///
    /// Need to verify:
    /// 1. Is mask properly isolating each slot's view of the cache?
    /// 2. Are indices correctly separating slots in the batch dimension?
    /// 3. Does the attention mechanism respect the batch dimension boundaries?
    #[allow(clippy::needless_range_loop)]
    pub fn indices_and_mask(
        &mut self,
        seq_len: usize,
        batch_mask: &[bool],
    ) -> Result<IndicesAndMask> {
        // mask shape is (b, h, t, k)
        let context = self.context;
        if self.context <= seq_len {
            return self.indices_and_mask_abs(seq_len, batch_mask);
        }
        let mut attention_masks = Vec::with_capacity(self.batch_size());
        let mut cache_indices = Vec::with_capacity(self.batch_size());
        for (batch_i, &batch_mask) in batch_mask.iter().enumerate() {
            if !batch_mask {
                let masks: Vec<Vec<f32>> = vec![vec![0.0; context]; seq_len];
                let indices = vec![self.indices[batch_i] as u32; seq_len];
                attention_masks.push(masks);
                cache_indices.push(indices);
            } else {
                let start_index = self.indices[batch_i];
                let start_pos = self.positions[batch_i];
                // DEBUG output removed
                let mut masks: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
                let mut indices = Vec::with_capacity(seq_len);
                let mut all_pos = vec![usize::MAX; context];
                if start_pos < context {
                    for i in 0..start_pos {
                        all_pos[i] = i;
                    }
                } else {
                    let offset = start_pos - start_index;
                    for i in 0..context {
                        all_pos[i] = if i < start_index {
                            i + offset
                        } else {
                            i + offset - context
                        };
                    }
                }

                // Build per-token indices and masks using a local counter so this
                // function is pure (no side effects). Position/indices state must
                // be advanced explicitly by the caller after the forward pass.
                let mut local_index = self.indices[batch_i];
                for seq_i in 0..seq_len {
                    // Compute cache index: use StreamingLLM if enabled, else simple wraparound
                    let cache_idx = if let Some(ref cfg) = self.streaming_config {
                        compute_streaming_index(seq_i + start_pos, cfg, context)
                    } else {
                        local_index
                    };

                    all_pos[cache_idx] = seq_i + start_pos;
                    indices.push(cache_idx as u32);

                    // Advance local counter (wrap inside context)
                    // Only used when StreamingLLM is disabled
                    if self.streaming_config.is_none() {
                        local_index += 1;
                        if local_index >= self.context {
                            local_index = 0;
                        }
                    }
                }
                // DEBUG output removed

                // Get evicted positions for this slot (if any)
                let empty_set = std::collections::HashSet::new();
                let evicted = if batch_i < self.evicted_positions.len() {
                    &self.evicted_positions[batch_i]
                } else {
                    &empty_set
                };

                for seq_i in 0..seq_len {
                    let my_pos = seq_i + start_pos;
                    let mask = all_pos
                        .iter()
                        .enumerate()
                        .map(|(cache_idx, &pos)| {
                            if pos > my_pos {
                                // Future position: causal mask
                                f32::NEG_INFINITY
                            } else if evicted.contains(&cache_idx) {
                                // Evicted position: mask out
                                f32::NEG_INFINITY
                            } else {
                                0.0
                            }
                        })
                        .collect::<Vec<f32>>();
                    masks.push(mask);
                }

                attention_masks.push(masks);
                cache_indices.push(indices);
            }
        }
        // Flattening the attention mask then using Tensor::from_vec rather using Tensor::new ends
        // up being almost 10x faster with candle 0.9.0. This has been fixed in candle 0.9.1.
        let attention_masks = attention_masks
            .into_iter()
            .flat_map(|m| m.into_iter().flatten())
            .collect::<Vec<f32>>();
        let mask = Tensor::from_vec(attention_masks, ((), 1, seq_len, context), self.device())?
            .to_dtype(self.dtype)?;
        let indices = Tensor::new(cache_indices, self.device())?;
        Ok(IndicesAndMask {
            indices,
            mask,
            active: batch_mask.to_vec(),
        })
    }

    /// Generate indices and masks for variable-length sequences (batched prefill)
    ///
    /// Unlike `indices_and_mask()` which assumes all slots process the same seq_len,
    /// this variant accepts per-slot sequence lengths, enabling proper batched prefill
    /// with variable-length prompts.
    ///
    /// # Arguments
    ///
    /// * `seq_lengths` - Number of tokens to process for each slot (0 = inactive)
    /// * `batch_mask` - Which batch slots are active
    ///
    /// # Returns
    ///
    /// IndicesAndMask with padded indices/masks (all slots have max_seq_len entries)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Batch of 3 requests with different prompt lengths
    /// let seq_lengths = vec![11, 6, 4, 0];  // 4th slot inactive
    /// let batch_mask = vec![true, true, true, false];
    /// let iam = builder.indices_and_mask_variable(&seq_lengths, &batch_mask)?;
    /// // Result: indices padded to max_len=11, shorter sequences padded with last index
    /// ```
    #[allow(clippy::needless_range_loop)]
    pub fn indices_and_mask_variable(
        &mut self,
        seq_lengths: &[usize],
        batch_mask: &[bool],
    ) -> anyhow::Result<IndicesAndMask> {
        if seq_lengths.len() != batch_mask.len() || seq_lengths.len() != self.batch_size() {
            anyhow::bail!(
                "seq_lengths and batch_mask must match batch_size: got {}, {}, expected {}",
                seq_lengths.len(),
                batch_mask.len(),
                self.batch_size()
            );
        }

        // Find the maximum sequence length (for padding)
        let max_seq_len = *seq_lengths.iter().max().unwrap_or(&1);

        if max_seq_len == 0 {
            // All slots inactive, return dummy data
            return Ok(self.indices_and_mask(0, batch_mask)?);
        }

        let context = self.context;
        if self.context <= max_seq_len {
            // Fall back to absolute positioning for very long sequences
            // Note: This may not work perfectly for variable lengths
            return Ok(self.indices_and_mask_abs(max_seq_len, batch_mask)?);
        }

        let mut attention_masks = Vec::with_capacity(self.batch_size());
        let mut cache_indices = Vec::with_capacity(self.batch_size());

        for (batch_i, &is_active) in batch_mask.iter().enumerate() {
            let slot_seq_len = seq_lengths[batch_i];

            if !is_active || slot_seq_len == 0 {
                // Inactive slot: generate dummy data padded to max_seq_len
                let masks: Vec<Vec<f32>> = vec![vec![0.0; context]; max_seq_len];
                let indices = vec![self.indices[batch_i] as u32; max_seq_len];
                attention_masks.push(masks);
                cache_indices.push(indices);
            } else {
                let start_index = self.indices[batch_i];
                let start_pos = self.positions[batch_i];
                let mut masks: Vec<Vec<f32>> = Vec::with_capacity(max_seq_len);
                let mut indices = Vec::with_capacity(max_seq_len);
                let mut all_pos = vec![usize::MAX; context];

                // Set up existing positions in cache
                if start_pos < context {
                    for i in 0..start_pos {
                        all_pos[i] = i;
                    }
                } else {
                    let offset = start_pos - start_index;
                    for i in 0..context {
                        all_pos[i] = if i < start_index {
                            i + offset
                        } else {
                            i + offset - context
                        };
                    }
                }

                // Generate indices for this slot's actual tokens
                let mut local_index = self.indices[batch_i];
                // DEBUG output removed
                for seq_i in 0..slot_seq_len {
                    all_pos[local_index] = seq_i + start_pos;
                    indices.push(local_index as u32);
                    local_index += 1;
                    if local_index >= self.context {
                        local_index = 0;
                    }
                }
                // DEBUG output removed

                // Pad indices to max_seq_len (repeat last valid index)
                let last_index = *indices.last().unwrap();
                while indices.len() < max_seq_len {
                    indices.push(last_index);
                }

                // Generate masks for actual tokens
                for seq_i in 0..slot_seq_len {
                    let my_pos = seq_i + start_pos;
                    let mask = all_pos
                        .iter()
                        .map(|&pos| {
                            if pos <= my_pos {
                                0.0
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                        .collect::<Vec<f32>>();
                    masks.push(mask);
                }

                // Pad masks to max_seq_len (all NEG_INFINITY = attend to nothing)
                let padding_mask = vec![f32::NEG_INFINITY; context];
                while masks.len() < max_seq_len {
                    masks.push(padding_mask.clone());
                }

                attention_masks.push(masks);
                cache_indices.push(indices);
            }
        }

        // Flatten and create tensors
        let attention_masks = attention_masks
            .into_iter()
            .flat_map(|m| m.into_iter().flatten())
            .collect::<Vec<f32>>();
        let mask = Tensor::from_vec(
            attention_masks,
            ((), 1, max_seq_len, context),
            self.device(),
        )?
        .to_dtype(self.dtype)?;
        let indices = Tensor::new(cache_indices, self.device())?;


        Ok(IndicesAndMask {
            indices,
            mask,
            active: batch_mask.to_vec(),
        })
    }

    /// Get the device this cache builder is using
    ///
    /// # Returns
    ///
    /// Reference to the CPU or GPU device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// **ABSOLUTE POSITION VARIANT**: Generate indices and mask using absolute positions
    ///
    /// This is an alternative implementation that creates attention masks based on
    /// absolute positions rather than cache indices. Used in some contexts where
    /// the full position information is needed.
    ///
    /// # Differences from `indices_and_mask()`
    ///
    /// - `indices_and_mask()`: Builds mask based on cached context length
    /// - `indices_and_mask_abs()`: Builds mask based on absolute token positions
    ///
    /// The absolute variant is useful when you need to track where you are in the
    /// full sequence, not just within the cache window.
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Number of new tokens being processed
    /// * `batch_mask` - Which batch slots are active
    ///
    /// # Returns
    ///
    /// IndicesAndMask with absolute position-aware attention masks
    #[allow(clippy::needless_range_loop)]
    fn indices_and_mask_abs(
        &mut self,
        seq_len: usize,
        batch_mask: &[bool],
    ) -> Result<IndicesAndMask> {
        let mask = self.get_mask_abs(seq_len, seq_len)?;
        let mut cache_indices = Vec::with_capacity(self.batch_size());
        for (batch_i, &batch_mask) in batch_mask.iter().enumerate() {
            if !batch_mask {
                let indices = vec![self.indices[batch_i] as u32; seq_len];
                cache_indices.push(indices);
            } else {
                let mut indices = Vec::with_capacity(seq_len);
                // Use a local counter to generate per-token indices without
                // mutating builder state. Caller is responsible for advancing
                // positions/indices after the forward pass.
                let mut local_index = self.indices[batch_i];
                for _ in 0..seq_len {
                    indices.push(local_index as u32);
                    local_index += 1;
                    if local_index >= self.context {
                        local_index = 0;
                    }
                }
                cache_indices.push(indices);
            }
        }
        let indices = Tensor::new(cache_indices, self.device())?;
        Ok(IndicesAndMask {
            indices,
            mask,
            active: batch_mask.to_vec(),
        })
    }

    /// **HELPER**: Generate absolute position attention mask
    ///
    /// Creates a causal attention mask based on absolute positions with context window constraints.
    ///
    /// # How the mask logic works
    ///
    /// For each position pair (i, j) in the mask:
    /// ```ignore
    /// if size1 + j > size2 + i || size1 + j + context < size2 + i {
    ///     NEG_INFINITY  // Cannot attend
    /// } else {
    ///     0.0  // Can attend
    /// }
    /// ```
    ///
    /// This creates:
    /// - Causal masking: Future tokens are masked out
    /// - Context window: Only attend within context-sized window
    ///
    /// # Arguments
    ///
    /// * `size1` - Number of query positions
    /// * `size2` - Number of key/value positions
    ///
    /// # Returns
    ///
    /// Tensor of shape [1, size1, size2] with NEG_INFINITY for masked, 0.0 for unmasked
    fn get_mask_abs(&self, size1: usize, size2: usize) -> Result<Tensor> {
        let context = self.context;
        let mask: Vec<_> = (0..size1)
            .flat_map(|i| {
                (0..size2).map(move |j| {
                    if size1 + j > size2 + i || size1 + j + context < size2 + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (size1, size2), self.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candlelight::core::IndexOp;

    #[test]
    fn test_set_position() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ParallelCacheBuilder::new(4, 512, DType::F32, &device)?;

        // Test setting positions
        cache.set_position(0, 5);
        cache.set_position(1, 6);
        cache.set_position(2, 7);

        assert_eq!(cache.get_position(0), 5);
        assert_eq!(cache.get_position(1), 6);
        assert_eq!(cache.get_position(2), 7);

        // Test that cache index wraps correctly
        cache.set_position(0, 512);
        assert_eq!(cache.get_position(0), 512);
        assert_eq!(cache.get_cache_index(0), 0); // 512 % 512 = 0

        cache.set_position(1, 1024);
        assert_eq!(cache.get_position(1), 1024);
        assert_eq!(cache.get_cache_index(1), 0); // 1024 % 512 = 0

        cache.set_position(2, 517);
        assert_eq!(cache.get_position(2), 517);
        assert_eq!(cache.get_cache_index(2), 5); // 517 % 512 = 5

        Ok(())
    }

    #[test]
    fn test_clear_slot() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ParallelCacheBuilder::new(4, 512, DType::F32, &device)?;

        // Set some positions
        cache.set_position(0, 10);
        cache.set_position(1, 20);

        // Reset a slot
        cache.reset_batch_index(0);

        assert_eq!(cache.get_position(0), 0);
        assert_eq!(cache.get_position(1), 20);
        assert!(cache.should_clear_slot(0));
        assert!(!cache.should_clear_slot(1));

        Ok(())
    }

    #[test]
    fn test_parallel_cache_builder() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = ParallelCacheBuilder::new(2, 5, DType::F32, &device)?;
        let inf = f32::INFINITY;

        let iam = cache.indices_and_mask(1, &[true, false])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[0], [0]]);
        assert_eq!(
            mask,
            [[[0.0, -inf, -inf, -inf, -inf]], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
        );

        // Advance position for slot 0 (caller must advance positions after forward)
        cache.set_position(0, cache.get_position(0) + 1);

        let iam = cache.indices_and_mask(1, &[true, false])?;
        let mask = iam.mask.i((.., 0))?.to_vec3::<f32>()?;
        assert_eq!(iam.indices.to_vec2::<u32>()?, [[1], [0]]);
        assert_eq!(
            mask,
            [[[0.0, 0.0, -inf, -inf, -inf]], [[0.0, 0.0, 0.0, 0.0, 0.0]]]
        );

        Ok(())
    }

    #[test]
    fn test_span_lifecycle() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(2, 512, DType::F32, &device)?;

        // Begin a span (first span has ID 1, not 0)
        let span_id = builder.begin_span(0, CacheTag::SystemPrompt, Some("sys".into()))?;
        assert_eq!(span_id, 1);
        assert!(builder.is_span_active(span_id));

        // Advance position
        builder.set_position(0, 10);

        // End the span
        builder.end_span(span_id)?;

        let span = builder.get_span(span_id).unwrap();
        assert_eq!(span.start_pos, 0);
        assert_eq!(span.end_pos, 10);
        assert_eq!(span.slot, 0);

        Ok(())
    }

    #[test]
    fn test_tag_region() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(2, 512, DType::F32, &device)?;

        let span_id = builder.tag_region(1, 5, 20, CacheTag::ToolOutput, Some("tool1".into()))?;

        let span = builder.get_span(span_id).unwrap();
        assert_eq!(span.start_pos, 5);
        assert_eq!(span.end_pos, 20);
        assert_eq!(span.slot, 1);
        assert_eq!(span.tag, CacheTag::ToolOutput);

        Ok(())
    }

    #[test]
    fn test_span_unique_names() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        let _span1 = builder.tag_region(0, 0, 10, CacheTag::Custom, Some("unique".into()))?;

        // Try to create another with same name
        let result = builder.tag_region(0, 10, 20, CacheTag::Custom, Some("unique".into()));
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_span_parent_child() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        let parent = builder.tag_region(0, 0, 50, CacheTag::ToolOutput, Some("parent".into()))?;
        let child = builder.tag_region(0, 50, 100, CacheTag::Custom, Some("child".into()))?;

        builder.set_span_parent(child, parent)?;

        let child_span = builder.get_span(child).unwrap();
        assert_eq!(child_span.parent, Some(parent));

        let parent_span = builder.get_span(parent).unwrap();
        assert!(parent_span.children.contains(&child));

        Ok(())
    }

    #[test]
    fn test_span_eviction_cascade() -> Result<()> {
        use crate::cache::{CacheTag, EvictionImpact};
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        let parent = builder.tag_region(0, 0, 50, CacheTag::ToolOutput, Some("parent".into()))?;
        let child = builder.tag_region(0, 50, 100, CacheTag::Custom, Some("child".into()))?;
        builder.set_span_parent(child, parent)?;

        // Evict parent should cascade to child
        let result = builder.evict_span(parent)?;

        assert_eq!(result.spans_affected.len(), 2);
        assert!(
            result
                .spans_affected
                .iter()
                .all(|(_, impact)| matches!(impact, EvictionImpact::FullyEvicted))
        );

        assert!(!builder.is_span_active(parent));
        assert!(!builder.is_span_active(child));

        Ok(())
    }

    #[test]
    fn test_evict_tagged() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        let _tool1 = builder.tag_region(0, 0, 10, CacheTag::ToolOutput, Some("tool1".into()))?;
        let _tool2 = builder.tag_region(0, 10, 20, CacheTag::ToolOutput, Some("tool2".into()))?;
        let _user = builder.tag_region(0, 20, 30, CacheTag::UserInput, Some("user1".into()))?;

        let result = builder.evict_tagged(CacheTag::ToolOutput)?;

        assert_eq!(result.spans_affected.len(), 2);
        assert!(builder.is_span_active(_user));
        assert!(!builder.is_span_active(_tool1));
        assert!(!builder.is_span_active(_tool2));

        Ok(())
    }

    #[test]
    fn test_cache_usage_info() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(2, 512, DType::F32, &device)?;

        let _s1 = builder.tag_region(0, 0, 10, CacheTag::SystemPrompt, None)?;
        let _s2 = builder.tag_region(0, 10, 20, CacheTag::UserInput, None)?;
        let _s3 = builder.tag_region(1, 0, 15, CacheTag::ToolOutput, None)?;

        let info = builder.get_cache_usage();

        assert_eq!(info.total_slots, 2);
        assert_eq!(info.active_span_count, 3);
        assert_eq!(info.per_tag_counts.get(&CacheTag::SystemPrompt), Some(&1));
        assert_eq!(info.per_tag_counts.get(&CacheTag::UserInput), Some(&1));
        assert_eq!(info.per_tag_counts.get(&CacheTag::ToolOutput), Some(&1));

        Ok(())
    }

    #[test]
    fn test_span_importance() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        let span = builder.tag_region(0, 0, 10, CacheTag::Custom, None)?;
        builder.set_span_importance(span, 0.8)?;

        let s = builder.get_span(span).unwrap();
        assert_eq!(s.importance, 0.8);

        // Test clamping
        builder.set_span_importance(span, 1.5)?;
        let s = builder.get_span(span).unwrap();
        assert_eq!(s.importance, 1.0);

        Ok(())
    }

    #[test]
    fn test_find_span_by_name() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        let span_id = builder.tag_region(0, 0, 10, CacheTag::Custom, Some("test_span".into()))?;

        let found = builder.find_span_by_name("test_span");
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, span_id);

        let not_found = builder.find_span_by_name("nonexistent");
        assert!(not_found.is_none());

        Ok(())
    }

    #[test]
    fn test_evict_named() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        let span_id = builder.tag_region(0, 0, 10, CacheTag::Custom, Some("deleteme".into()))?;

        assert!(builder.is_span_active(span_id));

        let result = builder.evict_named("deleteme")?;
        assert_eq!(result.spans_affected.len(), 1);
        assert!(!builder.is_span_active(span_id));

        // Try to evict non-existent name
        let result = builder.evict_named("nonexistent");
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_kv_insertion_at_start() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        // Simulate conversation: 100 tokens cached
        builder.set_position(0, 100);
        let span_id =
            builder.tag_region(0, 0, 100, CacheTag::UserInput, Some("original".into()))?;

        // Insert 50 tokens at position 0 (beginning)
        let result = builder.insert_context_at(0, 0)?;

        // Everything should be evicted since insertion is at start
        assert_eq!(result.positions_evicted.len(), 1);
        assert_eq!(result.positions_evicted[0].1, 0..100);

        // Position should reset to 0 for new content
        assert_eq!(builder.get_position(0), 0);

        // Span should be fully evicted
        assert_eq!(result.spans_affected.len(), 1);
        assert_eq!(result.spans_affected[0].0, span_id);
        match &result.spans_affected[0].1 {
            crate::cache::EvictionImpact::FullyEvicted => {}
            _ => panic!("Expected FullyEvicted"),
        }

        Ok(())
    }

    #[test]
    fn test_kv_insertion_at_middle() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        // Simulate conversation: 100 tokens cached
        builder.set_position(0, 100);
        let span_id =
            builder.tag_region(0, 0, 100, CacheTag::UserInput, Some("original".into()))?;

        // Insert at position 50 (middle)
        let result = builder.insert_context_at(0, 50)?;

        // Only positions 50-100 should be evicted
        assert_eq!(result.positions_evicted.len(), 1);
        assert_eq!(result.positions_evicted[0].1, 50..100);

        // Position should reset to 50
        assert_eq!(builder.get_position(0), 50);

        // Span should be partially evicted (0-50 remains)
        assert_eq!(result.spans_affected.len(), 1);
        match &result.spans_affected[0].1 {
            crate::cache::EvictionImpact::PartiallyEvicted { remaining } => {
                assert_eq!(remaining.len(), 1);
                assert_eq!(remaining[0], 0..50);
            }
            _ => panic!("Expected PartiallyEvicted"),
        }

        Ok(())
    }

    #[test]
    fn test_kv_insertion_at_end() -> Result<()> {
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        // Simulate conversation: 100 tokens cached
        builder.set_position(0, 100);

        // Insert at position 100 (end) - nothing to evict
        let result = builder.insert_context_at(0, 100)?;

        assert!(result.positions_evicted.is_empty());
        assert!(result.spans_affected.is_empty());

        // Position should stay at 100
        assert_eq!(builder.get_position(0), 100);

        Ok(())
    }

    #[test]
    fn test_kv_insertion_beyond_end() -> Result<()> {
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        builder.set_position(0, 100);

        // Insert beyond current position - nothing to evict
        let result = builder.insert_context_at(0, 150)?;

        assert!(result.positions_evicted.is_empty());
        assert!(result.spans_affected.is_empty());

        Ok(())
    }

    #[test]
    fn test_kv_insertion_with_multiple_spans() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        // Create multiple overlapping spans
        builder.set_position(0, 200);
        let span1 = builder.tag_region(0, 0, 50, CacheTag::SystemPrompt, Some("span1".into()))?;
        let span2 = builder.tag_region(0, 40, 100, CacheTag::UserInput, Some("span2".into()))?;
        let span3 = builder.tag_region(0, 90, 150, CacheTag::ToolOutput, Some("span3".into()))?;
        let span4 =
            builder.tag_region(0, 140, 200, CacheTag::ModelGeneration, Some("span4".into()))?;

        // Insert at position 75 - should affect span2, span3, span4
        let result = builder.insert_context_at(0, 75)?;

        // Check affected spans
        assert_eq!(result.spans_affected.len(), 3);

        let affected_ids: Vec<_> = result.spans_affected.iter().map(|(id, _)| *id).collect();
        assert!(!affected_ids.contains(&span1)); // Fully before insertion
        assert!(affected_ids.contains(&span2)); // Overlaps
        assert!(affected_ids.contains(&span3)); // Overlaps
        assert!(affected_ids.contains(&span4)); // Fully after

        Ok(())
    }

    #[test]
    fn test_kv_evict_range_partial_span() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        // Create a span covering 0-100
        let span_id = builder.tag_region(0, 0, 100, CacheTag::UserInput, Some("test".into()))?;

        // Evict middle section 30-70
        let result = builder.evict_range(0, 30, 70)?;

        // Span should be partially evicted with two remaining ranges
        assert_eq!(result.spans_affected.len(), 1);
        match &result.spans_affected[0].1 {
            crate::cache::EvictionImpact::PartiallyEvicted { remaining } => {
                assert_eq!(remaining.len(), 2);
                assert_eq!(remaining[0], 0..30);
                assert_eq!(remaining[1], 70..100);
            }
            _ => panic!("Expected PartiallyEvicted with split ranges"),
        }

        // Check span state
        let span = builder.span_registry.get(span_id).unwrap();
        match &span.state {
            crate::cache::SpanState::PartiallyEvicted { remaining_ranges } => {
                assert_eq!(remaining_ranges.len(), 2);
            }
            _ => panic!("Expected PartiallyEvicted state"),
        }

        Ok(())
    }

    #[test]
    fn test_reconstruct_after_insertion() {
        let cached = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let inserted = vec![100, 200, 300];

        // Insert at beginning
        let (seq, start) = ParallelCacheBuilder::reconstruct_after_insertion(&cached, 0, &inserted);
        assert_eq!(seq, vec![100, 200, 300, 10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(start, 0); // Everything needs reprocessing

        // Insert in middle
        let (seq, start) = ParallelCacheBuilder::reconstruct_after_insertion(&cached, 4, &inserted);
        assert_eq!(seq, vec![10, 20, 30, 40, 100, 200, 300, 50, 60, 70, 80]);
        assert_eq!(start, 4); // Only from insertion onwards

        // Insert at end
        let (seq, start) = ParallelCacheBuilder::reconstruct_after_insertion(&cached, 8, &inserted);
        assert_eq!(seq, vec![10, 20, 30, 40, 50, 60, 70, 80, 100, 200, 300]);
        assert_eq!(start, 8); // Only new tokens
    }

    #[test]
    fn test_kv_insertion_span_importance_preserved() -> Result<()> {
        use crate::cache::CacheTag;
        let device = Device::Cpu;
        let mut builder = ParallelCacheBuilder::new(1, 512, DType::F32, &device)?;

        builder.set_position(0, 100);
        let span_id =
            builder.tag_region(0, 20, 80, CacheTag::SystemPrompt, Some("important".into()))?;

        // Set high importance
        if let Some(span) = builder.span_registry.get_mut(span_id) {
            span.importance = 0.95;
        }

        // Insert at position 50 - partially evicts span
        let result = builder.insert_context_at(0, 50)?;

        // Importance should be preserved
        let span = builder.span_registry.get(span_id).unwrap();
        assert_eq!(span.importance, 0.95);

        // But state should reflect partial eviction
        match &span.state {
            crate::cache::SpanState::PartiallyEvicted { remaining_ranges } => {
                assert_eq!(remaining_ranges[0], 20..50);
            }
            _ => panic!("Expected PartiallyEvicted"),
        }

        Ok(())
    }
}
