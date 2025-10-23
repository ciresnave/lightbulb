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

use candle_core::{DType, Device, IndexOp, Result, Tensor};

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
        // Debug: log incoming IAM indices and builder positions for tracing
        if iam.indices.dims().len() <= 2 {
            if let Ok(idx_vec) = iam.indices.to_vec2::<u32>() {
                eprintln!(
                    "DEBUG ParallelKvCache::append: iam.indices.shape={:?} iam.indices={:?} iam.active={:?} positions={:?}",
                    iam.indices.dims(),
                    idx_vec,
                    iam.active,
                    // We can't access the outer BatchExecutor here; log builder positions indirectly
                    // by creating a temporary vector of zeroes as placeholder if needed
                    "<positions_hidden>"
                );
            } else {
                eprintln!(
                    "DEBUG ParallelKvCache::append: iam.indices.shape={:?} iam.active={:?} positions={:?}",
                    iam.indices.dims(),
                    iam.active,
                    "<positions_hidden>"
                );
            }
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
#[derive(Debug, Clone)]
pub struct ParallelCacheBuilder {
    context: usize,
    /// The current position in the stream for each slot (can exceed context)
    positions: Vec<usize>,
    /// The cache index where the next element will be stored for each slot
    indices: Vec<usize>,
    dtype: DType,
    device: Device,
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
        })
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
        let old_pos = self.positions[batch_index];
        let old_idx = self.indices[batch_index];
        self.positions[batch_index] = 0;
        self.indices[batch_index] = 0;
        crate::debug_cache!(
            "reset_batch_index: Slot {} position {} -> 0, index {} -> 0",
            batch_index,
            old_pos,
            old_idx
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
        if slot < self.positions.len() {
            let old_pos = self.positions[slot];
            self.positions[slot] = position;
            // Calculate cache index from position (wraps around context)
            self.indices[slot] = position % self.context;
            crate::debug_cache!(
                "set_position: Slot {} position {} -> {}",
                slot,
                old_pos,
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
                eprintln!(
                    "DEBUG indices_and_mask: slot={} start_pos={} start_index={} seq_len={} (decode path)",
                    batch_i, start_pos, start_index, seq_len
                );
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
                    all_pos[local_index] = seq_i + start_pos;
                    indices.push(local_index as u32);
                    // advance local counter (wrap inside context)
                    local_index += 1;
                    if local_index >= self.context {
                        local_index = 0;
                    }
                }
                eprintln!(
                    "DEBUG indices_and_mask: slot={} generated indices={:?}",
                    batch_i, indices
                );

                for seq_i in 0..seq_len {
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
                eprintln!(
                    "DEBUG indices_and_mask_variable: slot={} start_pos={} start_index={} slot_seq_len={} generating indices...",
                    batch_i, start_pos, start_index, slot_seq_len
                );
                for seq_i in 0..slot_seq_len {
                    all_pos[local_index] = seq_i + start_pos;
                    indices.push(local_index as u32);
                    local_index += 1;
                    if local_index >= self.context {
                        local_index = 0;
                    }
                }
                eprintln!(
                    "DEBUG indices_and_mask_variable: slot={} generated indices (before padding)={:?}",
                    batch_i, indices
                );

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

        // Debug: log final IAM before returning
        if let Ok(idx_vec) = indices.to_vec2::<u32>() {
            eprintln!(
                "DEBUG indices_and_mask_variable: final indices tensor shape={:?} data={:?}",
                indices.dims(),
                idx_vec
            );
        }
        eprintln!(
            "DEBUG indices_and_mask_variable: active_slots={:?} positions_before={:?}",
            batch_mask, self.positions
        );

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
    use candle_core::IndexOp;

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
}
