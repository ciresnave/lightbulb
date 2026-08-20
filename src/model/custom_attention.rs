//! Custom Batched Attention Layer for True Batched Inference
//!
//! This module implements a batched attention mechanism that can process
//! multiple sequences in parallel, achieving significant speedup over
//! sequential processing.
//!
//! # Architecture
//!
//! Structured like any batched-attention implementation, adapted for our
//! ScatteredKvCache infrastructure. The steps below are the standard
//! transformer sequence, not code taken from anywhere:
//!
//! 1. **Q/K/V Projections** - Linear transformations batched across sequences
//! 2. **RoPE** - Rotary Position Embeddings applied to Q/K
//! 3. **Batched Attention** - Compute attention over batched K/V from cache
//! 4. **Output Projection** - Transform back to hidden dimension
//!
//! # Key Difference from Standard Candle
//!
//! ```rust,ignore
//! // Standard Candle (Sequential):
//! for seq in batch {
//!     let logits = model.forward(&token, pos, &mut cache)?;  // One at a time
//! }
//!
//! // Our Batched Approach:
//! let logits_batch = batched_attention.forward(
//!     &tokens,       // [batch_size, seq_len, hidden]
//!     &positions,    // [batch_size, seq_len]
//!     &kv_cache,     // ScatteredKvCache (shared)
//!     &metadata,     // BatchMetadata
//! )?;  // Returns [batch_size, hidden]
//! ```

use crate::engine::{ParallelCacheBuilder, ParallelKvCache};
use crate::model::BatchMetadata;
use crate::model::quantizable_linear::QuantizableLinear;
use candlelight::core::{DType, Device, IndexOp, Module, Result, Tensor};
use candlelight::nn::VarBuilder;
use std::io::{Read, Seek};

/// FlashAttention-2 wrapper with feature gating
///
/// When the flash-attn feature is enabled and CUDA is available, this provides
/// highly optimized attention computation using FlashAttention-2 kernels.
#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candlelight::flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    candlelight::bail!("compile with '--features flash-attn'")
}

/// Batched Multi-Head Attention Layer
///
/// Processes multiple sequences in parallel using shared KV cache.
#[derive(Debug)]
pub struct BatchedAttention {
    // Linear projections
    q_proj: QuantizableLinear,
    k_proj: QuantizableLinear,
    v_proj: QuantizableLinear,
    o_proj: QuantizableLinear,

    // Model configuration
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,

    // Attention scale factor
    scale: f64,

    // FlashAttention-2 flag
    use_flash_attn: bool,

    // Device
    device: Device,

    // Whether to capture attention weights for eviction policies
    capture_attention: bool,
}

impl BatchedAttention {
    /// Create a new batched attention layer
    ///
    /// # Arguments
    /// * `hidden_size` - Model hidden dimension
    /// * `num_heads` - Number of attention heads
    /// * `num_kv_heads` - Number of key/value heads (for GQA)
    /// * `vb` - Variable builder for loading weights
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Load projection layers (no bias for Llama)
        let q_proj = QuantizableLinear::from_linear(candlelight::nn::linear_no_bias(
            hidden_size,
            num_heads * head_dim,
            vb.pp("q_proj"),
        )?);
        let k_proj = QuantizableLinear::from_linear(candlelight::nn::linear_no_bias(
            hidden_size,
            num_kv_heads * head_dim,
            vb.pp("k_proj"),
        )?);
        let v_proj = QuantizableLinear::from_linear(candlelight::nn::linear_no_bias(
            hidden_size,
            num_kv_heads * head_dim,
            vb.pp("v_proj"),
        )?);
        let o_proj = QuantizableLinear::from_linear(candlelight::nn::linear_no_bias(
            num_heads * head_dim,
            hidden_size,
            vb.pp("o_proj"),
        )?);

        let device = vb.device().clone();

        // Enable FlashAttention on CUDA with float16/bfloat16 when feature is available
        let use_flash_attn = cfg!(feature = "flash-attn") && device.is_cuda();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            use_flash_attn,
            device,
            capture_attention: false,
        })
    }

    /// Create a new BatchedAttention from GGUF quantized weights
    ///
    /// # Arguments
    /// * `hidden_size` - Model hidden dimension
    /// * `num_heads` - Number of attention heads
    /// * `num_kv_heads` - Number of key/value heads (for GQA)
    /// * `gguf_content` - GGUF file content with metadata and tensor info
    /// * `file` - Open file handle for reading tensor data
    /// * `device` - Device to load tensors on
    /// * `layer_idx` - Layer index for tensor naming (e.g., blk.0, blk.1, ...)
    /// * `name_mapper` - Tensor name mapper for architecture-agnostic loading
    pub fn from_gguf<R: Read + Seek>(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        gguf_content: &crate::gguf::Content,
        file: &mut R,
        device: &Device,
        layer_idx: usize,
        name_mapper: &crate::pruning::name_mapping::TensorNameMapper,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Map abstract names to concrete architecture-specific names
        let q_name = name_mapper
            .map_name(&format!("layer_{}.attention.query", layer_idx))
            .unwrap_or_else(|| format!("blk.{}.attn_q.weight", layer_idx));
        let k_name = name_mapper
            .map_name(&format!("layer_{}.attention.key", layer_idx))
            .unwrap_or_else(|| format!("blk.{}.attn_k.weight", layer_idx));
        let v_name = name_mapper
            .map_name(&format!("layer_{}.attention.value", layer_idx))
            .unwrap_or_else(|| format!("blk.{}.attn_v.weight", layer_idx));
        let o_name = name_mapper
            .map_name(&format!("layer_{}.attention.output", layer_idx))
            .unwrap_or_else(|| format!("blk.{}.attn_output.weight", layer_idx));

        // Load quantized projection tensors
        let q_tensor = gguf_content.tensor(file, &q_name, device)?;
        let k_tensor = gguf_content.tensor(file, &k_name, device)?;
        let v_tensor = gguf_content.tensor(file, &v_name, device)?;
        let o_tensor = gguf_content.tensor(file, &o_name, device)?;

        // Convert QTensor to QMatMul and wrap in QuantizableLinear
        let q_proj = QuantizableLinear::from_qmatmul(
            candlelight::core::quantized::QMatMul::from_qtensor(q_tensor)?,
        );
        let k_proj = QuantizableLinear::from_qmatmul(
            candlelight::core::quantized::QMatMul::from_qtensor(k_tensor)?,
        );
        let v_proj = QuantizableLinear::from_qmatmul(
            candlelight::core::quantized::QMatMul::from_qtensor(v_tensor)?,
        );
        let o_proj = QuantizableLinear::from_qmatmul(
            candlelight::core::quantized::QMatMul::from_qtensor(o_tensor)?,
        );

        // Enable FlashAttention on CUDA with float16/bfloat16 when feature is available
        let use_flash_attn = cfg!(feature = "flash-attn") && device.is_cuda();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
            use_flash_attn,
            device: device.clone(),
            capture_attention: false,
        })
    }

    /// Enable or disable attention weight capture
    ///
    /// When enabled, attention weights are returned from forward() for use
    /// by eviction policies like H2O. This adds memory overhead, so only
    /// enable when needed.
    pub fn set_capture_attention(&mut self, capture: bool) {
        self.capture_attention = capture;
    }

    /// Forward pass with batched sequences
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch_size, seq_len, hidden_size]
    /// * `index_pos` - Starting position for RoPE
    /// * `cos` - Pre-computed cosine tensor for RoPE
    /// * `sin` - Pre-computed sine tensor for RoPE
    /// * `batch_executor` - Contains ScatteredKvCache for K/V storage
    /// * `metadata` - Batch structure information
    /// * `layer_idx` - Which transformer layer (0..num_layers)
    ///
    /// # Returns
    /// Tuple of (output_tensor, optional_attention_weights)
    /// - output_tensor: [batch_size, seq_len, hidden_size]
    /// - attention_weights: [batch_size, num_heads, seq_q, seq_k] if capture_attention=true
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        index_pos: usize,
        cos: &Tensor,
        sin: &Tensor,
        cache_builder: &mut ParallelCacheBuilder,
        cache: &mut ParallelKvCache,
        metadata: &BatchMetadata,
        layer_idx: usize,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // === Step 1: Q/K/V Projections ===
        // Project input to query, key, value spaces
        let query_states = self.q_proj.forward(hidden_states)?;
        let key_states = self.k_proj.forward(hidden_states)?;
        let value_states = self.v_proj.forward(hidden_states)?;

        // === Step 2: Reshape for Multi-Head Attention ===
        // [batch_size, seq_len, hidden] -> [batch_size, num_heads, seq_len, head_dim]
        let q = query_states
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)? // [batch, heads, seq, dim]
            .contiguous()?;

        let k = key_states
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let v = value_states
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Detect whether this is a batched prefill early so we can decide how to apply RoPE.
        let is_batched_prefill = metadata.is_prefill
            && metadata.request_ids.len() > 1
            && metadata.sequence_lengths().iter().any(|&len| len > 1);

        // === Step 3: Apply RoPE (Rotary Position Embeddings) ===
        // Apply RoPE. For batched prefill we must apply RoPE per-sequence (positions start at 0
        // for each prompt). Therefore skip the global RoPE here when doing batched prefill and
        // apply it later to each extracted q/k slice.
        let (q, k) = if is_batched_prefill {
            (q, k)
        } else {
            (
                self.apply_rotary_emb(&q, index_pos, seq_len, cos, sin)?,
                self.apply_rotary_emb(&k, index_pos, seq_len, cos, sin)?,
            )
        };

        // === Step 3.5: Restructure for Batched Prefill ===
        // During batched prefill with concatenated sequences, we need to split and pad
        // Q/K/V so they all have uniform shapes for proper batched processing.

        let (q_batched, k_batched, v_batched, iam_opt) = if is_batched_prefill {
            crate::debug_cache!(
                "[BATCHED PREFILL] Restructuring {} concatenated sequences with lengths {:?}",
                metadata.request_ids.len(),
                metadata.sequence_lengths()
            );

            // Build batch_mask and seq_lengths for variable-length batched prefill
            // MUST match slot pool size for continuous batching
            let slot_pool_size = cache_builder.batch_size();
            let _num_sequences = metadata.request_ids.len();

            // Pad to slot pool size
            let mut batch_mask = vec![false; slot_pool_size];
            let mut seq_lengths = metadata.sequence_lengths();

            // Mark active slots based on request_ids
            for &req_id in &metadata.request_ids {
                if req_id < slot_pool_size {
                    batch_mask[req_id] = true;
                }
            }

            // Pad seq_lengths to match slot pool size
            seq_lengths.resize(slot_pool_size, 0);

            // Get variable-length indices and mask
            let iam = cache_builder
                .indices_and_mask_variable(&seq_lengths, &batch_mask)
                .map_err(|e| {
                    candlelight::core::Error::Msg(format!("Failed to get variable indices: {}", e))
                })?;

            // Split and pad Q/K/V to create uniform batched tensors
            // Current: [1, num_heads, concatenated_seq_len, head_dim]
            // Target: [num_requests, num_heads, max_seq_len, head_dim]
            let max_seq_len = *metadata.sequence_lengths().iter().max().unwrap();
            let num_requests = metadata.request_ids.len();

            let (_, num_heads_q, _, head_dim_q) = q.dims4()?;
            let (_, num_heads_k, _, head_dim_k) = k.dims4()?;

            crate::debug_cache!(
                "[BATCHED PREFILL] Splitting concatenated sequences (total_len={}) into {} padded batches (max_len={})",
                seq_len,
                num_requests,
                max_seq_len
            );

            let mut q_slices = Vec::new();
            let mut k_slices = Vec::new();
            let mut v_slices = Vec::new();

            for idx in 0..num_requests {
                // Use SequenceInfo to get the correct positions in the padded tensor
                let seq_info = &metadata.sequences[idx];
                let seq_start = seq_info.start_pos;
                let actual_len = seq_info.actual_length;

                // Extract each sequence from the padded tensor
                let mut q_seq = q.narrow(2, seq_start, actual_len)?;
                let mut k_seq = k.narrow(2, seq_start, actual_len)?;
                let v_seq = v.narrow(2, seq_start, actual_len)?;

                // For batched prefill, apply RoPE per-sequence so positions start at 0 for each
                // prompt. We apply RoPE before padding so actual_len is used.
                if is_batched_prefill {
                    q_seq = self.apply_rotary_emb(&q_seq, 0, actual_len, cos, sin)?;
                    k_seq = self.apply_rotary_emb(&k_seq, 0, actual_len, cos, sin)?;
                }

                // Pad to max_seq_len
                let q_padded = if actual_len < max_seq_len {
                    let padding = Tensor::zeros(
                        (1, num_heads_q, max_seq_len - actual_len, head_dim_q),
                        q_seq.dtype(),
                        q_seq.device(),
                    )?;
                    Tensor::cat(&[&q_seq, &padding], 2)?
                } else {
                    q_seq
                };

                let k_padded = if actual_len < max_seq_len {
                    let padding = Tensor::zeros(
                        (1, num_heads_k, max_seq_len - actual_len, head_dim_k),
                        k_seq.dtype(),
                        k_seq.device(),
                    )?;
                    Tensor::cat(&[&k_seq, &padding], 2)?
                } else {
                    k_seq
                };

                let v_padded = if actual_len < max_seq_len {
                    let padding = Tensor::zeros(
                        (1, num_heads_k, max_seq_len - actual_len, head_dim_k),
                        v_seq.dtype(),
                        v_seq.device(),
                    )?;
                    Tensor::cat(&[&v_seq, &padding], 2)?
                } else {
                    v_seq
                };

                q_slices.push(q_padded);
                k_slices.push(k_padded);
                v_slices.push(v_padded);
            }

            // Stack into batched tensors
            let q_batch = Tensor::cat(&q_slices, 0)?;
            let k_batch = Tensor::cat(&k_slices, 0)?;
            let v_batch = Tensor::cat(&v_slices, 0)?;

            crate::debug_cache!(
                "[BATCHED PREFILL] Batched shapes: Q={:?}, K={:?}, V={:?}",
                q_batch.dims(),
                k_batch.dims(),
                v_batch.dims()
            );

            (q_batch, k_batch, v_batch, Some(iam))
        } else {
            // Single request or decode: use tensors as-is
            (q.clone(), k.clone(), v.clone(), None)
        };

        // === Step 4: Update KV Cache ===
        // Note: K/V are stored with num_kv_heads in the cache
        // GQA head expansion happens later during attention computation

        let (k_full, v_full) = if is_batched_prefill {
            // Use the already-split k_batched and v_batched from Step 3.5
            // iam_opt contains the indices and mask generated earlier
            let iam = iam_opt.as_ref().unwrap();

            // Wrap K/V in KVTensor to expand to max_batch_size for cache
            let k_wrapped =
                crate::model::KVTensor::from_compact(&k_batched, cache_builder.batch_size())
                    .map_err(|e| {
                        candlelight::core::Error::Msg(format!("Failed to wrap batched K: {}", e))
                    })?;
            let v_wrapped =
                crate::model::KVTensor::from_compact(&v_batched, cache_builder.batch_size())
                    .map_err(|e| {
                        candlelight::core::Error::Msg(format!("Failed to wrap batched V: {}", e))
                    })?;

            // Append to cache
            let (k_cache, v_cache) = cache
                .append(k_wrapped.as_full(), v_wrapped.as_full(), iam)
                .map_err(|e| {
                    candlelight::core::Error::Msg(format!("Failed to append batched KV: {}", e))
                })?;

            (k_cache, v_cache)
        } else {
            // Single request or decode: use original unified path
            // Build batch_mask from request_ids
            let max_batch_size = cache_builder.batch_size();
            let mut batch_mask = vec![false; max_batch_size];
            for &req_id in &metadata.request_ids {
                if req_id < max_batch_size {
                    batch_mask[req_id] = true;
                }
            }

            let iam = cache_builder
                .indices_and_mask(seq_len, &batch_mask)
                .map_err(|e| {
                    candlelight::core::Error::Msg(format!("Failed to get indices and mask: {}", e))
                })?;

            // Wrap K/V in KVTensor to expand batch dimension to max_batch_size
            let k_wrapped =
                crate::model::KVTensor::from_compact(&k_batched, cache_builder.batch_size())
                    .map_err(|e| {
                        candlelight::core::Error::Msg(format!("Failed to wrap K tensor: {}", e))
                    })?;
            let v_wrapped =
                crate::model::KVTensor::from_compact(&v_batched, cache_builder.batch_size())
                    .map_err(|e| {
                        candlelight::core::Error::Msg(format!("Failed to wrap V tensor: {}", e))
                    })?;

            if layer_idx == 0 {
                // DEBUG output removed
            }

            let (k_full, v_full) = cache
                .append(k_wrapped.as_full(), v_wrapped.as_full(), &iam)
                .map_err(|e| {
                    candlelight::core::Error::Msg(format!("Failed to append KV: {}", e))
                })?;

            if layer_idx == 0 {
                // DEBUG output removed
            }

            (k_full, v_full)
        };

        // Store the attention mask (from iam_opt if batched prefill, else None for now)
        // Mask shape: [max_batch_size, 1, seq_q, full_cache_size]
        let attention_mask = iam_opt.as_ref().map(|iam| iam.mask.clone());

        // === Step 4.3: Narrow KV cache to valid sequence length ===
        // The cache returns the full buffer (e.g., 512 positions), but we only want
        // the valid portion that's been filled so far
        // For single-sequence case: valid_len = context_len + current_seq_len
        // For multi-sequence: each sequence may have different valid lengths
        let max_valid_len = metadata
            .context_lens
            .iter()
            .zip(metadata.sequence_lengths().iter())
            .map(|(ctx, seq)| ctx + seq)
            .max()
            .unwrap_or(1);

        let k_full = k_full.narrow(2, 0, max_valid_len)?;
        let v_full = v_full.narrow(2, 0, max_valid_len)?;

        // Also narrow the attention mask to match the narrowed K/V dimensions
        // Mask shape: [max_batch_size, 1, seq_q, full_cache_size]
        // Need to narrow dimension 3 (K dimension) to max_valid_len
        let attention_mask = attention_mask
            .map(|mask| {
                mask.narrow(3, 0, max_valid_len).map_err(|e| {
                    candlelight::core::Error::Msg(format!("Failed to narrow attention mask: {}", e))
                })
            })
            .transpose()?;

        if layer_idx == 0 {
            // DEBUG output removed
            if let Some(ref _mask) = attention_mask {
                // DEBUG output removed
            }
        }

        // === Step 4.5: Expand KV heads for GQA ===
        // If using Grouped Query Attention, repeat KV heads to match Q heads
        // This needs to be done AFTER narrowing the cache
        let (k_full, v_full) = if self.num_heads != self.num_kv_heads {
            let repeat_factor = self.num_heads / self.num_kv_heads;
            // Get dimensions AFTER narrowing
            let (_batch, _num_kv_heads, total_seq_len, _head_dim) = k_full.dims4()?;

            if layer_idx == 0 {
                // DEBUG output removed
            }

            // K/V shape from cache: [max_batch_size, num_kv_heads, total_seq, dim]
            // Target: [max_batch_size, num_heads, total_seq, dim]
            // Note: We must use the actual batch dimension from k_full (max_batch_size)
            // not the parameter batch_size (actual batch size)
            let cache_batch_size = k_full.dim(0)?;

            let k_expanded = k_full
                .unsqueeze(2)? // [max_batch, num_kv_heads, 1, seq, dim]
                .expand(&[
                    cache_batch_size,
                    self.num_kv_heads,
                    repeat_factor,
                    total_seq_len,
                    self.head_dim,
                ])?
                .reshape((
                    cache_batch_size,
                    self.num_heads,
                    total_seq_len,
                    self.head_dim,
                ))?;

            let v_expanded = v_full
                .unsqueeze(2)?
                .expand(&[
                    cache_batch_size,
                    self.num_kv_heads,
                    repeat_factor,
                    total_seq_len,
                    self.head_dim,
                ])?
                .reshape((
                    cache_batch_size,
                    self.num_heads,
                    total_seq_len,
                    self.head_dim,
                ))?;

            (k_expanded, v_expanded)
        } else {
            (k_full, v_full)
        };

        // === Step 5: Compute Attention ===
        // Use full K/V history (including current) for attention
        // k_full, v_full shape: [max_batch_size, num_heads, total_seq_len, head_dim]
        // where total_seq_len = historical_tokens + seq_len
        //
        // Need to narrow K/V to actual batch size for attention computation
        // Q has shape [batch_size, ...] but K/V have [max_batch_size, ...]
        // NOTE: Use q_batched.dim(0) instead of batch_size because in batched prefill,
        // q_batched has been reshaped to [num_requests, ...] but batch_size still reflects
        // the original input shape [1, concatenated_seq, ...]
        let actual_batch_size = q_batched.dim(0)?;

        if layer_idx == 0 {
            // DEBUG output removed
        }

        let k_for_attn = k_full.narrow(0, 0, actual_batch_size)?;
        let v_for_attn = v_full.narrow(0, 0, actual_batch_size)?;

        // Pass attention mask for batched prefill to handle padding
        // Need to narrow mask to actual_batch_size to match narrowed K/V
        let mask_for_attn = attention_mask
            .as_ref()
            .map(|mask| mask.narrow(0, 0, actual_batch_size))
            .transpose()?;

        let (attn_output, captured_weights) =
            self.compute_attention(&q_batched, &k_for_attn, &v_for_attn, mask_for_attn.as_ref())?;

        // Extract attention weights if requested (second element of tuple)
        let attn_weights = captured_weights;

        // === Step 6: Reshape and Output Projection ===
        // IMPORTANT: Must match input shape for residual connection!
        // Input shape: [batch_size, seq_len, hidden]  (e.g., [1, 1536, 576] for batched prefill)
        // For batched prefill, attn_output has shape: [num_requests, heads, max_seq_len, dim]
        // We need to return shape: [batch_size, seq_len, hidden] to match input
        let attn_output_for_proj = if is_batched_prefill {
            // We have [num_requests, heads, max_seq, dim]
            // Transpose to [num_requests, max_seq, heads, dim]
            let (num_reqs, num_h, max_seq, head_d) = attn_output.dims4()?;
            let transposed = attn_output.transpose(1, 2)?;

            // Reshape to [1, num_reqs * max_seq, heads * dim]
            let reshaped = transposed.reshape((1, num_reqs * max_seq, num_h * head_d))?;

            // Pad to match original seq_len if needed
            let current_seq = num_reqs * max_seq;
            if current_seq < seq_len {
                let padding = Tensor::zeros(
                    (1, seq_len - current_seq, num_h * head_d),
                    reshaped.dtype(),
                    reshaped.device(),
                )?;
                Tensor::cat(&[&reshaped, &padding], 1)?
            } else {
                reshaped
            }
        } else {
            // [batch, heads, seq, dim] -> [batch, seq, heads * dim]
            let (b, h, s, d) = attn_output.dims4()?;
            attn_output
                .transpose(1, 2)? // [batch, seq, heads, dim]
                .reshape((b, s, h * d))?
        };

        // Final output projection
        let output = self.o_proj.forward(&attn_output_for_proj)?;

        Ok((output, attn_weights))
    }

    /// Apply Rotary Position Embeddings using Candle's built-in implementation
    ///
    /// # Arguments
    /// * `x` - Tensor to apply RoPE to [batch, heads, seq, dim]
    /// * `index_pos` - Starting position in the sequence
    /// * `seq_len` - Length of the sequence
    /// * `cos` - Pre-computed cosine tensor
    /// * `sin` - Pre-computed sine tensor
    ///
    /// # Returns
    /// Tensor with RoPE applied [batch, heads, seq, dim]
    fn apply_rotary_emb(
        &self,
        x: &Tensor,
        index_pos: usize,
        seq_len: usize,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        // Extract cos and sin for the current position range
        let cos_slice = cos.narrow(0, index_pos, seq_len)?;
        let sin_slice = sin.narrow(0, index_pos, seq_len)?;

        // Input x is [batch, num_heads, seq, head_dim]
        // cos/sin are [seq, head_dim/2] (Llama-style RoPE)
        // We need to broadcast cos/sin to [batch, num_heads, seq, head_dim/2] for each half

        let shape = x.dims();
        let (batch, num_heads, seq, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
        let half_dim = head_dim / 2;

        // Broadcast cos/sin from [seq, head_dim/2] to [1, 1, seq, head_dim/2]
        // Then it will auto-broadcast to [batch, num_heads, seq, head_dim/2]
        let cos_slice = cos_slice.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, head_dim/2]
        let sin_slice = sin_slice.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, head_dim/2]

        // Explicitly broadcast to [batch, num_heads, seq, head_dim/2]
        let target_shape = vec![batch, num_heads, seq, half_dim];
        let cos_slice = cos_slice.broadcast_as(&target_shape[..])?;
        let sin_slice = sin_slice.broadcast_as(&target_shape[..])?;

        // Split x into two halves along head_dim
        let x1 = x.narrow(3, 0, half_dim)?;
        let x2 = x.narrow(3, half_dim, half_dim)?;

        // Apply RoPE formula:
        // x_out = [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        let out1 = ((&x1 * &cos_slice)? - (&x2 * &sin_slice)?)?;
        let out2 = ((&x1 * &sin_slice)? + (&x2 * &cos_slice)?)?;

        // Concatenate back
        Tensor::cat(&[out1, out2], 3)
    }
    /// Compute batched attention scores and apply to values
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, num_heads, seq_q, head_dim]
    /// * `k` - Key tensor [batch, num_kv_heads, seq_k, head_dim]
    /// * `v` - Value tensor [batch, num_kv_heads, seq_k, head_dim]
    /// * `mask` - Optional attention mask [batch, 1, seq_q, seq_k] where NEG_INFINITY blocks attention
    ///
    /// # Returns
    /// Tuple of (attention_output, optional_attention_weights)
    /// - attention_output: [batch, num_heads, seq_q, head_dim]
    /// - attention_weights: [batch, num_heads, seq_q, seq_k] if capture_attention=true
    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Get dimensions
        let (_batch, _num_heads, seq_q, _head_dim) = q.dims4()?;
        let (_batch_k, num_heads_k, seq_k, _head_dim_k) = k.dims4()?;

        // === FlashAttention-2 Path ===
        // Use FlashAttention when:
        // 1. Feature is enabled and we're on CUDA
        // 2. No complex attention masks (FlashAttention handles causal internally)
        // 3. K/V have same number of heads as Q (GQA already expanded)
        let use_flash = self.use_flash_attn
            && mask.is_none()
            && num_heads_k == self.num_heads
            && self.device.is_cuda();

        if use_flash {
            // FlashAttention expects (batch, seq_len, num_heads, head_dim)
            // Our tensors are (batch, num_heads, seq_len, head_dim)
            let q_flash = q.transpose(1, 2)?; // [batch, seq_q, num_heads, head_dim]
            let k_flash = k.transpose(1, 2)?; // [batch, seq_k, num_heads, head_dim]
            let v_flash = v.transpose(1, 2)?; // [batch, seq_k, num_heads, head_dim]

            // FlashAttention requires F16 or BF16
            let original_dtype = q.dtype();
            let flash_dtype = if self.device.is_cuda() {
                DType::F16
            } else {
                original_dtype
            };

            let q_flash = q_flash.to_dtype(flash_dtype)?;
            let k_flash = k_flash.to_dtype(flash_dtype)?;
            let v_flash = v_flash.to_dtype(flash_dtype)?;

            let softmax_scale = self.scale as f32;
            let causal = seq_q > 1; // Causal during prefill, non-causal during decode

            // Call FlashAttention
            let attn_output = flash_attn(&q_flash, &k_flash, &v_flash, softmax_scale, causal)?;

            // Convert back to original dtype and transpose back
            // [batch, seq_q, num_heads, head_dim] -> [batch, num_heads, seq_q, head_dim]
            let output_tensor = attn_output.to_dtype(original_dtype)?.transpose(1, 2)?;

            // FlashAttention doesn't expose attention weights, so return None
            return Ok((output_tensor, None));
        }

        // === Manual Attention Path (Fallback) ===
        // Note: GQA head expansion already done before calling this function
        // So k and v already have num_heads (Q heads), not num_kv_heads

        // === Compute Attention Scores ===
        // Q @ K^T: [batch, heads, seq_q, head_dim] @ [batch, heads, head_dim, seq_k]
        //       -> [batch, heads, seq_q, seq_k]
        let k_t = k.t()?; // Transpose - swaps last two dimensions
        let attn_weights = q.matmul(&k_t)?;

        // Scale by 1/sqrt(head_dim)
        let attn_weights = (attn_weights * self.scale)?;

        // === Apply ScatteredCache Mask ===
        // This mask prevents batch elements from attending to each other's cache positions
        let attn_weights = if let Some(cache_mask) = mask {
            // cache_mask shape: [batch, 1, seq_q, context]
            // attn_weights shape: [batch, heads, seq_q, seq_k]
            // The mask should broadcast across heads
            attn_weights.broadcast_add(cache_mask)?
        } else {
            attn_weights
        };

        // === Apply Causal Mask ===
        // For decoder, mask out future positions
        let attn_weights = if seq_q > 1 {
            // Only apply mask during prefill (seq_q > 1)
            let _mask = self.create_causal_mask(seq_q, seq_k)?;
            attn_weights.broadcast_add(&_mask)?
        } else {
            // During decode, we only attend to past (no masking needed)
            attn_weights
        };

        // === Softmax ===
        let attn_probs = candlelight::nn::ops::softmax_last_dim(&attn_weights)?;

        // TODO: Handle NaN values that can occur when all attention weights in a row are -inf
        // (This happens for padding tokens that shouldn't attend to anything)
        // Replace NaN with 0.0 to make padding tokens have zero attention
        // attn_probs = attn_probs.nan_to_num(0.0, f32::INFINITY, f32::NEG_INFINITY)?;

        // === ATTENTION SCORE VISUALIZATION (for batched prefill debugging) ===
        #[cfg(feature = "debug-attention-scores")]
        if seq_q > 1 {
            // Only visualize during prefill when there's potential for issues
            self.visualize_attention_scores(&attn_probs, seq_q, seq_k)?;
        }

        // === Apply Attention to Values ===
        // [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, head_dim]
        // -> [batch, heads, seq_q, head_dim]
        let attn_output = attn_probs.matmul(&v)?;

        // Optionally capture attention weights for eviction policies
        let captured_weights = if self.capture_attention {
            Some(attn_probs.clone())
        } else {
            None
        };

        Ok((attn_output, captured_weights))
    }

    /// Create causal attention mask
    ///
    /// Returns a mask where future positions are masked with -inf
    fn create_causal_mask(&self, seq_q: usize, seq_k: usize) -> Result<Tensor> {
        // Create mask: 0 for valid positions, -inf for masked positions
        // Shape: [seq_q, seq_k]
        let mut mask_data = vec![0.0f32; seq_q * seq_k];

        for i in 0..seq_q {
            for j in 0..seq_k {
                // Mask if query position i cannot attend to key position j
                // In causal attention: can only attend to positions <= i
                if j > i {
                    mask_data[i * seq_k + j] = f32::NEG_INFINITY;
                }
            }
        }

        let mask = Tensor::from_vec(mask_data, (seq_q, seq_k), &self.device)?;
        Ok(mask)
    }

    /// Visualize attention scores to debug batched attention
    ///
    /// Shows which positions each token is attending to, helping identify:
    /// - Tokens attending to padding positions (should be near-zero)
    /// - Tokens attending to future positions (should be near-zero for causal)
    /// - Distribution of attention across valid positions
    #[allow(dead_code)]
    fn visualize_attention_scores(
        &self,
        attn_probs: &Tensor,
        seq_q: usize,
        seq_k: usize,
    ) -> Result<()> {
        // attn_probs shape: [batch, heads, seq_q, seq_k]
        let (batch_size, num_heads, _, _) = attn_probs.dims4()?;

        eprintln!("\n╔════════════════════════════════════════════════════════════════");
        eprintln!("║ ATTENTION SCORE VISUALIZATION");
        eprintln!(
            "║ batch_size={}, num_heads={}, seq_q={}, seq_k={}",
            batch_size, num_heads, seq_q, seq_k
        );
        eprintln!("╚════════════════════════════════════════════════════════════════\n");

        // Visualize first batch element, first head (most important)
        let batch_idx = 0;
        let head_idx = 0;

        // Extract attention scores for this batch/head
        // Shape: [seq_q, seq_k]
        let scores = attn_probs.i((batch_idx, head_idx))?;
        let scores_vec = scores.to_vec2::<f32>()?;

        eprintln!("Batch {} Head {} Attention Pattern:", batch_idx, head_idx);
        eprintln!("(Each row = query position, shows where it attends)");
        eprintln!("");

        // Show header with key position indices
        eprint!("Q→K |");
        for k_pos in 0..seq_k.min(20) {
            // Limit width
            eprint!(" {:5}", k_pos);
        }
        if seq_k > 20 {
            eprint!(" ... +{}", seq_k - 20);
        }
        eprintln!(" | Top-3 Attended Positions");
        eprintln!("{}", "─".repeat(seq_k.min(20) * 6 + 50));

        // Show each query position
        for q_pos in 0..seq_q.min(15) {
            // Limit to first 15 query positions
            eprint!(" {:3} |", q_pos);

            let row = &scores_vec[q_pos];

            // Show attention weights (as percentage)
            for k_pos in 0..seq_k.min(20) {
                let prob = row[k_pos];
                let display = if prob < 0.001 {
                    "     ·".to_string() // Near-zero (masked)
                } else if prob >= 0.1 {
                    format!(" {:5.1}", prob * 100.0) // High attention
                } else {
                    format!(" {:5.2}", prob * 100.0) // Low attention
                };
                eprint!("{}", display);
            }

            // Find top-3 attended positions
            let mut indexed: Vec<(usize, f32)> =
                row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            eprint!(" | ");
            for (idx, (pos, prob)) in indexed.iter().take(3).enumerate() {
                if idx > 0 {
                    eprint!(", ");
                }
                eprint!("K{}:{:.1}%", pos, prob * 100.0);
            }

            eprintln!();
        }

        if seq_q > 15 {
            eprintln!("... ({} more query positions)", seq_q - 15);
        }

        // Summary statistics
        eprintln!("\n┌─ Summary Statistics ─────────────────────────────────────");

        // Check for attention to padding (last few positions if padded)
        let last_k_positions = 3;
        let mut padding_attention = 0.0f32;
        for q_pos in 0..seq_q {
            for k_offset in 0..last_k_positions.min(seq_k) {
                let k_pos = seq_k - 1 - k_offset;
                padding_attention += scores_vec[q_pos][k_pos];
            }
        }
        padding_attention /= (seq_q * last_k_positions) as f32;

        eprintln!(
            "│ Avg attention to last {} K positions: {:.4} ({:.2}%)",
            last_k_positions,
            padding_attention,
            padding_attention * 100.0
        );

        // Check for future attention (causal violation)
        let mut future_attention = 0.0f32;
        let mut future_count = 0;
        for q_pos in 0..seq_q {
            for k_pos in (q_pos + 1)..seq_k {
                future_attention += scores_vec[q_pos][k_pos];
                future_count += 1;
            }
        }
        if future_count > 0 {
            future_attention /= future_count as f32;
            eprintln!(
                "│ Avg attention to FUTURE positions: {:.4} ({:.2}%) [should be ~0]",
                future_attention,
                future_attention * 100.0
            );
        }

        // Check diagonal (self-attention strength)
        let mut self_attention = 0.0f32;
        let self_count = seq_q.min(seq_k);
        for i in 0..self_count {
            self_attention += scores_vec[i][i];
        }
        self_attention /= self_count as f32;
        eprintln!(
            "│ Avg diagonal (self) attention: {:.4} ({:.2}%)",
            self_attention,
            self_attention * 100.0
        );

        eprintln!("└───────────────────────────────────────────────────────────\n");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::batch_metadata::{BatchMetadata, SequenceInfo};

    #[test]
    fn test_attention_dimensions() {
        // Test that attention layer compiles and basic dimension logic works
        let hidden_size = 512;
        let num_heads = 8;
        let num_kv_heads = 8;
        let head_dim = hidden_size / num_heads;

        assert_eq!(head_dim, 64);
        assert_eq!(num_heads * head_dim, hidden_size);
    }

    #[test]
    fn test_head_dim_calculations() {
        // Test various configurations
        let configs = vec![
            (512, 8, 64),    // hidden=512, heads=8 -> head_dim=64
            (1024, 16, 64),  // hidden=1024, heads=16 -> head_dim=64
            (4096, 32, 128), // Llama 7B: hidden=4096, heads=32 -> head_dim=128
            (256, 4, 64),    // Small model
        ];

        for (hidden, heads, expected_dim) in configs {
            let head_dim = hidden / heads;
            assert_eq!(
                head_dim, expected_dim,
                "hidden={}, heads={} should give head_dim={}",
                hidden, heads, expected_dim
            );
        }
    }

    #[test]
    fn test_gqa_ratio() {
        // Test GQA (Grouped Query Attention) ratios
        // num_attention_heads must be divisible by num_key_value_heads

        // MHA (Multi-Head Attention): ratio = 1
        let num_heads = 8;
        let num_kv_heads = 8;
        assert_eq!(num_heads / num_kv_heads, 1);

        // GQA: Llama 3 8B has 32 query heads, 8 KV heads (ratio = 4)
        let num_heads = 32;
        let num_kv_heads = 8;
        assert_eq!(num_heads / num_kv_heads, 4);

        // GQA: Another example with ratio = 2
        let num_heads = 16;
        let num_kv_heads = 8;
        assert_eq!(num_heads / num_kv_heads, 2);
    }

    #[test]
    fn test_kv_cache_narrowing_logic() {
        // Test the logic for narrowing KV cache to valid positions

        // Scenario 1: Single request, first token (prefill)
        let context_lens = vec![0];
        let sequence_lengths = vec![5]; // 5-token prompt
        let max_valid_len = context_lens
            .iter()
            .zip(sequence_lengths.iter())
            .map(|(ctx, seq)| ctx + seq)
            .max()
            .unwrap_or(1);
        assert_eq!(max_valid_len, 5, "Prefill: should have 5 positions");

        // Scenario 2: Single request, decode step 3
        let context_lens = vec![3];
        let sequence_lengths = vec![1]; // Generating 1 token
        let max_valid_len = context_lens
            .iter()
            .zip(sequence_lengths.iter())
            .map(|(ctx, seq)| ctx + seq)
            .max()
            .unwrap_or(1);
        assert_eq!(
            max_valid_len, 4,
            "Decode at pos 3: should have 4 positions (0,1,2,3)"
        );

        // Scenario 3: Batch of 2 requests at different positions
        let context_lens = vec![5, 10];
        let sequence_lengths = vec![1, 1];
        let max_valid_len = context_lens
            .iter()
            .zip(sequence_lengths.iter())
            .map(|(ctx, seq)| ctx + seq)
            .max()
            .unwrap_or(1);
        assert_eq!(max_valid_len, 11, "Batch: should use max position (10+1)");

        // Scenario 4: Empty batch (edge case)
        let context_lens: Vec<usize> = vec![];
        let sequence_lengths: Vec<usize> = vec![];
        let max_valid_len = context_lens
            .iter()
            .zip(sequence_lengths.iter())
            .map(|(ctx, seq)| ctx + seq)
            .max()
            .unwrap_or(1);
        assert_eq!(max_valid_len, 1, "Empty batch should default to 1");
    }

    #[test]
    fn test_causal_mask_shape_logic() -> Result<()> {
        // Test causal mask dimensions
        let device = Device::Cpu;

        // Scenario 1: Single token attending to 1 position (decode step 0)
        let seq_q = 1;
        let seq_k = 1;
        // Mask shape should be [seq_q, seq_k] = [1, 1]
        // Only one attention score, no masking needed
        assert_eq!(seq_q, seq_k);

        // Scenario 2: Single token attending to 5 positions (decode step 4)
        let seq_q = 1;
        let seq_k = 5;
        // Mask shape [1, 5], all positions visible (no -inf needed)
        assert!(seq_q <= seq_k);

        // Scenario 3: 5 tokens (prefill), causal mask needed
        let seq_q = 5;
        let seq_k = 5;
        // Mask shape [5, 5], upper triangle should be masked
        // Expected pattern:
        // [0,    -inf, -inf, -inf, -inf]
        // [0,    0,    -inf, -inf, -inf]
        // [0,    0,    0,    -inf, -inf]
        // [0,    0,    0,    0,    -inf]
        // [0,    0,    0,    0,    0   ]

        let mask = (0..seq_q)
            .flat_map(|i| (0..seq_k).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
            .collect::<Vec<_>>();

        assert_eq!(mask.len(), seq_q * seq_k);

        // Verify upper triangle is masked
        assert_eq!(mask[1], f32::NEG_INFINITY); // Row 0, col 1
        assert_eq!(mask[7], f32::NEG_INFINITY); // Row 1, col 2

        // Verify diagonal and below are 0
        assert_eq!(mask[0], 0.0); // Row 0, col 0
        assert_eq!(mask[6], 0.0); // Row 1, col 1
        assert_eq!(mask[12], 0.0); // Row 2, col 2

        Ok(())
    }

    #[test]
    fn test_attention_score_scaling() {
        // Test attention score scaling factor
        let head_dim = 64;
        let scale_factor = 1.0 / (head_dim as f32).sqrt();

        assert!((scale_factor - 0.125).abs() < 1e-6);

        // Different head dimensions
        let head_dim = 128;
        let scale_factor = 1.0 / (head_dim as f32).sqrt();
        assert!((scale_factor - 0.088388).abs() < 1e-5);

        // Verify scaling reduces magnitude
        let score = 10.0;
        let scaled = score * scale_factor;
        assert!(scaled < score);
    }

    #[test]
    fn test_kv_cache_shape_after_narrowing() -> Result<()> {
        // Simulate KV cache narrowing
        let device = Device::Cpu;

        // Start with full cache buffer (512 positions)
        let batch_size = 2;
        let num_kv_heads = 4;
        let max_positions = 512;
        let head_dim = 64;

        let k_cache = Tensor::zeros(
            (batch_size, num_kv_heads, max_positions, head_dim),
            DType::F32,
            &device,
        )?;

        // Narrow to first 5 positions (context_lens=0 + sequence_lengths=5)
        let valid_positions = 5;
        let k_narrowed = k_cache.narrow(2, 0, valid_positions)?;

        assert_eq!(
            k_narrowed.dims(),
            &[batch_size, num_kv_heads, valid_positions, head_dim]
        );

        // Narrow to 10 positions (decode at position 9)
        let valid_positions = 10;
        let k_narrowed = k_cache.narrow(2, 0, valid_positions)?;

        assert_eq!(
            k_narrowed.dims(),
            &[batch_size, num_kv_heads, valid_positions, head_dim]
        );

        Ok(())
    }

    #[test]
    fn test_gqa_expansion_after_narrowing() -> Result<()> {
        // Test GQA: expanding KV heads to match query heads
        let device = Device::Cpu;

        let batch_size = 1;
        let num_kv_heads = 8;
        let num_query_heads = 32; // GQA: 4x expansion
        let seq_len = 5;
        let head_dim = 64;

        // Create KV tensor (narrowed)
        let k = Tensor::zeros(
            (batch_size, num_kv_heads, seq_len, head_dim),
            DType::F32,
            &device,
        )?;

        // Simulate GQA expansion (repeat each KV head 4 times)
        let num_groups = num_query_heads / num_kv_heads;
        assert_eq!(num_groups, 4);

        // After expansion: [batch, 32, seq_len, head_dim]
        // Each of the 8 KV heads gets repeated 4 times
        // Using repeat (dim, count) to expand along head dimension
        let k_expanded = k.repeat((1, num_groups, 1, 1))?;

        assert_eq!(
            k_expanded.dims(),
            &[batch_size, num_query_heads, seq_len, head_dim]
        );

        Ok(())
    }
    #[test]
    fn test_attention_output_shape() -> Result<()> {
        // Test final attention output shape
        let device = Device::Cpu;

        let batch_size = 2;
        let seq_len = 1; // Decode mode
        let num_heads = 8;
        let head_dim = 64;
        let hidden_size = num_heads * head_dim;

        // Attention output before reshape: [batch, heads, seq, head_dim]
        let attn_out = Tensor::zeros(
            (batch_size, num_heads, seq_len, head_dim),
            DType::F32,
            &device,
        )?;

        // After transpose and reshape: [batch, seq, hidden]
        let reshaped = attn_out
            .transpose(1, 2)? // [batch, seq, heads, head_dim]
            .reshape((batch_size, seq_len, hidden_size))?;

        assert_eq!(reshaped.dims(), &[batch_size, seq_len, hidden_size]);

        Ok(())
    }

    #[test]
    fn test_metadata_edge_cases_for_attention() {
        // Test edge cases in metadata that affect attention

        // Edge case 1: Empty batch
        let metadata = BatchMetadata {
            is_prefill: false,
            batch_size: 0,
            request_ids: vec![],
            context_lens: vec![],
            sequences: vec![],
        };

        assert_eq!(metadata.batch_size, 0);
        assert_eq!(metadata.request_ids.len(), 0);

        // Edge case 2: Single token, first position
        let metadata = BatchMetadata {
            is_prefill: true,
            batch_size: 1,
            request_ids: vec![0],
            context_lens: vec![0],
            sequences: vec![SequenceInfo::new_prefill(0, 1)],
        };

        assert_eq!(metadata.context_lens[0], 0);
        assert_eq!(metadata.sequence_lengths()[0], 1);

        // Edge case 3: Max sequence length
        let max_seq = 8192;
        let metadata = BatchMetadata {
            is_prefill: false,
            batch_size: 1,
            request_ids: vec![0],
            context_lens: vec![max_seq - 1],
            sequences: vec![SequenceInfo::new_decode(max_seq - 1)],
        };

        assert_eq!(metadata.context_lens[0], max_seq - 1);

        // Edge case 4: Mismatched array lengths (would be invalid)
        // In production, this should be validated
        let metadata = BatchMetadata {
            is_prefill: false,
            batch_size: 2,
            request_ids: vec![0, 1],
            context_lens: vec![5, 10],
            sequences: vec![SequenceInfo::new_decode(5), SequenceInfo::new_decode(10)],
        };

        assert_eq!(metadata.context_lens.len(), metadata.batch_size);
        assert_eq!(metadata.sequence_lengths().len(), metadata.batch_size);
    }
    #[test]
    fn test_softmax_temperature_effect() {
        // Test softmax attention score properties

        // Before softmax: raw attention scores
        let scores = vec![2.0_f32, 1.0_f32, 0.5_f32];

        // Softmax: exp(x) / sum(exp(x))
        let exp_scores: Vec<f32> = scores.iter().map(|&x| x.exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        let probs: Vec<f32> = exp_scores.iter().map(|&x| x / sum).collect();

        // Properties of softmax probabilities
        assert!(probs.iter().all(|&p| p >= 0.0 && p <= 1.0));
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        // Highest score gets highest probability
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }
    #[test]
    fn test_narrow_operation_preserves_data() -> Result<()> {
        // Verify narrowing doesn't corrupt data
        let device = Device::Cpu;

        // Create tensor with known values
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data.clone(), (4, 5), &device)?;

        // Narrow to first 3 columns
        let narrowed = tensor.narrow(1, 0, 3)?;

        assert_eq!(narrowed.dims(), &[4, 3]);

        // Verify data integrity
        let narrowed_data: Vec<Vec<f32>> = narrowed.to_vec2()?;
        assert_eq!(narrowed_data[0], vec![0.0, 1.0, 2.0]);
        assert_eq!(narrowed_data[1], vec![5.0, 6.0, 7.0]);

        Ok(())
    }

    #[test]
    fn test_batched_prefill_splitting() -> Result<()> {
        // Simulate num_requests=3, padded to 512 each, actual lengths [11,6,4]
        let device = Device::Cpu;
        let num_requests = 3;
        let padded = 512usize;
        let actual_lengths = vec![11usize, 6usize, 4usize];
        let total = padded * num_requests;

        // Create a tensor where each position contains its global token index (u32)
        // We'll expand to shape [1, 1, total, 1] to mimic [batch, heads, seq, dim]
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let q = Tensor::from_vec(data.clone(), (1, 1, total, 1), &device)?;

        // Build metadata with padded starts
        let request_ids: Vec<usize> = (0..num_requests).collect();
        let metadata = crate::model::batch_metadata::BatchMetadata::from_chunked_prefill_batch(
            request_ids,
            actual_lengths.clone(),
            vec![padded; num_requests],
        );

        // Perform the same splitting logic as in production
        let max_seq_len = *metadata.sequence_lengths().iter().max().unwrap();

        for idx in 0..num_requests {
            let seq_info = &metadata.sequences[idx];
            let seq_start = seq_info.start_pos;
            let actual_len = seq_info.actual_length;

            let q_seq = q.narrow(2, seq_start, actual_len)?;
            // Flatten and extract scalar values
            let vals = q_seq.flatten_all()?.to_vec1::<f32>()?;

            // Expect vals to be range(seq_start .. seq_start+actual_len)
            for i in 0..actual_len {
                let expected = (seq_start + i) as f32;
                assert_eq!(vals[i], expected, "Mismatch at idx {} for seq {}", i, idx);
            }
        }

        Ok(())
    }
}
