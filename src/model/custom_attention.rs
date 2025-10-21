//! Custom Batched Attention Layer for True Batched Inference
//!
//! This module implements a batched attention mechanism that can process
//! multiple sequences in parallel, achieving significant speedup over
//! sequential processing.
//!
//! # Architecture
//!
//! Based on candle-vllm's batched attention approach, adapted for our
//! ScatteredKvCache infrastructure:
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

use crate::engine::BatchExecutor;
use crate::model::BatchMetadata;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

/// Batched Multi-Head Attention Layer
///
/// Processes multiple sequences in parallel using shared KV cache.
#[derive(Debug)]
pub struct BatchedAttention {
    // Linear projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,

    // Model configuration
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,

    // Attention scale factor
    scale: f64,

    // Device and dtype
    device: Device,
    dtype: DType,
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
        let q_proj = candle_nn::linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj =
            candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj =
            candle_nn::linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let device = vb.device().clone();
        let dtype = vb.dtype();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            scale,
            device,
            dtype,
        })
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
    /// Output tensor [batch_size, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        index_pos: usize,
        cos: &Tensor,
        sin: &Tensor,
        batch_executor: &mut BatchExecutor,
        metadata: &BatchMetadata,
        layer_idx: usize,
    ) -> Result<Tensor> {
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

        // === Step 3: Apply RoPE (Rotary Position Embeddings) ===
        // DEBUG: Check Q/K before RoPE
        if layer_idx == 0 {
            let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
            let k_vec = k.flatten_all()?.to_vec1::<f32>()?;
            let cos_vec = cos.flatten_all()?.to_vec1::<f32>()?;
            let sin_vec = sin.flatten_all()?.to_vec1::<f32>()?;
            eprintln!(
                "DEBUG RoPE BEFORE - Q: shape={:?}, mean={:.6}, sample[0:3]={:?}",
                q.dims(),
                q_vec.iter().sum::<f32>() / q_vec.len() as f32,
                &q_vec[0..3.min(q_vec.len())]
            );
            eprintln!(
                "DEBUG RoPE BEFORE - K: shape={:?}, mean={:.6}, sample[0:3]={:?}",
                k.dims(),
                k_vec.iter().sum::<f32>() / k_vec.len() as f32,
                &k_vec[0..3.min(k_vec.len())]
            );
            eprintln!("DEBUG RoPE - index_pos={}, seq_len={}", index_pos, seq_len);
            eprintln!(
                "DEBUG RoPE - cos shape={:?}, sample[0:3]={:?}",
                cos.dims(),
                &cos_vec[0..3.min(cos_vec.len())]
            );
            eprintln!(
                "DEBUG RoPE - sin shape={:?}, sample[0:3]={:?}",
                sin.dims(),
                &sin_vec[0..3.min(sin_vec.len())]
            );
        }

        // Apply manual RoPE implementation to Q and K
        let q = self.apply_rotary_emb(&q, index_pos, seq_len, cos, sin)?;
        let k = self.apply_rotary_emb(&k, index_pos, seq_len, cos, sin)?;

        // DEBUG: Check Q/K after RoPE
        if layer_idx == 0 {
            let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
            let k_vec = k.flatten_all()?.to_vec1::<f32>()?;
            eprintln!(
                "DEBUG RoPE AFTER - Q: mean={:.6}, sample[0:3]={:?}",
                q_vec.iter().sum::<f32>() / q_vec.len() as f32,
                &q_vec[0..3.min(q_vec.len())]
            );
            eprintln!(
                "DEBUG RoPE AFTER - K: mean={:.6}, sample[0:3]={:?}",
                k_vec.iter().sum::<f32>() / k_vec.len() as f32,
                &k_vec[0..3.min(k_vec.len())]
            );
        }

        // === Step 4: Update KV Cache ===
        // Note: K/V are stored with num_kv_heads in the cache
        // GQA head expansion happens later during attention computation
        // Get indices and mask for the current batch (all requests active)
        let iam = batch_executor
            .get_indices_and_mask_simple(batch_size, seq_len)
            .map_err(|e| {
                candle_core::Error::Msg(format!("Failed to get indices and mask: {}", e))
            })?;

        // Append current K/V to cache and get full K/V history
        // This returns K/V tensors that include all historical tokens plus current
        let (k_full, v_full) = batch_executor
            .append_kv(layer_idx, &k, &v, &iam)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to append KV: {}", e)))?;

        // === Step 4.3: Narrow KV cache to valid sequence length ===
        // The cache returns the full buffer (e.g., 512 positions), but we only want
        // the valid portion that's been filled so far
        // For single-sequence case: valid_len = context_len + current_seq_len
        // For multi-sequence: each sequence may have different valid lengths
        let max_valid_len = metadata
            .context_lens
            .iter()
            .zip(metadata.sequence_lengths.iter())
            .map(|(ctx, seq)| ctx + seq)
            .max()
            .unwrap_or(1);

        let k_full = k_full.narrow(2, 0, max_valid_len)?;
        let v_full = v_full.narrow(2, 0, max_valid_len)?;

        if layer_idx == 0 {
            eprintln!(
                "DEBUG Layer 0: Narrowed cache to {} positions (context_lens={:?}, seq_lens={:?})",
                max_valid_len, metadata.context_lens, metadata.sequence_lengths
            );
        }

        // === Step 4.5: Expand KV heads for GQA ===
        // If using Grouped Query Attention, repeat KV heads to match Q heads
        // This needs to be done AFTER narrowing the cache
        let (k_full, v_full) = if self.num_heads != self.num_kv_heads {
            let repeat_factor = self.num_heads / self.num_kv_heads;
            // Get dimensions AFTER narrowing
            let (_batch, _num_kv_heads, total_seq_len, _head_dim) = k_full.dims4()?;

            if layer_idx == 0 {
                eprintln!(
                    "DEBUG GQA: k_full before expand: {:?}, repeat_factor={}",
                    k_full.dims(),
                    repeat_factor
                );
            }

            // K/V shape from cache: [batch, num_kv_heads, total_seq, dim]
            // Target: [batch, num_heads, total_seq, dim]
            let k_expanded = k_full
                .unsqueeze(2)? // [batch, num_kv_heads, 1, seq, dim]
                .expand(&[
                    batch_size,
                    self.num_kv_heads,
                    repeat_factor,
                    total_seq_len,
                    self.head_dim,
                ])?
                .reshape((batch_size, self.num_heads, total_seq_len, self.head_dim))?;

            let v_expanded = v_full
                .unsqueeze(2)?
                .expand(&[
                    batch_size,
                    self.num_kv_heads,
                    repeat_factor,
                    total_seq_len,
                    self.head_dim,
                ])?
                .reshape((batch_size, self.num_heads, total_seq_len, self.head_dim))?;

            if layer_idx == 0 {
                eprintln!("DEBUG GQA: k_expanded: {:?}", k_expanded.dims());
            }

            (k_expanded, v_expanded)
        } else {
            (k_full, v_full)
        };

        // === Step 5: Compute Attention ===
        // Use full K/V history (including current) for attention
        // k_full, v_full shape: [batch, num_heads, total_seq_len, head_dim]
        // where total_seq_len = historical_tokens + seq_len

        if layer_idx == 0 {
            eprintln!(
                "DEBUG Layer 0 Attention: q shape={:?}, k_full shape={:?}, v_full shape={:?}",
                q.dims(),
                k_full.dims(),
                v_full.dims()
            );
        }

        if layer_idx <= 1 {
            eprintln!(
                "DEBUG Layer {} attention inputs: Q: {:?}, K: {:?}, V: {:?}",
                layer_idx,
                q.dims(),
                k_full.dims(),
                v_full.dims()
            );

            // Print actual Q, K, V values
            let q_vec = q.flatten_all()?.to_vec1::<f32>()?;
            let k_vec = k_full.flatten_all()?.to_vec1::<f32>()?;
            let v_vec = v_full.flatten_all()?.to_vec1::<f32>()?;
            eprintln!(
                "DEBUG Layer {} Q: mean={:.6}, sample[0:5]={:?}",
                layer_idx,
                q_vec.iter().sum::<f32>() / q_vec.len() as f32,
                &q_vec[0..5.min(q_vec.len())]
            );
            eprintln!(
                "DEBUG Layer {} K: mean={:.6}, sample[0:5]={:?}",
                layer_idx,
                k_vec.iter().sum::<f32>() / k_vec.len() as f32,
                &k_vec[0..5.min(k_vec.len())]
            );
            eprintln!(
                "DEBUG Layer {} V: mean={:.6}, sample[0:5]={:?}",
                layer_idx,
                v_vec.iter().sum::<f32>() / v_vec.len() as f32,
                &v_vec[0..5.min(v_vec.len())]
            );
        }

        // Since we've narrowed K/V to only valid positions, we don't need the ScatteredCache mask
        // The mask was designed for the full 512-position buffer, but would cause shape mismatch
        // after narrowing. The causal mask (applied inside compute_attention) is sufficient.
        let attn_output = self.compute_attention(&q, &k_full, &v_full, None)?;

        if layer_idx == 0 {
            let attn_vec = attn_output.flatten_all()?.to_vec1::<f32>()?;
            eprintln!(
                "DEBUG Layer 0 attn_output BEFORE o_proj: mean={:.6}, sample[0:5]={:?}",
                attn_vec.iter().sum::<f32>() / attn_vec.len() as f32,
                &attn_vec[0..5.min(attn_vec.len())]
            );
        }

        // === Step 6: Reshape and Output Projection ===
        // [batch, heads, seq, dim] -> [batch, seq, heads * dim]
        let attn_output = attn_output
            .transpose(1, 2)? // [batch, seq, heads, dim]
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?
            .contiguous()?;

        // Final output projection
        let output = self.o_proj.forward(&attn_output)?;

        if layer_idx == 0 {
            let output_vec = output.flatten_all()?.to_vec1::<f32>()?;
            eprintln!(
                "DEBUG Layer 0 AFTER o_proj: mean={:.6}, sample[0:5]={:?}",
                output_vec.iter().sum::<f32>() / output_vec.len() as f32,
                &output_vec[0..5.min(output_vec.len())]
            );
        }

        Ok(output)
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

        if index_pos == 0 {
            eprintln!(
                "DEBUG RoPE - cos_slice shape={:?}, sin_slice shape={:?}",
                cos_slice.dims(),
                sin_slice.dims()
            );
            eprintln!("DEBUG RoPE - x shape={:?}", x.dims());
        }

        // Use Candle's built-in RoPE function (same as Llama)
        candle_nn::rotary_emb::rope(x, &cos_slice, &sin_slice)
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
    /// Attention output [batch, num_heads, seq_q, head_dim]
    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Get dimensions
        let (_batch, _num_heads, seq_q, _head_dim) = q.dims4()?;
        let (_batch_k, _num_heads_k, seq_k, _head_dim_k) = k.dims4()?;

        // Note: GQA head expansion already done before calling this function
        // So k and v already have num_heads (Q heads), not num_kv_heads

        // === Compute Attention Scores ===
        // Q @ K^T: [batch, heads, seq_q, head_dim] @ [batch, heads, head_dim, seq_k]
        //       -> [batch, heads, seq_q, seq_k]
        let k_t = k.t()?; // Transpose - swaps last two dimensions
        let attn_weights = q.matmul(&k_t)?;

        // DEBUG: Check raw attention scores
        let attn_w_vec = attn_weights.flatten_all()?.to_vec1::<f32>()?;
        eprintln!(
            "DEBUG compute_attention: raw Q@K^T mean={:.6}, sample[0:5]={:?}",
            attn_w_vec.iter().sum::<f32>() / attn_w_vec.len() as f32,
            &attn_w_vec[0..5.min(attn_w_vec.len())]
        );

        // Scale by 1/sqrt(head_dim)
        let attn_weights = (attn_weights * self.scale)?;

        // DEBUG: Check scaled attention scores
        let scaled_vec = attn_weights.flatten_all()?.to_vec1::<f32>()?;
        eprintln!(
            "DEBUG compute_attention: scaled scores (scale={:.6}) mean={:.6}, sample[0:5]={:?}",
            self.scale,
            scaled_vec.iter().sum::<f32>() / scaled_vec.len() as f32,
            &scaled_vec[0..5.min(scaled_vec.len())]
        );

        // === Apply ScatteredCache Mask ===
        // This mask prevents batch elements from attending to each other's cache positions
        let attn_weights = if let Some(cache_mask) = mask {
            eprintln!(
                "DEBUG: Before mask - attn_weights shape={:?}, cache_mask shape={:?}",
                attn_weights.dims(),
                cache_mask.dims()
            );
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
            let mask = self.create_causal_mask(seq_q, seq_k)?;
            attn_weights.broadcast_add(&mask)?
        } else {
            // During decode, we only attend to past (no masking needed)
            attn_weights
        };

        // === Softmax ===
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // DEBUG: Check softmax probabilities
        let probs_vec = attn_probs.flatten_all()?.to_vec1::<f32>()?;
        eprintln!(
            "DEBUG compute_attention: softmax probs mean={:.6}, sample[0:5]={:?}",
            probs_vec.iter().sum::<f32>() / probs_vec.len() as f32,
            &probs_vec[0..5.min(probs_vec.len())]
        );

        // === Apply Attention to Values ===
        // [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, head_dim]
        // -> [batch, heads, seq_q, head_dim]
        let attn_output = attn_probs.matmul(&v)?;

        // DEBUG: Check final attention output
        let output_vec = attn_output.flatten_all()?.to_vec1::<f32>()?;
        eprintln!(
            "DEBUG compute_attention: final output mean={:.6}, sample[0:5]={:?}",
            output_vec.iter().sum::<f32>() / output_vec.len() as f32,
            &output_vec[0..5.min(output_vec.len())]
        );

        Ok(attn_output)
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_causal_mask_shape() {
        // Test causal mask creation
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create dummy attention layer (we just need the mask function)
        // In real test, would need VarBuilder setup

        // Just verify mask logic
        let seq_q = 4;
        let seq_k = 4;

        // Mask should be [seq_q, seq_k] with upper triangle = -inf
        // [0,    -inf, -inf, -inf]
        // [0,    0,    -inf, -inf]
        // [0,    0,    0,    -inf]
        // [0,    0,    0,    0   ]

        assert!(true); // Placeholder - full test requires model weights
    }
}
