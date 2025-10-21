//! Model-agnostic batch manager for inference
//!
//! This module provides a generic batching interface that works with
//! any Candle transformer model (Llama, Mistral, Gemma, etc.).
//!
//! # Design Philosophy
//!
//! BatchManager is **model-agnostic** - it handles:
//! - Grouping requests into batches
//! - Coordinating forward passes
//! - Managing per-request KV caches
//! - Splitting batched outputs back to individual requests
//!
//! The actual model architecture (Llama, Mistral, etc.) is irrelevant to
//! the batching logic. All we need is a `forward()` method that accepts
//! tokens and returns logits.

use crate::engine::BatchExecutor;
use crate::model::BatchMetadata;
use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::models::llama::Cache;

/// Model-agnostic batch manager for inference
///
/// Generic over any model type that implements the required forward pass.
/// Works with Llama, Mistral, Gemma, Qwen, etc. without modification.
///
/// # Type Parameters
/// * `M` - Model type (e.g., `candle_transformers::models::llama::Llama`)
pub struct BatchManager<M> {
    model: M,
    batch_executor: BatchExecutor,
    device: Device,
}

impl<M> BatchManager<M>
where
    M: TransformerModel,
{
    /// Create a new batch manager around any Candle model
    ///
    /// # Arguments
    /// * `model` - Pre-loaded Candle model (Llama, Mistral, etc.)
    /// * `batch_executor` - Batch executor with KV cache management
    /// * `device` - Device for tensor operations
    pub fn new(model: M, batch_executor: BatchExecutor, device: Device) -> Self {
        Self {
            model,
            batch_executor,
            device,
        }
    }

    /// Forward pass for a batched decode step
    ///
    /// Processes multiple single-token decode requests in sequence.
    /// Each request gets its own KV cache and position tracking.
    ///
    /// # Arguments
    /// * `tokens` - Batched token tensor [batch_size] for decode
    /// * `metadata` - Batch structure description
    /// * `per_request_caches` - Individual cache for each request
    ///
    /// # Returns
    /// Batched logits tensor [batch_size, vocab_size]
    pub fn forward_decode_batch(
        &mut self,
        tokens: &Tensor,
        metadata: &BatchMetadata,
        per_request_caches: &mut [Cache],
    ) -> Result<Tensor> {
        // Verify this is a decode batch
        assert!(
            !metadata.is_prefill,
            "Use forward_prefill_batch for prefill"
        );
        assert_eq!(metadata.batch_size, per_request_caches.len());

        let mut logits_batch = Vec::with_capacity(metadata.batch_size);

        // Process each request sequentially
        // TODO: Explore true batched forward pass for 6x speedup
        for batch_idx in 0..metadata.batch_size {
            let position = metadata.slot_offsets[batch_idx];

            // Extract single token for this request
            let token = tokens.i(batch_idx)?.unsqueeze(0)?.unsqueeze(0)?; // [1, 1]

            // Model-agnostic forward pass
            let logits =
                self.model
                    .forward(&token, position, &mut per_request_caches[batch_idx])?;

            // Extract last token logits
            let last_logits = logits.i(logits.dim(0)? - 1)?;
            logits_batch.push(last_logits);
        }

        // Stack into [batch_size, vocab_size]
        Ok(Tensor::stack(&logits_batch, 0)?)
    }

    /// Forward pass for a batched prefill step
    ///
    /// Processes prompt tokens for multiple requests. Each prompt is
    /// processed independently with its own KV cache.
    ///
    /// # Arguments
    /// * `tokens` - Concatenated token tensor [total_tokens]
    /// * `metadata` - Batch structure with sequence boundaries
    /// * `per_request_caches` - Individual cache for each request
    ///
    /// # Returns
    /// Logits for last token of each sequence [batch_size, vocab_size]
    pub fn forward_prefill_batch(
        &mut self,
        tokens: &Tensor,
        metadata: &BatchMetadata,
        per_request_caches: &mut [Cache],
    ) -> Result<Tensor> {
        // Verify this is a prefill batch
        assert!(metadata.is_prefill, "Use forward_decode_batch for decode");
        assert_eq!(metadata.batch_size, per_request_caches.len());

        let mut logits_batch = Vec::with_capacity(metadata.batch_size);

        // Process each prompt sequentially
        for batch_idx in 0..metadata.batch_size {
            let (start, end) = metadata.sequence_range(batch_idx);

            // Extract tokens for this prompt
            let prompt_tokens = tokens.narrow(0, start, end - start)?;

            // Process entire prompt token by token
            let mut position = 0;
            let mut last_logits = None;

            let num_tokens = end - start;
            if num_tokens == 0 {
                anyhow::bail!(
                    "Empty prompt for batch_idx {}: start={}, end={}",
                    batch_idx,
                    start,
                    end
                );
            }

            for token_idx in 0..num_tokens {
                let token = prompt_tokens.i(token_idx)?.unsqueeze(0)?.unsqueeze(0)?; // [1, 1]

                // Model-agnostic forward pass
                let logits =
                    self.model
                        .forward(&token, position, &mut per_request_caches[batch_idx])?;

                last_logits = Some(logits.i(logits.dim(0)? - 1)?);
                position += 1;
            }

            logits_batch.push(last_logits.ok_or_else(|| {
                anyhow::anyhow!("No logits produced for batch_idx {}", batch_idx)
            })?);
        }

        // Stack into [batch_size, vocab_size]
        Ok(Tensor::stack(&logits_batch, 0)?)
    }

    /// Get reference to the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get reference to batch executor for cache management
    pub fn batch_executor(&self) -> &BatchExecutor {
        &self.batch_executor
    }

    /// Get mutable reference to batch executor
    pub fn batch_executor_mut(&mut self) -> &mut BatchExecutor {
        &mut self.batch_executor
    }

    /// Get mutable reference to the underlying model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

/// Trait defining the interface for transformer models
///
/// Any Candle model implementing this trait can be used with BatchManager.
/// This includes Llama, Mistral, Gemma, Qwen, and other decoder-only models.
pub trait TransformerModel {
    /// Forward pass for a single sequence
    ///
    /// # Arguments
    /// * `tokens` - Input token tensor [seq_len] or [1, seq_len]
    /// * `position` - Current position in sequence (for positional encoding)
    /// * `cache` - KV cache for this sequence
    ///
    /// # Returns
    /// Logits tensor [seq_len, vocab_size]
    fn forward(&mut self, tokens: &Tensor, position: usize, cache: &mut Cache) -> Result<Tensor>;
}

// Implement trait for Candle's Llama model
impl TransformerModel for candle_transformers::models::llama::Llama {
    fn forward(&mut self, tokens: &Tensor, position: usize, cache: &mut Cache) -> Result<Tensor> {
        // Candle's Llama::forward returns candle_core::Result, convert to anyhow::Result
        candle_transformers::models::llama::Llama::forward(self, tokens, position, cache)
            .map_err(|e| anyhow::anyhow!("Llama forward pass failed: {}", e))
    }
}

// Implement trait for Mistral (when needed)
// impl TransformerModel for candle_transformers::models::mistral::Model {
//     fn forward(&mut self, tokens: &Tensor, position: usize, cache: &mut Cache) -> Result<Tensor> {
//         candle_transformers::models::mistral::Model::forward(self, tokens, position, cache)
//     }
// }

// Implement trait for Gemma (when needed)
// impl TransformerModel for candle_transformers::models::gemma::Model {
//     fn forward(&mut self, tokens: &Tensor, position: usize, cache: &mut Cache) -> Result<Tensor> {
//         candle_transformers::models::gemma::Model::forward(self, tokens, position, cache)
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_manager_structure() {
        // Verify the generic structure compiles
        // Actual functionality tests require a loaded model
        assert!(true);
    }
}
