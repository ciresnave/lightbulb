//! Speculative Model Adapters
//!
//! This module provides adapters that wrap existing model implementations
//! to work with the SpeculativeModel trait.

use crate::cache::ParallelKvCache;
use crate::engine::speculative::SpeculativeModel;
use crate::model::batch_metadata::BatchMetadata;
use crate::model::custom_transformer::BatchedTransformer;
use anyhow::Result;
use candlelight::core::{Device, IndexOp, Tensor};

/// Adapter for BatchedTransformer to work with speculative decoding
pub struct BatchedTransformerAdapter {
    model: BatchedTransformer,
    cache_builder: crate::cache::ParallelCacheBuilder,
    caches: Vec<ParallelKvCache>,
    max_seq_len: usize,
}

impl BatchedTransformerAdapter {
    /// Create a new adapter wrapping a BatchedTransformer
    pub fn new(model: BatchedTransformer, max_seq_len: usize) -> Result<Self> {
        let config = model.config();

        // Create cache builder
        let cache_builder = crate::cache::ParallelCacheBuilder::new(
            1,              // batch_size: single slot for speculative decoding
            max_seq_len,    // context: maximum sequence length
            model.dtype(),  // dtype
            model.device(), // device
        )?;

        // Create per-layer KV caches
        let mut caches = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            caches.push(cache_builder.make_cache(
                config.num_key_value_heads, // num_kv_heads
                config.head_dim(),          // head_dim
            )?);
        }

        Ok(Self {
            model,
            cache_builder,
            caches,
            max_seq_len,
        })
    }

    /// Get reference to underlying model
    pub fn model(&self) -> &BatchedTransformer {
        &self.model
    }

    /// Get mutable reference to underlying model
    pub fn model_mut(&mut self) -> &mut BatchedTransformer {
        &mut self.model
    }
}

impl SpeculativeModel for BatchedTransformerAdapter {
    fn forward_logits(&mut self, tokens: &[u32], _position: usize) -> Result<Tensor> {
        // Create single-request batch metadata
        let metadata = BatchMetadata::from_sequences(&[tokens.to_vec()]);

        // Convert tokens to tensor
        let tokens_tensor = Tensor::new(tokens, self.model.device())?;

        // Forward pass through model
        let logits = self.model.forward(
            &tokens_tensor,
            &mut self.cache_builder,
            &mut self.caches,
            &metadata,
        )?;

        // Extract logits for last position
        // logits shape: [total_tokens, vocab_size]
        let last_logits = logits.i(tokens.len() - 1)?;

        Ok(last_logits)
    }

    fn device(&self) -> &Device {
        self.model.device()
    }

    fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }

    fn reset_cache(&mut self) {
        // Reset cache builder and all KV caches for new sequence
        let config = self.model.config();

        self.cache_builder = crate::cache::ParallelCacheBuilder::new(
            1,
            self.max_seq_len,
            self.model.dtype(),
            self.model.device(),
        )
        .expect("Failed to reset cache builder");

        self.caches.clear();
        for _ in 0..config.num_hidden_layers {
            self.caches.push(
                self.cache_builder
                    .make_cache(config.num_key_value_heads, config.head_dim())
                    .expect("Failed to create cache"),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_interface() {
        // Verify trait requirements compile
        fn _assert_speculative_model<T: SpeculativeModel>() {}
        // Full tests in integration tests with actual models
    }
}
