//! Batched Llama Inference using Candle's Built-in Model
//!
//! This module provides a thin wrapper around `candle_transformers::models::llama::Llama`
//! to handle batched inference with per-request KV caching.
//!
//! # Why Not Custom Implementation?
//!
//! Candle's Llama model already supports:
//! - Batched inputs: `[batch_size, seq_len]` → `[batch_size, vocab_size]`
//! - KV caching via `Cache` struct
//! - All transformer layers (attention, MLP, normalization)
//! - Optimized implementations
//!
//! We don't need to reimplement any of this!
//!
//! # What We Actually Need
//!
//! 1. **Per-request KV cache management**: Each request has its own cache
//! 2. **Batch assembly**: Group multiple requests into a single forward pass
//! 3. **Output routing**: Split batched outputs back to individual requests
//!
//! # Architecture
//!
//! ```text
//! BatchedLlama
//!   ├─> Candle's Llama (handles all model logic)
//!   ├─> HashMap<RequestId, Cache> (per-request KV caches)
//!   └─> Batch assembly/disassembly logic
//! ```

use candle_core::{DType, Device, Result, Tensor};
use candle_transformers::models::llama::{Cache, Config as LlamaConfig, Llama};
use std::collections::HashMap;

/// Request identifier
pub type RequestId = usize;

/// Batched Llama model for efficient multi-request inference
pub struct BatchedLlama {
    /// Candle's built-in Llama model (does all the heavy lifting!)
    model: Llama,
    
    /// Per-request KV caches
    /// Each request maintains its own cache to handle variable-length sequences
    caches: HashMap<RequestId, Cache>,
    
    /// Model configuration
    config: LlamaConfig,
    
    /// Device for tensor operations
    device: Device,
}

impl BatchedLlama {
    /// Create a new BatchedLlama from a pre-loaded Llama model
    pub fn new(model: Llama, config: LlamaConfig, device: Device) -> Self {
        Self {
            model,
            caches: HashMap::new(),
            config,
            device,
        }
    }
    
    /// Get or create a cache for a request
    fn get_or_create_cache(&mut self, request_id: RequestId) -> &mut Cache {
        self.caches.entry(request_id).or_insert_with(|| {
            Cache::new(
                self.config.num_hidden_layers,
                self.config.max_position_embeddings,
            )
        })
    }
    
    /// Remove a request's cache when it completes
    pub fn remove_cache(&mut self, request_id: RequestId) {
        self.caches.remove(&request_id);
    }
    
    /// Process a single request
    ///
    /// # Arguments
    /// * `request_id` - Unique identifier for this request
    /// * `input_ids` - Input token IDs, shape `[seq_len]`
    /// * `index_pos` - Current position in the sequence (for KV cache)
    ///
    /// # Returns
    /// Logits for next token, shape `[vocab_size]`
    pub fn forward_single(
        &mut self,
        request_id: RequestId,
        input_ids: &[u32],
        index_pos: usize,
    ) -> Result<Tensor> {
        // Create input tensor: [1, seq_len]
        let input_tensor = Tensor::new(input_ids, &self.device)?
            .unsqueeze(0)?; // Add batch dimension
        
        // Get cache for this request
        let cache = self.get_or_create_cache(request_id);
        
        // Forward pass through Candle's Llama
        // Returns [1, vocab_size] logits
        let logits = self.model.forward(&input_tensor, index_pos, cache)?;
        
        // Remove batch dimension: [vocab_size]
        logits.squeeze(0)
    }
    
    /// Process a batch of requests
    ///
    /// # Arguments
    /// * `batch` - List of (request_id, input_ids, index_pos) tuples
    ///
    /// # Returns
    /// Vec of logits tensors, one per request, each shape `[vocab_size]`
    pub fn forward_batch(
        &mut self,
        batch: &[(RequestId, Vec<u32>, usize)],
    ) -> Result<Vec<Tensor>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        
        // For now, process each request individually
        // TODO: Optimize by actually batching when sequences are same length
        let mut results = Vec::with_capacity(batch.len());
        
        for (request_id, input_ids, index_pos) in batch {
            let logits = self.forward_single(*request_id, input_ids, *index_pos)?;
            results.push(logits);
        }
        
        Ok(results)
    }
    
    /// Get the model configuration
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }
    
    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get number of active caches
    pub fn num_active_requests(&self) -> usize {
        self.caches.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarBuilder;
    
    #[test]
    #[ignore] // Requires model files
    fn test_batched_llama_single_request() -> anyhow::Result<()> {
        // This test demonstrates the simple API
        // In practice, you'd load a real model
        
        // Example usage:
        // let (model, config, device) = load_llama_model("path/to/model")?;
        // let mut batched = BatchedLlama::new(model, config, device);
        // 
        // let logits = batched.forward_single(
        //     request_id: 1,
        //     input_ids: &[1, 2, 3, 4],
        //     index_pos: 0,
        // )?;
        
        Ok(())
    }
}
