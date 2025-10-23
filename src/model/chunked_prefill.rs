//! Chunked prefill scheduler for optimal batching of variable-length prompts
//!
//! This module implements a hybrid chunking + padding strategy that enables:
//! - Parallel processing of multiple prompts with different lengths
//! - Early completion for shorter prompts (better time-to-first-token)
//! - Bounded memory usage through fixed chunk sizes
//! - Efficient GPU/CPU utilization through padding
//!
//! # Strategy
//!
//! Instead of processing entire prompts sequentially, we:
//! 1. Split long prompts into fixed-size chunks (e.g., 512 tokens)
//! 2. Pad shorter chunks to enable batching
//! 3. Process multiple requests together in each batch
//! 4. Allow short requests to complete early and start decoding
//!
//! # Example
//!
//! ```text
//! Requests: [2000 tokens, 500 tokens, 150 tokens]
//! Chunk size: 512
//!
//! Batch 1: [512, 500+12 pad, 150+362 pad] ← All 3 together
//! Batch 2: [512, -, -]                    ← Only long request
//! Batch 3: [512, -, -]                    ← Only long request
//! Batch 4: [464, -, -]                    ← Only long request
//!
//! Requests 2 & 3 start decoding after batch 1!
//! ```

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Configuration for chunked prefill strategy
#[derive(Debug, Clone)]
pub struct ChunkedPrefillConfig {
    /// Size of each chunk (e.g., 512 tokens)
    pub chunk_size: usize,
    /// Maximum number of requests to batch together
    pub max_batch_size: usize,
    /// Padding token ID (typically 0 or tokenizer.pad_token_id)
    pub pad_token_id: u32,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            max_batch_size: 4,
            pad_token_id: 0,
        }
    }
}

/// A request in the prefill phase with chunking state
#[derive(Debug, Clone)]
pub struct PrefillRequest {
    /// Unique request identifier
    pub request_id: String,
    /// All tokens to process (full prompt)
    pub tokens: Vec<u32>,
    /// Current position in the token stream
    pub offset: usize,
    /// Starting position for this request (for position embeddings)
    pub start_position: usize,
}

impl PrefillRequest {
    pub fn new(request_id: String, tokens: Vec<u32>) -> Self {
        Self {
            request_id,
            tokens,
            offset: 0,
            start_position: 0,
        }
    }

    /// Check if this request has more tokens to process
    pub fn has_more(&self) -> bool {
        self.offset < self.tokens.len()
    }

    /// Get the next chunk of tokens (up to chunk_size)
    pub fn next_chunk(&mut self, chunk_size: usize) -> Vec<u32> {
        let remaining = self.tokens.len() - self.offset;
        let chunk_len = remaining.min(chunk_size);
        let chunk = self.tokens[self.offset..self.offset + chunk_len].to_vec();
        self.offset += chunk_len;
        chunk
    }

    /// Get remaining tokens count
    pub fn remaining(&self) -> usize {
        self.tokens.len() - self.offset
    }

    /// Check if this is the last chunk
    pub fn is_last_chunk(&self, chunk_size: usize) -> bool {
        self.remaining() <= chunk_size
    }
}

/// A batched chunk ready for processing
#[derive(Debug, Clone)]
pub struct ChunkedBatch {
    /// Request IDs in this batch
    pub request_ids: Vec<String>,
    /// Token sequences (each may be padded to chunk_size)
    pub token_sequences: Vec<Vec<u32>>,
    /// Actual lengths before padding
    pub actual_lengths: Vec<usize>,
    /// Starting positions for each sequence
    pub positions: Vec<usize>,
    /// Padding applied to each sequence
    pub padding: Vec<usize>,
}

impl ChunkedBatch {
    /// Create a batched tensor from token sequences
    pub fn to_tensor(&self, dtype: DType, device: &Device) -> Result<Tensor> {
        if self.token_sequences.is_empty() {
            return Err(anyhow::anyhow!("Cannot create tensor from empty batch"));
        }

        // All sequences should have the same length (due to padding)
        let seq_len = self.token_sequences[0].len();
        let batch_size = self.token_sequences.len();

        // Flatten to 1D vec
        let mut flat_tokens = Vec::with_capacity(batch_size * seq_len);
        for seq in &self.token_sequences {
            if seq.len() != seq_len {
                return Err(anyhow::anyhow!(
                    "Inconsistent sequence lengths in batch: expected {}, got {}",
                    seq_len,
                    seq.len()
                ));
            }
            flat_tokens.extend_from_slice(seq);
        }

        // Create tensor with shape [batch_size, seq_len]
        let tensor = Tensor::from_vec(flat_tokens, (batch_size, seq_len), device)?;

        // Convert to appropriate dtype if needed
        match dtype {
            DType::U32 => Ok(tensor),
            _ => Ok(tensor.to_dtype(dtype)?),
        }
    }

    /// Create a flattened 1D tensor for BatchedTransformer input
    /// Returns tensor of shape [total_tokens] where total_tokens = batch_size * seq_len
    pub fn to_flat_tensor(&self, dtype: DType, device: &Device) -> Result<Tensor> {
        if self.token_sequences.is_empty() {
            return Err(anyhow::anyhow!("Cannot create tensor from empty batch"));
        }

        // All sequences should have the same length (due to padding)
        let seq_len = self.token_sequences[0].len();
        let batch_size = self.token_sequences.len();

        // Flatten to 1D vec
        let mut flat_tokens = Vec::with_capacity(batch_size * seq_len);
        for seq in &self.token_sequences {
            if seq.len() != seq_len {
                return Err(anyhow::anyhow!(
                    "Inconsistent sequence lengths in batch: expected {}, got {}",
                    seq_len,
                    seq.len()
                ));
            }
            flat_tokens.extend_from_slice(seq);
        }

        // Create 1D tensor
        let tensor = Tensor::from_vec(flat_tokens, batch_size * seq_len, device)?;

        // Convert to appropriate dtype if needed
        match dtype {
            DType::U32 => Ok(tensor),
            _ => Ok(tensor.to_dtype(dtype)?),
        }
    }

    /// Get the number of requests in this batch
    pub fn batch_size(&self) -> usize {
        self.request_ids.len()
    }

    /// Calculate efficiency (actual tokens / total tokens including padding)
    pub fn efficiency(&self) -> f64 {
        let actual: usize = self.actual_lengths.iter().sum();
        let total: usize = self.token_sequences.iter().map(|s| s.len()).sum();
        if total == 0 {
            0.0
        } else {
            actual as f64 / total as f64
        }
    }
}

/// Scheduler for chunked prefill with padding
pub struct ChunkedPrefillScheduler {
    config: ChunkedPrefillConfig,
}

impl ChunkedPrefillScheduler {
    pub fn new(config: ChunkedPrefillConfig) -> Self {
        Self { config }
    }

    /// Calculate optimal chunk size for a batch of requests
    ///
    /// Finds the maximum remaining tokens across all requests and rounds up
    /// to the nearest multiple of 32 (GPU warp/wave size) for efficiency.
    ///
    /// This minimizes padding waste while maintaining GPU-friendly alignment.
    fn calculate_dynamic_chunk_size(&self, requests: &[PrefillRequest]) -> usize {
        let max_remaining = requests
            .iter()
            .filter(|r| r.has_more())
            .map(|r| r.remaining())
            .max()
            .unwrap_or(0);

        if max_remaining == 0 {
            return self.config.chunk_size; // Fallback
        }

        // Round up to nearest multiple of 32 for GPU efficiency.
        // Cap at configured chunk_size to avoid excessive memory usage.
        let aligned = ((max_remaining + 31) / 32) * 32;
        aligned.min(self.config.chunk_size)
    }

    /// Create the next batch from available prefill requests
    ///
    /// This method selects up to `max_batch_size` requests and extracts
    /// the next chunk from each, applying padding as needed.
    ///
    /// **NEW**: Uses dynamic chunk sizing based on the longest request in the batch,
    /// minimizing padding waste while maintaining GPU-friendly alignment (multiples of 32).
    ///
    /// # Returns
    /// - `Some(ChunkedBatch)` if there are requests with remaining tokens
    /// - `None` if all requests are complete
    pub fn next_batch(&self, requests: &mut [PrefillRequest]) -> Option<ChunkedBatch> {
        // Filter requests that still have tokens
        let mut active_requests: Vec<&mut PrefillRequest> = requests
            .iter_mut()
            .filter(|r| r.has_more())
            .take(self.config.max_batch_size)
            .collect();

        if active_requests.is_empty() {
            return None;
        }

        // **NEW**: Calculate optimal chunk size for this specific batch
        // This minimizes padding waste while maintaining GPU-friendly alignment
        let chunk_size = self.calculate_dynamic_chunk_size(
            &active_requests
                .iter()
                .map(|r| (**r).clone())
                .collect::<Vec<_>>(),
        );

        eprintln!(
            "DEBUG CHUNKED: Dynamic chunk_size={} for {} requests",
            chunk_size,
            active_requests.len()
        );

        let mut request_ids = Vec::new();
        let mut token_sequences = Vec::new();
        let mut actual_lengths = Vec::new();
        let mut positions = Vec::new();

        // First pass: collect all chunks with their actual lengths
        // Note: next_chunk() mutates the request by advancing offset
        for request in &mut active_requests {
            let start_pos = request.start_position + request.offset;
            let chunk = request.next_chunk(chunk_size);
            let actual_len = chunk.len();

            request_ids.push(request.request_id.clone());
            token_sequences.push(chunk);
            actual_lengths.push(actual_len);
            positions.push(start_pos);
        }

        // Find the max length in this batch
        let max_len = actual_lengths.iter().copied().max().unwrap_or(0);

        eprintln!(
            "DEBUG CHUNKED: Batch actual_lengths={:?}, max_len={}",
            actual_lengths, max_len
        );

        // Second pass: pad all sequences to max_len
        let mut padding = Vec::new();
        for seq in token_sequences.iter_mut() {
            let pad_amount = max_len - seq.len();
            if pad_amount > 0 {
                seq.resize(max_len, self.config.pad_token_id);
            }
            padding.push(pad_amount);
        }

        Some(ChunkedBatch {
            request_ids,
            token_sequences,
            actual_lengths,
            positions,
            padding,
        })
    }

    /// Process all requests and return batches
    ///
    /// Continues creating batches until all requests are complete.
    pub fn schedule_all(&self, requests: &mut [PrefillRequest]) -> Vec<ChunkedBatch> {
        let mut batches = Vec::new();

        while let Some(batch) = self.next_batch(requests) {
            batches.push(batch);
        }

        batches
    }

    /// Get the configuration
    pub fn config(&self) -> &ChunkedPrefillConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefill_request_chunking() {
        let tokens: Vec<u32> = (0..1000).collect();
        let mut req = PrefillRequest::new("test-1".to_string(), tokens);

        assert!(req.has_more());
        assert_eq!(req.remaining(), 1000);

        // First chunk
        let chunk1 = req.next_chunk(512);
        assert_eq!(chunk1.len(), 512);
        assert_eq!(req.remaining(), 488);

        // Second chunk
        let chunk2 = req.next_chunk(512);
        assert_eq!(chunk2.len(), 488);
        assert_eq!(req.remaining(), 0);
        assert!(!req.has_more());
    }

    #[test]
    fn test_chunked_batch_creation() {
        let config = ChunkedPrefillConfig {
            chunk_size: 512,
            max_batch_size: 3,
            pad_token_id: 0,
        };

        let scheduler = ChunkedPrefillScheduler::new(config);

        let mut requests = vec![
            PrefillRequest::new("req-1".to_string(), (0..2000).collect()),
            PrefillRequest::new("req-2".to_string(), (0..500).collect()),
            PrefillRequest::new("req-3".to_string(), (0..150).collect()),
        ];

        // First batch should have all 3 requests
        let batch1 = scheduler.next_batch(&mut requests).unwrap();
        assert_eq!(batch1.batch_size(), 3);
        assert_eq!(batch1.actual_lengths[0], 512); // Full chunk
        assert_eq!(batch1.actual_lengths[1], 500); // Full prompt
        assert_eq!(batch1.actual_lengths[2], 150); // Full prompt

        // Req 2 and 3 should be done
        assert!(!requests[1].has_more());
        assert!(!requests[2].has_more());
        assert!(requests[0].has_more());
    }

    #[test]
    fn test_padding_strategy() {
        let config = ChunkedPrefillConfig {
            chunk_size: 512,
            max_batch_size: 4,
            pad_token_id: 99,
        };

        let scheduler = ChunkedPrefillScheduler::new(config);

        let mut requests = vec![
            PrefillRequest::new("short".to_string(), vec![1, 2, 3]), // 3 tokens
        ];

        let batch = scheduler.next_batch(&mut requests).unwrap();
        assert_eq!(batch.actual_lengths[0], 3);
        assert_eq!(batch.padding[0], 509); // 512 - 3
        assert_eq!(batch.token_sequences[0].len(), 512);
        assert_eq!(batch.token_sequences[0][3], 99); // Padded with 99
    }

    #[test]
    fn test_efficiency_calculation() {
        let batch = ChunkedBatch {
            request_ids: vec!["r1".to_string(), "r2".to_string()],
            token_sequences: vec![
                vec![1; 512], // Full chunk
                vec![2; 512], // Padded from 300
            ],
            actual_lengths: vec![512, 300],
            positions: vec![0, 0],
            padding: vec![0, 212],
        };

        let efficiency = batch.efficiency();
        assert!((efficiency - 0.7933).abs() < 0.001); // (512 + 300) / (512 + 512)
    }

    #[test]
    fn test_schedule_all() {
        let config = ChunkedPrefillConfig::default();
        let scheduler = ChunkedPrefillScheduler::new(config);

        let mut requests = vec![
            PrefillRequest::new("req-1".to_string(), (0..1500).collect()),
            PrefillRequest::new("req-2".to_string(), (0..400).collect()),
        ];

        let batches = scheduler.schedule_all(&mut requests);

        // Should have 3 batches:
        // Batch 1: both requests (512 + 400+112 pad)
        // Batch 2: req-1 only (512)
        // Batch 3: req-1 only (476)
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].batch_size(), 2);
        assert_eq!(batches[1].batch_size(), 1);
        assert_eq!(batches[2].batch_size(), 1);
    }
}
