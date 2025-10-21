//! Batch metadata for describing batched inference structure
//!
//! This module provides the `BatchMetadata` struct that describes how a batch
//! of requests is structured, enabling efficient batched forward passes while
//! handling variable-length sequences.

use candle_core::{Device, Result, Tensor};

/// Request ID type (numeric identifier for batch tracking)
///
/// Using usize for efficiency and simplicity. External systems can maintain
/// their own mapping from UUIDs/strings to these numeric IDs.
pub type RequestId = usize;

/// Describes the structure and composition of a batch for batched inference
///
/// This metadata is essential for processing batches with variable-length sequences,
/// tracking where each sequence starts/ends, and managing KV cache slot mappings.
///
/// # Examples
///
/// ## Decode Batch (most common - 90% of operations)
/// ```rust,ignore
/// // Three sequences, each generating one token:
/// // Seq1 at position 5, Seq2 at position 2, Seq3 at position 3
/// let metadata = BatchMetadata {
///     is_prefill: false,
///     batch_size: 3,
///     request_ids: vec![id1, id2, id3],
///     slot_offsets: vec![5, 2, 3],  // Current position in each sequence
///     sequence_lengths: vec![1, 1, 1],  // One token each
///     cu_seqlens: None,
///     max_seqlen: Some(1),
/// };
/// ```
///
/// ## Prefill Batch (variable-length prompts)
/// ```rust,ignore
/// // Three prompts: [1,2,3,4,5], [6,7], [8,9,10]
/// let metadata = BatchMetadata {
///     is_prefill: true,
///     batch_size: 3,
///     request_ids: vec![id1, id2, id3],
///     slot_offsets: vec![0, 0, 0],  // All start at position 0
///     sequence_lengths: vec![5, 2, 3],
///     cu_seqlens: Some(vec![0, 5, 7, 10]),  // Cumulative: [0, 5, 7, 10]
///     max_seqlen: Some(5),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct BatchMetadata {
    /// Whether this is a prefill batch (processing prompts) vs decode batch (generating tokens)
    pub is_prefill: bool,

    /// Number of sequences in this batch
    pub batch_size: usize,

    /// Request IDs for each sequence in the batch (for KV cache lookup)
    pub request_ids: Vec<RequestId>,

    /// Current position/offset in each sequence's KV cache
    /// - For decode: current generation position
    /// - For prefill: typically 0 (start of prompt)
    pub slot_offsets: Vec<usize>,

    /// Length of each sequence in the batch
    /// - For decode: typically all 1 (one token per request)
    /// - For prefill: variable (prompt lengths)
    pub sequence_lengths: Vec<usize>,

    /// Cumulative sequence lengths for concatenated prefill batches
    /// Format: [0, len1, len1+len2, len1+len2+len3, ...]
    /// Used to extract individual sequences from concatenated tensor
    /// Only Some(_) for prefill batches
    pub cu_seqlens: Option<Vec<usize>>,

    /// Maximum sequence length in this batch
    /// Used for attention mask construction
    pub max_seqlen: Option<usize>,

    /// Query start locations for FlashAttention optimization
    /// Marks where each query starts in concatenated tensor
    /// Format: [0, len1, len1+len2, ...] (typically same as cu_seqlens)
    /// Only Some(_) for prefill batches (decode doesn't need it)
    /// Will be used when FlashAttention is enabled (M2 milestone)
    pub query_start_loc: Option<Vec<usize>>,

    /// Context lengths: how many tokens each sequence has seen so far
    /// Used for continuous batching to track request history
    /// - Prefill: all zeros (starting fresh)
    /// - Decode: current position in generation (accumulated history)
    pub context_lens: Vec<usize>,
}

impl BatchMetadata {
    /// Create metadata for a decode batch (one token per request)
    ///
    /// Decode batches are the most common (90% of operations) and simplest:
    /// - Each request contributes exactly 1 token
    /// - All sequences have length 1
    /// - Positions are the current generation position for each request
    ///
    /// # Arguments
    /// * `request_ids` - IDs of requests in this batch
    /// * `positions` - Current position in each request's sequence
    ///
    /// # Returns
    /// BatchMetadata configured for decode batch processing
    pub fn from_decode_batch(request_ids: Vec<RequestId>, positions: Vec<usize>) -> Self {
        let batch_size = request_ids.len();

        Self {
            is_prefill: false,
            batch_size,
            request_ids,
            slot_offsets: positions.clone(),
            sequence_lengths: vec![1; batch_size], // All length 1
            cu_seqlens: None,                      // Not needed for decode
            max_seqlen: Some(1),
            query_start_loc: None,   // Not needed for decode
            context_lens: positions, // Position = context length for decode
        }
    }

    /// Create metadata for a prefill batch (variable-length prompts)
    ///
    /// Prefill batches handle initial prompt processing with variable lengths:
    /// - Prompts are concatenated into single tensor
    /// - cu_seqlens tracks boundaries between prompts
    /// - Used ~10% of time (once per request)
    ///
    /// # Arguments
    /// * `request_ids` - IDs of requests in this batch
    /// * `prompt_lengths` - Length of each prompt
    ///
    /// # Returns
    /// BatchMetadata configured for prefill batch processing
    pub fn from_prefill_batch(request_ids: Vec<RequestId>, prompt_lengths: Vec<usize>) -> Self {
        let batch_size = request_ids.len();

        // Build cumulative sequence lengths: [0, len1, len1+len2, ...]
        let mut cu_seqlens = vec![0];
        let mut total = 0;
        for &len in &prompt_lengths {
            total += len;
            cu_seqlens.push(total);
        }

        let max_seqlen = prompt_lengths.iter().copied().max();

        // Query start locations are the same as cu_seqlens boundaries (without the final total)
        let query_start_loc = cu_seqlens[..batch_size].to_vec();

        Self {
            is_prefill: true,
            batch_size,
            request_ids,
            slot_offsets: vec![0; batch_size], // All start at 0
            sequence_lengths: prompt_lengths,
            cu_seqlens: Some(cu_seqlens),
            max_seqlen,
            query_start_loc: Some(query_start_loc), // For FlashAttention
            context_lens: vec![0; batch_size],      // Starting fresh
        }
    }

    /// Create metadata from a list of token sequences (for testing)
    ///
    /// This is a convenience method for tests that simulates a prefill batch.
    /// Each inner Vec represents a sequence of token IDs.
    ///
    /// # Arguments
    /// * `sequences` - Vector of token sequences, e.g., vec![vec![1,2,3], vec![4,5]]
    ///
    /// # Returns
    /// BatchMetadata configured as a prefill batch with auto-generated request IDs
    ///
    /// Example
    /// ```ignore
    /// let sequences = vec![vec![1u32, 2, 3], vec![4u32, 5]];
    /// let metadata = BatchMetadata::from_sequences(&sequences);
    /// // Creates 2 sequences: request 0 (length 3), request 1 (length 2)
    /// ```
    pub fn from_sequences(sequences: &[Vec<u32>]) -> Self {
        let request_ids: Vec<RequestId> = (0..sequences.len()).collect();
        let prompt_lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
        Self::from_prefill_batch(request_ids, prompt_lengths)
    }

    /// Get the total number of tokens across all sequences in this batch
    ///
    /// For decode batches: equals batch_size (1 token each)
    /// For prefill batches: sum of all prompt lengths
    pub fn total_tokens(&self) -> usize {
        self.sequence_lengths.iter().sum()
    }

    /// Get the (start, end) indices for a specific sequence in a concatenated tensor
    ///
    /// Used to extract individual sequence results from batched output
    ///
    /// # Arguments
    /// * `seq_idx` - Index of sequence in batch (0..batch_size)
    ///
    /// # Returns
    /// (start_idx, end_idx) for slicing concatenated tensor
    ///
    /// # Panics
    /// Panics if seq_idx >= batch_size or if cu_seqlens is None for prefill batch
    pub fn sequence_range(&self, seq_idx: usize) -> (usize, usize) {
        assert!(seq_idx < self.batch_size, "Sequence index out of bounds");

        if self.is_prefill {
            let cu_seqlens = self
                .cu_seqlens
                .as_ref()
                .expect("Prefill batch must have cu_seqlens");
            (cu_seqlens[seq_idx], cu_seqlens[seq_idx + 1])
        } else {
            // Decode: each sequence is 1 token
            (seq_idx, seq_idx + 1)
        }
    }

    /// Get the index of the last token for a specific sequence
    ///
    /// This is the token whose logits we want for next-token prediction
    ///
    /// # Arguments
    /// * `seq_idx` - Index of sequence in batch (0..batch_size)
    ///
    /// # Returns
    /// Index of last token in the concatenated/batched tensor
    pub fn last_token_idx(&self, seq_idx: usize) -> usize {
        let (start, end) = self.sequence_range(seq_idx);
        end - 1
    }

    /// Create a position tensor for this batch
    ///
    /// Returns a tensor with shape:
    /// - Decode: [batch_size, 1] - one position per request
    /// - Prefill: [total_tokens] - position within each prompt
    ///
    /// # Arguments
    /// * `device` - Device to create tensor on
    ///
    /// # Returns
    /// Position tensor ready for model forward pass
    pub fn create_position_tensor(&self, device: &Device) -> Result<Tensor> {
        if self.is_prefill {
            // Prefill: [0..len1, 0..len2, 0..len3, ...]
            let mut positions = Vec::new();
            for &len in &self.sequence_lengths {
                for i in 0..len {
                    positions.push(i as u32);
                }
            }
            Tensor::new(positions, device)
        } else {
            // Decode: [pos1, pos2, pos3, ...]
            let positions: Vec<u32> = self.slot_offsets.iter().map(|&pos| pos as u32).collect();
            let tensor = Tensor::new(positions, device)?;
            tensor.reshape((self.batch_size, 1))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_batch_metadata() {
        let metadata = BatchMetadata::from_decode_batch(vec![1, 2, 3], vec![5, 2, 3]);

        assert_eq!(metadata.is_prefill, false);
        assert_eq!(metadata.batch_size, 3);
        assert_eq!(metadata.total_tokens(), 3);
        assert_eq!(metadata.max_seqlen, Some(1));
        assert!(metadata.cu_seqlens.is_none());

        // Check sequence ranges
        assert_eq!(metadata.sequence_range(0), (0, 1));
        assert_eq!(metadata.sequence_range(1), (1, 2));
        assert_eq!(metadata.sequence_range(2), (2, 3));

        // Check last token indices
        assert_eq!(metadata.last_token_idx(0), 0);
        assert_eq!(metadata.last_token_idx(1), 1);
        assert_eq!(metadata.last_token_idx(2), 2);
    }

    #[test]
    fn test_prefill_batch_metadata() {
        let metadata = BatchMetadata::from_prefill_batch(vec![1, 2, 3], vec![5, 2, 3]);

        assert_eq!(metadata.is_prefill, true);
        assert_eq!(metadata.batch_size, 3);
        assert_eq!(metadata.total_tokens(), 10);
        assert_eq!(metadata.max_seqlen, Some(5));

        // Check cumulative sequence lengths
        assert_eq!(metadata.cu_seqlens, Some(vec![0, 5, 7, 10]));

        // Check sequence ranges
        assert_eq!(metadata.sequence_range(0), (0, 5));
        assert_eq!(metadata.sequence_range(1), (5, 7));
        assert_eq!(metadata.sequence_range(2), (7, 10));

        // Check last token indices
        assert_eq!(metadata.last_token_idx(0), 4);
        assert_eq!(metadata.last_token_idx(1), 6);
        assert_eq!(metadata.last_token_idx(2), 9);
    }

    #[test]
    fn test_position_tensor_decode() {
        use candle_core::Device;

        let metadata = BatchMetadata::from_decode_batch(vec![1, 2], vec![5, 2]);

        let device = Device::Cpu;
        let positions = metadata.create_position_tensor(&device).unwrap();

        // Should be [2, 1] shaped tensor with values [[5], [2]]
        assert_eq!(positions.dims(), &[2, 1]);
        let data = positions.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(data, vec![5, 2]);
    }

    #[test]
    fn test_position_tensor_prefill() {
        use candle_core::Device;

        let metadata = BatchMetadata::from_prefill_batch(vec![1, 2], vec![3, 2]);

        let device = Device::Cpu;
        let positions = metadata.create_position_tensor(&device).unwrap();

        // Should be [5] shaped tensor with values [0, 1, 2, 0, 1]
        assert_eq!(positions.dims(), &[5]);
        let data = positions.to_vec1::<u32>().unwrap();
        assert_eq!(data, vec![0, 1, 2, 0, 1]);
    }
}
