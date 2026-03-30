//! Batch metadata for describing batched inference structure
//!
//! This module provides the `BatchMetadata` struct that describes how a batch
//! of requests is structured, enabling efficient batched forward passes while
//! handling variable-length sequences.

use candlelight::core::{Device, Result, Tensor};

/// Request ID type (numeric identifier for batch tracking)
///
/// Using usize for efficiency and simplicity. External systems can maintain
/// their own mapping from UUIDs/strings to these numeric IDs.
pub type RequestId = usize;

/// Information about a single sequence in a batch
///
/// This struct is the single source of truth for a sequence's position and length,
/// eliminating the need to keep separate arrays in sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceInfo {
    /// Starting position of this sequence in the concatenated tensor
    /// For example, in a batch of 3 sequences padded to 512 tokens each:
    /// - Sequence 0: start_pos = 0
    /// - Sequence 1: start_pos = 512
    /// - Sequence 2: start_pos = 1024
    pub start_pos: usize,

    /// Number of actual (non-padding) tokens in this sequence
    /// This is the true prompt/generation length, excluding any padding
    pub actual_length: usize,

    /// Total length including padding (equals actual_length if no padding)
    /// For padded batches, this is typically a uniform value (e.g., 512)
    /// For unpacked batches, this equals actual_length
    pub padded_length: usize,

    /// Current position/offset in this sequence's KV cache
    /// - For prefill: typically 0 (start of prompt)
    /// - For decode: current generation position
    pub cache_offset: usize,
}

impl SequenceInfo {
    /// Create a new sequence info for prefill (no padding)
    pub fn new_prefill(start_pos: usize, length: usize) -> Self {
        Self {
            start_pos,
            actual_length: length,
            padded_length: length,
            cache_offset: 0,
        }
    }

    /// Create a new sequence info for prefill with padding
    pub fn new_prefill_padded(
        start_pos: usize,
        actual_length: usize,
        padded_length: usize,
    ) -> Self {
        Self {
            start_pos,
            actual_length,
            padded_length,
            cache_offset: 0,
        }
    }

    /// Create a new sequence info for decode
    pub fn new_decode(cache_offset: usize) -> Self {
        Self {
            start_pos: 0,
            actual_length: 1,
            padded_length: 1,
            cache_offset,
        }
    }

    /// Get the ending position (exclusive) in the concatenated tensor
    pub fn end_pos(&self) -> usize {
        self.start_pos + self.padded_length
    }

    /// Check if this sequence has padding
    pub fn has_padding(&self) -> bool {
        self.padded_length > self.actual_length
    }
}

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
///     sequences: vec![
///         SequenceInfo::new_decode(5),
///         SequenceInfo::new_decode(2),
///         SequenceInfo::new_decode(3),
///     ],
///     context_lens: vec![5, 2, 3],
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
///     sequences: vec![
///         SequenceInfo::new_prefill(0, 5),
///         SequenceInfo::new_prefill(5, 2),
///         SequenceInfo::new_prefill(7, 3),
///     ],
///     context_lens: vec![0, 0, 0],
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

    /// Information about each sequence (position, lengths, cache offset)
    /// This is the single source of truth - replaces cu_seqlens, sequence_lengths, and slot_offsets
    pub sequences: Vec<SequenceInfo>,

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
    ///
    /// # Arguments
    /// * `request_ids` - IDs of requests to include in this batch
    /// * `positions` - Current position in each sequence (for KV cache lookup)
    /// * `context_lens` - How many tokens each sequence has generated so far
    ///
    /// # Returns
    /// BatchMetadata configured for decode batch processing
    pub fn from_decode_batch(
        request_ids: Vec<RequestId>,
        positions: Vec<usize>,
        context_lens: Vec<usize>,
    ) -> Self {
        let batch_size = request_ids.len();

        // For decode, each sequence generates 1 token
        let sequences = positions
            .iter()
            .map(|&pos| SequenceInfo::new_decode(pos))
            .collect();

        Self {
            is_prefill: false,
            batch_size,
            request_ids,
            sequences,
            context_lens,
        }
    }

    /// Get cumulative sequence lengths (for backward compatibility)
    /// Returns [0, len1, len1+len2, ...]
    pub fn cu_seqlens(&self) -> Vec<usize> {
        let mut cu_seqlens = vec![0];
        let mut total = 0;
        for seq in &self.sequences {
            total += seq.padded_length;
            cu_seqlens.push(total);
        }
        cu_seqlens
    }

    /// Get sequence lengths (actual lengths, not padded)
    pub fn sequence_lengths(&self) -> Vec<usize> {
        self.sequences.iter().map(|s| s.actual_length).collect()
    }

    /// Get slot offsets (cache positions)
    pub fn slot_offsets(&self) -> Vec<usize> {
        self.sequences.iter().map(|s| s.cache_offset).collect()
    }

    /// Get maximum sequence length in this batch
    pub fn max_seqlen(&self) -> usize {
        self.sequences
            .iter()
            .map(|s| s.actual_length)
            .max()
            .unwrap_or(0)
    }

    /// Get query start locations (for FlashAttention)
    pub fn query_start_loc(&self) -> Vec<usize> {
        self.sequences.iter().map(|s| s.start_pos).collect()
    }

    /// Create metadata for a prefill batch (variable-length prompts)
    ///
    /// Prefill batches handle initial prompt processing with variable lengths:
    /// - Prompts are concatenated into single tensor
    /// - Each sequence knows its start position and length
    /// - Used ~10% of time (once per request)
    ///
    /// # Arguments
    /// * `request_ids` - IDs of requests in this batch
    /// * `prompt_lengths` - Length of each prompt (actual lengths, not padded)
    ///
    /// # Returns
    /// BatchMetadata configured for prefill batch processing
    pub fn from_prefill_batch(request_ids: Vec<RequestId>, prompt_lengths: Vec<usize>) -> Self {
        let batch_size = request_ids.len();

        // Build sequences with cumulative start positions
        let mut sequences = Vec::with_capacity(batch_size);
        let mut start_pos = 0;
        for &length in &prompt_lengths {
            sequences.push(SequenceInfo::new_prefill(start_pos, length));
            start_pos += length;
        }

        Self {
            is_prefill: true,
            batch_size,
            request_ids,
            sequences,
            context_lens: vec![0; batch_size], // Starting fresh
        }
    }

    /// Create metadata for a chunked prefill batch with padding
    ///
    /// This handles the case where sequences are padded to uniform length for batching.
    /// Uses SequenceInfo to track both actual and padded lengths for each sequence.
    ///
    /// # Arguments
    /// * `request_ids` - IDs of requests in this batch
    /// * `actual_lengths` - True number of tokens in each sequence (without padding)
    /// * `padded_lengths` - Length of each sequence after padding (typically all equal)
    ///
    /// # Example
    /// ```ignore
    /// // Three prompts: [11 tokens, 6 tokens, 4 tokens], each padded to 512
    /// // Tensor structure: [512 tokens (11 real + 501 pad), 512 tokens (6 real + 506 pad), 512 tokens (4 real + 508 pad)]
    /// let metadata = BatchMetadata::from_chunked_prefill_batch(
    ///     vec![id1, id2, id3],
    ///     vec![11, 6, 4],      // actual_lengths
    ///     vec![512, 512, 512], // padded_lengths
    /// );
    /// // sequences[0]: start_pos=0, actual=11, padded=512
    /// // sequences[1]: start_pos=512, actual=6, padded=512
    /// // sequences[2]: start_pos=1024, actual=4, padded=512
    /// ```
    pub fn from_chunked_prefill_batch(
        request_ids: Vec<RequestId>,
        actual_lengths: Vec<usize>,
        padded_lengths: Vec<usize>,
    ) -> Self {
        let batch_size = request_ids.len();

        // Build sequences with cumulative start positions based on PADDED lengths
        let mut sequences = Vec::with_capacity(batch_size);
        let mut start_pos = 0;
        for (&actual, &padded) in actual_lengths.iter().zip(&padded_lengths) {
            sequences.push(SequenceInfo::new_prefill_padded(start_pos, actual, padded));
            start_pos += padded;
        }

        Self {
            is_prefill: true,
            batch_size,
            request_ids,
            sequences,
            context_lens: vec![0; batch_size],
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
        self.sequences.iter().map(|s| s.padded_length).sum()
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
    /// Panics if seq_idx >= batch_size
    pub fn sequence_range(&self, seq_idx: usize) -> (usize, usize) {
        assert!(seq_idx < self.batch_size, "Sequence index out of bounds");
        let seq = &self.sequences[seq_idx];
        (seq.start_pos, seq.end_pos())
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
        let (_start, end) = self.sequence_range(seq_idx);
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
            for seq in &self.sequences {
                for i in 0..seq.actual_length {
                    positions.push(i as u32);
                }
            }
            Tensor::new(positions, device)
        } else {
            // Decode: [pos1, pos2, pos3, ...]
            let positions: Vec<u32> = self
                .sequences
                .iter()
                .map(|s| s.cache_offset as u32)
                .collect();
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
        let metadata =
            BatchMetadata::from_decode_batch(vec![1, 2, 3], vec![5, 2, 3], vec![5, 2, 3]);

        assert_eq!(metadata.is_prefill, false);
        assert_eq!(metadata.batch_size, 3);
        assert_eq!(metadata.total_tokens(), 3);
        assert_eq!(metadata.max_seqlen(), 1);
        assert_eq!(metadata.cu_seqlens(), vec![0, 1, 2, 3]);

        // Check sequence ranges - decode sequences all start at 0 (single token each)
        assert_eq!(metadata.sequence_range(0), (0, 1));
        assert_eq!(metadata.sequence_range(1), (0, 1));
        assert_eq!(metadata.sequence_range(2), (0, 1));

        // Check last token indices - for decode, each is the last (and only) token
        assert_eq!(metadata.last_token_idx(0), 0);
        assert_eq!(metadata.last_token_idx(1), 0);
        assert_eq!(metadata.last_token_idx(2), 0);
    }

    #[test]
    fn test_prefill_batch_metadata() {
        let metadata = BatchMetadata::from_prefill_batch(vec![1, 2, 3], vec![5, 2, 3]);

        assert_eq!(metadata.is_prefill, true);
        assert_eq!(metadata.batch_size, 3);
        assert_eq!(metadata.total_tokens(), 10);
        assert_eq!(metadata.max_seqlen(), 5);

        // Check cumulative sequence lengths
        assert_eq!(metadata.cu_seqlens(), vec![0, 5, 7, 10]);

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
        use candlelight::core::Device;

        let metadata = BatchMetadata::from_decode_batch(vec![1, 2], vec![5, 2], vec![5, 2]);

        let device = Device::Cpu;
        let positions = metadata.create_position_tensor(&device).unwrap();

        // Should be [2, 1] shaped tensor with values [[5], [2]]
        assert_eq!(positions.dims(), &[2, 1]);
        let data = positions.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(data, vec![5, 2]);
    }

    #[test]
    fn test_position_tensor_prefill() {
        use candlelight::core::Device;

        let metadata = BatchMetadata::from_prefill_batch(vec![1, 2], vec![3, 2]);

        let device = Device::Cpu;
        let positions = metadata.create_position_tensor(&device).unwrap();

        // Should be [5] shaped tensor with values [0, 1, 2, 0, 1]
        assert_eq!(positions.dims(), &[5]);
        let data = positions.to_vec1::<u32>().unwrap();
        assert_eq!(data, vec![0, 1, 2, 0, 1]);
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_empty_batch() {
        let metadata = BatchMetadata::from_decode_batch(vec![], vec![], vec![]);

        assert_eq!(metadata.batch_size, 0);
        assert_eq!(metadata.total_tokens(), 0);
        assert_eq!(metadata.context_lens.len(), 0);
        assert_eq!(metadata.sequence_lengths().len(), 0);
    }

    #[test]
    fn test_single_sequence_prefill() {
        let metadata = BatchMetadata::from_prefill_batch(vec![0], vec![10]);

        assert!(metadata.is_prefill);
        assert_eq!(metadata.batch_size, 1);
        assert_eq!(metadata.total_tokens(), 10);
        assert_eq!(metadata.sequence_range(0), (0, 10));
        assert_eq!(metadata.last_token_idx(0), 9);
    }

    #[test]
    fn test_single_sequence_decode() {
        let metadata = BatchMetadata::from_decode_batch(vec![0], vec![5], vec![5]);

        assert!(!metadata.is_prefill);
        assert_eq!(metadata.batch_size, 1);
        assert_eq!(metadata.total_tokens(), 1);
        assert_eq!(metadata.context_lens[0], 5);
        assert_eq!(metadata.sequence_range(0), (0, 1));
    }

    #[test]
    fn test_max_batch_size() {
        // Test with large batch size
        let batch_size = 128;
        let request_ids: Vec<usize> = (0..batch_size).collect();
        let positions: Vec<usize> = (0..batch_size).map(|i| i * 2).collect();

        let metadata =
            BatchMetadata::from_decode_batch(request_ids, positions.clone(), positions.clone());

        assert_eq!(metadata.batch_size, batch_size);
        assert_eq!(metadata.total_tokens(), batch_size);
        assert_eq!(metadata.context_lens, positions);
    }

    #[test]
    fn test_varying_sequence_lengths() {
        // Prefill with very different sequence lengths
        let sequence_lengths = vec![1, 10, 100, 5, 50];
        let request_ids = vec![0, 1, 2, 3, 4];

        let metadata = BatchMetadata::from_prefill_batch(request_ids, sequence_lengths.clone());

        assert!(metadata.is_prefill);
        assert_eq!(metadata.total_tokens(), 1 + 10 + 100 + 5 + 50);
        assert_eq!(metadata.max_seqlen(), 100);

        // Verify cumulative lengths
        let expected_cu = vec![0, 1, 11, 111, 116, 166];
        assert_eq!(metadata.cu_seqlens(), expected_cu);
    }

    #[test]
    fn test_decode_varying_context_lens() {
        // Decode batch with requests at different positions
        let request_ids = vec![0, 1, 2, 3, 4];
        let positions = vec![0, 5, 10, 50, 100];

        let metadata =
            BatchMetadata::from_decode_batch(request_ids, positions.clone(), positions.clone());

        assert!(!metadata.is_prefill);
        assert_eq!(metadata.context_lens, positions);

        // Each decode sequence generates 1 token, all start at position 0
        for i in 0..5 {
            assert_eq!(metadata.sequence_range(i), (0, 1));
        }
    }

    #[test]
    fn test_mismatched_lengths_edge_case() {
        // With new design, we can't create mismatched metadata - SequenceInfo prevents it
        // This test can be removed or updated to test SequenceInfo construction
        let sequences = vec![
            SequenceInfo::new_prefill(0, 5),
            SequenceInfo::new_prefill(5, 10),
        ];

        let metadata = BatchMetadata {
            is_prefill: true,
            batch_size: 2,
            request_ids: vec![0, 1],
            context_lens: vec![0, 1, 2], // Extra element (doesn't matter for this test)
            sequences,
        };

        // Accessing beyond bounds would panic, so keep assertions safe
        assert_eq!(metadata.batch_size, 2);
    }

    #[test]
    fn test_sequence_range_bounds() {
        let metadata = BatchMetadata::from_prefill_batch(vec![0, 0, 0], vec![5, 3, 7]);

        // Valid indices
        assert_eq!(metadata.sequence_range(0), (0, 5));
        assert_eq!(metadata.sequence_range(1), (5, 8));
        assert_eq!(metadata.sequence_range(2), (8, 15));

        // Out of bounds would panic in actual impl (not tested here)
    }

    #[test]
    fn test_last_token_idx_bounds() {
        let metadata = BatchMetadata::from_prefill_batch(vec![0, 0], vec![10, 20]);

        assert_eq!(metadata.last_token_idx(0), 9); // First 10 tokens: 0-9
        assert_eq!(metadata.last_token_idx(1), 29); // Next 20 tokens: 10-29
    }

    #[test]
    fn test_zero_sequence_length() {
        // Edge case: sequence with length 0 (shouldn't happen in practice)
        let metadata = BatchMetadata::from_prefill_batch(vec![0, 0], vec![5, 0]);

        assert_eq!(metadata.total_tokens(), 5);
        assert_eq!(metadata.sequence_range(0), (0, 5));
        assert_eq!(metadata.sequence_range(1), (5, 5)); // Empty range
    }

    #[test]
    fn test_large_sequence_length() {
        // Test with very long sequence (8k tokens)
        let max_seq = 8192;
        let metadata = BatchMetadata::from_prefill_batch(vec![0], vec![max_seq]);

        assert_eq!(metadata.total_tokens(), max_seq);
        assert_eq!(metadata.max_seqlen(), max_seq);
        assert_eq!(metadata.last_token_idx(0), max_seq - 1);
    }

    #[test]
    fn test_position_tensor_single_token() -> Result<()> {
        use candlelight::core::Device;

        let metadata = BatchMetadata::from_decode_batch(vec![0], vec![0], vec![0]);
        let device = Device::Cpu;
        let positions = metadata.create_position_tensor(&device)?;

        assert_eq!(positions.dims(), &[1, 1]);
        let data = positions.to_vec2::<u32>()?;
        assert_eq!(data, vec![vec![0]]);

        Ok(())
    }

    #[test]
    fn test_position_tensor_max_position() -> Result<()> {
        use candlelight::core::Device;

        let max_pos = 8191;
        let metadata = BatchMetadata::from_decode_batch(vec![0], vec![max_pos], vec![max_pos]);
        let device = Device::Cpu;
        let positions = metadata.create_position_tensor(&device)?;

        let data = positions.to_vec2::<u32>()?;
        assert_eq!(data, vec![vec![max_pos as u32]]);

        Ok(())
    }

    #[test]
    fn test_total_tokens_calculation() {
        // Decode: total_tokens = sum of sequence_lengths (all 1 for decode)
        let metadata =
            BatchMetadata::from_decode_batch(vec![0, 1, 2], vec![5, 10, 15], vec![5, 10, 15]);
        assert_eq!(metadata.total_tokens(), 3);

        // Prefill: total_tokens = sum of sequence_lengths
        let metadata = BatchMetadata::from_prefill_batch(vec![0, 1], vec![10, 20]);
        assert_eq!(metadata.total_tokens(), 30);
    }

    #[test]
    fn test_request_ids_assignment() {
        let request_ids = vec![0, 1, 2];
        let positions = vec![0, 5, 10];

        let metadata =
            BatchMetadata::from_decode_batch(request_ids.clone(), positions.clone(), positions);

        // Request IDs should match what we passed
        assert_eq!(metadata.request_ids, request_ids);
    }

    #[test]
    fn test_slot_offsets_decode() {
        // In decode mode, slot_offsets = positions
        let request_ids = vec![0, 1, 2];
        let positions = vec![3, 7, 15];

        let metadata =
            BatchMetadata::from_decode_batch(request_ids, positions.clone(), positions.clone());

        assert_eq!(metadata.slot_offsets(), positions);
    }

    #[test]
    fn test_slot_offsets_prefill() {
        // In prefill mode, slot_offsets are all 0 (start of prompts)
        let request_ids = vec![0, 1, 2];
        let sequence_lengths = vec![5, 3, 7];

        let metadata = BatchMetadata::from_prefill_batch(request_ids, sequence_lengths);

        assert_eq!(metadata.slot_offsets(), vec![0, 0, 0]);
    }

    #[test]
    fn test_max_seqlen_decode() {
        let metadata = BatchMetadata::from_decode_batch(vec![0, 1], vec![5, 10], vec![5, 10]);

        // In decode mode, max_seqlen = 1 since all sequences generate 1 token
        assert_eq!(metadata.max_seqlen(), 1);
    }

    #[test]
    fn test_max_seqlen_prefill_single() {
        let metadata = BatchMetadata::from_prefill_batch(vec![0], vec![42]);

        assert_eq!(metadata.max_seqlen(), 42);
    }

    #[test]
    fn test_max_seqlen_prefill_batch() {
        let metadata = BatchMetadata::from_prefill_batch(vec![0, 1, 2], vec![5, 100, 25]);

        // Max of 5, 100, 25
        assert_eq!(metadata.max_seqlen(), 100);
    }
}
