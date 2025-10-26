//! StreamingLLM-style KV cache eviction policy
//!
//! Implements the StreamingLLM approach for constant-memory long-context generation:
//! - **Attention Sinks**: Keep first N tokens (typically 4-8) always
//! - **Sliding Window**: Keep most recent M tokens
//! - **Middle Eviction**: When full, evict from the middle (between sinks and window)
//!
//! Benefits:
//! - Constant memory usage beyond window size
//! - Minimal quality degradation (attention sinks stabilize)
//! - Works on any hardware (CPU/GPU)
//!
//! References:
//! - "Efficient Streaming Language Models with Attention Sinks"
//! - https://arxiv.org/abs/2309.17453

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Configuration for StreamingLLM-style KV cache management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Enable StreamingLLM policy
    pub enabled: bool,

    /// Number of initial tokens to keep as "attention sinks" (typically 4-8)
    ///
    /// These tokens are never evicted and help stabilize attention patterns.
    /// The paper shows that models naturally develop attention sinks at the start.
    pub sink_size: usize,

    /// Size of the sliding window for recent context (e.g., 1024, 2048)
    ///
    /// The most recent window_size tokens are always kept.
    /// This provides the immediate context for generation.
    pub window_size: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sink_size: 4,      // 4 attention sink tokens (typical)
            window_size: 2048, // 2K recent tokens
        }
    }
}

impl StreamingConfig {
    /// Create a new StreamingLLM config
    ///
    /// # Arguments
    /// * `sink_size` - Number of initial attention sink tokens (typically 4-8)
    /// * `window_size` - Size of sliding window for recent context (e.g., 1024, 2048)
    ///
    /// # Example
    /// ```ignore
    /// let config = StreamingConfig::new(4, 2048);
    /// ```
    pub fn new(sink_size: usize, window_size: usize) -> Self {
        Self {
            enabled: true,
            sink_size,
            window_size,
        }
    }

    /// Total cache size needed (sinks + window)
    pub fn total_cache_size(&self) -> usize {
        self.sink_size + self.window_size
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.enabled {
            if self.sink_size == 0 {
                anyhow::bail!("StreamingLLM sink_size must be > 0");
            }
            if self.window_size == 0 {
                anyhow::bail!("StreamingLLM window_size must be > 0");
            }
            if self.sink_size >= self.window_size {
                anyhow::bail!(
                    "StreamingLLM sink_size ({}) must be < window_size ({})",
                    self.sink_size,
                    self.window_size
                );
            }
        }
        Ok(())
    }
}

/// Compute which cache indices to use for the next token(s) with StreamingLLM eviction
///
/// This implements the core StreamingLLM logic:
/// 1. If position < total_cache_size, use sequential positions (no eviction yet)
/// 2. Once full, start evicting from the middle:
///    - Keep [0..sink_size) as attention sinks
///    - Keep [cache_size - window_size..cache_size) as recent window
///    - Write new tokens to the eviction zone (middle)
///
/// # Arguments
/// * `position` - Current sequence position (can exceed cache size)
/// * `config` - StreamingLLM configuration
/// * `cache_size` - Total cache capacity
///
/// # Returns
/// The cache index where the next token should be written
///
/// # Example
/// ```ignore
/// let config = StreamingConfig::new(4, 1024);  // 4 sinks + 1024 window
/// let cache_size = 1028;  // Total cache size
///
/// // First 1028 tokens: sequential indices
/// assert_eq!(compute_streaming_index(0, &config, cache_size), 0);
/// assert_eq!(compute_streaming_index(100, &config, cache_size), 100);
///
/// // Token 1028: cache full, start evicting from middle
/// // Sinks: [0..4), Window: [1024..1028), Eviction zone: [4..1024)
/// assert_eq!(compute_streaming_index(1028, &config, cache_size), 4);  // Overwrite position 4
/// assert_eq!(compute_streaming_index(1029, &config, cache_size), 5);  // Overwrite position 5
/// ```
pub fn compute_streaming_index(
    position: usize,
    config: &StreamingConfig,
    cache_size: usize,
) -> usize {
    if !config.enabled {
        // No streaming: use position directly (will wrap if position >= cache_size)
        return position % cache_size;
    }

    let total_size = config.total_cache_size();

    if total_size > cache_size {
        // Config asks for more space than cache provides - fallback to simple wraparound
        return position % cache_size;
    }

    if position < total_size {
        // Cache not full yet: use sequential positions
        return position;
    }

    // Cache is full: implement StreamingLLM eviction
    //
    // Layout:
    // [0..sink_size)                          - Attention sinks (never evicted)
    // [sink_size..cache_size-window_size)     - Eviction zone (circular buffer)
    // [cache_size-window_size..cache_size)    - Recent window (protected)
    //
    // The recent window slides forward as new tokens arrive, but sinks stay fixed.

    let eviction_zone_start = config.sink_size;
    let eviction_zone_size = cache_size - config.sink_size - config.window_size;

    if eviction_zone_size == 0 {
        // No middle zone (sink + window fills entire cache)
        // Just overwrite the last position of sinks
        return config.sink_size - 1;
    }

    // Compute offset within eviction zone (circular)
    let offset_in_eviction = (position - total_size) % eviction_zone_size;
    eviction_zone_start + offset_in_eviction
}

/// Compute cache indices for a batch of tokens (used for prefill)
///
/// For each token in the sequence, computes where it should be written in cache.
///
/// # Arguments
/// * `start_position` - Starting sequence position
/// * `seq_len` - Number of tokens to compute indices for
/// * `config` - StreamingLLM configuration
/// * `cache_size` - Total cache capacity
///
/// # Returns
/// Vector of cache indices, one per token
pub fn compute_streaming_indices(
    start_position: usize,
    seq_len: usize,
    config: &StreamingConfig,
    cache_size: usize,
) -> Vec<usize> {
    (0..seq_len)
        .map(|i| compute_streaming_index(start_position + i, config, cache_size))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_validation() {
        // Valid config
        let config = StreamingConfig::new(4, 2048);
        assert!(config.validate().is_ok());

        // Invalid: sink_size == 0
        let mut bad_config = config.clone();
        bad_config.sink_size = 0;
        assert!(bad_config.validate().is_err());

        // Invalid: window_size == 0
        let mut bad_config = config.clone();
        bad_config.window_size = 0;
        assert!(bad_config.validate().is_err());

        // Invalid: sink >= window
        let mut bad_config = config.clone();
        bad_config.sink_size = 2048;
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_streaming_disabled() {
        let config = StreamingConfig {
            enabled: false,
            sink_size: 4,
            window_size: 1024,
        };

        // When disabled, should just use modulo
        assert_eq!(compute_streaming_index(0, &config, 100), 0);
        assert_eq!(compute_streaming_index(50, &config, 100), 50);
        assert_eq!(compute_streaming_index(100, &config, 100), 0); // Wrap
        assert_eq!(compute_streaming_index(150, &config, 100), 50); // Wrap
    }

    #[test]
    fn test_streaming_before_full() {
        let config = StreamingConfig::new(4, 1024);
        let cache_size = 1028; // 4 + 1024

        // Before cache is full: sequential indices
        for i in 0..1028 {
            assert_eq!(
                compute_streaming_index(i, &config, cache_size),
                i,
                "Position {} should map to index {}",
                i,
                i
            );
        }
    }

    #[test]
    fn test_streaming_eviction() {
        let config = StreamingConfig::new(4, 16); // 4 sinks + 16 window
        let cache_size = 24; // Total: 24

        // Eviction zone: [4..8) (4 positions between sinks and window)
        // Layout:
        // [0..4)   - Sinks (never evicted)
        // [4..8)   - Eviction zone (4 positions)
        // [8..24)  - Recent window (16 positions)

        // Position 24: cache full, start evicting from position 4
        assert_eq!(compute_streaming_index(24, &config, cache_size), 4);
        assert_eq!(compute_streaming_index(25, &config, cache_size), 5);
        assert_eq!(compute_streaming_index(26, &config, cache_size), 6);
        assert_eq!(compute_streaming_index(27, &config, cache_size), 7);

        // Position 28: wrap back to position 4 (circular within eviction zone)
        assert_eq!(compute_streaming_index(28, &config, cache_size), 4);
        assert_eq!(compute_streaming_index(29, &config, cache_size), 5);
    }

    #[test]
    fn test_streaming_indices_batch() {
        let config = StreamingConfig::new(4, 8);
        let cache_size = 16; // 4 + 4 (eviction) + 8 (window)

        // Prefill 5 tokens starting at position 0
        let indices = compute_streaming_indices(0, 5, &config, cache_size);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);

        // Prefill 5 tokens starting at position 16 (cache full, evicting)
        let indices = compute_streaming_indices(16, 5, &config, cache_size);
        // Should write to eviction zone: [4, 5, 6, 7, 4]
        assert_eq!(indices, vec![4, 5, 6, 7, 4]);
    }

    #[test]
    fn test_no_eviction_zone() {
        // Edge case: sink + window exactly fills cache (no middle zone)
        let config = StreamingConfig::new(4, 20);
        let cache_size = 24; // Exactly sink + window

        // Once full, should just overwrite last sink position
        assert_eq!(compute_streaming_index(24, &config, cache_size), 3);
        assert_eq!(compute_streaming_index(25, &config, cache_size), 3);
    }
}
