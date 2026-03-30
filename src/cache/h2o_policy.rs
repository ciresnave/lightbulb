use super::eviction_policy::EvictionPolicy;
/// H2O (Heavy Hitters Oracle) eviction policy.
///
/// Tracks cumulative attention scores for each token position and evicts
/// tokens with the lowest attention scores (i.e., tokens that have been
/// "ignored" the most during generation).
///
/// Based on the H2O paper: https://arxiv.org/abs/2306.14048
/// "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"
use std::collections::HashMap;

/// Configuration for H2O eviction policy.
#[derive(Debug, Clone)]
pub struct H2OConfig {
    /// Whether H2O tracking is enabled.
    pub enabled: bool,

    /// Number of recent tokens to always keep (similar to attention sinks).
    /// These tokens are never evicted regardless of attention scores.
    pub num_recent_to_keep: usize,

    /// Decay factor for attention scores (0.0 to 1.0).
    /// Higher values give more weight to recent attention.
    /// 1.0 = no decay (all history equally weighted)
    /// 0.9 = recent attention weighted slightly higher
    pub decay_factor: f32,
}

impl Default for H2OConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_recent_to_keep: 4, // Similar to StreamingLLM attention sinks
            decay_factor: 0.95,    // Slight preference for recent attention
        }
    }
}

/// Per-token metadata for H2O policy.
#[derive(Debug, Clone)]
pub struct TokenMetadata {
    /// Cumulative attention score (sum of all attention this token received).
    pub cumulative_attention: f32,

    /// Number of generation steps this token has been present.
    pub steps_present: usize,

    /// The original sequence position of this token.
    pub original_position: usize,
}

impl TokenMetadata {
    pub fn new(position: usize) -> Self {
        Self {
            cumulative_attention: 0.0,
            steps_present: 0,
            original_position: position,
        }
    }

    /// Average attention score per step.
    pub fn average_attention(&self) -> f32 {
        if self.steps_present == 0 {
            0.0
        } else {
            self.cumulative_attention / (self.steps_present as f32)
        }
    }
}

/// H2O policy state tracking attention scores per cache slot.
#[derive(Debug, Clone)]
pub struct H2OPolicy {
    /// Configuration for H2O.
    config: H2OConfig,

    /// Metadata for each cache slot (slot_id -> metadata).
    slot_metadata: HashMap<usize, TokenMetadata>,

    /// Current generation step.
    current_step: usize,
}

impl H2OPolicy {
    pub fn new(config: H2OConfig) -> Self {
        Self {
            config,
            slot_metadata: HashMap::new(),
            current_step: 0,
        }
    }

    /// Update attention scores for all tokens after a generation step.
    ///
    /// # Arguments
    /// * `attention_weights` - Attention weights from the model (shape: [num_heads, seq_len, seq_len])
    ///   or pre-aggregated across heads (shape: [seq_len, seq_len])
    /// * `cache_positions` - Mapping from cache slot to sequence position
    pub fn update_attention_scores(
        &mut self,
        attention_weights: &[Vec<f32>], // [query_pos][key_pos] -> attention weight
        cache_positions: &HashMap<usize, usize>, // slot -> seq_position
    ) {
        self.current_step += 1;

        // Aggregate attention received by each key position
        let mut position_attention: HashMap<usize, f32> = HashMap::new();

        for query_attention in attention_weights {
            for (key_pos, &attn_weight) in query_attention.iter().enumerate() {
                *position_attention.entry(key_pos).or_insert(0.0) += attn_weight;
            }
        }

        // Apply decay and update slot metadata
        for (&slot_id, &seq_position) in cache_positions.iter() {
            let metadata = self
                .slot_metadata
                .entry(slot_id)
                .or_insert_with(|| TokenMetadata::new(seq_position));

            // Apply decay to existing cumulative attention
            metadata.cumulative_attention *= self.config.decay_factor;

            // Add new attention for this step
            if let Some(&new_attention) = position_attention.get(&seq_position) {
                metadata.cumulative_attention += new_attention;
            }

            metadata.steps_present += 1;
        }
    }

    /// Compute eviction scores for all cache slots.
    /// Higher score = higher priority for eviction.
    ///
    /// # Arguments
    /// * `cache_size` - Total cache size
    /// * `num_used` - Number of slots currently in use
    /// * `cache_positions` - Mapping from cache slot to sequence position
    ///
    /// # Returns
    /// Vector of (slot_id, eviction_score) pairs, sorted by score (highest first).
    pub fn compute_eviction_scores(
        &self,
        cache_size: usize,
        num_used: usize,
        cache_positions: &HashMap<usize, usize>,
    ) -> Vec<(usize, f32)> {
        if !self.config.enabled {
            return Vec::new();
        }

        let mut scores = Vec::with_capacity(num_used);

        // Find the most recent tokens (based on sequence position)
        let max_position = cache_positions.values().max().copied().unwrap_or(0);
        let recent_threshold = max_position.saturating_sub(self.config.num_recent_to_keep);

        for (&slot_id, &seq_position) in cache_positions.iter() {
            // Recent tokens are never evicted (score = -infinity)
            if seq_position > recent_threshold {
                scores.push((slot_id, f32::NEG_INFINITY));
                continue;
            }

            // Compute eviction score based on attention
            let metadata = self.slot_metadata.get(&slot_id);
            let eviction_score = if let Some(metadata) = metadata {
                // Lower average attention = higher eviction priority
                // Invert so that low attention -> high score
                let avg_attention = metadata.average_attention();
                if avg_attention > 0.0 {
                    1.0 / avg_attention
                } else {
                    f32::INFINITY // No attention at all = highest eviction priority
                }
            } else {
                // No metadata = token was never attended to
                f32::INFINITY
            };

            scores.push((slot_id, eviction_score));
        }

        // Sort by eviction score (highest first)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
    }

    /// Get the slot with the highest eviction priority.
    pub fn select_eviction_candidate(
        &self,
        cache_size: usize,
        num_used: usize,
        cache_positions: &HashMap<usize, usize>,
    ) -> Option<usize> {
        let scores = self.compute_eviction_scores(cache_size, num_used, cache_positions);

        // Return the slot with the highest eviction score (if finite)
        scores
            .first()
            .filter(|(_, score)| score.is_finite())
            .map(|(slot, _)| *slot)
    }

    /// Clear metadata for a slot (called when slot is evicted).
    pub fn clear_slot(&mut self, slot_id: usize) {
        self.slot_metadata.remove(&slot_id);
    }

    /// Reset all state (useful for starting new generation).
    pub fn reset(&mut self) {
        self.slot_metadata.clear();
        self.current_step = 0;
    }

    /// Get token metadata by slot ID (for span-level attention aggregation).
    ///
    /// Used by the segmented eviction policy to aggregate per-token attention
    /// scores into per-span relevance scores.
    pub fn get_token_metadata(&self, slot_id: usize) -> Option<&TokenMetadata> {
        self.slot_metadata.get(&slot_id)
    }

    /// Get all token metadata (for iteration by external policies).
    pub fn all_token_metadata(&self) -> &HashMap<usize, TokenMetadata> {
        &self.slot_metadata
    }
}

impl EvictionPolicy for H2OPolicy {
    fn compute_eviction_scores(
        &self,
        cache_size: usize,
        num_used: usize,
        cache_positions: &HashMap<usize, usize>,
    ) -> Vec<(usize, f32)> {
        self.compute_eviction_scores(cache_size, num_used, cache_positions)
    }

    fn select_eviction_candidate(
        &self,
        cache_size: usize,
        num_used: usize,
        cache_positions: &HashMap<usize, usize>,
    ) -> Option<usize> {
        self.select_eviction_candidate(cache_size, num_used, cache_positions)
    }

    fn name(&self) -> &str {
        "H2O"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_metadata_average() {
        let mut metadata = TokenMetadata::new(0);
        assert_eq!(metadata.average_attention(), 0.0);

        metadata.cumulative_attention = 5.0;
        metadata.steps_present = 2;
        assert_eq!(metadata.average_attention(), 2.5);
    }

    #[test]
    fn test_h2o_protects_recent_tokens() {
        let config = H2OConfig {
            enabled: true,
            num_recent_to_keep: 4,
            decay_factor: 1.0,
        };

        let policy = H2OPolicy::new(config);

        // Create cache with positions 0..10
        let mut cache_positions = HashMap::new();
        for i in 0..10 {
            cache_positions.insert(i, i);
        }

        let scores = policy.compute_eviction_scores(10, 10, &cache_positions);

        // Positions 7, 8, 9 should have NEG_INFINITY (recent tokens)
        // Positions 0..6 should have INFINITY (no attention data yet)
        let recent_scores: Vec<_> = scores
            .iter()
            .filter(|(slot, _)| *slot >= 7)
            .map(|(_, score)| *score)
            .collect();

        assert!(
            recent_scores
                .iter()
                .all(|&s| s.is_infinite() && s.is_sign_negative())
        );
    }

    #[test]
    fn test_h2o_evicts_low_attention() {
        let config = H2OConfig {
            enabled: true,
            num_recent_to_keep: 2,
            decay_factor: 1.0,
        };

        let mut policy = H2OPolicy::new(config);

        // Create cache with 5 positions
        let mut cache_positions = HashMap::new();
        for i in 0..5 {
            cache_positions.insert(i, i);
        }

        // Simulate attention: position 1 gets lots of attention, position 0 gets none
        let attention_weights = vec![
            vec![0.1, 0.5, 0.2, 0.1, 0.1],   // Query 0 attends mostly to position 1
            vec![0.0, 0.7, 0.2, 0.05, 0.05], // Query 1 attends heavily to position 1
        ];

        policy.update_attention_scores(&attention_weights, &cache_positions);

        let candidate = policy.select_eviction_candidate(5, 5, &cache_positions);

        // Position 0 should be selected (lowest attention, not recent)
        assert_eq!(candidate, Some(0));
    }

    #[test]
    fn test_h2o_decay() {
        let config = H2OConfig {
            enabled: true,
            num_recent_to_keep: 1,
            decay_factor: 0.5, // Strong decay
        };

        let mut policy = H2OPolicy::new(config);

        let mut cache_positions = HashMap::new();
        cache_positions.insert(0, 0);
        cache_positions.insert(1, 1);

        // First step: position 0 gets attention
        let attention1 = vec![vec![1.0, 0.0]];
        policy.update_attention_scores(&attention1, &cache_positions);

        assert_eq!(
            policy.slot_metadata.get(&0).unwrap().cumulative_attention,
            1.0
        );

        // Second step: position 1 gets attention, position 0 decays
        let attention2 = vec![vec![0.0, 1.0]];
        policy.update_attention_scores(&attention2, &cache_positions);

        // Position 0 should have decayed: 1.0 * 0.5 = 0.5
        assert_eq!(
            policy.slot_metadata.get(&0).unwrap().cumulative_attention,
            0.5
        );
        // Position 1 should have: 0.0 * 0.5 + 1.0 = 1.0
        assert_eq!(
            policy.slot_metadata.get(&1).unwrap().cumulative_attention,
            1.0
        );
    }

    #[test]
    fn test_h2o_disabled() {
        let config = H2OConfig {
            enabled: false,
            ..Default::default()
        };

        let policy = H2OPolicy::new(config);

        let mut cache_positions = HashMap::new();
        cache_positions.insert(0, 0);

        let scores = policy.compute_eviction_scores(1, 1, &cache_positions);
        assert!(scores.is_empty());
    }
}
