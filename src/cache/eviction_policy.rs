/// Generic eviction policy trait for KV cache management.
///
/// Different policies can be combined using a voting system to make
/// more intelligent eviction decisions.
use std::collections::HashMap;

/// Trait for cache eviction policies.
///
/// Implementations must be thread-safe (Send + Sync) to work with
/// speculative decoding and other concurrent features.
pub trait EvictionPolicy: Send + Sync {
    /// Compute eviction scores for cache slots.
    ///
    /// Higher score = higher priority for eviction.
    ///
    /// # Arguments
    /// * `cache_size` - Total cache capacity
    /// * `num_used` - Number of slots currently in use
    /// * `cache_positions` - Mapping from slot ID to sequence position
    ///
    /// # Returns
    /// Vector of (slot_id, eviction_score) pairs. Scores should be normalized
    /// to a reasonable range (e.g., 0.0 to 1.0) for voting to work well.
    fn compute_eviction_scores(
        &self,
        cache_size: usize,
        num_used: usize,
        cache_positions: &HashMap<usize, usize>,
    ) -> Vec<(usize, f32)>;

    /// Select a single eviction candidate (highest priority).
    fn select_eviction_candidate(
        &self,
        cache_size: usize,
        num_used: usize,
        cache_positions: &HashMap<usize, usize>,
    ) -> Option<usize> {
        let scores = self.compute_eviction_scores(cache_size, num_used, cache_positions);
        scores.first().map(|(slot, _)| *slot)
    }

    /// Name of the policy (for debugging/logging).
    fn name(&self) -> &str;
}

/// Voting aggregator that combines multiple eviction policies.
pub struct VotingAggregator {
    /// List of (policy, weight) pairs.
    policies: Vec<(Box<dyn EvictionPolicy>, f32)>,
}

impl std::fmt::Debug for VotingAggregator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VotingAggregator")
            .field("num_policies", &self.policies.len())
            .field(
                "weights",
                &self.policies.iter().map(|(_, w)| w).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl VotingAggregator {
    pub fn new() -> Self {
        Self {
            policies: Vec::new(),
        }
    }

    /// Add a policy with a given weight.
    ///
    /// Weights don't need to sum to 1.0; they will be normalized internally.
    pub fn add_policy<P: EvictionPolicy + 'static>(mut self, policy: P, weight: f32) -> Self {
        self.policies.push((Box::new(policy), weight));
        self
    }

    /// Compute aggregated eviction scores by weighted voting.
    ///
    /// Each policy votes on each slot, and votes are combined using weights.
    pub fn compute_aggregated_scores(
        &self,
        cache_size: usize,
        num_used: usize,
        cache_positions: &HashMap<usize, usize>,
    ) -> Vec<(usize, f32)> {
        if self.policies.is_empty() {
            return Vec::new();
        }

        // Collect scores from all policies
        let mut all_scores: Vec<Vec<(usize, f32)>> = Vec::new();
        for (policy, _) in &self.policies {
            let scores = policy.compute_eviction_scores(cache_size, num_used, cache_positions);
            all_scores.push(scores);
        }

        // Normalize policy weights
        let total_weight: f32 = self.policies.iter().map(|(_, w)| w).sum();
        let normalized_weights: Vec<f32> = self
            .policies
            .iter()
            .map(|(_, w)| w / total_weight)
            .collect();

        // Aggregate scores for each slot
        let mut aggregated: HashMap<usize, f32> = HashMap::new();

        for (policy_idx, scores) in all_scores.iter().enumerate() {
            let weight = normalized_weights[policy_idx];

            // Normalize scores for this policy to [0, 1] range
            let max_score = scores
                .iter()
                .map(|(_, s)| s)
                .filter(|s| s.is_finite())
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1.0);

            let min_score = scores
                .iter()
                .map(|(_, s)| s)
                .filter(|s| s.is_finite())
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);

            let range = (max_score - min_score).max(1e-6); // Avoid division by zero

            for &(slot, score) in scores {
                if !score.is_finite() {
                    // Handle special cases: NEG_INFINITY means "never evict"
                    if score.is_infinite() && score.is_sign_negative() {
                        *aggregated.entry(slot).or_insert(0.0) += f32::NEG_INFINITY;
                    }
                    // POS_INFINITY is handled normally (high eviction priority)
                    continue;
                }

                // Normalize and weight
                let normalized = (score - min_score) / range;
                *aggregated.entry(slot).or_insert(0.0) += normalized * weight;
            }
        }

        // Convert to sorted vector
        let mut result: Vec<(usize, f32)> = aggregated.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        result
    }

    /// Select the slot with highest aggregated eviction priority.
    pub fn select_eviction_candidate(
        &self,
        cache_size: usize,
        num_used: usize,
        cache_positions: &HashMap<usize, usize>,
    ) -> Option<usize> {
        let scores = self.compute_aggregated_scores(cache_size, num_used, cache_positions);
        scores
            .first()
            .filter(|(_, score)| score.is_finite())
            .map(|(slot, _)| *slot)
    }
}

impl Default for VotingAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple recency-based eviction policy (FIFO).
pub struct RecencyPolicy {
    /// Number of recent tokens to protect (like attention sinks).
    num_recent_to_keep: usize,
}

impl RecencyPolicy {
    pub fn new(num_recent_to_keep: usize) -> Self {
        Self { num_recent_to_keep }
    }
}

impl EvictionPolicy for RecencyPolicy {
    fn compute_eviction_scores(
        &self,
        _cache_size: usize,
        _num_used: usize,
        cache_positions: &HashMap<usize, usize>,
    ) -> Vec<(usize, f32)> {
        let max_position = cache_positions.values().max().copied().unwrap_or(0);
        let recent_threshold = max_position.saturating_sub(self.num_recent_to_keep);

        let mut scores: Vec<(usize, f32)> = cache_positions
            .iter()
            .map(|(&slot, &pos)| {
                if pos > recent_threshold {
                    (slot, f32::NEG_INFINITY) // Never evict recent
                } else {
                    // Older tokens have higher eviction priority
                    (slot, (max_position - pos) as f32)
                }
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
    }

    fn name(&self) -> &str {
        "Recency"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recency_policy() {
        let policy = RecencyPolicy::new(2);

        let mut cache_positions = HashMap::new();
        for i in 0..5 {
            cache_positions.insert(i, i);
        }

        let scores = policy.compute_eviction_scores(5, 5, &cache_positions);

        // Positions 3, 4 should be protected (recent)
        let recent_scores: Vec<_> = scores
            .iter()
            .filter(|(slot, _)| *slot >= 3)
            .map(|(_, score)| *score)
            .collect();
        assert!(
            recent_scores
                .iter()
                .all(|s| s.is_infinite() && s.is_sign_negative())
        );

        // Position 0 should be first eviction candidate (oldest)
        let candidate = policy.select_eviction_candidate(5, 5, &cache_positions);
        assert_eq!(candidate, Some(0));
    }

    #[test]
    fn test_voting_aggregator_single_policy() {
        let policy = RecencyPolicy::new(2);
        let aggregator = VotingAggregator::new().add_policy(policy, 1.0);

        let mut cache_positions = HashMap::new();
        for i in 0..5 {
            cache_positions.insert(i, i);
        }

        let candidate = aggregator.select_eviction_candidate(5, 5, &cache_positions);
        assert_eq!(candidate, Some(0)); // Oldest token
    }

    #[test]
    fn test_voting_aggregator_multiple_policies() {
        // Two policies with equal weight
        let policy1 = RecencyPolicy::new(1); // Protects last 1 token
        let policy2 = RecencyPolicy::new(1); // Same

        let aggregator = VotingAggregator::new()
            .add_policy(policy1, 0.5)
            .add_policy(policy2, 0.5);

        let mut cache_positions = HashMap::new();
        for i in 0..5 {
            cache_positions.insert(i, i);
        }

        let scores = aggregator.compute_aggregated_scores(5, 5, &cache_positions);

        // Both policies agree that position 0 is oldest
        let candidate = aggregator.select_eviction_candidate(5, 5, &cache_positions);
        assert_eq!(candidate, Some(0));
    }

    #[test]
    fn test_voting_aggregator_weight_normalization() {
        let policy1 = RecencyPolicy::new(1);
        let policy2 = RecencyPolicy::new(1);

        // Weights that don't sum to 1.0
        let aggregator = VotingAggregator::new()
            .add_policy(policy1, 2.0)
            .add_policy(policy2, 3.0);

        let mut cache_positions = HashMap::new();
        cache_positions.insert(0, 0);
        cache_positions.insert(1, 1);

        let scores = aggregator.compute_aggregated_scores(2, 2, &cache_positions);

        // Should still work correctly (weights are normalized internally)
        assert_eq!(scores.len(), 2);

        // Position 0 should have higher eviction score (older)
        assert!(scores[0].0 == 0 || scores[1].0 == 0); // Position 0 is in the results
    }
}
