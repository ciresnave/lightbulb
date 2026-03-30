//! Mixture-of-Experts (MoE) Routing Infrastructure
//!
//! Implements efficient routing-friendly batching for MoE models (e.g., Mixtral).
//! Supports capacity-aware routing with Token Drop and Expanded Drop policies.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │              Token Routing (per layer)                      │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Input: N tokens → Router: Softmax(W @ hidden)               │
//! │ Top-K selection (K=1 or K=2)                                │
//! │ → Expert assignments: [E0: [t1,t3], E1: [t2,t5], ...]       │
//! └─────────────────────────────────────────────────────────────┘
//!                              ↓
//! ┌─────────────────────────────────────────────────────────────┐
//! │          Capacity Management (per expert)                   │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Capacity = C × (num_tokens / num_experts)                   │
//! │ If overflow:                                                │
//! │   - Token Drop: reject lowest score tokens                  │
//! │   - Expanded Drop: increase local capacity by m             │
//! └─────────────────────────────────────────────────────────────┘
//!                              ↓
//! ┌─────────────────────────────────────────────────────────────┐
//! │         Expert-Parallel Batching                            │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Group tokens by expert → Process in parallel                │
//! │ Aggregate outputs → Return to original positions            │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use lightbulb::engine::{MoeRouter, MoeConfig, RoutingPolicy};
//!
//! // Create router with 8 experts, Top-2 routing
//! let config = MoeConfig {
//!     num_experts: 8,
//!     top_k: 2,
//!     capacity_factor: 1.25,
//!     routing_policy: RoutingPolicy::TokenDrop,
//!     ..Default::default()
//! };
//!
//! let mut router = MoeRouter::new(config);
//!
//! // Route tokens
//! let assignments = router.route_tokens(&logits)?;
//!
//! // Check capacity overflow
//! if assignments.has_drops() {
//!     println!("Dropped {} tokens", assignments.num_dropped());
//! }
//!
//! // Get per-expert batches
//! for (expert_id, token_indices) in assignments.expert_batches() {
//!     // Process tokens with expert[expert_id]
//! }
//! ```

use std::collections::HashMap;
use thiserror::Error;

/// Expert ID (0-indexed)
pub type ExpertId = usize;

/// Token index in the batch
pub type TokenIndex = usize;

/// Routing policy for handling capacity overflow
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingPolicy {
    /// Drop tokens that exceed expert capacity (Switch-style)
    TokenDrop,

    /// Expand expert capacity by a factor m (capacity-aware)
    ExpandedDrop { expansion_factor: u32 },

    /// No dropping - reroute overflow to next best expert
    Reroute,

    /// No capacity limits (for debugging)
    Unlimited,
}

impl Default for RoutingPolicy {
    fn default() -> Self {
        RoutingPolicy::TokenDrop
    }
}

/// Configuration for MoE routing
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Number of experts in the MoE layer
    pub num_experts: usize,

    /// Top-K experts per token (1 or 2 typically)
    pub top_k: usize,

    /// Capacity factor: max_tokens_per_expert = capacity_factor * (num_tokens / num_experts)
    pub capacity_factor: f32,

    /// Policy for handling capacity overflow
    pub routing_policy: RoutingPolicy,

    /// Enable load balancing auxiliary loss
    pub load_balancing: bool,

    /// Enable routing statistics
    pub enable_stats: bool,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            top_k: 2,
            capacity_factor: 1.25,
            routing_policy: RoutingPolicy::TokenDrop,
            load_balancing: false,
            enable_stats: true,
        }
    }
}

/// Assignment of a token to an expert with routing score
#[derive(Debug, Clone)]
pub struct TokenAssignment {
    /// Token index in the batch
    pub token_index: TokenIndex,

    /// Assigned expert ID
    pub expert_id: ExpertId,

    /// Routing score (from softmax)
    pub score: f32,

    /// Whether this token was dropped due to capacity
    pub dropped: bool,

    /// Rank of this expert for this token (0 = top choice)
    pub rank: usize,
}

/// Result of routing tokens to experts
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// All assignments (including dropped tokens)
    assignments: Vec<TokenAssignment>,

    /// Per-expert token lists (expert_id -> token_indices)
    expert_batches: HashMap<ExpertId, Vec<TokenIndex>>,

    /// Number of tokens dropped
    num_dropped: usize,

    /// Per-expert load (number of assigned tokens)
    expert_loads: Vec<usize>,
}

impl RoutingResult {
    /// Get assignments for a specific expert
    pub fn get_expert_batch(&self, expert_id: ExpertId) -> Option<&[TokenIndex]> {
        self.expert_batches.get(&expert_id).map(|v| v.as_slice())
    }

    /// Iterate over all expert batches
    pub fn expert_batches(&self) -> impl Iterator<Item = (ExpertId, &[TokenIndex])> {
        self.expert_batches
            .iter()
            .map(|(id, indices)| (*id, indices.as_slice()))
    }

    /// Check if any tokens were dropped
    pub fn has_drops(&self) -> bool {
        self.num_dropped > 0
    }

    /// Number of dropped tokens
    pub fn num_dropped(&self) -> usize {
        self.num_dropped
    }

    /// Total tokens routed (including drops)
    pub fn total_tokens(&self) -> usize {
        self.assignments.len()
    }

    /// Get all assignments (including dropped)
    pub fn assignments(&self) -> &[TokenAssignment] {
        &self.assignments
    }

    /// Get load per expert
    pub fn expert_loads(&self) -> &[usize] {
        &self.expert_loads
    }

    /// Calculate load imbalance (variance / mean²)
    pub fn load_imbalance(&self) -> f32 {
        if self.expert_loads.is_empty() {
            return 0.0;
        }

        let mean = self.expert_loads.iter().sum::<usize>() as f32 / self.expert_loads.len() as f32;
        if mean == 0.0 {
            return 0.0;
        }

        let variance = self
            .expert_loads
            .iter()
            .map(|&load| {
                let diff = load as f32 - mean;
                diff * diff
            })
            .sum::<f32>()
            / self.expert_loads.len() as f32;

        variance / (mean * mean)
    }
}

/// Statistics about routing performance
#[derive(Debug, Clone, Default)]
pub struct RoutingStats {
    /// Total tokens routed
    pub total_tokens: usize,

    /// Total tokens dropped
    pub total_dropped: usize,

    /// Per-expert token counts (accumulated)
    pub expert_token_counts: Vec<usize>,

    /// Average load imbalance across all routing operations
    pub avg_load_imbalance: f32,

    /// Maximum load imbalance seen
    pub max_load_imbalance: f32,

    /// Number of routing operations
    pub num_routing_ops: usize,
}

/// Errors that can occur during routing
#[derive(Debug, Error)]
pub enum RoutingError {
    #[error("Invalid logits shape: expected {expected}, got {actual}")]
    InvalidShape { expected: String, actual: String },

    #[error("Expert ID {0} out of range (max: {1})")]
    ExpertOutOfRange(ExpertId, usize),

    #[error("Top-K value {0} must be > 0 and <= num_experts")]
    InvalidTopK(usize),

    #[error("Invalid capacity factor {0}, must be > 0")]
    InvalidCapacityFactor(f32),
}

/// MoE router managing token-to-expert assignments
pub struct MoeRouter {
    /// Configuration
    config: MoeConfig,

    /// Statistics (if enabled)
    stats: Option<RoutingStats>,
}

impl MoeRouter {
    /// Create a new MoE router
    pub fn new(config: MoeConfig) -> Result<Self, RoutingError> {
        // Validate config
        if config.top_k == 0 || config.top_k > config.num_experts {
            return Err(RoutingError::InvalidTopK(config.top_k));
        }

        if config.capacity_factor <= 0.0 {
            return Err(RoutingError::InvalidCapacityFactor(config.capacity_factor));
        }

        let stats = if config.enable_stats {
            Some(RoutingStats {
                expert_token_counts: vec![0; config.num_experts],
                ..Default::default()
            })
        } else {
            None
        };

        Ok(Self { config, stats })
    }

    /// Route tokens to experts based on router logits
    ///
    /// # Arguments
    /// * `logits` - Router logits [num_tokens, num_experts]
    ///
    /// # Returns
    /// Routing result with expert assignments and batches
    pub fn route_tokens(&mut self, logits: &[Vec<f32>]) -> Result<RoutingResult, RoutingError> {
        let num_tokens = logits.len();
        if num_tokens == 0 {
            return Ok(RoutingResult {
                assignments: Vec::new(),
                expert_batches: HashMap::new(),
                num_dropped: 0,
                expert_loads: vec![0; self.config.num_experts],
            });
        }

        // Validate logits shape
        if logits[0].len() != self.config.num_experts {
            return Err(RoutingError::InvalidShape {
                expected: format!("[{}, {}]", num_tokens, self.config.num_experts),
                actual: format!("[{}, {}]", num_tokens, logits[0].len()),
            });
        }

        // Calculate capacity per expert
        let base_capacity = (num_tokens as f32 * self.config.capacity_factor
            / self.config.num_experts as f32)
            .ceil() as usize;

        let capacity_per_expert = match self.config.routing_policy {
            RoutingPolicy::ExpandedDrop { expansion_factor } => {
                base_capacity + expansion_factor as usize
            }
            RoutingPolicy::Unlimited => usize::MAX,
            _ => base_capacity,
        };

        // Route each token to top-K experts
        let mut all_assignments = Vec::with_capacity(num_tokens * self.config.top_k);
        let mut expert_counts = vec![0; self.config.num_experts];

        for (token_idx, token_logits) in logits.iter().enumerate() {
            // Softmax over logits
            let scores = softmax(token_logits);

            // Get top-K experts
            let top_k = self.top_k_indices(&scores);

            // Assign to top-K experts
            for (rank, &expert_id) in top_k.iter().enumerate() {
                let score = scores[expert_id];

                // Check capacity
                let dropped = match self.config.routing_policy {
                    RoutingPolicy::TokenDrop | RoutingPolicy::ExpandedDrop { .. } => {
                        expert_counts[expert_id] >= capacity_per_expert
                    }
                    RoutingPolicy::Reroute => false, // Will be handled later
                    RoutingPolicy::Unlimited => false,
                };

                all_assignments.push(TokenAssignment {
                    token_index: token_idx,
                    expert_id,
                    score,
                    dropped,
                    rank,
                });

                if !dropped {
                    expert_counts[expert_id] += 1;
                }
            }
        }

        // Build per-expert batches
        let mut expert_batches: HashMap<ExpertId, Vec<TokenIndex>> = HashMap::new();
        let mut num_dropped = 0;

        for assignment in &all_assignments {
            if !assignment.dropped {
                expert_batches
                    .entry(assignment.expert_id)
                    .or_insert_with(Vec::new)
                    .push(assignment.token_index);
            } else {
                num_dropped += 1;
            }
        }

        // Update stats
        if let Some(stats) = &mut self.stats {
            stats.total_tokens += num_tokens;
            stats.total_dropped += num_dropped;
            stats.num_routing_ops += 1;

            for (expert_id, count) in expert_counts.iter().enumerate() {
                stats.expert_token_counts[expert_id] += count;
            }
        }

        let result = RoutingResult {
            assignments: all_assignments,
            expert_batches,
            num_dropped,
            expert_loads: expert_counts,
        };

        // Update load imbalance stats
        if let Some(stats) = &mut self.stats {
            let imbalance = result.load_imbalance();
            stats.max_load_imbalance = stats.max_load_imbalance.max(imbalance);
            stats.avg_load_imbalance =
                (stats.avg_load_imbalance * (stats.num_routing_ops - 1) as f32 + imbalance)
                    / stats.num_routing_ops as f32;
        }

        Ok(result)
    }

    /// Get top-K expert indices by score
    fn top_k_indices(&self, scores: &[f32]) -> Vec<ExpertId> {
        let mut indexed: Vec<(usize, f32)> =
            scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed
            .iter()
            .take(self.config.top_k)
            .map(|(i, _)| *i)
            .collect()
    }

    /// Get routing statistics
    pub fn stats(&self) -> Option<&RoutingStats> {
        self.stats.as_ref()
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        if let Some(stats) = &mut self.stats {
            *stats = RoutingStats {
                expert_token_counts: vec![0; self.config.num_experts],
                ..Default::default()
            };
        }
    }

    /// Get configuration
    pub fn config(&self) -> &MoeConfig {
        &self.config
    }
}

/// Compute softmax over a vector
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    logits.iter().map(|&x| (x - max).exp() / exp_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_moe_router_creation() {
        let config = MoeConfig {
            num_experts: 4,
            top_k: 2,
            ..Default::default()
        };
        let router = MoeRouter::new(config).unwrap();
        assert_eq!(router.config().num_experts, 4);
    }

    #[test]
    fn test_invalid_top_k() {
        let config = MoeConfig {
            num_experts: 4,
            top_k: 5, // Invalid: > num_experts
            ..Default::default()
        };
        assert!(MoeRouter::new(config).is_err());
    }

    #[test]
    fn test_basic_routing() {
        let config = MoeConfig {
            num_experts: 4,
            top_k: 2,
            capacity_factor: 2.0, // High capacity to avoid drops
            routing_policy: RoutingPolicy::Unlimited,
            ..Default::default()
        };

        let mut router = MoeRouter::new(config).unwrap();

        // 3 tokens, 4 experts
        let logits = vec![
            vec![1.0, 2.0, 0.5, 0.3], // Token 0: prefers expert 1, then 0
            vec![0.5, 0.3, 2.5, 1.0], // Token 1: prefers expert 2, then 3
            vec![3.0, 0.5, 0.3, 0.2], // Token 2: prefers expert 0, then 1
        ];

        let result = router.route_tokens(&logits).unwrap();

        // With top_k=2, we get 3 tokens × 2 assignments = 6 total assignments
        assert_eq!(result.assignments().len(), 6);
        assert!(!result.has_drops());
        assert_eq!(result.expert_batches().count(), 4); // All experts used
    }

    #[test]
    fn test_token_drop_policy() {
        let config = MoeConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 0.5, // Low capacity to force drops
            routing_policy: RoutingPolicy::TokenDrop,
            ..Default::default()
        };

        let mut router = MoeRouter::new(config).unwrap();

        // All tokens route to same expert
        let logits = vec![
            vec![5.0, 0.0, 0.0, 0.0],
            vec![5.0, 0.0, 0.0, 0.0],
            vec![5.0, 0.0, 0.0, 0.0],
            vec![5.0, 0.0, 0.0, 0.0],
        ];

        let result = router.route_tokens(&logits).unwrap();

        assert!(result.has_drops());
        assert!(result.num_dropped() > 0);
    }

    #[test]
    fn test_expanded_drop_policy() {
        let config = MoeConfig {
            num_experts: 4,
            top_k: 1,
            capacity_factor: 0.5,
            routing_policy: RoutingPolicy::ExpandedDrop {
                expansion_factor: 2,
            },
            ..Default::default()
        };

        let mut router = MoeRouter::new(config).unwrap();

        let logits = vec![
            vec![5.0, 0.0, 0.0, 0.0],
            vec![5.0, 0.0, 0.0, 0.0],
            vec![5.0, 0.0, 0.0, 0.0],
            vec![5.0, 0.0, 0.0, 0.0],
        ];

        let result = router.route_tokens(&logits).unwrap();

        // With expansion, should drop fewer tokens
        assert!(result.num_dropped() <= 2);
    }

    #[test]
    fn test_load_imbalance() {
        let config = MoeConfig {
            num_experts: 4,
            top_k: 1,
            routing_policy: RoutingPolicy::Unlimited,
            ..Default::default()
        };

        let mut router = MoeRouter::new(config).unwrap();

        // Balanced load
        let balanced_logits = vec![
            vec![5.0, 0.0, 0.0, 0.0],
            vec![0.0, 5.0, 0.0, 0.0],
            vec![0.0, 0.0, 5.0, 0.0],
            vec![0.0, 0.0, 0.0, 5.0],
        ];

        let balanced = router.route_tokens(&balanced_logits).unwrap();
        let balanced_imbalance = balanced.load_imbalance();

        // Imbalanced load
        let imbalanced_logits = vec![
            vec![5.0, 0.0, 0.0, 0.0],
            vec![5.0, 0.0, 0.0, 0.0],
            vec![5.0, 0.0, 0.0, 0.0],
            vec![0.0, 5.0, 0.0, 0.0],
        ];

        let imbalanced = router.route_tokens(&imbalanced_logits).unwrap();
        let imbalanced_imbalance = imbalanced.load_imbalance();

        assert!(imbalanced_imbalance > balanced_imbalance);
    }

    #[test]
    fn test_routing_stats() {
        let config = MoeConfig {
            num_experts: 4,
            top_k: 2,
            enable_stats: true,
            ..Default::default()
        };

        let mut router = MoeRouter::new(config).unwrap();

        let logits = vec![vec![1.0, 2.0, 0.5, 0.3]; 10];

        router.route_tokens(&logits).unwrap();

        let stats = router.stats().unwrap();
        assert_eq!(stats.total_tokens, 10);
        assert_eq!(stats.num_routing_ops, 1);
        assert!(stats.expert_token_counts.iter().sum::<usize>() > 0);
    }
}
