use lightbulb::cache::{EvictionPolicy, H2OConfig, H2OPolicy, RecencyPolicy, VotingAggregator};
/// Demonstration of the voting eviction system combining multiple policies.
///
/// This example shows how H2O, StreamingLLM, and Recency policies can be
/// combined using weighted voting to make more intelligent eviction decisions.
use std::collections::HashMap;

fn main() {
    println!("=== Multi-Policy Voting System Demo ===\n");

    // Create a simple scenario with 10 cache slots
    let cache_size = 10;
    let mut cache_positions: HashMap<usize, usize> = HashMap::new();
    for i in 0..cache_size {
        cache_positions.insert(i, i);
    }

    println!("Cache state: 10 slots, positions 0-9\n");

    // 1. Test individual policies

    println!("--- Individual Policy Decisions ---\n");

    // H2O Policy
    println!("1. H2O Policy (attention-based):");
    let h2o_config = H2OConfig {
        enabled: true,
        num_recent_to_keep: 2, // Protect last 2 tokens
        decay_factor: 0.95,
    };
    let mut h2o_policy = H2OPolicy::new(h2o_config);

    // Simulate attention: positions 2, 5, 7 get high attention
    let attention_weights = vec![
        vec![0.05, 0.05, 0.3, 0.1, 0.05, 0.25, 0.05, 0.1, 0.03, 0.02],
        vec![0.03, 0.04, 0.28, 0.08, 0.06, 0.27, 0.06, 0.12, 0.04, 0.02],
    ];
    h2o_policy.update_attention_scores(&attention_weights, &cache_positions);

    let h2o_candidate =
        h2o_policy.select_eviction_candidate(cache_size, cache_size, &cache_positions);
    println!("   High attention: positions 2, 5, 7");
    println!("   Protected (recent): positions 8, 9");
    println!("   H2O recommends evicting: position {:?}", h2o_candidate);
    println!("   (Low attention positions preferred)\n");

    // Recency Policy
    println!("2. Recency Policy (FIFO):");
    let recency_policy = RecencyPolicy::new(2); // Protect last 2 tokens
    let recency_candidate =
        recency_policy.select_eviction_candidate(cache_size, cache_size, &cache_positions);
    println!("   Protected (recent): positions 8, 9");
    println!(
        "   Recency recommends evicting: position {:?}",
        recency_candidate
    );
    println!("   (Oldest non-protected position)\n");

    // 2. Test voting aggregator

    println!("--- Weighted Voting System ---\n");

    // Scenario 1: H2O-dominated (70% H2O, 30% Recency)
    println!("Scenario 1: H2O-dominated weighting (70% H2O, 30% Recency)");
    let h2o_config = H2OConfig {
        enabled: true,
        num_recent_to_keep: 2,
        decay_factor: 0.95,
    };
    let mut h2o_policy_1 = H2OPolicy::new(h2o_config);
    h2o_policy_1.update_attention_scores(&attention_weights, &cache_positions);

    let recency_policy_1 = RecencyPolicy::new(2);

    let aggregator_1 = VotingAggregator::new()
        .add_policy(h2o_policy_1, 0.7)
        .add_policy(recency_policy_1, 0.3);

    let candidate_1 =
        aggregator_1.select_eviction_candidate(cache_size, cache_size, &cache_positions);
    println!("   Voting result: evict position {:?}", candidate_1);
    println!("   (H2O's attention-based decision is prioritized)\n");

    // Scenario 2: Balanced weighting (50% H2O, 50% Recency)
    println!("Scenario 2: Balanced weighting (50% H2O, 50% Recency)");
    let h2o_config = H2OConfig {
        enabled: true,
        num_recent_to_keep: 2,
        decay_factor: 0.95,
    };
    let mut h2o_policy_2 = H2OPolicy::new(h2o_config);
    h2o_policy_2.update_attention_scores(&attention_weights, &cache_positions);

    let recency_policy_2 = RecencyPolicy::new(2);

    let aggregator_2 = VotingAggregator::new()
        .add_policy(h2o_policy_2, 0.5)
        .add_policy(recency_policy_2, 0.5);

    let candidate_2 =
        aggregator_2.select_eviction_candidate(cache_size, cache_size, &cache_positions);
    let scores_2 = aggregator_2.compute_aggregated_scores(cache_size, cache_size, &cache_positions);

    println!("   Voting result: evict position {:?}", candidate_2);
    println!("   (Compromise between attention and age)\n");

    println!("   Aggregated eviction scores (top 5):");
    for (i, (slot, score)) in scores_2.iter().take(5).enumerate() {
        if score.is_finite() {
            println!("     {}. Position {}: score = {:.3}", i + 1, slot, score);
        }
    }
    println!();

    // Scenario 3: Recency-dominated (30% H2O, 70% Recency)
    println!("Scenario 3: Recency-dominated weighting (30% H2O, 70% Recency)");
    let h2o_config = H2OConfig {
        enabled: true,
        num_recent_to_keep: 2,
        decay_factor: 0.95,
    };
    let mut h2o_policy_3 = H2OPolicy::new(h2o_config);
    h2o_policy_3.update_attention_scores(&attention_weights, &cache_positions);

    let recency_policy_3 = RecencyPolicy::new(2);

    let aggregator_3 = VotingAggregator::new()
        .add_policy(h2o_policy_3, 0.3)
        .add_policy(recency_policy_3, 0.7);

    let candidate_3 =
        aggregator_3.select_eviction_candidate(cache_size, cache_size, &cache_positions);
    println!("   Voting result: evict position {:?}", candidate_3);
    println!("   (Recency's age-based decision is prioritized)\n");

    // 3. Show detailed comparison

    println!("--- Summary Comparison ---\n");
    println!("Attention pattern: High=[2,5,7], Medium=[3,6], Low=[0,1,4,8,9]");
    println!("Protected (recent): [8,9]");
    println!();
    println!("Policy Recommendations:");
    println!(
        "  H2O alone:           Position {:?} (prioritizes low attention)",
        h2o_candidate
    );
    println!(
        "  Recency alone:       Position {:?} (prioritizes oldest)",
        recency_candidate
    );
    println!("  Voting (70% H2O):    Position {:?}", candidate_1);
    println!("  Voting (50% each):   Position {:?}", candidate_2);
    println!("  Voting (70% Recency): Position {:?}", candidate_3);
    println!();
    println!("Conclusion: Voting allows flexible policy blending based on use case.");
    println!("- Memory-constrained: favor H2O (keep important tokens)");
    println!("- Streaming workload: favor Recency (maintain temporal locality)");
    println!("- General purpose: balanced weights (compromise)");
}
