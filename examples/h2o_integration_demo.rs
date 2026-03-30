use candlelight::core::{DType, Device};
/// Integration demonstration of H2O + Voting with ParallelCacheBuilder
///
/// This example shows how to use the H2O policy and voting system with
/// ParallelCacheBuilder for intelligent cache management.
///
/// Since we don't have actual attention weights from a model yet, we simulate
/// the attention patterns to demonstrate the tracking and scoring functionality.
use lightbulb::cache::{
    H2OConfig, H2OPolicy, ParallelCacheBuilder, RecencyPolicy, VotingAggregator,
};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== H2O + Voting Integration with ParallelCacheBuilder ===\n");

    let device = Device::Cpu;
    let batch_size = 4;
    let context = 10;

    // Create ParallelCacheBuilder
    let mut builder = ParallelCacheBuilder::new(batch_size, context, DType::F32, &device)?;

    println!("Created ParallelCacheBuilder:");
    println!("  Batch size: {}", batch_size);
    println!("  Context size: {}", context);
    println!();

    // Set up H2O policy
    let h2o_config = H2OConfig {
        enabled: true,
        num_recent_to_keep: 2,
        decay_factor: 0.95,
    };
    let h2o_policy = H2OPolicy::new(h2o_config.clone());
    builder.set_h2o_policy(Some(h2o_policy));

    println!("Configured H2O Policy:");
    println!("  Enabled: {}", h2o_config.enabled);
    println!(
        "  Recent tokens protected: {}",
        h2o_config.num_recent_to_keep
    );
    println!("  Decay factor: {}", h2o_config.decay_factor);
    println!();

    // Set up voting aggregator
    let h2o_for_voting = H2OPolicy::new(h2o_config);
    let recency_policy = RecencyPolicy::new(2);

    let aggregator = VotingAggregator::new()
        .add_policy(h2o_for_voting, 0.7) // 70% weight on attention
        .add_policy(recency_policy, 0.3); // 30% weight on recency

    builder.set_voting_aggregator(Some(aggregator));

    println!("Configured Voting Aggregator:");
    println!("  Policies: H2O (70%), Recency (30%)");
    println!();

    // Simulate processing tokens for multiple slots
    println!("--- Simulating Token Processing ---\n");

    // Slot 0: Process 5 tokens
    for pos in 0..5 {
        builder.set_position(0, pos);
    }
    println!("Slot 0: Processed 5 tokens (positions 0-4)");

    // Slot 1: Process 8 tokens
    for pos in 0..8 {
        builder.set_position(1, pos);
    }
    println!("Slot 1: Processed 8 tokens (positions 0-7)");

    // Slot 2: Process 3 tokens
    for pos in 0..3 {
        builder.set_position(2, pos);
    }
    println!("Slot 2: Processed 3 tokens (positions 0-2)");

    println!();

    // Simulate attention weights for the cache
    println!("--- Simulating Attention Patterns ---\n");

    // Create realistic attention patterns:
    // - High attention to positions 1 and 4 (important content)
    // - Low attention to positions 0, 2, 3 (filler/low value)
    let attention_weights = vec![
        vec![0.05, 0.30, 0.10, 0.08, 0.25, 0.12, 0.05, 0.03, 0.01, 0.01], // Query 0
        vec![0.03, 0.32, 0.08, 0.06, 0.28, 0.13, 0.06, 0.02, 0.01, 0.01], // Query 1
        vec![0.06, 0.28, 0.12, 0.10, 0.22, 0.11, 0.07, 0.02, 0.01, 0.01], // Query 2
    ];

    println!("Attention pattern:");
    println!("  High attention: positions 1, 4");
    println!("  Medium attention: positions 5");
    println!("  Low attention: positions 0, 2, 3, 6+");
    println!();

    // Update H2O with simulated attention
    builder.update_attention_scores(&attention_weights);
    println!("✓ Updated H2O policy with attention weights");
    println!();

    // Get eviction recommendations
    println!("--- Eviction Recommendations ---\n");

    if let Some(h2o) = builder.h2o_policy() {
        // Get current cache positions
        let positions = builder.positions();
        let cache_positions: HashMap<usize, usize> = (0..batch_size)
            .map(|slot| (slot, positions[slot]))
            .collect();

        // H2O-only recommendation
        let h2o_scores = h2o.compute_eviction_scores(context, batch_size, &cache_positions);
        println!("H2O Policy Scores (top 5 candidates for eviction):");
        for (i, (slot, score)) in h2o_scores.iter().take(5).enumerate() {
            if score.is_finite() {
                println!("  {}. Slot {}: score = {:.3}", i + 1, slot, score);
            } else if score.is_infinite() && score.is_sign_negative() {
                println!("  {}. Slot {}: PROTECTED (recent)", i + 1, slot);
            } else {
                println!("  {}. Slot {}: PRIORITY (no attention)", i + 1, slot);
            }
        }
        println!();
    }

    if let Some(aggregator) = builder.voting_aggregator() {
        let positions = builder.positions();
        let cache_positions: HashMap<usize, usize> = (0..batch_size)
            .map(|slot| (slot, positions[slot]))
            .collect();

        let aggregated_scores =
            aggregator.compute_aggregated_scores(context, batch_size, &cache_positions);
        println!("Voting Aggregator Scores (H2O 70% + Recency 30%):");
        for (i, (slot, score)) in aggregated_scores.iter().take(batch_size).enumerate() {
            if score.is_finite() {
                println!("  {}. Slot {}: score = {:.3}", i + 1, slot, score);
            } else {
                println!("  {}. Slot {}: PROTECTED", i + 1, slot);
            }
        }
        println!();

        let eviction_candidate =
            aggregator.select_eviction_candidate(context, batch_size, &cache_positions);
        if let Some(slot) = eviction_candidate {
            println!("✓ Voting recommendation: Evict slot {}", slot);
            println!(
                "  (Slot {} has lowest combined attention + oldest position)",
                slot
            );
        } else {
            println!("  No eviction candidate (all slots protected)");
        }
    }

    println!();
    println!("--- Summary ---\n");
    println!("Infrastructure complete:");
    println!("  ✓ H2O policy tracks cumulative attention per slot");
    println!("  ✓ Voting aggregator combines multiple policies");
    println!("  ✓ ParallelCacheBuilder maintains slot->position mapping");
    println!("  ✓ update_attention_scores() ready for real attention weights");
    println!();
    println!("Next steps:");
    println!("  1. Expose attention weights from custom_attention.rs");
    println!("  2. Call builder.update_attention_scores() after each forward pass");
    println!("  3. Use voting_aggregator.select_eviction_candidate() for cache management");
    println!("  4. Implement explicit eviction logic based on use case");

    Ok(())
}
