//! Demonstration of runtime slot pool adjustment
//!
//! This example shows how `SlotPoolMonitor` dynamically adjusts the slot pool
//! size based on actual workload patterns.
//!
//! Scenarios:
//! 1. **Startup**: Pool starts with hardware-calculated size
//! 2. **Load spike**: Queue builds → monitor recommends growing
//! 3. **Steady state**: Optimal utilization → no adjustment
//! 4. **Memory pressure**: Large contexts → monitor recommends shrinking
//!
//! Run: `cargo run --example runtime_slot_adjustment --release`

use lightbulb::engine::slot_monitor::{AdjustmentConfig, SlotPoolMonitor};
use lightbulb::engine::slot_pool::{Request, SlotPool, SlotState};
use lightbulb::hardware::batch_sizing::ModelMemoryProfile;

fn main() {
    println!("=== Runtime Slot Pool Adjustment Demo ===\n");

    // Simulated model (similar to 7B Llama)
    let model_profile = ModelMemoryProfile {
        weights_bytes: 6 * 1024 * 1024 * 1024, // 6GB
        num_layers: 32,
        hidden_size: 4096,
        num_kv_heads: 32,
        context_window: 512,
    };

    // Available system memory (simulate 16GB GPU)
    let available_memory = 16 * 1024 * 1024 * 1024u64;

    // Conservative config for demo (faster adjustments)
    let config = AdjustmentConfig {
        target_utilization: 0.7,
        shrink_threshold: 0.75, // Shrink earlier for demo
        grow_threshold: 0.5,
        max_adjustment_fraction: 0.2,  // Larger adjustments for demo
        adjustment_cooldown_secs: 5,   // Shorter cooldown
        min_samples_for_adjustment: 5, // Fewer samples needed
    };

    let mut monitor = SlotPoolMonitor::with_config(model_profile.clone(), 2, config);
    let mut pool = SlotPool::new(10); // Start with 10 slots

    println!("📊 Initial configuration:");
    println!("  - Max slots: {}", pool.max_slots());
    println!(
        "  - Available memory: {:.2} GB",
        available_memory as f64 / 1e9
    );
    println!("  - Target utilization: 70%");
    println!("  - Shrink threshold: 75%");
    println!("  - Grow threshold: 50%\n");

    // === Scenario 1: Load spike (should grow) ===
    println!("--- Scenario 1: Load Spike ---");
    println!("Simulating 15 incoming requests with 10-slot pool...\n");

    for i in 0..15 {
        let request = Request {
            id: format!("req_{}", i),
            prompt_tokens: vec![1; 50], // 50-token prompts
            max_new_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
        };
        pool.submit_request(request);
    }

    // Simulate multiple batches with high queue
    for batch_num in 0..10 {
        let positions = collect_active_positions(&pool);
        let pending_count = pool.pending_count();

        monitor.record_batch(positions.len(), pending_count, &positions);

        if batch_num % 3 == 0 {
            let stats = monitor.get_statistics();
            println!(
                "Batch {}: {} active, {} pending, {:.2} MB avg memory",
                batch_num,
                positions.len(),
                pending_count,
                stats.avg_memory_bytes as f64 / 1e6
            );
        }

        // Check for adjustment recommendation
        if let Some(new_size) = monitor.should_adjust(pool.max_slots(), available_memory) {
            println!(
                "\n✅ Monitor recommends: {} → {} slots",
                pool.max_slots(),
                new_size
            );

            // Wait for safe point (no active requests for demo)
            simulate_batch_completion(&mut pool);

            if pool.can_resize() {
                pool.resize_to(new_size).unwrap();
                monitor.record_adjustment();
                println!("✓ Resized to {} slots\n", new_size);
            }
        }

        simulate_token_generation(&mut pool, 10);
    }

    // === Scenario 2: Steady state ===
    println!("\n--- Scenario 2: Steady State ---");
    println!("Running with optimal load...\n");

    // Submit moderate load
    for i in 15..25 {
        let request = Request {
            id: format!("req_{}", i),
            prompt_tokens: vec![1; 50],
            max_new_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
        };
        pool.submit_request(request);
    }

    for batch_num in 0..8 {
        let positions = collect_active_positions(&pool);
        let pending_count = pool.pending_count();

        monitor.record_batch(positions.len(), pending_count, &positions);

        if batch_num % 3 == 0 {
            let stats = monitor.get_statistics();
            println!(
                "Batch {}: {} active, {} pending, {:.2} MB avg memory",
                batch_num,
                positions.len(),
                pending_count,
                stats.avg_memory_bytes as f64 / 1e6
            );
        }

        // Check for adjustment
        if let Some(new_size) = monitor.should_adjust(pool.max_slots(), available_memory) {
            println!(
                "\n✅ Monitor recommends: {} → {} slots",
                pool.max_slots(),
                new_size
            );
        } else if batch_num == 0 {
            println!("✓ No adjustment needed (optimal utilization)\n");
        }

        simulate_token_generation(&mut pool, 10);
    }

    // === Scenario 3: Memory pressure (should shrink) ===
    println!("\n--- Scenario 3: Memory Pressure ---");
    println!("Simulating large contexts approaching memory limits...\n");

    // Simulate long-running requests with large contexts
    for batch_num in 0..15 {
        let positions = collect_active_positions(&pool);
        let pending_count = pool.pending_count();

        // Simulate growing contexts (approaching max context window)
        let large_positions: Vec<usize> = positions.iter().map(|&p| p.max(400)).collect();

        monitor.record_batch(large_positions.len(), pending_count, &large_positions);

        if batch_num % 5 == 0 {
            let stats = monitor.get_statistics();
            let utilization = stats.avg_memory_bytes as f64 / available_memory as f64;
            println!(
                "Batch {}: {} active, {:.2} MB avg memory ({:.1}% utilization)",
                batch_num,
                large_positions.len(),
                stats.avg_memory_bytes as f64 / 1e6,
                utilization * 100.0
            );
        }

        // Check for adjustment
        if let Some(new_size) = monitor.should_adjust(pool.max_slots(), available_memory) {
            println!("\n⚠️  High memory pressure detected!");
            println!(
                "✅ Monitor recommends: {} → {} slots",
                pool.max_slots(),
                new_size
            );

            simulate_batch_completion(&mut pool);

            if pool.can_resize() {
                pool.resize_to(new_size).unwrap();
                monitor.record_adjustment();
                println!("✓ Resized to {} slots to reduce memory usage\n", new_size);
            }
        }

        simulate_token_generation(&mut pool, 5);
    }

    // Final summary
    println!("\n=== Summary ===");
    let stats = monitor.get_statistics();
    println!("Final slot count: {}", pool.max_slots());
    println!(
        "Peak memory usage: {:.2} MB",
        stats.peak_memory_bytes as f64 / 1e6
    );
    println!("Avg active slots: {:.1}", stats.avg_active_slots);
    println!("\n✅ Runtime adjustment keeps memory usage bounded while maximizing throughput!");
}

/// Collect current token positions from active slots
fn collect_active_positions(pool: &SlotPool) -> Vec<usize> {
    let mut positions = Vec::new();
    for slot_id in 0..pool.max_slots() {
        if let Some(slot) = pool.get_slot_state(slot_id) {
            if let SlotState::Active {
                current_position, ..
            } = slot
            {
                positions.push(*current_position);
            }
        }
    }
    positions
}

/// Simulate token generation (advance positions)
fn simulate_token_generation(pool: &mut SlotPool, tokens: usize) {
    for slot_id in 0..pool.max_slots() {
        if let Some(SlotState::Active {
            current_position,
            max_new_tokens,
            tokens_generated,
            ..
        }) = pool.get_slot_state(slot_id)
        {
            let new_position = current_position + tokens;
            let new_tokens_generated = tokens_generated + tokens;

            // Mark finished if max tokens reached
            if new_tokens_generated >= *max_new_tokens {
                let _ = pool.mark_finished(slot_id);
            } else {
                // Would update position in real implementation
                // (simplified for demo)
            }
        }
    }

    // Free finished slots and allocate pending
    pool.free_finished_slots();
    pool.allocate_pending_requests();
}

/// Simulate batch completion (finish all active requests for safe resize)
fn simulate_batch_completion(pool: &mut SlotPool) {
    for slot_id in 0..pool.max_slots() {
        if matches!(pool.get_slot_state(slot_id), Some(SlotState::Active { .. })) {
            let _ = pool.mark_finished(slot_id);
        }
    }
    pool.free_finished_slots();
}
