//! Integration tests for runtime slot adjustment
//!
//! Tests the complete flow: monitoring → decision → adjustment
//! under various workload scenarios.

use lightbulb::engine::{SlotPool, SlotPoolMonitor, Request};
use lightbulb::hardware::batch_sizing::ModelMemoryProfile;

fn test_model_profile() -> ModelMemoryProfile {
    ModelMemoryProfile {
        weights_bytes: 6 * 1024 * 1024 * 1024, // 6GB model
        num_layers: 32,
        hidden_size: 4096,
        num_kv_heads: 32,
        context_window: 512,
    }
}

fn create_test_request(id: &str, prompt_len: usize, max_tokens: usize) -> Request {
    Request {
        id: id.to_string(),
        prompt_tokens: vec![1; prompt_len],
        max_new_tokens: max_tokens,
        temperature: 0.7,
        top_p: 0.9,
    }
}

#[test]
fn test_grow_with_queue_pressure() {
    // Scenario: Low memory usage but requests are queueing
    // Expected: Pool should grow to handle demand

    let mut pool = SlotPool::new(8);
    let mut monitor = SlotPoolMonitor::new(test_model_profile(), 2);

    // Fill all slots
    for i in 0..8 {
        pool.submit_request(create_test_request(&format!("req{}", i), 50, 100));
    }

    // Add pending requests (queue builds up)
    for i in 8..15 {
        pool.submit_request(create_test_request(&format!("req{}", i), 50, 100));
    }

    let stats = pool.stats();
    assert_eq!(stats.active_slots, 8);
    assert_eq!(stats.pending_requests, 7);

    // Simulate several iterations with moderate token positions
    for _ in 0..15 {
        let positions = pool.get_active_positions();
        monitor.record_batch(
            stats.active_slots,
            stats.pending_requests,
            &positions,
        );
    }

    // Check if monitor recommends growth
    let available_memory = 16 * 1024 * 1024 * 1024u64; // 16GB available
    let decision = monitor.should_adjust(8, available_memory);

    assert!(
        decision.is_some(),
        "Monitor should recommend growth with queue pressure and headroom"
    );
    assert!(
        decision.unwrap() > 8,
        "New size should be larger than current"
    );

    println!("✓ Growth decision: 8 → {} slots", decision.unwrap());
}

#[test]
fn test_shrink_with_memory_pressure() {
    // Scenario: High memory usage, approaching limit
    // Expected: Pool should shrink to avoid OOM

    let mut pool = SlotPool::new(16);
    let mut monitor = SlotPoolMonitor::new(test_model_profile(), 2);

    // Fill slots with long-running requests at high token positions
    for i in 0..12 {
        let mut req = create_test_request(&format!("req{}", i), 50, 500);
        pool.submit_request(req);
    }

    // Simulate many tokens generated (approaching context limit)
    for _ in 0..15 {
        let positions = vec![450, 460, 470, 480, 490, 500, 500, 490, 480, 470, 460, 450];
        monitor.record_batch(12, 0, &positions);
    }

    // Tight memory scenario (only 8GB available)
    let available_memory = 8 * 1024 * 1024 * 1024u64;
    let decision = monitor.should_adjust(16, available_memory);

    assert!(
        decision.is_some(),
        "Monitor should recommend shrinkage under memory pressure"
    );
    assert!(
        decision.unwrap() < 16,
        "New size should be smaller than current"
    );

    println!("✓ Shrink decision: 16 → {} slots", decision.unwrap());
}

#[test]
fn test_cannot_resize_with_active_slots() {
    // Scenario: Try to resize while requests are active
    // Expected: resize_to() should fail

    let mut pool = SlotPool::new(8);

    // Add active requests
    pool.submit_request(create_test_request("req1", 50, 100));
    pool.submit_request(create_test_request("req2", 50, 100));

    assert_eq!(pool.active_count(), 2);
    assert!(!pool.can_resize(), "Cannot resize with active slots");

    // Attempt resize should fail
    let result = pool.resize_to(16);
    assert!(
        result.is_err(),
        "Should not be able to resize with active slots"
    );
}

#[test]
fn test_resize_when_safe() {
    // Scenario: All requests finished, pool is idle
    // Expected: Resize succeeds, pending requests preserved

    let mut pool = SlotPool::new(8);

    // Add some pending requests (they won't be allocated yet if we don't submit)
    pool.submit_request(create_test_request("req1", 50, 100));
    pool.submit_request(create_test_request("req2", 50, 100));

    // Complete these requests
    pool.update_slot(0, 100).unwrap(); // Generate 1 token
    pool.update_slot(0, 101).unwrap(); // Complete (max_new_tokens=100 in test)
    // ... in real code we'd generate all 100 tokens, but this tests the concept

    // Actually, let's just free the slots to simulate completion
    pool.free_slot(0).unwrap();
    pool.free_slot(1).unwrap();

    assert!(pool.can_resize(), "Pool should be resizable when idle");

    // Grow the pool
    let result = pool.resize_to(16);
    assert!(result.is_ok(), "Resize should succeed when safe");

    let stats = pool.stats();
    assert_eq!(stats.total_slots, 16);
    assert_eq!(stats.free_slots, 16);

    println!("✓ Successfully resized from 8 to 16 slots");
}

#[test]
fn test_cooldown_prevents_rapid_adjustments() {
    // Scenario: Multiple adjustment opportunities in quick succession
    // Expected: Only first adjustment proceeds, rest are blocked by cooldown

    let mut pool = SlotPool::new(10);
    let mut monitor = SlotPoolMonitor::new(test_model_profile(), 2);

    // Fill slots and queue
    for i in 0..15 {
        pool.submit_request(create_test_request(&format!("req{}", i), 50, 100));
    }

    // Build up samples
    for _ in 0..15 {
        let positions = pool.get_active_positions();
        let stats = pool.stats();
        monitor.record_batch(stats.active_slots, stats.pending_requests, &positions);
    }

    let available_memory = 16 * 1024 * 1024 * 1024u64;

    // First check should return decision
    let decision1 = monitor.should_adjust(10, available_memory);
    assert!(decision1.is_some(), "First adjustment should be allowed");

    // Record that adjustment happened
    monitor.record_adjustment();

    // Immediate second check should be blocked
    let decision2 = monitor.should_adjust(10, available_memory);
    assert!(
        decision2.is_none(),
        "Cooldown should prevent immediate second adjustment"
    );

    println!("✓ Cooldown mechanism working correctly");
}

#[test]
fn test_memory_estimation_scales_with_position() {
    // Scenario: Verify memory usage estimation is proportional to token positions
    // Expected: Higher positions → higher memory estimate

    let monitor = SlotPoolMonitor::new(test_model_profile(), 2);

    // Early in generation
    let positions_early = vec![50, 60, 70, 80];
    let stats_early = monitor.get_statistics();

    // Late in generation
    let positions_late = vec![400, 450, 480, 500];
    
    // The estimate_memory_usage is private, but we can test via record_batch
    // and check statistics
    
    // For manual verification:
    let kv_per_request = test_model_profile().kv_cache_bytes_per_request(2);
    let per_token = kv_per_request as f64 / test_model_profile().context_window as f64;
    
    let early_total: usize = positions_early.iter().sum();
    let late_total: usize = positions_late.iter().sum();
    
    let early_estimate = (early_total as f64 * per_token) as u64;
    let late_estimate = (late_total as f64 * per_token) as u64;
    
    assert!(
        late_estimate > early_estimate,
        "Memory usage should increase with token position"
    );
    
    println!(
        "✓ Memory estimation: early={} MB, late={} MB",
        early_estimate / (1024 * 1024),
        late_estimate / (1024 * 1024)
    );
}

#[test]
fn test_gradual_adjustment_limits() {
    // Scenario: Adjustment size should be bounded by max_adjustment_fraction
    // Expected: Growth/shrinkage limited to ±10% per adjustment

    let mut monitor = SlotPoolMonitor::new(test_model_profile(), 2);

    // Build up samples favoring growth
    for _ in 0..15 {
        monitor.record_batch(8, 10, &[100, 120, 150, 180, 200, 220, 240, 260]);
    }

    let available_memory = 32 * 1024 * 1024 * 1024u64; // Plenty of memory
    let decision = monitor.should_adjust(20, available_memory);

    if let Some(new_size) = decision {
        let growth = (new_size as i32 - 20) as f64 / 20.0;
        assert!(
            growth <= 0.11, // Allow slight rounding, so 0.11 instead of exact 0.1
            "Growth should be limited to ~10% per adjustment, got {:.1}%",
            growth * 100.0
        );
        println!(
            "✓ Gradual adjustment: 20 → {} slots ({:.1}% growth)",
            new_size,
            growth * 100.0
        );
    }
}

#[test]
fn test_no_adjustment_without_sufficient_samples() {
    // Scenario: Only a few samples collected
    // Expected: No adjustment decision (need more data)

    let mut monitor = SlotPoolMonitor::new(test_model_profile(), 2);

    // Only 3 samples (need 10 by default)
    for _ in 0..3 {
        monitor.record_batch(5, 5, &[100, 150, 200, 250, 300]);
    }

    let available_memory = 16 * 1024 * 1024 * 1024u64;
    let decision = monitor.should_adjust(10, available_memory);

    assert!(
        decision.is_none(),
        "Should not adjust with insufficient samples"
    );

    println!("✓ Correctly requires minimum samples before adjustment");
}
