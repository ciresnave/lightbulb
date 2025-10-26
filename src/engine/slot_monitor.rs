//! Runtime monitoring and dynamic adjustment for SlotPool
//!
//! This module adds adaptive slot pool sizing based on actual resource usage.
//! Unlike the static `calculate_optimal_batch_size()` calculation, this monitors
//! real workload patterns and adjusts slot count dynamically.
//!
//! # Design
//!
//! - **Monitor**: Track per-request memory, peak usage, queue depth
//! - **Adjust**: Grow when queue builds with headroom, shrink when memory pressure
//! - **Gradual**: ±10% adjustments per window to avoid thrashing
//! - **Safe**: Only adjust when no active requests are in flight
//!
//! # Example
//!
//! ```ignore
//! let monitor = SlotPoolMonitor::new(model_profile, dtype_bytes);
//!
//! // After each batch
//! monitor.record_batch_stats(&pool, actual_memory_used);
//!
//! // Periodically check for adjustment
//! if let Some(new_size) = monitor.should_adjust(&pool) {
//!     if pool.can_resize() {
//!         pool.resize_to(new_size)?;
//!     }
//! }
//! ```

use crate::hardware::batch_sizing::ModelMemoryProfile;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Tracks resource usage patterns for dynamic slot adjustment
#[derive(Debug)]
pub struct SlotPoolMonitor {
    /// Model memory profile for KV cache estimation
    model_profile: ModelMemoryProfile,

    /// Bytes per tensor element (2 for f16, 4 for f32)
    dtype_bytes: usize,

    /// Recent memory measurements (sliding window)
    memory_samples: VecDeque<MemorySample>,

    /// Maximum sample window size
    max_samples: usize,

    /// Last adjustment timestamp
    last_adjustment: Option<Instant>,

    /// Minimum time between adjustments (prevent thrashing)
    adjustment_cooldown: Duration,

    /// Configuration for adjustment policy
    config: AdjustmentConfig,
}

/// Single memory measurement
#[derive(Debug, Clone)]
struct MemorySample {
    timestamp: Instant,
    /// Total memory used by active requests (bytes)
    total_memory_bytes: u64,
    /// Number of active slots at time of measurement
    active_slots: usize,
    /// Number of pending requests in queue
    pending_requests: usize,
}

/// Configuration for adjustment policy
#[derive(Debug, Clone)]
pub struct AdjustmentConfig {
    /// Target memory utilization (0.0-1.0, default 0.7)
    pub target_utilization: f64,

    /// Shrink threshold: adjust when memory exceeds this (default 0.8)
    pub shrink_threshold: f64,

    /// Grow threshold: adjust when utilization below this + queue not empty (default 0.5)
    pub grow_threshold: f64,

    /// Maximum adjustment per window (fraction of current size, default 0.1 = ±10%)
    pub max_adjustment_fraction: f64,

    /// Minimum cooldown between adjustments (default 30 seconds)
    pub adjustment_cooldown_secs: u64,

    /// Minimum samples needed before adjustment (default 10)
    pub min_samples_for_adjustment: usize,
}

impl Default for AdjustmentConfig {
    fn default() -> Self {
        Self {
            target_utilization: 0.7,
            shrink_threshold: 0.8,
            grow_threshold: 0.5,
            max_adjustment_fraction: 0.1,
            adjustment_cooldown_secs: 30,
            min_samples_for_adjustment: 10,
        }
    }
}

impl SlotPoolMonitor {
    /// Create a new monitor with default configuration
    pub fn new(model_profile: ModelMemoryProfile, dtype_bytes: usize) -> Self {
        Self::with_config(model_profile, dtype_bytes, AdjustmentConfig::default())
    }

    /// Create a new monitor with custom configuration
    pub fn with_config(
        model_profile: ModelMemoryProfile,
        dtype_bytes: usize,
        config: AdjustmentConfig,
    ) -> Self {
        Self {
            model_profile,
            dtype_bytes,
            memory_samples: VecDeque::new(),
            max_samples: 100, // Keep last 100 measurements
            last_adjustment: None,
            adjustment_cooldown: Duration::from_secs(config.adjustment_cooldown_secs),
            config,
        }
    }

    /// Record current memory usage for a batch
    ///
    /// Call this after each batch processing to maintain sliding window
    /// of memory measurements.
    ///
    /// # Arguments
    ///
    /// * `active_slots` - Number of currently active slots
    /// * `pending_requests` - Number of requests in queue
    /// * `current_positions` - Token positions for each active slot (for KV cache estimation)
    pub fn record_batch(
        &mut self,
        active_slots: usize,
        pending_requests: usize,
        current_positions: &[usize],
    ) {
        // Estimate memory usage based on current token positions
        let total_memory = self.estimate_memory_usage(current_positions);

        let sample = MemorySample {
            timestamp: Instant::now(),
            total_memory_bytes: total_memory,
            active_slots,
            pending_requests,
        };

        self.memory_samples.push_back(sample);

        // Maintain sliding window
        while self.memory_samples.len() > self.max_samples {
            self.memory_samples.pop_front();
        }
    }

    /// Estimate memory usage for current batch based on token positions
    ///
    /// Uses per-token KV cache calculation: for each active request,
    /// memory = current_position × (per_token_kv_cache_bytes)
    fn estimate_memory_usage(&self, current_positions: &[usize]) -> u64 {
        let full_context_bytes = self
            .model_profile
            .kv_cache_bytes_per_request(self.dtype_bytes);
        let per_token_bytes = full_context_bytes as f64 / self.model_profile.context_window as f64;

        current_positions
            .iter()
            .map(|&pos| (pos as f64 * per_token_bytes) as u64)
            .sum()
    }

    /// Check if slot pool should be adjusted
    ///
    /// Returns Some(new_size) if adjustment is recommended, None otherwise.
    ///
    /// # Decision Logic
    ///
    /// **Grow** when:
    /// - Memory utilization < grow_threshold (50%)
    /// - Pending queue is not empty (demand exists)
    /// - Cooldown period has elapsed
    /// - Sufficient samples collected
    ///
    /// **Shrink** when:
    /// - Memory utilization > shrink_threshold (80%)
    /// - Cooldown period has elapsed
    /// - Sufficient samples collected
    pub fn should_adjust(
        &self,
        current_max_slots: usize,
        available_memory_bytes: u64,
    ) -> Option<usize> {
        // Check cooldown
        if let Some(last_adj) = self.last_adjustment {
            if last_adj.elapsed() < self.adjustment_cooldown {
                return None;
            }
        }

        // Need enough samples for reliable decision
        if self.memory_samples.len() < self.config.min_samples_for_adjustment {
            return None;
        }

        // Calculate statistics from recent samples
        let stats = self.calculate_statistics();

        // Calculate current memory utilization
        let utilization = stats.avg_memory_bytes as f64 / available_memory_bytes as f64;

        // Decision: Should we grow or shrink?
        if utilization > self.config.shrink_threshold {
            // Memory pressure: shrink slot pool
            let reduction =
                (current_max_slots as f64 * self.config.max_adjustment_fraction).ceil() as usize;
            let new_size = current_max_slots.saturating_sub(reduction).max(2);

            if new_size < current_max_slots {
                return Some(new_size);
            }
        } else if utilization < self.config.grow_threshold && stats.avg_pending_requests > 0.0 {
            // Headroom available + demand exists: grow slot pool
            let increase =
                (current_max_slots as f64 * self.config.max_adjustment_fraction).ceil() as usize;
            let new_size = current_max_slots + increase;

            // Sanity check: don't grow beyond what memory can support
            let kv_per_request = self
                .model_profile
                .kv_cache_bytes_per_request(self.dtype_bytes);
            let max_supportable = (available_memory_bytes / kv_per_request) as usize;

            let new_size = new_size.min(max_supportable).min(128); // Cap at 128 slots

            if new_size > current_max_slots {
                return Some(new_size);
            }
        }

        None
    }

    /// Mark that an adjustment was made (resets cooldown)
    pub fn record_adjustment(&mut self) {
        self.last_adjustment = Some(Instant::now());
    }

    /// Calculate statistics from recent samples
    fn calculate_statistics(&self) -> MemoryStatistics {
        if self.memory_samples.is_empty() {
            return MemoryStatistics::default();
        }

        let count = self.memory_samples.len() as f64;

        let avg_memory = self
            .memory_samples
            .iter()
            .map(|s| s.total_memory_bytes)
            .sum::<u64>() as f64
            / count;

        let peak_memory = self
            .memory_samples
            .iter()
            .map(|s| s.total_memory_bytes)
            .max()
            .unwrap_or(0);

        let avg_pending = self
            .memory_samples
            .iter()
            .map(|s| s.pending_requests as f64)
            .sum::<f64>()
            / count;

        let avg_active = self
            .memory_samples
            .iter()
            .map(|s| s.active_slots as f64)
            .sum::<f64>()
            / count;

        MemoryStatistics {
            avg_memory_bytes: avg_memory as u64,
            peak_memory_bytes: peak_memory,
            avg_pending_requests: avg_pending,
            avg_active_slots: avg_active,
        }
    }

    /// Get current statistics summary
    pub fn get_statistics(&self) -> MemoryStatistics {
        self.calculate_statistics()
    }
}

/// Aggregated statistics from monitoring window
#[derive(Debug, Default, Clone)]
pub struct MemoryStatistics {
    pub avg_memory_bytes: u64,
    pub peak_memory_bytes: u64,
    pub avg_pending_requests: f64,
    pub avg_active_slots: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model_profile() -> ModelMemoryProfile {
        ModelMemoryProfile {
            weights_bytes: 6 * 1024 * 1024 * 1024, // 6GB
            num_layers: 32,
            hidden_size: 4096,
            num_kv_heads: 32,
            context_window: 512,
        }
    }

    #[test]
    fn test_memory_estimation() {
        let monitor = SlotPoolMonitor::new(test_model_profile(), 2);

        // 3 requests at different positions
        let positions = vec![100, 200, 256];
        let memory = monitor.estimate_memory_usage(&positions);

        // Should be proportional to sum of positions
        assert!(memory > 0);
        println!(
            "Estimated memory for {:?}: {} MB",
            positions,
            memory / (1024 * 1024)
        );
    }

    #[test]
    fn test_grow_decision() {
        let mut monitor = SlotPoolMonitor::new(test_model_profile(), 2);

        // Simulate low utilization with pending requests
        for _ in 0..15 {
            monitor.record_batch(
                4,                     // 4 active slots
                5,                     // 5 pending requests
                &[100, 150, 200, 250], // moderate positions
            );
        }

        let available_memory = 16 * 1024 * 1024 * 1024u64; // 16GB
        let decision = monitor.should_adjust(10, available_memory);

        // Should recommend growing (low util + queue)
        assert!(decision.is_some());
        assert!(decision.unwrap() > 10);
    }

    #[test]
    fn test_shrink_decision() {
        let mut monitor = SlotPoolMonitor::new(test_model_profile(), 2);

        // Simulate high memory usage
        for _ in 0..15 {
            monitor.record_batch(
                8,                                         // 8 active slots
                0,                                         // no pending
                &[450, 480, 500, 510, 500, 490, 480, 470], // near max context
            );
        }

        // Tight memory: only 1GB available (8 slots × ~384MB each ≈ 3GB > 80% of 1GB)
        let available_memory = 1024 * 1024 * 1024u64; // 1GB (very tight)
        let decision = monitor.should_adjust(10, available_memory);

        // Should recommend shrinking (high memory pressure)
        assert!(decision.is_some(), "Expected shrink recommendation");
        assert!(decision.unwrap() < 10, "Expected smaller size than 10");
    }

    #[test]
    fn test_cooldown_prevents_thrashing() {
        let mut monitor = SlotPoolMonitor::new(test_model_profile(), 2);

        // Record enough samples
        for _ in 0..15 {
            monitor.record_batch(4, 5, &[100, 150, 200, 250]);
        }

        let available_memory = 16 * 1024 * 1024 * 1024u64;

        // First adjustment should trigger
        let decision1 = monitor.should_adjust(10, available_memory);
        assert!(decision1.is_some());

        // Record the adjustment
        monitor.record_adjustment();

        // Second call immediately after should be blocked by cooldown
        let decision2 = monitor.should_adjust(10, available_memory);
        assert!(decision2.is_none());
    }
}
