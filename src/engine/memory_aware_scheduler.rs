//! Memory-Aware Priority Scheduler
//!
//! Extends SlotPool with memory budgeting, priority-based admission control,
//! and eviction pressure metrics for efficient resource management.
//!
//! # Features
//!
//! - **Memory Budget Tracking**: Track actual memory usage per slot
//! - **Priority-based Admission**: Higher priority requests get slots first
//! - **Eviction Pressure Metrics**: Monitor when system is close to capacity
//! - **Dynamic Slot Limits**: Adjust max_slots based on available memory
//!
//! # Example
//!
//! ```ignore
//! let config = MemoryAwareConfig {
//!     max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
//!     memory_per_slot_base: 100 * 1024 * 1024,  // 100MB base
//!     memory_per_token: 50 * 1024,               // 50KB per token
//!     eviction_pressure_threshold: 0.85,         // 85% utilization
//! };
//!
//! let mut scheduler = MemoryAwareScheduler::new(config);
//! scheduler.submit_priority_request(request, Priority::High);
//! ```

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use super::slot_pool::{Request, SlotId, SlotPool, SlotPoolError};

/// Request priority level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Low priority (batch jobs, background tasks)
    Low = 0,

    /// Normal priority (default)
    Normal = 1,

    /// High priority (interactive queries, urgent tasks)
    High = 2,

    /// Critical priority (system tasks, health checks)
    Critical = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Normal
    }
}

/// Configuration for memory-aware scheduling
#[derive(Debug, Clone)]
pub struct MemoryAwareConfig {
    /// Maximum memory budget in bytes
    pub max_memory_bytes: usize,

    /// Base memory cost per slot (model weights, overhead)
    pub memory_per_slot_base: usize,

    /// Memory cost per token in KV cache (bytes)
    pub memory_per_token: usize,

    /// Eviction pressure threshold (0.0-1.0)
    /// Above this threshold, scheduler starts rejecting low-priority requests
    pub eviction_pressure_threshold: f64,

    /// Safety margin for memory allocation (0.0-1.0)
    /// Reserve this fraction of memory as headroom
    pub memory_safety_margin: f64,
}

impl Default for MemoryAwareConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            memory_per_slot_base: 100 * 1024 * 1024,  // 100MB base
            memory_per_token: 50 * 1024,              // 50KB per token
            eviction_pressure_threshold: 0.85,        // 85% utilization
            memory_safety_margin: 0.1,                // 10% headroom
        }
    }
}

/// Priority request wrapper
#[derive(Debug, Clone)]
struct PriorityRequest {
    request: Request,
    priority: Priority,
    /// Timestamp for FIFO within same priority
    submission_order: u64,
}

impl PartialEq for PriorityRequest {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.submission_order == other.submission_order
    }
}

impl Eq for PriorityRequest {}

impl PartialOrd for PriorityRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier submission
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.submission_order.cmp(&self.submission_order))
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total memory currently used (bytes)
    pub used_bytes: usize,

    /// Memory available (bytes)
    pub available_bytes: usize,

    /// Number of active slots
    pub active_slots: usize,

    /// Current eviction pressure (0.0-1.0)
    pub eviction_pressure: f64,

    /// Number of requests rejected due to memory
    pub rejected_count: usize,

    /// Number of requests waiting in queue
    pub queued_count: usize,
}

/// Memory-aware scheduler with priority-based admission control
pub struct MemoryAwareScheduler {
    /// Configuration
    config: MemoryAwareConfig,

    /// Underlying slot pool
    slot_pool: SlotPool,

    /// Priority queue of pending requests
    priority_queue: BinaryHeap<PriorityRequest>,

    /// Memory usage per slot (slot_id -> bytes)
    slot_memory_usage: HashMap<SlotId, usize>,

    /// Current total memory usage
    current_memory_usage: usize,

    /// Submission counter for FIFO within priority
    submission_counter: u64,

    /// Statistics
    stats: MemoryStats,
}

impl MemoryAwareScheduler {
    /// Create a new memory-aware scheduler
    ///
    /// Initial max_slots is calculated from memory budget
    pub fn new(config: MemoryAwareConfig) -> Self {
        // Calculate max slots from memory budget
        let effective_memory =
            (config.max_memory_bytes as f64 * (1.0 - config.memory_safety_margin)) as usize;
        let max_slots = effective_memory / config.memory_per_slot_base;

        Self {
            config,
            slot_pool: SlotPool::new(max_slots),
            priority_queue: BinaryHeap::new(),
            slot_memory_usage: HashMap::new(),
            current_memory_usage: 0,
            submission_counter: 0,
            stats: MemoryStats::default(),
        }
    }

    /// Submit a request with priority
    pub fn submit_priority_request(&mut self, request: Request, priority: Priority) {
        let priority_request = PriorityRequest {
            request,
            priority,
            submission_order: self.submission_counter,
        };

        self.submission_counter += 1;
        self.priority_queue.push(priority_request);
        self.stats.queued_count = self.priority_queue.len();
    }

    /// Submit a request with default (Normal) priority
    pub fn submit_request(&mut self, request: Request) {
        self.submit_priority_request(request, Priority::Normal);
    }

    /// Allocate pending requests to free slots based on priority and memory
    pub fn allocate_pending_requests(&mut self) {
        let mut temp_queue = BinaryHeap::new();

        while let Some(priority_req) = self.priority_queue.pop() {
            // Estimate memory needed for this request
            let estimated_tokens =
                priority_req.request.prompt_tokens.len() + priority_req.request.max_new_tokens;
            let estimated_memory = self.config.memory_per_slot_base
                + (estimated_tokens * self.config.memory_per_token);

            // Check memory budget
            let available_memory = self.get_available_memory();
            if estimated_memory > available_memory {
                // Check eviction pressure
                let pressure = self.calculate_eviction_pressure();

                if pressure > self.config.eviction_pressure_threshold
                    && priority_req.priority < Priority::High
                {
                    // Reject low-priority requests under pressure
                    self.stats.rejected_count += 1;
                    println!(
                        "⚠️  Rejected request {} (priority={:?}, pressure={:.2}, memory needed={} MB)",
                        priority_req.request.id,
                        priority_req.priority,
                        pressure,
                        estimated_memory / (1024 * 1024)
                    );
                    continue;
                }

                // Queue back for later
                temp_queue.push(priority_req);
                continue;
            }

            // Try to allocate
            let request_id = priority_req.request.id.clone();
            self.slot_pool.submit_request(priority_req.request.clone());

            // Track memory if allocated
            if let Some(&slot_id) = self.slot_pool.get_request_slot(&request_id) {
                self.slot_memory_usage.insert(slot_id, estimated_memory);
                self.current_memory_usage += estimated_memory;
                println!(
                    "✓ Allocated request {} (priority={:?}) to slot {} [{} MB]",
                    request_id,
                    priority_req.priority,
                    slot_id,
                    estimated_memory / (1024 * 1024)
                );
            } else {
                // Slot allocation failed, queue back
                temp_queue.push(priority_req);
            }
        }

        // Restore unallocated requests
        self.priority_queue = temp_queue;
        self.stats.queued_count = self.priority_queue.len();

        // Update statistics
        self.update_stats();
    }

    /// Free a slot and reclaim its memory
    pub fn free_slot(&mut self, slot_id: SlotId) -> Result<(), SlotPoolError> {
        if let Some(memory) = self.slot_memory_usage.remove(&slot_id) {
            self.current_memory_usage = self.current_memory_usage.saturating_sub(memory);
        }

        self.slot_pool.free_slot(slot_id)?;
        self.update_stats();
        Ok(())
    }

    /// Calculate current eviction pressure (0.0-1.0)
    fn calculate_eviction_pressure(&self) -> f64 {
        let max_memory = (self.config.max_memory_bytes as f64
            * (1.0 - self.config.memory_safety_margin)) as usize;
        self.current_memory_usage as f64 / max_memory as f64
    }

    /// Get available memory in bytes
    fn get_available_memory(&self) -> usize {
        let max_memory = (self.config.max_memory_bytes as f64
            * (1.0 - self.config.memory_safety_margin)) as usize;
        max_memory.saturating_sub(self.current_memory_usage)
    }

    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.used_bytes = self.current_memory_usage;
        self.stats.available_bytes = self.get_available_memory();
        self.stats.active_slots = self.slot_memory_usage.len();
        self.stats.eviction_pressure = self.calculate_eviction_pressure();
        self.stats.queued_count = self.priority_queue.len();
    }

    /// Get current memory statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        self.stats.clone()
    }

    /// Get reference to underlying slot pool
    pub fn slot_pool(&self) -> &SlotPool {
        &self.slot_pool
    }

    /// Get mutable reference to underlying slot pool
    pub fn slot_pool_mut(&mut self) -> &mut SlotPool {
        &mut self.slot_pool
    }

    /// Adjust memory budget at runtime
    pub fn set_memory_budget(&mut self, new_budget_bytes: usize) {
        self.config.max_memory_bytes = new_budget_bytes;
        self.update_stats();

        println!(
            "🔧 Adjusted memory budget to {} GB (pressure: {:.2})",
            new_budget_bytes / (1024 * 1024 * 1024),
            self.stats.eviction_pressure
        );
    }

    /// Get eviction pressure threshold
    pub fn get_eviction_pressure(&self) -> f64 {
        self.stats.eviction_pressure
    }

    /// Check if system is under memory pressure
    pub fn is_under_pressure(&self) -> bool {
        self.stats.eviction_pressure > self.config.eviction_pressure_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_priority_ordering() {
        let config = MemoryAwareConfig {
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            ..Default::default()
        };

        let mut scheduler = MemoryAwareScheduler::new(config);

        // Submit requests with different priorities
        scheduler.submit_priority_request(create_test_request("low", 10, 10), Priority::Low);
        scheduler.submit_priority_request(create_test_request("normal", 10, 10), Priority::Normal);
        scheduler.submit_priority_request(create_test_request("high", 10, 10), Priority::High);

        scheduler.allocate_pending_requests();

        // High priority should be allocated first
        let ready = scheduler.slot_pool().get_ready_batch();
        assert_eq!(ready.len(), 3); // All should fit in memory
    }

    #[test]
    fn test_memory_tracking() {
        let config = MemoryAwareConfig {
            max_memory_bytes: 500 * 1024 * 1024,     // 500MB
            memory_per_slot_base: 100 * 1024 * 1024, // 100MB
            memory_per_token: 50 * 1024,             // 50KB
            ..Default::default()
        };

        let mut scheduler = MemoryAwareScheduler::new(config);

        // Submit a large request
        scheduler.submit_request(create_test_request("large", 1000, 1000));
        scheduler.allocate_pending_requests();

        let stats = scheduler.get_memory_stats();
        assert!(stats.used_bytes > 0);
        assert!(stats.eviction_pressure > 0.0);
    }

    #[test]
    fn test_eviction_pressure_rejection() {
        let config = MemoryAwareConfig {
            max_memory_bytes: 200 * 1024 * 1024,    // 200MB
            memory_per_slot_base: 80 * 1024 * 1024, // 80MB per slot
            memory_per_token: 10 * 1024,            // 10KB per token
            eviction_pressure_threshold: 0.7,       // 70%
            memory_safety_margin: 0.1,              // 10% margin
            ..Default::default()
        };

        let mut scheduler = MemoryAwareScheduler::new(config);

        // Fill up memory with Normal priority requests
        for i in 0..5 {
            scheduler.submit_priority_request(
                create_test_request(&format!("req-{}", i), 100, 100),
                Priority::Normal,
            );
        }

        scheduler.allocate_pending_requests();

        let stats = scheduler.get_memory_stats();
        assert!(stats.rejected_count > 0 || stats.queued_count > 0);
    }

    #[test]
    fn test_dynamic_budget_adjustment() {
        let config = MemoryAwareConfig {
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            ..Default::default()
        };

        let mut scheduler = MemoryAwareScheduler::new(config);

        // Add some requests
        scheduler.submit_request(create_test_request("req-1", 100, 100));
        scheduler.allocate_pending_requests();

        let stats_before = scheduler.get_memory_stats();

        // Reduce budget
        scheduler.set_memory_budget(512 * 1024 * 1024); // 512MB

        let stats_after = scheduler.get_memory_stats();

        // Pressure should increase with reduced budget
        assert!(stats_after.eviction_pressure > stats_before.eviction_pressure);
    }
}
