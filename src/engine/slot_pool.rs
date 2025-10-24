//! Slot Pool for Continuous Batching
//!
//! This module implements a dynamic slot allocation system that enables continuous
//! batching: requests arrive and complete independently, and each inference iteration
//! processes whatever slots are ready (variable batch size 1-N).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                       SlotPool (64 slots)                   │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Slot 0: Active (req-1, pos=50, prefill=false)               │
//! │ Slot 1: Free                                                │
//! │ Slot 2: Active (req-2, pos=120, prefill=false)              │
//! │ Slot 3: Active (req-3, pos=0, prefill=true)   ← New request│
//! │ Slot 4: Free                                                │
//! │ ...                                                         │
//! │ Slot 63: Finished (req-0, waiting for pickup)               │
//! └─────────────────────────────────────────────────────────────┘
//!
//! Each iteration:
//!   1. Allocate new requests to free slots
//!   2. Batch all Active slots together (variable size!)
//!   3. Process batch (unified prefill/decode)
//!   4. Mark completed slots as Finished
//!   5. Return completed results
//! ```
//!
//! # Key Differences from Static Batching
//!
//! **Static Batching (Old):**
//! - Fixed batch size (e.g., 8 requests)
//! - Wait until batch is full
//! - All requests process together
//! - All must finish before new batch starts
//!
//! **Continuous Batching (New):**
//! - Memory-limited slots (e.g., 64 based on VRAM)
//! - Variable batch size (1-64 per iteration)
//! - Requests complete independently
//! - New requests fill free slots immediately
//!
//! # Phase 1 Scope
//!
//! This initial implementation keeps prefill and decode unified in the same
//! forward pass (your existing BatchedTransformer handles both). This simplifies
//! the state machine and gets us the main benefits of continuous batching:
//!
//! - No waiting for batches to fill
//! - No head-of-line blocking
//! - 2-3x higher throughput on concurrent workloads
//!
//! Phase 2 (later) will separate prefill/decode for additional 1.5-2x speedup.

use std::collections::{HashMap, VecDeque};
use thiserror::Error;

/// Unique identifier for a request
pub type RequestId = String;

/// Slot index in the pool (0..max_slots)
pub type SlotId = usize;

/// Request submitted to the slot pool
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request identifier
    pub id: RequestId,

    /// Input prompt tokens
    pub prompt_tokens: Vec<u32>,

    /// Maximum tokens to generate (not including prompt)
    pub max_new_tokens: usize,

    /// Sampling temperature
    pub temperature: f64,

    /// Top-p sampling threshold
    pub top_p: f64,
}

/// State of a single slot in the pool
#[derive(Debug, Clone)]
pub enum SlotState {
    /// Slot is available for a new request
    Free,

    /// Slot is actively processing a request
    Active {
        /// Request being processed
        request_id: RequestId,

        /// Tokens generated so far (not including prompt)
        tokens_generated: usize,

        /// Maximum tokens to generate
        max_new_tokens: usize,

        /// Current position in sequence (prompt_len + tokens_generated)
        current_position: usize,

        /// Whether this slot is still in initial prefill phase
        /// (true = processing prompt, false = generating tokens)
        is_prefill: bool,

        /// All tokens generated so far (for final result)
        generated_tokens: Vec<u32>,
    },

    /// Slot processing is complete, result ready for pickup
    Finished {
        /// Request that was processed
        request_id: RequestId,

        /// All generated tokens
        result: Vec<u32>,
    },
}

/// Token generated during an inference step
#[derive(Debug, Clone)]
pub struct CompletedToken {
    /// Request that generated this token
    pub request_id: RequestId,

    /// The generated token
    pub token: u32,

    /// Whether this was the final token for this request
    pub is_final: bool,
}

/// Errors that can occur during slot pool operations
#[derive(Debug, Error)]
pub enum SlotPoolError {
    #[error("No free slots available (all {0} slots in use)")]
    NoFreeSlots(usize),

    #[error("Slot {0} is not in active state")]
    SlotNotActive(SlotId),

    #[error("Request {0} not found in active slots")]
    RequestNotFound(RequestId),

    #[error("Invalid slot ID {0} (max slots: {1})")]
    InvalidSlotId(SlotId, usize),
}

/// Slot pool managing dynamic batch formation
pub struct SlotPool {
    /// All slots (indexed by SlotId)
    slots: Vec<SlotState>,

    /// Maximum number of slots (memory-limited)
    max_slots: usize,

    /// Map from request ID to slot ID for fast lookup
    request_to_slot: HashMap<RequestId, SlotId>,

    /// Queue of pending requests waiting for free slots
    pending_requests: VecDeque<Request>,
}

impl SlotPool {
    /// Create a new slot pool with the specified capacity
    ///
    /// # Arguments
    /// * `max_slots` - Maximum number of concurrent requests (memory-limited)
    ///
    /// # Example
    /// ```ignore
    /// let pool = SlotPool::new(64);  // 64 slots based on VRAM capacity
    /// ```
    pub fn new(max_slots: usize) -> Self {
        Self {
            slots: vec![SlotState::Free; max_slots],
            max_slots,
            request_to_slot: HashMap::new(),
            pending_requests: VecDeque::new(),
        }
    }

    /// Submit a request to the pool
    ///
    /// If a free slot is available, the request is allocated immediately.
    /// Otherwise, it's queued for later allocation.
    ///
    /// # Arguments
    /// * `request` - The request to process
    pub fn submit_request(&mut self, request: Request) {
        if let Some(slot_id) = self.try_allocate_slot(&request) {
            println!("✓ Allocated request {} to slot {}", request.id, slot_id);
        } else {
            println!("⏳ Queued request {} (no free slots)", request.id);
            self.pending_requests.push_back(request);
        }
    }

    /// Try to allocate pending requests to any newly freed slots
    ///
    /// This should be called after processing each batch to fill slots
    /// that became available.
    pub fn allocate_pending_requests(&mut self) {
        while let Some(request) = self.pending_requests.pop_front() {
            if self.try_allocate_slot(&request).is_none() {
                // No free slots, put it back and stop
                self.pending_requests.push_front(request);
                break;
            }
        }
    }

    /// Try to allocate a request to a free slot
    ///
    /// Returns Some(slot_id) if successful, None if no slots available
    fn try_allocate_slot(&mut self, request: &Request) -> Option<SlotId> {
        // Find first free slot
        let slot_id = self
            .slots
            .iter()
            .position(|s| matches!(s, SlotState::Free))?;

        // Allocate it
        let prompt_len = request.prompt_tokens.len();
        self.slots[slot_id] = SlotState::Active {
            request_id: request.id.clone(),
            tokens_generated: 0,
            max_new_tokens: request.max_new_tokens,
            current_position: prompt_len, // Start after prompt
            is_prefill: true,             // Initial prefill phase
            generated_tokens: Vec::new(),
        };

        self.request_to_slot.insert(request.id.clone(), slot_id);

        Some(slot_id)
    }

    /// Get all slots that are ready for processing
    ///
    /// Returns a vector of active slot IDs that should be batched together
    /// in the next forward pass.
    pub fn get_ready_batch(&self) -> Vec<SlotId> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, state)| match state {
                SlotState::Active { .. } => Some(idx),
                _ => None,
            })
            .collect()
    }

    /// Update a slot after generating a token
    ///
    /// # Arguments
    /// * `slot_id` - The slot to update
    /// * `token` - The newly generated token
    ///
    /// # Returns
    /// * `Ok(Some(completed))` - Request is complete
    /// * `Ok(None)` - Request continues
    /// * `Err(_)` - Slot is not active
    pub fn update_slot(
        &mut self,
        slot_id: SlotId,
        token: u32,
    ) -> Result<Option<CompletedToken>, SlotPoolError> {
        if slot_id >= self.max_slots {
            return Err(SlotPoolError::InvalidSlotId(slot_id, self.max_slots));
        }

        let state = &mut self.slots[slot_id];

        match state {
            SlotState::Active {
                request_id,
                tokens_generated,
                max_new_tokens,
                current_position,
                is_prefill,
                generated_tokens,
            } => {
                // After first token generation, we're done with prefill
                *is_prefill = false;

                // Add token to result
                generated_tokens.push(token);
                *tokens_generated += 1;
                *current_position += 1;

                // Check if we're done (TODO: also check for EOS token)
                let is_complete = *tokens_generated >= *max_new_tokens;

                let completed = CompletedToken {
                    request_id: request_id.clone(),
                    token,
                    is_final: is_complete,
                };

                if is_complete {
                    // Move to finished state
                    let req_id = request_id.clone();
                    let result = generated_tokens.clone();

                    *state = SlotState::Finished {
                        request_id: req_id.clone(),
                        result,
                    };

                    self.request_to_slot.remove(&req_id);
                }

                Ok(Some(completed))
            }
            _ => Err(SlotPoolError::SlotNotActive(slot_id)),
        }
    }

    /// Get the state of a specific slot
    pub fn get_slot_state(&self, slot_id: SlotId) -> Option<&SlotState> {
        self.slots.get(slot_id)
    }

    /// Mark a slot as free, making it available for new requests
    ///
    /// This should be called after picking up finished results.
    pub fn free_slot(&mut self, slot_id: SlotId) -> Result<(), SlotPoolError> {
        if slot_id >= self.max_slots {
            return Err(SlotPoolError::InvalidSlotId(slot_id, self.max_slots));
        }

        // Remove from request mapping if still there
        if let SlotState::Finished { request_id, .. } = &self.slots[slot_id] {
            self.request_to_slot.remove(request_id);
        }

        self.slots[slot_id] = SlotState::Free;
        Ok(())
    }

    /// Get statistics about the pool
    pub fn stats(&self) -> SlotPoolStats {
        let mut stats = SlotPoolStats::default();
        stats.total_slots = self.max_slots;
        stats.pending_requests = self.pending_requests.len();

        for state in &self.slots {
            match state {
                SlotState::Free => stats.free_slots += 1,
                SlotState::Active { is_prefill, .. } => {
                    stats.active_slots += 1;
                    if *is_prefill {
                        stats.prefilling_slots += 1;
                    } else {
                        stats.decoding_slots += 1;
                    }
                }
                SlotState::Finished { .. } => stats.finished_slots += 1,
            }
        }

        stats
    }

    /// Check if all slots are idle (no active or finished requests)
    pub fn is_idle(&self) -> bool {
        self.slots.iter().all(|s| matches!(s, SlotState::Free)) && self.pending_requests.is_empty()
    }

    /// Get the number of active requests
    pub fn active_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s, SlotState::Active { .. }))
            .count()
    }
}

/// Statistics about slot pool utilization
#[derive(Debug, Default, Clone)]
pub struct SlotPoolStats {
    pub total_slots: usize,
    pub free_slots: usize,
    pub active_slots: usize,
    pub prefilling_slots: usize,
    pub decoding_slots: usize,
    pub finished_slots: usize,
    pub pending_requests: usize,
}

impl SlotPoolStats {
    /// Calculate utilization as a percentage (0.0 - 1.0)
    pub fn utilization(&self) -> f64 {
        if self.total_slots == 0 {
            return 0.0;
        }
        (self.active_slots + self.finished_slots) as f64 / self.total_slots as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_allocation() {
        let mut pool = SlotPool::new(4);

        let req1 = Request {
            id: "req1".to_string(),
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 10,
            temperature: 0.7,
            top_p: 0.9,
        };

        pool.submit_request(req1);

        let stats = pool.stats();
        assert_eq!(stats.active_slots, 1);
        assert_eq!(stats.free_slots, 3);

        let batch = pool.get_ready_batch();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0], 0);
    }

    #[test]
    fn test_slot_completion() {
        let mut pool = SlotPool::new(4);

        let req = Request {
            id: "req1".to_string(),
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 2,
            temperature: 0.7,
            top_p: 0.9,
        };

        pool.submit_request(req);

        // Generate first token
        let result1 = pool.update_slot(0, 100).unwrap();
        assert!(result1.is_some());
        assert!(!result1.unwrap().is_final);

        // Generate second token (should complete)
        let result2 = pool.update_slot(0, 101).unwrap();
        assert!(result2.is_some());
        assert!(result2.unwrap().is_final);

        // Check slot is now finished
        match pool.get_slot_state(0) {
            Some(SlotState::Finished { .. }) => (),
            _ => panic!("Expected finished state"),
        }
    }

    #[test]
    fn test_pending_queue() {
        let mut pool = SlotPool::new(2);

        // Fill all slots
        pool.submit_request(Request {
            id: "req1".to_string(),
            prompt_tokens: vec![1],
            max_new_tokens: 10,
            temperature: 0.7,
            top_p: 0.9,
        });

        pool.submit_request(Request {
            id: "req2".to_string(),
            prompt_tokens: vec![2],
            max_new_tokens: 10,
            temperature: 0.7,
            top_p: 0.9,
        });

        // This should go to pending queue
        pool.submit_request(Request {
            id: "req3".to_string(),
            prompt_tokens: vec![3],
            max_new_tokens: 10,
            temperature: 0.7,
            top_p: 0.9,
        });

        let stats = pool.stats();
        assert_eq!(stats.active_slots, 2);
        assert_eq!(stats.pending_requests, 1);

        // Free a slot and try to allocate pending
        pool.free_slot(0).unwrap();
        pool.allocate_pending_requests();

        let stats = pool.stats();
        assert_eq!(stats.active_slots, 2);
        assert_eq!(stats.pending_requests, 0);
    }
}
