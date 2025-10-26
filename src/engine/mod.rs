//! Inference Engine Components
//!
//! This module contains the core scheduling and execution infrastructure
//! for continuous batching and request management.

pub mod slot_monitor;
pub mod slot_pool;
pub mod speculative;

pub use slot_monitor::{AdjustmentConfig, MemoryStatistics, SlotPoolMonitor};
pub use slot_pool::{
    CompletedToken, Request as SlotPoolRequest, RequestId, SlotId, SlotPool, SlotPoolError,
    SlotPoolStats, SlotState,
};
pub use speculative::{SpeculativeConfig, SpeculativeDecoder, SpeculativeModel, SpeculativeStats};

// Legacy types for backward compatibility (will be migrated to SlotPool)
use anyhow::Result;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;

/// Legacy Request type (to be replaced by SlotPoolRequest)
#[derive(Debug, Clone)]
pub struct Request {
    pub id: String,
    pub prompt: String,
    pub max_new_tokens: usize,
}

/// Legacy RequestState enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    Pending,
    Decoding,
    Completed,
}

/// Legacy RequestContext (to be replaced by SlotPool's internal state)
#[derive(Debug, Clone)]
pub struct RequestContext {
    pub request: Request,
    pub state: RequestState,
    pub tokens_generated: usize,
    pub position: usize,
    pub cache_index: Option<usize>,
    pub generated_tokens: Vec<u32>,
}

impl RequestContext {
    pub fn new(request: Request) -> Self {
        Self {
            request,
            state: RequestState::Pending,
            tokens_generated: 0,
            position: 0,
            cache_index: None,
            generated_tokens: Vec::new(),
        }
    }

    pub fn assign_cache_index(&mut self, index: usize) {
        self.cache_index = Some(index);
    }

    pub fn start_decoding(&mut self) {
        self.state = RequestState::Decoding;
    }

    pub fn record_token(&mut self) {
        self.tokens_generated += 1;
        self.position += 1;
    }

    pub fn complete(&mut self) {
        self.state = RequestState::Completed;
    }

    pub fn should_continue(&self) -> bool {
        (self.state == RequestState::Decoding)
            && self.tokens_generated < self.request.max_new_tokens
    }
}

/// Legacy RequestQueue
#[derive(Clone)]
pub struct RequestQueue {
    pending: Arc<Mutex<VecDeque<RequestContext>>>,
}

impl RequestQueue {
    pub fn new() -> Self {
        Self {
            pending: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub fn submit(&self, req: Request) -> Result<String> {
        let id = req.id.clone();
        let ctx = RequestContext::new(req);
        self.pending.lock().push_back(ctx);
        Ok(id)
    }

    pub fn pop(&self) -> Option<RequestContext> {
        self.pending.lock().pop_front()
    }

    pub fn len(&self) -> usize {
        self.pending.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.lock().is_empty()
    }
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Legacy BatchConfig
#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
}

impl BatchConfig {
    pub fn new(max_batch_size: usize, max_batch_tokens: usize) -> Self {
        Self {
            max_batch_size,
            max_batch_tokens,
        }
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_batch_tokens: 2048,
        }
    }
}

/// Legacy BatchAssembler
pub struct BatchAssembler {
    config: BatchConfig,
}

impl BatchAssembler {
    pub fn new(config: BatchConfig) -> Self {
        Self { config }
    }

    pub fn assemble_batch(&self, queue: &RequestQueue) -> Vec<RequestContext> {
        let mut batch = Vec::new();
        let mut total_tokens = 0;
        let mut overflow = Vec::new();

        while batch.len() < self.config.max_batch_size {
            let ctx = match queue.pop() {
                Some(ctx) => ctx,
                None => break,
            };

            let request_tokens = ctx.request.max_new_tokens;

            if total_tokens + request_tokens > self.config.max_batch_tokens {
                overflow.push(ctx);
                break;
            }

            total_tokens += request_tokens;
            batch.push(ctx);
        }

        for ctx in overflow {
            let req = ctx.request;
            let _ = queue.submit(req);
        }

        batch
    }
}

// Re-export cache types for direct use throughout the codebase
pub use crate::cache::parallel_cache_builder::{
    IndicesAndMask, ParallelCacheBuilder, ParallelKvCache,
};
