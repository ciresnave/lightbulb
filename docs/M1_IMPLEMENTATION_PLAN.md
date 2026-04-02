# M1 (0.2) Implementation Plan: Core Engine

**Status**: IN PROGRESS  
**Target Version**: 0.2.0  
**Prerequisites**: M0 (0.1) complete ✅

---

## Overview

M1 focuses on transforming Lightbulb from a single-request synchronous inference engine into a production-ready concurrent inference server with:
1. **Continuous Batching** - Handle multiple concurrent requests efficiently
2. **Paged KV Management** - Robust memory management for KV caches
3. **Observability** - Comprehensive metrics and tracing

---

## Current State (M0 Baseline)

### Existing Components

**`lightbulb/src/engine.rs`** (67 lines)
- ✅ `Request` struct (id, prompt, max_new_tokens)
- ✅ `KvPageRef` struct (layer, start_pos, len)
- ✅ `KvPager` stub (basic page allocation per layer)
- ✅ `Scheduler` struct (single-request synchronous)
- ✅ `Scheduler::run_single()` - delegates to closure for generation

**`lightbulb/src/cache.rs`** (12 lines)
- ✅ `KvCacheConfig` struct (block_size)
- ✅ `KvCache` stub

**`lightbulb/src/main.rs`**
- ✅ CLI with `LocalLlamaGen` command
- ✅ Single-request generation working

**`lightbulb/tests/integration_local_model.rs`**
- ✅ Integration test for local model loading

**Infrastructure Dependencies:**
- ✅ `tokio` available (async runtime)
- ✅ `tracing` available (observability)
- ✅ `parking_lot` available (synchronization)
- ✅ `crossbeam` available (concurrent data structures)

---

## Architecture Design

### High-Level Flow

```
┌─────────────────────────────────────────────────────────┐
│                   External Interface                    │
│  (HTTP API, CLI commands, test harness)                │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│              Scheduler (async, batching)                │
│                                                          │
│  • Request queue (priority, FIFO)                       │
│  • Batch assembly (prefill vs decode)                   │
│  • State tracking (per-request)                         │
│  • Continuous loop (schedule → execute → update)        │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│                KvPager (memory management)              │
│                                                          │
│  • Page allocation/deallocation                         │
│  • Layer-wise bookkeeping                               │
│  • Eviction policies (LRU for now)                      │
│  • Zero-copy handoff to Candle                          │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│                Candle Model Backend                     │
│                                                          │
│  • Forward pass (prefill/decode)                        │
│  • KV cache integration                                 │
│  • Sampling                                             │
└─────────────────────────────────────────────────────────┘
```

### Request Lifecycle

```
1. Submit → Queue (pending)
2. Schedule → Assemble batch (prefill or decode)
3. Execute → Forward pass + sampling
4. Update → Advance state, check completion
5. Complete → Return result
```

---

## Implementation Tasks

### Phase 1: Request Management (Week 1)

#### 1.1 Request State Machine

**File**: `lightbulb/src/engine.rs`

**Add**:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    Pending,      // Waiting to be scheduled
    Prefilling,   // Currently in prefill phase
    Decoding,     // Currently generating tokens
    Completed,    // Finished successfully
    Failed,       // Error occurred
}

pub struct RequestContext {
    pub request: Request,
    pub state: RequestState,
    pub tokens_generated: usize,
    pub kv_pages: Vec<KvPageRef>,
    pub created_at: std::time::Instant,
    pub first_token_at: Option<std::time::Instant>,
    pub completed_at: Option<std::time::Instant>,
}
```

**Why**: Track per-request state for scheduling decisions and metrics.

**Acceptance**: 
- RequestContext compiles
- State transitions valid
- Unit tests for state machine

#### 1.2 Request Queue

**File**: `lightbulb/src/engine.rs`

**Add**:
```rust
use crossbeam::queue::SegQueue;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub struct RequestQueue {
    pending: Arc<SegQueue<RequestContext>>,
    active: Arc<RwLock<HashMap<String, RequestContext>>>,
}

impl RequestQueue {
    pub fn new() -> Self { ... }
    pub fn submit(&self, req: Request) -> Result<()> { ... }
    pub fn pop_pending(&self) -> Option<RequestContext> { ... }
    pub fn update_state(&self, id: &str, state: RequestState) -> Result<()> { ... }
    pub fn complete(&self, id: &str) -> Option<RequestContext> { ... }
}
```

**Why**: Thread-safe queue for concurrent request submission/processing.

**Acceptance**:
- Submit/pop operations thread-safe
- State updates atomic
- No data races under concurrent load
- Test with 8+ concurrent threads

---

### Phase 2: Batch Assembly (Week 2)

#### 2.1 Batch Types

**File**: `lightbulb/src/engine.rs`

**Add**:
```rust
#[derive(Debug)]
pub struct PrefillBatch {
    pub requests: Vec<RequestContext>,
    pub prompt_lens: Vec<usize>,
    pub total_tokens: usize,
}

#[derive(Debug)]
pub struct DecodeBatch {
    pub requests: Vec<RequestContext>,
    pub total_tokens: usize,
}

pub enum Batch {
    Prefill(PrefillBatch),
    Decode(DecodeBatch),
}
```

**Why**: Separate prefill (parallel) from decode (sequential) for optimal batching.

**Acceptance**:
- Batch construction correct
- Token counts accurate
- Memory estimates reasonable

#### 2.2 Batch Assembler

**File**: `lightbulb/src/engine.rs`

**Add**:
```rust
pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
    pub prefill_priority: bool, // Prioritize new requests
}

impl Scheduler {
    fn assemble_batch(
        &self,
        queue: &RequestQueue,
        config: &BatchConfig,
    ) -> Option<Batch> {
        // 1. Check for pending prefill requests
        // 2. If none, gather decode requests
        // 3. Respect max_batch_size and max_batch_tokens
        // 4. Return None if no work available
        ...
    }
}
```

**Why**: Smart batching improves throughput without sacrificing TTFT.

**Acceptance**:
- Prefill batches prioritized when `prefill_priority = true`
- Token limits respected
- Mixed batches avoided (prefill OR decode, not both)

---

### Phase 3: KV Page Management (Week 3)

#### 3.1 Enhanced KvPager

**File**: `lightbulb/src/cache.rs`

**Current** (stub):
```rust
pub struct KvPager {
    pub use_kv_cache: bool,
    layers: usize,
    pages_per_layer: Vec<usize>,
}
```

**Enhance to**:
```rust
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};

pub struct KvPage {
    pub layer: usize,
    pub page_id: usize,
    pub data: Option<Vec<u8>>, // Placeholder for actual tensor data
    pub ref_count: usize,
    pub last_access: std::time::Instant,
}

pub struct KvPager {
    pub use_kv_cache: bool,
    layers: usize,
    
    // Per-layer page pools
    free_pages: Vec<Mutex<VecDeque<usize>>>,
    pages: Vec<Mutex<HashMap<usize, KvPage>>>,
    
    // Configuration
    page_size: usize, // Tokens per page
    max_pages_per_layer: usize,
}

impl KvPager {
    pub fn new(use_kv_cache: bool, page_size: usize) -> Self { ... }
    
    pub fn attach(&mut self, layers: usize, max_pages: usize) { ... }
    
    pub fn alloc_page(&self, layer: usize) -> Result<KvPageRef> {
        // 1. Try to get free page from pool
        // 2. If pool empty, check if under max_pages limit
        // 3. If at limit, evict LRU page
        // 4. Return page reference
        ...
    }
    
    pub fn free_page(&self, page_ref: KvPageRef) -> Result<()> {
        // 1. Decrement ref_count
        // 2. If ref_count == 0, return to free pool
        ...
    }
    
    pub fn get_page(&self, page_ref: KvPageRef) -> Result<&KvPage> {
        // Retrieve page for reading
        ...
    }
    
    fn evict_lru(&self, layer: usize) -> Result<usize> {
        // Find oldest page with ref_count == 0
        // Free it and return page_id
        ...
    }
}
```

**Why**: Production-grade memory management prevents OOM and enables long contexts.

**Acceptance**:
- Stable under 10k token decode
- No memory leaks (free after use)
- Eviction works correctly under pressure
- Zero-copy handoff possible (design ready, impl later)

#### 3.2 Request-to-KV Mapping

**File**: `lightbulb/src/engine.rs`

**Add**:
```rust
impl RequestContext {
    pub fn allocate_kv_pages(&mut self, pager: &KvPager, layers: usize) -> Result<()> {
        for layer in 0..layers {
            let page = pager.alloc_page(layer)?;
            self.kv_pages.push(page);
        }
        Ok(())
    }
    
    pub fn free_kv_pages(&self, pager: &KvPager) -> Result<()> {
        for page_ref in &self.kv_pages {
            pager.free_page(*page_ref)?;
        }
        Ok(())
    }
}
```

**Why**: Each request owns its KV cache pages; cleanup on completion.

**Acceptance**:
- Pages allocated on request start
- Pages freed on request completion/failure
- No page leaks after 1000 request cycles

---

### Phase 4: Continuous Batching Loop (Week 4)

#### 4.1 Async Scheduler

**File**: `lightbulb/src/engine.rs`

**Transform Scheduler from**:
```rust
pub struct Scheduler;

impl Scheduler {
    pub fn run_single<F>(&self, req: &Request, mut generate_fn: F) -> Result<String>
    where F: FnMut(&str, usize) -> Result<String>
    { ... }
}
```

**To**:
```rust
use tokio::sync::{mpsc, RwLock};
use tokio::task;

pub struct Scheduler {
    queue: Arc<RequestQueue>,
    pager: Arc<KvPager>,
    config: BatchConfig,
    running: Arc<RwLock<bool>>,
}

impl Scheduler {
    pub fn new(config: BatchConfig, pager: Arc<KvPager>) -> Self {
        Self {
            queue: Arc::new(RequestQueue::new()),
            pager,
            config,
            running: Arc::new(RwLock::new(false)),
        }
    }
    
    pub async fn submit(&self, req: Request) -> Result<mpsc::Receiver<String>> {
        // 1. Create channel for response
        // 2. Submit to queue
        // 3. Return receiver
        ...
    }
    
    pub async fn start<F>(&self, mut execute_batch: F) -> Result<()>
    where
        F: FnMut(Batch) -> Result<Vec<(String, Option<String>)>> + Send + 'static,
    {
        *self.running.write().await = true;
        
        while *self.running.read().await {
            // 1. Assemble batch from queue
            let batch = match self.assemble_batch(&self.queue, &self.config) {
                Some(b) => b,
                None => {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    continue;
                }
            };
            
            // 2. Execute batch (model forward pass)
            let results = execute_batch(batch)?;
            
            // 3. Update request states
            for (req_id, token_opt) in results {
                if let Some(token) = token_opt {
                    // Token generated, continue decoding
                    self.queue.update_state(&req_id, RequestState::Decoding)?;
                } else {
                    // Request complete
                    self.queue.complete(&req_id);
                }
            }
            
            // 4. Brief yield to avoid busy-wait
            tokio::task::yield_now().await;
        }
        
        Ok(())
    }
    
    pub async fn stop(&self) {
        *self.running.write().await = false;
    }
}
```

**Why**: Continuous batching enables high throughput with concurrent requests.

**Acceptance**:
- Handle N>=8 concurrent requests
- No OOM on CPU (with reasonable batch sizes)
- Graceful shutdown
- Async tasks don't deadlock

#### 4.2 Correctness Testing

**File**: `lightbulb/tests/continuous_batching_test.rs` (new)

**Add**:
```rust
#[tokio::test]
async fn test_concurrent_requests_deterministic() {
    // 1. Create scheduler with fixed seed
    // 2. Submit 8 requests with different prompts
    // 3. Collect results
    // 4. Compare token-by-token with single-request baseline
    // 5. Assert identical outputs (no batching artifacts)
}

#[tokio::test]
async fn test_no_oom_cpu() {
    // 1. Create scheduler with limited batch size
    // 2. Submit 16 concurrent requests (stress test)
    // 3. Assert all complete without panic/OOM
}
```

**Acceptance**: 
- Token-by-token regression passes (batched == unbatched)
- 8+ concurrent requests complete successfully
- Memory usage bounded

---

### Phase 5: Observability (Week 5)

#### 5.1 Metrics Infrastructure

**File**: `lightbulb/src/metrics.rs` (new)

**Add**:
```rust
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct Metrics {
    // Counters
    requests_submitted: AtomicUsize,
    requests_completed: AtomicUsize,
    requests_failed: AtomicUsize,
    tokens_generated: AtomicU64,
    
    // Gauges
    active_requests: AtomicUsize,
    kv_bytes_used: AtomicU64,
    
    // Histograms (simplified for MVP)
    ttft_sum_ms: AtomicU64,
    ttft_count: AtomicUsize,
    tokens_per_sec_sum: AtomicU64,
    tokens_per_sec_count: AtomicUsize,
}

impl Metrics {
    pub fn new() -> Self { ... }
    
    pub fn record_request_submitted(&self) { ... }
    pub fn record_request_completed(&self, ctx: &RequestContext) { ... }
    pub fn record_token_generated(&self) { ... }
    pub fn set_active_requests(&self, count: usize) { ... }
    pub fn set_kv_bytes_used(&self, bytes: u64) { ... }
    
    pub fn snapshot(&self) -> MetricsSnapshot { ... }
}

#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub requests_submitted: usize,
    pub requests_completed: usize,
    pub requests_failed: usize,
    pub tokens_generated: u64,
    pub active_requests: usize,
    pub kv_bytes_used: u64,
    pub avg_ttft_ms: f64,
    pub avg_tokens_per_sec: f64,
}
```

**Why**: Atomic counters enable lock-free metrics collection.

**Acceptance**:
- Metrics don't slow down hot path (< 1% overhead)
- Snapshot consistent
- Thread-safe under concurrent load

#### 5.2 Tracing Integration

**File**: `lightbulb/src/engine.rs`

**Add tracing spans**:
```rust
use tracing::{info_span, instrument};

impl Scheduler {
    #[instrument(skip(self, execute_batch))]
    pub async fn start<F>(&self, mut execute_batch: F) -> Result<()> {
        // ... existing code with spans:
        
        let batch_span = info_span!("assemble_batch");
        let _guard = batch_span.enter();
        let batch = self.assemble_batch(...)?;
        drop(_guard);
        
        let execute_span = info_span!("execute_batch", 
            batch_type = ?batch,
            batch_size = batch.len()
        );
        let _guard = execute_span.enter();
        let results = execute_batch(batch)?;
        drop(_guard);
        
        // ... etc
    }
}
```

**Why**: Distributed tracing enables performance debugging.

**Acceptance**:
- Spans visible in tracing output
- Nested spans correct
- Span overhead < 2% on hot path

#### 5.3 Metrics Exporter

**File**: `lightbulb/src/metrics.rs`

**Add**:
```rust
#[cfg(feature = "metrics-export")]
pub fn export_json(metrics: &Metrics) -> String {
    let snapshot = metrics.snapshot();
    serde_json::to_string_pretty(&snapshot).unwrap()
}

#[cfg(feature = "metrics-export")]
pub fn export_stdout(metrics: &Metrics) {
    let snapshot = metrics.snapshot();
    println!("=== Lightbulb Metrics ===");
    println!("Requests: {}/{} (submitted/completed)", 
        snapshot.requests_submitted, snapshot.requests_completed);
    println!("Tokens generated: {}", snapshot.tokens_generated);
    println!("Active requests: {}", snapshot.active_requests);
    println!("Avg TTFT: {:.2}ms", snapshot.avg_ttft_ms);
    println!("Avg tokens/sec: {:.2}", snapshot.avg_tokens_per_sec);
    println!("KV bytes used: {} ({:.2}MB)", 
        snapshot.kv_bytes_used, 
        snapshot.kv_bytes_used as f64 / 1024.0 / 1024.0);
}
```

**Cargo.toml**:
```toml
[features]
default = []
metrics-export = ["serde_json"]

[dependencies]
serde_json = { version = "1.0", optional = true }
```

**Why**: Feature-gated export keeps binary lean for production.

**Acceptance**:
- JSON export valid
- Stdout export readable
- Feature flag works correctly

---

## Testing Strategy

### Unit Tests

**File**: `lightbulb/src/engine.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_request_state_transitions() {
        // Valid: Pending → Prefilling → Decoding → Completed
        // Invalid: Decoding → Prefilling
    }
    
    #[test]
    fn test_kv_pager_alloc_free() {
        // Allocate pages, free them, verify no leaks
    }
    
    #[test]
    fn test_kv_pager_eviction() {
        // Fill pager to capacity, force eviction, verify LRU
    }
    
    #[test]
    fn test_batch_assembly() {
        // Submit prefill + decode requests
        // Verify prefill batch assembled first
    }
}
```

### Integration Tests

**File**: `lightbulb/tests/continuous_batching_test.rs`

```rust
#[tokio::test]
async fn test_8_concurrent_requests() {
    // End-to-end test with real (tiny) model
}

#[tokio::test]
async fn test_correctness_vs_baseline() {
    // Token-by-token regression test
}

#[tokio::test]
async fn test_10k_token_decode() {
    // Long-context stability test
}
```

### Acceptance Criteria Validation

| Criterion                            | Test                           | Pass? |
| ------------------------------------ | ------------------------------ | ----- |
| N>=8 concurrent requests without OOM | `test_8_concurrent_requests`   | ⏳     |
| Token-by-token correctness           | `test_correctness_vs_baseline` | ⏳     |
| Stable under 10k token decode        | `test_10k_token_decode`        | ⏳     |
| No KV cache corruption               | `test_kv_cache_integrity`      | ⏳     |
| Metrics < 1% overhead                | `bench_metrics_overhead`       | ⏳     |

---

## Dependencies and Configuration

### Cargo.toml Updates

```toml
[dependencies]
tokio = { version = "1.0", features = ["full"] }
tracing = "0.1"
tracing-subscriber = "0.3"
parking_lot = "0.12"
crossbeam = "0.8"
serde = { version = "1.0", features = ["derive"] }
serde_json = { version = "1.0", optional = true }

[features]
default = []
metrics-export = ["serde_json"]

[dev-dependencies]
tokio-test = "0.4"
criterion = "0.5"
```

### Configuration File

**File**: `lightbulb.toml` (example)

```toml
[scheduler]
max_batch_size = 8
max_batch_tokens = 4096
prefill_priority = true

[kv_cache]
page_size = 64  # tokens per page
max_pages_per_layer = 1024
use_kv_cache = true

[metrics]
export_interval_secs = 5
export_format = "json"  # or "stdout"
```

---

## Migration Path

### Backward Compatibility

The existing `Scheduler::run_single()` method remains functional for M0 use cases:

```rust
impl Scheduler {
    // NEW: Async continuous batching (default)
    pub async fn start<F>(&self, execute_batch: F) -> Result<()> { ... }
    
    // OLD: Single-request synchronous (deprecated but kept)
    pub fn run_single<F>(&self, req: &Request, generate_fn: F) -> Result<String> { ... }
}
```

CLI commands can opt into async batching via flag:

```bash
# Old synchronous mode (M0)
lightbulb local-llama-gen --model-dir ./model --prompt "Hello"

# New async batching mode (M1)
lightbulb local-llama-gen --model-dir ./model --prompt "Hello" --async-batch
```

---

## Timeline

| Week  | Focus              | Deliverables                                    |
| ----- | ------------------ | ----------------------------------------------- |
| **1** | Request Management | RequestContext, RequestQueue, unit tests        |
| **2** | Batch Assembly     | PrefillBatch, DecodeBatch, assemble_batch logic |
| **3** | KV Page Management | Enhanced KvPager, eviction, zero-copy design    |
| **4** | Continuous Loop    | Async Scheduler, integration tests              |
| **5** | Observability      | Metrics, tracing, exporters                     |

**Total**: 5 weeks (~1.25 months)

---

## Risks and Mitigations

### Risk 1: Candle KV Cache Integration Complexity

**Risk**: Candle's KV cache API may not support zero-copy handoff as designed.

**Mitigation**: 
- Design KvPager with abstraction layer
- Implement copy-based approach first, optimize later
- Document Candle API limitations for upstream contribution

### Risk 2: Deadlocks in Async Scheduler

**Risk**: Complex async locks could deadlock under concurrent load.

**Mitigation**:
- Use `parking_lot` for sync locks (faster, easier to reason about)
- Minimize lock scope
- Extensive concurrency testing with tools like `loom` (if needed)

### Risk 3: Metric Overhead

**Risk**: Atomic operations on hot path could degrade performance.

**Mitigation**:
- Feature-gate metrics export
- Use relaxed memory ordering where safe
- Benchmark before/after metrics integration

---

## Success Criteria

M1 is **COMPLETE** when:

✅ **Continuous Batching**:
- Handles ≥8 concurrent requests without OOM on CPU
- Token-by-token correctness matches single-request baseline (fixed seed)

✅ **Paged KV Management**:
- Stable under 10k token decode across multiple requests
- No cache corruption detected in stress tests
- Zero-copy handoff design documented (implementation optional)

✅ **Observability**:
- Metrics (TTFT, tok/s, active reqs, kv-bytes) exposed
- Feature-gated exporter (stdout or JSON)
- Tracing spans functional

✅ **Testing**:
- All unit tests pass
- Integration tests validate 8+ concurrent requests
- No regressions in M0 functionality

---

## Next Steps (Post-M1)

After M1 completion, proceed to **M1.5** (Hardware Adaptivity):
- Integrate `system-analysis` crate for hardware detection
- Automatic backend selection (CPU/CUDA/ROCm)
- Dynamic model size selection based on available resources

Then **M2** (Performance Enablers):
- StreamingLLM-style KV policy
- Prefix KV caching
- FlashAttention integration

---

## References

- **Current Code**: `lightbulb/src/engine.rs`, `lightbulb/src/cache.rs`
- **Tests**: `lightbulb/tests/integration_local_model.rs`
- **Dependencies**: Tokio, Tracing, Parking Lot, Crossbeam
- **Research**: MemOSA paper (tiered KV orchestration), vLLM (continuous batching)

---

**Document Status**: Draft v1.0  
**Author**: GitHub Copilot  
**Date**: October 19, 2025  
**Next Review**: Week 1 completion (Request Management phase)
