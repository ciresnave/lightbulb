# Continuous Batching Implementation Design

**Date:** November 24, 2025  
**Status:** Design Phase  
**Target:** Dynamic request scheduling for 20-40% GPU utilization improvement

---

## Executive Summary

**Goal:** Implement continuous batching (aka "iteration-level batching") to allow requests to join and leave batches dynamically at each decode step.

**Expected Benefits:**
- 20-40% better GPU utilization
- Lower average latency (no waiting for batch to fill)
- Higher throughput at variable load
- Better tail latency (p95, p99)

**Implementation Effort:** 3-4 weeks  
**Complexity:** Medium-High

---

## Problem Statement

### Current System: Static Batching

**How it works:**
1. Collect N requests
2. Form batch of size N
3. Generate tokens for all N requests
4. Wait for ALL requests to complete
5. Release batch, collect new requests

**Issues:**
```
Request Lifecycle:
A: ████████████████ (16 tokens)
B: ████████ (8 tokens, waits 8 steps idly)
C: ████ (4 tokens, waits 12 steps idly)
   ↑
   Batch starts

Wasted GPU cycles: B waits 8 steps, C waits 12 steps
Latency: A blocks B and C from finishing early
```

### Desired: Continuous Batching

**How it should work:**
1. Maintain request queue
2. Form batch from available requests
3. Generate ONE token per request
4. Remove completed requests
5. Add new requests from queue
6. GOTO step 3 (dynamic batch)

**Benefits:**
```
Request Lifecycle:
A: ████████████████ (16 tokens)
B: ████████ (8 tokens, finishes early)
C: ████ (4 tokens, finishes early)
D:         ████████ (joins mid-flight)
E:             ████ (joins later)
   ↑       ↑   ↑
   New requests join dynamically

✓ GPU always busy (batch size adapts)
✓ Lower latency (finish early)
✓ Higher throughput (no idle time)
```

---

## Architecture Design

### 1. Request Queue

```rust
pub struct ContinuousBatchScheduler {
    // Pending requests waiting to be processed
    pending_queue: VecDeque<RequestContext>,
    
    // Active requests currently being processed
    active_batch: Vec<RequestContext>,
    
    // Configuration
    config: SchedulerConfig,
    
    // Statistics
    stats: SchedulerStats,
}

pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub min_batch_size: usize, // Don't run with batch < min (inefficient)
    pub max_wait_time_ms: u64, // Max time to wait for batch to fill
    pub prefill_batch_size: usize, // Separate limit for prefill
}
```

### 2. Batch Formation Logic

```rust
impl ContinuousBatchScheduler {
    pub fn get_next_batch(&mut self) -> Option<Vec<RequestContext>> {
        let now = Instant::now();
        
        // Separate prefill and decode requests
        let (prefill, decode) = self.partition_by_phase();
        
        // Prioritize decode (lower latency impact)
        let mut batch = decode;
        
        // Add prefill requests if space available
        let remaining_slots = self.config.max_batch_size - batch.len();
        let prefill_to_add = prefill.into_iter()
            .take(remaining_slots.min(self.config.prefill_batch_size));
        batch.extend(prefill_to_add);
        
        // Enforce minimum batch size (unless waiting too long)
        if batch.len() < self.config.min_batch_size {
            let oldest_wait = self.oldest_request_age();
            if oldest_wait < self.config.max_wait_time_ms {
                return None; // Wait for more requests
            }
        }
        
        Some(batch)
    }
    
    fn partition_by_phase(&mut self) -> (Vec<RequestContext>, Vec<RequestContext>) {
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        
        for req in self.active_batch.drain(..) {
            match req.phase() {
                Phase::Prefill => prefill.push(req),
                Phase::Decode => decode.push(req),
            }
        }
        
        // Add new requests from queue
        while let Some(req) = self.pending_queue.pop_front() {
            if prefill.len() + decode.len() >= self.config.max_batch_size {
                self.pending_queue.push_front(req); // Put back
                break;
            }
            prefill.push(req);
        }
        
        (prefill, decode)
    }
}
```

### 3. Dynamic Batch Adjustment

```rust
impl ParallelModelManager {
    pub fn step_continuous(&mut self) -> Result<Vec<CompletedRequest>> {
        // Get next batch (may add/remove requests)
        let mut batch = self.scheduler.get_next_batch()
            .ok_or(anyhow!("No requests to process"))?;
        
        // Allocate cache for new requests
        for req in &batch {
            if !self.cache.has_slot(req.id()) {
                self.cache.allocate_slot(req.id())?;
            }
        }
        
        // Forward pass
        let tokens = self.model.forward_batch(&mut batch)?;
        
        // Update request states
        let mut completed = Vec::new();
        let mut still_active = Vec::new();
        
        for mut req in batch {
            if req.should_continue() {
                still_active.push(req);
            } else {
                // Cleanup cache
                self.cache.free_slot(req.id())?;
                completed.push(req.into_completed());
            }
        }
        
        // Put active requests back for next iteration
        self.scheduler.active_batch = still_active;
        
        Ok(completed)
    }
}
```

### 4. Prefill/Decode Separation

**Challenge:** Prefill and decode have different compute characteristics.

- **Prefill:** Compute-bound, O(n²) attention, benefits from large batches
- **Decode:** Memory-bound, O(n) attention, benefits from smaller batches

**Solution:** Separate scheduling policies

```rust
pub enum BatchType {
    PrefillOnly,      // Only prefill requests
    DecodeOnly,       // Only decode requests
    Mixed(Ratio),     // Mix prefill + decode (ratio: prefill/decode)
}

impl ContinuousBatchScheduler {
    pub fn select_batch_type(&self) -> BatchType {
        let prefill_count = self.count_prefill_requests();
        let decode_count = self.count_decode_requests();
        
        match (prefill_count, decode_count) {
            (0, 0) => BatchType::DecodeOnly, // Default
            (0, _) => BatchType::DecodeOnly,
            (_, 0) => BatchType::PrefillOnly,
            (p, d) => {
                // Use mixed if both exist and load is balanced
                if p as f32 / d as f32 < 0.5 {
                    // Mostly decode, prioritize decode-only batches
                    BatchType::DecodeOnly
                } else {
                    // Mixed batch: 25% prefill, 75% decode
                    BatchType::Mixed(Ratio::new(1, 3))
                }
            }
        }
    }
}
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

**Tasks:**
1. ✅ Design `ContinuousBatchScheduler` struct
2. Implement request queue (VecDeque)
3. Add `get_next_batch()` logic
4. Implement batch formation rules

**Deliverables:**
- `src/engine/continuous_scheduler.rs` (new file)
- Unit tests for scheduling logic

### Phase 2: Integration (Week 2)

**Tasks:**
1. Integrate scheduler into `ParallelModelManager`
2. Add `step_continuous()` method
3. Handle cache allocation/deallocation per step
4. Implement completed request extraction

**Deliverables:**
- Updated `parallel_model_manager.rs`
- Integration tests

### Phase 3: Prefill/Decode Separation (Week 3)

**Tasks:**
1. Implement `BatchType` selection logic
2. Add prefill-only and decode-only batch modes
3. Optimize mixed batching ratio
4. Benchmark prefill vs decode performance

**Deliverables:**
- Optimized scheduling policies
- Performance benchmarks

### Phase 4: Optimizations (Week 4)

**Tasks:**
1. Add priority queue for requests (latency-sensitive)
2. Implement preemption (pause low-priority requests)
3. Add request deadline tracking
4. Optimize for tail latency (p95, p99)

**Deliverables:**
- Advanced scheduling features
- Production-ready system

---

## Performance Analysis

### Expected Improvements

#### Scenario 1: Variable Request Arrival
**Current (Static Batching):**
```
Batch 1: [A, B] (wait for 2 requests)
  Generate 10 tokens
  A finishes at token 5
  B finishes at token 10
  GPU idle waiting for A to finish

Batch 2: [C, D] (wait for 2 requests)
  ...
```
**Throughput:** ~50% GPU utilization (waiting time + idle time)

**Continuous Batching:**
```
Step 1: [A, B]
Step 2: [A, B, C] (C joins)
Step 3: [A, B, C, D] (D joins)
Step 4: [A, B, C, D]
Step 5: [B, C, D] (A completes)
...
```
**Throughput:** ~85% GPU utilization (always busy)
**Improvement:** **70% increase in throughput**

#### Scenario 2: Bursty Traffic
**Current:**
- Requests arrive in bursts
- Must wait for batch to fill
- High latency variance

**Continuous:**
- Process available requests immediately
- No wait time for batch formation
- Lower p95/p99 latency

**Latency Improvement:**
- P50: -10-20% (small benefit)
- P95: -30-50% (large benefit)
- P99: -50-70% (huge benefit)

---

## Challenges & Solutions

### Challenge 1: KV Cache Fragmentation

**Problem:** Continuous batching causes scattered cache allocation/deallocation.

**Solution:** Use PagedAttention-style memory management
- Allocate cache in fixed-size pages (e.g., 16 tokens)
- Maintain free page list
- Link pages for long sequences

```rust
pub struct PagedKvCache {
    pages: Vec<Page>,           // Physical pages
    free_pages: Vec<PageId>,    // Available pages
    slot_to_pages: HashMap<SlotId, Vec<PageId>>, // Logical → physical
}
```

**Benefit:** No fragmentation, efficient memory use

### Challenge 2: Attention Computation Complexity

**Problem:** Variable batch sizes require dynamic attention mask construction.

**Solution:** Pre-compute attention masks or use FlashAttention's built-in masking
```rust
// FlashAttention handles variable lengths natively
flash_attn(
    &q, &k, &v,
    softmax_scale,
    causal: true, // Automatically masks future tokens
)
```

### Challenge 3: Fairness

**Problem:** Long-running requests may starve new requests.

**Solution:** Implement fairness policies
- Round-robin scheduling
- Token budget per request (preempt after N tokens)
- Priority levels (interactive > batch)

```rust
pub enum SchedulingPolicy {
    FIFO,              // First-in-first-out
    RoundRobin,        // Rotate through requests
    ShortestRemaining, // Prioritize nearly-done requests
    Priority(u8),      // Explicit priority levels
}
```

---

## Testing Strategy

### Unit Tests
- Request queue operations (enqueue, dequeue, reorder)
- Batch formation logic (edge cases: empty, overflow)
- Phase partitioning (prefill vs decode)

### Integration Tests
- Single request correctness
- Multiple requests (variable lengths)
- Dynamic joining/leaving
- Cache allocation/deallocation

### Performance Tests
- Throughput vs static batching
- Latency (P50, P95, P99)
- GPU utilization (target >85%)
- Memory efficiency

### Stress Tests
- High request rate (1000+ req/s)
- Bursty traffic patterns
- Long-running requests mixed with short
- Cache pressure (>90% capacity)

---

## Configuration

### Recommended Settings

**Low Latency (Interactive Chat):**
```toml
[scheduler]
max_batch_size = 8
min_batch_size = 1
max_wait_time_ms = 10
prefill_batch_size = 4
```

**High Throughput (Batch Processing):**
```toml
[scheduler]
max_batch_size = 32
min_batch_size = 8
max_wait_time_ms = 100
prefill_batch_size = 16
```

**Balanced (Production):**
```toml
[scheduler]
max_batch_size = 16
min_batch_size = 4
max_wait_time_ms = 50
prefill_batch_size = 8
```

---

## Rollout Plan

### Phase 1: Experimental Flag
- Add `--continuous-batching` CLI flag
- Default to static batching (stable)
- Gather user feedback

### Phase 2: Opt-In
- Enable continuous batching via config
- Document benefits and trade-offs
- Provide migration guide

### Phase 3: Default
- Make continuous batching default
- Deprecate static batching mode
- Remove old code after 1-2 releases

---

## Monitoring

### Key Metrics
- `scheduler_queue_length` - Pending requests
- `scheduler_batch_size` - Current batch size
- `scheduler_join_rate` - Requests/sec joining batch
- `scheduler_completion_rate` - Requests/sec completing
- `scheduler_gpu_utilization` - % GPU busy

### Alerts
- Queue length > 100 (capacity issue)
- Avg batch size < min_batch_size (underutilization)
- Completion rate < join rate (backlog growing)

---

## References

### Papers
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

### Implementations
- [vLLM](https://github.com/vllm-project/vllm) - Reference continuous batching
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - In-flight batching
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)

---

**Next Steps:**
1. Create `src/engine/continuous_scheduler.rs` skeleton
2. Write unit tests for scheduling logic
3. Benchmark current static batching baseline
4. Implement Phase 1 (core infrastructure)

**Status:** 📝 Design complete, ready for implementation
