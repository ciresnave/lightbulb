# Lightbulb v1.0 Release Roadmap

> ## ⚠️ SUPERSEDED — this is not a status source
> 
> **`ROADMAP.md` is the single roadmap.** Its VERIFIED STATUS block carries
> figures measured at a named ref. This document's status claims were made at
> a state the project has left and have **not** been re-validated — treat them
> as provenance, not as findings.
> 
> It has **128 unchecked boxes and 0 checked**, and its own Phase 1 is
> COMPLETE — for months it declared the project blocked on work that had
> already shipped. Kept for provenance.

**Goal**: Production-ready inference server with streaming, authentication, and memory-aware scheduling

> ⚠️ **This document is stale and is kept for provenance.** Its phases were
> written against a state the project left long ago, and Phase 1 — which every
> later phase lists as a dependency — is complete. Do not plan from the
> timeline below. Current work is in `docs/superpowers/specs/` and
> `docs/superpowers/plans/`.

**Status as written**: "Core features implemented but server has compilation
errors. MemoryAwareScheduler exists but not integrated."

**Status verified 2026-08-15**: the server compiles (`cargo check -j 4 --bins`
→ exit 0), 643 library tests pass, and `MemoryAwareScheduler` is exported from
`src/engine/mod.rs:56`. Phases 2–5 have not been re-validated and may be stale
in the same direction.

---

## Phase 1: Fix Compilation Errors (Priority: CRITICAL) — ✅ COMPLETE

> **This phase is done. It was completed long before this note was added, and
> the document was never updated — so for months it declared the project
> blocked on work that had already shipped.** Anyone planning off the schedule
> below should read this section first.

**Verified complete 2026-08-15:**

```
$ cargo check -j 4 --bins
Finished `dev` profile [unoptimized + debuginfo] target(s) in 44.34s
EXIT=0
```

Warnings only (four unread struct fields in `lightbulb-cli`). Corroborating:
`cargo test -j 4 --lib` → `643 passed; 0 failed; 14 ignored`, and the server
serves real completions over HTTP in the integration suites.

Each sub-item below was resolved, in several cases by exactly the fix this
document prescribed:

| item | prescribed fix | state |
| --- | --- | --- |
| 1.1 E0255 module conflict | rename local `middleware` module | done — `src/api/mod.rs:30` reads `pub mod auth_middleware;` |
| 1.1 E0432 `engine::Scheduler` | create or drop the import | done — `src/engine/mod.rs:56` exports `MemoryAwareScheduler` and friends |
| 1.2 E0425 `from_fn_with_state` | qualify or alias the axum import | done — compiles |
| 1.3 E0282 type annotations | annotate `admin.rs:294` | done — compiles |
| 1.4 E0382/E0373 borrow errors | clone before move in `chat.rs` | done — compiles |
| 1.5 E0308 type mismatches | align `inference_tx` channel types | done — compiles |

**The timeline in "Summary" below is therefore measured from a state that no
longer exists**, since every later phase lists Phase 1 as a dependency. The
remaining phases have not been re-validated against the current tree and may be
stale in the same direction — treat their status claims as unverified rather
than as findings.

For current work, see `docs/superpowers/specs/` and `docs/superpowers/plans/`.

---

## Phase 2: Server Integration Testing (Priority: HIGH)
**Estimated Time**: 1-2 days  
**Dependencies**: Phase 1 complete

### 2.1 Database Setup Validation
**Action Items**:
- [ ] Verify `demo-setup.ps1` runs migrations successfully
- [ ] Confirm all four migrations apply without errors
- [ ] Test bootstrap admin key insertion
- [ ] Validate PostgreSQL connection from server

---

### 2.2 Server Startup Testing
**Action Items**:
- [ ] Run `cargo run --release --bin lightbulb` manually
- [ ] Verify server binds to port 8080
- [ ] Check logs for initialization errors
- [ ] Confirm health check endpoint responds: `curl http://localhost:8080/health`

---

### 2.3 API Endpoint Testing
**Action Items**:
- [ ] Test admin endpoint: `POST /v1/lightbulb/admin/api-keys`
  - Create user API key with admin token
  - Verify key returned and stored in database
  - Test role validation (user/admin/llm)
  - Test optional expiration parameter

- [ ] Test OpenAI endpoints with user key:
  - `GET /v1/models` - list available models
  - `POST /v1/chat/completions` (non-streaming)
  - `POST /v1/chat/completions` (streaming with `stream: true`)

- [ ] Test rate limiting:
  - Verify per-minute limits enforced
  - Confirm 429 response when exceeded

- [ ] Test authentication:
  - Verify requests without API key fail (401)
  - Verify requests with invalid key fail (401)
  - Verify requests with expired key fail (401)

---

### 2.4 CLI Testing
**Action Items**:
- [ ] Test CLI in non-streaming mode:
  ```bash
  cargo run --release --bin lightbulb-cli -- --api-key $LIGHTBULB_USER_KEY
  ```

- [ ] Test CLI in streaming mode:
  ```bash
  cargo run --release --bin lightbulb-cli -- --api-key $LIGHTBULB_USER_KEY --stream
  ```

- [ ] Verify conversation history maintained
- [ ] Test system prompt functionality
- [ ] Verify environment variable loading

---

### 2.5 Streaming Validation
**Action Items**:
- [ ] Confirm SSE format compliance (OpenAI-compatible)
- [ ] Verify token-by-token emission (not batched)
- [ ] Check `[DONE]` marker sent on completion
- [ ] Test error handling during streaming
- [ ] Validate finish_reason in final chunk

---

### Phase 2 Validation
**Success Criteria**:
- [ ] Server starts without crashes
- [ ] All API endpoints respond correctly
- [ ] Streaming works end-to-end
- [ ] CLI can interact with server
- [ ] Authentication and rate limiting functional

---

## Phase 3: MemoryAwareScheduler Integration (Priority: HIGH)
**Estimated Time**: 3-5 days  
**Dependencies**: Phase 2 complete

### 3.1 Architecture Design
**Current State**:
- ModelRunner processes InferenceJobs directly via channel
- MemoryAwareScheduler exists but bypassed
- SlotPool manages continuous batching but unused

**Target Architecture**:
```
Request → ApiServer → MemoryAwareScheduler → SlotPool → ModelRunner → Response
                            ↓
                    Priority Queue + Memory Tracking
```

**Design Decisions**:
- [ ] Define mapping: InferenceJob → Request (scheduler format)
- [ ] Determine callback mechanism: How scheduler notifies when slots available
- [ ] Plan batch coordination: How ModelRunner gets batched requests from SlotPool
- [ ] Memory estimation: Calculate KV cache requirements per request

---

### 3.2 Request Mapping Layer
**Files**: `src/engine/scheduler_bridge.rs` (NEW)

**Action Items**:
- [ ] Create bridge module to convert between formats
- [ ] Implement `InferenceJob → Request` conversion:
  ```rust
  pub fn job_to_request(job: InferenceJob) -> Request {
      Request {
          request_id: job.request_id,
          prompt_token_len: job.prompt_tokens.len(),
          max_output_tokens: job.max_tokens,
          priority: job.priority.unwrap_or(Priority::Normal),
      }
  }
  ```

- [ ] Implement reverse mapping for responses
- [ ] Handle streaming vs. complete response modes
- [ ] Maintain mapping table: RequestId → InferenceJob metadata

---

### 3.3 Scheduler Integration in ModelRunner
**Files**: `src/engine/model_runner.rs`

**Current Flow**:
```rust
loop {
    let job = rx.recv()?;
    // Process immediately
    let result = model_manager.decode(...);
    job.resp.send(result)?;
}
```

**Target Flow**:
```rust
loop {
    select! {
        // Receive new jobs and submit to scheduler
        Some(job) = rx.recv() => {
            let request = job_to_request(job);
            scheduler.submit_request(request, job.priority);
            pending_jobs.insert(request.request_id, job);
        }
        
        // Process batch when scheduler allocates slots
        Some(batch) = scheduler.get_ready_batch() => {
            let results = model_manager.forward_batch(batch)?;
            
            for (request_id, result) in results {
                if let Some(job) = pending_jobs.remove(&request_id) {
                    send_response(job, result);
                }
            }
        }
    }
}
```

**Action Items**:
- [ ] Add scheduler field to ModelRunner struct
- [ ] Implement dual-channel architecture:
  - Job submission channel (existing)
  - Batch ready notification channel (new)
- [ ] Create pending jobs HashMap: `RequestId → InferenceJob`
- [ ] Implement batch retrieval from scheduler
- [ ] Update `forward_batch` to handle multiple requests
- [ ] Preserve streaming support in batched mode

---

### 3.4 Memory Accounting
**Files**: `src/engine/model_runner.rs`, `src/engine/memory_aware_scheduler.rs`

**Action Items**:
- [ ] Calculate per-request memory requirements:
  ```rust
  fn estimate_memory(prompt_len: usize, max_output: usize, head_dim: usize, num_heads: usize) -> usize {
      let total_seq_len = prompt_len + max_output;
      let kv_cache_size = total_seq_len * head_dim * num_heads * 2 * 4; // 2 for K+V, 4 bytes per f32
      let activation_size = ...; // Model-specific
      kv_cache_size + activation_size
  }
  ```

- [ ] Update scheduler config with model parameters:
  ```rust
  MemoryAwareConfig {
      max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
      memory_per_slot_base: 100 * 1024 * 1024,
      memory_per_token: calculate_memory_per_token(model_config),
      eviction_pressure_threshold: 0.85,
      memory_safety_margin: 0.1,
  }
  ```

- [ ] Track actual memory usage during inference
- [ ] Implement memory pressure callbacks
- [ ] Add emergency request rejection when OOM risk detected

---

### 3.5 Priority and Fairness
**Files**: `src/engine/memory_aware_scheduler.rs`, `src/api/openai/chat.rs`

**Action Items**:
- [ ] Expose priority parameter in API:
  ```rust
  pub struct ChatCompletionRequest {
      // ... existing fields
      #[serde(default)]
      pub priority: Option<String>, // "low", "normal", "high", "critical"
  }
  ```

- [ ] Map string priority to enum:
  ```rust
  fn parse_priority(s: Option<String>) -> Priority {
      match s.as_deref() {
          Some("low") => Priority::Low,
          Some("high") => Priority::High,
          Some("critical") => Priority::Critical,
          _ => Priority::Normal,
      }
  }
  ```

- [ ] Implement FIFO-within-priority in scheduler (already done)
- [ ] Add metrics for priority queue depth by level
- [ ] Test fairness under load (ensure low priority eventually processed)

---

### 3.6 Continuous Batching Implementation
**Files**: `src/engine/model_runner.rs`, `src/engine/slot_pool.rs`

**Action Items**:
- [ ] Implement dynamic batch sizing:
  ```rust
  fn build_batch(scheduler: &MemoryAwareScheduler, max_batch: usize) -> Vec<Request> {
      let mut batch = Vec::new();
      while batch.len() < max_batch {
          if let Some(req) = scheduler.try_admit_next() {
              batch.push(req);
          } else {
              break; // No more admissible requests
          }
      }
      batch
  }
  ```

- [ ] Handle partial batch completion:
  - Some requests finish early (shorter sequences)
  - Continue batching with remaining requests + new arrivals
  
- [ ] Implement slot reuse:
  - When request finishes, immediately release slot
  - Notify scheduler to admit next waiting request

- [ ] Add batch coordination metrics:
  - Average batch size
  - Slot utilization %
  - Wait time per priority level

---

### Phase 3 Validation
**Tests**:
- [ ] Single request works (no regression)
- [ ] Multiple concurrent requests batched correctly
- [ ] Memory limits prevent OOM
- [ ] Priority ordering respected under load
- [ ] Streaming still works in batched mode
- [ ] Slots released and reused properly

**Load Testing**:
- [ ] 10 concurrent requests (should batch efficiently)
- [ ] 100 concurrent requests (should queue and process)
- [ ] Memory pressure test (exceed budget, verify rejection)
- [ ] Priority test (high priority bypasses low priority queue)

---

## Phase 4: Production Hardening (Priority: MEDIUM)
**Estimated Time**: 2-3 days  
**Dependencies**: Phase 3 complete

### 4.1 Error Handling
**Action Items**:
- [ ] Graceful degradation when model fails
- [ ] Timeout handling for long-running requests
- [ ] Proper error responses for all failure modes
- [ ] Request cancellation support
- [ ] Circuit breaker for repeated failures

---

### 4.2 Observability
**Action Items**:
- [ ] Structured logging (JSON format)
- [ ] Request tracing with correlation IDs
- [ ] Metrics endpoint (`/metrics` with Prometheus format)
  - Request latency histograms
  - Throughput (requests/sec)
  - Error rates by endpoint
  - Queue depth by priority
  - Memory usage over time
  - Slot utilization

- [ ] Health check improvements:
  - Deep health (check database, model loaded)
  - Liveness vs readiness probes

---

### 4.3 Configuration Management
**Action Items**:
- [ ] Move hardcoded values to config file
- [ ] Support environment variable overrides
- [ ] Validate configuration on startup
- [ ] Document all configuration options
- [ ] Provide example configs for common scenarios

---

### 4.4 Security Hardening
**Action Items**:
- [ ] Rate limiting per endpoint
- [ ] Request size limits (prevent DoS)
- [ ] API key rotation support
- [ ] Audit log retention policy
- [ ] HTTPS/TLS support
- [ ] CORS configuration

---

### 4.5 Documentation
**Action Items**:
- [ ] API reference documentation
- [ ] Architecture documentation
- [ ] Deployment guide
- [ ] Performance tuning guide
- [ ] Troubleshooting guide
- [ ] Migration guide (v0.x → v1.0)

---

## Phase 5: Testing and Validation (Priority: HIGH)
**Estimated Time**: 2-3 days  
**Dependencies**: Phases 1-4 complete

### 5.1 Unit Tests
**Action Items**:
- [ ] Test coverage for scheduler integration
- [ ] Test coverage for memory accounting
- [ ] Test coverage for batch coordination
- [ ] Test coverage for priority queue
- [ ] Test coverage for API endpoints

---

### 5.2 Integration Tests
**Action Items**:
- [ ] End-to-end streaming test
- [ ] Multi-user concurrent test
- [ ] Authentication and authorization test
- [ ] Rate limiting test
- [ ] Database failure recovery test

---

### 5.3 Performance Tests
**Action Items**:
- [ ] Latency benchmarks (P50, P95, P99)
- [ ] Throughput benchmarks (tokens/sec)
- [ ] Memory efficiency tests
- [ ] Batch size impact analysis
- [ ] Priority impact on latency

---

### 5.4 Stress Tests
**Action Items**:
- [ ] Sustained load test (1 hour at target RPS)
- [ ] Burst traffic test
- [ ] Memory pressure test
- [ ] Connection leak test
- [ ] Database connection pool exhaustion test

---

## Release Checklist

### Pre-Release
- [ ] All compilation errors fixed
- [ ] All tests passing (unit, integration, performance)
- [ ] Documentation complete and reviewed
- [ ] Security audit completed
- [ ] Performance benchmarks meet targets
- [ ] Demo scripts work on fresh install

### Release Artifacts
- [ ] Binary builds for major platforms
- [ ] Docker image published
- [ ] Helm chart (if Kubernetes deployment supported)
- [ ] Release notes
- [ ] Migration guide
- [ ] Security advisory (if applicable)

### Post-Release
- [ ] Monitor production deployments
- [ ] Gather user feedback
- [ ] Hotfix process ready
- [ ] v1.1 roadmap defined

---

## Timeline Estimate

| Phase                           | Duration | Cumulative |
| ------------------------------- | -------- | ---------- |
| Phase 1: Fix Compilation Errors | 1-2 days | 2 days     |
| Phase 2: Integration Testing    | 1-2 days | 4 days     |
| Phase 3: Scheduler Integration  | 3-5 days | 9 days     |
| Phase 4: Production Hardening   | 2-3 days | 12 days    |
| Phase 5: Testing & Validation   | 2-3 days | 15 days    |

**Total Estimated Time**: ~3 weeks (15 business days)

**Buffer for unknowns**: +1 week

**Target Release Date**: ~4 weeks from Phase 1 start

---

## Risk Assessment

### High Risk
- **Scheduler Integration Complexity**: Most complex change, affects core request flow
  - **Mitigation**: Feature flag to toggle between direct and scheduled processing
  - **Mitigation**: Incremental rollout (scheduler without batching, then add batching)

- **Performance Regression**: Scheduler overhead may increase latency
  - **Mitigation**: Benchmark before/after, optimize hot paths
  - **Mitigation**: Keep direct processing as fallback for single requests

### Medium Risk
- **Borrow Checker Issues**: Rust ownership in async context is tricky
  - **Mitigation**: Clone data liberally initially, optimize later
  - **Mitigation**: Use Arc/Mutex where needed for shared state

- **Memory Accounting Accuracy**: Estimates may not match actual usage
  - **Mitigation**: Conservative estimates with safety margin
  - **Mitigation**: Runtime monitoring and adjustment

### Low Risk
- **API Compatibility**: Changes should be backward compatible
  - **Mitigation**: Version API endpoints if breaking changes needed

---

## Success Metrics

### Functional
- ✅ Server compiles and runs without errors
- ✅ All API endpoints functional
- ✅ Streaming works correctly
- ✅ Authentication and rate limiting work
- ✅ Scheduler manages memory and priority correctly

### Performance
- **Latency**: P95 latency < 200ms (excluding model inference time)
- **Throughput**: Handle 100+ concurrent requests
- **Memory**: Stay within configured budget (no OOM crashes)
- **Batch Efficiency**: Average batch size > 4 under load

### Reliability
- **Uptime**: Server runs 24+ hours without restart
- **Error Rate**: < 0.1% errors under normal load
- **Recovery**: Graceful handling of database/model failures

---

## Notes

- **Incremental Approach**: Each phase is independently testable
- **Feature Flags**: Consider adding flags for:
  - Enable/disable scheduler (use direct processing as fallback)
  - Enable/disable batching (use per-request processing)
  - Enable/disable memory limits (allow unlimited for testing)

- **Backwards Compatibility**: Existing demo scripts and CLI should work unchanged

- **Future Enhancements** (post v1.0):
  - Multi-model support (route requests to different models)
  - Model warm-up and preloading
  - KV cache compression (H2O, quantization)
  - Speculative decoding
  - Multi-GPU support (already scaffolded)
