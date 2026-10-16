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

> ⚠️ **Audited against automated test coverage 2026-10-14, at `origin/main` `cee4e324`.** This
> phase's items were written as a MANUAL testing checklist. Re-auditing them found most are
> actually about whether an **automated** test covers the claim, not whether someone ran the
> step by hand once — so each item below is now checked only when a specific test function
> proves it, named inline. Unchecked items are not "not yet run manually"; they are **genuinely
> untested**, confirmed by grep with a positive control (the same query style finding real hits
> elsewhere), not by absence alone. One item (`[DONE]` marker, 2.5) is not a missing test — the
> behavior itself does not exist in the server.
>
> `[x] (gated)` means the test exists and asserts the claim, but is `#[ignore] // Requires
> PostgreSQL` and does **not** run under plain `cargo test` — only `cargo test -- --ignored`
> against a live Postgres instance. Treat these as "would catch a regression if run with a DB,"
> not "runs in CI by default."

### 2.1 Database Setup Validation
**Action Items**:
- [ ] Verify `demo-setup.ps1` runs migrations successfully — NOT COVERED. `grep -rn
  "demo-setup\|demo_setup" tests/ src/` → 0 hits (control: the same style grep for `"bootstrap"`
  finds real hits in `demo-setup.ps1`, confirming the search works).
- [ ] Confirm all five migrations (`V1`–`V5`, not four — the item itself is stale on the count)
  apply without errors — NOT COVERED. No test runs or validates migration application;
  `tests/api_integration_tests.rs`'s `create_test_server()` assumes a pre-migrated DB.
- [ ] Test bootstrap admin key insertion — NOT COVERED. `provision_test_api_key`
  (`tests/api_integration_tests.rs:44-69`) inserts an arbitrary admin-role key for test
  convenience; that is not the `demo-setup.ps1` bootstrap path.
- [ ] Validate PostgreSQL connection from server — NOT COVERED. `health_check`
  (`src/api/mod.rs:708-710`) never touches `db_pool`; no test asserts DB-connectivity reporting.

---

### 2.2 Server Startup Testing
**Action Items**:
- [ ] Run `cargo run --release --bin lightbulb` manually — inherently a manual step, no automated
  test runs the binary's `main()`.
- [ ] Verify server binds to port 8080 — NOT COVERED. `ApiConfig::default().bind_address` is never
  asserted by `test_api_config_default` (`src/api/mod.rs:717-722`), which only checks the
  `enable_*` flags.
- [ ] Check logs for initialization errors — NOT COVERED. No test captures/asserts startup log
  output. (A narrower, different claim IS tested: `api_server_new_refuses_default_config_without_auth`,
  `src/api/mod.rs:828-842`, asserts `ApiServer::new` errors when auth isn't configured.)
- [x] (gated) Confirm health check endpoint responds — `test_health_check`,
  `tests/api_integration_tests.rs:125-142`: builds the real router, `GET /health`, asserts `200`.

---

### 2.3 API Endpoint Testing
**Action Items**:
- [ ] Test admin endpoint `POST /v1/lightbulb/admin/api-keys` — create key, verify returned/stored
  in DB — NOT COVERED. Every `admin.rs` test uses `db_pool: None` and stops at the `503` before
  the DB insert path (`create_api_key_allows_an_admin_key_requesting_admin_role`,
  `src/api/admin.rs:473-491`); no test with a real pool exercises the insert or response shape.
- [x] — role validation (user/admin/llm) — `src/api/admin.rs:432-514`:
  `create_api_key_refuses_a_user_key_requesting_admin_role` (433),
  `create_api_key_refuses_an_llm_key_requesting_admin_role` (452),
  `create_api_key_allows_an_admin_key_requesting_admin_role` (474),
  `create_api_key_allows_an_admin_key_requesting_llm_role` (502). Minor residual gap: no case for
  an `llm` key requesting `user`/`llm`.
- [ ] — optional expiration parameter — NOT COVERED. `expires_in_seconds`/`expires_at`
  (`src/api/admin.rs:225,305,349-377`) is implemented but no test ever sets or checks it.
- [ ] `GET /v1/models` list available models — NOT COVERED, and the item itself is wrong: the
  route is registered as `post(models::list_models)` (`src/api/openai/mod.rs:19`), not `GET`.
  `src/api/openai/models.rs` has no test module at all.
- [x] (gated for the full-router case) `POST /v1/chat/completions` non-streaming — two layers:
  handler-level, `a_contract_completion_is_recorded_once_with_its_final_finish_reason`
  (`src/api/openai/chat.rs:2933-3013`); full-router, `test_chat_completions_with_auth`
  (`tests/api_integration_tests.rs:175-205`, `#[ignore] // Requires PostgreSQL`).
- [x] (handler-level only) `POST /v1/chat/completions` streaming (`stream: true`) —
  `a_streaming_completion_is_recorded_once_with_the_reason_the_runner_reported`
  (`src/api/openai/chat.rs:2827-2871`) calls `chat_completions()` directly, not through the
  auth-wrapped router — no test covers streaming combined with real auth/rate-limit middleware.
- [ ] Rate limiting — per-minute limits enforced — NOT COVERED. `rate_limit_middleware`
  (`src/api/auth_middleware.rs:274-340`) exists but has zero tests. (`test_rate_limiter`,
  `src/engine/production_providers.rs:601-607`, is an unrelated `RateLimiter` struct for
  knowledge-base providers, not the HTTP API limiter — confirmed by reading it.)
- [ ] Rate limiting — 429 when exceeded — NOT COVERED, same reason; `TOO_MANY_REQUESTS`
  (`auth_middleware.rs:327`) is never reached by a test. (A *different* 429 path, the pre-auth
  attempt limiter, IS tested, but it is a distinct mechanism from the per-key rate limit here.)
- [x] (gated) Auth — missing key → 401 — `test_chat_completions_without_auth`
  (`tests/api_integration_tests.rs:144-173`, `#[ignore] // Requires PostgreSQL`).
- [ ] Auth — invalid key → 401 — NOT COVERED. `grep -i "invalid.*key" tests/` → 0 relevant hits
  (control: the same grep style found an unrelated real hit elsewhere, confirming it runs).
  `auth_middleware.rs`'s own test module (447-492) only tests `effective_client_ip`, not the
  key-lookup/401 branch (`auth_middleware.rs:226-237`).
- [ ] Auth — expired key → 401 — NOT COVERED. `grep -i "expired" tests/` → 0 hits. The
  `expires_at > NOW()` check (`auth_middleware.rs:208`) is implemented but never exercised.

---

### 2.4 CLI Testing
**Updated 2026-10-14**: `non_stream_chat`/`stream_chat` take `client`/`base_url` as plain
parameters, so they're testable against a real local server (axum, already a dependency) with no
mocking library needed. The remaining three items are inside `main()`'s interactive stdin loop,
genuinely harder to test without stdin mocking — still open.
- [x] Test CLI in non-streaming mode — `non_stream_chat_returns_the_assistant_content_and_usage`,
  `non_stream_chat_surfaces_the_servers_error_message` (`lightbulb-cli.rs` test module)
- [x] Test CLI in streaming mode —
  `stream_chat_concatenates_token_deltas_and_stops_at_the_done_sentinel`,
  `stream_chat_surfaces_the_servers_error_message`
- [ ] Verify conversation history maintained (`conversation: Vec<Message>`, `lightbulb-cli.rs:135-225`)
- [ ] Test system prompt functionality (`args.system`, `lightbulb-cli.rs:138-141`)
- [ ] Verify environment variable loading (`LIGHTBULB_API_KEY`, `lightbulb-cli.rs:119`)

---

### 2.5 Streaming Validation
**Action Items**:
- [ ] Confirm SSE format compliance (OpenAI-compatible) — NOT COVERED as its own claim. Tests
  parse `data: ` lines incidentally while testing other things; no test asserts
  `Content-Type: text/event-stream` or general framing (`grep -n "text/event-stream" src/api/openai/chat.rs`
  → 0 hits).
- [ ] Verify token-by-token emission (not batched) — NOT COVERED. Every streaming test stub
  (`stub_runner_with`, `chat.rs:2038-2069`) sends exactly one `StreamItem::Token` before `Done`;
  no test sends multiple tokens to prove they arrive unbatched.
- [ ] Check `[DONE]` marker sent on completion — **NOT A MISSING TEST, A MISSING BEHAVIOR.**
  `grep -n "DONE" src/` finds only the *client* parsing `[DONE]` (`lightbulb-cli.rs:331`) from a
  server that never sends it — `create_chat_stream` (`chat.rs:1313-1501`) ends the stream with a
  `finish_reason`-carrying JSON frame, never a literal `data: [DONE]` line. A real
  OpenAI-incompatibility; filed as its own item, not fixed here.
- [ ] Test error handling during streaming — NOT COVERED. The `Err(e) =>` SSE-error-frame branch
  exists (`chat.rs:1447-1457`) and is fed from three production call sites in
  `model_runner.rs`, but no test constructs a stub that sends an `Err` through the stream channel.
- [x] Validate finish_reason in final chunk —
  `a_streaming_completion_is_recorded_once_with_the_reason_the_runner_reported`
  (`chat.rs:2827-2871`) asserts `frames[1]["choices"][0]["finish_reason"]` for both `Stop` and
  `Length`.

---

### Phase 2 Validation — reconciled, not re-checked wholesale
**Of ~24 original line items, as of 2026-10-14: 7 covered (2 of those gated behind a live
Postgres), 17 not covered.** The `[DONE]` SSE marker gap was a real missing behavior, now fixed
(#138). The CLI binary's two network-facing functions now have tests (this entry); its three
stdin-loop items remain open. The success criteria below were never independently meaningful
(they just restate the items above at a coarser grain) and are retired as a separate checklist
rather than re-audited. The real open gap left, in priority order if this phase is picked up
again: the per-key rate limiter has zero test coverage at all (needs a live Postgres to verify
here, not available in this environment).

---

## Phase 3: MemoryAwareScheduler Integration (Priority: HIGH)
**Estimated Time**: 3-5 days
**Dependencies**: Phase 2 complete — **not actually true in practice**: per the 2026-10-14 audit
above, Phase 2 is 5/24 covered, not complete, and this project's own `ROADMAP.md` already
documents that this file's phase dependencies don't hold in general (its Phase 1 shipped while
the doc still called the whole project blocked). The PM explicitly directed starting Phase 3's
slice 1 (see `ROADMAP.md`'s "M4 — MemoryAwareScheduler integration: slice plan") without waiting
on Phase 2 — a deliberate call, not an oversight.

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

> ⚠️ **Audited against real code 2026-10-14, at `origin/main` `58396d3`.** Same method as the
> Phase 2 audit: each item checked against actual code (file:line evidence), not re-run as a
> manual checklist. Unlike Phase 2 (5/24 covered, mostly just untested), **Phase 4 is
> substantially unimplemented**: roughly 3/27 fully done, 6/27 partial, ~18/27 genuinely missing
> — the unchecked `[ ]` boxes below are, for once, an accurate reading, not stale pessimism.
> Several gaps are worse than "missing" — a real defect wearing the shape of a feature:
> - **The admin `/metrics`-adjacent endpoint is a hardcoded zero stub.** `system_metrics()`
>   (`src/api/admin.rs:209-219`) literally returns `cpu_usage_percent: 0.0`,
>   `memory_total_bytes: 0`, `gpu_metrics: vec![]`, with a `// TODO` comment — not partially
>   working, never wired to anything. There is no Prometheus-format endpoint at all (no
>   `prometheus`/`metrics` crate in `Cargo.toml`).
> - **`PipelineConfig.timeout_ms` is dead configuration.** Has a real default
>   (`Some(300_000)`, `src/engine/pipeline.rs:284`) and looks like a working timeout, but is
>   never read anywhere outside its own struct. `grep -rn "tokio::time::timeout" src/` → 0 hits
>   in the whole crate — there is no request-timeout enforcement anywhere, despite the config
>   field implying there is.
> - **API key management is create-only.** No revoke or rotate endpoint exists
>   (`grep -rni "revoke\|delete.*api_key" src/` → 0 hits) — a compromised key cannot be
>   invalidated through the API at all.
> - **`tracing-subscriber`'s `json` Cargo feature isn't enabled** (`Cargo.toml:137`:
>   `features = ["env-filter"]` only) — "structured JSON logging" isn't an unused capability,
>   the crate can't produce it without a dependency change first.
> - CORS (`src/api/mod.rs:532-537`) and rate limiting (`auth_middleware.rs:274-340`) both have
>   real, serving-path mechanisms — not dead code — but neither has the granularity its item
>   implies: CORS is unconditionally `allow_origin(Any)` with no policy knob at all, and the
>   rate limiter is one global per-API-key limit with no per-endpoint logic, silently disabled
>   whenever no database is configured.

### 4.1 Error Handling
**Action Items**:
- [x] Graceful degradation when model fails — `queue_error_response()`
  (`src/api/openai/chat.rs:486-496`) maps `QueueFullError` → 503, everything else → 500 with a
  JSON body; `grep -n "panic!\|\.expect(" src/engine/model_runner.rs` → 0 hits.
- [ ] Timeout handling for long-running requests — NOT DONE. `tokio::time::timeout` has 0 uses
  anywhere in `src/` (positive control: the same grep style for `StatusCode::` finds 94 real
  hits). See `PipelineConfig.timeout_ms` note above — the config field exists, the enforcement
  does not.
- [ ] Proper error responses for all failure modes — PARTIAL. Real differentiated 400/503/500
  responses exist for validation and queue/runner failures, but request-size-DoS and timeout
  failure modes have no error path because the underlying checks (below) don't exist.
- [ ] Request cancellation support — NOT DONE. `grep -rn "CancellationToken" src/` → 0 hits; the
  streaming handler has no logic reacting to a dropped client connection.
- [ ] Circuit breaker for repeated failures — NOT DONE. `grep -rin "breaker" src/` → 0 hits.

---

### 4.2 Observability
**Action Items**:
- [x] Structured logging (JSON format) — DONE (Phase 4 hardening item 4, #154). The
  `tracing-subscriber` `json` feature is enabled; `lightbulb::logging::init_tracing`, shared by
  `main.rs` and `lightbulb-probe.rs`, switches to `fmt::layer().json()` when
  `LIGHTBULB_LOG_FORMAT=json` is set (default stays plain-text, unchanged for existing
  deployments). Per-request `println!`/`eprintln!` call sites in `src/api/admin.rs` and
  `src/api/auth_middleware.rs` still bypass tracing and would mix plain lines into JSON output
  during live serving -- same class of defect, not fixed here, tracked as its own item.
- [ ] Request tracing with correlation IDs — NOT DONE. `TraceLayer::new_for_http()`
  (`src/api/mod.rs:538`) is tower-http's internal span logging, not a client-visible correlation
  ID; `grep -rn "x-request-id\|SetRequestIdLayer\|MakeRequestUuid" src/` → 0 hits. The
  `RequestId` types that do exist (`slot_pool.rs:60`, `batched_llama.rs:36`,
  `batch_metadata.rs:13`) are internal scheduler bookkeeping, never propagated to HTTP.
- [ ] Metrics endpoint (`/metrics` with Prometheus format) — NOT DONE, worse than missing (see
  note above: hardcoded-zero stub, wrong route, wrong format, no dependency).
  - [ ] Request latency histograms — NOT DONE (`grep -rn "histogram\|Histogram" src/` finds only
    an unrelated GGUF dtype-count comment)
  - [ ] Throughput (requests/sec) — NOT DONE
  - [ ] Error rates by endpoint — NOT DONE
  - [ ] Queue depth by priority — NOT DONE
  - [ ] Memory usage over time — NOT DONE
  - [ ] Slot utilization — NOT DONE
- [x] Health check improvements — DONE. `GET /health` stays the bare liveness check
  (`health_check()`, `src/api/mod.rs`); added `GET /health/ready` (`readiness_check`) for
  deep health, both outside the auth stack so a load balancer with no bearer token can read
  either. Checks `db_pool` (via a real `SELECT 1`, when a database is configured) and
  `inference_tx` (vs. `config.models_dir`, so "never asked for a model" reads
  `not_configured` and "asked for one and it failed to start" reads `not_loaded` and blocks
  readiness) fresh on every call. Returns 503 + JSON detail when not ready, 200 when ready.

---

### 4.3 Configuration Management
**Action Items**:
- [ ] Move hardcoded values to config file — NOT DONE. No app config file format exists (only
  `Cargo.toml`/`rust-toolchain.toml`); `rate_limit_per_minute`, `model_max_batch_size`,
  `model_context_length`, `model_max_queue_depth`, and all of `MemoryAwareConfig`
  (`main.rs:72-107`) are hardcoded constants with no file or env override.
- [x] Support environment variable overrides — PARTIAL, credited as done for what it covers:
  `main.rs:24-69` reads 8 real env vars (`DATABASE_URL`, `LIGHTBULB_BIND_ADDRESS`,
  `LIGHTBULB_MODELS_DIR`, `LIGHTBULB_DEFAULT_MODEL`, `LIGHTBULB_JWT_SECRET`,
  `LIGHTBULB_TRUSTED_PROXIES`, `LIGHTBULB_MAX_AUTH_ATTEMPTS_PER_MINUTE_PER_IP`,
  `SECURITY_ALERT_EMAIL`) — but the hardcoded values above have no env equivalent at all, so
  coverage is partial, not systemic.
- [x] Validate configuration on startup — PARTIAL, real and wired:
  `validate_auth_policy()` (`src/api/mod.rs:216-245`) is actually called from `ApiServer::new`
  (`mod.rs:318`) and refuses to start on an unsafe no-DB/no-auth combination — but it validates
  only the auth policy, nothing else (no checks on batch sizes, context length, port conflicts).
- [ ] Document all configuration options — NOT DONE. All 8 env vars above are documented only as
  inline Rust comments; `grep -rln "LIGHTBULB_BIND_ADDRESS\|LIGHTBULB_JWT_SECRET" docs/ *.md` →
  0 hits.
- [ ] Provide example configs for common scenarios — NOT DONE. No `.env.example` or
  `config.example.*` file exists.

---

### 4.4 Security Hardening
**Action Items**:
- [ ] Rate limiting per endpoint — PARTIAL (see note above): real, Postgres-backed, but one
  global per-key limit applied uniformly, not per-endpoint, and a no-op with no database.
- [ ] Request size limits (prevent DoS) — NOT DONE. `grep -rn "RequestBodyLimitLayer\|
  DefaultBodyLimit\|body_limit" src/` → 0 hits anywhere.
- [ ] API key rotation support — NOT DONE (see note above: create-only, no revoke).
- [ ] Audit log retention policy — NOT DONE. `grep -rni "retention\|purge" src/` → only an
  unrelated KV-cache-compression comment about token retention, not audit logs.
- [x] HTTPS/TLS support — DONE, confirmed live serving-path code, not a dead type:
  `ApiServer::serve()` (`src/api/mod.rs:542-650+`) builds a real `rustls::ServerConfig`, binds a
  second `TcpListener` for HTTPS, and runs a manual `TlsAcceptor` accept loop alongside HTTP.
- [ ] CORS configuration — NOT DONE as a *policy* (see note above): the mechanism exists, wide
  open, with no config knob — "configuration" implies a choice, and there isn't one.

---

### 4.5 Documentation
**Action Items**:
- [x] API reference documentation — DONE. `docs/API.md` (508 lines) is a genuine
  endpoint-by-endpoint reference with request/response schemas.
- [ ] Architecture documentation — PARTIAL. `docs/M3_6_MULTI_GPU_ARCHITECTURE.md` and a README
  subsection cover only the multi-GPU slice; no overall system/request-flow architecture doc
  exists.
- [ ] Deployment guide — PARTIAL. README has a substantial TLS/SSL configuration section, but no
  general deployment guidance (process manager, scaling, container deployment).
- [ ] Performance tuning guide — NOT DONE. No dedicated file; `docs/` keyword hits are all
  unrelated internal design docs (quantization research, kernel fusion analysis).
- [ ] Troubleshooting guide — PARTIAL. README's "Troubleshooting" subsection is scoped only to
  TLS, not general operational troubleshooting.
- [ ] Migration guide (v0.x → v1.0) — NOT DONE. `docs/fuel-migration-*.md` are about the internal
  candle→Fuel backend port, unrelated to a user-facing version-upgrade guide.

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
