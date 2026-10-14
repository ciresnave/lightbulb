# Changelog

## 0.8.7

### Docs
- Phase 4 hardening item 1 (`PipelineConfig.timeout_ms`, PM-tasked): marked every
  `PipelineConfig` field as inert with a doc comment, rather than building an async execution
  driver nobody asked for. `Pipeline`/`PipelineBuilder` have zero callers anywhere and no
  `async fn execute` — there is no running path for a timeout (or any other field) to govern.
  Added a ROADMAP.md entry for the missing execution driver as its own, separately-prioritized
  feature. No behaviour change.

## 0.8.6

### Docs
- Audited V1_ROADMAP.md's Phase 4 ("Production Hardening") against real code, same method as
  the earlier Phase 2 audit: ~3/27 items fully done, 6/27 partial, ~18/27 genuinely missing.
  Unlike Phase 2 (mostly untested-but-working), several Phase 4 gaps are real defects wearing
  the shape of a feature: the admin metrics endpoint is a hardcoded-zero stub with a `// TODO`
  behind it; a `timeout_ms` config field exists but is never read anywhere (no request-timeout
  enforcement exists at all); API keys can be created but never revoked or rotated; and
  `tracing-subscriber`'s `json` feature isn't even enabled, so "structured logging" can't be
  turned on without a dependency change first. No code change.

## 0.8.5

### Docs
- **Corrects a mistake in #146 (0.8.4).** That PR's slice plan called `Op::PagedAttn`'s GPU
  execution path "a genuinely open risk, not yet resolved," having read only the problem
  statement at the top of `tests/gpu_paged_vs_contiguous.rs`'s doc comment and not far enough
  into the same file to find it already contains a real, `nsys`-traced, independently
  cross-checked measurement: no fused CUDA kernel, a host round-trip every token, ~10x the
  per-token cost of contiguous decode. Confirmed on this machine (real CUDA GPU present) by
  reading the existing result rather than re-running it. No code change.

## 0.8.4

### Docs
- Sizing/slice plan for research-track item 1 ("wire prefix-matching into the Fuel serving
  path"), PM-tasked. Found the real dependency chain is deeper than the original note described:
  two unwired primitives (`BlockPrefixIndex`, `BatchedPagedDecoder`), both built against Fuel's
  `PagedSessionScheduler` rather than the `SessionScheduler` actually in use; `PagedDecodeModel`
  has zero implementors anywhere (Fuel or lightbulb) today, though `fuel-model-llama` already has
  the matching inherent method, making a lightbulb-owned newtype wrapper the fix rather than new
  cross-repo engineering; Qwen3 has no equivalent yet; and `Op::PagedAttn`'s actual GPU execution
  path is an open, previously-contested question per `tests/gpu_paged_vs_contiguous.rs`. 5 slices
  proposed. **Not started** — sizing only, build begins once the PM confirms priority. No code
  change. Scope note: the default candlelight path is unaffected by any of this; `fuel-engine` is
  opt-in and not at parity.

## 0.8.3

### Docs
- Corrected ROADMAP.md's research-track item 2 (the `strftime_now` live-date rendering hazard):
  it was already fixed the same day as the review that surfaced it (#107, 2026-09-30) and the
  roadmap entry was never updated to say so. No code change.

## 0.8.2

### Docs / Tests
- `src/bin/lightbulb-cli.rs`'s network-facing functions (`non_stream_chat`, `stream_chat`) now
  have tests, run against a real local server (axum, already a dependency — no mocking library
  needed): success + usage parsing, error-message surfacing, and SSE delta concatenation down to
  the `[DONE]` sentinel. No code change to the binary itself. The three stdin-loop items (history,
  system prompt, env var loading) remain untested — harder to test without mocking stdin, left
  open in `V1_ROADMAP.md`.

## 0.8.1

### Docs
- Corrected ROADMAP.md's M4 slice plan: slice 2 (request admission) landed as a read-only dry-run
  check (`MemoryAwareScheduler::check_admission`), not the `submit_request`/
  `allocate_pending_requests` flow the plan originally described, so it does not make
  `AppState.scheduler`'s internal state live. Slice 3 (`scheduler_stats`/`scheduler_queue` wiring)
  and the priority field in slice 4 are deferred to slice 5 (continuous batching), which is what
  will actually give the scheduler concurrent state worth reporting. No code change.

## 0.8.0

### Changed
- **Breaking:** `/v1/chat/completions` and `/v1/completions` now reject a request with `503` if
  its own estimated cost (a `chars/4 + max_tokens` heuristic) alone exceeds the scheduler's
  configured memory budget, before dispatch. Board item M4 slice 2 — same shape and precedent as
  #103 (a new rejection path where there was none before). With the default 8 GB budget, no
  realistic request changes behavior; the breaking surface is the *possibility* of a new 503, not
  an observed regression in default configurations.

## 0.7.2

### Docs / CI
- CodeRipper portfolio-triage hygiene fixes (WSP-001, SUP-008, DOC-009), no behaviour change:
  - Added `.github/CODEOWNERS` (`* @ciresnave`).
  - Pinned `ci.yml`'s three `uses:` to full commit SHAs (`actions/checkout@...` v4.4.0,
    `Swatinem/rust-cache@...` v2.9.2, `actions/setup-python@...` v5.6.0), each with the version in
    a trailing comment. Added a top-level `permissions: contents: read` (the single `gates` job
    needs nothing more).
  - Added `GLOSSARY.md` (59 tracked doc pages clears the >=10-page DOC-009 threshold).
  - Verified before/after with a local build of `coderipper` 0.6.27 (`origin/main` 4375a87):
    before, WSP-001/DOC-009/SUP-008 = 1/1/5 findings, matching the triage doc exactly; after, all
    three clean.

## 0.7.1

### Docs
- Provenance audit of the repo's third-party-derived files (PM-tasked, per CireSnave's
  no-plagiarism rule, `PROVENANCE.md`): one real fork (`src/cache/parallel_cache_builder.rs`, from
  `huggingface/candle`'s `kv_cache.rs`) and two already-resolved items (the ABI-interface ruling in
  `src/backend/marlin_ffi.rs`, and the 2026-08-19 deletion of the unattributed `kernels/` CUDA
  sources). Comment-only fix: the fork's header asserted an unverifiable `Copyright (c) 2023
  Hugging Face` line (candle's own licence text names no holder); corrected to name the origin
  repository instead of an unverified legal holder. No code bytes changed.
  Follow-up: cloned `huggingface/candle` to resolve the two open gaps left by the initial audit
  (no commit pinned, no blob diff) — pinned the fork to `3b35cfca9` (date-bounded) and confirmed
  the derivation with a verbatim-matched comment plus struct/function-level comparison, not a
  naive whole-file diff (which would have measured 13x of unrelated growth instead of fork
  fidelity).

## 0.7.0

### Changed
- **Breaking:** `ModelRunner::start` gained a required `cache_stats: SharedCacheStats` parameter
  (both the candlelight and fuel-engine arms), same shape and reasoning as #134's
  `max_queue_depth` addition. Board item M4 slice 1: wires the `cache_stats` admin endpoint to
  the prefix cache's real, already-live stats instead of a hardcoded mock. `fuel-engine` accepts
  but never writes it — that path has no prefix-cache concept today.

## 0.6.2

### Fixed
- `/v1/chat/completions` streaming responses (both the real model-runner path and the no-model
  fallback) now send a literal `data: [DONE]` line as the final SSE frame, matching the OpenAI
  streaming contract. Found during the V1_ROADMAP Phase 2 test-coverage audit: the server never
  sent it at all (not a missing test, a missing behavior) — `lightbulb-cli` already parsed for it
  on the client side, so this closes a real client/server mismatch, not just adds coverage.

## 0.6.1

### Docs
- Audited V1_ROADMAP.md's Phase 2 manual-testing checklist against the real test suite: 5/24
  items genuinely covered (2 gated behind `#[ignore] // Requires PostgreSQL`), 19/24 not covered.
  Found the missing `data: [DONE]` SSE terminator is a real implementation gap, not a missing
  test (tracked as its own fix, next). Added a 5-slice plan for Phase 3 (MemoryAwareScheduler
  integration) to ROADMAP.md. No code changes.

## 0.6.0

### Changed
- **Breaking:** `/v1/chat/completions` now rejects `top_p` values other than `1.0` with a 400,
  instead of silently dropping the field (`ChatCompletionRequest` had no `top_p` field at all).
  Same pattern and precedent as #103's `n`/`stop`/completions'-`top_p` fix: a caller sending the
  spec default (`1.0`, or omitting the field) sees no change; a caller sending a meaningful
  `top_p` previously got a clean-but-wrong 200 and now gets an explicit 400 naming the value,
  since nucleus sampling is not wired into either decode path.

## 0.5.2

### Changed
- `kiss-ref-core`/`kiss-ops-vocab` dev-dependencies bumped 0.2.0 -> 0.4.0 (the actual crates.io
  latest; closes a roadmap-flagged drift). 0.4.0 is a breaking release upstream
  (`#[non_exhaustive]` on several enums, tightened NaN-quietness in `diff`, changed
  `reduce_var`/`reduce_std` form, bool comparison dtype) but none of those surfaces are exercised
  by `tests/kissref_differential.rs` -- verified by running the full differential and lib suites
  against the bumped version, not assumed from the changelog. Dev-dependency only, no lightbulb
  API change.

## 0.5.1

### Fixed
- A disconnecting client in Complete (non-streaming) mode previously ran its job to completion
  unnoticed: nothing checked whether the response channel was still alive until the final send
  silently failed. `run_jobs` now checks whether the response channel is closed before every
  generation step and stops as soon as the client is gone, instead of burning a full generation's
  worth of compute for a result nobody can read.

## 0.5.0 (2026-10-12)

### Changed
- **Breaking:** requests to the model runner queued without bound (`ModelRunner::start` used an
  unbounded `std::sync::mpsc::channel` for both the candlelight and fuel-engine backends), so a
  client that keeps submitting faster than the single-threaded runner can drain grows the queue
  without limit. `ModelRunner::start` now takes a `max_queue_depth` and creates a bounded
  `std::sync::mpsc::sync_channel` instead, returning a `SyncSender`. Callers enqueue with
  `try_send` and get `TrySendError::Full` back once the queue is at capacity; every HTTP entry
  point (`/v1/completions`, `/v1/chat/completions` non-streaming and streaming, the
  output-contract retry loop) maps that into a 503 via a new `QueueFullError`, distinct from a
  500 so a client knows this one is safe to retry. `ApiConfig` gains `model_max_queue_depth`
  (default 32); `main.rs` and the probe pass it through to `ModelRunner::start`.

## 0.4.2

### Changed
- The KV-cache disk tier (`cache::tiered_storage::FileDiskStore`) now writes atomically through
  `persistant` 0.4.1 (crates.io) instead of raw `std::fs` calls, and creates a sibling
  `<dir>.atomic-scratch` directory beside every `FileDiskStore` base directory — a new on-disk
  artifact, swept of any abandoned temp file at open. Reading files left behind by the old writer
  is unaffected: `FileDiskStore` still reads whatever plain file is at `base_dir/<key>`.
  - `base_dir` is resolved (symlinks and relative components, including a bare `.`) before the
    scratch sibling is derived from it, so the sibling stays outside the real root directory.
  - The sweep only runs once this `FileDiskStore` holds the scratch directory's lock file
    exclusively (a fresh `create_new`) — a scratch directory another live store still holds is
    left untouched rather than swept blind.
