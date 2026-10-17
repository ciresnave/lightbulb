# Changelog

## 0.10.11

### Added
- PersistAnt adoption Step 4 (docs/PERSISTANT_ADOPTION_DESIGN.md §5): the
  last call-site step. `rotate_api_key`'s one transaction -- the row-locked
  `SELECT ... FOR UPDATE`, the replacement key's `INSERT ... RETURNING`,
  and the old key's `UPDATE ... SET revoked_at`, all inside one
  `backend.begin(Isolation::ReadCommitted)` / `PgTx` transaction -- ported
  to `AppState::pg_backend`. Only `src/api/mod.rs`'s pool construction
  itself (step 5) still uses `deadpool_postgres`.
- Two helpers, `lock_api_key_for_rotation` and `insert_rotated_key`,
  extracted from `rotate_api_key` proactively (every prior step needed a
  Codacy method-length fix; this one didn't).

### Verified
- A real throwaway Postgres (Docker): all 13 `tests/api_integration_tests.rs`
  tests pass, including `test_rotating_an_api_key_revokes_the_old_and_issues_a_live_new_one`
  -- a genuine behavioral check (the rotated-out token is confirmed
  `401`, the new one confirmed live), not just a status-code check, so
  the transaction's atomicity is directly exercised. All 3
  `tests/persistant_migrate_tests.rs` tests pass. fmt clean; clippy
  baseline unchanged (594 diagnostics, 50 lints, none new or increased).
- The design doc's own step 4 description was already accurate (one
  function, the transaction) -- no correction needed this time, unlike
  steps 2 and 3.

## 0.10.10

### Added
- PersistAnt adoption Step 3 (docs/PERSISTANT_ADOPTION_DESIGN.md §5): the
  5 single-statement writes -- `create_api_key`'s insert, `revoke_api_key`'s
  update (both `admin.rs`), `pre_auth_attempt_limiter_middleware`'s
  `auth_attempt_counts` upsert, `rate_limit_middleware`'s `api_key_usage`
  upsert, and `audit_middleware`'s best-effort `audit_logs` insert (all
  three `auth_middleware.rs`) -- now go through `AppState::pg_backend`.
  `rotate_api_key`'s one transaction is untouched (step 4).

### Fixed
- `client_ip`/`api_key_id` insert parameters bind through `Value::Text`
  with an explicit `::inet`/`::uuid` cast on the PARAMETER -- `Value` has
  no INET variant either (the same gap step 2 found for UUID columns), and
  these are inserts into typed columns, not lookups against a stored
  value, so the cast belongs on the bind side here, not a column.

### Corrected
- `docs/PERSISTANT_ADOPTION_DESIGN.md`'s step 3 description (§5 item 3)
  claimed this step was 3 call sites ("`create_api_key`'s insert,
  `revoke_api_key`'s update, the rate-limit counter's `ON CONFLICT ... DO
  UPDATE ... RETURNING`"). That undercounted: there are TWO
  rate-limit-shaped upsert counters, not one (the pre-auth attempt limiter
  and the per-key rate limiter are structurally identical and both single-
  statement writes), and `audit_middleware`'s plain insert was omitted
  entirely. The true scope is 5 call sites. Fixed directly in the doc, not
  just noted here, per the portfolio's retraction rule -- the §1 Scope
  section's own file-level totals (4 in `auth_middleware.rs`, 6 in
  `admin.rs`) were already correct and needed no change.

## 0.10.9

### Added
- PersistAnt adoption Step 2 (docs/PERSISTANT_ADOPTION_DESIGN.md §5): the
  2 genuinely read-only call sites -- `admin.rs`'s `revoke_api_key`
  self-lockout check and `auth_middleware.rs`'s bearer-hash lookup -- now
  read through `AppState::pg_backend` (`persistant_sqlx::PgBackend`,
  built alongside the existing `db_pool` rather than replacing it). Every
  other call site is unchanged.
- `AppState::pg_backend: Option<persistant_sqlx::PgBackend>`, constructed
  in `ApiServer::new` from the same `database_url` as `db_pool`, `None`
  under the same condition.

### Fixed
- **Found while porting**: `persistant-sqlx`'s row decoder has no case for
  Postgres's `UUID` column type at all (only
  `BOOL`/`INT2`/`INT4`/`INT8`/`FLOAT4`/`FLOAT8`/`TEXT`/`VARCHAR`/`NAME`/
  `BPCHAR`/`BYTEA`) -- selecting a raw `uuid` column through `PgBackend::
  query` errors with `RelError::Unsupported`. Both ported queries now
  cast the `id` column to `::text` in the SQL itself; it decodes as
  `Value::Text` and is parsed back to a `Uuid` in Rust.

### Corrected
- `docs/PERSISTANT_ADOPTION_DESIGN.md`'s step 2 description (§5 item 2,
  and the related row in §2's table) claimed "the 3 single-row lookups
  (`admin.rs:420,567,709`) plus `auth_middleware.rs`'s hash lookup" were
  this step's scope. That was wrong: `admin.rs:420` is `create_api_key`'s
  `INSERT ... RETURNING` (a write, Step 3), and `admin.rs:678`/`:709` are
  both inside `rotate_api_key`'s one transaction (Step 4), not read-only
  call sites at all. The true Step 2 scope is exactly 2 call sites:
  `admin.rs:567` and `auth_middleware.rs:209`. Fixed directly in the doc
  (not just noted here), per the portfolio's retraction rule.

## 0.10.8

### Added
- PersistAnt adoption Step 1 (docs/PERSISTANT_ADOPTION_DESIGN.md §5):
  `src/api/persistant_migrate.rs` ports the 6 `migrations/V{n}__*.sql`
  files to `persistant_relational::Migration`, run via
  `persistant_sqlx::PgBackend` ALONGSIDE (not instead of) the existing
  `refinery` runner -- nothing is wired into the startup path yet, that
  is step 5. Not committed to `[dependencies]` until now:
  `persistant-sqlx = "0.7.4"`, `persistant-relational = "0.7.4"`.
- A Uuid<->`Value` helper (`uuid_to_value`/`value_to_uuid`), per the
  design doc §7 Q3's answer: stringify through `Value::Text` rather
  than change the schema, keeping `UUID` columns identical to today.
- `tests/persistant_migrate_tests.rs` (3 new `#[ignore] // Requires
  PostgreSQL` tests): proves `PgBackend::migrate` applies cleanly on
  top of a database `refinery` already migrated (the design doc §7
  Q1's non-destructiveness requirement), that the resulting schema
  matches the original migrations table-by-table and index-by-index,
  and that a second `migrate()` run is a pure no-op via PersistAnt's
  own ledger.

### Fixed
- **A literal, byte-identical port of the 6 `.sql` files is NOT
  non-destructive against a database refinery already migrated.**
  None of their `CREATE INDEX` statements have `IF NOT EXISTS`, and
  V6's `ALTER TABLE ... ADD COLUMN` has no existence guard; PersistAnt's
  own migration ledger (`_sqlx_migrations`) is a different table from
  refinery's, so it starts empty on such a database and genuinely tries
  to run every statement fresh on first contact. Every `CREATE INDEX`
  in the ported migrations now has `IF NOT EXISTS`; V6's `ADD COLUMN`
  is wrapped in a guarded `DO` block. The migration SQL's literal text
  changed for idempotency; the resulting schema did not.

## 0.10.7

### Added
- `docs/PERSISTANT_ADOPTION_DESIGN.md`: design-first plan for swapping
  lightbulb's admin/auth Postgres path (src/api/auth_middleware.rs,
  src/api/admin.rs, 6 migration files) onto PersistAnt 0.7.4's
  persistant-sqlx::PgBackend, per the PM's instruction following the
  auth-framework/PersistAnt adoption-fit analysis. Covers the target
  API shape, the migration path, two staging options (recommends
  incremental-by-call-site over big-bang), a step order, and an
  explicit rollback plan including the one stage (the pool/ledger
  swap itself) a plain PR revert does not cover. Docs only -- no
  dependency added, no source code or migration touched.

### Corrected
- The 0.10.6 entry below stated the auth-framework/PersistAnt
  adoption-fit verdict as first written, which read both crates from
  stale local working-tree checkouts. `docs/AUTH_PERSISTANT_ADOPTION_FIT.md`
  itself was corrected in #167 (commit 6a55165) against current
  `origin/main` for both repos, but this CHANGELOG entry was not
  updated at the time -- a retraction that didn't reach as far as the
  original claim. Corrected verdict, current as of PersistAnt 0.7.4 /
  auth-framework 0.6.0-rc71: PersistAnt is **adoptable** (not blocked --
  its `persistant-sqlx::PgBackend` now has a certified relational
  contract covering every primitive lightbulb's persistence uses).
  auth-framework's hash-at-rest gap is **closed**; its no-per-key-role
  and no-atomic-rotation gaps are **still open**.

## 0.10.6

### Added
- `docs/AUTH_PERSISTANT_ADOPTION_FIT.md`: adoption-fit analysis for
  auth-framework and PersistAnt against lightbulb's current API-key
  auth and Postgres persistence (board 196, CIRESNAVE-EXPECTATIONS
  §6.14). Docs only, no code or dependency change.

## 0.10.5

### Updated
- system-analysis 0.2 -> 0.3.0 (pulls hardware-query 0.3.0 transitively,
  closing the gap where lightbulb could not take hardware-query 0.3.0
  directly: system-analysis 0.2.1's published `hardware-query = "^0.2.0"`
  did not admit it). build-linux (Docker) and gates (native Windows, all
  6 steps: fmt, lockfile, doc, test, featgate, clippy) both green; clippy
  baseline unchanged (593 diagnostics, 51 lints, none new or increased).

## 0.10.4

### Added
- A synthetic Llama-shaped checkpoint generator for the checkpoint/GGUF
  `#[ignore]` fixture plan (`docs/CHECKPOINT_FIXTURE_FEASIBILITY.md`):
  `tests/support/mod.rs` (reached from integration tests via
  `#[path = "support/mod.rs"] mod support;`, and from `#[cfg(test)]` unit
  tests under `src/` via `include!` from the new
  `src::test_models::synthetic_checkpoint` module -- one file, two reach
  paths, so there is exactly one copy to keep correct). Writes a tiny
  (few-KB) `config.json` + `tokenizer.json` + `model.safetensors` with
  random f32 weights, in the same HuggingFace directory layout the real
  loaders already read. Verified end to end by a new unconditional (no
  `#[ignore]`, no env var, no real checkpoint) unit test,
  `test_models::tests::synthetic_checkpoint_loads`, which writes the
  fixture and loads it through the real `load_llama_f32_from_dir`.
- Adds `safetensors` as a dev-dependency (already present transitively;
  this makes it a direct one for the generator to use).

### Known gap, corrected from the feasibility note
- The feasibility note characterized `model_fuel_golden.rs`'s 7 tests as
  "zero additional design work beyond having a fixture at all" -- true for
  loading a checkpoint, not true for this file's specific harness. Its
  golden fixture lives at one hardcoded path
  (`tests/fixtures/model_fuel_golden/v1/tinyllama_f32_greedy.json`);
  pointing `TINYLLAMA_DIR` at the synthetic checkpoint and running
  `capture_golden_fixture` would overwrite that real, human-reviewed
  artifact rather than create a separate one. Un-ignoring those 7 needs a
  parallel fixture path (and capture/compare functions) for the synthetic
  checkpoint, not a one-line env-var swap -- a real, if modest, follow-up,
  not a zero-effort one. Likewise `decoder.rs`'s
  `stepwise_logits_match_the_reference_loop_exactly` hardcodes TinyLlama
  token ids (e.g. `7483`) that exceed the synthetic checkpoint's smaller
  vocabulary -- each test needs its own check against the fixture's actual
  dimensions, not a blanket "any valid checkpoint works" assumption.
  Correcting this now, before it's acted on as if it were true.

## 0.10.3

### Added
- Feasibility note for a synthetic checkpoint/GGUF test fixture (HANDOFF
  item 2): classifies all 62 real checkpoint/GGUF-gated `#[ignore]` tests
  as needing real trained weights (15) vs. architecture/shape/contract only
  (43, synthetic-fixture candidates) vs. special-cased (4). Docs only, no
  implementation (`docs/CHECKPOINT_FIXTURE_FEASIBILITY.md`).

## 0.10.2

### Changed
- Takes `hardware-query` 0.2.2, which fixes a Linux compile bug in that
  crate (missing `use std::fs`, 3 undefined driver-version functions under
  `#[cfg(target_os = "linux")]`); `system-analysis` was already at its own
  latest (0.2.1).
- Adds an isolated `build-linux` CI job (`ubuntu-latest`, `cargo build
  --workspace`) now that the blocker is gone, replacing the long-standing
  `gates`-job comment explaining why Linux wasn't covered.

## 0.10.1

### Added
- Phase 4 config-management: 9 new environment variable overrides for values that were
  previously hardcoded with no override at all --
  `LIGHTBULB_RATE_LIMIT_PER_MINUTE`/`_MODEL_MAX_BATCH_SIZE`/`_MODEL_CONTEXT_LENGTH`/
  `_MODEL_MAX_QUEUE_DEPTH`/`_MAX_MEMORY_BYTES`/`_MEMORY_PER_SLOT_BASE`/`_MEMORY_PER_TOKEN`/
  `_EVICTION_PRESSURE_THRESHOLD`/`_MEMORY_SAFETY_MARGIN` (`main.rs`). Every one keeps the
  previous hardcoded value as its default, so an existing deployment that sets none of them
  is unaffected.
- Two new startup validations, closing the rest of this item: `validate_server_config`
  (`src/api/mod.rs`) refuses a `0` batch size/context length/queue depth/rate limit;
  `MemoryAwareConfig::validate` (`src/engine/memory_aware_scheduler.rs`) refuses a `0` memory
  value or an eviction-pressure/safety-margin fraction outside its valid range. Both run at
  startup, before the server (or scheduler) does anything with the value.

## 0.10.0

### Changed
- **Breaking:** `GET /v1/lightbulb/admin/scheduler/queue` and `/admin/scheduler/stats` no
  longer return `0`/`0.0`/`[]` for fields with no real backing -- a hardcoded-zero stub found
  while adding `/metrics` (0.9.4/0.9.5), which looked like a real (idle) reading and was not.
  Every such field is now `null`, and both responses gained a new `unavailable_reason` string
  explaining why, pointing to `GET /metrics` for request count/latency/throughput.
  `scheduler/stats.memory_usage_bytes` is the one field that stayed a real number (same
  `live_system` source `/admin/metrics` already reads); `scheduler/queue`'s
  `pending_requests`/`running_requests`/`queue` have no real backing at all and are `null`
  unconditionally -- the scheduler's priority queue is never populated by the live request
  path. A caller parsing these fields as plain numbers (rather than nullable) will break.

## 0.9.5

### Added
- Phase 4 observability: request latency, throughput and error-rate-by-endpoint, completing
  the `/metrics` item started in 0.9.4. `request_metrics_middleware` (`src/api/mod.rs`) is the
  outermost router layer -- times and counts EVERY request, `/health`/`/metrics` included.
  One histogram (`lightbulb_http_request_duration_seconds`, the exporter's default quantile
  summary, no hand-picked buckets) and one counter (`lightbulb_http_requests_total`, labeled
  `route`+`method`+`status`); throughput and error rate are both PromQL computations over that
  one counter, not separate metrics. `route` comes from `axum::extract::MatchedPath` -- the
  registered route PATTERN, never the resolved URI, which would be unbounded cardinality and
  would leak which specific ids/keys were requested to an unauthenticated `/metrics` caller.
  Added a periodic `run_upkeep()` background task (`ApiServer::new`): the first metric needing
  it, since `install_recorder`/`build_recorder` don't spawn one automatically and 0.9.4's plain
  gauges never recorded histogram/summary data that would need draining.

## 0.9.4

### Added
- Phase 4 observability, v1 slice: `GET /metrics` in Prometheus text format
  (`ApiConfig.enable_metrics_endpoint`, default `true`), unauthenticated same as
  `/health`/`/health/ready`. Two gauges, no labels: `lightbulb_memory_total_bytes`/
  `_used_bytes` (the same live `sysinfo` sampler the admin `metrics` endpoint already reads)
  and `lightbulb_queue_depth` (new `AppState.queue_depth`, an RAII counter via
  `engine::model_runner::QueueDepthGuard` spanning enqueue-to-response-delivered, not a raw
  channel length -- `std::sync::mpsc::sync_channel` exposes none). New dependencies `metrics`
  and `metrics-exporter-prometheus` (latest crates.io versions; the latter with
  `default-features = false` to keep its push-gateway/second-listener features, and the
  `aws-lc-rs` they'd pull in, out of the tree). Request latency/throughput/error-rate, queue
  depth by priority, and slot utilization are NOT in this slice -- see `V1_ROADMAP.md` for why
  each is deferred.

## 0.9.3

### Fixed
- Phase 4 audit's "Request cancellation support" was marked NOT DONE on a wrong premise ("the
  streaming handler has no logic reacting to a dropped client connection"). It does: both
  response modes already stop generation on client disconnect via channel-drop detection --
  Complete mode proactively (`resp_tx.is_closed()` before every step, #133), streaming
  reactively (`stream_tx.send(...).is_err()` after one token, the same mechanism #155's timeout
  path relies on). No behavioral change; added the missing regression-pin test for the
  streaming path (Complete mode already had one) and corrected the roadmap. No
  `CancellationToken` added -- nothing today needs that abstraction over a mechanism that
  already works.

## 0.9.2

### Added
- Phase 4 hardening (audit item): `GET /health/ready` for deep readiness, next to the existing
  bare-liveness `/health` (unchanged). Checks the database (a real `SELECT 1`, when one is
  configured) and the model runner (`inference_tx` vs. `config.models_dir`, so "never asked
  for a model" reads `not_configured` -- ready -- and "asked for one and it failed to start"
  reads `not_loaded` -- not ready) fresh on every call. `200` + JSON detail when ready, `503`
  + the same shape otherwise. Both routes stay outside the auth stack, same as before, so a
  load balancer with no bearer token can read either.

## 0.9.1

### Fixed
- `postgres_integration` CI job widened to run all 12 Postgres-gated tests in
  `tests/api_integration_tests.rs` (was 4, by name filter) -- CireSnave's "#[ignore] should be
  a last resort." Running the other 8 for the first time surfaced a real cross-test
  contamination risk: every test shared one hardcoded fake peer IP, and the pre-auth attempt
  limiter's attempt counter (`auth_attempt_counts`) persists in Postgres across a whole CI run,
  keyed by IP alone -- a test that deliberately saturates its bucket could leak into an
  unrelated test's result depending on run order, which `cargo test` does not guarantee.
  Fixed by giving each test its own fake IP and truncating that table at the top of
  `create_test_server`.
- `test_pre_auth_attempt_limiter_is_keyed_per_ip`'s "different peer" varied only the port, not
  the IP (127.0.0.1:9999 vs the shared 127.0.0.1:54321) -- since the limiter keys by IP alone,
  this never actually exercised per-IP keying. Mutation-checked against a live Postgres in CI:
  reverting to that shape makes the test fail (429 instead of 401), confirming the fixed
  version (a genuinely different IP) is what the test needs to mean what it claims.

## 0.9.0

### Changed
- **Breaking:** Phase 4 hardening -- "Timeout handling for long-running requests" from the
  Phase 4 audit (#148): `tokio::time::timeout` had zero uses anywhere in `src/`, and
  `resp_rx.await` in `openai::chat::run_inference_once` (the chokepoint every non-streaming
  completion and the n>1 retry loop shares) had no upper bound at all -- a stuck or
  pathologically slow generation held the HTTP connection open indefinitely.
  `/v1/chat/completions` (non-streaming) and `/v1/completions` now return `504` after
  `ApiConfig.request_timeout_ms` (default 300000ms / 5 minutes, env
  `LIGHTBULB_REQUEST_TIMEOUT_MS`) if the model runner hasn't answered. New response code where
  there was none before -- same shape as #103/#142's precedent for a new rejection path.
  No explicit cancellation mechanism exists (`CancellationToken` has zero uses in `src/`), but
  dropping the response channel on timeout closes it, and `model_runner::run_jobs`'s normal
  complete-mode loop checks whether that channel is closed before every generation step -- a
  timeout does stop generation there. A different runner implementation that doesn't make that
  check would keep running past the timeout regardless.
  - Sourcery review caught three further gaps before merge, all fixed in the same PR: the
    streaming path (`stream: true`) had no timeout at all -- a stalled runner left an SSE
    client waiting forever, the exact hang this item otherwise fixes. Added a per-chunk idle
    timeout on the streaming path: if no token arrives within `request_timeout_ms` of the
    previous one, the stream emits one `timeout_error` SSE frame and ends (followed by the
    usual `[DONE]`).
  - The contract-retry and fallback-contract paths started a fresh `request_timeout_ms` on
    every attempt, so a request retried 3 times could hold the HTTP connection open up to 3x
    the configured limit before returning. `execute_contract` now computes one deadline for
    the whole call and passes each attempt only the time remaining, so the total is bounded by
    `request_timeout_ms` regardless of attempt count.
  - `LIGHTBULB_REQUEST_TIMEOUT_MS=0` was accepted and meant every request timed out
    immediately (`tokio::time::timeout(0)` always expires). Rejected at startup now, with a
    named error.

## 0.8.11

### Added
- Phase 4 hardening item 4 (PM-tasked): `tracing-subscriber`'s `json` feature wasn't enabled,
  so "structured logging" couldn't be turned on without a dependency change first (found by
  the Phase 4 audit, #148). Investigated first: enabling just the Cargo feature alone has no
  observable effect by itself -- nothing called `.json()`. Enabled the feature and wired a
  runtime toggle: `LIGHTBULB_LOG_FORMAT=json` switches `main.rs`'s tracing subscriber to JSON
  output; any other value (including unset, the default) keeps today's plain-text format, so
  no existing deployment's logs change under it. Verified both formats by actually running the
  binary (not just compiling): plain-text unchanged, `LIGHTBULB_LOG_FORMAT=json` produces real
  structured JSON lines.

## 0.8.10

### Added
- Phase 4 hardening item 3 (PM-tasked): API keys could be created but never revoked or
  rotated -- the only way to disable one was deleting the row directly in the database,
  outside any API path. Added `DELETE /v1/lightbulb/admin/api-keys/{id}` (revoke) and
  `POST /v1/lightbulb/admin/api-keys/{id}/rotate` (issue a replacement with the same
  role/description, revoke the original). Purely additive: no existing key's behavior
  changes, so this is a feature addition, not a breaking bump.
  - `revoke_api_key` refuses to revoke the caller's own key unless another unexpired,
    unrevoked admin key exists -- otherwise it is the request that locks every admin out
    of every admin route, including the one that could undo it.
  - `rotate_api_key` inserts the replacement and revokes the original in one database
    transaction: a crash between the two halves can never leave both keys live or
    neither live.
  - Migration `V6__add_api_key_revocation.sql` adds `revoked_at`; the auth lookup in
    `auth_middleware.rs` now also requires `revoked_at IS NULL`. Stays one indexed read
    on the hot path (keyed on the already-unique, already-indexed `key_hash`); the new
    `revoked_at` index serves the cold paths (an admin listing, the lockout check) instead.
  - Extends the `postgres_integration` CI job (#153): the revoked-key-is-rejected behavior
    needed a test that actually runs, not an `#[ignore]`-only assertion. Runs three new
    tests by name filter, alongside #153's pre-auth-limiter regression test -- the
    remaining pre-existing `#[ignore]`d integration tests stay unexecuted in CI, unrelated
    to this item.

## 0.8.9

### Fixed
- **Critical, pre-existing bug:** every authenticated request returned 503 when a database
  was configured. `pre_auth_attempt_limiter_middleware` (security audit item 3, 2026-10-06)
  bound a `String` (the client IP's `.to_string()`) to a `$1::inet`-cast parameter in its
  rate-limiting query. `String`'s `ToSql` only covers TEXT-family OIDs, so encoding failed on
  *every* call, unconditionally -- not past-the-threshold, every single one -- and this
  middleware runs in front of every route whenever `database_url` is configured. Fixed by
  binding the already-in-scope `client_ip: std::net::IpAddr` directly; `IpAddr` has a native
  `ToSql` for `INET`.
  - Found via a new `postgres_integration` CI job, this repo's first CI job that actually
    runs any of this file's Postgres-gated tests against a real database -- none of the 8
    pre-existing `#[ignore]`d integration tests gated on Postgres have ever executed in CI
    before this. New regression test
    `test_pre_auth_attempt_limiter_does_not_503_an_ordinary_request` reproduces it with a
    single, ordinary, nowhere-near-the-rate-limit request.
  - Whether any real lightbulb deployment has ever actually run with `database_url`
    configured is not something this lane has visibility into -- noted here rather than
    assumed either way, since that determines whether this was a live-production outage or
    an always-broken, never-exercised configuration path.

## 0.8.8

### Fixed
- Phase 4 hardening item 2 (PM-tasked): the admin `metrics` endpoint
  (`/v1/lightbulb/admin/metrics`) reported a hardcoded `cpu_usage_percent: 0.0,
  memory_total_bytes: 0, memory_used_bytes: 0` behind a `// TODO` comment, never wired to
  anything. Added `sysinfo` as a direct dependency (`system` feature only) and a persistent
  `AppState.live_system` sampler (CPU usage is a diff between refreshes, so a fresh `System`
  per request would always read 0); `cpu_usage_percent`/`memory_total_bytes`/
  `memory_used_bytes` are now real. `gpu_metrics` stays an honest empty `Vec`: neither
  `sysinfo` nor `system-analysis` (already a dep) exposes live GPU utilization or
  temperature, only static capability (name, VRAM size, compute capability) -- that needs
  vendor-specific NVML bindings, tracked as a new ROADMAP item, not built here.

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
