# Changelog

## 0.19.42

### Fixed
- CodeRipper OPS-011 `cli-basics`: `lightbulb`, `lightbulb-cli` and `lightbulb-probe` answer `--version`/`-V`/`--help`/`-h` with exit 0 and output before any logging, configuration or database access; an unknown flag exits 2. The server now parses flags with clap (it scanned `env::args()` for `--no-auth` before). Log output carries no ANSI escapes unless stdout is a terminal and `NO_COLOR` is unset. New `tests/cli_basics.rs` runs all three binaries.

### Changed
- **Breaking (CLI):** `lightbulb` rejects flags other than `--no-auth`, `--help` and `--version` (exit 2); it used to ignore them.

## 0.19.41

### Changed
- Docs only: superseded banners on AWQ_PHASE_1_SUMMARY, PROGRESS_SUMMARY, IMPLEMENTATION_NEXT_STEPS, AWQ_KERNEL_SETUP, GLOSSARY and PRUNING_IMPLEMENTATION (ROADMAP D4); `docs/PATENT-SWEEP.md` corrected: it named a `pruning` module that applies masks, but that code was removed 2026-10-22 (idea 14, rows 12 and 14; verdicts unchanged).

## 0.19.40

### Removed
- **Breaking (schema):** migration V8 drops the unused `user_sessions` table and its four indexes (ROADMAP A9; nothing in `src/` ever read or wrote it). Irreversible; V3 is unchanged so existing databases converge.

## 0.19.39

### Changed
- `docs/PATENT-SWEEP.md`: read in full the independents of CN120144300A, CN113383346A, US20220318633A1 and the previously partly read US8838659B2, US11132610B2, US20250061316A1, US20240370699A1, US12362764B2 (rows 8-15); all clear on Google text. Still INCOMPLETE: search coverage for ideas 1-2, 6, 11, 12.

## 0.19.38

### Changed
- `docs/PATENT-SWEEP.md`: ideas 3, 6, 9-14 searched and recorded; more patents read (US11544597B2, US9098492B2, US20250061316A1, US20240370699A1, US12362764B2, Amazon family partly); source of each read and unreproducible queries stated. Still INCOMPLETE.

## 0.19.37

### Added
- `.coderipper.toml`: SEC-002 waivers (reason, owner, removal condition) for 9 files whose commits hold placeholder tokens or SQL column names that gitleaks reads as keys; each hit read at its commit, none is a credential.

## 0.19.36

### Fixed
- `src/api/mod.rs` test helper: matching `tokio_postgres::config::Host` is refutable on cfg(unix) (E0005 under CodeRipper on Linux); now a `match` with a panic arm.

## 0.19.35

### Changed
- Every tracked `.rs` file (136) now starts with `// SPDX-License-Identifier: MIT OR Apache-2.0` (prepend only, no existing line changed). Licence files unchanged.

## 0.19.34

### Changed
- `ROADMAP.md` research track item 3: Cache-to-Cache (arXiv:2510.03215) added as prior art to read when the track opens.

## 0.19.33

### Changed
- `docs/PATENT-SWEEP.md`: searches for ideas 4, 5, 7, 8 recorded, three more patents read (US12198030B1, US20250322047A1, CN120952055B), cross-reference to the Fuel lane for ideas 1-2, search-surface weakness stated. Still incomplete.

## 0.19.32

### Added
- `docs/PATENT-SWEEP.md`: in-progress patent sweep record (idea list, two Nvidia claim reads). Explicitly not a clearance; searches still to run.

## 0.19.31

### Changed
- Docs: superseded banner on 12 docs describing code removed in #195/#196 (`src/model/`, `kv_compression`, `multi_gpu` parallel modules, `tests/multi_gpu_validation.rs`). ROADMAP D4 partly done.

## 0.19.30

### Fixed
- README/QUICKSTART/DEMO: Fuel pin updated to 0.25.10, the removed `local-llama-gen` quickstart replaced with the server flow, the multi-GPU section replaced with what actually exists, QUICKSTART `--config`/`config.toml` replaced with the real env vars, DEMO `sessions` table name corrected to `user_sessions`. ROADMAP D3.

## 0.19.29

### Added
- `LIGHTBULB_CORS_ORIGINS` (comma-separated) restricts CORS to the listed origins; unset keeps allow-any. Responses now carry `x-request-id` (the caller's, else a minted UUID), exposed to browsers via CORS. ROADMAP A7 partly closed; audit-log retention remains open.

## 0.19.28

### Added
- `examples/prefix_sharing_bench.rs`: `LB_ON_FIRST=1` runs the prefix-sharing arm before the baseline arm, so a time-capped benchmark window still yields the arm under test. Output and the OFF/ON comparison are unchanged.

## 0.19.27

### Added
- ROADMAP slice 6: CPU re-measure with prefix sharing off vs on against local Fuel main `ab97c5c21` (sharer TTFT ~60 s -> 2.3 s, bit-identical, n=1 rep, two runs), the GPU windows that produced no numbers and why, and the Fuel lane's CUDA TTFT figure cited as theirs. Audit row B4 marked done (#234).

## 0.19.26

### Changed
- The request-body cap is now explicit: `MAX_REQUEST_BODY_BYTES` (2 MiB) applied with `DefaultBodyLimit::max` in `build_router`. This is the same figure axum 0.8 already applied to `Json`/`Bytes`/`String` extractors, so behaviour is unchanged; ROADMAP audit row A7 was wrong that there was "no limit". Tests pin both the cap and the axum default it matches.

## 0.19.25

### Fixed
- `OutputContractSpec::Json` now parses: `try_parse` returned `None` for every input, so a Json contract could never succeed. It accepts the whole text, a Markdown code fence body, or the first parseable `{..}`/`[..]` span, and yields `ContractOutput::Json { value }`. Bare scalars count only as the whole text. (ROADMAP A3)

## 0.19.24

### Removed
- `src/hub.rs` and the `hf-hub` dependency. The file was not declared in `lib.rs` (never compiled) and had no callers; the server takes local model paths only. Network fetch is a future `mlmf-source-hf` crate (PM ruling 2026-10-25, ROADMAP audit B13).

## 0.19.23

### Changed
- Serving now runs Fuel's `PagedSchedulePolicy::Priority` (was `AdmissionOrder`), so the `X-Lightbulb-Priority` header decides the ORDER sessions are served within each scheduler tick (prefill, then decode) and who gets pool blocks first under pressure; it was recorded and ignored before. It does not cap how many sessions advance per tick (`max_batch` only chunks the batched forward), so the effect under light load is small. Ties keep admission order, so equal-priority traffic is unchanged. New test `serving_policy_serves_a_later_higher_priority_session_first_within_a_tick` (with an `AdmissionOrder` contrast arm). PM ruling 2026-10-25 (ROADMAP audit B4).

## 0.19.22

### Fixed
- `GET /v1/models` now works (it was registered POST-only, so every OpenAI client got 405) and lists what is actually loaded: the runner's model (`default_model`) while one is loaded, an empty list otherwise. The hardcoded `lightbulb-default`/`lightbulb-7b`/`lightbulb-13b` roster is gone. `POST /v1/models` still answers the same way.

### Added
- `GET /v1/models/{model}`: the model object, or an OpenAI-shaped 404 (`model_not_found`). ROADMAP audit row A1.

## 0.19.21

### Fixed
- The twelve `/v1/lightbulb/{knowledge,reasoning,state,tools}/*` routes now answer `501 Not Implemented` with a typed error instead of a success body that did nothing (`knowledge/add` returned `success: true` and a random UUID and stored nothing). The ~300 lines of unused request/response types are gone. PM ruling 2026-10-25 (ROADMAP audit A2).
- `src/api/mod.rs` module doc no longer claims auth "via auth-framework" (not a dependency; auth is the `api_keys` table with SHA-256 hashes) or lists the reserved routes as working features (A10 doc line).

## 0.19.20

### Added
- ROADMAP: spec-to-roadmap audit section (rows A1-A12 serving/API, B1-B14 engine/Fuel, C1-C5 tests, D1-D5 docs/provenance, plus the existing ROADMAP lines that contradict the code). Docs only.

## 0.19.19

### Added
- `docs/THIRD_PARTY_NOTICES.md`: credits `src/backend/marlin_ffi.rs`'s origin (`guoqingbao/attention.rs`, MIT; CireSnave's 2026-08-20 interface ruling unchanged), records removed third-party material, and says plainly that a per-dependency licence list is not yet generated.

### Fixed
- README no longer says the licence files and third-party notices are missing (they exist / now exist). `PROVENANCE.md` notes `parallel_cache_builder.rs` is gone.

## 0.19.18

### Changed
- `ModelRunner::start`/`start_with_options` no longer ignore `dtype`: anything other than unset or `f32` (also `float32`/`fp32`) returns the typed `UnsupportedDtype` error (recover with `downcast_ref`) naming the request and why, instead of a log warning and silent f32. All in-repo callers already pass `"f32"`. BF16/F16 weights remain unwired (needs fuel#489, Fuel 0.25.13+, not yet on crates.io).

## 0.19.17

### Changed
- ROADMAP slice 6: Fuel-lane root cause of the TTFT (prefill re-plans per token; fix in Fuel, #227) and the GPU attempt 2026-10-25 (OFF arm only, no verdict).

## 0.19.16

### Fixed
- `BlockPrefixIndexFor::record` is now first-writer-wins. It used to re-point an already-indexed chain at the newest prompt's session, so when that session finished `forget_donor` dropped the chain and the still-live original donor was no longer found: with prefix sharing on, a live donor served exactly ONE sharer and every later request ran cold (measured 2026-10-24, CPU, `cache_stats.hits` 2 of 4). New test `a_finished_sharer_does_not_orphan_the_live_donor`.

### Added
- `examples/prefix_sharing_bench.rs`: prefix sharing OFF vs ON through the real `ModelRunner` (TTFT, decode tok/s, no-share control, greedy bit-identity OFF vs ON, host load at start/end). ROADMAP slice 6.

### Changed
- CI: re-pinned the advisory CodeRipper portfolio check to coderipper `ee9c395` (0.9.24).

## 0.19.15

### Changed
- Deps: `fuel`/`fuel-*` 0.24.0 -> 0.25.10 (all eight crates; latest on crates.io). Breaking in Fuel 0.25.0 (#477): `fuel_loaders::quantized::gguf_file::Content::read(&mut File)` is gone; the three call sites (`src/gguf/mod.rs`, `loader_gguf.rs`, `loader_gguf_qwen3.rs`) now use `Content::open(&Path)`. Fuel documents four malformed-input-only metadata differences vs `read` (duplicate keys first-wins, no trailing-NUL strip, non-0/1 Bool bytes accepted, non-UTF-8 keys rejected); Lightbulb uses Fuel's `Content` only to derive numeric config, strings come from its own parser.

### Fixed
- `tests/gguf_e2e_weight_load_census.rs`: `stories15M-q4_0`/`q8_0` now load (the `freq_base` gap is closed in Fuel), so `KNOWN_FAILURES` is empty and ROADMAP's GGUF corpus section says so.

## 0.19.13

### Added
- ROADMAP research-track item 1, slice 3: `ServingDriver` can now EXPLOIT a `BlockPrefixIndex` hit -- register the donor's blocks as a Fuel `PrefixId` and admit via `PagedSessionScheduler::add_session_sharing_prefix` -- instead of only detecting it. Off by default, so serving behavior is unchanged. Toggle: `ApiConfig::model_prefix_sharing` / `EngineOptions::prefix_sharing` (new `ModelRunner::start_with_options`; `start` is unchanged and uses the defaults), set from the `LIGHTBULB_PREFIX_SHARING` env var read only in `main.rs` (documented in README). Not to be enabled by default until a measured benchmark shows a win (ROADMAP row). A full-block-aligned repeat shares one block fewer (Fuel rejects a share with no suffix); registrations are keyed by exact `(donor, blocks)`; any Fuel refusal falls back to a cold admission.

## 0.19.12

### Fixed
- `src/quantization/mod.rs` module doctest imported the removed `candlelight` crate and the deleted Norm Tweaking types, so `cargo test --doc` failed to compile it. Example trimmed to the dequantize/requantize calls that still exist.

## 0.19.11

### Changed
- Deps: `fuel`/`fuel-*` 0.22.4 -> 0.24.0 (all eight crates; latest on crates.io per the standing dependencies rule). No source changes needed: `cargo check --all-targets --features fuel-engine` is clean.

## 0.19.10

### Changed
- CI: re-pinned the advisory CodeRipper portfolio check to coderipper `d14a22b` (installs 0.9.14; the old pin installed 0.9.6), per the coderipper lane's repin notice.

## 0.19.9

### Fixed
- **ROADMAP.md: corrected M2's cache-management and FlashAttention "✅ COMPLETE" claims,
  contradicted (not just unverified) by the Candle removal (#195).** StreamingLLM/H2O/
  voting/cache-span eviction all cite `ParallelCacheBuilder` as their integration point --
  removed 2026-10-22 (`src/engine/mod.rs:329`) -- and cite three demo files
  (`streaming_llm_demo.rs`, `h2o_integration_demo.rs`, `span_management_demo.rs`) that no
  longer exist. The underlying cache modules survive, but their would-be Fuel-era
  successor (`SpanBlockMap`/`EvictionPlan` in `src/model_fuel/policies.rs`) has zero callers
  outside its own file -- same "built, never wired into the live driver" pattern as
  `batched.rs` and pre-slice-3 `BlockPrefixIndex`. The "hash-and-reuse prefill KV ... TTFT
  reduction >15% ... measured" claim is specifically false: that mechanism
  (`BlockPrefixIndex`) has been detection-only to date; real wiring is a separate, in-flight
  ROADMAP slice, not yet merged as of this correction.
  FlashAttention (M3.4) is a feature-gated no-op stub (`pub fn available() -> bool { true
  }`) behind a `flash-attn` Cargo feature that no longer exists in `Cargo.toml`, and its two
  cited evidence files (`benchmark_flashattention.rs`, `flash_attention_tests.rs`) are both
  gone. Each claim marked `⚠️ CONTRADICTED` in place pointing to one evidence block, per the
  pattern in #218/#219.

## 0.19.8

### Fixed
- **ROADMAP.md: corrected 23 false "✅ INTEGRATED" claims for 10 crates that were dead
  workspace scaffolding, not removed features.** `dynctx`, `mocopr-server`, `mocopr-client`,
  `web-server-abstraction`, `infra-storage`, `infra-fingerprinting`, `infra-network`,
  `infra-consensus`, `auto-discovery`, `coalescent` are none of them dependencies of
  `lightbulb` today, and `git log -S` on `src/` finds zero commits ever referencing any of
  their module paths there. Traced to the repo's initial commit (`98acb6a`), which scaffolded
  a multi-crate workspace including these as local path members / unused version pins; the
  very next commit to touch `Cargo.toml` (`633122f`, the project's second commit) replaced
  the whole file with the flat single-package structure still in use today -- these crates
  existed for exactly one commit, before any of the "INTEGRATED" prose citing them was
  written. Full evidence in `ROADMAP.md`'s new "dead scaffold crates" section; each of the 23
  sites marked `⚠️ CONTRADICTED` in place rather than repeating the evidence.

## 0.19.7

### Fixed
- **ROADMAP.md: corrected 3 false "✅ INTEGRATED: `distributed-config` crate" claims.** The crate
  was a direct dependency from the initial commit until #213 (2026-10-23) dropped it as unused,
  but `git log --all -S"distributed_config" -- src/` finds zero commits ever referencing that
  identifier under `src/` -- it was never actually called from any code, at any point in the
  project's history, not just stale after removal. Flagged (not fixed) the same claim pattern for
  several other never-dependency crates (`mocopr-server`, `web-server-abstraction`, `infra-*`,
  `coalescent`, `dynctx`, `candle-cuda-vmm`) elsewhere in the document's far-future milestones, for
  a follow-up docs-only audit PR.

### Changed
- **Repin the advisory CodeRipper `portfolio` CI job** (`.github/workflows/ci.yml`) to
  `ciresnave/coderipper/.github/workflows/portfolio.yml@30e012b7d400b10d88c1e294bd485e49975afc1e`
  (PM notice, 2026-10-24: fixes the default-branch ratchet red and a concurrent-run install
  failure in that upstream workflow). Verified the SHA and the workflow file both exist at it
  via `gh api` before applying. CI-only; no code behavior changes.

## 0.19.6

### Added
- **Advisory, non-blocking CodeRipper portfolio CI check** (CireSnave, 2026-10-23: "Roll out
  advisory-only, non-blocking CodeRipper checks everywhere.") New `portfolio` job in
  `.github/workflows/ci.yml`, calling CodeRipper's own reusable `portfolio.yml` workflow (public
  repo, pinned by SHA like every other action in this file:
  `ciresnave/coderipper/.github/workflows/portfolio.yml@c665210ad0972cf3ddb7bc81bdd7800c3248a938`).
  Not a required branch-protection check and no rule is silenced -- the PM's own gate treats this
  job as informational; the PM separately tracks baseline findings as individual ROADMAP rows per
  failing rule, not as CI failures to fix in this PR.

## 0.19.5

### Removed
- **`BatchedPagedDecoder`/`src/model_fuel/batched.rs` (2,585 lines), ROADMAP "research track item
  1" slice 4 reconciliation (PM-tasked, priority over slice 3).** Zero callers anywhere outside
  its own file and its own tests -- never wired into `ServingDriverInner`, `ModelRunner::start`,
  or anything else live; `scheduler_driver_paged.rs`'s `FuelPagedSchedulerDriver` is the one
  actually driving production serving. Built to fill a gap its own doc comment named directly
  ("there is no batched paged decode in Fuel at the HEAD this was written against"), which Fuel
  has since closed itself: `PagedSessionScheduler::step_batched` (Fuel 0.22.4, already used by
  `FuelPagedSchedulerDriver::step`) is exactly that missing upstream primitive. Its own
  measurements independently argued against keeping it regardless: a real benchmark recorded in
  its module doc showed its hand-rolled batched paged decode (fresh graph every call, no
  plan-once `DecodeSession`) running 5.58x-11.11x slower per sequence-token than the contiguous
  baseline, non-amortizing with batch size. Full decision and evidence: `ROADMAP.md`'s slice 4
  entry; full removed content in git history (same precedent as `scheduler_driver.rs`'s own
  deletion in 0.19.0). Verified green before removal (`cargo test --lib model_fuel::batched`,
  12 passed/0 failed/2 ignored) and after (`cargo test --lib` 647/647).

## 0.19.4

### Added
- **`InferenceJob.priority` wired through to Fuel's `set_priority` (M4 slice 5.4c completion).**
  `SchedulerDriver` (`src/engine/scheduled_runner.rs`) gained a `set_priority(id, priority: i32)`
  method, default no-op (same pattern as `handle_control`) so every existing implementor needs no
  change unless it has a real priority concept. `run_scheduled_jobs`'s admission loop now reads
  `job.priority` (parsed from the `X-Lightbulb-Priority` header) and calls
  `driver.set_priority(id, priority as i32)` right after a successful `try_admit`, logging (never
  failing the admission over) an error. `FuelPagedSchedulerDriver` is the one real implementor,
  forwarding to `PagedSessionScheduler::set_priority`; `ServingDriver` forwards to whichever of its
  three concrete arms (`F32`/`QuantizedGguf`/`QuantizedGgufQwen3`) is active. Proven by a new test
  (`FakeDriver` records every `set_priority` call it receives) rather than by compiling alone.
  **No live ordering effect yet**: every `BatchedServeConfig::policy` still defaults to
  `PagedSchedulePolicy::AdmissionOrder` -- Fuel's own doc says `set_priority` is read only under
  `PagedSchedulePolicy::Priority` and otherwise ignored. The value is recorded correctly; switching
  the default policy is a separate, not-yet-made decision, out of scope here.

## 0.19.3

### Changed
- **`try_connect_once` now uses `PgBackend::connect_with` with an explicit `TlsPolicy`**, instead
  of hand-building a `sqlx::PgPoolOptions` pool. `connect_with`'s own default (`TlsPolicy::Require`)
  overrides whatever `sslmode` a `DATABASE_URL` carries, so swapping it in with defaults could have
  silently broken any currently-working non-`Require` connection -- a new `tls_policy_for` picks the
  policy from `DATABASE_URL` itself instead: loopback host (test/local Postgres, usually started
  without certs) -> `TlsPolicy::Disable`, regardless of any `sslmode` on the URL; non-loopback with
  an explicit `sslmode=...` -> honored as given (`allow` maps to the closest equivalent, `Prefer`);
  non-loopback with no `sslmode` at all -> `Require` (the one case this picks something stronger
  than sqlx's own implicit `Prefer` default -- a tightening, never a weakening). Checked against a
  local docker Postgres container over loopback (`Disable` path; the full Postgres-gated
  `persistant_migrate_tests`/`api_integration_tests` suites, 16/16 + 3/3, pass against it with a
  fresh, `initdb`-created container) and against a `sslmode`-free non-loopback URL by code
  inspection (not independently checked against a live non-loopback Postgres in this session).
  Direct `sqlx` dependency removed (unused after this change); new direct `url = "2.5"` dependency
  added to parse `sslmode` and the loopback host from `DATABASE_URL`.

### Fixed
- **Security: `tokio` 1.53.2 -> 1.54.1, fixing RUSTSEC-2026-0386** (`AsyncFd::new` and related safe
  constructors violate I/O safety, patched >=1.51.6,<1.52.0 or >=1.53.3; verified against the
  advisory text and crates.io directly). Found via this repo's own `cargo deny` gate (added in
  0.19.2) while finishing this PR -- `cargo update -p tokio` was the whole fix.

## 0.19.1

### Fixed
- **Security: `ctutils` 0.4.2 -> 0.4.5, fixing RUSTSEC-2026-0385** (non-constant-time mask
  generation / `u128` comparisons on `ctutils::Choice`, patched >= 0.4.4; verified against the
  advisory text and crates.io directly, not just the notice). Transitive dependency only -- no
  direct pin in `Cargo.toml`, so `cargo update -p ctutils` was the whole fix. `cargo test --lib`
  658/658, `cargo fmt --all --check` and `clippy_gate.py` clean.

## 0.19.2

### Fixed
- **Security: RUSTSEC-2026-0258 (h2 0.3.27, "unbounded empty DATA frames") removed by dropping
  `distributed-config`.** `cargo tree -i h2@0.3.27` traced the whole chain to one unused direct
  dependency: `distributed-config = "0.1"` (only version ever published) pulled `reqwest 0.11.27`
  (`hyper 0.14` -> `h2 0.3.27`) and `jsonschema 0.17.1` transitively, and a repo-wide grep found
  zero actual uses of `distributed_config` in `src/` or anywhere else -- dropping the dependency
  removes the whole chain (confirmed: `h2`/`jsonschema`/`reqwest 0.11.x`/`hyper-tls 0.5` are all
  gone from `Cargo.lock`; only the modern `h2 0.4.20` remains).

### Added
- **`cargo deny` CI gate** (portfolio rule: CI and branch protection everywhere,
  CIRESNAVE-EXPECTATIONS.md §6.5), mirroring coderipper's/UniSQL's `deny` job and pinned action
  version. New `deny.toml`: real vulnerabilities fail the job; `unmaintained = "none"` and
  `yanked = "warn"` mirror `cargo audit`'s own existing default treatment of this tree's 4
  current unmaintained crates and one yanked `chacha20` (all pre-existing, not independently
  actionable from this repo, already non-blocking under `cargo audit`) rather than introducing a
  stricter failure mode for them on day one. License allow-list adds `MPL-2.0` (file-level/weak
  copyleft) and `CC0-1.0` (public-domain-equivalent) beyond coderipper's/UniSQL's own list --
  lightbulb's real tree needs them (`cssparser`/`cssparser-macros`/`dtoa-short`/`option-ext`/
  `selectors` use MPL-2.0; `tiny-keccak` uses CC0-1.0), confirmed by running `cargo deny check
  licenses` against this repo, not copied blind.

## 0.19.0

### Changed
- **Dependencies: `fuel`/`fuel-*` 0.20.71 -> 0.22.4 (CireSnave's standing newest-deps rule;
  publish independently confirmed via crates.io API + sparse index). No code changes needed for
  the bump itself.
- **Paged-scheduler port, slice 2 (ROADMAP research-track item 1): all three serving arms swapped
  from `SessionScheduler` to `PagedSessionScheduler`.** New `src/model_fuel/
  scheduler_driver_paged.rs::FuelPagedSchedulerDriver` replaces the deleted `scheduler_driver.rs::
  FuelSchedulerDriver` for `ServingDriverInner::{F32,QuantizedGguf,QuantizedGgufQwen3}` alike --
  Fuel 0.22.4 implements `PagedDecodeModel` for all three concrete model types, not just the one
  slice 1 originally sized against. `set_priority` is exposed as a new non-trait method on the
  driver (not yet called from `try_admit` -- that wiring is separate, tracked in
  `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` entry #2).
  **Breaking for anything depending on admission backpressure shape:** `PagedSessionScheduler::
  add_session` is optimistic (no up-front capacity pre-check), so `try_admit` can no longer return
  `AdmitError::NoCapacity` -- a session that can't fit is admitted and fails in isolation during
  its first `step`, surfacing as `FinishOutcome::Failed` instead (proven by a real test, not just
  documented). `SchedulePolicy::Batched{max_batch}` became `step_batched(max_batch)`, called
  unconditionally every tick (`max_batch=1` degrades to serial, proven equivalent by test).
  Verified: `cargo test --lib` 658/658 (incl. two pre-existing `scheduled_llama_engine` tests now
  exercising the Paged path), 5 new tests in `scheduler_driver_paged.rs`, `cargo fmt --all --check`
  and `clippy_gate.py` clean.

## 0.18.5

### Changed
- **Dependencies: `persistant` 0.4.1 -> 0.9.24, `persistant-sqlx`/`persistant-relational` 0.7.4 ->
  0.9.24 (CireSnave's standing newest-deps rule).** No lightbulb code changes needed: `cargo check
  --tests` and `cargo test --lib` (657/657) pass unmodified, and the full Postgres-gated suite
  (`persistant_migrate_tests` + `api_integration_tests`, 19/19) passes against a real local
  Postgres container, so the three-crate API lightbulb actually touches (`PgBackend`,
  `RelationalBackend`/`RelationalTx`, `Value`/`Row`/`RelError`, `persistant::blocking::Store`)
  held stable across the jump.

## 0.18.4

### Added
- **Failures-only, strict per-IP brute-force limiter (issue #130).** A second,
  separate counter from `pre_auth_attempt_limiter_middleware`'s flood limiter
  (which counts every request, successful or not, and so must stay generous):
  `auth_middleware` now records a failure (new `auth_failure_counts` table,
  migration V7) only on a REAL authentication failure — missing/malformed
  header, or a hash matching no live key, never a 500 or a success — and
  escalates to `429` once `max_auth_failures_per_minute_per_ip` (default 10/min)
  is crossed within the window, instead of the ordinary `401`. The security
  alert hook (`security_event.rs`) now fires from THIS crossing, not the flood
  limiter's, per the PM's instruction on #127's review ("item 4's alert
  threshold must key off that, not off this flood limiter, so alerts are not
  noise") — the flood limiter still 429s on its own threshold but no longer
  dispatches a `SecurityEvent`. Fails open on a database error recording the
  failure (returns the ordinary 401): unlike the pre-auth flood limiter, this
  sits after a real authentication decision has already been made, so a
  counter-write failure here is an observability gap, not an unmetered-
  guessing hole. Three new Postgres-gated tests (`#[ignore]`, run by
  `postgres_integration`): threshold escalation, per-IP keying, and the core
  claim that successful requests never arm this counter — verified against a
  real local Postgres container, not just compiled.

## 0.18.3

### Changed
- **Paged-scheduler port slice 2 re-sized against Fuel's merged `#456` (PM-tasked, 2026-10-23).**
  `ROADMAP.md`: slice 1 (`LlamaPagedAdapter`) is moot — Fuel implements `PagedDecodeModel`
  directly on its own `Llama3Model`/`QuantizedLlama3Model`, confirmed by reading
  `fuel-model-llama`/`fuel-transformers` at Fuel's merge commit. That same reading narrows slice
  2's scope: only `ServingDriverInner::QuantizedGguf` can swap (the plain `fuel_model_llama::
  LlamaModel` the `F32` arm uses never gets the trait). Two blockers found and recorded:
  `PagedSessionScheduler` has no per-session incremental-token accessor (breaks streaming;
  Fuel-side fix targeted for 0.22.2, no scope-cut accepted), and its `new()` takes no
  `SchedulePolicy`, breaking the one-generic-`run_batched`-for-all-three-variants pattern (needs
  a parallel `run_batched_paged`). `set_priority` already exists on `PagedSessionScheduler`,
  closing the parked M4 5.4c priority-wiring gap with no further Fuel work once slice 2 lands.
  No lightbulb code changes — `step()` cannot compile without the missing accessor, and shipping
  the rest uncompiled/unverified was declined.

## 0.18.2

### Changed
- **Commit the AWQ independent-reference fixture; retract the AWQ loader-wiring slice plan
  (PM-tasked, 2026-10-23).** `scripts/awq_reference/` (a WSL2+autoawq 0.2.9 reference-generation
  harness) and its output (`tests/fixtures/awq_reference/v1/qwen3_0_6b_awq_greedy.json`, real GPU
  provenance) are committed per PM ruling (`docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` entry #4): the
  fixture validates any AWQ implementation's correctness, not only the one it was originally
  built against. **No Rust-side comparison test is added**: `src/loaders/awq.rs` and
  `src/model/awq_qwen3.rs`, the code `ROADMAP.md`'s "AWQ loader wiring: slice plan" sized a
  three-slice plan against, were themselves deleted in the candlelight removal (#195, 0.16.0,
  2026-10-22) — one day after that plan was written, before its own slice 2 part 2 reference
  comparison ever ran. `ROADMAP.md` retracted that section in place (not just caveated) and added
  a new "AWQ validation — blocked on a verified Fuel-side path" row naming the three
  not-yet-checked candidates elsewhere in the portfolio (`fuel_transformers::quant_w4a16`,
  `baracuda`'s Marlin, `mlmf-awq`) that might eventually consume this fixture. Two other stale
  AWQ claims found while fixing the above (an M3 status-table cell, a "CPU-unusable" bail-site
  list naming the same deleted files) corrected in place for the same reason.

## 0.18.1

### Changed
- **`cache_clear` real implementation (board PM task, 2026-10-22).** Was a literal `TODO` stub
  that always returned `cleared_entries: 0`/`freed_bytes: 0` and touched nothing. Now sends a new
  `ControlMsg::ClearPrefixCache` to the serving thread over a new administrative control channel
  (`ControlSender`/`ControlMsg`, `src/engine/model_runner.rs`) -- `ModelRunner::start` now returns
  `(InferenceRequestSender, ControlSender)` instead of just the inference sender, and
  `AppState.cache_control: Option<ControlSender>` carries it. `engine::scheduled_runner::
  run_scheduled_jobs`'s loop polls `ctrl_rx` every iteration (including the idle-wait branch, now
  `recv_timeout(50ms)` instead of a blocking `recv()`, so a control message reaches the loop even
  with zero inference jobs in flight) and calls the new `SchedulerDriver::handle_control` (default
  no-op, so `FakeDriver` and every other existing implementor needed no change).
  `ServingDriver::handle_control`/`clear_prefix_cache` (`src/model_fuel/scheduled_llama_engine.rs`)
  rebuilds `BlockPrefixIndexFor` from scratch and zeroes `SharedCacheStats` to match -- the HTTP
  handler never writes `cache_stats` directly, only the serving thread does, so there is exactly
  one writer and no race between an admin clear and live detection. The response reports a
  snapshot taken BEFORE sending the clear signal (the clear itself is necessarily async).
  `layer_indices`/`older_than_seconds` stay honestly unsupported -- the index has no per-layer or
  per-age structure to select against.

## 0.18.0

### Changed (breaking)
- **`cache_stats`/`CACHE_METRICS_LIVE` real again (board PM task, 2026-10-22, ROADMAP research
  track item 1 / `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` entry #1).** `ServingDriver`
  (`src/model_fuel/scheduled_llama_engine.rs`) now carries a
  `BlockPrefixIndexFor<SessionId>` — the chain-hash prefix index
  `src/model_fuel/policies.rs`'s `BlockPrefixIndex` already built and proved correct for Fuel's
  block-pool `SessionHandle`, generalized (`BlockPrefixIndexFor<D>`, zero other callers so this
  is safe) so a caller with no live `KvBlockPool` session can still use it — and records/looks
  up every admitted prompt's token prefix on the live Fuel path, writing real
  `hits`/`misses`/`insertions`/`entries`/`evictions` into `SharedCacheStats`.
  **Detection only, not exploitation**: a detected hit is NOT spliced into the donor's KV blocks
  and does NOT skip prefill yet — that needs `fuel_inference::multi_session::
  PagedSessionScheduler` (`add_session_sharing_prefix`), a different scheduler than the
  currently-wired `SessionScheduler`, out of scope for this slice. `CacheStatsResponse` never
  surfaces a "tokens saved" figure, so nothing here claims one. `total_size_bytes` stays an
  honest `0`: the index stores hash chains, not KV bytes. **Breaking**: `/v1/lightbulb/admin/
  cache/stats` now reports `status: "live"` with real values instead of
  `unavailable_on_backend` — a behavior change on an existing endpoint's response shape.
  `src/api/admin.rs`'s cache-stats regression-guard tests rewritten to assert real values
  (idle-zero + primed-real), mirroring the `scheduler_stats`/`scheduler_queue` pattern #204 set.

## 0.17.3

### Changed
- **Fuel 0.20.71 published; bumped the dependency pin (all `fuel`/`fuel-*` crates, and
  `baracuda-*` alpha.140 -> alpha.147 transitively).** Two flips, both from HANDOFF.md:
  1. **`tests/gguf_e2e_weight_load_census.rs`'s `KNOWN_FAILURES`**: Fuel's `derive_config` now
     defaults `llama.attention.head_count_kv` for pre-GQA exports, closing that gap. Re-measured
     against 0.20.71: `stories15M-q4_0.gguf`/`stories15M-q8_0.gguf` still fail, now for a
     DIFFERENT, previously-masked key (`llama.rope.freq_base`, also not defaulted) — updated
     `KNOWN_FAILURES` to the new cause rather than removing the entries; still 9/11 weighted
     corpus files load end-to-end, unchanged.
  2. **`scheduler_stats`/`scheduler_queue` are live again.** `ServingDriver`
     (`src/model_fuel/scheduled_llama_engine.rs`) now mirrors every real admission/departure on
     the Fuel serving path into the shared `MemoryAwareScheduler` (`mirror_admission`/
     `mirror_departure`), rebuilding what the deleted candlelight `run_jobs` path used to do
     (M4 slices 5.4a/5.4b), against the live Fuel path instead. Split the old combined
     `SCHEDULER_AND_CACHE_METRICS_LIVE` flag into `SCHEDULER_METRICS_LIVE` (now `true`) and
     `CACHE_METRICS_LIVE` (**still `false`**): the cache-stats gap does NOT close the same way --
     the currently-wired `fuel_inference::multi_session::SessionScheduler` has no prefix-cache
     concept at all (no prefix sharing; that's research track item 1's unwired territory), so
     there is nothing real to report yet. Flipping it anyway would have reintroduced exactly the
     "never-written value passed off as real" defect #201 existed to stop, just with a fabricated
     non-zero instead of a zero. Updated `src/api/admin.rs`'s regression-guard tests to assert
     real values for the scheduler fields; `cache_stats`'s stay unchanged (still asserting
     `unavailable_on_backend`).

## 0.17.2

### Fixed
- **ROADMAP.md's M4 slice 5.4c still described plumbing into `admit_pending`/`run_jobs`**, the
  candlelight-era path deleted entirely in #195 — corrected in place: that plumbing target is
  gone regardless of API shape.
- **Found while fixing the above (Sourcery, this PR): two conflicting PM rulings on 5.4c's
  priority API shape.** `docs/API-EXTENSIONS.md` documents Option B (the `X-Lightbulb-Priority`
  header, already implemented) per a 2026-10-21 ruling; this PR's own first draft recorded
  Option A (a top-level `priority` body field) per a same-day 2026-10-22 ruling made without
  reading `docs/API-EXTENSIONS.md` first. **RESOLVED 2026-10-22: Option B stands, Option A
  withdrawn.** `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` entry #2 and `ROADMAP.md`'s 5.4c note now
  say so. Neither option is wired to the live Fuel path, so nothing shipped was ever wrong.
  Docs only, no code changed.

## 0.17.1

### Fixed
- **ROADMAP.md still described candlelight as the default shipping path in three places** —
  the top "Note on Candle-first" banner, the "What is actually true" status section's
  Candle→Fuel port paragraph and file-count table (found during Sourcery review of this PR,
  missed by the first pass), and the prefix-matching research-track section's own scope
  correction — all retracted in place 2026-10-22 (not just caveated): candlelight was removed
  entirely in PR #195, Fuel is the sole, unconditional serving path, `fuel-engine` is a no-op.
  The prefix-matching section's blast-radius claim is corrected to distinguish which slices
  actually change live serving behavior (2 and 3) from prep/decision/blocked ones (1, 4, 5) —
  not every slice there touches the only production path equally. Also fixed: the "fuel-engine
  arm of `ModelRunner::start`" phrasing, which implied a feature-gated dispatch that no longer
  exists; and an overbroad "predate" disclaimer that swept in post-port evidence it shouldn't
  have. Docs only, no code changed.

## 0.17.0

### Changed
- **BREAKING: `cache_stats`/`scheduler_stats`/`scheduler_queue` report a typed
  `{"status":"live"|"unavailable_on_backend","value"?}` for every field that depends on the
  shared `SharedCacheStats`/`SharedScheduler` handles, instead of a raw number/`null`.** Fuel's
  `ModelRunner::start` never writes either handle (`docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` gaps
  #1/#2) — now that Fuel is the sole serving path (PR #195), those endpoints were reporting a
  permanently-stale construction-time zero as if it were a genuine idle reading, which the
  gaps doc itself flagged as worse than the state before candlelight's removal. Affected
  fields: `cache_stats`'s `total_entries`/`total_size_bytes`/`hit_rate`/`eviction_count`;
  `scheduler_queue`'s `pending_requests`/`running_requests`; `scheduler_stats`'s
  `active_slots`/`queued_requests`/`scheduler_memory_used_bytes`/`eviction_pressure`. All
  report `unavailable_on_backend` today. Gated on a single new constant,
  `model_runner::SCHEDULER_AND_CACHE_METRICS_LIVE` (currently `false`) — flip it, and wire the
  handles, once Fuel's own admission/cache-counter mechanisms publish (cache counters 0.20.64,
  scheduler stats 0.20.65) and this shape goes live without further endpoint changes.
  `memory_usage_bytes` (system memory, unrelated to either handle) is unaffected.

## 0.16.5

### Fixed
- **README.md still described candlelight as the default, shipping backend** — removed
  entirely in PR #195 (2026-10-22). Rewrote the "Backends" section: Fuel is now the sole
  engine, `fuel-engine` is a no-op flag, dependencies are pinned by crates.io version
  (`0.20.61`, PR #196) rather than git revision. Added a pointer to
  `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` for the two real capability gaps the removal opened.
  No code changed; version bumped per the standing convention.

## 0.16.4

### Fixed
- **Retracted two stale ROADMAP.md sections describing code deleted in PR #195 (candlelight
  removal) as though it still existed.** "M3.6 — Multi-GPU Inference" (two call sites: the
  top-of-file honesty-audit summary and the detailed M3.6 section) described `src/multi_gpu/
  {distributed_cache,pipeline_parallel,tensor_parallel}.rs`'s eight runtime-bail sites and
  `ParallelCacheBuilder` — every one of those files and that type were deleted entirely in the
  candlelight removal (zero external callers, confirmed before deletion). "M4 —
  MemoryAwareScheduler integration: slice plan" described `run_jobs`/`admit_pending`/
  `EngineModel`/`ParallelModelManager` and the real `SharedScheduler` wiring slices 5.4a/5.4b
  landed into `run_jobs` — all of it removed in the same PR, confirmed by that file's own
  pre-removal comment ("`run_jobs` itself survives only as the candlelight path's job loop").
  `cache_stats`/`scheduler_stats`/`scheduler_queue`/real priority admission are consequently
  back to reporting nothing real on the current (Fuel-only) path — the exact hole this whole
  M4 plan existed to fill — already tracked as a real, open capability gap in
  `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` entries #1/#2, not re-derived here. Added prominent
  `SUPERSEDED` banners at each site rather than deleting the detailed history, which records
  real work and real lessons (the `EngineModel::step_batch` batch-shape hazard in particular)
  that whoever rebuilds this against Fuel should read.

No code changed; `Cargo.toml`/`Cargo.lock` bumped per the versioning convention
(every pushed change bumps), verified with `cargo metadata --locked`.

## 0.16.3

### Added
- **Re-created the end-to-end GGUF weight-load census through Fuel's own loader**
  (`tests/gguf_e2e_weight_load_census.rs`), replacing `tests/gguf_serving_census.rs` (deleted
  in PR #195 along with its `ParallelModelManager::load_gguf` target). Calls `model_fuel::
  loader_gguf::load_quantized_llama_gguf`/`loader_gguf_qwen3::load_quantized_qwen3_gguf`
  directly — the real, live serving-path loaders — not just metadata acceptance.
  **Result: 9 of 11 weighted corpus files load end-to-end, including all three IQ-coded files**
  (`IQ3_XS`/`IQ4_XS`/`Q2_K`), confirming the IQ dtype-gap closure (0.16.2) holds at
  weight-decode time, not just metadata acceptance.
  **New gap found, routed to the Fuel lane, not fixed here**: `stories15M-q4_0.gguf`/
  `stories15M-q8_0.gguf` fail — Fuel's `derive_config` requires `llama.attention.head_count_kv`
  and does not default it. These are pre-GQA karpathy-era exports that never declared the key
  (kv heads == attention heads is implied by its absence, per llama.cpp's own convention); not
  malformed, not an IQ-dtype issue. Pinned as `KNOWN_FAILURES` in the new test so either
  direction (fixed upstream, or a new/different failure) is caught rather than silently passing.
  `ROADMAP.md`'s GGUF corpus section and `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`'s entry #11
  updated with the full writeup.

Verified clean: `cargo check --lib --tests --examples --benches --features fuel-engine`,
`cargo fmt --check`, `python scripts/clippy_gate.py`, `cargo metadata --locked`, `cargo doc
--no-deps --lib`, and `cargo test --test gguf_e2e_weight_load_census -- --ignored --nocapture`
against the real corpus.

## 0.16.2

### Fixed
- **Re-measured the IQ dtype gap on the real GGUF corpus — RESOLVED.** Was "3 of 13 weighted
  files refuse on an unrepresentable IQ/K-quant tensor dtype" (measured 2026-09-09, parked on
  the Fuel migration). After PR #195 (candlelight removed) and PR #196 (Fuel 0.20.61, GAP-211
  B1's widened `GgmlDType` table): **0 of 11 weighted corpus files refuse**, including the
  three carrying 180-210 tensors each of dtype 20/21/23 (IQ4_NL/IQ3_S/IQ4_XS). Updated
  `tests/gguf_dtype_census.rs`'s accepted-dtype table (`CANDLE_ACCEPTS` → `FUEL_ACCEPTS`,
  matching Fuel 0.20.61's real table) and rewrote its regression-guard test (`the_refused_
  number_is_a_dtype_code_and_not_a_tensor_index` → `the_iq_dtype_gap_stays_closed`) to assert
  zero refusals going forward rather than requiring at least one — the old test's own "both
  arms or nothing" guard correctly caught that its original scenario no longer has any subject
  matter in this corpus. `ROADMAP.md`'s entry updated with the dated re-measurement and its
  ref. Still open, not claimed here: end-to-end weight-loading verification through Fuel's own
  loader (`tests/gguf_serving_census.rs`, deleted in PR #195, not yet re-created) — this
  re-measurement is metadata/tensor-directory acceptance only.

Verified clean: `cargo check --lib --tests --examples --benches --features fuel-engine`,
`cargo fmt --check`, `python scripts/clippy_gate.py`, `cargo metadata --locked`, and
`cargo test --test gguf_dtype_census -- --ignored --nocapture` against the real corpus.

## 0.16.1

### Changed
- **Moved the `fuel` dependency family off the git source onto crates.io** (Fuel 0.20.61, all
  40 crates, published 2026-10-22; CireSnave's standing rule, CIRESNAVE-EXPECTATIONS §6.8: no
  cross-repo deps by relative path/git where crates.io has them). `fuel`/`fuel-cpu-backend`/
  `fuel-model-llama`/`fuel-transformers`/`fuel-loaders`/`fuel-ir`/`fuel-inference`/
  `fuel-cuda-backend` all pinned to `"0.20.61"` by version. The main crate publishes under the
  name `fuel` directly -- no `package = "fuel-core"` rename needed as it was for the git
  source; crates.io's own `fuel-core` (v0.48.x) is the unrelated Fuel blockchain client and was
  confirmed NOT accidentally pulled (checked `Cargo.lock`: no `fuel-core` entry anywhere).

### Fixed
- **Found and fixed a stale test while verifying the migration**:
  `tests/chat_template_render.rs`'s `a_gguf_with_an_unsupported_tensor_dtype_falls_through_
  and_says_why` pinned the OLD behavior -- a GGUF whose tensor directory names an unmodeled
  quantization used to make the whole file unreadable (metadata included). Fuel 0.20.61
  (GAP-211 B1) made `fuel_formats::gguf::Content::read` lenient: an unsupported tensor dtype is
  now recorded and skipped rather than aborting the read, so metadata (including
  `tokenizer.chat_template`) resolves normally regardless of what any individual tensor's dtype
  is. Verified this is unconditional (not dtype-specific) with a deliberately-invalid dtype code
  (`100`, never a valid GGML type). Renamed to
  `a_gguf_with_an_unsupported_tensor_dtype_still_resolves_its_metadata` and rewritten to assert
  the new (better) behavior. Also updated a stale comment in `src/gguf/mod.rs` asserting "the
  IQ codebook family is rejected by candle AND by fuel", which this exact migration made false.
  **Known follow-up, not done here**: GAP-211 also widened `GgmlDType::from_u32` to accept the
  entire IQ family (codes 16-23, 29), TQ (34, 35), and MXFP4/NVFP4 (39, 40) -- the "3 of 13
  weighted GGUF corpus files refuse on an unrepresentable dtype" figure in `HANDOFF.md`/
  `ROADMAP.md` is now stale and needs re-measuring against the real corpus
  (`tests/gguf_dtype_census.rs`), not assumed fixed.
- `cargo audit`: unchanged from pre-migration (`h2`/`chacha20`/etc. findings are pre-existing,
  confirmed by diffing `Cargo.lock` against `origin/main`'s, not introduced by this migration).

Verified clean: `cargo check --lib --tests --examples --benches --features fuel-engine`,
`cargo fmt --check`, `cargo test --tests` (CI's skip list), `cargo test --features fuel-engine
--tests --no-run`, `cargo doc --no-deps --lib`, `cargo metadata --locked`, `python
scripts/clippy_gate.py`.

## 0.16.0

### Breaking
- **Removed the `candlelight` dependency entirely** (CireSnave, 2026-10-22, verbatim: "I've
  got an easy solution for Lightbulb. Remove all dependencies on Candle and Candlelight and
  then start fixing the errors that leaves."). The default, non-opt-in serving path is now
  unconditionally Fuel-backed (`src/model_fuel/`) -- the `fuel-engine` feature flag that used
  to select it is now a no-op kept only so existing callers (including CI) don't break.
  Removed: `ParallelModelManager` and its 16 supporting `src/model/*.rs` files; `run_jobs`/
  `EngineModel` and ~10 helper functions in `src/engine/model_runner.rs` (confirmed dead by
  that file's own pre-removal comment: Fuel's path never called `run_jobs`); `AwqQwen3` +
  Marlin kernels (`src/backend/marlin.rs`) + the whole `src/loaders/` module (zero production
  callers -- `load_awq_llama` was never wired in); the `lora`, `pruning` (algorithm body --
  `name_mapping` survives) and `quantization::norm_tweaking` modules (zero callers); the
  Tensor-typed halves of `src/cache/{prefix_cache,tiered_storage}.rs` (the byte-level
  `DiskStore`/plain `PrefixCacheConfig`/`PrefixCacheStats` survive -- real Fuel dependencies);
  the entire `ParallelCacheBuilder`/`ParallelKvCache`/`kv_compression`/`tensor_codec` cluster
  (confirmed superseded, not a gap -- `tests/parallel_kv_cache.rs`'s own pre-removal doc
  comment already named Fuel's `kv_block_pool` as "a strict superset"); candlelight-typed
  multi-GPU tensor/pipeline parallelism (zero callers, never wired to the 8 known multi-GPU
  bail sites); `SpeculativeModel`/`SpeculativeDecoder` (zero callers; `SpeculativeConfig`/
  `SpeculativeStats` survive). `src/gguf/mod.rs`'s `Value`/`TensorInfo`/`Content` repointed to
  `fuel_loaders::quantized::gguf_file`'s API-compatible equivalents (confirmed field-for-field
  match against candle-core 0.10.2's shapes) rather than deleted, since this is genuinely live
  code (GGUF metadata introspection used throughout the test suite).
  Deleted (not ported -- real porting work, not a mechanical swap, tracked as follow-up in
  `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`): `tests/gguf_serving_census.rs`,
  `tests/enhanced_correctness_tests.rs`, `tests/flash_attention_tests.rs`,
  `tests/multi_gpu_validation.rs`, `tests/fused_rmsnorm_{perf,parity}.rs`, 30 of 34
  `examples/*.rs`, the sole `[[bench]]` target. Fixed two silu/sigmoid tests in
  `tests/kissref_differential.rs` that claimed to test "the real code path" -- true via
  candlelight, no longer true now that the real path is `fuel::lazy::Tensor::silu()`
  (`src/model_fuel/batched.rs:975`); kept the harness-self-test by swapping in plain f32 math.
  Found and fixed along the way: `ModelRunner::start`'s Fuel implementation was still
  `#[cfg(feature = "fuel-engine")]`-gated even though it is now the only implementation --
  `src/api/mod.rs` calls it unconditionally, so the default (non-`fuel-engine`) build no longer
  compiled until this was made unconditional too.
  **Real capability gaps opened** (candlelight provided these; Fuel's live path does not, and
  this is NOT a Lightbulb-side bug to fix -- see `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md` for
  the full writeup and routing): the `cache_stats`/`scheduler_stats`/`scheduler_queue` admin
  endpoints silently go dark (M4 slices 5.4a-c's real admission/departure mirroring and
  prefix-cache stats had no Fuel-side equivalent to move to); f16/bf16 dtype selection for
  SafeTensors models (pre-existing Fuel limitation, no longer avoidable via the candlelight
  fallback). AWQ quantized inference is recorded as likely-superseded-but-unverified: three
  other portfolio projects (fuel, baracuda, mlmf) have their own quantization/Marlin code,
  parity not yet checked from this side. The WSL2 AWQ reference-fixture methodology
  (`scripts/awq_reference/`) is kept regardless -- it validates any AWQ implementation,
  including whichever one (if any) eventually replaces this.
  `cargo check --lib --tests --examples --benches --features fuel-engine`, `cargo fmt --check`,
  `cargo test --tests` (CI's skip list), `cargo test --features fuel-engine --tests --no-run`,
  `cargo doc --no-deps --lib`, `cargo metadata --locked`, and `python scripts/clippy_gate.py`
  (baseline updated: 610 -> 289 diagnostics, a real drop from deleting ~40 files, not a
  measurement artifact) are all clean.

## 0.15.0

### Fixed
- **AWQ slice 2 (part 1): `AwqQwen3` used the wrong `head_dim` for every real Qwen3 dense
  checkpoint.** `Qwen3RotaryEmbedding::new` and `Qwen3AwqAttention::new` both computed
  `head_dim = hidden_size / num_attention_heads` directly, but every Qwen3 dense model
  (confirmed on `Qwen/Qwen3-0.6B`'s and the AWQ slice-2 target `Orion-zhen/Qwen3-0.6B-AWQ`'s
  `config.json`, both of which set `hidden_size: 1024`, `num_attention_heads: 16` -- formula:
  64 -- and `head_dim: 128` explicitly) sets `head_dim` independently of that ratio, the same
  way `fuel_transformers`' GGUF Qwen3 reader already treats it (read from
  `qwen3.attention.key_length`, never derived). `AwqQwen3::new` sized `q_proj`/`k_proj`/
  `v_proj`/`o_proj` for the wrong shape and failed with a `qweight` shape mismatch on exactly
  this checkpoint -- the deeper reason slice 1's tied-embedding fix alone was not enough to
  load it. Fixed: `Qwen3Config` gains `head_dim: Option<usize>` (`#[serde(default)]`, matching
  HF's own optional field) and `effective_head_dim()`, which every call site now uses instead
  of recomputing the formula. Red-first, synthetic `VarBuilder` fixture (`head_dim: Some(256)`
  against a config whose formula gives 64, CPU-only, no GPU/corpus dependency): demonstrated
  failing against the pre-fix code (`"Failed to load q_proj: Failed to load qweight"`), passing
  after.
- AWQ checkpoint pinned for slice 2's end-to-end proof: `Orion-zhen/Qwen3-0.6B-AWQ` at commit
  `9f607bee5fca19cba0bd2b383ba804c89385ea18` (no tagged release; HEAD of `main` as of
  2026-10-22). Licence: the AWQ repo itself is `gpl-3.0`; its base model `Qwen/Qwen3-0.6B` is
  `apache-2.0`. Recorded in `src/test_models.rs` (`AWQ_MODEL_REPO`, `AWQ_BASE_MODEL_REPO`,
  `AWQ_PINNED_REVISION`) alongside a `LIGHTBULB_AWQ_MODEL_DIR`-resolving `awq_qwen3_dir()`,
  mirroring `tinyllama_dir`'s convention. The end-to-end GPU generation-vs-reference-
  implementation test itself is blocked on a Windows packaging gap in `autoawq` (no prebuilt
  wheel for this platform/Python combination; its `triton` dependency has no Windows wheel on
  PyPI either) -- not started; see the PM thread for the options.

### Breaking
- `Qwen3Config` (`src/model/awq_qwen3.rs`) gains a mandatory `head_dim: Option<usize>` field
  for direct struct construction (Rust struct literals don't honor `#[serde(default)]`; only
  `serde` deserialization does) -- same class of change, and reason, as 0.14.0's
  `tie_word_embeddings` bump.

## 0.14.0

### Added
- **AWQ slice 1: `AwqQwen3` now loads tied-embedding Qwen3 checkpoints (PM-tasked 2026-10-21).**
  Every Qwen3 dense model up to ~4B ships `tie_word_embeddings: true` -- `lm_head` is a
  shared-tensor alias to `embed_tokens.weight` (confirmed via a safetensors header range-fetch
  on `Orion-zhen/Qwen3-0.6B-AWQ`: `__metadata__` carries `{"lm_head.weight":
  "model.embed_tokens.weight"}`), never its own AWQ-quantized tensor set. `AwqQwen3::new`
  previously built an `AwqLinear` for `lm_head` unconditionally, failing on every such
  checkpoint. A new `LmHead` enum (`Quantized`/`Tied`) checks the tensor's actual presence
  first, the config flag only as a fallback. Red-first, with a synthetic in-memory
  `VarBuilder::from_tensors` fixture (no downloaded weights): demonstrated failing against the
  pre-fix code, passing after. `cargo test --lib`: 889 passed, 0 failed, 25 ignored.
- ROADMAP.md's AWQ slice plan corrected: this is now slice 1 (done); the previously-sized
  "prove correctness end-to-end against a real checkpoint" work moves to slice 2.

### Breaking
- `Qwen3Config` (`src/model/awq_qwen3.rs`) gains a mandatory `tie_word_embeddings: bool` field
  for direct struct construction (Rust struct literals don't honor `#[serde(default)]`; only
  `serde` deserialization does) -- same class of change as 5.4a/M4's priority field, hence the
  second-number bump pre-1.0.

## 0.13.5

### Added
- **Multi-GPU hardware limitation note + AWQ loader wiring slice plan (PM-tasked 2026-10-21).**
  (1) Recorded that this box has exactly one GPU, so the 8 known multi-GPU bail sites
  (`src/multi_gpu/*.rs`) are UNVERIFIED BY HARDWARE here -- design/unit review only, no
  multi-GPU work started. (2) ROADMAP's "Phase 3 IN PROGRESS: Model loader integration" claim
  for AWQ was wrong -- measured at 0% wired: `load_awq_llama` has zero callers anywhere, no
  `--awq` flag exists, and `ParallelModelManager::load`/`load_adaptive` both call
  `load_local_llama` unconditionally. Sized as a 2-slice plan: slice 1 proves the existing
  loader/model correct standalone against a real small AWQ checkpoint and a pinned reference
  (needs finding a small-enough model first); slice 2 wires it into the real batched serving
  engine (not sized yet -- `AwqQwen3`'s forward pass doesn't match the engine's batched
  interface, and which fix is right depends on slice 1's findings). Docs only, no code changed.

## 0.13.4

### Changed
- **Parked ROADMAP.md's "task A" (IQ3_XS/IQ4_XS/Q2_K dtype refusals) as blocked on the Fuel
  migration (PM-tasked 2026-10-21).** candlelight's vendored `candle-core` 0.10.2 (the shipping
  default path) is missing dequantization kernels for the entire IQ codebook family (dtype
  codes 16-29) -- 180-210 of 272 tensors in each of the 3 refusing files, not "one tensor."
  Fuel's own GAP-211 B2 already implemented this class of decode for Fuel's path (`0.20.14`);
  writing the same kernels into candlelight now, a fork lightbulb is migrating away from, would
  be throwaway work. Docs only, no code changed.

## 0.13.3

### Fixed
- **`stories15M-q4_0.gguf`'s "unexplained refusal" in `gguf_serving_census`, closed (PM-tasked
  2026-10-21).** Root cause: it is `LLAMA_SPM_VOCAB_SHA256`'s own oracle-verified 32000-token
  vocabulary with exactly one token (id 30143) corrupted by this file's own conversion, from
  `"\u{feff}"` (BOM) to `""` -- confirmed by a token-by-token diff against three independent
  sources (its own sibling `stories15M-q8_0.gguf`, `tinyllama-1.1b-chat-v1.0.Q4_0.gguf`,
  `ggml-vocab-llama-spm.gguf`), not a new or different vocabulary. Added
  `STORIES15M_Q4_VOCAB_SHA256` to `spm_derivation_warrant`'s allowlist, with the strongest
  warrant of the four weaker entries (an oracle-verified parent vocabulary, not "no oracle
  exists"). Red-first test:
  `stories15m_q4_0_shares_the_llama_vocabulary_minus_one_corrupted_token` -- demonstrated
  failing with the allowlist entry removed (`the allowlist entry for this digest is missing`),
  passing with it restored. `gguf_serving_census` (local corpus only, never hosted CI) goes
  from 9/13 to 10/13 weighted files loading, 0 "unexplained" refusals (was 1).

## 0.13.2

### Fixed
- **Retracted ROADMAP.md's stale "GGUF loads 1 model in 30" claim (PM-tasked
  2026-10-21).** It was never re-derived against this repo's own later-measured
  numbers (2026-09-09: 23/30 tokenizers rebuilt, 8/11 weighted files load) and
  pointed at a pre-tokenizer-rule avenue that is EXHAUSTED, not open -- every
  `tokenizer.ggml.pre` value in the local corpus already has a verified
  implementation or a documented permanent refusal. Replaced with today's
  re-measurement (`main` `f475ed8`, corpus `C:\Models`, 32 files, run only
  against a local `LIGHTBULB_GGUF_CORPUS` -- never hosted CI): 24/32 tokenizers
  rebuild, 13 weighted files (9 load, 3 refuse on an unrepresentable IQ/K-quant
  dtype, 1 refuses for a new, as-yet-unexplained reason -- tracked as its own
  follow-up PR, not fixed here). Docs only, no code changed.

### Added
- **PersistAnt adoption status, added to the Fuel-migration sizing doc (PM-asked 2026-10-21).**
  ROADMAP.md now records that PersistAnt adoption is already complete in shipped code
  (`persistant-sqlx`/`persistant-relational` 0.7.4, "Step 5") for lightbulb's Postgres-backed
  API keys, audit logs, sessions, rate-limit usage and auth-attempt-count storage --
  `tokio-postgres`/`refinery`/`postgres-types` remain only in `[dev-dependencies]`, deliberately,
  for upgrade-path and raw-schema tests. Nothing blocked.

### Fixed
- **RustSec lock-only update.** `cargo update` for the semver-compatible fixes:
  `rustls` 0.23.45 -> 0.23.46 (RUSTSEC-2026-0346/0347/0348/0349), `rustls-webpki` 0.103.15 ->
  0.103.16 (RUSTSEC-2026-0382), `webpki-root-certs` 1.0.9 -> 1.0.10 (RUSTSEC-2026-0383),
  `webpki-roots` 1.0.9 -> 1.0.10 (RUSTSEC-2026-0381; the OTHER `webpki-roots` line in the lock,
  0.26.11, pulled by `ureq` v2.12.1 via our own `hf-hub` v0.4.3 dependency, is **not** flagged by
  any advisory -- confirmed via `cargo audit`, not assumed), `h2` 0.4.15 -> 0.4.20 and
  `quinn-proto` 0.11.16 -> 0.11.19 (RUSTSEC-2026-0258/0338/0339/0340/0341/0342, found by the same
  audit pass though not in the PM's original list). `cargo check --bins` and `cargo test --lib`
  (886 passed) both green after the bump.
- **Not fixable from this repo, reported rather than attempted:** `zip` 7.2.0 (pulled by
  `candle-core` via `candlelight`) and `zip` 8.6.0 (pulled by `fuel-core` and others via `fuel`)
  both need >=9.0.0 (RUSTSEC-2026-0367/0368/0369) -- both are transitive pins one major version
  below the fix, owned by candlelight's and fuel's own `Cargo.toml`, not lightbulb's; `cargo
  update -p zip` cannot cross a major-version boundary a dependent itself declared. Likewise
  `h2` 0.3.27 (RUSTSEC-2026-0258, pulled by `distributed-config` v0.1.0's own `reqwest` v0.11
  dependency) has no newer `distributed-config` release to move to (checked crates.io: `0.1.0`
  is still the newest). Both need the owning upstream crate to bump its own dependency, not a
  lightbulb-side fix.

## 0.13.0

### Added
- **M4 slice 5.4c: real request priority, Option B (PM-ruled 2026-10-21).** `InferenceJob` gains
  a `priority: Priority` field, read from the `X-Lightbulb-Priority` HTTP header on
  `/v1/chat/completions` and `/v1/completions` (never the OpenAI-compatible JSON body -- see
  `docs/API-EXTENSIONS.md`), defaulting to `Normal` when the header is absent or unparseable.
  `admit_pending` now submits via `MemoryAwareScheduler::submit_priority_request` with the job's
  own priority instead of the plain `submit_request` (which always submitted `Normal`
  regardless) -- under eviction pressure, a `High`/`Critical` request is now queued rather than
  rejected where a `Low`/`Normal` one in the identical state is rejected outright. Red-first
  test: `admit_pending_honors_the_jobs_own_priority_not_always_normal`
  (`src/engine/model_runner.rs`), which fails against the pre-fix code (both priorities reject
  identically) and passes against the fix.

### Breaking
- `InferenceJob` (`src/engine/model_runner.rs`) gains a mandatory `priority: Priority` field --
  same class of change as 5.4a's `AppState.scheduler` type change (0.11.6 -> 0.12.0), hence the
  second-number bump pre-1.0. Every in-tree construction site is updated; an out-of-tree
  constructor would need to add the field.

## 0.12.4

### Added
- **Fuel dependency migration sizing (PM-tasked 2026-10-21).** ROADMAP.md gets a new section
  sizing the move from lightbulb's git-pinned Fuel rev to crates.io: `fuel` and its 12 companion
  crates are all published at `0.20.14` (checked live against the crates.io API), the pinned rev
  predates Fuel's GAP-348 `fuel-core` dissolution (so `Cargo.toml`'s `package = "fuel-core"`
  alias names a package that no longer exists upstream), and lightbulb's own exposure is lower
  than expected -- zero `fuel_core::` path references anywhere in `src/`, only nine
  `fuel::lazy::` facade-path usages across five files. Docs only, no dependency or code change;
  this is the sizing that must land before the migration itself, per the PM's ordering.

## 0.12.3

### Changed
- **M4 slice 5.5 verification finding: the `run_jobs` complexity refactor (ROADMAP.md slice plan
  item 6) appears already done**, as a side effect of extraction landed across slices 5.1-5.4b,
  not as dedicated 5.5 work. `run_jobs` is now 46 lines with the streaming/complete-mode branches
  already split into named helpers (`stream_job_step`, `finish_complete_job`,
  `demux_step_results`, `admit_pending`, `remove_cancelled_before_step`, `fail_whole_batch`);
  manual cyclomatic count on the remaining body is ~7, against item 6's cited limit of 12 (it was
  18 pre-extraction). All 10 `engine::model_runner::tests::*` unit tests pass. Flagged as a
  finding rather than closed outright: this is a manual estimate, not a read of Codacy's own
  number, since no PR in the slice plan touched `run_jobs` for its own sake to generate one. Docs
  only, no code changed.

## 0.12.2

### Added
- **M4 slice 5.4c deferred, with the real decision laid out.** Real `Priority` is blocked on how
  a caller specifies it, not on sizing: OpenAI's chat/completions schema has no priority field.
  ROADMAP.md now lays out the two real options (a non-standard top-level JSON field vs. a custom
  HTTP header) with their actual tradeoffs, a recommendation (the header, for an OpenAI-
  *compatible* API specifically), and an explicit answer to "does this need CireSnave" (no --
  single-project API extension, not a cross-portfolio or resource decision). Docs only, no code
  changed.

## 0.12.1

### Added
- **M4 slice 5.4b: `scheduler_stats`/`scheduler_queue` read the real shared scheduler**
  (`MemoryAwareScheduler::get_memory_stats()`), not `unavailable_reason`/`null` always. Real now:
  `scheduler_queue`'s `pending_requests`/`running_requests`; `scheduler_stats`'s new
  `active_slots`/`queued_requests`/`scheduler_memory_used_bytes`/`eviction_pressure` fields. Still
  unavailable, correctly: `queue` (the per-request list -- `priority_queue` has no enumeration
  API) and `scheduler_stats`'s `total_requests`/`completed_requests`/`failed_requests`/
  `average_latency_ms`/`throughput_tokens_per_sec` (`GET /metrics` is the right home, per the
  PM's 2026-10-16 ruling, not a duplicate counter here).
- **`cache_clear`'s `_scheduler` binding decided: left alone.** Calling `free_slot` on active
  admission slots from a "clear the prefix cache" endpoint would kill in-flight requests, not
  clear a cache -- the scheduler was never the right lever for this endpoint.
- **Corrected in place**, per the retraction rule: `scheduler_queue_reports_unavailable_not_zero`
  asserted "always null," which is now simply wrong on an idle scheduler (a real `0`). Renamed to
  `scheduler_queue_reports_real_zero_not_null_when_idle` with corrected assertions, plus two new
  tests (`..._reports_real_occupancy_once_a_request_is_admitted`, one per endpoint) priming the
  shared scheduler directly and reading the real occupancy back.
- `cargo test --lib`: 884 passed, 0 failed, 24 ignored (was 882/0/24 -- the +2 are the new
  occupancy tests). `scripts/clippy_gate.py`: clean. Both checkpoint-needing tests re-run by hand
  against real TinyLlama after this change: both still pass.

## 0.12.0

### Changed (breaking)
- **M4 slice 5.4a: `AppState.scheduler` is now `SharedScheduler` (`Arc<Mutex<MemoryAwareScheduler>>`),
  not `Arc<MemoryAwareScheduler>`** -- a public-field type change (pre-1.0, hence the 0.n bump).
  `check_admission`'s own `&self`-only design (slice 2) can't support `submit_request`/
  `allocate_pending_requests`/`free_slot`, which all need `&mut self`; interior mutability is what
  lets both the HTTP layer (reading) and `run_jobs` (writing, on its own thread) hold the same
  scheduler. `ApiServer::new`'s `scheduler` parameter and every `MemoryAwareScheduler` construction
  site across the tree (production and test fixtures) updated to match.
- `run_jobs` now mirrors real admission and departure into the shared scheduler's own `SlotPool`
  bookkeeping, not just its own `batch`/`modes`: `admit_pending` calls `submit_request` +
  `allocate_pending_requests` when a job joins (a chars/4 placeholder for `prompt_tokens`, the
  same heuristic `check_admission` already uses -- real tokenization hasn't happened yet at
  admission time); every removal site calls the new `release_scheduler_slot` helper,
  unconditionally (the scheduler mirror has no "the backend already freed it" case to skip,
  unlike `abandon_if_not_completed`). No real admission limit enforced yet -- still this sub-slice's
  scope, not 5.4a's.
- Red-first test `run_jobs_mirrors_real_admission_and_departure_into_the_shared_scheduler`: two
  concurrent jobs, the scheduler's tracked occupancy reaches 2 while both are in flight (impossible
  before -- nothing called `submit_request`/`free_slot` from `run_jobs` at all) and returns to 0
  once both complete.
- Re-verified both prior checkpoint tests (slice 5.1's cache-reuse-safety test, the cache-leak fix)
  against the real TinyLlama checkpoint after this change: both still pass.
- `cargo test --lib`: 882 passed, 0 failed, 24 ignored. `scripts/clippy_gate.py`: clean, no new or
  increased lints.

## 0.11.6

### Added
- **M4 slice 5.4 sub-plan.** Reading `MemoryAwareScheduler`/`SlotPool` before slicing found a real
  shape mismatch: `SlotPool` is its own independent bookkeeping state machine, not something that
  reads or drives `run_jobs`'s real stepping -- wiring it in means keeping a second, parallel
  bookkeeping structure in sync, not replacing the real one. Also: `AppState.scheduler` is
  `Arc<MemoryAwareScheduler>` (no interior mutability) today, but `submit_request`/`free_slot`
  need `&mut self`. ROADMAP.md now sizes 5.4 into 5.4a (make the scheduler mutably shareable, sync
  real admission/departure, red-first), 5.4b (wire `scheduler_stats`/`scheduler_queue` to the real
  shared scheduler), 5.4c (real `Priority`, sized but not designed -- the API-body shape is a real
  decision for whoever reviews it). Docs only, no code changed.

## 0.11.5

### Changed
- **M4 slice 5.3: `run_jobs` now steps every live request together, in one `step_batch` call per
  round** (`src/engine/model_runner.rs`), instead of one job running alone to completion before
  the next is even admitted (5.2's shape). New helpers: `admit_pending` (grows the active set,
  no cap yet -- that's slice 5.4's `SlotPool` job), `remove_cancelled_before_step` (the multi-job
  counterpart of the old `resp_tx.is_closed()` precheck for `Complete` mode), `demux_step_results`
  (routes each round's result to its own request's channel, using a per-job previous-token-count
  snapshot to detect a new token without depending on `step_batch`'s own unread/unreconciled
  return value), `fail_whole_batch` (a `step_batch` error aborts every active job together, per
  the trait's own doc -- no per-request granularity exists to do otherwise).
- Red-first test `run_jobs_steps_concurrent_requests_together_without_cross_contamination`: two
  jobs submitted together, `step_batch` observed with `batch.len() >= 2` while both are in
  flight, each still produces its own correct, uncontaminated output. Fails against pre-5.3 code
  (`batch.len()` was always 1); passes against this.
- **Corrected in place**, per the retraction rule: slice 5.2's acceptance test asserted "no
  interleaving between jobs" as its own stated temporary scope, not a permanent guarantee --
  interleaving is the entire point of 5.3, so that assertion is now simply wrong. Replaced with
  the real invariant (above), not merely annotated.
- Re-verified the slice-5.1-adjacent cache-leak fix (#179) against 5.3's new concurrent
  admission: still passes, and for a sound reason -- a cancelled job is removed in the same round
  it's stepped, before any other job can join it in the active batch.
- `cargo test --lib`: 881 passed, 0 failed, 24 ignored (fast suite) + the checkpoint-needing
  tests individually confirmed passing. `scripts/clippy_gate.py`: clean, no new or increased
  lints (one genuinely new `clippy::while_let_loop` finding on the admission loop's own
  `try_recv` drain, fixed by using `while let` directly, not suppressed).

## 0.11.4

### Fixed
- **Cache-slot leak on request cancellation -- a real, pre-existing DoS, found while designing
  M4 slice 5.3, not introduced by it.** `ParallelModelManager::release_cache_index` was only ever
  called from `forward_batch`'s own completion branch. `run_jobs`'s four cancellation/error exit
  paths (a closed `Complete` receiver, a closed `Streaming` receiver, and a `step_batch` error on
  either mode) all dropped `RequestContext` without a final call that would reach that branch --
  leaking the cache index permanently. With the production default `model_max_batch_size: 8`, 8
  such cancellations over the server's uptime (cumulative, not concurrent) exhausted the pool;
  the 9th request's prefill hit `allocate_cache_index().expect("No available cache indices in
  pool")` and panicked the model-runner thread, permanently killing inference with no clean
  recovery short of a process restart.
- **Reproduced live before fixing**, not just read: new test
  `a_cancelled_request_leaks_its_cache_slot_and_the_next_request_panics` (real TinyLlama,
  `max_batch_size: 1` -- one cancellation is enough to exhaust a one-slot pool, same mechanism,
  fewer repetitions) panicked against the pre-fix code exactly as described, then passed once
  fixed.
- **Fix:** new `EngineModel::abandon(&mut self, ctx: &RequestContext)` trait method (default
  no-op), called once by `run_jobs` per job whenever `ctx.cache_index.is_some() && ctx.state !=
  RequestState::Completed` -- i.e. a slot was allocated and the backend's own completion path
  never got to release it. `ParallelModelManager`'s impl calls the now-`pub(crate)`
  `release_cache_index`. Not published to crates.io (confirmed: crates.io returns 404 for
  `lightbulb`), so no GHSA/RustSec advisory applies.
- `cargo test --lib`: 881 passed, 0 failed, 24 ignored (the checkpoint-needing reproduction test
  is skipped by this count, same policy as slice 5.1's). `scripts/clippy_gate.py`: clean, no new
  or increased lints.

## 0.11.3

### Changed
- **M4 slice 5.2: `run_jobs`'s admission loop is non-blocking once a job is already in hand**,
  with no intended behavior change. Previously `while let Ok(job) = rx.recv()` blocked on the
  channel between every single job; now a `pending: VecDeque<InferenceJob>` drains everything
  already queued (`rx.try_recv()`) before each job is taken off the front, falling back to a
  blocking `recv()` only when `pending` is empty. Jobs still process one at a time, in submission
  order, each fully to completion -- this is pure plumbing for slice 5.3's real concurrent
  stepping, not a behavior change by itself.
- New test `run_jobs_processes_already_queued_jobs_one_at_a_time_in_submission_order`
  (`src/engine/model_runner.rs`): three jobs with different `max_new_tokens` (2, 3, 1), submitted
  before `run_jobs` is called so the channel already holds all three on entry -- exactly the
  "jobs already queued" case the drain touches. Confirmed green against the admission loop
  **before** the refactor (baseline) and **after** (unchanged): same processing order
  (`["j1","j1","j2","j2","j2","j3"]`, no interleaving), same per-job token counts and finish
  reasons.
- `cargo test --lib`: 881 passed, 0 failed, 24 ignored (was 880/0/24 -- the +1 is the new test;
  every pre-existing `run_jobs`-calling test in this module still passes unchanged).
  `scripts/clippy_gate.py`: clean, no new or increased lints.

## 0.11.2

### Added
- **M4 slice 5.1: measured, not just read, that cache-slot reuse is safe across unrelated
  requests.** New test `cache_index_reused_by_a_later_unrelated_request_is_not_corrupted`
  (`tests/parallel_model_manager_integration.rs`): request A runs to completion and frees its
  cache index, unrelated request B is admitted into a fresh batch and necessarily reuses that
  index (`max_batch_size: 1`); B's output is compared bit-for-bit against a control run of B
  alone on a separate model instance. Measured against a real TinyLlama checkpoint, CPU f32:
  passes -- identical tokens both ways. De-risks `EngineModel::step_batch`'s documented worry
  about a scheduler stepping a changing set of live requests corrupting cache state, before any
  of slice 5.2+ depends on the answer.
- **Deliberately not `#[ignore]`d**, per CireSnave's standing rule (EXPECTATIONS §5.1g/5.1h): a
  test that can run on the self-hosted runner runs there rather than being silently excluded via
  `#[ignore]`. The self-hosted job is being routed (PM); until it lands, `ci.yml`'s hosted `test`
  step skips this one test by name instead.
- ROADMAP.md's "Test coverage limits" section gets a self-hosted-runner debt inventory: every
  other `#[ignore]` in the tree, by file, categorized (Postgres-covered-already /
  checkpoint-needing / GPU-needing / mixed / not-yet-triaged) with counts from `grep -c`.

## 0.11.1

### Added
- **M4 slice 5 (continuous batching) sub-slice plan**, per the PM's "sub-sliced, each slice
  red-first" direction. ROADMAP.md now sizes it into 5.1-5.5: 5.1 verifies (doesn't assume) that
  `ParallelModelManager::forward_batch` already tolerates the variable-live-request-set call
  pattern continuous batching needs, before anything in `run_jobs` depends on it; 5.2 restructures
  `run_jobs`'s admission loop with no behavior change; 5.3 is the real concurrent-stepping change;
  5.4 wires `MemoryAwareScheduler`'s `SlotPool` in for real `scheduler_stats`/`scheduler_queue`/
  priority data (closing what slices 3/4 deferred here); 5.5 is the already-known `run_jobs`
  complexity refactor, sequenced last. Docs only, no code changed.

## 0.11.0

### Removed (breaking)
- **Deleted `src/engine/pipeline.rs` (`Pipeline`/`PipelineBuilder`/`PipelineConfig`/`Stage`/etc.),
  dead since it was added.** PM-directed decision (board follow-up on the 2026-10-14 Phase 4
  audit's "Pipeline execution driver" finding): decide wire-or-delete with evidence. Evidence for
  delete: zero callers anywhere outside the file and its own blanket re-export (`git grep` across
  `src/ tests/ examples/ benches/`); no concrete roadmap plan names it as an integration point
  (CR.1-CR.4 target `ModelRunner`'s inference loop directly; `docs/SCS_MODEL_SUPPORT.md`'s
  "pipeline" is an unrelated, separately-planned `ScsInferencePipeline`); and unlike its two
  siblings from the same origin commit (`KnowledgeBase`, `Decomposition`, both of which did get
  real callers), this module never found one across the project's whole history. Building a real
  async execution driver for it would be new feature work with no identified requester, not a
  gap-fill. `lightbulb::engine::{Pipeline, PipelineBuilder, PipelineConfig, PipelineError,
  PipelineId, PipelineStats, Stage, StageData, StageId, StageResult, StageState, StageType,
  CacheRef}` are no longer part of the public API -- hence the 0.n major bump, pre-1.0. See
  ROADMAP.md's "Pipeline execution driver" entry for the full evidence trail.

## 0.10.14

### Corrected
- **ROADMAP.md's M4 slice 1 was stale-marked open.** It shipped in #139:
  `src/api/admin.rs::cache_stats` reports the real, shared
  `AppState.cache_stats` handle (not the hardcoded-zero mock), proven by
  `cache_stats_reports_the_shared_handles_real_values_not_a_mock` -- both
  confirmed present and real at `origin/main` before writing this fix, not
  just the merged PR's title. Corrected the entry in place to match slice
  2's existing "DONE, lightbulb#NNN" format. Docs only, no code changed.

## 0.10.13

### Fixed
- **PM hold on PR#173, item 1 -- start-degraded behavior restored.**
  `ApiServer::new` now calls a new `connect_with_retry` helper instead of
  `PgBackend::connect` directly: 5 attempts, each capped at a 2s connect
  timeout, with 200ms-doubling-to-1600ms delays between tries -- at most
  ~13s of total patience. **Measured, not assumed:** a first version of
  this fix called `PgBackend::connect` directly in the retry loop and
  measured ~155s to exhaust 5 attempts (sqlx's own `acquire_timeout`
  defaults to 30s per attempt, and `connect()` gives no way to override
  it), nowhere near the "momentary blip" this retry is supposed to cost.
  Fixed by building the `sqlx::PgPool` directly with an explicit 2s
  `acquire_timeout`, then handing it to `PgBackend::from_pool` (the same
  public constructor `connect()` uses internally) -- see
  `try_connect_once`. `sqlx` is now a direct dependency for exactly this
  (PersistAnt already pulls in the identical version transitively; this
  does not reintroduce a separate Postgres driver).

  If the database is still unreachable once the retries are exhausted,
  the server now starts in the SAME degraded state it always has when
  `DATABASE_URL` is merely unset (`pg_backend: None`, auth/audit/
  rate-limiting disabled) instead of `ApiServer::new` itself returning an
  error. This restores the pre-Step-5 `deadpool_postgres` behavior
  (always lazy, so a bad connection string there also only ever surfaced
  as a readiness-check failure on first use) rather than introducing a
  new, stricter failure mode. Tradeoff stated explicitly: a genuinely
  misconfigured `DATABASE_URL` (a typo, not a blip) now starts
  successfully with auth silently disabled rather than failing fast at
  startup -- the same tradeoff this repo always had before Step 5's
  eager `connect()` briefly removed it. Proven end to end (not just by
  reading the code) by a new test,
  `api_server_new_starts_degraded_on_unreachable_database`: measured
  ~15s against a genuinely unreachable address, down from the ~155s the
  first version of this fix would have cost.
- **PM hold item 2 -- red-first test for `CheckStatus::Unreachable`.**
  Added `check_database_reports_unreachable_when_a_live_backend_goes_away`
  (`src/api/mod.rs`): a real throwaway Postgres behind a small local TCP
  proxy this test runs itself, so the initial connect and one `Ok` check
  succeed through it, then the proxy is severed (stops accepting AND
  drops every connection already open) and a second check must report
  `Unreachable`. Two simpler approaches were tried and measured NOT to
  work (both still in the test's own doc comment): `pg_terminate_backend`
  alone (sqlx's pool silently opens a fresh connection on the next query),
  and additionally revoking the role's `CONNECT` privilege (the test role
  is a superuser in both this repo's own Docker verification and CI's
  native `initdb`-created cluster -- confirmed directly against both --
  and superusers bypass `CONNECT` privilege checks entirely).
- **Found while verifying, not part of the PM's hold but blocking
  `postgres_integration` CI on the already-pushed commit:**
  `tests/persistant_migrate_tests.rs`'s `migrate_with_refinery` helper now
  resets the schema (`DROP SCHEMA public CASCADE; CREATE SCHEMA public;`)
  before running refinery. The CI job runs `api_integration_tests` and
  this file back to back against the SAME shared Postgres cluster within
  one job; once Step 5 made `api_integration_tests` (via `ApiServer::new`)
  migrate through `PgBackend::migrate()` instead of refinery, that run
  left behind the same tables/indexes this file's raw refinery run also
  tries to create -- none of the original `.sql` files' `CREATE INDEX`
  statements have `IF NOT EXISTS` (only `persistant_migrate`'s ported
  copies do), so refinery failed with "relation ... already exists".
  Confirmed in the real CI log for run 39058146345. Verified fixed by
  reproducing CI's exact sequence (`api_integration_tests` then
  `persistant_migrate_tests`, same cluster, same container) twice.

## 0.10.12

### Added
- **PersistAnt adoption complete.** Step 5 (docs/PERSISTANT_ADOPTION_DESIGN.md
  §5 item 5), the last step: `persistant_sqlx::PgBackend` is now the sole
  database path for the admin/auth Postgres surface. `ApiServer::new` builds
  only a `PgBackend` and runs `backend.migrate(...)` as the sole migration
  path; `deadpool_postgres`, `tokio_postgres` and `refinery` are gone from
  this repo's `[dependencies]` (kept as dev-dependencies only, scoped to
  `tests/persistant_migrate_tests.rs`'s own simulation of a pre-Step-5
  deployment -- see that file's `migrate_with_refinery` helper).
- Ported the `auth_attempt_counts` cleanup background task and
  `check_database` (the readiness-check helper) to `pg_backend`. Deleted
  `get_db_client` (admin.rs) -- dead code, zero callers after Steps 1-4.

### Fixed (finding)
- **`PgBackend::connect` is EAGER**, unlike the prior
  `deadpool_postgres::Pool::builder(...).build()`, which never touched the
  network until the first `.get()`. `connect()` opens a real connection and
  runs `SHOW server_version` before returning. This is a genuine behavior
  change: a momentarily-unreachable database now fails `ApiServer::new`
  itself (the server refuses to start) rather than starting and reporting
  `not_configured`/`unreachable` via `/health/ready` until the database
  comes up. Flagged for the PM's attention, not silently absorbed.
  `readiness_check_is_not_ready_when_database_is_unreachable` (a red test
  that injected a lazily-built pool pointed at an unreachable address) had
  no direct equivalent for the same reason -- replaced with
  `connecting_to_an_unreachable_database_fails`, which proves the same
  claim one layer lower (at `connect()` itself); `check_database`'s
  `CheckStatus::Unreachable` arm (connects fine, a later query fails) has
  no red test post-Step-5, a real coverage gap, noted rather than hidden.

### Corrected
- `docs/PERSISTANT_ADOPTION_DESIGN.md`'s §5 item 6 ("drop `user_sessions`
  as part of this work") and §7 Open Question 2 are corrected in place:
  the PM's actual answer -- *"carry user_sessions forward unchanged;
  dropping the dead table is its own later PR (irreversible = own
  decision)"* -- overrides the doc's original text, which was written
  before that answer. `user_sessions` (`V3`, dead schema) is unchanged by
  this PR; dropping it is deferred to its own future PR.

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
