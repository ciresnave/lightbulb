# Changelog

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
