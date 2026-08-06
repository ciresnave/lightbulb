# Engine-wiring Fuel baseline

**Date:** 2026-08-05

## FUEL_BASELINE

```
FUEL_BASELINE=12df216b89a832cd20408c146227b0af8375b805
```

Short SHA: `12df216b`

Commit: `fix(decode)!: key model identity on a never-recycled counter, not weight
pointers` (2026-08-05, CireSnave). The `!` marks a declared breaking change to
the decode path. This is the tip of `origin/main` in the Fuel repository as of
this dispatch — every later measurement in this plan (Task 10's sweep and
onward) quotes this SHA as the Fuel state it ran against.

This commit is at or after `af93e318` (the commit that landed `PlanOnce` as
the paged-decode default), so a sweep run against this baseline measures the
configuration that actually ships, not the stale `Replan` default that
`fuel-lightbulb-port` was previously detached at (`99dbf231`).

## Worktree update

`C:/Projects/fuel-lightbulb-port` (the path-dependency source for `fuel`,
`fuel-cpu-backend`, `fuel-cuda-backend`, and `fuel-inference` in this repo's
`Cargo.toml`) moved:

- **Before:** detached HEAD at `99dbf231` (`docs(lazy): correct stale
  paged-decode f32-only comments`), plus an uncommitted local modification to
  `fuel-cuda-backend/src/baracuda/attention.rs` (a baracuda alpha.78 ABI fix
  authored in this repo, adding `a`/`a_mean` optional-output parameters to the
  `FlashDecodingRun` FFI alias).
- **After:** detached HEAD at `12df216b89a832cd20408c146227b0af8375b805`
  (`origin/main` tip at fetch time), working tree clean.

`git checkout main` (as literally specified) failed — Fuel's `main` branch was
already checked out in the read-only shared worktree at `C:/Projects/fuel`
(`fatal: 'main' is already used by worktree at 'C:/Projects/fuel'`), and git
worktrees enforce one checkout per branch. This was not anticipated by the
task brief. Landed on `origin/main`'s tip via `git checkout --detach
origin/main` instead, which achieves the same functional worktree state (files
on disk match main's tip) without contending for the `main` branch ref itself
— Lightbulb's path dependency only cares about the checked-out file contents,
not which ref/branch owns them.

## attention.rs local edits

Before updating, verified that `origin/main`'s copy of
`fuel-cuda-backend/src/baracuda/attention.rs` already carries an equivalent
(and better-commented) fix: the `FlashDecodingRun` type alias declares `a: *mut
std::ffi::c_void` and `a_mean: *mut std::ffi::c_void` between `y` and
`workspace`, identical in shape and position to the local diff, and the call
site passes `core::ptr::null_mut()` for both — same behavior as the local
edit. Confirmed via `git show origin/main:fuel-cuda-backend/src/baracuda/attention.rs`
after fetch, and by walking `git log --oneline -- fuel-cuda-backend/src/baracuda/attention.rs`,
which shows `fe4fc446 fix(cuda)!: flash_decoding ABI — baracuda alpha.78 added
a/a_mean outputs` as the commit that landed it upstream. (The brief's dispatch
note cited `38d68f86` for this; the SHA differs from `fe4fc446`, likely due to
history rewriting/rebasing on Fuel's side, but the file content match is what
was actually checked, not the SHA.)

Since upstream already carries the equivalent fix, the local edit was
redundant. Discarded it with:

```
git checkout -- fuel-cuda-backend/src/baracuda/attention.rs
```

before checking out `origin/main`. No work was lost — the change is present
upstream in a superior form (clearer comments explaining the null-encoding and
the `a_mean` requires-`a` relationship).

## `cargo check --lib` result (Lightbulb, post-update)

Ran in `C:\Projects\lightbulb` (not workspace-wide — this is the crate root
here, so `--lib` alone is correct per the standing constraint about
`tensor-tools`):

```
cargo check --lib 2>&1 | tail -60; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Result: **`CARGO EXIT: 0`**. Compiled clean — 44 pre-existing dead-code/unused-field
warnings (unrelated to Fuel, e.g. `MemorySample::timestamp`,
`Qwen3RotaryEmbedding::head_dim`, `WandaScorer::config`), zero errors. No Fuel
API drift affected Lightbulb's call sites despite the breaking-change marker
(`!`) on the tip commit's decode-identity rework — that rework's surface
apparently doesn't touch the parts of Fuel's API that Lightbulb currently
calls. No call-site fixes were required.

## Summary

| | Before | After |
|---|---|---|
| Fuel worktree HEAD | `99dbf231` (detached) | `12df216b89a832cd20408c146227b0af8375b805` (detached) |
| attention.rs | local uncommitted diff (redundant) | clean, matches upstream |
| `cargo check --lib` | not run this session | exit 0, 0 errors |

`FUEL_BASELINE=12df216b89a832cd20408c146227b0af8375b805` is the SHA every
later measurement in this plan (Task 10's sweep and onward) must quote.
