# `model_fuel` → Engine Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Lightbulb's HTTP server able to serve real completions through the
Fuel-backed model path (`src/model_fuel/`), which today nothing calls.

**Architecture:** A Cargo feature `fuel-engine` swaps the body of
`ModelRunner::start` between the existing `candlelight` path and a new Fuel one.
The ~100-line job loop is shared via a generic, tensor-free `EngineModel` trait
so neither path duplicates it. Inside `model_fuel`, a `FuelDecoder` trait
abstracts *which* Fuel model type decodes, so the currently-blocked quantized
type arrives later as one added impl. `SessionState` owns the persistent
`DecodeSession` so no caller can accidentally take the 223×-slower path.

**Tech Stack:** Rust 2024, `fuel` (= `fuel-core` renamed, path dependency),
`tokenizers`, `axum` 0.8, `tower` 0.5 (`util` feature, for `ServiceExt::oneshot`),
`anyhow`.

## Global Constraints

Copied from `docs/superpowers/specs/2026-08-05-engine-wiring-design.md`. Every
task's requirements implicitly include these.

- **`C:\Projects\fuel` is READ-ONLY.** It is a shared working tree across
  multiple sessions. Never run a mutating git operation there. Read-only queries
  (`git log`, `git show`, `git diff`) are fine. Lightbulb's own worktree
  `C:\Projects\fuel-lightbulb-port` is fine to manipulate.
- **Never run workspace-wide `cargo check`/`test` in the Fuel repo.**
  `tensor-tools` has a standing break and is a default member. Always
  `-p <crate>`.
- **One `cargo` invocation at a time.** The build-directory lock serializes them.
- **Never trust a background task's exit code when piping.** `cmd | tail` reports
  `tail`'s status. Always echo `${PIPESTATUS[0]}` explicitly.
- **All GPU runs go through** `pwsh C:\Projects\fuel-crash-vmm\scripts\gpu-run.ps1 -Project lightbulb -- <cmd>`.
  It is a machine-wide mutex; bypassing it preceded a kernel bugcheck on
  2026-07-31.
- **`api/mod.rs` is not modified by this plan.** `ModelRunner::start`'s signature
  is unchanged.
- **`engine/` stays tensor-free.** No Fuel or candlelight tensor type may appear
  in any signature under `src/engine/`.
- **Numerical-drift claims are asserted on logits, never on tokens.** Argmax
  hides drift: a ~1.7e-3 perturbation flips neither greedy nor seeded-temperature
  tokens (measured by Fuel, 2026-08-05).
- **Measurement harnesses fail rather than skip** on a missing checkpoint or
  absent GPU. An early `return` from a `#[test]` is a PASS.
- The `fuel-cuda` build is ~24 minutes. Always `cargo check` before `cargo build`.

---

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `src/model_fuel/device.rs` | create | `select()` — the one place a `Device` is chosen. |
| `src/model_fuel/session.rs` | create | `SessionState` — KV cache + context + `DecodeSession` + position. Owns the persistence seam. |
| `src/model_fuel/decoder.rs` | create | `FuelDecoder` trait + `impl for LlamaModel`. The Axis-B seam. |
| `src/model_fuel/engine_model.rs` | create | `FuelEngineModel` — tokenizer, EOS, sampling, `RequestContext` mapping. Implements `EngineModel`. |
| `src/model_fuel/loader.rs` | modify | `LoadedLlama` gains `tokenizer`, `eos`, `device`. |
| `src/model_fuel/loader_f32.rs` | modify | Switch config parsing to `LlamaFullConfig`; populate the new fields. |
| `src/model_fuel/mod.rs` | modify | Declare the four new modules. |
| `src/engine/model_runner.rs` | modify | `EngineModel` trait, generic `run_jobs`, `cfg` swap. |
| `Cargo.toml` | modify | Add the `fuel-engine` feature. |
| `tests/fuel_engine_http.rs` | create | The acceptance gate. |
| `tests/gpu_paged_vs_contiguous.rs` | modify | Phase B: the batch-size sweep. |

**`src/api/mod.rs` is untouched.**

---

## Phase Structure

**Phase A (Tasks 1–9)** — sub-project 1. Ends with the server serving real
completions through Fuel.

**Phase B (Task 10)** — sub-project 2's *measurement*, which produces the
decision that sub-project 2's implementation needs.

**Why the batching implementation is not in this plan.** Which batched path to
build is an open empirical question: `build_batched_decode_logits` rejects
ragged `cached_len` outright, and `forward_paged_step_batched` costs 10.4× at
**k=1** — the one k that batching exists to avoid. Writing tasks for "whichever
wins" would require placeholder steps, which this plan format forbids and which
would be dishonest besides. Task 10 produces the curve; the batching
implementation gets a short follow-up plan once it exists. **Task 10 has no code
dependency on Tasks 1–9 and may be executed at any point after Task 1.**

---

## Task 1: Update the Fuel worktree and pin the baseline

Everything downstream compiles against this. Fuel's `PlanOnce` default landed on
main at `af93e318`; Lightbulb's worktree is detached at `99dbf231`, which
predates it. A sweep run before this task measures a configuration that no
longer ships — worse than a wrong number, because it looks current.

**Files:**
- Modify: `C:\Projects\fuel-lightbulb-port` (git worktree state only, no file edits)
- Create: `docs/superpowers/plans/2026-08-05-engine-wiring-baseline.md`

**Interfaces:**
- Consumes: nothing.
- Produces: a recorded `FUEL_BASELINE` commit SHA that every later measurement
  quotes.

- [ ] **Step 1: Record the current state before changing it**

```bash
cd C:/Projects/fuel-lightbulb-port && git status --short && git log --oneline -1
```

Expected: detached at `99dbf231`, plus local modifications to
`fuel-cuda-backend/src/baracuda/attention.rs` (the alpha.78 ABI fix). Note
whether those modifications are still present — Fuel landed an equivalent at
`38d68f86`, so they may now be redundant.

- [ ] **Step 2: Fetch and update to main**

```bash
cd C:/Projects/fuel-lightbulb-port && git fetch origin && git checkout main && git pull --ff-only origin main && git log --oneline -1
```

Expected: a commit at or after `af93e318` (main was `1c640648` on 2026-08-05).
If `git checkout main` fails because local `attention.rs` edits conflict, and
Step 1 showed Fuel already carries the equivalent fix, discard them with
`git checkout -- fuel-cuda-backend/src/baracuda/attention.rs` and retry. If Fuel
does *not* carry it, stash instead and re-apply.

- [ ] **Step 3: Verify Lightbulb still compiles against the updated Fuel**

```bash
cargo check --lib 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: `CARGO EXIT: 0`. If Fuel's API drifted, fix the call sites now —
discovering it here is far cheaper than discovering it inside Task 4.

- [ ] **Step 4: Record the baseline**

Write `docs/superpowers/plans/2026-08-05-engine-wiring-baseline.md` containing
the exact commit SHA from Step 2, the date, and the `cargo check` result. Every
measurement in Task 10 quotes this SHA.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-08-05-engine-wiring-baseline.md
git commit -m "chore(fuel): Pin the Fuel baseline for engine wiring

Updates the fuel-lightbulb-port worktree from detached 99dbf231 to main,
which carries PlanOnce as the paged decode default (af93e318 onward).

Before this, a batch sweep would have measured Replan under a harness
whose unset default is also Replan — a stale number for a configuration
that no longer ships, which is worse than a wrong one because it looks
current."
```

---

## Task 2: Device selection in exactly one place

**Files:**
- Create: `src/model_fuel/device.rs`
- Modify: `src/model_fuel/mod.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `pub fn select() -> fuel::Device`.

Three callers picking a device independently is how a model and its KV cache end
up on different ones. This is the only place that chooses.

- [ ] **Step 1: Write the failing test**

Append to `src/model_fuel/device.rs` (created in Step 3, so write the file with
this test block in place):

```rust
#[cfg(test)]
mod tests {
    /// `select()` returns a usable device and does not panic.
    ///
    /// Deliberately weak on WHICH device — that depends on features and
    /// hardware. The claim under test is that selection is total: there is no
    /// build configuration in which it fails to produce one.
    #[test]
    fn select_returns_a_usable_device() {
        let dev = super::select();
        // Realizing a trivial graph proves the device is not merely
        // constructed but actually drivable.
        let a = fuel::lazy::LazyTensor::from_f32(vec![2.0, 3.0], (1usize, 2usize), &dev);
        let w = a.const_f32_like(vec![1.0, 0.0, 0.0, 1.0], (2usize, 2usize));
        let y = a.matmul(&w).expect("matmul on the selected device");
        assert_eq!(y.realize_f32(), vec![2.0, 3.0]);
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cargo test --lib model_fuel::device 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: FAIL — `cannot find function 'select'` (or "file not found for module
`device`" until Step 3 wires `mod.rs`).

- [ ] **Step 3: Implement**

`src/model_fuel/device.rs`:

```rust
//! Where the Fuel path's device is chosen — and the only place it is.
//!
//! `generate.rs` originally hardcoded `Device::cpu()`. Spreading that choice
//! across the loader, the session and the runner is how a model and its KV
//! cache end up on different devices, which fails deep inside `realize` as a
//! confusing byte-count error rather than at the point of the mistake.

use fuel::Device;

/// The device this build serves on.
///
/// CUDA when `fuel-cuda` is enabled and a device is actually present; CPU
/// otherwise.
///
/// **Falling back to CPU is deliberate.** A server that refuses to start on a
/// CPU-only host is worse than one that starts slow, and the log line names
/// which was chosen so a surprising benchmark has somewhere to look.
///
/// The measurement harness (`tests/gpu_paged_vs_contiguous.rs`) makes the
/// OPPOSITE choice and fails outright, because a benchmark that silently
/// measures CPU is an artifact rather than a result.
pub fn select() -> Device {
    #[cfg(feature = "fuel-cuda")]
    {
        match fuel_cuda_backend::CudaDevice::new(0) {
            Ok(d) => {
                tracing::info!("model_fuel: CUDA device 0 selected");
                return Device::from(d);
            }
            Err(e) => {
                tracing::warn!(
                    "model_fuel: `fuel-cuda` is enabled but CUDA device 0 is unavailable \
                     ({e:?}); falling back to CPU"
                );
            }
        }
    }
    tracing::info!("model_fuel: CPU selected");
    Device::cpu()
}
```

Add to `src/model_fuel/mod.rs`, after the existing `pub mod batched;` line:

```rust
pub mod device;
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cargo test --lib model_fuel::device 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: PASS, 1 test.

- [ ] **Step 5: Commit**

```bash
git add src/model_fuel/device.rs src/model_fuel/mod.rs
git commit -m "feat(model_fuel): Choose the device in exactly one place

generate.rs hardcoded Device::cpu(). Three callers choosing independently
is how a model and its KV cache end up on different devices, which fails
inside realize as a byte-count error rather than at the mistake.

CPU fallback when fuel-cuda is on but no device is present is deliberate:
a server that refuses to start on a CPU-only host is worse than one that
starts slow. The measurement harness makes the opposite choice."
```

---

## Task 3: The loader carries a tokenizer and EOS

Without EOS the runner can only stop on `max_new_tokens` — every request runs to
the limit and emits text past the model's stop. That is a visible correctness
bug, not a performance detail.

**Files:**
- Modify: `src/model_fuel/loader.rs` (the `LoadedLlama` struct at `:39`)
- Modify: `src/model_fuel/loader_f32.rs:52-57` (config parsing)

**Interfaces:**
- Consumes: `model_fuel::device::select()` from Task 2.
- Produces:
  - `LoadedLlama { model: LlamaModel, config: LlamaConfig, tokenizer: tokenizers::Tokenizer, eos: Option<fuel::lazy_llama_full::LlamaEosToks>, device: fuel::Device }`
  - `LoadedLlama::is_eos(&self, tok: u32) -> bool`
  - `load_llama_f32_from_dir(dir: &Path) -> anyhow::Result<LoadedLlama>` (signature unchanged)

- [ ] **Step 1: Write the failing test**

Append to `src/model_fuel/loader_f32.rs`'s existing `mod tests`:

```rust
    /// The loader must surface the checkpoint's EOS token.
    ///
    /// Asserts the CONCRETE value (TinyLlama's `</s>` is 2), not merely that
    /// the field is `Some`. `Some(garbage)` would pass an is-present check and
    /// then never fire during generation, producing exactly the runaway output
    /// this field exists to prevent.
    #[test]
    #[ignore = "needs the TinyLlama checkpoint"]
    fn loader_surfaces_eos_and_tokenizer() -> Result<()> {
        let Some(dir) = tinyllama_dir() else {
            panic!("no TinyLlama snapshot — this test asserts loader behaviour, so it fails rather than skipping");
        };
        let loaded = load_llama_f32_from_dir(&dir)?;

        assert!(loaded.is_eos(2), "TinyLlama's </s> is token 2; EOS did not parse");
        assert!(!loaded.is_eos(0), "token 0 is <unk>, not EOS");

        // The tokenizer must round-trip, not merely exist.
        let ids: Vec<u32> = loaded
            .tokenizer
            .encode("The capital of France is", true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        assert!(!ids.is_empty(), "tokenizer produced no ids");
        Ok(())
    }
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cargo test --lib model_fuel::loader_f32 -- --ignored 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: FAIL to compile — `no method named 'is_eos'` and `no field 'tokenizer'`.

- [ ] **Step 3: Extend `LoadedLlama`**

In `src/model_fuel/loader.rs`, replace the struct at `:39-44` with:

```rust
/// A loaded Llama-shape model, plus everything a serving loop needs to drive it.
///
/// `config` is retained because callers need `vocab_size` to slice logits and
/// `n_layers`/`n_kv_heads`/`head_dim` to size a KV cache.
///
/// `tokenizer`, `eos` and `device` were added when this path was wired into the
/// engine. All three load EAGERLY: a server that accepts a request and only
/// then discovers it cannot detokenize has converted a startup error into a
/// per-request one.
pub struct LoadedLlama {
    pub model: LlamaModel,
    pub config: fuel::lazy::LlamaConfig,
    pub tokenizer: tokenizers::Tokenizer,
    /// `None` when `config.json` carries no `eos_token_id`. Generation then
    /// stops only on `max_new_tokens`, which is why the loader logs a warning.
    pub eos: Option<fuel::lazy_llama_full::LlamaEosToks>,
    /// The device the weights were loaded onto. A `SessionState` built against
    /// a different one is a byte-count error deep inside `realize`.
    pub device: fuel::Device,
}
```

And add to the existing `impl LoadedLlama` block:

```rust
    /// `true` if `tok` ends generation for this checkpoint.
    ///
    /// `false` when the checkpoint declares no EOS — generation then runs to
    /// `max_new_tokens`, which is the honest behaviour for a model that never
    /// declared a stop.
    pub fn is_eos(&self, tok: u32) -> bool {
        self.eos.as_ref().is_some_and(|e| e.is_eos(tok))
    }
```

- [ ] **Step 4: Switch `loader_f32` to the config that carries EOS**

**[verified]** `loader_f32.rs:55` parses via `Llama2cConfig::from_hf_json_str`,
whose `LlamaConfig` has no EOS field. `LlamaFullConfig` parses the same HF
`config.json`, carries `eos_token_id: Option<LlamaEosToks>`, and its
`to_lazy_config()` yields the identical `LlamaConfig` — so nothing downstream
changes.

Replace `src/model_fuel/loader_f32.rs:53-57` with:

```rust
    let config_str = std::fs::read_to_string(dir.join("config.json"))
        .with_context(|| format!("reading config.json in {}", dir.display()))?;
    // `LlamaFullConfig`, NOT `Llama2cConfig`. Both parse this same file into
    // the same `LlamaConfig` via `to_lazy_config()`, but only this one retains
    // `eos_token_id`. Without it the engine can stop only on max_new_tokens.
    let full = fuel::lazy_llama_full::LlamaFullConfig::from_hf_json_str(&config_str)
        .map_err(|e| anyhow::anyhow!("parsing config.json: {e:?}"))?;
    let config: LlamaConfig = full.to_lazy_config();
```

Then, at the end of the function where `LoadedLlama` is constructed, populate
the new fields. Load the tokenizer eagerly and warn on a missing EOS:

```rust
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer.json in {}: {e}", dir.display()))?;
    if full.eos_token_id.is_none() {
        tracing::warn!(
            "config.json in {} declares no eos_token_id; generation will stop \
             only on max_new_tokens",
            dir.display()
        );
    }
    Ok(LoadedLlama {
        model,
        config,
        tokenizer,
        eos: full.eos_token_id,
        device: super::device::select(),
    })
```

Also update `src/model_fuel/loader.rs`'s own `load_llama_from_dir` to populate
the three new fields the same way — it constructs the same struct and will not
compile otherwise.

- [ ] **Step 5: Run the test to verify it passes**

```bash
cargo test --lib model_fuel::loader_f32 -- --ignored 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/model_fuel/loader.rs src/model_fuel/loader_f32.rs
git commit -m "feat(model_fuel): Carry the tokenizer and EOS on LoadedLlama

LoadedLlama was {model, config}. Serving needs a tokenizer and a stop
condition, and had neither: generate.rs's test loaded a tokenizer
separately and passed Some(2) as a literal.

loader_f32.rs:55 parsed config.json through Llama2cConfig, which has no
eos field. LlamaFullConfig parses the SAME file, keeps eos_token_id, and
its to_lazy_config() yields the identical LlamaConfig — so this is a
strictly-more-information swap with no downstream change.

Without EOS the engine can stop only on max_new_tokens: every request
runs to the limit and emits text past the model's stop. That is a
correctness bug, not a performance one.

The test asserts is_eos(2) specifically rather than eos.is_some().
Some(garbage) passes a presence check and then never fires."
```

---

## Task 4: `SessionState` — the persistence seam, made unforgettable

**Files:**
- Create: `src/model_fuel/session.rs`
- Modify: `src/model_fuel/mod.rs`

**Interfaces:**
- Consumes: `LoadedLlama` from Task 3.
- Produces:
  - `pub struct SessionState`
  - `SessionState::new(config: &fuel::lazy::LlamaConfig, max_seq_len: usize, device: &fuel::Device) -> anyhow::Result<Self>`
  - `SessionState::position(&self) -> usize`
  - `SessionState::parts(&mut self) -> (&mut KvCache, &mut InferenceContext)` — `pub(crate)`
  - `SessionState::advance(&mut self, n: usize)` — `pub(crate)`

**REVISED 2026-08-05 after Fuel landed `forward_decode_step`** (`lazy.rs:8675`,
main). The original design held an `Option<DecodeSession>` here so a caller
could not forget to thread it. Fuel has since made the fast path reachable at
the plain `(tokens, cache, ctx)` shape by carrying the held plan on the
`InferenceContext` — so there is no longer a fourth argument to forget, and a
field for it here would be ceremony.

`SessionState` still earns its place: it keeps the KV cache, the context and the
position coherent as one unit, and `new()` is the only place `with_capacity`
(rather than the capture-defeating `with_dims`) is chosen.

- [ ] **Step 1: Write the failing test**

In `src/model_fuel/session.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh session starts at position 0 with an allocated cache.
    ///
    /// Weak by design — the interesting assertions about `SessionState` are
    /// numerical and live in Task 5, which compares logits. This one exists so
    /// `new()` cannot silently start mid-sequence.
    #[test]
    fn new_session_starts_at_position_zero() {
        let cfg = fuel::lazy::LlamaConfig {
            vocab_size: 32,
            dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        };
        let dev = crate::model_fuel::device::select();
        let st = SessionState::new(&cfg, 16, &dev).expect("allocating a tiny KV cache");
        assert_eq!(st.position(), 0);
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cargo test --lib model_fuel::session 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: FAIL — module `session` not found.

- [ ] **Step 3: Implement**

`src/model_fuel/session.rs`:

```rust
//! One sequence's decode state — and the reason it is a type rather than three
//! locals.
//!
//! # Why this exists
//!
//! To keep the three things one sequence needs — KV cache, inference context,
//! position — as a single unit, and to make `with_capacity` the only way a
//! cache gets built here.
//!
//! `with_capacity`, not `with_dims`, is the load-bearing part. `with_dims`
//! grows the cache by REPLACEMENT, which rebuilds the graph every step and
//! defeats both plan reuse and capture.
//!
//! # A hazard this type no longer has to defend against
//!
//! Fuel originally exposed plan reuse only via
//! `forward_with_kv_context_persistent`, whose `&mut Option<DecodeSession>`
//! fourth argument you could not write unless you already knew `DecodeSession`
//! existed. The call a consumer naturally writes was the slow one — measured on
//! CUDA at **5,901 ms/token** against **26.47 ms/token**, though note that
//! comparison also toggled capture, so it bounds rather than measures either.
//!
//! Fuel diagnosed it as a REACHABILITY defect and shipped `forward_decode_step`
//! (`lazy.rs:8675`), which carries the held plan on the `InferenceContext` and
//! takes the plain `(tokens, cache, ctx)` shape. There is no fourth argument to
//! forget, so this type does not need to own one.

use anyhow::Result;

use fuel::inference_context::{InferenceContext, KvCache};
use fuel::lazy::LlamaConfig;
use fuel::{DType, Device};

/// One sequence's KV cache, inference context, and position.
///
/// The decode plan is NOT held here — it rides on the `InferenceContext`, which
/// is what `forward_decode_step` reads and writes.
pub struct SessionState {
    cache: KvCache,
    ctx: InferenceContext,
    position: usize,
}

impl SessionState {
    /// Allocate a session able to hold `max_seq_len` tokens.
    ///
    /// `with_capacity`, NOT `with_dims` — and this is load-bearing. `with_dims`
    /// grows the cache by REPLACEMENT, rebuilding the graph every step and
    /// defeating both plan reuse and capture. `with_capacity` pre-allocates
    /// `[1, n_kv_heads, max_seq_len, head_dim]` per layer and appends via
    /// `Op::WriteSlice` at a runtime offset, giving a graph that is stable
    /// across steps.
    ///
    /// Fuel enforces this: the persistent forward rejects a `with_dims` cache
    /// outright rather than silently taking the slow path.
    ///
    /// Memory is real and allocated up front:
    /// `n_layers * 2 * n_kv_heads * max_seq_len * head_dim * sizeof(dtype)`.
    /// TinyLlama at 2048 f32 is ~92 MiB.
    pub fn new(config: &LlamaConfig, max_seq_len: usize, device: &Device) -> Result<Self> {
        let cache = KvCache::with_capacity(
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            max_seq_len,
            DType::F32,
            device,
        )
        .map_err(|e| anyhow::anyhow!("allocating KV cache: {e:?}"))?;
        Ok(Self {
            cache,
            ctx: InferenceContext::new(device.clone()),
            position: 0,
        })
    }

    /// Tokens committed to the cache so far.
    pub fn position(&self) -> usize {
        self.position
    }

    /// The two pieces a forward needs, borrowed together.
    ///
    /// `pub(crate)` and returned as a pair so `decoder.rs` can drive a forward
    /// without the fields becoming public.
    pub(crate) fn parts(&mut self) -> (&mut KvCache, &mut InferenceContext) {
        (&mut self.cache, &mut self.ctx)
    }

    /// Record that `n` tokens were committed.
    pub(crate) fn advance(&mut self, n: usize) {
        self.position += n;
    }
}
```

Add to `src/model_fuel/mod.rs`:

```rust
pub mod session;
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cargo test --lib model_fuel::session 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model_fuel/session.rs src/model_fuel/mod.rs
git commit -m "feat(model_fuel): Group one sequence's decode state

SessionState keeps KV cache, inference context and position as one unit,
and makes with_capacity the only way a cache is built here — with_dims
grows by replacement, rebuilding the graph every step and defeating both
plan reuse and capture.

It deliberately does NOT hold an Option<DecodeSession>. Fuel originally
exposed plan reuse only through a fourth argument you could not write
unless you knew DecodeSession existed, so the natural call was the slow
one. Fuel diagnosed that as a reachability defect and shipped
forward_decode_step, which carries the plan on the InferenceContext at
the plain (tokens, cache, ctx) shape. Nothing left to forget."
```

---

## Task 5: `FuelDecoder` — the seam a quantized model drops into

**Files:**
- Create: `src/model_fuel/decoder.rs`
- Modify: `src/model_fuel/mod.rs`

**Interfaces:**
- Consumes: `SessionState::{parts, advance}` from Task 4.
- Produces:
  - `pub trait FuelDecoder { fn prefill(&self, tokens: &[u32], st: &mut SessionState) -> Result<Vec<f32>>; fn step(&self, token: u32, st: &mut SessionState) -> Result<Vec<f32>>; }`
  - `impl FuelDecoder for fuel::lazy::LlamaModel`

- [ ] **Step 1: Write the failing test**

The claim under test is **numerical**, so it is asserted on logits with
`maxdiff == 0`, never on tokens. Argmax is a discretization that hides drift: a
regression must move a logit across the top-2 gap before one token changes, so a
token-comparison test passes whether or not the maths changed. Fuel measured a
~1.7e-3 perturbation flipping neither greedy nor seeded-temperature tokens.

In `src/model_fuel/decoder.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tinyllama_dir() -> Option<PathBuf> {
        let p = PathBuf::from(
            "C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        );
        p.join("model.safetensors").is_file().then_some(p)
    }

    /// Stepping through `FuelDecoder` produces bit-identical logits to the
    /// original `generate_greedy` loop.
    ///
    /// **Asserted on logits with maxdiff == 0, deliberately not on tokens.**
    /// This test's sole purpose is detecting numerical drift, and argmax is the
    /// operation that hides it — a token comparison would pass whether or not
    /// the decode maths changed, which is evidence carrying no information.
    #[test]
    #[ignore = "needs the TinyLlama checkpoint; slow on CPU"]
    fn stepwise_logits_match_the_reference_loop_exactly() -> anyhow::Result<()> {
        let Some(dir) = tinyllama_dir() else {
            panic!("no TinyLlama snapshot — this test asserts numerical behaviour, so it fails rather than skipping");
        };
        let loaded = crate::model_fuel::loader_f32::load_llama_f32_from_dir(&dir)?;
        let prompt: Vec<u32> = vec![1, 450, 7483, 310, 3444, 338];

        // Reference: the shape generate_greedy uses — one persistent session
        // across prefill and every decode step.
        let mut ref_st =
            SessionState::new(&loaded.config, prompt.len() + 4, &loaded.device)?;
        let mut reference = loaded.model.prefill(&prompt, &mut ref_st)?;
        for _ in 0..3 {
            let next = crate::model_fuel::generate::argmax(&reference);
            reference = loaded.model.step(next, &mut ref_st)?;
        }

        // Under test: an independently constructed session doing the same.
        let mut st = SessionState::new(&loaded.config, prompt.len() + 4, &loaded.device)?;
        let mut got = loaded.model.prefill(&prompt, &mut st)?;
        for _ in 0..3 {
            let next = crate::model_fuel::generate::argmax(&got);
            got = loaded.model.step(next, &mut st)?;
        }

        assert_eq!(got.len(), reference.len(), "logit row width changed");
        let maxdiff = got
            .iter()
            .zip(reference.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            maxdiff, 0.0,
            "logits drifted by {maxdiff:e} — decode is not reproducible step-for-step"
        );
        assert_eq!(st.position(), ref_st.position(), "position accounting diverged");
        Ok(())
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cargo test --lib model_fuel::decoder 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: FAIL — module `decoder` not found.

- [ ] **Step 3: Implement**

`src/model_fuel/decoder.rs`:

```rust
//! Which Fuel model type decodes — abstracted, so a second one can arrive.
//!
//! # Why a trait for a single implementor
//!
//! Because the second implementor exists and is next.
//!
//! GGUF/quantized Llama is `QuantizedLlama3Model::from_gguf`, which wraps
//! `Llama3Model`. When this plan was written `Llama3Model` had no `KvCache` and
//! no `forward_with_kv_context*` at all, so serving through it would have
//! re-run the whole prefix per token. Lightbulb filed that gap; Fuel landed
//! `Llama3Model::forward_with_kv_context_persistent`
//! (`lazy_llama_full.rs:382`) the same day, and the quantized wrapper inherits
//! it through its `inner`.
//!
//! So quantized support is now `impl FuelDecoder for Llama3Model` and nothing
//! else — no change to the session, the engine model, or the runner. It is
//! deliberately NOT in this plan: the runner should be proven end-to-end on one
//! model path before a second one is debugged alongside it.
//!
//! **Two things to know when that impl is written.** Fuel's paged tier does not
//! yet thread the LLaMA-3.1 frequency override, so `PagedDecodeModel` is a
//! separate supertrait and handing a `Llama3Model` to `PagedSessionScheduler` is
//! a compile error rather than a silent wrong answer. And a scaled-vs-unscaled
//! parity test on a short prompt proves nothing: the RoPE band edges sit at
//! `original_max_position_embeddings / *_freq_factor` = 8192, and at positions
//! 0..6 the two differ by 1.1e-7.

use anyhow::Result;

use fuel::lazy::LlamaModel;

use super::session::SessionState;

/// A model that can prefill a prompt and then decode one token at a time
/// against a held [`SessionState`].
pub trait FuelDecoder {
    /// Consume the whole prompt in one forward. Returns the final logit row.
    fn prefill(&self, tokens: &[u32], st: &mut SessionState) -> Result<Vec<f32>>;

    /// Consume one token. Returns the resulting logit row.
    fn step(&self, token: u32, st: &mut SessionState) -> Result<Vec<f32>>;
}

impl FuelDecoder for LlamaModel {
    fn prefill(&self, tokens: &[u32], st: &mut SessionState) -> Result<Vec<f32>> {
        // `forward_decode_step` for BOTH prefill and decode, deliberately.
        // At `seq != 1` it falls back to the rebuild path without building a
        // plan; the first `seq == 1` token builds it; later tokens rebind and
        // skip optimize. Output is byte-identical either way, so there is no
        // reason for the two arms to call different entry points — and one
        // entry point is one fewer place to get the fast path wrong.
        let (cache, ctx) = st.parts();
        let logits = self
            .forward_decode_step(tokens, cache, ctx)
            .map_err(|e| anyhow::anyhow!("prefill forward: {e:?}"))?;
        st.advance(tokens.len());
        Ok(logits)
    }

    fn step(&self, token: u32, st: &mut SessionState) -> Result<Vec<f32>> {
        let (cache, ctx) = st.parts();
        let logits = self
            .forward_decode_step(&[token], cache, ctx)
            .map_err(|e| anyhow::anyhow!("decode forward: {e:?}"))?;
        st.advance(1);
        Ok(logits)
    }
}
```

Add to `src/model_fuel/mod.rs`:

```rust
pub mod decoder;
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cargo test --release --lib model_fuel::decoder -- --ignored --nocapture 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: PASS. **Run in release** — debug is ~10× slower on numeric code.

- [ ] **Step 5: Commit**

```bash
git add src/model_fuel/decoder.rs src/model_fuel/mod.rs
git commit -m "feat(model_fuel): Add the FuelDecoder seam for a second model type

A trait with one implementor, because the second is already specified and
blocked: QuantizedLlama3Model wraps Llama3Model, which has no KvCache and
no forward_with_kv_context* at all. An ask is filed with Fuel; when it
lands, quantized support is one added impl and nothing else.

The parity test asserts LOGITS with maxdiff == 0, not tokens. Its sole
purpose is detecting numerical drift and argmax is what hides drift — a
token comparison passes whether or not the maths changed."
```

---

## Task 6: `EngineModel` — the tensor-free seam, plus the candlelight impl

Pure refactor: no behaviour changes. Doing it separately from Task 7 means a
reviewer can reject the abstraction without rejecting the Fuel path.

**Files:**
- Modify: `src/engine/model_runner.rs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub(crate) trait EngineModel { fn step_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>>; fn decode_text(&self, tokens: &[u32], skip_special: bool) -> Result<String>; }`
  - `fn run_jobs<M: EngineModel>(model: M, rx: Receiver<InferenceJob>)`

- [ ] **Step 1: Add the trait and the candlelight impl**

Insert into `src/engine/model_runner.rs`, above `pub struct ModelRunner`:

```rust
/// What the job loop needs from a model, and nothing more.
///
/// **Deliberately tensor-free.** No Fuel or candlelight type appears here, which
/// is what keeps `engine/` free of the tensor layer — the property the port
/// design names as the actual differentiator worth protecting.
///
/// Already batch-shaped, so adding real batching changes the job-draining loop
/// and the implementations, not this trait.
pub(crate) trait EngineModel {
    /// Advance every request in `batch` by at most one token.
    ///
    /// Returns one entry per request, **positionally aligned with `batch`**.
    ///
    /// `Some(tok)` means that request produced `tok` this step. **`None` means
    /// it produced no token and that is not an error** — it is mid-chunked-
    /// prefill or already stopped. Errors are the `Err` arm and abort the whole
    /// batch. An implementation returning `Err` where `None` is meant turns
    /// ordinary prefill progress into a failed request.
    fn step_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>>;

    /// Detokenize.
    fn decode_text(&self, tokens: &[u32], skip_special: bool) -> Result<String>;
}

impl EngineModel for crate::model::ParallelModelManager {
    fn step_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>> {
        self.forward_batch(batch)
    }

    fn decode_text(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        self.decode(tokens, skip_special)
    }
}
```

- [ ] **Step 2: Extract the job loop, unchanged**

Move the body of the `Ok(mut manager) => { ... }` arm (currently
`model_runner.rs:93-211`) into a free generic function, replacing every
`manager.forward_batch(&mut batch)` with `model.step_batch(&mut batch)` and every
`manager.decode(...)` with `model.decode_text(...)`. **Change nothing else** —
not the tool-call degradation, not the ordering, not the error fanout.

```rust
/// The job loop. Identical for every backend, which is why it is generic rather
/// than duplicated per `cfg`.
fn run_jobs<M: EngineModel>(mut model: M, rx: Receiver<InferenceJob>) {
    while let Ok(job) = rx.recv() {
        // ... body moved verbatim from the Ok(mut manager) arm ...
    }
    println!("Model runner thread exiting (receiver closed)");
}
```

Generic, not `Box<dyn>`, so there is no dynamic dispatch in the decode path.

- [ ] **Step 3: Verify nothing changed**

```bash
cargo check --lib 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: `CARGO EXIT: 0`.

```bash
cargo test --lib engine:: 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: same pass/fail counts as before this task. Record both numbers — a
pure refactor that changes a test count did not refactor purely.

- [ ] **Step 4: Commit**

```bash
git add src/engine/model_runner.rs
git commit -m "refactor(engine): Extract a tensor-free EngineModel seam

Pure refactor, no behaviour change: the job loop becomes generic over a
two-method trait, and ParallelModelManager implements it by forwarding to
methods it already has. src/model/ is untouched.

The trait carries no tensor types, which is what keeps engine/ free of
the tensor layer. It is already batch-shaped, so real batching later
changes the job-draining loop and the impls rather than this trait.

Generic rather than Box<dyn> — no dynamic dispatch in the decode path.

Documents step_batch's return contract, which the type does not convey:
None means 'no token this step' (mid-chunked-prefill at
parallel_model_manager.rs:300, or stopped at :673), NOT an error. An
implementation returning Err there turns prefill progress into a failed
request."
```

---

## Task 7: `FuelEngineModel` — sampling, EOS, and request mapping

**Files:**
- Create: `src/model_fuel/engine_model.rs`
- Modify: `src/model_fuel/mod.rs`

**Interfaces:**
- Consumes: `LoadedLlama` (Task 3), `SessionState` (Task 4), `FuelDecoder` (Task 5), `EngineModel` (Task 6).
- Produces:
  - `pub struct FuelEngineModel`
  - `FuelEngineModel::load(model_dir: &Path, context_length: usize) -> anyhow::Result<Self>`
  - `impl crate::engine::model_runner::EngineModel for FuelEngineModel`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Greedy selection is deterministic; sampled selection at the same seed is
    /// reproducible and at different seeds is not.
    ///
    /// The third assertion is the one that can fail informatively: if
    /// `select_token` ignored the seed, the first two would still pass.
    #[test]
    fn select_token_is_greedy_at_zero_and_seeded_above() {
        let logits = vec![0.1, 3.0, 0.2, 2.9, 0.3];
        assert_eq!(select_token(&logits, 0.0, 1), 1, "temperature 0 must be argmax");
        assert_eq!(select_token(&logits, 0.0, 999), 1, "greedy must ignore the seed");
        assert_eq!(
            select_token(&logits, 1.0, 7),
            select_token(&logits, 1.0, 7),
            "same seed must reproduce"
        );
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cargo test --lib model_fuel::engine_model 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: FAIL — module `engine_model` not found.

- [ ] **Step 3: Implement**

`src/model_fuel/engine_model.rs`:

```rust
//! The Fuel path, wearing the engine's interface.
//!
//! Everything above the realize boundary is Lightbulb's: which token to pick,
//! when to stop, whose request runs. Fuel supplies the forward pass, the KV
//! cache and the graph. That split is the one the port is built on.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

use crate::engine::model_runner::EngineModel;
use crate::engine::{RequestContext, RequestState};

use super::decoder::FuelDecoder;
use super::loader::LoadedLlama;
use super::session::SessionState;

/// Pick a token from a realized logit row.
///
/// `temperature <= 0.0` is greedy, routed through the EXISTING public
/// `generate::argmax` rather than a copy — a second implementation of a
/// tie-break is a second thing to keep in sync, and the golden pins that one.
fn select_token(logits: &[f32], temperature: f64, seed: u64) -> u32 {
    if temperature <= 0.0 {
        return super::generate::argmax(logits);
    }
    let mut l = logits.to_vec();
    crate::sampling::apply_temperature(&mut l, temperature as f32);
    crate::sampling::sample_from_logits(&l, seed) as u32
}

/// A loaded Fuel model plus one decode session per in-flight request.
pub struct FuelEngineModel {
    loaded: LoadedLlama,
    /// Keyed by `RequestContext.request.id`. Entries are created on the
    /// request's first step and dropped when it completes — a session holds a
    /// pre-allocated KV cache (~92 MiB for TinyLlama at 2048 f32), so leaking
    /// them leaks real memory.
    ///
    /// # Do not "optimise" this into a slot pool that reuses contexts
    ///
    /// One `SessionState` per REQUEST — each with its own `InferenceContext`
    /// and its own `KvCache` — is correct and confirmed correct by Fuel
    /// (2026-08-05). The obvious next optimisation is a fixed pool of slots
    /// that reuses a context across successive requests to avoid rebuilding
    /// the decode plan per request. **That is unsafe as Fuel stands**, and it
    /// fails silently:
    ///
    /// - The held plan's KV Arcs are bound ONCE at build time and mutate in
    ///   place via `Op::WriteSlice`. `rebind_and_realize_prebuilt` rebinds
    ///   `token_ids`/`rope_cos`/`rope_sin`/`mask`/offset and **not** the KV
    ///   nodes — so a plan is welded to the exact `KvCache` it was built on.
    /// - `DecodeSession::is_valid_for`'s key is `(seq, max_seq_len, n_layers,
    ///   cache_dtype)` — pure geometry. It cannot distinguish two `KvCache`
    ///   instances of the same shape, which in a slot pool is the normal case.
    ///
    /// Retire request A from a slot, admit B with a fresh same-shaped cache,
    /// reuse the slot's context: the key matches, the plan is reused, and B
    /// decodes over A's KV buffers. No error, plausible tokens, one tenant
    /// reading another's context.
    ///
    /// If plan reuse across requests is wanted later, the safe shape is a slot
    /// owning a PERSISTENT `KvCache` and reusing **cache and context together**,
    /// resetting the cache between requests rather than allocating a new one.
    /// Tell Fuel first — they offered to add cache identity to the validity key
    /// so the unsafe version fails loudly instead of silently.
    sessions: HashMap<String, SessionState>,
    context_length: usize,
    /// Per-request sampling seed source. Incremented per selection so two
    /// tokens in one request do not share a seed.
    seed_counter: u64,
}

impl FuelEngineModel {
    /// Load a SafeTensors checkpoint directory.
    ///
    /// Fails here rather than at first request if the tokenizer or config is
    /// missing — see `LoadedLlama`.
    pub fn load(model_dir: &Path, context_length: usize) -> Result<Self> {
        let loaded = super::loader_f32::load_llama_f32_from_dir(model_dir)?;
        Ok(Self {
            loaded,
            sessions: HashMap::new(),
            context_length,
            seed_counter: 0,
        })
    }

    /// Advance one request by at most one token.
    fn step_one(&mut self, ctx: &mut RequestContext) -> Result<Option<u32>> {
        match ctx.state {
            RequestState::Completed | RequestState::AwaitingToolResult { .. } => Ok(None),
            RequestState::Pending => {
                let ids: Vec<u32> = self
                    .loaded
                    .tokenizer
                    .encode(ctx.request.prompt.as_str(), true)
                    .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?
                    .get_ids()
                    .to_vec();

                // Cache capacity is fixed at allocation, so it must cover
                // prompt + generation. Capped at context_length because the KV
                // cache is pre-allocated and an oversized request would
                // otherwise allocate unboundedly.
                let want = ids.len() + ctx.request.max_new_tokens + 1;
                let max_seq_len = want.min(self.context_length);
                if ids.len() >= max_seq_len {
                    anyhow::bail!(
                        "prompt of {} tokens does not fit in a context of {}",
                        ids.len(),
                        max_seq_len
                    );
                }

                let mut st =
                    SessionState::new(&self.loaded.config, max_seq_len, &self.loaded.device)?;
                let logits = self.loaded.model.prefill(&ids, &mut st)?;
                self.sessions.insert(ctx.request.id.clone(), st);

                self.seed_counter += 1;
                let tok = select_token(&logits, ctx.request_temperature(), self.seed_counter);
                ctx.generated_tokens.push(tok);
                ctx.start_decoding();
                ctx.record_token();
                if self.loaded.is_eos(tok) {
                    ctx.complete();
                    self.sessions.remove(&ctx.request.id);
                }
                Ok(Some(tok))
            }
            RequestState::Decoding => {
                let last = *ctx
                    .generated_tokens
                    .last()
                    .ok_or_else(|| anyhow::anyhow!("decoding with no previous token"))?;
                let st = self
                    .sessions
                    .get_mut(&ctx.request.id)
                    .ok_or_else(|| anyhow::anyhow!("no session for request {}", ctx.request.id))?;
                let logits = self.loaded.model.step(last, st)?;

                self.seed_counter += 1;
                let tok = select_token(&logits, ctx.request_temperature(), self.seed_counter);
                ctx.generated_tokens.push(tok);
                ctx.record_token();
                if self.loaded.is_eos(tok) {
                    ctx.complete();
                    self.sessions.remove(&ctx.request.id);
                }
                Ok(Some(tok))
            }
        }
    }
}

impl EngineModel for FuelEngineModel {
    fn step_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>> {
        let mut out = Vec::with_capacity(batch.len());
        for ctx in batch.iter_mut() {
            out.push(self.step_one(ctx)?);
        }
        Ok(out)
    }

    fn decode_text(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        self.loaded
            .tokenizer
            .decode(tokens, skip_special)
            .map_err(|e| anyhow::anyhow!("detokenizing: {e}"))
    }
}
```

**`RequestContext` carries no temperature** — `InferenceJob` does, and the job
loop discards it when building the context. Add to `src/engine/mod.rs`'s
`RequestContext`:

```rust
    /// Sampling temperature carried from the originating `InferenceJob`.
    /// `0.0` means greedy. Defaults to `0.0` so existing constructors are
    /// unchanged and deterministic.
    pub temperature: f64,
```

initialize it to `0.0` in `RequestContext::new`, add the accessor:

```rust
    pub fn request_temperature(&self) -> f64 {
        self.temperature
    }
```

and set it in `run_jobs` where the context is built:
`ctx.temperature = job.temperature;`

- [ ] **Step 4: Run the test to verify it passes**

```bash
cargo test --lib model_fuel::engine_model 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/model_fuel/engine_model.rs src/model_fuel/mod.rs src/engine/mod.rs src/engine/model_runner.rs
git commit -m "feat(model_fuel): Implement EngineModel over the Fuel path

Maps RequestContext state onto a per-request SessionState: Pending
tokenizes and prefills, Decoding steps one token, Completed yields None.
EOS completes the request and drops its session — a session holds a
pre-allocated KV cache (~92 MiB for TinyLlama at 2048 f32), so leaking
them leaks real memory.

Also fixes a live bug: InferenceJob carries a temperature that nothing
read, so every request was greedy regardless of what the client asked.
RequestContext now carries it, defaulting to 0.0 so existing constructors
stay deterministic.

Greedy routes through the existing public generate::argmax rather than a
copy — a second tie-break implementation is a second thing to keep in
sync, and the golden pins that one."
```

---

## Task 8: The `fuel-engine` feature swap

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/engine/model_runner.rs`

**Interfaces:**
- Consumes: `FuelEngineModel::load` (Task 7), `run_jobs` (Task 6).
- Produces: `ModelRunner::start` behaviour under `--features fuel-engine`.
  **Signature unchanged.**

- [ ] **Step 1: Add the feature**

In `Cargo.toml`'s `[features]`:

```toml
    # Serve through the Fuel path (`src/model_fuel/`) instead of candlelight.
    #
    # A BUILD-TIME choice rather than a runtime one, so one binary serves one
    # backend. Accepted cost: a candlelight/Fuel A/B needs two builds and two
    # processes. Acceptable because the acceptance gate asserts content over
    # HTTP rather than comparing the two runners — three of the four suites the
    # port design named as the parity oracle no longer compile.
    #
    # `--all-features` selects Fuel. That build is already broken for an
    # unrelated reason (`cuda` needs `candlelight/cuda`, whose candle-kernels
    # v0.10.2 build script fails under CUDA 13.3), so no working configuration
    # changes meaning.
    fuel-engine = []
```

- [ ] **Step 2: Split `start` on the feature**

In `src/engine/model_runner.rs`, replace the single `pub fn start` with two
`cfg`-gated definitions. The candlelight one keeps the existing body verbatim,
ending in `run_jobs(manager, rx)`. The Fuel one:

```rust
    /// Start the model runner thread, serving through the Fuel path.
    ///
    /// `dtype` is accepted and IGNORED: this path loads all weights as f32 by
    /// construction. Fuel's CPU backend has no `[F32, BF16, F32]` matmul
    /// kernel, and while the optimizer will insert a promoting cast, that cast
    /// is value-lossless but NOT accumulation-preserving. Silently honouring a
    /// "bf16" request would change numerics against every golden captured on
    /// the f32 path. The parameter stays for signature compatibility.
    #[cfg(feature = "fuel-engine")]
    pub fn start(
        model_path: impl Into<PathBuf>,
        _max_batch_size: usize,
        context_length: usize,
        dtype: Option<String>,
    ) -> Result<InferenceRequestSender> {
        let model_path = model_path.into();
        let (tx, rx): (Sender<InferenceJob>, Receiver<InferenceJob>) = std::sync::mpsc::channel();

        if let Some(d) = dtype.as_deref()
            && d != "f32"
        {
            tracing::warn!(
                "fuel-engine ignores dtype={d:?} and loads f32; see the module docs on \
                 accumulation-preserving casts"
            );
        }

        std::thread::spawn(move || {
            println!("Loading Fuel model from {}", model_path.display());
            match crate::model_fuel::engine_model::FuelEngineModel::load(
                &model_path,
                context_length,
            ) {
                Ok(model) => {
                    println!("Fuel model loaded at {}", model_path.display());
                    run_jobs(model, rx);
                }
                Err(e) => {
                    eprintln!("Failed to load Fuel model at {}: {:#}", model_path.display(), e);
                    drain_with_error(rx, &format!("model load failed: {e}"));
                }
            }
        });

        Ok(tx)
    }
```

Extract the existing load-failure drain into a shared helper so both paths use
it rather than duplicating the `ResponseMode` match:

```rust
/// Answer every queued job with an error, then exit.
///
/// Shared by both backends: a load failure must not leave clients blocked on a
/// channel that will never produce.
fn drain_with_error(rx: Receiver<InferenceJob>, msg: &str) {
    while let Ok(job) = rx.recv() {
        match job.response_mode {
            ResponseMode::Complete(resp_tx) => {
                let _ = resp_tx.send(Err(anyhow::anyhow!("{}", msg)));
            }
            ResponseMode::Streaming(stream_tx) => {
                let _ = stream_tx.send(Err(anyhow::anyhow!("{}", msg)));
            }
        }
    }
}
```

- [ ] **Step 3: Verify both configurations compile**

```bash
cargo check --lib 2>&1 | tail -10; echo "CANDLELIGHT EXIT: ${PIPESTATUS[0]}"
```

Expected: `CANDLELIGHT EXIT: 0`.

```bash
cargo check --lib --features fuel-engine 2>&1 | tail -10; echo "FUEL EXIT: ${PIPESTATUS[0]}"
```

Expected: `FUEL EXIT: 0`. Run these **sequentially** — one cargo invocation at a
time.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml src/engine/model_runner.rs
git commit -m "feat(engine): Add the fuel-engine feature swap

ModelRunner::start gains a cfg-gated Fuel body. api/mod.rs is untouched
and the signature is unchanged, so the swap is invisible to callers.

dtype is accepted and ignored on the Fuel path, with a warning. Fuel's
CPU backend has no [F32, BF16, F32] matmul kernel; the optimizer inserts
a promoting cast that is value-lossless but NOT accumulation-preserving,
so honouring a bf16 request would change numerics against every golden
captured on the f32 path.

The load-failure drain is now shared rather than duplicated per backend —
a load failure must not leave clients blocked on a channel that will
never produce."
```

---

## Task 9: The acceptance gate — real HTTP, asserted content

**Files:**
- Create: `tests/fuel_engine_http.rs`

**Interfaces:**
- Consumes: everything above.
- Produces: the gate that says sub-project 1 is done.

- [ ] **Step 1: Write the failing test**

```rust
//! The acceptance gate for the Fuel-backed runner.
//!
//! Asserts the completion is COHERENT ENGLISH CONTINUING THE PROMPT, not merely
//! that a 200 came back. A model wired up wrongly — transposed projection,
//! mis-set RoPE base, wrong norm placement — still returns tokens; it just
//! returns nonsense. Status code, token count, and absence of panic all pass on
//! garbage. Only reading the text catches it.
//!
//! Goes through the REAL router and the REAL handler via `ServiceExt::oneshot`,
//! so routing, JSON deserialization, the channel and the runner are all
//! exercised. Only the TCP socket is skipped, and the socket is not what is at
//! risk.
//!
//! Run: `cargo test --release --features fuel-engine --test fuel_engine_http -- --ignored --nocapture`
#![cfg(feature = "fuel-engine")]

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use lightbulb::api::{ApiConfig, AppState};
use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler, ModelRunner};

fn tinyllama_dir() -> Option<PathBuf> {
    let p = PathBuf::from(
        "C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    );
    p.join("model.safetensors").is_file().then_some(p)
}

#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint; minutes on CPU"]
async fn fuel_runner_serves_a_coherent_completion_over_http() {
    let dir = tinyllama_dir()
        .expect("no TinyLlama snapshot — this is an acceptance gate, so it fails rather than skipping");

    let tx = ModelRunner::start(&dir, 1, 512, Some("f32".to_string()))
        .expect("starting the Fuel model runner");

    let state = AppState {
        scheduler: Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default())),
        config: ApiConfig::default(),
        db_pool: None,
        inference_tx: Some(tx),
    };

    let app = lightbulb::api::openai::routes().with_state(state);

    // temperature 0.0 PINNED, not defaulted: this gate asserts content, so a
    // future default that makes sampling stochastic would make it flaky.
    let body = serde_json::json!({
        "model": "tinyllama",
        "prompt": "The capital of France is",
        "max_tokens": 8,
        "temperature": 0.0,
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .expect("router returned no response");

    assert_eq!(resp.status(), StatusCode::OK, "completion request failed");

    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("response was not JSON");
    let text = v["choices"][0]["text"]
        .as_str()
        .expect("no choices[0].text in the response")
        .to_string();

    eprintln!("completion: {text:?}");

    assert!(!text.trim().is_empty(), "the model returned no text");
    assert!(
        text.to_lowercase().contains("paris"),
        "expected the continuation to name Paris, got {text:?} — the server \
         responds but the model is producing nonsense, which points at the \
         wiring (projection layout, RoPE base, norm placement) rather than the \
         plumbing"
    );
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cargo test --release --features fuel-engine --test fuel_engine_http -- --ignored --nocapture 2>&1 | tail -30; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: FAIL. Most likely a compile error first — `AppState`'s field set or
`MemoryAwareScheduler::new`'s signature may differ. Fix the test against the real
signatures; do not change `api/` to suit the test.

- [ ] **Step 3: Make it pass**

There is no new production code in this task by design — if Tasks 1–8 are
correct, this passes. If it fails at runtime, the failure is the finding.
Diagnose in this order, because it separates plumbing from maths:

1. **Non-200** — the handler or the channel. Plumbing.
2. **200, empty text** — `should_continue()`/state transitions in `step_one`.
3. **200, fluent but wrong content** — the model wiring. Compare against
   `model_fuel::generate::generate_greedy` on the same prompt; if that also
   produces nonsense the defect predates this plan.
4. **200, correct content but never stops** — EOS. Verify `is_eos` against
   Task 3's test.

- [ ] **Step 4: Confirm the candlelight path still builds**

```bash
cargo check --lib --tests 2>&1 | tail -10; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: `CARGO EXIT: 0`. `#![cfg(feature = "fuel-engine")]` must exclude the
whole file, so this test must not break a default build.

- [ ] **Step 5: Commit**

```bash
git add tests/fuel_engine_http.rs
git commit -m "test(engine): Gate the Fuel runner on a real HTTP completion

Asserts the completion is coherent English continuing the prompt, not
that a 200 came back. A model wired up wrongly — transposed projection,
mis-set RoPE base, wrong norm placement — still returns tokens; it just
returns nonsense, and status code, token count and absence of panic all
pass on garbage.

Goes through the real router and handler via ServiceExt::oneshot, so
routing, JSON, the channel and the runner are exercised. Only the TCP
socket is skipped and the socket is not what is at risk.

Fails rather than skips on a missing checkpoint: an early return from a
#[test] is a PASS, and a gate that reports success having verified
nothing is the artifact this project exists to stop producing.

temperature is pinned to 0.0 in the request rather than defaulted, so a
future sampling default cannot make a content assertion flaky."
```

---

## Phase B

## Task 10: The batch-size sweep

Produces the decision sub-project 2's implementation needs. **No code dependency
on Tasks 2–9** — it extends an existing harness against Fuel directly, and needs
only Task 1's worktree update.

**Files:**
- Modify: `tests/gpu_paged_vs_contiguous.rs`

**Interfaces:**
- Consumes: Task 1's `FUEL_BASELINE` SHA.
- Produces: a committed measurement table; the paged-vs-contiguous decision.

- [ ] **Step 1: Make the unset environment variable an error**

Currently `""` maps to `Replan`. Since Fuel flipped the driver default to
`PlanOnce` (`af93e318`), a bare run now measures the **opposite** of the shipped
configuration while looking current. Replace the match arm at `:228-232`:

```rust
    let plan = match std::env::var("LB_PLAN").unwrap_or_default().as_str() {
        "once" => PagedDecodePlan::PlanOnce,
        "replan" => PagedDecodePlan::Replan,
        // NOT a default. Fuel's driver default is now PlanOnce, so mapping
        // unset to Replan would silently measure a configuration that no
        // longer ships — and a stale number that looks current is worse than
        // a wrong one.
        "" => panic!("LB_PLAN must be set explicitly to \"once\" or \"replan\"; an arm that does not name its mode is not a control"),
        other => panic!("LB_PLAN must be \"once\" or \"replan\"; got {other:?}"),
    };
```

- [ ] **Step 2: Add the k-sweep with per-window timing**

Parameterize both arms over `k` from `LB_BATCH` (default `1`), looping k
sessions per step. Record **prefill and steady-state decode separately**:

**Three windows, not two, and the boundary placement is load-bearing.**

**[verified by Fuel 2026-08-05, in code]** The decode plan is built on the
**first decode token**, on both routes — never during prefill:

- Paged: `fuel-inference/src/multi_session.rs:1278` prefill calls
  `forward_paged_step`, the non-persistent re-planning entry, and builds no
  session. `:1320` decode calls `forward_paged_step_persistent`, which is where
  the plan is built.
- Contiguous: `forward_with_kv_context_persistent` given the whole prompt sees
  `seq != 1`, drops any session and falls back to the rebuild path *without*
  building one. The build lands on the first decode token here too.

So the first decode token carries a one-time plan-build cost that no later token
pays. Whether it sits inside or outside the steady-state window changes what the
`PlanOnce` arm's number means — and that is invisible in a two-window split.

```rust
// Three windows. `first_decode_ms` is separated because that is where the
// plan is built (Fuel, verified in code 2026-08-05) — folding it into the
// steady-state mean charges PlanOnce a one-time cost as if it recurred, and
// at low token counts that is enough to move the ratio.
let prefill_ms: f64 = /* wall time of the prefill call */;
let first_decode_ms: f64 = /* wall time of decode token 0 alone */;
let steady_ms_per_token: f64 = /* wall time of decode tokens 1..n, divided by (n-1) */;
```

Report all three at every `k`. State explicitly in the results table which side
of the boundary token 0 falls on.

**Instrument validity, since this is what the disagreement turned out to be
about.** A *within-arm* warm-up-vs-steady ratio is an invalid instrument for
plan reuse — it reads ~1.0 regardless. A *cross-arm* steady-state comparison
(Replan vs PlanOnce) is the valid one, and is what produced the 29.7× in
`0e3fc36`. This sweep does the latter. Do not substitute the former.

- [ ] **Step 3: Add the capture-isolation arm**

**This corrects a confound in Lightbulb's own headline number.** The
5,901 → 26.47 ms/token comparison in `0e3fc36` toggled persistence **and**
capture together: `tests/gpu_paged_vs_contiguous.rs:301` calls
`forward_with_kv_context_captured` on the fast arm and `forward_with_kv_context`
on the slow one. So 223× is an upper bound on each and a measurement of neither.

Fuel is holding a decision on it — CUDA-graph capture
(`forward_with_kv_context_captured`) is called from tests only, no production
route captures, and they have declined to default it on a number that cannot
isolate it.

Add a third contiguous configuration, holding persistence constant:

| Arm | Entry point |
| --- | --- |
| persistent, no capture | `forward_decode_step` |
| persistent, with capture | `forward_with_kv_context_captured` |

Same tokens, same model, same commit. Report ms/token and DtoH bytes/token. The
delta is capture's actual contribution. **k=1 alone answers this** — if the full
sweep gets expensive, run this arm at k=1 only.

- [ ] **Step 4: Record DtoH bytes per token alongside milliseconds**

At each `k`, capture both from the existing `nsys` path. If bytes/token stays
flat (~2.5 GB) while `k` grows, the host round-trip is per-token overhead and
amortizes across the batch; if it scales with `k`, it does not.

**Corrected 2026-08-05, and worth stating because the wrong version was already
propagating.** An earlier draft claimed this slope decides monolithic-kernel vs
small-primitives for a paged CUDA kernel. Fuel retracted that: *any* GPU
implementation removes the host round-trip equally, so the slope characterises
the **current broken placement**, not a difference between two candidate
futures. Baracuda had built a prioritisation rule on the wrong version before it
was caught.

The slope is still worth measuring — it tells us whether the round-trip
amortises across a batch, which bears on how urgent the fix is — but do not
report it as a kernel-shape discriminator.

- [ ] **Step 5: Run the sweep**

```bash
pwsh C:\Projects\fuel-crash-vmm\scripts\gpu-run.ps1 -Project lightbulb -- cargo test --release --features fuel-cuda --test gpu_paged_vs_contiguous -- --ignored --nocapture
```

for `k ∈ {1, 2, 4, 8, 16}` × `LB_PLAN ∈ {once, replan}`, both arms naming their
mode. **All GPU runs go through `gpu-run.ps1`** — it is a machine-wide mutex.

Also record `session_realize_count` at each point. "The field says `PlanOnce`"
and "the persistent path actually ran" are different claims, and only the second
is worth anything.

- [ ] **Step 6: Report the curve, including if it is boring**

A paged step that is **not** flat in `k` is the more consequential result: it
means the round-trip scales with batch and kernel work must target the plumbing
rather than the attention math.

**If contiguous fails admission on ragged lengths, that is the finding.** Record
the rejection. Do not synthesize a uniform batch to obtain a number that
describes no real workload.

- [ ] **Step 7: Commit the measurement**

Commit the harness change and the results table, quoting Task 1's `FUEL_BASELINE`
SHA. State explicitly which `k` (if any) inverts the 10.4× ratio, and whether
DtoH bytes/token was flat or scaling.

---

## Self-Review

**Spec coverage.** §1 Axis A → Task 8. §2 Axis B → Task 5. §3 files → all tasks.
§4 loader/EOS → Task 3. §5 sampling → Task 7, device → Task 2. §6 gate → Task 9,
drift test → Task 5, EOS test → Task 3, sampling determinism → Task 7. §7
batching → Task 10. §8 error handling → Task 8 (`drain_with_error`), Task 7
(per-request errors). §9 stale worktree → Task 1.

**One deliberate gap:** the spec's §7 "runner-side work, once the path is
chosen" (drain N jobs, admission control) has no task, because it cannot be
written without Task 10's result. Called out under Phase Structure rather than
left silent.

**Type consistency.** `SessionState::{new, position, parts, advance}` as defined
in Task 4 are the names used in Tasks 5 and 7. `FuelDecoder::{prefill, step}`
from Task 5 match Task 7's call sites. `EngineModel::{step_batch, decode_text}`
from Task 6 match Task 7's impl and Task 8's `run_jobs`. `LoadedLlama`'s five
fields from Task 3 are the ones read in Tasks 4, 5 and 7.

**Known risk, flagged not hidden:** Task 9's `AppState` construction and
`MemoryAwareScheduler::new` signature are written from the struct definition at
`src/api/mod.rs:114-126` but not compile-checked. Task 9 Step 2 expects a
compile error there first and says to fix the test, not `api/`.
