# Wiring `model_fuel` into the serving engine — design

**Date:** 2026-08-05
**Status:** approved, ready for an implementation plan
**Supersedes nothing.** Extends `2026-07-28-fuel-port-design.md` step 5 ("wire
batched decode") with the seam that step assumed and never specified.

---

## Summary

`src/model_fuel/` is a working Fuel-backed Llama path — loader, capture-shaped
decode loop, batched paged decoder, 2,618 lines of retained policy — that
**nothing calls**. The engine, the API and the CLI reference it nowhere; the
server still runs `candlelight` through `ParallelModelManager`.

This spec defines the seam that connects them, in two sub-projects, plus the
Fuel-side blocker that keeps a third out of scope.

---

## Context: the seam is one file and three methods

**[verified 2026-08-05]** Lightbulb's entire engine→model coupling:

- `src/api/mod.rs:174` calls `ModelRunner::start(path, max_batch, ctx_len, dtype)`.
- `src/engine/model_runner.rs` spawns a thread owning a `ParallelModelManager`
  and uses exactly `forward_batch`, `decode`, and (internally) `tokenize`.
- Everything else under `engine/` is tensor-free, as
  `2026-07-28-fuel-port-design.md` predicted in its subsystem table.

**The shipped path does not batch.** `model_runner.rs:109` is
`let mut batch = vec![ctx];` — one `RequestContext`, built from one job pulled
sequentially off a blocking `mpsc`. Despite the name, `forward_batch` is never
called with more than one request in production. Matching today's *observable*
behaviour therefore needs only single-sequence decode, which
`model_fuel::generate::generate_greedy` already performs.

This makes real batching **new capability**, not a port of something existing —
and it is why it gets its own sub-project rather than riding along.

---

## Scope: three sub-projects, forced order

| | Sub-project | Depends on | In this spec |
| --- | --- | --- | --- |
| **1** | Runner seam + single-sequence parity | — | Yes |
| **2** | Real batching | 1 (code), nothing (measurement) | Yes |
| **3** | GGUF + quantized | 1, **and a Fuel change** | **No — blocked** |

### Why sub-project 3 is out of scope

**[verified 2026-08-05, independently confirmed by Fuel peer `7te4v7wk`]**

The quantized Llama path is `QuantizedLlama3Model::from_gguf`
(`fuel-core/src/lazy_quantized_llama.rs:240`), which wraps `Llama3Model`
(`fuel-core/src/lazy_llama_full.rs:319`).

`Llama3Model`'s complete method set is `new`, `forward(tokens, start_pos)`,
`forward_embeds`, `forward_hidden_embeds`. **No `KvCache`, no
`forward_with_kv_context*`, no `forward_paged_step*`.** `LlamaModel` — the
safetensors type `model_fuel` uses — has all six.

Positive controls, because an empty search is not an absence:

- Ours: `pub fn (load_)?from_gguf` matches **14** constructors across
  `fuel-core` (gemma3, glm4, lfm2, llama, phi3, qwen2, qwen3, qwen3_moe,
  smollm3, t5, whisper, +2 in `lazy.rs`). The grep works.
- Fuel's, run independently on their side:
  `grep -c "forward_with_kv_context\|forward_paged_step"` returns **0** in both
  `lazy_llama_full.rs` and `lazy_quantized_llama.rs`, against **235** in
  `lazy.rs`.

Serving GGUF through this today re-runs the entire prefix on every token.
Against the 26.47 ms/token measured for persistent contiguous decode
(`0e3fc36`), that is not a serving path.

`PhiModel` has **both** `from_gguf` (`lazy.rs:11746`) and
`forward_with_kv_context_persistent` (`lazy.rs:11276`) in the same impl block,
so the combination exists in `fuel-core` — for Phi, not Llama.

**Ask filed 2026-08-05** with Fuel (peer `7te4v7wk`), who verified it and
routed it to the owner for sequencing. Requested shape: `_persistent` first, on
`Llama3Model` rather than the quantized wrapper — the wrapper is documented as
existing "solely to label the quantization origin", and putting it on
`Llama3Model` means the GGUF path inherits it. A non-persistent
`forward_with_kv_context` landing alone would close the gap on paper and leave
the path unservable.

**Sub-project 3 gets its own spec when that lands.** Sub-project 1's Axis B
trait (§3) is shaped so it arrives as one added impl.

---

## §1 — Axis A: how the two runners coexist

Two independent axes. Conflating them would produce one trait serving two
unrelated lifetimes:

- **Axis A — candlelight vs Fuel.** Temporary. Dies when `src/model/` is deleted.
- **Axis B — within Fuel, safetensors-f32 vs GGUF-quantized.** Permanent.

**Decision: Axis A is a Cargo feature, `fuel-engine`.**

`src/api/mod.rs` is **untouched** and `ModelRunner::start`'s signature is
**unchanged**. The swap is entirely inside `model_runner.rs`.

Accepted cost: one binary cannot serve both, so any candlelight/Fuel A/B needs
two builds and two processes. This is acceptable because the parity gate (§6)
does not compare against candlelight.

Known and accepted: `--all-features` selects the Fuel runner. That build is
already broken for an unrelated reason — `cuda` requires `candlelight/cuda`,
whose `candle-kernels v0.10.2` build script fails under CUDA 13.3 — so no
working configuration changes meaning.

### Avoiding a duplicated job loop

The ~100-line job loop (streaming/complete arms, tool-call graceful
degradation, load-failure fanout) is shared via a small trait, made generic so
there is no dynamic dispatch:

```rust
// src/engine/model_runner.rs
trait EngineModel {
    fn step_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>>;
    fn decode_text(&self, tokens: &[u32], skip_special: bool) -> Result<String>;
}

fn run_jobs<M: EngineModel>(model: M, rx: Receiver<InferenceJob>) { /* existing loop */ }

#[cfg(not(feature = "fuel-engine"))]
pub fn start(..) -> Result<InferenceRequestSender> { /* spawns ParallelModelManager */ }

#[cfg(feature = "fuel-engine")]
pub fn start(..) -> Result<InferenceRequestSender> { /* spawns the Fuel model */ }
```

`ParallelModelManager` implements `EngineModel` by forwarding to methods it
already has — no changes to `src/model/`.

**`step_batch`'s return contract** (stated because the Fuel implementation must
match it and it is not self-evident from the type). **[verified]**
`parallel_model_manager.rs` builds `vec![None; batch.len()]` and fills slots
positionally: index `i` corresponds to `batch[i]`. `Some(tok)` means that
request produced `tok` this step. **`None` means the request produced no token
this step and this is not an error** — it is mid-chunked-prefill (`:300` in the
function's body) or already stopped (`:673`). Errors are the `Err` arm of the
`Result`, and they abort the whole batch. A Fuel implementation that returns
`Err` where candlelight returns `None` would turn ordinary prefill progress into
a failed request.

**The trait carries no tensor types**, so `engine/` stays tensor-free, which the
port design names as the actual differentiator worth protecting.

**The trait is already batch-shaped** (`&mut [RequestContext]`). Sub-project 2
changes the job-draining loop and the Fuel implementation; it does not change
this trait.

Retiring Axis A later is deleting one impl and two `cfg` attributes.

---

## §2 — Axis B: the model trait inside `model_fuel`

The seam promised to Fuel. A quantized model arrives as one added impl:

```rust
// src/model_fuel/decoder.rs
pub trait FuelDecoder {
    fn prefill(&self, tokens: &[u32], st: &mut SessionState) -> Result<Vec<f32>>;
    fn step(&self, token: u32, st: &mut SessionState) -> Result<Vec<f32>>;
}
```

`impl FuelDecoder for LlamaModel` now. `impl FuelDecoder for Llama3Model` when
the Fuel change lands — no other file changes.

This trait lives in `model_fuel/`, not `engine/`, because it names Fuel types.

---

## §3 — File structure

| File | Status | Responsibility |
| --- | --- | --- |
| `src/model_fuel/session.rs` | **new** | `SessionState`: `KvCache` + `InferenceContext` + `Option<DecodeSession>` + position. Owns the persistence seam. |
| `src/model_fuel/device.rs` | **new** | `select()` — the single place a device is chosen. |
| `src/model_fuel/decoder.rs` | **new** | `FuelDecoder` trait + the `LlamaModel` impl. |
| `src/model_fuel/engine_model.rs` | **new** | Tokenizer, EOS, sampling, `RequestContext` ↔ session mapping. The `EngineModel` impl. |
| `src/model_fuel/loader.rs` | modify | `LoadedLlama` gains `tokenizer` and `eos`. |
| `src/model_fuel/loader_f32.rs` | modify | Same, on the f32 path. |
| `src/engine/model_runner.rs` | modify | `EngineModel` trait, generic job loop, `cfg` swap. |
| `src/api/mod.rs` | **untouched** | — |

### Why `SessionState` is its own type

It exists to make the persistence seam unforgettable. A caller holding a raw
`KvCache` can call `forward_with_kv_context` and lose 223× silently, which is
exactly what happened once already and is what the harness in
`tests/gpu_paged_vs_contiguous.rs` was built to measure. `SessionState` owns the
`Option<DecodeSession>` so `step()` cannot be written without it.

Fuel's own audit (peer `kblt7uwd`, 2026-08-05) characterises
`forward_with_kv_context` as the raw rebuild primitive with no persistent
sibling reachable at the same call shape — a discoverability defect rather than
a bad default. Either way the cost to a hand-rolling consumer is identical, and
this type is how Lightbulb stops paying it.

---

## §4 — Loader: EOS and the tokenizer are the real gaps

`LoadedLlama` is `{ model, config }`. It carries **no tokenizer and no EOS**.
`generate.rs`'s test loads a tokenizer separately and passes `Some(2)` as a
literal.

**[verified]** `loader_f32.rs:55` parses `config.json` via
`Llama2cConfig::from_hf_json_str`, which yields a `LlamaConfig` carrying no EOS
token.

**Decision: switch to `LlamaFullConfig::from_hf_json_str`**
(`fuel-core/src/lazy_llama_full.rs:133`), which parses the HF `config.json` and
carries `LlamaEosToks` with `is_eos(tok)` (`:96`). Its `to_lazy_config()`
(`:196`) still yields the `LlamaConfig` the model needs, so nothing downstream
changes. Reachable as `fuel::lazy_llama_full::*` — `fuel` is `fuel-core`
renamed (`Cargo.toml:152`) and `lazy_llama_full` is `pub mod` at
`fuel-core/src/lib.rs:78`.

`LoadedLlama` gains:

- `tokenizer: tokenizers::Tokenizer`, loaded from `tokenizer.json` in the same
  directory, at load time.
- `eos: Option<LlamaEosToks>`.

**Both load eagerly and fail at load, not at first request.** A server that
accepts a job and only then discovers it cannot detokenize has converted a
startup error into a per-request error.

Without EOS the runner can only stop on `max_new_tokens` — every request runs to
the limit and emits text past the model's stop. That is a visible correctness
bug, not a performance issue.

---

## §5 — Sampling and device

### Sampling

`generate.rs`'s module doc commits Lightbulb to owning token selection, citing
Fuel's consumer contract §15. `InferenceJob` carries `temperature: f64`, which
`argmax` currently ignores — so today every request would be greedy regardless
of what the client asked for.

The Fuel path uses `crate::sampling`:
`apply_temperature` → `top_k_filter`/`top_p_filter` → `sample_from_logits`.

**`temperature <= 0.0` means greedy**, routed through
`model_fuel::generate::argmax` — the existing public function, not a copy, so
the tie-break stays pinned to one implementation. This keeps the §6 gate
deterministic.

### Device

`generate.rs` hardcodes `Device::cpu()`. `SessionState::new` takes a `&Device`
rather than choosing one, so the choice is made in exactly one place:

```rust
// src/model_fuel/device.rs
/// The device this build serves on. CUDA when the feature is on and a device
/// is present; CPU otherwise.
pub fn select() -> Device;
```

Selection lives here — not in the loader, the session, or the runner — because
three callers picking a device independently is how a model and its KV cache end
up on different ones. `LoadedLlama` records the `Device` it was loaded onto so
`SessionState` cannot be built against a different one.

Falling back to CPU when `fuel-cuda` is on but no device is present is
deliberate: a server that refuses to start on a CPU-only host is worse than one
that starts slow, and the log line names which was chosen. **The measurement
harness makes the opposite choice** — `tests/gpu_paged_vs_contiguous.rs` fails
rather than falling back, because a benchmark that silently measures CPU is an
artifact.

Not optional: the port targets GPU serving, and every measurement backing this
spec was taken on CUDA.

---

## §6 — Verification

### The gate

An integration test starts the server with `fuel-engine`, POSTs to
`/v1/completions`, and asserts the returned completion is **coherent English
continuing the prompt** — the same standard `generate.rs`'s
`generates_coherent_text_end_to_end` already holds itself to.

A model wired up wrongly — transposed projection, mis-set RoPE base, wrong norm
placement — still returns tokens. Only reading the text catches it. Token count,
HTTP 200, and absence of panic all pass on garbage.

**The test fails rather than skips** on a missing checkpoint. An early return
from `#[test]` is a PASS, and a gate that reports success having verified
nothing is the artifact this project has spent the most effort learning to stop
producing.

Deliberately **not** cross-runner parity against candlelight: three of the four
suites the port design named as the oracle no longer compile, and the
feature-flag decision means comparison needs two processes.

### Supporting tests

- `SessionState::step` ≡ `generate_greedy` on the same prompt and checkpoint,
  **compared on logits with `maxdiff == 0`, not on tokens.**

  **Token equality is a vacuous oracle here.** Argmax is a discretization that
  discards nearly all of the signal: a decode-maths regression has to move a
  logit far enough to cross the top-2 gap before a single token changes, so a
  token-comparison test passes whether or not the maths changed. Fuel measured
  this directly on 2026-08-05 — a real ~1.7e-3 mis-positioning perturbation
  flipped **neither** greedy nor seeded-temperature tokens — and it bit them
  before they caught it. Their prefix-sharing correctness work is asserted at
  the logit level for the same reason.

  This is the one supporting test whose whole purpose is detecting numerical
  drift, so it is the one that must not be measured through argmax.
- EOS stops generation strictly before `max_new_tokens` on a prompt whose greedy
  continuation is known to terminate. This must be able to fail: it asserts a
  *shorter* output, which a broken EOS cannot produce.
- Sampling: `temperature == 0.0` produces identical output across two runs;
  `temperature > 0.0` with distinct seeds does not.

---

## §7 — Sub-project 2: batching

### Two candidate paths

**[verified 2026-08-05]**

| | `build_batched_decode_logits` (`lazy.rs:8977`) | `forward_paged_step_batched` (`lazy.rs:7995`) |
| --- | --- | --- |
| Ragged `cached_len` | **Rejected** — uniformity gate at `:9021` | **Supported** — per-row RoPE position and slot index |
| Minimum k | 2 | 1 |
| Persistent sibling | **None** | takes pool + sessions |

Positive control for the missing persistent sibling: `DecodeSession` appears
**64** times in `lazy.rs`. The absence is real. Confirmed independently by Fuel
peer `kblt7uwd` while auditing the contiguous route: `SessionScheduler` holds a
session and calls the persistent forward, but the batched arm calls
`build_batched_decode_logits` with no session threaded.

The uniformity gate is expected to disqualify contiguous batching for a serving
engine: requests arrive at different times with different prompt lengths, and
the gate demands every cache sit at an identical `cached_len`.

### The measurement decides, not this document

**SUPERSEDED 2026-08-06 by the measurement this section commissioned, and then
CORRECTED again same-day.** The 10.4× below is a **debug-build** number. At
release, on Fuel `8771997e`, the first release measurement put the pair at
~2.6× (paired ratios 4.43 / 2.59 / 2.09, n=3) — but that used a steady window
that smeared the one-time CUDA-graph capture-build token into the captured
arm's average, understating the captured arm's speed and so understating the
paged penalty against it. Corrected, the same pair is **paged/captured 3.98×**
(paired ratios 3.98 / 4.63 / 3.98, n=3), and **paged/uncaptured contiguous is
not resolvably different from 1×** (0.92 / 1.52 / 1.08 — the spread straddles
1.0 against a ~2.2× process noise floor; direction not established, say "not
resolvably different," never "equal").

The consequence that emerged from the two numbers together: **at k=1 the
paged penalty is capture-shaped, not paging-shaped.** Paged decode performs
about like contiguous decode *without* capture (ratio ~1×, unresolved), and
the entire ~4× gap to *captured* contiguous is the same ~4× capture is worth
on the contiguous path by itself (see `model_fuel/mod.rs` rule 3). Paging
itself is not shown to cost anything at k=1 — what costs is the captured
fast path being unavailable to the paged arm, not paging.

Caveats unchanged by the correction: **k=1 only** — k≥4 OOMs on this 8 GB
card, so the batching regime this section exists to inform remains
unmeasured; **n=3** against a ~2.2× noise floor; direction (no inversion
observed at k=1) is solid, no second significant figure is defended. The
reasoning below stands — it argued correctly that a k=1 ratio cannot decide a
batching question.

The measured 10.4× paged penalty (`0e3fc36`) is a **k=1** number, and k=1 is
precisely the case batching exists to avoid. A paged step is one graph over k
rows; if its cost is near-flat in k, the ratio inverts somewhere. Where it
inverts is not derivable by reading — and this project has a documented history
of three sessions, three positions and six retractions on exactly this class of
question, settled only by instrument.

**Sweep k ∈ {1, 2, 4, 8, 16} on both paths.** Extends the existing
`tests/gpu_paged_vs_contiguous.rs` and `pwsh scripts/gpu-run.ps1` harness.

Requirements on the sweep:

1. **Both arms name their mode explicitly.** Fuel flipped
   `PagedSessionScheduler::new` from `Replan` to `PlanOnce` on
   2026-08-05; a harness arm that inherits a constructor default stops being a
   control the moment the default moves.
2. **Unset `LB_PLAN` is an error, not a default.** It currently maps to
   `Replan`, which since the flip is the opposite of the shipped configuration.
   A bare run must not silently measure non-production.
3. **Record DtoH bytes/token alongside ms/token at every k.** If bytes/token
   stays flat (~2.5 GB) while k grows, the host round-trip is per-token overhead
   and amortizes across the batch; if it scales with k, it does not. That single
   distinction is a direct input to Fuel's backend-agnostic paged-attention
   design and to whether Baracuda targets one fused kernel or several
   primitives.
4. **Report the curve even if it is flat and boring.** A paged step that is
   *not* flat in k is the more consequential result, because it means the
   round-trip scales with batch and the kernel work must target the plumbing
   rather than the attention math.
5. **If contiguous fails admission on ragged lengths, that is the finding.**
   Record the rejection, do not synthesize a uniform batch to obtain a number
   that describes no real workload.

### Sequencing note

**The sweep has no code dependency on sub-project 1.** It extends an existing
harness against Fuel directly. It can therefore run before, during, or after the
runner work, and the implementation plan should not serialize it behind
sub-project 1 unnecessarily.

Fuel and Baracuda are both holding design decisions on this curve.

### Runner-side work, once the path is chosen

Drain up to `model_max_batch_size` jobs from the `mpsc` instead of one, hold N
`RequestContext`s, and admit against `can_admit_prompts()` /
`blocks_required_batch()` — Fuel supplies the inputs, Lightbulb makes the
decision, per the port design's "keep — ours by contract" row.

---

## §8 — Error handling

| Condition | Behaviour |
| --- | --- |
| Model, tokenizer or config missing at load | Existing behaviour: log, then drain the queue answering each job with the load error. Preserved unchanged. |
| Per-step forward failure | Propagate to that job's response channel; mark the request `Completed`; the runner thread survives. One poisoned request must not take down the server. |
| Tokenizer encode/decode failure | Same as a step failure — per-request, not fatal. |
| Batch admission failure (sub-project 2) | Requeue the rejected requests; run the batch that fits. Never drop a job silently. |

---

## §9 — Risks

| Risk | Assessment |
| --- | --- |
| **Stale Fuel worktree** | `Cargo.toml:152` points at `C:/Projects/fuel-lightbulb-port/fuel-core`, detached at `99dbf231`. Fuel's `PlanOnce` default is on `feat/persistent-decode-default` @ `23b81462`, not yet merged. **The worktree must be updated before the sweep**, or it measures a stale default and reports a number for a configuration that no longer ships. |
| **`fuel-cuda` build time** | The CUDA build is ~24 minutes on this machine. Plan tasks so a compile error is caught by `cargo check` before a full build. |
| **Sampling changes the gate** | The behavioural gate asserts content, so it must run greedy. If a future default makes temperature non-zero, the gate becomes flaky. Pinned by asserting `temperature == 0.0` in the test request rather than relying on a default. |
| **`EngineModel` shaped by candlelight** | The trait takes `&mut [RequestContext]`, which comes from the path being retired. Judged acceptable: it is an engine-level contract ("advance these requests one token"), carries no tensor types, and is already the shape sub-project 2 needs. |

---

## Out of scope

- GGUF and quantized serving (sub-project 3 — blocked on Fuel).
- Deleting `src/model/` or the `candlelight` dependency. That is the port
  design's step 6 and needs parity across more than this seam.
- Prefix caching, segmented cache, tiered storage, tool-call detection on the
  Fuel path. `ParallelModelManager` has them; the Fuel path will not at first.
  They are additive and each deserves its own decision against Fuel's
  equivalents, per the port design's subsystem table.

  **Fuel's prefix-sharing equivalent landed on main 2026-08-05 (`1c640648`)** —
  rung-1, same absolute positions, prefix at offset 0, which covers the
  shared-system-prompt case. Surface is `KvBlockPool::{register_prefix,
  splice_prefix_from, release_prefix}`, drivable directly without adopting
  Fuel's scheduler. `splice_prefix_from` returns the shared token count, so
  "prefill only `prompt[shared..]`" is enforced rather than conventional.
  Rung-2 (position-shifted / mid-prompt donation, needing RoPE delta-rotation)
  is deferred on Fuel's side. Recorded here so the eventual prefix-cache
  decision compares against something concrete rather than re-deriving it.
- Speculative decoding.
- Multi-GPU placement — withdrawn as a category error in the port design (D3).
