# Porting Lightbulb from Candle to Fuel — design

**Status**: draft, 2026-07-28. Design agreed; implementation plan not yet written.
**Counterpart docs (Fuel side)**: `fuel/docs/architecture/15-consumer-contract.md` (v0.2),
`fuel/docs/fuel-consumer-seam.md` Annex A (the Fuel-side survey of this codebase).

---

## Summary

Lightbulb moves off `candlelight` (a Candle fork) onto Fuel. This is not a tensor-library
swap. Candle is eager; Fuel is a lazy graph with an optimizer that rewrites, fuses, and
places the DAG before anything computes. The port is therefore a **rewrite of Lightbulb's
tensor core and a subtraction of everything Lightbulb built to work around Candle's
eagerness** — while its top half, which never touches tensors, stays.

The end state is a materially thinner Lightbulb: it keeps reasoning orchestration, the
OpenAI-compatible server, and constrained generation, and it hands mechanism — kernels,
fusion, placement, KV allocation — to Fuel.

---

## Context

### What Lightbulb is today

~66k LOC in a single crate on `candlelight` + `mlmf`. Tensor coupling is concentrated,
not spread:

| Subsystem | LOC | Files touching tensors | Character |
| --- | ---: | :-: | --- |
| `engine/` | 16.0k | **1 / 25** | reasoning orchestration; reaches tensors only via `speculative.rs` |
| `cache/` | 9.4k | 5 / 12 | 5 tensor files; the other 7 are policy |
| `model/` | 8.9k | **16 / 17** | the tensor core — the real port surface |
| `api/` | 2.3k | **0 / 9** | OpenAI-compatible server |
| `contracts/` | 1.3k | **0 / 6** | constrained generation |
| `multi_gpu/` | 1.8k | 4 / 6 | placement / sharding |
| `loaders/`, `memory/`, `backend/` | ~2.1k | most | weight loading, estimation, Marlin FFI |

The Candle API surface actually used is ~40 distinct items: `Tensor`/`DType`/`Device`/
`Error`, `VarBuilder`, `Linear`/`embedding`/`RmsNorm`, `ops::{silu, softmax_last_dim}`,
`QMatMul`/`QTensor`/`gguf_file`, the `llama` model types, `LogitsProcessor`, `flash_attn`.

### What Fuel is

Not a Candle replacement at the same layer. Fuel is a full ML stack: an IR (`fuel-ir`),
a graph (`fuel-graph`), an optimizer and dispatch layer (`fuel-dispatch`), a backend
contract, and five backends (CPU, CUDA, Metal, Vulkan, MKL/AOCL). **Fuel is lazy-only** —
`Tensor::matmul()` returns a graph node; nothing computes until `.realize()`.

This is the load-bearing difference. Laziness enables a class of optimization Candle
structurally could not do: cross-op fusion chosen at plan time, arm selection among
equivalent implementations, captured-graph replay, tolerance-driven precision choice.

### Verified findings that shaped this design

1. **`fuel-inference` already exists and overlaps Lightbulb ~1:1.** 6,231 LOC, 153 unit
   tests, covering eviction (LRU/H2O/weighted-voting), prefix caching, StreamingLLM,
   speculative decoding, chunked prefill, segmented eviction, KV compression (KIVI/R-KV/
   low-rank), a memory-aware scheduler, MoE routing, tiered storage, context compression,
   tool calls, and sampling. The corresponding Lightbulb surface is ~8.4k LOC.
   **[verified]** Nothing depends on `fuel-inference` yet — Lightbulb would be its first
   consumer, so expect integration defects that unit tests don't catch.
2. **Model parity is better than Fuel's own gap list stated.** `fuel-transformers/src/
   _models_retired/` holds the dead *eager* ports; the live models are **157 `lazy_*`
   modules in `fuel-core`**, with `lazy::LlamaModel` canonical and `lazy_llama_full.rs`
   covering Llama-3.1 RoPE scaling and full HF `config.json`. Working end-to-end binaries
   exist: `llama-lazy`, `qwen-lazy`, `gemma-lazy`, plus CUDA/Vulkan variants. This gives
   the rewrite a working reference to copy. Correction filed to Fuel; Annex A.3 to be updated.
3. **There is no `CustomOp` in Fuel, by design.** The primitive `Op` enum is
   build-time-closed. Custom kernels go through the kernel binding table
   (`fuel_dispatch::extend_global_bindings`) or the fused-op registry (which requires a
   total, never-panicking `decompose` into the existing basis).
4. **`fuel-inference` and `fuel-training` sit *above* the consumer seam.** Raised as a
   contract question (§15 refuses admission/eviction/fairness, yet `fuel-inference` ships
   a scheduler doing exactly that); resolved same-day in §15 v0.2 — the seam binds
   Foundation, not the repository. Adopting a shipped toolkit is a consumer choice, never
   an obligation.
5. **`fuel-core` is mid-retirement** into `fuel-ir`/`fuel-hardware`/`fuel-memory`. Import
   paths will move under us. Don't hard-code deep paths without checking.

---

## Decisions

### D1 — Parallel path in the same repo

Eager→lazy cannot be half-done inside one compiling crate: `candlelight::Tensor` and
Fuel's `Tensor` are different types, and `model/` is 16-of-17 files coupled.

```
lightbulb/
  src/model/        <- candlelight, frozen, serves as the oracle
  src/model_fuel/   <- new lazy graph path
  tests/            <- run BOTH, assert parity
switch when green -> delete src/model/ and the candlelight dependency
```

Lightbulb's existing correctness suites (`batched_transformer_correctness.rs`,
`model_correctness.rs`, `correctness_tests.rs`, `fused_rmsnorm_parity.rs`) become the
parity gate rather than being rewritten. The old path stays runnable until the new one
passes, which is what makes a rewrite of this size safe.

### D2 — Batched decode is the first end-to-end target

Chosen over a single-sequence greedy slice. Batched multi-sequence decode over a shared
KV pool is Lightbulb's reason to exist, so the first thing proven is the thing that
matters. **This makes Fuel's in-flight KV block-pool allocator a critical-path
dependency** rather than a later adoption step — see Risks.

### D3 — Multi-GPU placement goes to Fuel

`multi_gpu/` (1,767 LOC of tensor parallelism, pipeline parallelism, distributed cache)
is deleted rather than ported. Lightbulb supplies the device set as a C-5 constraint and
Fuel's optimizer decides sharding and placement. Placement among equivalent
implementations is squarely Fuel's under §15.

Caveat: Fuel's multi-device coherence protocol is a placeholder today (`authority` and
`version` fields exist in `inference_context.rs`, activated in Phase J). Single-device
correctness lands first regardless; this decision governs what we *don't* rebuild.

---

## Subsystem disposition

The governing rule, per Eric: **adopt what Fuel supplies unless Lightbulb's is
demonstrably better; where Lightbulb's is better and it is mechanism, upstream it into
Fuel** rather than keeping a private copy.

| Disposition | Subsystems | Notes |
| --- | --- | --- |
| **Keep untouched** | `engine/` (less `speculative.rs` and the overlap set), `api/`, `contracts/`, `server.rs`, `tls.rs`, `hub.rs`, `bin/` | Tensor-free. The actual differentiator. |
| **Rewrite as graph construction** | `model/` | Against `lazy::LlamaModel` as reference. |
| **Delete — Fuel decides** | `model/fused_kernels.rs`, `model/fused_rmsnorm.rs`, `engine/mixed_precision.rs`, `hardware/batch_sizing.rs`, `memory/estimate.rs`, `multi_gpu/` | Each pre-commits to an implementation the optimizer should choose. |
| **Diff, then adopt/upstream** | `cache/{eviction_policy, h2o_policy, streaming_policy, segmented_eviction_policy, prefix_cache, kv_compression, tiered_storage}`, `engine/{moe_router, speculative, memory_aware_scheduler, context_compression, tool_call, streaming_context}`, `model/chunked_prefill`, `sampling.rs` | ~8.4k LOC against `fuel-inference`'s 6,231. |
| **Replace with Fuel equivalents** | `loaders/` → `LazyVarBuilder` + `fuel-formats`; `gguf/`, `quantization/` → `fuel-quantized` | `mlmf` dependency also drops (it is itself Candle-based). |
| **Benchmark, then likely delete** | `backend/marlin.rs`, `backend/marlin_ffi.rs` | Fuel has `baracuda/quant_w4a16.rs` (Marlin's territory) and `qmatmul` landed a total Q4_0 decompose. A total decompose guarantees correctness, not speed — measure on our shapes before deleting. If Fuel is slower, the fix is a backend kernel registration routed through Fuel, not a Lightbulb-private path. |

### Things Lightbulb currently decides that it must stop deciding

The rule for the rewrite: **express what you want computed plus your constraints; never
pick the implementation.** Every site where Lightbulb selects a kernel, a fusion, a dtype,
or a device is a decision handed back to the optimizer:

- hand-fused kernels → the fused-op registry and JIT fusion
- manual mixed precision → a **tolerance budget** (C-5)
- memory estimation → **asking** Fuel for headroom (C-1)
- manual placement → supply the device set, let Fuel place

### Capture-shaped decode

Fuel's `CapturedRun` (captured-graph replay) measured **10.4× on TinyLlama-1.1B decode,
byte-exact**. Earning it requires the decode step to be capture-shaped: a stable graph,
runtime-offset KV writes, no per-token graph rebuilding, no host-side branching inside the
step. Cheap to design in from the start; expensive to retrofit. `model_fuel/` is built to
this constraint from its first commit.

---

## Plan

1. **Prove the substrate.** Run `llama-lazy` on our hardware, get one token out, confirm
   the `CapturedRun` path. No Lightbulb code. De-risks everything below and teaches the
   lazy idiom against a working reference.
2. **The `fuel-inference` diff.** Module-by-module over the ~8.4k LOC overlap set. One of
   three verdicts each — adopt Fuel's, upstream ours, or genuinely diverge — with
   *adopt* as the default and the burden of proof on Lightbulb. Output is a table
   reviewed before any port code is written. Two upstream candidates already visible:
   `kv_compression.rs` (1,998 LOC vs Fuel's 742) and `segmented_eviction_policy.rs` (844
   vs 551); both name the same strategies on each side, so this may be depth rather than
   breadth. Each verdict also flags **which modules want the allocator underneath** —
   `fuel-inference`'s `tiered_storage.rs` is effectively a C-3-lossy consumer written
   before C-3 existed and has never been wired to an allocator, so our `tiered_storage.rs`
   and its counterpart should be judged against the allocator, not against each other in
   isolation.
3. **Rewrite `model/` into `model_fuel/`** as graph construction, capture-shaped, deleting
   the hand-fusion and mixed-precision machinery rather than translating it.
4. **Audit the 70 value-extraction sites** (`to_vec1`/`to_vec2`/`to_scalar`) and 57
   `.forward()` calls. Classify each as a legitimate realize boundary (logits → sampling)
   or as hidden dynamic control flow that must become a graph construct or an explicit
   realize.
5. **Wire batched decode onto Fuel's KV allocator** (C-1 admissibility for the uniformity
   gate, C-3 evict/restore, refcounted COW splice for shared prefixes), with Lightbulb's
   retained policies driving it.
6. **Flip the switch**: parity suite green → delete `src/model/`, `candlelight`, `mlmf`,
   `multi_gpu/`, and the Marlin FFI if the benchmark says so.
7. **Upstream** whatever won its diff and belongs on Fuel's side of the seam.

---

## Risks

| Risk | Assessment |
| --- | --- |
| **Eager→lazy semantics** | The largest single risk, and it is ours, not a Fuel gap. The 70 extraction sites are where hidden host-side control flow lives. Step 4 exists to find them before they become silent correctness bugs. |
| **D2 makes the KV allocator critical path** | **Substantially de-risked 2026-07-28**: Increment 2 part 1 landed at `cae56435` — refcount-aware partial evict with an honest `{freed, still_shared}` report, geometry-keyed `PoolCapacity` for C-1, and a born-red splice×evict hazard test. Part 2 (device-backed pools, materializing `block_table` for `Op::PagedAttn`) is next. Read `kv_block_pool.rs`'s module doc before designing cache policies onto it. Remaining exposure is part 2's timing, not the allocator's existence. |
| **`fuel-inference` has zero consumers** | Unit-tested, never integrated. We will find the integration defects. Budget for it; report them upstream rather than forking. |
| **`fuel-core` is being renamed under us** | Import paths will move. Don't hard-code deep paths; prefer re-exported surfaces and check before pinning. |
| **Marlin performance parity is unproven** | A total decompose guarantees correctness and analyzability, not that the fast path matches a hand-tuned kernel. Benchmark before deleting. |
| **Multi-device coherence is a placeholder** | D3 deletes `multi_gpu/` on the strength of a protocol that activates in Phase J. Single-device correctness does not depend on it; multi-GPU capability does. |
| **`candlelight`'s divergence from Candle is uncatalogued** | It is a fork. Anything it changed relative to stock Candle is an unknown port input; surfaces during the parity phase. |

---

## Consumer-seam feedback loop

Lightbulb is Fuel's first real Class-A (inference host) consumer, and §15 is explicitly a
work in progress. Friction gets reported raw to the seam owner rather than worked around,
in four categories: a clause that doesn't cover our case (contract gap), a needed
capability that doesn't exist (roadmap gap), **Fuel deciding something we needed to
decide** (mechanism/policy violation — the most valuable kind), and having to reach past
the seam (API defect).

One such report has already round-tripped: the `fuel-inference` placement ambiguity was
raised and resolved in §15 v0.2 the same day (`aa512e95`), with a decisions-log entry.
Two other corrections from this session — the model-parity gap and `fuel-inference`'s
"unit-tested but never integrated" status — are now recorded in Annex A.3/A.4.

The test to apply before filing next time: **would replacing it require forking Fuel?**
If yes, it is Foundation and the refusals bind. If we can simply not depend on the crate,
it is above the seam and is a default we may ignore.

### Working agreements with the Fuel repo

- `C:\Projects\fuel` is a **shared working tree across three sessions — read-only for us.**
  No git operations there.
- **Never run workspace-wide `cargo check`/`cargo test`** in the Fuel repo. `tensor-tools`
  has a standing break and is a default member, so a bare root `cargo check` fails for
  unrelated reasons. Always `-p <crate>`, and one cargo invocation at a time (the build-dir
  lock serializes; parallel invocations thrash).
- Routing: paged-KV allocation → session `2eymo83p`; contract/seam/architecture →
  `trpe1mc5`.
- Small task we own as `fuel-inference`'s first consumer: its `lib.rs` header still names
  a `fuel-nn` crate that doesn't exist (lines 7, 76, 80); the real surface is `fuel-core`'s
  `lazy_nn_*` modules. Fix as part of integrating it.

---

## Out of scope

- Training. Lightbulb is an inference host; `fuel-training` is not in this port.
- Rewriting `engine/`'s reasoning layer, `api/`, or `contracts/`. They are tensor-free and
  stay as they are.
- Splitting the single 66k-LOC crate. Worth doing eventually, but bundling it with a
  tensor-layer rewrite would make both harder to review.
