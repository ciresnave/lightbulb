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

1. **`fuel-inference` is not a duplicate of Lightbulb — it is a policy layer written
   against Foundation mechanisms that were never built.** 6,231 LOC, 153 unit tests, zero
   consumers **[verified]**. The full 13-module diff (completed 2026-07-28, see
   [Appendix: diff results](#appendix-fuel-inference-diff-results)) overturned the initial
   "~8.4k LOC overlapping near-1:1" premise: **exactly one module is a clean adoption.**

   The framing that explains it (credit: Fuel's seam owner): `tiered_storage` defers byte
   movement to a "caller/runtime" that didn't exist; `SpanRegistry` names groups no
   allocator could refcount; `eviction` scores attention snapshots nobody was producing.
   All three sit above the same hole, which the in-flight KV allocator fills. That also
   explains 153 tests + zero consumers as one phenomenon rather than a coincidence — and
   it reframes this port: Lightbulb is the first consumer to actually stand on that layer,
   which is why gaps are surfacing now and in this order.
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
| **Delete — Fuel decides** | `model/fused_kernels.rs`, `model/fused_rmsnorm.rs`, `engine/mixed_precision.rs`, `multi_gpu/`, and the *estimation half* of `hardware/batch_sizing.rs` + `memory/estimate.rs` | Each pre-commits to an implementation the optimizer should choose. **Re-audit pending**: entries here must be checked file-by-file for mixed responsibilities — see the correction below. |
| **Keep — ours by contract** | `hardware/batch_sizing.rs`'s `calculate_optimal_batch_size()` + `RuntimeBatchAdjuster` | **[verified 2026-07-28]** Fuel provides the batched *arm* (`SchedulePolicy::Batched { max_batch }` + uniformity gate) but **no dynamic sizing** — `max_batch` is a caller-supplied constant and `fuel-inference/scheduler.rs` has zero batch-sizing references. §15 assigns "which work to coalesce" to the consumer. Fuel supplies the inputs (`free_blocks()`, `blocks_required_batch()`); we make the decision. |
| **Adopt Fuel's wholesale** | `cache/streaming_policy.rs` | The only clean adoption of the 13. Fuel's has `position_ids` RoPE remapping + `select_keep`/`select_evict`; ours is index arithmetic. |
| **Compose — Fuel's structure, our capability** | `cache/{kv_compression, eviction_policy, h2o_policy, prefix_cache, tiered_storage, segmented_eviction_policy}`, `engine/speculative`, `model/chunked_prefill` | The dominant outcome. Adopt Fuel's traits/registries; **upstream** our capabilities behind them. See appendix. |
| **Not overlapping — name collisions** | `engine/tool_call.rs` (499), `engine/streaming_context.rs` (479) | 978 LOC that was never overlap. Ours are token-level tool detection (CR.1) and streaming *context injection*; Fuel's are a tool schema registry and StreamingLLM sink tokens. Keep both. |
| **Judgment call, not capability** | `engine/memory_aware_scheduler.rs`, `engine/moe_router.rs` | Near-parity. Fuel's scheduler is **not** a stub (`try_admit` + pressure threshold + queue + budget accounting, verified). Ours differs by *coupling* — it extends our `SlotPool` (647) + `slot_monitor` (410). Both sides are policy, so genuinely optional under §15 v0.2. |
| **Replace with Fuel equivalents** | `loaders/` → `LazyVarBuilder` + `fuel-formats`; `gguf/`, `quantization/` → `fuel-quantized` | `mlmf` dependency also drops (it is itself Candle-based). |
| **Benchmark, then delete** | `backend/marlin.rs`, `backend/marlin_ffi.rs`, `model/awq_qwen3.rs` | **[verified 2026-07-28 on main]** `fuel-cuda-backend/src/baracuda/quant_w4a16.rs` ships **both** Marlin and AWQ natively: `marlin_gemm_f16` (:54), `marlin_can_implement_f16` (:104), `awq_gemm_f16` (:128), `awq_can_implement_f16` (:186), `AwqWeight::matmul_f16` (:444), plus `nf4_dequantize_{f16,bf16,f32}`. Neither is a Baracuda ask. Benchmark on our shapes for parity, then delete the FFI — there is no capability gap to cover, only a performance question. |

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

1. **Prove the substrate.** ~~Run `llama-lazy`, get one token out.~~ **Attempted
   2026-07-28 — it fails**, and finding that on day one was the whole point of doing it
   first. Root cause (Fuel-side, fix in flight): an unbuilt CPU mixed-precision matmul
   kernel; the model builds `[F32, BF16, F32]` and CPU registered only uniform `[T,T,T]`.
   A second, separate defect made it hard to diagnose — `plan.rs` prints a hardcoded
   `available backends: []` that ignores the real table; also being fixed. **Resume when
   the fix lands** (weights are cached; turnaround is minutes), then confirm `CapturedRun`.
2. ~~**The `fuel-inference` diff.**~~ **DONE 2026-07-28** — all 13 modules; see the
   appendix. Outcome inverted the premise: 1 clean adopt, 8 compose, 2 non-overlaps.
   The work this generates is mostly **upstreaming into Fuel**, not deleting from
   Lightbulb.
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
| **`fuel-inference` has zero consumers** | Unit-tested, never integrated. We will find the integration defects. Budget for it; report them upstream rather than forking. Confirmed by the diff: it is a policy layer standing on unbuilt Foundation mechanisms. |
| **CPU is an F32-only world for weights** | **[verified 2026-07-28 by Fuel]** Mixed `[F32, BF16, F32]` matmul is a CUDA-only capability; the CPU backend registered only uniform `[T,T,T]`. Nothing states this — the `matmul` builder and `apply_linear`'s docstring imply it's valid and it fails at realize. **Consequence for D1**: the parity oracle casts weights to F32 at load. This is arguably *better* for parity anyway — it removes BF16 rounding as a confound when isolating genuine Candle-vs-Fuel divergence — so we take it regardless of whether general CPU mixed precision lands. |
| **No runnable reference for the path we need** | `llama-lazy` exercises `lazy::LlamaModel::forward` (via a documented thin wrapper) with **no KV cache**, and never touches `Llama3Model` RoPE scaling, `InferenceContext`, `KvCache`, or `CapturedRun`. So the *decoder* has a smoke test; the *serving* path — KV cache, batched decode, capture-shaped replay — has **no runnable example at all**. That is the path this port needs, and we will be writing the first one. |
| **`fuel-core` is being renamed under us** | Import paths will move. Don't hard-code deep paths; prefer re-exported surfaces and check before pinning. |
| **Marlin performance parity is unproven** | Downgraded 2026-07-28. Fuel ships the *same* Marlin kernel natively (`marlin_gemm_f16`) plus a native AWQ path, so this is no longer "does a capability exist" but only "does it match on our shapes." Benchmark before deleting; no fallback path needs designing. |
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

## Appendix: `fuel-inference` diff results

Completed 2026-07-28. Method: public-surface comparison, module by module. **Limitation,
stated because it bit us elsewhere: this establishes *capability* differences, not
*behavioural* ones. Two implementations can expose identical APIs and differ under load.
Where we adopt Fuel's, our existing tests are the check; where no test exists, that is a
gap to name rather than assume away.**

| Module | Verdict | Basis |
| --- | --- | --- |
| `streaming_policy` | **Adopt Fuel's** | Fuel has `position_ids` RoPE remapping + `select_keep`/`select_evict`; ours is index math |
| `kv_compression` | **Compose** | Adopt `CompressedKv`/`KvCompressor` traits (ours is a closed `CompressionPolicy` enum); **upstream** our `QuantGranularity::{PerHead, PerGroup}` + `per_head_scales` (Fuel's `KiviConfig` has *only* `bits`) and our relationship-aware strategy (no Fuel counterpart) |
| `eviction` + `h2o` | **Compose** | Adopt Fuel's `EvictionContext`/trait/`VotingAggregator` (`Box<dyn>` beats our generic builder); **upstream** our stateful H2O — Fuel's `H2oPolicy` is a unit struct scoring a passed-in snapshot and structurally cannot accumulate heavy-hitters, which is most of what H2O is |
| `prefix_cache` | **Compose** | Adopt Fuel's core (`longest_prefix_match`); keep our stats — `hit_rate`, `avg_saved_tokens`, `current_size_bytes`, `check_would_hit` |
| `speculative` | **Compose** | Fuel's `verify_draft` + stats is a verification *primitive*; ours is a *driver* (`SpeculativeModel` trait, `SpeculativeDecoder::generate_tokens`). Splits exactly on the mechanism/policy line |
| `tiered_storage` | **Compose** | Fuel's is metadata-only *by explicit design* ("does not move actual tensors"); ours moves bytes (`cpu_kv_layers`, `FileDiskStore`). The "caller/runtime" it defers to is the KV allocator |
| `chunked_prefill` | **Compose** | Fuel's is a zero-copy single-sequence iterator; ours is cross-request batch scheduling + tensor materialization |
| `segmented_eviction` | **Compose** | **Fuel already has a span vocabulary** — `SpanId`, `SpanKind`, `SpanRegistry::register(label, kind, range)`, `EvictionPlan`. Ours adds parent/child hierarchy with cycle detection, `importance: f32`, and `EvictionImpact` |
| `tool_call` | **Not overlapping** | Fuel = schema/registry/text parsing; ours = token-level streaming detection (CR.1). Adopt Fuel's registry for our `src/tools/` |
| `streaming_context` | **Not overlapping** | Ours is context *injection* (`StreamingContextProvider`, `on_token`), unrelated to StreamingLLM |
| `scheduler` | **Judgment** | Fuel's is real, not a stub. Difference is coupling to our `SlotPool` |
| `moe_routing` | **Upstream candidate** | Our `load_imbalance()` + `RoutingStats` are a **C-4** instance — the measurement that reveals a wrong `capacity_factor` before tokens drop. Initially undersold as "polish"; corrected |
| `sampling` | **Consumer policy** | Resolved earlier: host-side post-processing over realized logits |

**Net**: 1 clean adopt, 2 non-overlaps, 8 compose, 2 judgment calls. The dominant outcome
is **upstreaming**, not deletion. On current evidence Fuel gains stateful H2O
accumulation, KIVI granularity control, a relationship-aware compression strategy, and
routing observability.

### Span ownership — the extent/meaning split

Raised by Eric: why do spans stay on Lightbulb's side? They largely shouldn't. A span has
two separable halves:

- **Extent** — a contiguous run of blocks with an identity, evicted/restored atomically.
  **Mechanism.** Fuel's `splice` already does this anonymously; two consumers
  (ours + `fuel-inference`) want span-atomic eviction; and the per-block `EvictReport`
  attribution we negotiated exists *only* because we reassemble group outcomes from block
  outcomes. Captured as a deferred, two-consumer-specified Fuel increment.
- **Meaning** — `CacheTag`, `name`, `importance: f32`, parent/child dependency,
  `fact_key` linking a segment to a KnowledgeBase fact. **Ours**, unambiguously:
  `importance` is valuation, and §15 assigns selection among competing work to the consumer.

Corroboration: `fuel-inference`'s `SpanInfo` carries extent + `SpanKind` and stops there —
two codebases landing on nearly the same decomposition independently.

---

## Out of scope

- Training. Lightbulb is an inference host; `fuel-training` is not in this port.
- Rewriting `engine/`'s reasoning layer, `api/`, or `contracts/`. They are tensor-free and
  stay as they are.
- Splitting the single 66k-LOC crate. Worth doing eventually, but bundling it with a
  tensor-layer rewrite would make both harder to review.
