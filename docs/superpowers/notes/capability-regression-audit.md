# Capability-regression audit: what could old Lightbulb do that new Lightbulb can't?

**Date**: 2026-07-29. **Fuel tree**: `C:\Projects\fuel-lightbulb-port` @ `13279179`.

**Why this exists.** Fuel's reverse gap list is *forward*-facing — what the consumer needs
from Fuel to proceed. This is a different axis: **capability regression across the port.** The
first instance was found by chasing a falsifier, not by looking, which is a poor way to
discover them.

**Eric's framing, which shapes what to do with a finding**: if old-Lightbulb functionality
isn't expressible on Fuel, that's a discussion with Fuel / Baracuda / KISS — and the outcome
may be that the capability belongs in another project, at which point Lightbulb *uses* it
rather than reimplementing it. A gap is not automatically Lightbulb's problem to solve.

**Evidence level**: everything below is `[read]` from source unless marked. **Nothing here
was executed.** Coverage is partial and the unchecked list is stated, not omitted.

---

## Confirmed regression

### Attention observability — arm-gated

**H2O and R-KV need the per-key column-sum of post-softmax attention weights**
(`a_t[k] = Σ_q probs[q][k]`), every step, every layer.

- **Decomposed arm**: fine. `registry/flash_attn.rs:285` materialises `probs` as a real node.
- **Fused (FlashAttn) arm**: impossible by design — avoiding the `[B,Hq,Sq,Sk]`
  materialisation *is* the op's value.

**Consequence**: attention-driven eviction and FlashAttention are mutually exclusive, making
arm choice a **C-5 constraint set per deployment**. Lightbulb's eviction policy catalogue is
therefore **not uniformly available** — `h2o_policy` and R-KV are arm-gated; `streaming_policy`,
recency, and `segmented_eviction_policy` are not.

**Status**: open four-way discussion (Lightbulb / Fuel / Baracuda / KISS). The route is a
second output slot via `output_views` (mechanism exists, used by `selective_scan` and
`ssd_chunk_scan`), at O(Sk) rather than O(Sq·Sk). Baracuda is establishing whether their
kernel generator can emit that shape. Full analysis: `falsifier-1-attention-arm.md`.

---

## No regression — verified

### LoRA — Fuel's half is *stronger*, and the two are complementary

| | What it does |
| --- | --- |
| **Lightbulb** `src/lora/` (671 LOC) | **Ingestion**: `LoraAdapter::load`, multiple `LoraFormat`s, `validate` against base tensors, name mapping, `merge_into(base, scale)` — permanent merge |
| **Fuel** `lazy_nn/lora.rs` | **Layer math**: `LazyLoraLinear`, `y = base(x) + (alpha/rank)·x@A@B`, over a frozen `WeightStorage::{F32, BF16, Q4_0}` |

**This is a capability *gain*.** Lightbulb could only *merge* adapters into base weights. Fuel's
`WithLoRA` serves **unmerged** adapters over a frozen — and optionally **Q4_0-quantized** —
base, which is exactly the shared-immutable-base + per-tenant-delta shape multi-tenant adapter
serving needs. Lightbulb's loader/validator/name-mapper is the piece Fuel doesn't have and
should stay consumer-side.

### GGUF quantized inference

`fuel-core/src/lazy_quantized_llama.rs` — `QuantizedLlama3Model::from_gguf` (`:240`).
`WeightStorage::Q4_0` carries GGML blocks directly, dispatching through `Op::QMatMul`.

### Marlin / AWQ 4-bit

**[verified 2026-07-28]** `fuel-cuda-backend/src/baracuda/quant_w4a16.rs` ships both natively:
`marlin_gemm_f16` (:54), `awq_gemm_f16` (:128), `AwqWeight::matmul_f16` (:444),
`nf4_dequantize_{f16,bf16,f32}`. Lightbulb's Marlin FFI and `awq_qwen3.rs` are deletion
candidates, not gaps.

---

## Constraint rather than regression

### No `CustomOp` escape hatch

Candle let Lightbulb drop in arbitrary CUDA via `CustomOp1`/`CustomOp3`. Fuel's primitive `Op`
enum is **build-time-closed** by design — an opaque node would be invisible to the optimizer's
base-map analysis, which is the whole machine.

Replacements: the **kernel binding table** (`fuel_dispatch::extend_global_bindings`, runtime-
extensible) for a new *implementation* of an existing op; the **fused-op registry** (a total,
never-panicking `decompose` + a re-fusion `pattern`) for a new *identity*.

**Not a regression in capability, but a real change in how custom kernels are contributed** —
and it routes work to Baracuda/Fuel rather than into Lightbulb, which is the intended direction.

---

## Resolved — was the near-miss

### Multi-GPU — model sharding, NOT op placement

**The most dangerous entry in this audit**, because it was one decision away from deleting
working code.

The port originally said: delete `multi_gpu/` (1,767 LOC), hand placement to Fuel's optimizer.
That conflated two capabilities. **Fuel's multi-device story is *op placement*** — distributing
operations across a supplied device set. **Lightbulb's is *model sharding*** — splitting one
layer's weights across GPUs.

**[verified]** ours: `ShardingStrategy::{ColumnWise, RowWise, Hybrid}`, `TensorShard`,
`TensorShard::all_reduce` after a sharded matmul, and 728 LOC partitioning layers across
pipeline stages. **[verified by Fuel]** greps for `tensor_parallel`/`data_parallel` across
`fuel-dispatch`/`fuel-core`/`fuel-graph`: **zero implementation hits.**

**Resolution (Eric)**: multi-device belongs in **Fuel**, with our code as *one possible guide*
and an explicit caution — establish each piece belongs before building it. Per-file placement
is in the spec's D3.

**Operative constraint: `multi_gpu/` stays un-deleted until the Fuel-side equivalent exists
*and is verified*.** Deleting on the strength of a plan is the same error one step removed
from deleting on the strength of a module name.

**How it was caught**: reclassifying it from "resolved" to "uncertain". Had it stayed
"resolved", the deletion would have proceeded on an assumption nobody had tested.

---

## Audited 2026-07-29 (second pass)

### Speculative decoding — no regression, and the thing that could have broken it didn't

`SpeculativeModel::forward_logits(&mut self, tokens, position) -> Result<Tensor>` plus
`reset_cache`; `SpeculativeDecoder` drives two implementors (draft + target), each with its own
KV cache.

**The risk was graph affinity, not the trait.** Every Fuel `from_*` constructor mints a new
graph, and tensors from different graphs cannot be combined — so two resident models are two
graphs. Speculative decoding survives because **draft and target never meet at the tensor
level**: the accept/reject test compares *logits*, and
`DecodeModel::forward_with_kv_context_persistent` returns **`Vec<f32>`** — already realized.
The comparison is host-side arithmetic over two independent graphs' outputs.

So: two `DecodeModel` implementors, two `InferenceContext`s, comparison after realize. **The
realize-at-logits boundary is what makes multi-model arrangements work**, which is worth
knowing before designing anything else wanting two models resident.

### Pruning — no regression, but it is a **C-6 consumer**, and that matters

`PruningMask::apply(weights)`, `score_weights(weights, activations)`, and
`WandaScorer::accumulate_activations(&mut self, activations: &Tensor)`.

Wanda-style scoring is `|weight| x ||activation||`, so it **needs activations from a forward
pass** — observability of an intermediate, structurally the *same shape* as the H2O finding.

**But it is the other C-6 regime.** Pruning is offline one-time calibration, not per-token
per-layer. That is exactly the "occasional" case §15's original C-6 anticipated: observation
changes the plan, the cost is reported through C-4, and paying for one broken fusion is an
evaluable trade.

**So Lightbulb now has one consumer in each C-6 regime** — pruning (occasional, already served)
and H2O/R-KV (hot-path, which forced the v0.3 amendment). Having both named makes the regime
split concrete rather than hypothetical, and shows it was not invented for a single awkward
case.

---

## Audited 2026-07-29 (third pass) — and it corrects an earlier recommendation

### KV compression — the public surface substantially overstates what is implemented

**This is the most consequential finding in the audit, because it invalidates advice I gave
Fuel.** My 13-module `fuel-inference` diff classified `kv_compression` as *"compose — adopt
Fuel's traits, upstream our KIVI granularity control and relationship-aware strategy."* That
verdict was read off the public surface. The implementations do not back it.

**[verified] Low-rank compression does not compute a low-rank approximation.**
`compute_low_rank` (`:1070`) says so in its own comment — *"we'll use random projection as a
proxy. TODO: Replace with proper SVD when available in Candle"* — and returns
`Tensor::randn(...)` for `U`, `Tensor::ones(...)` for `S`, and `Tensor::randn(...)` for `Vt`.
**The input tensor is discarded.** It is not an approximation of anything; it is noise with the
right shape.

**[verified] KIVI's `PerGroup` granularity panics.** `todo!("Grouped quantization not yet
implemented")` (`:446`). `PerHead` and `PerChannel` are real. **`PerGroup` is precisely the
option that made our config look richer than Fuel's `bits`-only `KiviConfig`** — and it is the
one I recommended upstreaming.

**[verified] KIVI has a suspected correctness bug in the real path too.** `compute_scales`
derives a symmetric scale (`abs().max / (2^bits − 1)`), so `tensor / scale` spans
`[−max_val, +max_val]`. `quantize` then applies `clamp(0.0, max_val)` with
`zero_points: None` (`:492`). On signed KV activations that **clamps every negative value to
zero**, discarding half the range. Flagged as suspected rather than asserted — I have not
traced every path — but the symmetric-scale/asymmetric-clamp mismatch is visible in the code.

**[verified] Relationship-aware compression is scaffolding.** Its clustering is *"TODO:
Implement more sophisticated clustering (DBSCAN, hierarchical)"* (`:772`) and its causal
analysis *"TODO: Implement proper causal graph analysis"* (`:839`). Also
*"TODO: Implement proper chaining of multiple compressors"* (`:145`), so composing strategies
is unimplemented.

**Consequences:**

1. **The upstreaming offer is withdrawn** for KIVI granularity and relationship-aware. Offering
   Fuel a `todo!()` and a TODO-scaffolded strategy would have been worse than offering nothing.
   Corrected to Fuel directly.
2. **`kv_compression.rs`'s 1,998 LOC vs Fuel's 742 is not depth — it is unfinished breadth.**
   The size delta I cited as evidence of richer capability partly measures stubs.
3. **No regression to record**: a capability that never worked cannot be lost. Low-rank and
   `PerGroup` are *absences in both systems*, not port casualties.
4. **SVD is absent from Fuel and from KISS's op vocabulary** (checked both). So if low-rank KV
   compression is ever wanted for real, the decomposition has to come from somewhere — per the
   placement rubric, an op vocabulary question for KISS and a numerics question for a backend,
   not something Lightbulb should hand-roll again.

**The method lesson, which generalises past this module.** The diff's stated limitation was
*"this establishes capability differences, not behavioural ones."* That limitation is exactly
what bit. A module can expose a rich, well-documented, plausibly-typed API and do nothing
behind it — and reading the API cannot tell you which. **`grep -n "todo!\|unimplemented!"` over
a module before claiming it as a capability is a five-second check that would have caught all
of this**, and it is now the first thing to run on any module proposed for upstreaming.

---

## Audited 2026-07-29 (fourth pass) — the named items are complete

**Gate zero first, per the rubric.** Lightbulb: `chunked_prefill.rs` clean,
`tiered_storage.rs` clean, `contracts/` one hit. Fuel: `chunked_prefill.rs`,
`tiered_storage.rs`, `kv_block_pool_device.rs` all clean.

**And gate zero on the third upstreaming claim, which had never been run.** A.4's
*"dominant outcome is upstreaming"* rested on three claimed gains; two were withdrawn above.
**`h2o_policy.rs` passes gate zero — no stubs — and it is genuinely wired**:
`custom_transformer.rs:590`/`:745` feed `update_attention_scores` from the real forward pass
(the two extraction sites deliberately preserved when the dead-debug realizes were removed),
through `parallel_cache_builder.rs:862` and `segmented_eviction_policy.rs:597`. **So stateful
H2O accumulation is real**, the surviving upstream candidate is verified rather than assumed,
and the attention-observability finding that drove §15 v0.3 and Baracuda's kernel prototype
rests on an implemented capability.

### Chunked prefill — no regression, expressible

Complementary as the diff recorded (Fuel: zero-copy single-sequence iterator; ours:
cross-request batch scheduling + tensor materialization). The open question was *"at production
shapes"* — i.e. whether Fuel supports **incremental** prefill into a live KV cache.

It does. `DecodeModel::forward_with_kv_context_persistent(tokens, cache, ctx, session)` takes
`tokens: &[u32]` documented as *"full prompt on prefill, the last token on decode"* and mutates
the cache in place. Chunked prefill is therefore repeated calls over successive slices — no new
mechanism required.

### Tiered storage's disk tier — **already largely built in Fuel**; corrected 2026-07-29

**My first answer under-credited what exists, and Eric caught it.** I had said the disk tier
was "expressible via `read_block`/`write_block`, consumer owns storage," which implied Fuel
carried only byte-level primitives. It carries much more than that.

**`fuel-inference/src/tiered_storage.rs` is multi-tier storage with a disk tier**, and it is
further along than the audit credited:

| Present in Fuel | |
| --- | --- |
| `Tier::{Gpu = 0, Cpu = 1, Disk = 2}` + `is_faster_than` | the full tier model, disk included |
| `SegmentMeta { key, position_range, size_bytes, tier, access_count }` | placement tracking |
| **`position_range` "preserved across tier moves for correct positional embedding re-injection"** | **the RoPE-phase requirement, designed in from the start** |
| `TieredStore` with `gpu_budget`/`cpu_budget`, `gpu_used`/`cpu_used`/`disk_used` | per-tier byte budgets and accounting |
| `register` / `demote` / `promote` / `remove` / `get` | lifecycle |
| `candidates_for_demotion(tier, needed)` | demotion selection |
| `touch` + `access_count` | LRU recency |
| `TierTransfer { key, from, to, size_bytes, position_range }` | a descriptor of the move to execute |

**What is genuinely absent is only the byte movement**, and it is *explicitly* delegated —
`TierTransfer` is documented as *"Describes a tier transfer the caller must execute"*, with
*"caller must store positions with the data."* The module header says it *"does not move actual
tensors (that responsibility belongs to the caller / runtime)."*

**So the missing piece is narrow and has two halves:**

1. **Device side — exists.** `DeviceKvPool::read_block(layer, kind, phys) -> Vec<f32>` and
   `write_block(layer, kind, phys, &[f32])` move blocks between the device pool and host.
   **f32-only today**, with a byte/dtype-generic form tracked for the CUDA bf16 pool.
2. **Disk side — absent in Fuel.** Nothing writes host bytes to a file and reads them back.
   Lightbulb's `FileDiskStore` does exactly this.

**Revised placement.** Byte movement to and from disk is *mechanism*, not policy — per the
rubric it belongs in Fuel, completing what is already started there, rather than staying a
consumer implementation. That makes Lightbulb's `tiered_storage.rs` overlap **larger** than the
diff's "complementary" verdict implied: Fuel has model + budgets + policy + position
preservation; we have those *plus* the I/O. What is distinctively ours narrows to the
`FileDiskStore` backend and the `fact_key` link to the KnowledgeBase (`<RETRIEVE:key>`), which
is genuinely consumer semantics.

**Corollary — a positive one.** The position-preservation property I raised with the allocator
session as a RoPE risk is *already* a stated invariant on both `SegmentMeta` and `TierTransfer`.
Two independent designs converged on carrying position ranges through tier moves, which is
decent evidence it is the right invariant rather than a Lightbulb quirk.

### The mmap decision collapses Cpu and Disk — but not yet

**Eric: "In Fuel, we had decided that all storage on the host would be memory mapped to a file.
That means that the file tier may not be separate from the host tier any more. What is in host
memory is on disk."**

**Status, checked rather than assumed** — the distinction matters because it changes whether
the entry above is current:

| | State |
| --- | --- |
| Decided | **Yes** (Eric), recorded as a session-memory project entry `project_unified_durable_tensor_store` |
| Implemented | **No.** `fuel_memory::Storage` is `{ inner: BackendStorage, dtype, bundle, stype }` — *"Backend variant + the bytes themselves"*. No file mapping. `inference_context.rs:1038` says the persistent map is *"today a simple in-memory HashMap"* with the mmap-backed store as future work |
| In Fuel's architecture docs | **No.** `storage-unification.md:778` explicitly scopes it out: *"Storage on disk / memory mapping. Out of scope here."* |

**So the disk-tier entry above describes today accurately, and has a shelf life.** Once the
durable store lands, host and disk stop being distinct *locations* and become one mapping at
differing *residency*.

**Three consequences worth designing for now:**

1. **`TieredStore`'s three-tier model becomes two tiers plus residency.** `Tier::Cpu` and
   `Tier::Disk` would describe the same bytes, so `cpu_used`/`disk_used` stop being independent
   budgets and `demote(key, Tier::Disk)` stops being a byte-moving operation. The tier model
   would want revisiting rather than reinterpreting.

2. **It collides with C-1, and this is the part that matters to an inference host.** C-1
   requires Fuel to report headroom *in the consumer's admission unit*. If host residency is
   the kernel's page-cache decision, resident bytes become an **observation, not a budget** —
   we cannot admit sessions against a quantity we do not control. The control surface shifts
   from allocation to `mlock`/`madvise`, and "free host bytes" stops being answerable in the
   way admission needs. **This should be raised at the seam before the durable store lands**,
   because it changes what C-1 can promise for host-tier KV.

3. **Our `FileDiskStore` probably becomes unnecessary** — a good outcome, and it narrows
   "distinctively ours" further, to just the `fact_key` KnowledgeBase link.

**What does not change**: position preservation across tier moves. A promoted segment still
needs its original positions for correct RoPE phase whether the bytes moved or merely became
resident.

### Structured output contracts — no Fuel bearing, one pre-existing gap

Entirely host-side: parses generated *text*, no tensors anywhere. Fuel is irrelevant to it, so
no regression is possible.

Gate zero did find a gap **in Lightbulb, pre-dating the port**:
`OutputContractSpec::Json => None` (`validation.rs:117`) — the `Json` variant is declared and
always fails to parse. `EnumChoice`, `TaggedFields`, and `CommitBlock` are real. Recorded here
because the audit found it, but it is a Lightbulb TODO rather than a port risk, and per the
absence-in-both-systems rule it is not the audit's business to fix.

---

## Stub census — how much of Lightbulb is actually implemented?

Run because `kv_compression`'s surface overstated its implementation badly enough to withdraw
two upstreaming offers, and the obvious worry was that the same error sat in other verdicts.
**It does not.**

**Gate zero across all of `src/`**: exactly **one** `todo!()`/`unimplemented!()` — the known
`QuantGranularity::PerGroup` in `kv_compression.rs:446`.

**But gate zero is a weak probe, because the worst case wasn't one.** Low-rank compression is
not a `todo!()`; it is a function that computes its input, discards it, and returns
`Tensor::randn`. So the sharper probe is **synthetic data in a production path**:

`Tensor::randn` outside `#[cfg(test)]` appears at five sites. Three are **false positives** —
doc-comment examples in ```` ```ignore ```` blocks (`fused_kernels.rs:45/46`,
`tensor_parallel.rs:50`). The remaining two are `kv_compression.rs:1092/1099`, the low-rank
stub already recorded.

**Result: `kv_compression` is the outlier, not the pattern.** `tensor_parallel::from_full_tensor`
is real (bounds-checked, genuine sharding), which also matters for the decision that Lightbulb's
multi-GPU code serves as *one possible guide* for Fuel's multi-device work — the guide is not
scaffolding.

**Two probes worth keeping**, since one alone would have missed the case that cost the most:

1. `grep -n "todo!\|unimplemented!"` — catches declared-unimplemented.
2. `Tensor::randn`/synthetic constructors outside `#[cfg(test)]` — catches
   **silently-unimplemented**, which is the more dangerous form because it type-checks, runs,
   and returns plausibly-shaped output.

The second is domain-specific; the general form is *"does this function's output actually
depend on its input?"* — which is what a stub cannot fake and a reader cannot see from a
signature.

---

## NOT yet audited

Listed so coverage is honest rather than implied:

- **Anything reaching past Candle's public API** the way attention observability reached into
  `probs`. That is the shape to look for, and it is not enumerable by reading module lists.
