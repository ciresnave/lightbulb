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

## NOT yet audited

Listed so coverage is honest rather than implied:

- **Chunked prefill** at production shapes.
- **Tiered storage's disk tier** — `DeviceKvPool` evict/restore is byte-exact, but whether an
  `Externalized` handle can back onto consumer-supplied disk storage is unconfirmed.
- **KV compression** (KIVI, low-rank) as graph transforms — R-KV is covered by the attention gap;
  the other two are unexamined.
- **Structured output contracts** — believed host-side and tensor-free, unverified.
- **Anything reaching past Candle's public API** the way attention observability reached into
  `probs`. That is the shape to look for, and it is not enumerable by reading module lists.
