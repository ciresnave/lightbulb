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

## Uncertain — flagged, not resolved

### Multi-GPU placement

The port hands `multi_gpu/` (1,767 LOC of tensor/pipeline parallelism + distributed cache) to
Fuel's optimizer, supplying the device set as a C-5 constraint. But **Fuel's multi-device
coherence protocol is a placeholder** — `inference_context.rs`'s `authority`/`version` fields
exist and no protocol consults them; Phase J activates it.

Single-device correctness doesn't depend on this. **Multi-GPU capability does**, and until
Phase J lands this is an unverified assumption rather than a verified equivalence. Worth
checking before anything depends on it.

---

## NOT yet audited

Listed so coverage is honest rather than implied:

- **Speculative decoding** — two models in one process/graph, draft-then-verify. `fuel-inference`
  has `verify_draft`; whether the *two-model* arrangement is expressible is unchecked.
- **Chunked prefill** at production shapes.
- **Tiered storage's disk tier** — `DeviceKvPool` evict/restore is byte-exact, but whether an
  `Externalized` handle can back onto consumer-supplied disk storage is unconfirmed.
- **KV compression** (KIVI, low-rank) as graph transforms — R-KV is covered by the attention gap;
  the other two are unexamined.
- **Structured output contracts**, **pruning** — believed host-side and tensor-free, unverified.
- **Anything reaching past Candle's public API** the way attention observability reached into
  `probs`. That is the shape to look for, and it is not enumerable by reading module lists.

---

## Method note

The one confirmed regression was found by chasing a specific falsifier. That is not a method.
A better one: for each Lightbulb capability, ask **"what did this need from Candle that was
not a tensor operation?"** — observability of an intermediate, a custom kernel, a device
placement decision, a dtype choice. Those are the places where a lazy, optimizer-owning
framework legitimately differs, and therefore where regressions live.
