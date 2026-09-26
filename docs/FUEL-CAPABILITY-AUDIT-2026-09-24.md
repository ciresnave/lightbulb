# Fuel capability audit — 2026-09-24 (second pass, post-CI-fix)

Answers CireSnave's question via the PM: *"how capable is [Lightbulb] purely on Fuel and what still
needs to be done."* Method: every subsystem still importing `candlelight` gets one verdict — **DELETE**
(fuel provides it, this is reimplementation), **WIRE** (fuel provides it, we must call it), **PORTABLE**
(a kernel is missing on one backend but exists on another — Unpopped can move it), **BUILD** (fuel
genuinely lacks it anywhere), **KEEP** (Lightbulb's own concern, not an engine concern). Every verdict is
backed by a read of fuel's actual crates via `gh api` against `origin/main` — never the local
`C:\Projects\fuel` checkout (known stale all day), never inferred from Lightbulb's own absence of an
import. Three parallel sub-investigations did the file-by-file reads; this document consolidates them
and states plainly where a row is weaker evidence (name-matched from context, rate-limited, etc.).

⚠️ **Provenance gap, stated rather than papered over: measured against `origin/main` on 2026-09-24;
the exact SHA read at the time was not recorded.** Fuel's `main` moved repeatedly that same day (the
llama/phi extraction, the `fuel-loaders` rename, and more since) and has moved further since this
document was written, so every claim below should be treated as describing fuel's crates as they stood
sometime on 2026-09-24, not as reproducible against today's `origin/main` — a re-run against the
current tip may find crates renamed, moved, or changed underneath this reading. Re-verify rather than
trust before acting on any single row for implementation.

## The one metric that matters

```
files under src/ importing candlelight: 42   (unchanged from this morning's 42 — no port code
                                               landed today; today was CI unblock + this audit)
```

**This is the only number that should be repeated.** A `fuel::`-import count is noise — CireSnave's own
design intent is that one `fuel::` import can replace several candlelight ones, so that count could sit
anywhere forever without meaning progress or its absence.

## Per-subsystem verdicts

| File | Provides | Verdict | Evidence |
|---|---|---|---|
| `src/model/awq_qwen3.rs` | AWQ 4-bit Qwen3 + Marlin | **PORTABLE** | `fuel-cuda-backend/src/baracuda/quant_w4a16.rs`: real `awq_gemm_f16`/`awq_can_implement_f16`, doc explicitly: *"AWQ (Phase 48)... Loads HuggingFace `*-AWQ` checkpoints with no repack."* CUDA/baracuda only; fuel's own doc says *"No Fuel `OpKind` dispatches here yet... once `QuantFormat::{Awq,Marlin,NF4}` land, the dispatchers call into here."* Kernel exists, fuel's own dispatch glue doesn't yet — same shape as the GGUF gaps found earlier today. Overturns this morning's "COULD NOT DETERMINE" (a keyword grep missed the doc context a full read found). |
| `src/loaders/awq.rs` | AWQ checkpoint parsing | **PORTABLE** | Same evidence. No fuel-side AWQ *loader* found, only the kernel — the loading glue is the gap, not the math. |
| `src/model/quantizable_linear.rs` | Regular/Quantized linear dispatch enum | **DELETE** | `fuel::lazy::WeightStorage` already unifies F32/BF16/Q4_0/LoRA behind one type via `apply_linear`, used throughout `fuel-parallel`'s `Linear`. |
| `src/model/{custom_transformer,custom_transformer_block,custom_attention,batch_metadata,chunked_prefill,decode_state,kv_tensor,mlp_wrapper,parallel_model_manager}.rs`, `batched_llama_wrapper.rs` | Hand-rolled batched inference on candlelight tensors | **DELETE** | Not replaced by an external fuel crate — replaced by **Lightbulb's own `src/model_fuel/batched.rs`**, already implementing batched paged attention, admission control, RoPE-batched decode on Fuel's `Tensor`/`KvGeometry`/`DeviceKvPool`. The replacement already exists in-repo; only the old code needs deleting. `batched_llama_wrapper.rs`'s own header says "Legacy reference (unused)" — already dead. |
| `src/model/fused_rmsnorm.rs` | Fused RMSNorm (CUDA via candle-layer-norm, CPU fallback) | **DELETE/WIRE** | Fuel has this at multiple layers: `.slang` cross-backend shader source (`fuel-kernels-source`), a backend-agnostic graph op (`fuel-graph/src/registry/rms_norm_last_dim.rs`), CUDA impl (`fuel-cuda-backend/src/baracuda/norm.rs`), model-facing module (`fuel-nn/src/modules/norm.rs::RmsNorm`). |
| `src/model/fused_kernels.rs` | Hand-written `fused_linear_silu`/`fused_matmul_add` | **DELETE**, low confidence | Fuel is graph-first/lazy by architecture — positioned to fuse these at the optimizer level rather than via named functions. Did not confirm the optimizer performs these *specific* fusions — weakest verdict in this table. |
| `src/model/speculative_adapters.rs` | Adapts candlelight models to a `SpeculativeModel` trait | **WIRE** | `fuel-inference/src/speculative.rs` (confirmed earlier today) has real draft/verify accept-reject math via caller-supplied closures. Needs an adapter, not new algorithm. |
| `src/quantization/{mod,norm_tweaking}.rs` | Weight-quantization calibration (norm tweaking) | **BUILD**, likely small | No fuel equivalent found — a training/calibration-time algorithm, plausibly genuinely absent from an inference-focused framework. Not exhaustively searched. |
| `src/multi_gpu/topology.rs` | Device discovery/memory/P2P — **mostly hardcoded placeholders** (`// TODO: Candle API for memory info`, fixed 80GB guess, hardcoded P2P=true) | **DELETE** | `fuel-hardware::probe::ProbeReport::probe_all()` (real per-device memory), `enumerate::{Cuda,Vulkan,Cpu}Enumerator`, `transfer_cost::BandwidthMatrix::measure` (real measured H2D/D2H bandwidth), `fuel-dispatch::topology::SystemTopology`. **Fuel's version is measured; Lightbulb's is a guess.** Probably the cleanest, easiest win in this whole audit. |
| `src/multi_gpu/config.rs` | `ParallelismMode`/`MultiGPUConfig` strategy-selection policy | **WIRE** | The size-based heuristic itself is Lightbulb's own, no fuel equivalent (fuel reports hardware facts, picks no strategy) — but must be rewired onto fuel-hardware/fuel-dispatch topology instead of candlelight's `DeviceTopology::discover()`. |
| `src/multi_gpu/tensor_parallel.rs` (Column/Row) | Sharded matmul | **DELETE** | `fuel-parallel::tensor_parallel::{ColumnParallel,RowParallel}`, zero `bail!`/`todo!`. |
| `src/multi_gpu/tensor_parallel.rs` (`Hybrid`) | Combined sharding | **CONFIRMED ABSENT — real question, not resolved** | fuel-lane confirmed: no third variant. The sharding dimension type is `ShardDim` (`fuel-parallel/src/tensor_parallel.rs:179`), exactly two values (`Column`/`Row`) — corrected from an earlier "`TensorParallelConfig`" naming guess; `ColumnParallel`/`RowParallel` are separate structs, not enum variants. Still unclear whether `Hybrid` is a real requirement or an invented strategy nobody needs. |
| `src/multi_gpu/pipeline_parallel.rs` | GPipe scheduling | **DELETE** | `fuel-parallel::pipeline_parallel::{GPipe, OneForwardOneBackward}`, generic scheduler not tied to a tensor type, zero `bail!`/`todo!`. |
| `src/multi_gpu/distributed_cache.rs` (`Sharded`/`Hybrid` cache-sync **storage**) | Cross-device KV cache storage | ⚠️ **BUILD — CONFIRMED, the one place today fuel is BEHIND Lightbulb, not ahead** | Fuel lane confirmed structurally, with a positive control: `fuel-parallel/src/distributed_cache.rs` is exhaustively coordination-only — four items (`CacheShardInfo`, `SyncEvent`, `CacheSyncProtocol`, `CacheRoutingHint`), none store cache data, all track shard/rank/position metadata. `fuel-inference`'s `SessionState` holds exactly one `KvCache` + one `InferenceContext`, and `InferenceContext::new` takes a single `Device`. Grep of `fuel-inference/src/*.rs` for `shard`/`multi_device`/`cross_device`/`multi_gpu` → **zero** (the same pattern hits real code in `fuel-parallel`'s `comm.rs`/`device_group.rs`/`tensor_parallel.rs`, so the query works and the zero is real, not a broken grep). **There is no `CacheStrategy`-shaped enum anywhere in fuel — `Sharded` is not even declared as a concept, let alone implemented.** Lightbulb's own `distributed_cache.rs:165` at least *names* the strategy and marks it unimplemented; fuel hasn't modelled it as a type at all. **Open architectural question, not yet answered: does this get built in fuel or in Lightbulb?** CireSnave's multi-GPU box will exercise this immediately, so it can't stay open indefinitely. |
| `src/cache/kv_compression.rs` | KIVI/R-KV/low-rank KV compression | **DELETE**, name-matched not independently re-verified this pass | `fuel-inference::kv_compress` states the same three strategies. Recommend a direct side-by-side read before treating as settled. |
| `src/cache/prefix_cache.rs` | Hash-keyed prefix reuse | **DELETE**, name-matched not independently re-verified this pass | `fuel-inference::prefix_cache`, same stated purpose. Same caveat. |
| `src/cache/parallel_cache_builder.rs` | Forked from Candle's `ScatteredCacheBuilder` (attributed, SPDX'd) | **DELETE, high confidence** | Lightbulb's own `src/model_fuel/policies.rs` (already built, part of the fuel port) already does this job on `fuel::kv_block_pool`. The replacement already exists in-repo. |
| `src/cache/tensor_codec.rs` | Tensor↔bytes serialization for disk tiering | **COULD NOT DETERMINE** | `fuel-inference::tiered_storage` handles tier bookkeeping; no confirmed tensor↔bytes codec found. `fuel::safetensors` might serve this role, not checked. |
| `src/cache/tiered_storage.rs` | GPU→CPU→Disk KV demotion + Lightbulb's own `KnowledgeBase`/`[KB:key]` retrieval | **SPLIT** | Tiering mechanics: `fuel-inference::tiered_storage::TieredStore`/`Tier`/`SegmentMeta` — **DELETE** (name-matched, mechanics confirmed to exist). `KnowledgeBase` retrieval-token integration: **KEEP** — a Lightbulb product feature, no engine equivalent exists or should. |
| `src/memory/{estimate,speculative,utils}.rs` | Pure memory-footprint arithmetic | **KEEP** | Capacity-planning logic, not an engine capability. `candlelight::core::DType` used only as a size-lookup enum key — trivially swappable for `fuel::DType` whenever candlelight goes. Not real work either way. |
| `src/gguf/mod.rs` | GGUF metadata/tokenizer reading (Lightbulb's own, extensively verified this repo's whole history) | **SPLIT** | The underlying byte-level types (`candlelight::core::quantized::gguf_file::{Value,TensorInfo,QTensor}`) are candle's, not Lightbulb's — **DELETE/WIRE** candidate once rebuilt on `fuel-formats::gguf`/`fuel::quantized::gguf_mmap`. But the metadata-interpretation LOGIC (tokenizer extraction, architecture refusal rules, dtype histograms) is Lightbulb's own verified work — **KEEP**, just needs a data-source swap, not a rewrite. Found while writing this doc, not previously flagged — Lightbulb's "independently tested GGUF reader" (cited in PR #92 today) is itself not fully candlelight-free at the primitive-type level. |
| `src/backend/marlin.rs` | Marlin int4 W4A16 CUDA matmul | **PORTABLE** | Same `fuel-cuda-backend/src/baracuda/quant_w4a16.rs` as AWQ above — Marlin is explicitly named alongside AWQ/NF4 in that file's doc header, same "kernel exists, dispatcher doesn't yet" shape. |
| `src/hardware/mod.rs` | Device enum with `Cuda`/`Metal` real, `Rocm`/`Vulkan` **hardcoded TODO stubs that fall back to CPU** | **DELETE/WIRE** | Fuel already supports Vulkan for real (`fuel/vulkan` feature, `fuel-vulkan-backend`, confirmed working in the per-backend audit below) — candlelight's own `Device` enum doesn't. Fuel is strictly ahead of what this file can express today. |
| `src/lora/mod.rs` | LoRA adapter loading via candlelight safetensors | **WIRE** | `fuel-nn/src/modules/mod.rs` declares `pub mod lora` — LoRA is part of fuel-core's stated NN surface (confirmed both by this module listing and by an earlier internal doc listing "linear, embedding, norm, activation, **lora**, quantizable_linear, moe, conv, sequential" as fuel-core's NN surface). |
| `src/engine/speculative.rs` | Type-only candlelight import (`Device`, `Tensor`) around Lightbulb's own scheduling logic | **WIRE** | Same target as `speculative_adapters.rs` above — `fuel-inference::speculative`. |
| `src/pruning/{mod,gguf_application}.rs` | Wanda-style activation/magnitude pruning, applied to a loaded GGUF model | **KEEP** | An offline model-compression algorithm, not something an inference engine provides. Needs a `Tensor`-type swap to `fuel::lazy::Tensor` eventually, not a fuel equivalent — the algorithm is Lightbulb's own. |
| `src/tools/mod.rs` | Test-only candlelight `DType` reference | **KEEP** (trivial) | Test code, migrates for free whenever the surrounding module does. |
| `src/lib.rs` | Doctest-style smoke example + `LlamaEosToks` reference | **KEEP** (trivial) | Documentation/example code, not production path. |

## More misleading pointers, found while writing this doc and by the fuel lane

The fourth and fifth false pointers found today (after `model_runner.rs:327`, fixed in #91; `deny.toml`'s
"we don't expose raw RSA operations"; and fuel's own example claiming to mirror an eager module that
404s) — this pattern is now recurring often enough to name as its own category, not four unrelated
incidents:

- ⚠️ **`fuel-parallel/src/distributed_cache.rs`'s own doc comment claims cache storage "lives in
  `fuel_inference`". That claim is FALSE, not merely unconfirmed** — the fuel lane's structural scan
  (with a positive control proving the query works) found no cross-device storage anywhere in
  `fuel-inference`. Not this repo's comment to fix, but the exact shape of misdirection this section
  exists to catch, and it fooled two independent readings today (this lane's first pass, and the
  assumption baked into `distributed_cache.rs`'s own row before the fuel lane's answer landed) before a
  structural scan caught it.
- **`gguf/mod.rs`'s framing of itself as an "independently tested GGUF reader"** (the framing this lane
  itself used in PR #92's body today) is accurate for the *interpretation logic* but not for the
  *underlying types* — `crate::gguf::Content` still imports
  `candlelight::core::quantized::gguf_file::{Value, TensorInfo, QTensor}` directly (see table row
  above). Nobody wrote a false comment here — but a future reader (including this lane, hours ago)
  could reasonably assume "Lightbulb's own GGUF reader" means "candlelight-free," and it doesn't yet.
  Worth a doc-comment addition in `gguf/mod.rs` stating this split plainly, as its own small fix — not
  done in this document.

## What works TODAY with `--features fuel-engine` and zero candlelight

Unchanged from this morning's status doc plus today's PR #92 (pending gate): one architecture (Llama),
two checkpoint formats (SafeTensors f32 via `load_llama_f32_from_dir`, GGUF-quantized via
`QuantizedLlama3Model::from_gguf` — the latter verified reaching fuel's own dequant dispatch and
blocked there on non-Q4_0 tensors, see #92), CPU only (no multi-GPU, no fused kernels beyond what
`fuel::lazy` already does internally), no LoRA, no AWQ, no speculative decoding, no tensor/pipeline
parallelism. Every one of those absences was checked THIS session and found to be **Lightbulb not
calling something fuel already has** (see table above) rather than a fuel-side hole — **with one
confirmed exception: cross-device KV cache storage, which fuel does not have either.**

⚠️ **SUPERSEDED 2026-09-26, this paragraph left as-taken rather than rewritten (this is an audit, not
a live status page — see the provenance note at the top).** The "blocked there on non-Q4_0 tensors"
finding above was true when measured on 2026-09-24 and is now **false**: `fuel#244` (merged
2026-09-25T22:46Z) centralized GGUF dequantization across all 10 `fuel-transformers` quantized-model
files, wiring 14 of `GgmlDType`'s 15 variants including `Q6_K`. Re-measured against fuel `main`
`580540f` (2026-09-26T00:33Z) directly — see `src/engine/model_runner.rs`'s comment (fixed in #91/#92
the same night) for the re-derived claim and its own citation correction. The current real limitation
is `Q8_1` (typed error, GAP-125) and GGUF's IQ/TQ/MXFP4/NVFP4 families (unmodeled by `GgmlDType` at
all, per that type's own doc comment — not GAP-339/`fuel#247`, a different, unrelated hazard this
lane cited wrongly once and retracted). This paragraph's *other* claims (no multi-GPU, no LoRA, no
AWQ, no speculative decoding, no tensor/pipeline parallelism) are unaffected by tonight's five fuel
merges (#244/#245/#246/#247/#253) and were re-checked against their titles/bodies before writing this
note — none of the other four touch AWQ, LoRA, speculative decoding, or parallelism.

## Milestone: what stands between today and `fuel-engine` becoming DEFAULT

In priority order, per the PM's sequencing (cheapest/highest-value first), and using this audit's own
verdicts:

1. **DELETE `src/multi_gpu/topology.rs`, wire `fuel-hardware`** — cleanest win, hardcoded stub replaced
   by a measured implementation, no design decisions needed.
2. ⚠️ **UPDATED 2026-09-26 — this precondition is now MET.** *(Original text, dated 2026-09-24, for
   the record: "Finish PR #92's GGUF path once fuel wires K-quant dequant (already filed, fuel#243-
   adjacent per the PM) — the only architecture currently usable end-to-end needs this to serve
   real-world checkpoints, not just the one local Q4_0-with-Q6_K-output file." The `#243`-adjacent
   citation was itself imprecise; the actual fix is `fuel#244`.)* **`fuel#244` (merged
   2026-09-25T22:46Z) wires K-quant dequant, `Q6_K` included — confirmed against fuel `main` `580540f`.
   Finishing PR #92's GGUF path no longer waits on fuel; the remaining GGUF-side fuel dependency is the
   config-derivation gap `fuel#246` (merged 2026-09-25T23:42Z) already closed with
   `fuel_model_loader::quantized::config_from_gguf::derive_config` — Lightbulb's own dated copy in
   `loader_gguf.rs` can be deleted once the fuel lane confirms it's safe to depend on (in progress as
   of this edit, not yet done).**
3. **Replace the Llama-only loader with `fuel-transformers`' ~90-architecture zoo** — the single
   largest capability jump available, and per today's audit, requires no new fuel work at all.
4. **Wire `fuel-parallel`** for tensor/pipeline parallelism, **delete `src/multi_gpu/`**'s
   candlelight-based Column/Row/GPipe implementations (keeping only the genuinely-Lightbulb
   `config.rs` strategy-selection policy, rewired). ⚠️ **Cross-device KV cache storage does NOT ride
   along with this step for free — it is confirmed BUILD, not DELETE, the one place today fuel is
   behind Lightbulb rather than ahead (fuel has no `CacheStrategy`-shaped type at all). This needs an
   architectural decision (fuel or Lightbulb?) before step 4 can be considered complete, and
   CireSnave's multi-GPU box will need an answer, not just the tensor/pipeline half.** `Hybrid` TP
   stays open pending real-requirement confirmation, lower urgency.
5. **Wire `fuel-inference::speculative`** for speculative decoding.
6. **Flip `fuel-engine` from opt-in to default**, delete `candlelight`, delete this session's
   `[patch]` block in `Cargo.toml` along with it.

Steps 1-5 are almost entirely **DELETE/WIRE**, matching CireSnave's own prior ("fuel should be
relatively complete on some backends") — confirmed correct by today's evidence, not merely assumed.
**BUILD work found today is small**: quantization calibration (`norm_tweaking.rs`), and two genuinely
open questions (`Hybrid` TP, cross-device cache storage) that need the fuel lane's answer before they
can even be classified.

## Per-backend completeness

Method: direct file reads via `gh api repos/ciresnave/fuel/contents/<path>` (GitHub's code-search API
rate-limited mid-audit from this session's earlier heavy use; fell back to direct-fetch + local grep,
which has a much higher limit). This is a **spot-check of key files per backend, not an exhaustive
per-crate sweep** — stated as a limit, not hidden.

| Backend | Verdict | Evidence |
|---|---|---|
| `fuel-cpu-backend` | Complete on files checked | 15 source files; `ops.rs`/`quantized.rs` spot-checked, zero `todo!`/`unimplemented!`/"not yet implemented". Optional MKL/AOCL acceleration via thin FFI-loader shim crates (`fuel-aocl-cpu-backend`, `fuel-mkl-cpu-backend` — 3 files each, not independent backends). |
| `fuel-cuda-backend` | Complete on files checked | 14 files + `baracuda/` subdir; `Cargo.toml` pulls the full baracuda FFI stack (driver, cuBLAS, cuRAND, NVRTC, CUTLASS) unconditionally, plus optional cuDNN/NCCL — production depth, not a stub. `flash_attn.rs`'s only restrictions are ordinary kernel preconditions (F16/BF16 only, no strided inputs), not missing capability. |
| `fuel-vulkan-backend` | **Missing flash-attention specifically — PORTABLE, not BUILD** | 9 files, depends on a real published `vulkane = "0.16.0"`. Has `capture.rs`/`residency.rs` like CUDA does, but no `flash_attn.rs` equivalent. CUDA/baracuda has it; this is exactly the kernel-porting case Unpopped exists for, per CireSnave's own account. |
| `fuel-metal-backend` | **Not determined** | 6 files, has its own `quantized.rs`; not spot-checked deeply given time. Lower priority — not in CireSnave's stated backend focus (CUDA/Vulkan/CPU). |
| `fuel-aocl-cpu-backend` / `fuel-mkl-cpu-backend` | N/A — not independent backends | Thin binding-library shims extending `fuel-cpu-backend`'s optional acceleration paths. |

**Kernel-seam mechanism** (`fuel-kernel-seam/src/lib.rs`, read in full): two distinct dispatch layers,
not one. (1) A **static capability registry** (`fuel-dispatch::dispatch::CapabilityRegistry`) — each
backend registers `(OpKind, DType)` pairs it supports; `find_backend_for` walks registered backends in
order (GPU before CPU by convention) and returns the first match or a typed `NoBackendForOp` error,
never a panic. (2) A **JIT kernel-seam** — Fuel asks Baracuda's real `Synthesizer` to fuse a kernel for
a subgraph at runtime, cost-gated, off the hot path — this is the mechanism matching CireSnave's
"Unpopped ports a kernel once one backend has it" narrative, specifically for *fused* kernels.

**Dispatch is per-operation, confirmed by reading `fuel-dispatch/src/dispatch.rs` directly** —
`find_backend_for(op, dtype)` is called per op per dtype, not once per process. A single graph can
legitimately route different ops to different devices within one execution — the mechanism a
heterogeneous multi-GPU PCIe-switch box needs. **Not fully confirmed**: `fuel-hardware`'s actual
device-*selection* policy for heterogeneous hardware (as opposed to the dispatch mechanism itself,
which is confirmed) — ran out of time this pass.

**Bottom line, stated as a measurement rather than a restated prior**: the evidence collected today is
consistent with CireSnave's expectation that fuel is relatively complete on CPU/CUDA/Vulkan. The one
concrete backend gap found (Vulkan flash-attention) is exactly the PORTABLE shape he described. This
does not mean the audit is exhaustive — Metal is unchecked, and the non-exhaustive per-backend sweep
means further `todo!`/`bail!` sites could exist unfound.

## What I could not determine (collected, not scattered)

- ~~Whether cross-device sharded KV cache storage exists anywhere in fuel~~ — **RESOLVED by the fuel
  lane: it does not, anywhere, confirmed with a positive control.** See the BUILD verdict above; the
  open item now is architectural (fuel or Lightbulb builds it), not factual.
- Whether `Hybrid` sharding (`ShardDim` has only `Column`/`Row`) is a real requirement anyone needs, or
  an invented strategy.
- Whether fuel's lazy-graph optimizer actually performs the two specific fusions
  `fused_kernels.rs` hand-writes (architectural claim found, not a specific fusion-pass test).
- `src/cache/tensor_codec.rs`'s fuel equivalent, if any.
- Full depth of `fuel-metal-backend`.
- `fuel-hardware`'s heterogeneous-device *selection* policy (dispatch mechanism confirmed; policy not).
- Exhaustive (non-spot-check) `todo!`/`bail!` sweep across all backend crates — GitHub's search API
  rate-limited mid-session; what's reported here is a spot-check, named as such throughout.
