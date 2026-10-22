# Gaps opened by removing Candle/candlelight (2026-10-22)

CireSnave, 2026-10-22, verbatim: "I've got an easy solution for Lightbulb.
Remove all dependencies on Candle and Candlelight and then start fixing the
errors that leaves." And: "If Lightbulb loses functionality by killing its
CandleLight dependencies and Fuel doesn't have the capabilities necessary to
provide that functionality, that highlights possible gaps in Fuel. Then its a
matter of deciding whether to add that functionality to Fuel or if it has
been superceded by something better."

This document is the list. Each entry: what was lost, what used it, whether
Fuel/mlmf has it today, and a view on add-to-Fuel vs superseded. The PM routes
the add-to-Fuel items; nothing here was re-implemented in Lightbulb to paper
over a gap — deleted code stayed deleted.

Context: candlelight was already NOT the only path (`model_fuel`/`fuel-engine`
has existed since the Candle→Fuel port began; see `ROADMAP.md`), but it was
still the DEFAULT. This removal makes the Fuel path unconditional. Of the ~34
`src/` files that imported candlelight directly, most had **zero production
callers even before removal** (dead weight accumulated during the port) —
those are noted as "dead code, not a capability loss" below, not real gaps.

## Real capability gaps (candlelight provided this; Fuel's live path does not)

### 1. `cache_stats` admin endpoint goes dark
**What it was:** `ModelRunner::start`'s `SharedCacheStats` parameter, written
by `run_jobs` after every job (prefix-cache hit/miss/eviction counters), read
by the `cache_stats` HTTP admin endpoint without going through the inference
channel (M4 slice 1).
**Used by:** `src/api/admin.rs`'s `cache_stats` handler.
**Fuel today:** `ModelRunner::start`'s Fuel implementation
(`src/engine/model_runner.rs`) accepts the parameter for signature
compatibility and never writes it — confirmed in its own doc comment before
this removal. The endpoint now always reads the zero value the `Arc<Mutex<_>>`
was constructed with.
**View:** genuine functionality loss, not superseded. Fuel's
`FuelSchedulerDriver`/`run_scheduled_jobs` path (board item 97) has no
prefix-cache concept of its own to report from — this would be new Fuel-side
instrumentation, not a Lightbulb-side wiring fix. Add to Fuel if the admin
endpoint's data is still wanted; otherwise the endpoint should be changed to
say "not available on this backend" rather than silently reporting zeros,
which is the current (actually worse) state and not fixed by this removal.

### 2. `scheduler_stats`/`scheduler_queue` admin endpoints go dark
**What it was:** `SharedScheduler` (`MemoryAwareScheduler`), mirrored by
`run_jobs` on every admission/departure (M4 slices 5.4a/5.4b/5.4c — real
priority admission, slot-pool observability, the `X-Lightbulb-Priority`
header).
**Used by:** `src/api/admin.rs`'s `scheduler_stats`/`scheduler_queue`
handlers.
**Fuel today:** same as above — accepted, never written. Fuel's scheduling
(board item 97) doesn't go through `MemoryAwareScheduler`/`SlotPool` at all.
**View:** real loss. This was recent, deliberate work (M4 slices 5.4a-c,
PRs #182/#183/#186) that the Fuel path never had. Add to Fuel if priority
admission matters to the production serving path; otherwise these three M4
slices' work is dead on the shipping build starting now, which the PM/
CireSnave should know explicitly rather than discover later.

**⚠️ Two conflicting PM rulings on the same question — not resolved by this doc, flagged for
the PM.** `docs/API-EXTENSIONS.md` documents **Option B** (the `X-Lightbulb-Priority` header
above, `low`/`normal`/`high`/`critical`, invalid values silently default — **PM-ruled
2026-10-21**, already implemented: `InferenceJob.priority`/`priority_from_headers` in
`src/api/openai/chat.rs`/`src/engine/model_runner.rs`, dead on the Fuel path per this entry's
"Fuel today" line, same as everything else here). Asked again without that prior ruling surfacing,
**PM-ruled 2026-10-22: Option A** — optional top-level `priority` field on chat/completions,
values `"low"|"normal"|"high"`, default `normal`, unknown value = 400 with a typed error, same
convention as vLLM. **Neither has been implemented against the live Fuel path**, so nothing is
shipped-and-wrong yet, but `docs/API-EXTENSIONS.md` currently asserts Option B is the answer and
would need updating (or retracting) if Option A is the one that stands. Which ruling governs is
for the PM to say, not this doc. Once settled, whichever option it is plumbs to a new Fuel-side
`set_priority` hook once Fuel publishes a version carrying it; **Fuel's ordering hook stays
parked until lightbulb holds a consumer for it** — tell the Fuel lane once this is plumbed. Not
started: this needs the add-to-Fuel decision above made first
(there is nothing in Fuel yet to plumb to), then Fuel's crates.io publish.

### 3. Dtype selection (f16/bf16) for SafeTensors models
**What it was:** `ModelRunner::start`'s `dtype` parameter, honored by the
candlelight path to load weights as f16/bf16 for memory/speed.
**Fuel today:** `dtype` is accepted and IGNORED — Fuel's CPU backend has no
`[F32, BF16, F32]` matmul kernel and always loads f32 (confirmed in the Fuel
`ModelRunner::start`'s own pre-existing doc comment, not something this
removal changed).
**View:** pre-existing Fuel limitation, not newly caused by this removal —
recording it here only because it is now the ONLY path, so the limitation is
no longer avoidable by the candlelight fallback. Not immediately my call to
size; flagging for the PM to route.

### 4. AWQ 4-bit quantized inference (Marlin kernels)
**What it was:** `AwqQwen3` (`src/model/awq_qwen3.rs`, ~1115 lines after
slice 1/2's fixes) + Marlin CUDA kernels (`src/backend/marlin.rs`, ~527
lines) + `src/loaders/awq.rs` — a complete, correct (per AWQ slice 1/2's
red-first tests) AWQ loader that this lane was actively validating end-to-end
against a real checkpoint (`Orion-zhen/Qwen3-0.6B-AWQ`) when this task
arrived.
**Used by:** `loaders::load_awq_llama` had **zero production callers**
(confirmed before this removal, HANDOFF.md 2026-10-22) — tests and dead
examples only. So this is "a complete, correct, but never-wired capability",
not a production regression.
**Fuel/mlmf today:** per the portfolio PM's earlier measurement (2026-10-22,
cross-project line counts): `fuel_transformers`' `quant_w4a16` (~745 lines),
`baracuda`'s Marlin (~2.5k lines), `mlmf-awq` (~975 lines) all exist
elsewhere in the portfolio. **Not yet verified from Lightbulb's side** whether
any of them is AWQ-format-compatible (same `qweight`/`qzeros`/`scales`
packing, same `gemm` version) or just a different quant scheme under a
similar name — that check is real work, not done here.
**View:** likely superseded rather than a gap to fill, given three other
projects already have quantization code in this space — but I have not
verified parity, so I'm not calling it closed. The WSL2 fixture + the
independent-reference comparison methodology built for this (see
`scripts/awq_reference/`, kept per PM direction) validates ANY AWQ
implementation's correctness, including whichever one (if any) eventually
replaces this.
**Kept, not deleted:** `scripts/awq_reference/` and its fixture generation —
PM, 2026-10-22: "Keep the WSL2 fixture: it validates ANY AWQ path, including
Fuel's after the migration."

### 5. Speculative decoding (draft/target model verification)
**What it was:** `SpeculativeModel` trait (candlelight Tensor-typed) +
`SpeculativeDecoder` (the accept/reject algorithm), `src/engine/
speculative.rs`. `SpeculativeConfig`/`SpeculativeStats` (candlelight-free)
survive.
**Used by:** nothing outside this module (confirmed: zero callers of
`SpeculativeDecoder`/`SpeculativeModel` anywhere in src/tests/examples,
even before removal) -- `SpeculativeConfig` is read by `init.rs` but never
fed to a `SpeculativeDecoder`.
**Fuel/mlmf today:** not checked.
**View:** looks like it was never wired into the live serving path at all
(not even the candlelight one) -- a designed-but-unintegrated feature, not a
regression from this removal. Lower priority than the admin-endpoint gaps
above; flagging rather than deciding.

### 6. Model pruning (Wanda 2:4 structured + tail pruning)
**What it was:** `WandaScorer` (activation-aware weight scoring), `TailPruner`
(layer removal), `PruningLoader`/`PruningValidator` (manifest application +
perplexity validation), `PruningManifest`/`PruningMask` (Tensor-typed data),
`gguf_application.rs` (applying a manifest to a GGUF file). ~1800 lines total.
**Used by:** nothing — confirmed zero callers of any of these anywhere in
src/tests/examples, even before removal. `apply_manifest_to_gguf`'s only
references were its own file and a bare `pub use` re-export that nothing
called.
**Fuel/mlmf today:** not checked.
**View:** like speculative decoding, this looks designed-but-never-integrated
rather than a regression — M5's acceptance criteria (2:4 structured speedup,
tail-pruning accuracy) describe a feature that was built but never wired to
any model-loading or serving path. Flagging rather than deciding; lower
priority than the admin-endpoint gaps.

### 7. Norm Tweaking (post-quantization accuracy recovery)
**What it was:** `NormTweaker`/`apply_norm_tweaking` (`src/quantization/
norm_tweaking.rs`) — LayerNorm/RMSNorm calibration for 1.5-3% accuracy
recovery on quantized models (M3.10).
**Used by:** nothing — zero external callers, confirmed before removal.
**Fuel/mlmf today:** not checked.
**View:** same pattern as speculative decoding and pruning: designed per the
module's own doc comment ("Post-Quantization Optimizations") but never wired
to any quantization or serving path. Flagging, not deciding.

### 8. Prefix KV cache (candlelight implementation)
**What it was:** `PrefixKvCache`/`PrefixKvEntry` (`src/cache/prefix_cache.rs`)
— hash-and-reuse caching for shared prompt prefixes, Tensor-typed KV storage.
`PrefixCacheConfig`/`PrefixCacheStats` (plain) survive.
**Used by:** nothing live — `src/model_fuel/policies.rs`'s own doc comment
already called this implementation "candle-typed AND the wrong model" for
Fuel's prefix-matching design before this removal; it only ever referenced
the type in discussion, never constructed one.
**Fuel/mlmf today:** `model_fuel/policies.rs` describes (per its module doc)
a content-addressed prefix-matching layer already built on donor-must-stay-
open copy-on-write — this is the LIVE mechanism gap #1's cache_stats
ultimately should read from, per that file's own design notes. Not yet
checked whether it's wired to anything that writes `PrefixCacheStats` today.
**View:** superseded by design (Fuel's own prefix-matching layer was always
meant to replace this), not a gap to fill with the old implementation.

### 9. Tiered KV-cache demotion (GPU -> RAM -> disk)
**What it was:** `TieredStorageManager`/`DemotedSegment`/`StorageTier`
(`src/cache/tiered_storage.rs`) — demoting evicted KV segments through
RAM then disk, with `<RETRIEVE:key>`-triggered promotion back to GPU.
`DiskStore`/`FileDiskStore` (the candlelight-free byte-level disk backend)
survive — real dependency of `model_fuel/policies.rs`.
**Used by:** nothing live — `model_fuel/policies.rs` only referenced the
Tensor-typed manager/segment in its own design-notes doc comment.
**Fuel/mlmf today:** not checked whether an equivalent GPU/RAM/disk tiering
exists elsewhere in the portfolio.
**View:** likely designed-ahead-of-integration like the pruning/speculative
entries above, though this one is less certain — the `<RETRIEVE:key>`
mechanism it mentions sounds like it could tie to the KnowledgeBase feature
elsewhere in this crate. Flagging for the PM to judge rather than guessing.

## Dead code removed, not a capability loss (zero production callers before this removal)

- **`ParallelModelManager`** and its ~16 supporting `src/model/*.rs` files
  (`batch_metadata`, `batched_llama_wrapper`, `chunked_prefill`,
  `custom_attention`, `custom_transformer[_block]`, `decode_state`,
  `fused_kernels`, `fused_rmsnorm`, `kv_tensor`, `mlp_wrapper`,
  `quantizable_linear`, `speculative_adapters`) — all reachable only through
  each other and `ParallelModelManager` itself, never from outside
  `src/model/` except `model_runner.rs`'s now-removed `not(fuel-engine)`
  branch. `src/model/batched_llama.rs` was already orphaned (not even
  declared in `model/mod.rs`) before this removal.
- **`run_jobs`/`EngineModel` and its ~10 helper functions**
  (`src/engine/model_runner.rs`) — confirmed by that file's OWN pre-removal
  comment: "`run_jobs` itself survives only as the candlelight
  (`ParallelModelManager`) path's job loop." Fuel's path goes through
  `scheduled_llama_engine::run_fuel_engine`/`FuelSchedulerDriver` entirely
  separately and never called `run_jobs`.
- **`local_llama_generate`/`load_local_llama`/`load_gguf_llama`** (`src/
  lib.rs`, `src/loaders/mod.rs`) — `load_gguf_llama` was already documented
  as "the DEAD path" in `tests/gguf_architecture_refusal.rs`'s own module
  doc, superseded by `crate::gguf::require_llama_architecture`/
  `require_qwen3_architecture` (already live on Fuel's path). The whole
  `src/loaders/` module had zero real (non-doc-comment) external callers.
- **Multi-GPU tensor/pipeline parallelism** (`src/multi_gpu/
  {distributed_cache,pipeline_parallel,tensor_parallel}.rs`, M3.6) — zero
  external callers even before removal; never wired into any of the 8 known
  multi-GPU bail sites. UNVERIFIED BY HARDWARE regardless (this box has one
  GPU — HANDOFF.md). `config.rs`/`topology.rs` survive (candlelight-free).
  **Not yet checked** whether Fuel/mlmf has multi-GPU support — low priority
  to check given no production caller existed either way.
- **Deleted test files** that existed only to exercise the dead code above:
  `tests/integration_local_model.rs`, `tests/gguf_architecture_refusal.rs`,
  `tests/parallel_model_manager_integration.rs`, `tests/
  multi_gpu_validation.rs`. Also 30 of 34 `examples/*.rs`, and the only
  `[[bench]]` target (`benches/batched_inference_benchmark.rs`, ParallelModelManager
  speedup claims).

### 10. Parallel KV cache builder + compression (the engine's biggest candlelight cluster)
**What it was:** `ParallelCacheBuilder`/`ParallelKvCache`/`IndicesAndMask`/
`CacheUsageInfo` (`src/cache/parallel_cache_builder.rs`, ported from an
upstream fork, own SPDX header), plus its two support modules
`kv_compression.rs` (KIVI/low-rank/relationship-aware KV compression) and
`tensor_codec.rs`. Also deleted `tests/parallel_kv_cache.rs` and
`tests/kv_append_perf_probe.rs`, which existed only to test this cluster.
**Used by:** `ParallelModelManager` (deleted in the same pass) was its ONLY
production caller, confirmed via `git show HEAD~15:src/model/
parallel_model_manager.rs` before removal (`use crate::engine::
{ParallelCacheBuilder, ParallelKvCache, ...}`, `cache_builder: ParallelCacheBuilder`
field).
**Fuel/mlmf today:** `tests/parallel_kv_cache.rs`'s OWN doc comment, written
before this removal, already says it: "The Fuel-side equivalent of the
deleted lifecycle is `fuel::kv_block_pool`
(`open`/`append`/`filled_tokens`/`discard`/`evict_blocks`/`splice`), driven
by `lightbulb::model_fuel::policies`. That path is block-granular rather
than a fixed `batch × context` rectangle, so it is a strict superset — it
can express prefix sharing and ragged lengths, which the old API could not."
**View:** confirmed superseded, not a gap — this is the one entry in this
document with a pre-existing, in-repo citation saying so, not my own
assessment. `kv_compression.rs`'s KIVI/low-rank compression techniques were
never independently wired to the Fuel path either; if anyone wants that
specific accuracy/memory tradeoff on Fuel's KV cache, that would be new work
against `fuel::kv_block_pool`, not a restoration of this code.

### 11. GGUF serving census lost its measurement target (test coverage gap, not a Fuel gap) — RE-CREATED 2026-10-22
**What it was:** `tests/gguf_serving_census.rs` — measured how many corpus
checkpoints `ParallelModelManager::load_gguf` ("the reader that is actually
reachable", per its own doc) could load, and why each failure happened. This
is the source of HANDOFF's "13 weighted files, 10/13 load, 3/13 refuse"
figures.
**Deleted, not ported**, because repointing it correctly requires matching
Fuel's GGUF loader's (`model_fuel::loader_gguf`/`loader_gguf_qwen3`) actual
calling convention and failure-classification shape (architecture refusal
vs. dtype refusal vs. something else) — real work, not a mechanical
candlelight swap, and getting the classification wrong would produce a
confidently-wrong census, which is worse than no census.
**RE-CREATED as `tests/gguf_e2e_weight_load_census.rs`**, calling Fuel's
real loaders directly: 9 of 11 weighted corpus files load end-to-end
(including all 3 IQ-coded ones — the dtype gap closure holds at
weight-decode time too), 2 fail on a real, different, Fuel-side gap
(`llama.attention.head_count_kv` not defaulted for pre-GQA exports). See
`ROADMAP.md`'s GGUF corpus section for the full writeup. Also deleted for the
same reason: `tests/flash_attention_tests.rs` (candlelight's FlashAttention vs.
manual-attention parity — no Fuel equivalent exists to compare against, so
there is nothing to port this to, only to re-derive if FlashAttention parity
matters on the Fuel path) and `tests/enhanced_correctness_tests.rs` (493
lines of `ParallelModelManager` correctness regression tests — batch
isolation, cross-batch-size consistency, etc. The properties are generic and
worth re-deriving against Fuel's model manager, but that is a real port, not
a mechanical one).

## Resolution status

Every file originally importing candlelight has been resolved one way or
another: deleted as dead weight (the large majority — see above), repointed
to an API-compatible Fuel type (`src/gguf/mod.rs`'s `Value`/`TensorInfo`/
`Content`, confirmed field-compatible with `fuel_loaders::quantized::
gguf_file`), or split to keep only the candlelight-free parts a live Fuel
caller actually needs (`src/cache/{prefix_cache,tiered_storage}.rs`,
`src/engine/speculative.rs`, `src/multi_gpu/config.rs`'s `ShardingStrategy`).
`cargo check --lib --tests --examples --features fuel-engine` is clean (zero
candlelight references, zero compile errors) as of this writing.

`src/hardware/mod.rs` needed no change — its one "candlelight" hit was
already a comment, not a dependency.
