# Fuel/MLMF migration — PR sequence

Companion to `docs/fuel-migration-manifest.md` (what moves where) and `docs/candle-to-fuel-translation-guide.md` (how serving-layer code changes shape). This orders the work into PRs that each build, pass tests, and are independently deployable — no PR should leave the tree in a state that only compiles with a later PR also applied. **Research + plan only; no code from this document until the PM approves it.**

Ground rule carried from HANDOFF: **the GGUF reader move to mlmf goes last** — mlmf is still designing its own mmap-accessor prerequisite for it (manifest §6), and lightbulb's current GGUF reader is a real, independently-tested, working zero-copy path. Don't destabilize the one format-reading path that works while mlmf's side is still being decided.

Each PR below is scoped so `cargo build`/`cargo test` (default features) and `cargo build/test --features fuel-engine` both stay green at every step — the two paths currently coexist (`src/model` on candlelight, `src/model_fuel` on fuel), and this sequence keeps them coexisting until the PR that explicitly flips the default.

---

## ⚠️ Resequencing (2026-10-01, PM-approved, not a scope/risk change — see HANDOFF.md)

Original PRs 2, 3, 5, 6, and the `loaders/mod.rs` half of PR 4 turned out NOT to be independent, contrary
to how they were scoped below. Found while implementing PR 1: the manifest's "does an equivalent exist
at the destination" check didn't verify "what does the EXISTING caller need back" — every one of these
items is a type/trait that candlelight-typed caller files (`custom_attention.rs`, `custom_transformer.rs`,
`custom_transformer_block.rs`, `mlp_wrapper.rs`, `parallel_model_manager.rs`, `batched_llama*.rs`,
`speculative_adapters.rs`) construct from candlelight `VarBuilder`s or hold as fields — you cannot swap
the wrapped type without those callers' weight-construction code already being on fuel, which is PR 12's
job, not a standalone PR. **All five are MOVED into an expanded PR 12, below** — see that entry for the
full real caller list per item. PRs 1, 7, 8, 9, and the `lib.rs`-only half of PR 4 remain independent,
confirmed by checking every production caller (not just the destination), and PR 1 and PR 7 are done
(lightbulb#109, lightbulb#110).

---

## Phase A — zero-coupling deletes and swaps (confirmed zero production callers beyond the file's own tests, checked per-item — not just "a fuel destination exists")

**PR 1 — `multi_gpu::topology` → `fuel-hardware`. DONE — lightbulb#109.**
Replaced the hand-rolled `DeviceTopology::discover()` (hardcoded 80GB guess, no real P2P/interconnect) with real `fuel::probe::ProbeReport`/`fuel::topology::SystemTopology` measurements. Manifest ref: §3.

**PR 4 — `src/lib.rs`'s `hello_generate`/`local_llama_generate` → delete.**
Confirmed zero callers anywhere outside `src/lib.rs` itself (re-checked specifically for this resequencing — the `loaders/mod.rs` half of the original PR 4 is NOT safe, see PR 12). Superseded by `src/model_fuel/generate.rs`. Manifest §7 (partial — the manifest's own §7 bundled this with `loaders/mod.rs`, which this resequencing splits apart).

**PR 7 — `hardware/mod.rs::to_device()` translation. DONE — lightbulb#110.**
`candlelight::core::Device` (enum) → `fuel::Device` (struct). Confirmed zero production callers before touching it (`model_selection.rs` uses `InferenceBackend` as a value, never calls `.to_device()`). Real capability gain: fuel has working Vulkan, candlelight's Vulkan arm was always a CPU-fallback stub — version-bump judgment call per CireSnave's breaking-version test, not yet made (PR not yet merged).

**PR 8 — `memory/{estimate,speculative,utils}.rs` DType translation.**
Replace hand-rolled `match dtype {...}` byte-width lookups with `fuel::DType::size_in_bytes()`. Confirmed zero external callers of `WeightMemory`/`ActivationMemory`/`KvCacheMemory` (checked for this resequencing). Translation guide §4 notes this *fixes* a latent bug (today's default-arm fallback silently mis-sizes `I8`/`I16`/`I32`) — flag in the PR description as a correctness fix, not just a mechanical swap.

**PR 9 — `tools/mod.rs` test-only import swap.**
Trivial, bundle with whichever earlier PR touches the module `tools/mod.rs`'s tests actually exercise, or land standalone — no risk either way.

---

## Phase B — multi-GPU parallelism + the candlelight model-path deletion (expanded 2026-10-01)

**PR 10 — `multi_gpu::tensor_parallel` Column/Row → `fuel-parallel::tensor_parallel::{ColumnParallel,RowParallel}`.**
Depends on PR 1 (fuel-hardware topology already swapped in, so this PR's device-group construction has a consistent topology source). `ShardingStrategy::Hybrid`'s bail stays as-is — manifest §3 confirms no fuel destination exists for it yet; do not invent one in this PR.

**PR 11 — `multi_gpu::pipeline_parallel` GPipe → `fuel-parallel::pipeline_parallel::GPipe`.**
Independent of PR 10 (different sharding axis) — can land before, after, or in parallel with it. `PipeDream`/`Interleaved1F1B` bails stay, same reasoning as PR 10's `Hybrid`.

**PR 12 (EXPANDED 2026-10-01) — the candlelight model-path deletion. Absorbs original PRs 2, 3, 5, 6, and the `loaders/mod.rs` half of PR 4.**
Not independently sub-deployable in the old sense — these all share real callers with each other and with the files this PR already deleted. One coordinated PR (or a tightly internally-ordered stack of commits within it, verified green at each step), not N parallel-mergeable PRs:
- `model/quantizable_linear.rs` → `fuel-nn::modules::quantizable_linear` (was PR 2). Real callers: `custom_attention.rs`, `custom_transformer.rs`, `mlp_wrapper.rs` (all deleted by this PR) — their candlelight `VarBuilder`-based construction must move to fuel's `WeightStorage` construction as part of the same change, not before it.
- `model/fused_rmsnorm.rs` → `fuel-nn::modules::norm::RmsNorm` (was PR 3). Real callers: `custom_transformer.rs`, `custom_transformer_block.rs` (both deleted by this PR). Same `candle-layer-norm` dependency/`[patch]`-block check as originally scoped, run it once this PR has actually removed every candlelight consumer.
- `src/loaders/mod.rs` (whole file) → delete (the half of old PR 4 that isn't safe standalone). Real caller: `parallel_model_manager.rs` (deleted by this PR). `loaders/awq.rs` (Phase C, kept) must not import from `loaders/mod.rs`'s Llama-specific code — check that boundary.
- `cache/parallel_cache_builder.rs` → delete, superseded by `src/model_fuel/policies.rs` (was PR 5, narrower half). Real callers found: `custom_attention.rs`, `custom_transformer.rs`, `custom_transformer_block.rs`, `parallel_model_manager.rs`, `batched_llama_wrapper.rs` (all deleted by this PR) — **but also** `speculative_adapters.rs` (not deleted by this PR, see below) **and** `multi_gpu/{distributed_cache.rs,pipeline_parallel.rs}` (not deleted by this PR, PR 10/11's territory). **This PR's deletion of `parallel_cache_builder.rs` is gated on ALL of its callers being gone — do not delete it here if `speculative_adapters.rs` or the two `multi_gpu` files still reference it.** Track as an exit condition, not an assumption.
- `cache/prefix_cache.rs` → wire `fuel-inference::prefix_cache` (was PR 5, other half). Real caller: `parallel_model_manager.rs` (deleted by this PR) — narrower than `parallel_cache_builder.rs`, no known residual caller outside this PR's scope, but re-verify at PR time rather than trusting this note.
- `engine/speculative.rs` → wire `fuel-inference::speculative`, **together with** `model/speculative_adapters.rs`'s own port (was PR 6, now explicitly bundled — `speculative_adapters.rs` is `engine/speculative.rs`'s one real caller and is itself candlelight-entangled, so wiring the trait without also porting its one caller leaves dead/broken code). This is what closes out `parallel_cache_builder.rs`'s remaining caller from the bullet above (minus the two `multi_gpu` files).
- `model/parallel_model_manager.rs`, `model/batched_llama*.rs`, `model/custom_transformer*.rs`, `model/custom_attention.rs`, `model/batch_metadata.rs`, `model/chunked_prefill.rs`, `model/decode_state.rs`, `model/kv_tensor.rs`, `model/mlp_wrapper.rs` → delete, confirm `model_fuel/batched.rs` covers every deleted code path (the original PR 12 content, manifest §1, ~6,700 candlelight lines across 9 files). Run the existing `model_fuel` test suite plus `tests/fuel_engine_http.rs`'s acceptance gate (HANDOFF: last run by hand 2026-08-08, re-run it, don't trust that result) before deleting.
- Exit condition for the WHOLE expanded PR 12: `grep -rln "candlelight" src/model/ src/loaders/mod.rs src/cache/parallel_cache_builder.rs src/cache/prefix_cache.rs src/engine/speculative.rs src/model/speculative_adapters.rs` → zero files. Whether the non-`fuel-engine` default build path itself flips is still Phase D's job (PR 18), not this one's — this PR removes the dead weight, Phase D removes the choice.

**Still blocked, not sequenced — needs a PM/architecture ruling first, per manifest §3's open items (unaffected by the resequencing above):**
- `ShardingStrategy::Hybrid`, `PipelineStrategy::{PipeDream,Interleaved1F1B}` — no fuel destination exists; open whether these are real requirements.
- `multi_gpu::distributed_cache` sharded/hybrid cache **storage** — confirmed fuel gap (fuel only has coordination, no storage layer), confirmed genuinely unscoped by the fuel lane (2026-10-01: an unresolved design fork in fuel's own architecture, not just unsized). Deferred past this whole sequence. `parallel_cache_builder.rs`'s two residual `multi_gpu` callers (above) wait on whatever PR eventually resolves this, not on PR 12.

---

## Phase C — quantization, pruning, LoRA, AWQ/Marlin (mixed destinations, some gated on mlmf)

**PR 13 — `quantization/norm_tweaking.rs`, `pruning/{mod.rs,gguf_application.rs}` → port into fuel.**
Both confirmed genuinely absent from fuel (manifest §4, positive-controlled greps). This is real algorithm-porting work, not a wiring change — scope as a fuel-repo PR (or a fuel-repo change landed first, then a lightbulb PR switching the call site), not a lightbulb-only change. Sequence after Phase B since neither has a dependency on it, just lower urgency.

**PR 14 — `quantization/mod.rs`, `quantization/gguf_ops.rs` → port dequant/format logic into fuel, using `fuel-quantized`.**
Manifest §4 flags this as not yet symbol-compared against `fuel-quantized`'s actual dequant API — this PR's first commit should be that comparison (could shrink to a smaller diff than the file's current size suggests, or could reveal it's not a clean swap). Do this before PR 16 (GGUF reader move) since the GGUF reader move assumes quantization's dependency on `crate::gguf::Content` is already resolved on the fuel side.

**PR 15 — `backend/{marlin.rs,marlin_ffi.rs}`, `model/awq_qwen3.rs`, `loaders/awq.rs` → port into fuel, once fuel's `QuantFormat::{Awq,Marlin,NF4}` dispatch lands.**
**Blocked on fuel-side work** (manifest §1, §2: kernel exists in `fuel-cuda-backend`, dispatcher does not) — this PR cannot start until the fuel lane ships that dispatch glue. Message the fuel lane before scheduling this PR's start date.

**PR 16 — `pruning/name_mapping.rs`, LoRA loading half of `lora/mod.rs` → mlmf, once mlmf promotes `TensorNameMapper` out of its legacy crate.**
**Blocked on mlmf-side work** (manifest §4: mlmf's own migration plan sequences this as its item 7, gated on a crate promotion/rewrite already scheduled on mlmf's side — migrating onto the legacy crate now means migrating twice). Message the mlmf lane before scheduling.

**PR 17 — LoRA runtime-application half of `lora/mod.rs` → `fuel-nn::modules::lora`.**
Independent of PR 16 (different half of the same file, different destination) — can land first if convenient, since `fuel-nn::modules::lora`'s `WeightStorage::WithLoRA` shape is already confirmed matching (manifest §4).

---

## Phase D — flip the default, retire candlelight

**PR 18 — flip `fuel-engine` from opt-in feature to default; delete the `src/model` candlelight path entirely (whatever Phase B/C left of it), delete the `candlelight` dependency and its `[patch]` blocks from `Cargo.toml`.**
This is the version-bump moment the portfolio's breaking-version test (CLAUDE.md §9, already applied twice this lane per HANDOFF) almost certainly classifies as breaking — a caller who was getting candlelight's serving behavior (whatever residual differences Phase B/C didn't already converge) now unconditionally gets fuel's. Requires every prior PR in this sequence to have landed; this is the one genuinely sequential gate in the whole plan. Do not attempt this before `tests/fuel_engine_http.rs`'s acceptance gate (HANDOFF: last run 2026-08-08, per this lane's own prior finding) has been re-run and passed on current fuel `origin/main`, not trusted from memory.

---

## Phase E — the GGUF reader (explicitly last, per HANDOFF)

**PR 19 — `src/gguf/{mod.rs,parser.rs}` → `mlmf-gguf`, once mlmf ships the zero-copy mmap tensor-slice accessor.**
**Blocked on mlmf's own prerequisite** (manifest §6: mlmf's own inventory doc already sequences this correctly — its mmap-source-file primitive needs to reach `mlmf-gguf` before lightbulb switches, or lightbulb takes a real perf regression trading a zero-copy path for a copying one). Do not start this PR until the mlmf lane confirms that prerequisite is done. This also retires `require_llama_architecture`'s hardcoded refusal gate (`src/gguf/mod.rs:2301`) only if the replacement's per-architecture geometry handling is verified at least as strict — HANDOFF's standing warning about this gate producing wrong numbers instead of an error applies with full force to whatever replaces it; do not relax it as a side effect of this move.

**PR 20 (final, only if PR 19 also carries `hub.rs`) — resolve the `hub.rs` open question (manifest §6) before or alongside PR 19.**
`hub.rs`'s HF Hub download logic doesn't cleanly fit either side of the mlmf/fuel split — mlmf's architecture deliberately excludes network I/O. This needs a PM ruling (keep in lightbulb as a selection-adjacent concern, or extend mlmf's charter) before this PR can be scoped at all; listed here only so it isn't forgotten, not because its content is decided.

---

## Summary table

| Phase | PRs | Blocked on |
|---|---|---|
| A — zero-coupling deletes/swaps | 1, 4, 7, 8, 9 | Nothing. **1 and 7 done** (lightbulb#109, #110). Old 2/3/5/6 and `loaders/mod.rs`-half-of-4 MOVED into 12 (2026-10-01 resequencing) |
| B — multi-GPU parallelism + candlelight model-path deletion | 10–12 | PR 1 (topology) for PR 10; PR 12 is now the expanded cluster absorbing old 2/3/5/6 — see its exit condition; `Hybrid`/`PipeDream`/distributed-cache storage explicitly excluded pending a ruling |
| C — quantization/pruning/LoRA/AWQ | 13–17 | PR 15 blocked on fuel-side dispatch work; PR 16 blocked on mlmf-side crate promotion |
| D — flip default | 18 | All of A–C |
| E — GGUF reader + hub.rs | 19–20 | mlmf's mmap-accessor prerequisite (19); PM ruling on `hub.rs` (20) |
