# Fuel/MLMF migration — PR sequence

Companion to `docs/fuel-migration-manifest.md` (what moves where) and `docs/candle-to-fuel-translation-guide.md` (how serving-layer code changes shape). This orders the work into PRs that each build, pass tests, and are independently deployable — no PR should leave the tree in a state that only compiles with a later PR also applied. **Research + plan only; no code from this document until the PM approves it.**

Ground rule carried from HANDOFF: **the GGUF reader move to mlmf goes last** — mlmf is still designing its own mmap-accessor prerequisite for it (manifest §6), and lightbulb's current GGUF reader is a real, independently-tested, working zero-copy path. Don't destabilize the one format-reading path that works while mlmf's side is still being decided.

Each PR below is scoped so `cargo build`/`cargo test` (default features) and `cargo build/test --features fuel-engine` both stay green at every step — the two paths currently coexist (`src/model` on candlelight, `src/model_fuel` on fuel), and this sequence keeps them coexisting until the PR that explicitly flips the default.

---

## Phase A — zero-coupling deletes and swaps (no cross-file dependency, safe to do in any order, listed cheapest-first)

**PR 1 — `multi_gpu::topology` → `fuel-hardware`.**
Delete `src/multi_gpu/topology.rs`'s hand-rolled `DeviceTopology::discover()` (hardcoded 80GB guess, `// TODO: Candle API for memory info`), replace with `fuel-hardware::probe::ProbeReport::probe_all()`. No other file depends on this one's internals beyond `multi_gpu/config.rs`'s consumption of its output shape — update that call site in the same PR. Manifest ref: §3. Independently deployable: this subsystem isn't on the request-serving hot path yet (multi-GPU isn't wired to the fuel-engine feature at all today).

**PR 2 — `model/quantizable_linear.rs` → `fuel-nn::modules::quantizable_linear`.**
Direct match, single-file delete. Manifest ref: §1.

**PR 3 — `model/fused_rmsnorm.rs` → `fuel-nn::modules::norm::RmsNorm`.**
Delete the file; in the same PR, `grep` the whole tree for any other `candle-layer-norm` consumer before removing that git dependency + the `[patch]` block in `Cargo.toml` (manifest §1 flags this check explicitly — don't drop the patch block on the assumption this was the only user without verifying).

**PR 4 — `src/lib.rs`'s `hello_generate`/`local_llama_generate` + `src/loaders/mod.rs` (whole file) → delete.**
Both fully superseded by `src/model_fuel/{generate.rs,loader.rs,loader_f32.rs,loader_gguf.rs}` already in-repo. Manifest §7. Confirm no other file calls `crate::loaders::load_local_llama` before deleting (`loaders/awq.rs`, kept until Phase C, must not import from `loaders/mod.rs`'s Llama-specific code — check that boundary in this PR).

**PR 5 — `cache/parallel_cache_builder.rs`, `cache/prefix_cache.rs` → delete, wire `fuel-inference::prefix_cache`.**
`parallel_cache_builder.rs` is already superseded by `src/model_fuel/policies.rs` in-repo (pure delete). `prefix_cache.rs` needs `fuel-inference` added as a real `src/` dependency (today it's only a test-only dependency per the translation guide §1 — this PR is also where that Cargo.toml change happens) and the call sites repointed at `PrefixCache::{new,lookup,insert}`. Translation guide §1.

**PR 6 — `engine/speculative.rs` → wire `fuel-inference::speculative`.**
Bigger shape change than PR 5: fuel's `verify_draft` takes realized host `Vec<f32>` logprobs, not live `Tensor`s (translation guide §2) — the caller must realize logits before calling verify, so this PR touches whatever currently calls `SpeculativeModel::forward_logits` as well as the trait itself. Do this after PR 5 so `fuel-inference` is already a real dependency, not because of a code dependency between the two.

**PR 7 — `hardware/mod.rs::to_device()` translation.**
Swap `candlelight::core::Device` (enum) for `fuel::Device` (struct, `Device::cpu()`/`cuda_backend::device_if_available()`/`metal_backend::device_if_available()`/`vulkan_backend::new_device()`) per translation guide §3. This one is a **real capability gain** (fuel has working Vulkan; candlelight's Vulkan arm was always a CPU-fallback stub) — worth calling out in the PR description since it changes observable behavior on Vulkan-capable hardware, which per CireSnave's own versioning rule (portfolio CLAUDE.md §9) means this PR needs a version bump, likely minor (new capability, not breaking — verify against the "does a conforming caller's observed outcome change for a request it could already legally send" test before deciding major vs minor).

**PR 8 — `memory/{estimate,speculative,utils}.rs` DType translation.**
Replace hand-rolled `match dtype {...}` byte-width lookups with `fuel::DType::size_in_bytes()`. Translation guide §4 notes this *fixes* a latent bug (today's default-arm fallback silently mis-sizes `I8`/`I16`/`I32`) — flag in the PR description as a correctness fix, not just a mechanical swap, since it may change memory-estimate output for any caller already passing those dtypes (versioning judgment call, same test as PR 7).

**PR 9 — `tools/mod.rs` test-only import swap.**
Trivial, bundle with whichever earlier PR touches the module `tools/mod.rs`'s tests actually exercise, or land standalone — no risk either way.

---

## Phase B — multi-GPU parallelism (each strategy independently swappable; `Hybrid`/`PipeDream`/distributed-cache storage explicitly NOT included, see blockers below)

**PR 10 — `multi_gpu::tensor_parallel` Column/Row → `fuel-parallel::tensor_parallel::{ColumnParallel,RowParallel}`.**
Depends on PR 1 (fuel-hardware topology already swapped in, so this PR's device-group construction has a consistent topology source). `ShardingStrategy::Hybrid`'s bail stays as-is — manifest §3 confirms no fuel destination exists for it yet; do not invent one in this PR.

**PR 11 — `multi_gpu::pipeline_parallel` GPipe → `fuel-parallel::pipeline_parallel::GPipe`.**
Independent of PR 10 (different sharding axis) — can land before, after, or in parallel with it. `PipeDream`/`Interleaved1F1B` bails stay, same reasoning as PR 10's `Hybrid`.

**PR 12 — `model/parallel_model_manager.rs`, `model/batched_llama*.rs`, and the rest of `model/custom_transformer*.rs`/`custom_attention.rs`/`batch_metadata.rs`/`chunked_prefill.rs`/`decode_state.rs`/`kv_tensor.rs`/`mlp_wrapper.rs` → delete, confirm `model_fuel/batched.rs` covers every deleted code path.**
This is the largest single PR in the sequence by line count (manifest §1: ~6,700 candlelight lines across 9 files) but the replacement (`model_fuel/batched.rs`) already exists and is already tested under `--features fuel-engine`. The PR's job is verifying coverage, not writing new logic — run the existing `model_fuel` test suite plus any acceptance test in `tests/fuel_engine_http.rs` (per HANDOFF: last run by hand, re-run it as part of this PR's verification, don't trust a stale 2026-08-08 result) before deleting the candlelight-path files. This PR is also the natural point to decide whether the default (non-`fuel-engine`) build path gets deleted here or in a later, separate "flip the default" PR — recommend the latter (Phase D below), so this PR stays a pure move/delete without also changing which path serves production traffic.

**Blocked, not sequenced — needs a PM/architecture ruling first, per manifest §3's open items:**
- `ShardingStrategy::Hybrid`, `PipelineStrategy::{PipeDream,Interleaved1F1B}` — no fuel destination exists; open whether these are real requirements.
- `multi_gpu::distributed_cache` sharded/hybrid cache **storage** — confirmed fuel gap (fuel only has coordination, no storage layer); needs a ruling on whether this gets built in fuel or lightbulb before any PR touches it.

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
| A — zero-coupling deletes/swaps | 1–9 | Nothing; can start immediately once approved |
| B — multi-GPU parallelism | 10–12 | PR 1 (topology) for PR 10; `Hybrid`/`PipeDream`/distributed-cache storage explicitly excluded pending a ruling |
| C — quantization/pruning/LoRA/AWQ | 13–17 | PR 15 blocked on fuel-side dispatch work; PR 16 blocked on mlmf-side crate promotion |
| D — flip default | 18 | All of A–C |
| E — GGUF reader + hub.rs | 19–20 | mlmf's mmap-accessor prerequisite (19); PM ruling on `hub.rs` (20) |
