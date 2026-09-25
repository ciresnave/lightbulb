# Fuel port — measured status, 2026-09-24

**Measured at `origin/main` `29cabf4d`** (verified: `git ls-remote origin main` at time of
writing returned `29cabf4dc1383356a1e43852b685ec7f4f75ab2a`, same SHA the PM cited). This is a
status document, not a plan. No port code, no version bump. Every number below is tagged
MEASURED (command + output, or file:line) or ASSUMED.

---

## 0. Correcting the PM's file-count method

MEASURED. The PM's count (`total 123`) is right; a first pass here using
`git ls-files -- 'src/**/*.rs'` gave 112 and looked like a discrepancy. It wasn't — `git`
pathspec `**` glob magic does not match files directly inside `src/` itself (only inside a
subdirectory of it), so it silently dropped 11 top-level files (`src/lib.rs`, `src/main.rs`,
`src/server.rs`, etc.).

```
git ls-files | grep -c '^src/.*\.rs$'                      → 123
git ls-files -- 'src/**/*.rs' | wc -l                       → 112   (undercounts; pathspec bug)
find src -name '*.rs' | wc -l                                → 123   (agrees with the correct method)
```

Positive control: the 11-file gap is exactly the set of `src/*.rs` files with no subdirectory,
confirmed by `comm` between both listings.

**Candlelight/fuel importer counts, corrected:**

```
files under src/ containing "use candlelight|extern crate candlelight|candlelight::"   42
files under src/ containing "use fuel|extern crate fuel|fuel::" (loose substring)      17  ← includes false positives
files that actually `use fuel::...` / `fuel_cuda_backend::...` (crate-level import)    10
```

The loose count (17) is inflated by files that merely mention the word "fuel" in a comment or
cfg string (`fuel-engine` feature checks, PR-comment provenance notes) without importing
anything — e.g. `src/lib.rs` (only `pub mod model_fuel;`), `src/gguf/mod.rs`,
`src/engine/model_runner.rs` (cfg-gates only, imports `crate::model_fuel::...` not the `fuel`
crate). The real importer count is **10**: all 10 live under `src/model_fuel/`, plus
`src/api/chat_template.rs` mentions `fuel_ir::SymId` only in a doc comment (not an import —
excluded on inspection).

So: **123 total, 42 candlelight-importing, 10 fuel-crate-importing** (PM's "12" likely came
from a method close to the loose 17-count, minus a couple of exclusions — the discrepancy is
in method, not in the underlying fact that the port is small). The port has grown by roughly
the same handful of files the PM already flagged as stalled-looking; this doesn't change that
read.

---

## 1. Does `fuel-engine` build and test today?

MEASURED, both commands run to completion just now, full output kept (not head/grep'd):

```
$ cargo check --features fuel-engine
   Checking lightbulb v0.1.0 (C:\Projects\lightbulb)
warning: ... (46 warnings total: 42 in lib, 4 in bin "lightbulb-cli"; all unused-import/
             unused-variable/dead-code/deprecated-fn, none in model_fuel/)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1m 12s
EXITCODE:0
```

```
$ cargo test --features fuel-engine --lib
test result: ok. 712 passed; 0 failed; 22 ignored; 0 measured; 0 filtered out; finished in 7.84s
EXITCODE:0
```

**Both pass, cleanly, today.** This is a stronger position than "should work" — it is a green
build and a green unit-test run with the feature on.

**But this answer needs a caveat the raw numbers hide (feeds directly into Q4):**

```
$ cargo test --lib                     # default features, no fuel-engine
test result: ok. 712 passed; 0 failed; 22 ignored; 0 measured; 0 filtered out; finished in 6.72s
$ diff <(sort test-name-list --features default) <(sort test-name-list --features fuel-engine)
1 line differs: only the "finished in Ns" timing
```

**`--features fuel-engine` changes zero unit tests that run.** `src/model_fuel/*` compiles and
its 62 tests run unconditionally, feature flag or not — `#[cfg(feature = "fuel-engine")]` in
`src/engine/model_runner.rs` gates only which model *serves HTTP traffic*
(`src/engine/model_runner.rs:224,294`), not which unit tests execute. So "cargo test passes
with fuel-engine" is true but is not evidence the feature flag does anything to test
selection — see Q4 for what *is* gated.

---

## 2. What is parity, concretely? (the work-plan list)

MEASURED by reading `src/model_fuel/` against `src/model/` + `src/loaders/` + `src/multi_gpu/`.
This is not exhaustive prose-scanning — it's a direct diff of what each path's loader/runner
can construct and run.

**Fuel path today (`src/model_fuel/`, all file:line current):**

- One architecture, hardcoded: `fuel::lazy::LlamaModel` via
  `FuelEngineModel::load` (`src/model_fuel/engine_model.rs`). No architecture dispatch table —
  candlelight's `awq_qwen3.rs`, `custom_transformer*.rs` have no fuel counterpart.
- **SafeTensors only, f32 only.** `src/engine/model_runner.rs:322-338`: `.gguf` paths are
  explicitly refused with `"fuel-engine does not yet support GGUF models... Quantized/GGUF
  support is pending `impl FuelDecoder for Llama3Model`"`. The `dtype` parameter to
  `ModelRunner::start` is accepted and silently ignored (`model_runner.rs:304-310`,
  `model_runner.rs:288-293` doc) — always loads f32, by design (accumulation-preserving-cast
  reasons documented in `src/model_fuel/mod.rs:24-29`), not as a stopgap with a plan.
- **No AWQ / quantized-weight inference.** `src/loaders/awq.rs` and
  `src/model/quantizable_linear.rs` (candlelight) have no fuel equivalent — MEASURED via
  `grep -rn "awq\|AWQ\|quantiz" src/model_fuel/`: the only hits are doc comments in
  `decoder.rs`/`policies.rs` stating quantized support is *pending*, and one already-fixed
  item (`policies.rs:28`, grouped quantization now returns `Err` instead of `todo!`).
- **No speculative decoding wired to a real draft model.** `grep -rln speculative
  src/model_fuel/` hits only `mod.rs`'s prose; `src/model/speculative_adapters.rs`
  (candlelight) has no fuel counterpart.
- **No multi-GPU model parallelism.** `grep -rln "multi_gpu\|ParallelModelManager"
  src/model_fuel/` hits only `batched.rs` and `engine_model.rs`, and in both cases it's the
  *batching* code (multiple sequences on one device), not `src/multi_gpu/`'s tensor/pipeline
  parallelism across devices — `src/model/parallel_model_manager.rs` has no fuel counterpart at
  all.
- **No LoRA.** `grep -rln "lora\|LoRA" src/model_fuel/ src/model/` — zero hits in
  `model_fuel/`, so this is missing from *both* paths, not a fuel-specific gap; not counted in
  the port's work plan.
- **The acceptance gate that WOULD prove HTTP-level Llama parity has never been run in CI**
  (see Q4) — so even the one architecture Fuel does implement has no continuously-verified
  claim behind it, only a manually-run one last recorded 2026-08-08.

**So the parity list, in priority order a port plan would use:**
1. GGUF/quantized-weight loading (`impl FuelDecoder for Llama3Model`, already named as the next
   step in `decoder.rs:15`)
2. Architecture dispatch beyond Llama (AWQ-Qwen3, custom transformer variants)
3. AWQ / general quantized inference
4. Multi-device model parallelism (tensor/pipeline — separate from Fuel's own batching)
5. Speculative decoding
6. Continuous (CI) verification of the one path that does exist, not just manual runs

This list **is** the work plan the PM asked for; nothing above is a plan for how to build any
of it, only what is missing today.

---

## 3. Is multi-GPU real?

MEASURED just now, same eight sites `ROADMAP.md`'s VERIFIED STATUS block named on 2026-09-01/02
(line numbers shifted ~8 lines from unrelated edits since; same functions, same file):

```
src/multi_gpu/distributed_cache.rs:173   CacheSyncStrategy::Sharded  → bail!("Sharded cache strategy not yet implemented")
src/multi_gpu/distributed_cache.rs:177   CacheSyncStrategy::Hybrid   → bail!("Hybrid cache strategy not yet implemented")
src/multi_gpu/pipeline_parallel.rs:361   PipelineStrategy::PipeDream        (execute)              → bail!(...)
src/multi_gpu/pipeline_parallel.rs:365   PipelineStrategy::Interleaved1F1B  (execute)              → bail!(...)
src/multi_gpu/pipeline_parallel.rs:422   PipelineStrategy::PipeDream        (execute_with_model)   → bail!(...)
src/multi_gpu/pipeline_parallel.rs:426   PipelineStrategy::Interleaved1F1B  (execute_with_model)   → bail!(...)
src/multi_gpu/tensor_parallel.rs:222     ShardingStrategy::Hybrid   (constructor) → bail!("Hybrid sharding strategy not yet implemented")
src/multi_gpu/tensor_parallel.rs:248     ShardingStrategy::Hybrid   (forward)     → bail!("Hybrid sharding strategy not yet implemented")
```

**All eight still bail today, unchanged in substance since the 2026-09-02 fix landed.**
Confirmed each enum (`CacheSyncStrategy`, `PipelineStrategy`, `ShardingStrategy`) is still
`pub` with these variants still selectable (`distributed_cache.rs:8`, `pipeline_parallel.rs:110`,
`tensor_parallel.rs:6`).

What the 2026-09-02 work *did* fix, and remains fixed (re-checked, not re-derived — this part
is ASSUMED stable from the roadmap's own record, not independently re-verified line-by-line
today): `CacheSyncStrategy::Replicated` went from silently returning `Ok(())` while writing
nothing to an honest `Err` naming the missing write plan
(`distributed_cache.rs:166-168`, confirmed present in the same read above); the two
non-bailing tensor-parallel strategies (`ColumnWise`/`RowWise`) had their `#[ignore]` removed
and a broadcast-add bug fixed; `DeviceTopology::discover()`'s non-termination and
`recommend_strategy`'s two panics were fixed.

**Net: 8 of the original 8 bail sites are still bails. 3 of 3 `CacheSyncStrategy` variants are
still 0-of-3-implemented (all honest now, none working). `ShardingStrategy` is 2-of-3 working
(Column/Row), `PipelineStrategy` is 1-of-3 working (GPipe only).** A PCIe switch with several
heterogeneous GPUs — the hardware this is being bought for — is exactly the configuration that
would first reach `Hybrid` sharding or `PipeDream`/`Interleaved1F1B` scheduling as a natural
choice for heterogeneous devices, and those are precisely the paths that still fail at runtime.

---

## 4. What is the test suite's state, and does it cover the fuel path?

MEASURED, three separate signals, none inferred from the others:

**a) `cargo test --lib` (unit tests, both feature sets) — already shown in Q1: 712 passed, 0
failed, 22 ignored, byte-identical test list with or without `fuel-engine`.** 62 of those 712
are in `model_fuel::*` (`grep -c "^test model_fuel::"` on the fuel-engine run output = 62); of
those, most that touch a real checkpoint are `#[ignore]`d (11 of the 62, needing the TinyLlama
snapshot). So the fuel path's *unit-level* logic (batching admission, KV-pool policies, RoPE,
paged-attention-vs-dense oracle checks) **is exercised continuously**, on every default-feature
CI run, without the flag.

**b) `cargo test --tests` (integration suites) — read from `.github/workflows/ci.yml:133-172`,
which carries its own measured comment block, re-derived by that file's author across two CI
runs (32498482991, 32500847148) and quoted here rather than re-run, since re-running would only
reproduce the same documented number:**

```
812 passed, 0 failed, 66 ignored, 30 suites
  15 suites ran something
   7 compiled to an empty binary (no unit tests — src/main.rs, two src/bin targets — expected)
   8 compiled but every test in them was #[ignore]'d
```

**Four integration files are `#![cfg(feature = "fuel-engine")]`-gated at file scope and do not
even compile under default CI features**, let alone run:
`tests/api_result_metadata.rs`, `tests/chat_template_e2e.rs`, `tests/fuel_engine_http.rs`,
`tests/gpu_paged_vs_contiguous.rs`. Two of those four — `chat_template_e2e` and
`fuel_engine_http` — are named in the CI file itself as *"THE BEHAVIOURAL ACCEPTANCE GATES for
the chat-template / GGUF work"* and *"the tests most worth running"*, and default CI runs zero
of them.

**c) `cargo test --features fuel-engine --tests --no-run`** (`.github/workflows/ci.yml:193-197`)
compiles those four files (and their 9 tests) so they can't silently bit-rot into
non-compiling state, but **`--no-run` never executes a single one.** Confirmed by reading
`tests/fuel_engine_http.rs:79-95` directly: its own file header states *"THIS FILE IS NEVER RUN
BY CI, BY DECISION"* (CireSnave ruled 2026-08-27: no GPU CI runner, a cost decision) and that
the file's claim *"is verified by a person running it on demand, and by nothing else."* Its two
tests (`fuel_runner_serves_a_coherent_completion_over_http`,
`fuel_runner_serves_a_default_temperature_completion`) are the only tests in this repo that
drive an HTTP request through the real router into a real `FuelEngineModel` and assert on
decoded text content — i.e. the only thing that would catch a wiring bug (transposed
projection, wrong RoPE base, wrong norm placement) that still returns 200 with plausible-shaped
garbage. **Last actually run and recorded: 2026-08-08**, per the comments in that file
(6-trial EOS-rate table at lines 379-384) — 47 days before this document, at a different `main`.

**Answer to Q4 directly: the 643→712-test suite exercises the fuel path's unit-level logic on
every CI run, but the one test that would prove end-to-end HTTP-served-Llama-produces-correct-
text has not run in CI ever, and was last run by hand 47 days ago.** A green CI board today says
nothing about whether that specific claim (Fuel serves a coherent completion) still holds — it
was true 47 days and an unknown number of Fuel `origin/main` commits ago, and nobody has
re-checked it since. Positive control for "would a break here be caught": `model_fuel_golden.rs`
(Tier 3, regression-only against Lightbulb's own past output, not an independent oracle) *does*
run in CI-adjacent form for the cheap checks, but its own doc header states plainly it "answers
only 'is this the same?'", not "is this right?" — it would not have caught the class of bug
`fuel_engine_http.rs` exists to catch.

---

## 5. Fuel import classification (three buckets, per PM's follow-up task)

MEASURED. All 10 fuel-importing files live under `src/model_fuel/`. **Zero of them write
`fuel_core::` directly** — every import goes through the `fuel` facade crate
(`use fuel::...`, `use fuel_cuda_backend::CudaDevice` in one dev-path). Positive control: `grep
-rn "fuel_core::" src/` → 0 hits; the same query against `fuel_inference::` phrasing in
`src/model_fuel/mod.rs` doc comments *does* hit, confirming the grep isn't silently broken.

Read `fuel/Cargo.toml` (fetched via `gh api repos/ciresnave/fuel/contents/fuel/Cargo.toml` at
today's fuel `main`, not the local `C:\Projects\fuel` checkout — that checkout is the one
`src/backend/marlin_ffi.rs:91` already warns is measured 65 commits stale, so it was not read
for this): `fuel` is a **facade crate**, re-exporting fuel-core's (and now fuel-transformers')
public surface 1:1, enforced by a `cargo public-api` byte-identical-surface gate per feature
set. Its own header says this exists precisely so a consumer's `use fuel::…` "resolves
identically" across the dissolution. **This means Lightbulb is already positioned correctly on
the letter of the PM's rule** ("do not write new `fuel_core::` paths") — it never did.

That does not settle the substance, though: the facade's *promise* is path-stability, not that
every symbol's underlying crate is already decided. Classifying what we actually import,
checked against `fuel-core/src/lib.rs`'s current module list (fetched from GitHub, same method)
and the `fuel-inference` crate doc's stated layer boundaries:

**Bucket 1 — STABLE, no fuel_core coupling at all:**
- `fuel_cuda_backend::CudaDevice` (`src/model_fuel/device.rs:25`) — one of the already-split
  backend crates the PM named as completed-B0. Port against it now, no caveat.

**Bucket 2 — reached via the `fuel` facade, backed by fuel-core content the fuel-inference
doc does NOT name as one of the two open questions, but also does not claim is staying put:**
- `fuel::inference_context::{InferenceContext, KvCache}` — `fuel-inference/src/lib.rs`'s own
  doc comment calls this "the production executor"'s persistent per-context storage and
  contrasts it with a *retired* eager-Tensor-era module, which reads as settled machinery, not
  an open question — but that's an inference from prose, not a stated decision. ASSUMED
  reasonably stable; not verified with the fuel lane.
- `fuel::kv_block_pool`, `fuel::kv_block_pool_device::{DeviceKvPool, BlockKind}`,
  `fuel::safetensors::MmapedSafetensors` — all currently `pub mod`s directly in
  `fuel-core/src/lib.rs` (confirmed by GitHub content fetch), none named in fuel-inference's
  "what is here" list, none called out by the PM as decided either way.
- `fuel::{DType, Device, Shape}` — the fuel-inference crate doc's own "Layer placement" diagram
  states fuel-core's designated post-dissolution surface is *"tensors, devices, autograd, AND
  the NN surface"* — Device/DType/Shape are exactly that, so ASSUMED to be staying in fuel-core
  rather than moving, but stated as an inference from that diagram, not confirmed with the fuel
  lane.

**Bucket 3 — UNDECIDED, matches the PM's two named open questions exactly:**
- `fuel::lazy::Tensor` (`src/model_fuel/batched.rs:293`) — this is literally the ~8,000-line
  `Tensor` in `fuel-core/src/lazy.rs` the PM asked about.
- `fuel::lazy::{LlamaModel, LlamaConfig, LlamaWeights, WeightStorage}`
  (`src/model_fuel/batched.rs:290`, `decoder.rs:30`, `loader.rs:35`, `loader_f32.rs:35-39`,
  `session.rs:31`) — `LlamaModel` is literally the PM's second named open question
  (`LlamaModel`/`PhiModel`, ~16,000 lines, same file). `LlamaConfig`/`LlamaWeights`/
  `WeightStorage` are supporting types in that same `lazy.rs` module; their home follows
  whatever `Tensor`/`LlamaModel` decision lands.

**Lightbulb's requirement for bucket 3, stated as measured usage, not preference:**
Lightbulb constructs and passes `fuel::lazy::Tensor` in every one of the 10 fuel-importing
files (it's the activation/weight type threading through batching, decode, KV pool ops, and
the smoke test). It constructs `LlamaModel`/`LlamaConfig`/`LlamaWeights` in exactly 3 places
(`loader.rs`, `loader_f32.rs`, `session.rs`) — narrower, load-time-only usage. **Lightbulb
needs `Tensor` without pulling in a model zoo** (it never imports `fuel_transformers` today);
it needs `LlamaModel` only at load and does not care which crate it lives in as long as it
doesn't drag unrelated model-zoo weight or training code into a serving binary. This is
offered as the fuel lane's input for item 52, not a vote for `fuel-transformers` — that's
their call to make from real Tensor-user counts across all fuel consumers, not just this one.

**Next action on this bucket:** message `gf5jcpe8` (fuel lane) directly with this measurement
before writing anything against those paths — not done in this document, queued as the
immediate follow-up.

---

## 6. Per-gap: is fuel missing this, or does Lightbulb just not call it?

MEASURED against fuel's actual crates at `origin/main` (via `gh api`, not the stale local
checkout — same caveat as §5). Method: `gh api repos/ciresnave/fuel/contents/<path>` to list
directories and read files, plus `gh api search/code -f q='"<symbol>" repo:ciresnave/fuel'` to
locate implementations. Checked in the PM's stated priority order.

**Lightbulb's current direct-dependency footprint on fuel, for context:** `Cargo.toml`
declares exactly two fuel deps — `fuel-inference` and (implicitly, via `fuel`) `fuel-core`. `src/model_fuel/` imports only the `fuel` facade (§5). `fuel_inference` is used **only in two
integration tests** (`tests/gpu_paged_vs_contiguous.rs`, `tests/paged_plan_once.rs` —
`fuel_inference::multi_session::{KvBudget, PagedSessionScheduler}`), never in `src/`. `fuel-parallel`, `fuel-transformers`, `fuel-formats`, `fuel-quantized` are **not Lightbulb
dependencies at all** — `fuel-transformers` and `fuel-quantized`/`fuel-formats` arrive only
*transitively* through `fuel`/`fuel-core` in `Cargo.lock` (confirmed: `grep '^name =
"fuel-parallel"$' Cargo.lock` → 0 hits; `fuel-transformers` → 1 hit, transitive only, `grep -rn
"use fuel_transformers" src/` → 0 hits).

### Multi-device model parallelism — **(b) PRESENT IN FUEL, NOT WIRED IN LIGHTBULB**

Fuel ships a dedicated crate, **`fuel-parallel`** (`description = "Multi-GPU parallelism
primitives for the Fuel ML framework"`), with `src/{comm,device_group,distributed_cache,
pipeline_parallel,tensor_parallel,topology}.rs` — the same module names as Lightbulb's own
`src/multi_gpu/`, but a different, apparently more mature implementation:

- **Tensor parallel**: `ColumnParallel`/`RowParallel` sharding, built on
  `fuel::lazy::WeightStorage::apply_linear` — the doc states this "buys F32, BF16, Q4_0 and
  LoRA weights for free," unlike Lightbulb's hand-rolled `matmul`-only version.
  `grep -n "not yet implemented\|todo!\|bail!"` over all six `fuel-parallel/src/*.rs` files →
  **0 hits**, in any file.
- **Pipeline parallel**: `ScheduleKind::{GPipe, OneForwardOneBackward}` — GPipe and a real 1F1B
  scheduler (not a bail), with a test asserting 1F1B's bubble time is ≤ GPipe's.
- **Topology**: descriptive-only (`DeviceInfo`, `Link`, `Interconnect`; explicitly "no CUDA or
  Metal API calls" in the module doc) — a different design from Lightbulb's own
  `DeviceTopology::discover()`, which is the one that had the non-terminating
  `cuda_if_available` loop. Fuel's version sidesteps that class of bug by never probing itself.
- **Distributed cache**: a coordination *protocol* (`CacheShardInfo`, `CacheSyncProtocol`,
  `SyncEvent`), explicitly documented as NOT the storage layer ("the actual cache storage...
  lives in `fuel_inference`"). This is a different abstraction from Lightbulb's
  `CacheSyncStrategy::{Replicated,Sharded,Hybrid}` enum, so it is not a drop-in replacement —
  wiring it up is a redesign, not a re-export.
- Has a live multi-backend test (`fuel-parallel/tests/tri_backend_device_group_live.rs`),
  naming CUDA/Vulkan devices explicitly rather than falling back to CPU — the failure mode that
  made Lightbulb's own `ColumnWise`/`RowWise` tests run zero times for months (ROADMAP.md,
  2026-09-02 finding) does not appear to be present here, though **I did not run this test
  myself (no GPU in this environment) — that it exists and is structured to name real devices
  is measured; that it currently passes on real hardware is ASSUMED from its presence, not
  independently verified.**

**One real caveat, not fuel-has-everything**: fuel-parallel's `ShardingStrategy` has no
`Hybrid` variant at all — only `ColumnParallel`/`RowParallel` exist. Lightbulb's own
`ShardingStrategy::Hybrid` bail therefore has **no fuel equivalent to wire to** — this one
specific sub-gap may genuinely be (a), or may simply be a strategy nobody has needed (fuel's
two-strategy design might be the intentionally complete set). **I could not determine which.**
Same open question for `CacheSyncStrategy::{Sharded,Hybrid}` specifically, since fuel's
protocol layer doesn't map onto that enum directly — establishing whether `fuel_inference`'s
storage side actually implements sharded/hybrid cache storage (as opposed to just coordination)
needs a read of `fuel_inference`'s cache modules that I have not done; **flagged as not yet
determined, not asserted either way.**

**Bottom line for the PM's sequencing question: the multi-GPU gap that matters most for the
new hardware — actually running tensor/pipeline-parallel inference across multiple heterogeneous
devices at all — is (b).** Fuel has it, Lightbulb has never depended on `fuel-parallel` and
`src/multi_gpu/` was never wired to talk to fuel's tensor type in the first place (it predates
the port and operates on `candlelight` tensors). The eight bail sites are in a module that isn't
even on the fuel port's critical path — the real fix is wiring `model_fuel/` to `fuel-parallel`
and `fuel_inference`'s distributed cache storage, then likely deleting `src/multi_gpu/` rather
than fixing its bails.

### GGUF / quantized loading — **(b) PRESENT IN FUEL, NOT WIRED IN LIGHTBULB, and more so than I expected**

`fuel-formats/src/` has `gguf.rs`, `ggml.rs`, `imatrix.rs`, `pickle.rs`, `safetensors.rs` — real
parsers, not stubs (file listing via `gh api`, not inferred from crate description).
`fuel-quantized/src/` has `k_quants.rs`, `cpu.rs`, `avx.rs`, `neon.rs` — real dequant/kernel
code with SIMD variants, not a placeholder.

**Further than that: `fuel-transformers` already ships GGUF-quantized model constructors for
specific architectures**, found via `gh api search/code -f q='"fn from_gguf"
repo:ciresnave/fuel'`: `lazy_quantized_llama.rs`, `lazy_quantized_phi3.rs`,
`lazy_quantized_qwen2.rs`, `lazy_quantized_qwen3.rs`, `lazy_quantized_qwen3_moe.rs`,
`lazy_quantized_gemma3.rs`, `lazy_quantized_glm4.rs`, `lazy_quantized_t5.rs`,
`lazy_quantized_lfm2.rs`, `lazy_quantized_smollm3.rs`, `lazy_quantized_whisper.rs` — each with
its own `from_gguf` constructor. This directly contradicts the assumption I'd have made from
Lightbulb's side alone (`src/engine/model_runner.rs:327`'s message, "GGUF/other Candle-supported
quant formats... Quantized/GGUF support is pending `impl FuelDecoder for Llama3Model`") — that
message describes Lightbulb's own wiring gap, not a fuel capability gap. **Lightbulb's own
comment is accurate about Lightbulb, and reads (without the crate-level check) like it's about
fuel — worth fixing in `model_runner.rs` since a future reader would draw the same wrong
inference I almost did.**

### Architecture dispatch beyond Llama — **(b) PRESENT IN FUEL, NOT WIRED IN LIGHTBULB, overwhelmingly**

`fuel-transformers/src/models/mod.rs` declares ~90+ `pub mod lazy_*` architectures (partial
list, alphabetical prefix only, via `gh api`): `lazy_bert`, `lazy_chatglm`, `lazy_deepseek2`,
`lazy_falcon`, `lazy_gemma`/`gemma2`/`gemma3`/`gemma4_*`, `lazy_glm4`, `lazy_granite`,
`lazy_lfm2`, `lazy_llama_full`, `lazy_mamba`/`mamba2`, `lazy_qwen*` (not shown in the head read,
but the quantized-constructor search above confirms `lazy_quantized_qwen2/3/3_moe` exist), and
more — this is a genuine model zoo, not a handful of variants. Lightbulb's `model_fuel/`
hand-rolls its own `LlamaModel`-only loader against the bare `fuel::lazy` facade instead of
using this zoo at all.

### AWQ — **COULD NOT DETERMINE**

`gh api search/code -f q='AWQ repo:ciresnave/fuel'` found no crate implementing the AWQ format
specifically (hits were unrelated: `quant_w4a16.rs`, `quant_scale.rs`, generic quantization
infra). `fuel-cuda-backend`'s W4A16 kernel is the same *numeric family* as AWQ (4-bit weight,
16-bit activation) but AWQ checkpoints carry a specific packing/scale format I did not verify
fuel's loader accepts. **Genuinely unresolved — do not treat this as (a) or (b) without a
follow-up read of the AWQ checkpoint format against `fuel-quantized`'s loader,** which I have
not done. Lowest priority per the PM's ordering, so not pursued further today.

### Speculative decoding — **(b) PRESENT IN FUEL, NOT WIRED IN LIGHTBULB**

`fuel-inference/src/speculative.rs` implements real draft-then-verify accept/reject
math (Leviathan et al. 2023 / Chen et al. 2023), taking caller-supplied closures for the draft
and target models — not a stub. Lightbulb has zero references to `fuel_inference::speculative`
anywhere (`grep -rn "fuel_inference" .` → only the two `multi_session` test imports noted
above). Lowest priority per the PM's ordering, so not pursued further today, but the
finding is unambiguous.

### Answering the PM's actual question

**Mostly (b).** Of the five gaps: multi-device parallelism, GGUF/quantized loading, and
architecture dispatch are all substantially (b) — the capability exists in fuel crates
Lightbulb does not yet depend on or call. Speculative decoding is (b) as well, more simply (a
single module, unused). AWQ is genuinely undetermined. **None of the five came back as clearly
(a) missing-in-fuel.** Per the PM's own decision rule, this points at concentrating the
immediate work in Lightbulb (wiring), not standing down this lane in favor of fuel-side
capability work — with the caveat that `Hybrid` tensor-parallel sharding and `Sharded`/`Hybrid`
cache *storage* (as opposed to coordination) remain open questions that could still turn out to
need fuel-side work; I would not close those off without the follow-up reads named above.

---

## Summary table

| Question | Answer | Confidence |
|---|---|---|
| Does `fuel-engine` build/test? | Yes, both exit 0, today | MEASURED |
| Does the flag change which tests run? | No — identical test list with/without it | MEASURED |
| Parity gap | 6-item list above (GGUF, arch dispatch, AWQ, multi-GPU, speculative, CI verification) | MEASURED |
| Multi-GPU real? | 8/8 original bail sites still bail; 3/6 strategies now honest-but-unimplemented, 3/6 working | MEASURED |
| Test suite covers fuel path? | Unit-level yes (62 tests, every CI run); end-to-end HTTP/content correctness no (never run in CI, last manual run 2026-08-08) | MEASURED |
| Fuel import buckets | 1 stable, ~6 modules assumed-settled-but-unconfirmed, 2 symbol-families (Tensor, LlamaModel-family) confirmed undecided | MIXED — see caveats |
| Gaps: fuel-missing vs Lightbulb-unwired | Multi-GPU, GGUF/quantized, architecture dispatch, speculative decoding: mostly (b) unwired, fuel already has them. AWQ: undetermined. `Hybrid` TP/cache-sync specifically: undetermined | MOSTLY MEASURED, two open items flagged |

No port code was written. No version was bumped.
