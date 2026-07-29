# Fuel Port Phase 0: Pre-Port Cleanup and Substrate Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dead-debug realize sites that would destroy Fuel's fusion, establish a buildable current Fuel checkout, and produce a written record of Fuel's *actual* API surface — the input Plan 2 (the `model_fuel/` rewrite) cannot be written without.

**Architecture:** Three independent deliverables in dependency order. Task 1 is pure Lightbulb cleanup with zero Fuel dependency and standalone value (it fixes a latent performance bug today). Task 2 unblocks building Fuel at all. Task 3 is a spike whose deliverable is a *document* of verified signatures, not code. Task 4 runs the one experiment that could kill the attention-reduction route early and cheaply.

**Tech Stack:** Rust 2024 edition, `candlelight` (outgoing), Fuel (`fuel`, `fuel-core`, `fuel-ir`, `fuel-dispatch`, `fuel-inference`), `cargo`, PowerShell/Git Bash on Windows.

## Global Constraints

- **`C:\Projects\fuel` is a shared working tree across three sessions — read-only. Never run mutating git operations there.** Read-only queries (`git log`, `git show`, `git diff`, `git worktree list`) are fine.
- **Never run workspace-wide `cargo check` / `cargo test` in the Fuel repo.** `tensor-tools` has a standing break and is a default member, so a bare root invocation fails for unrelated reasons. Always `-p <crate>`.
- **One cargo invocation at a time.** The build-directory lock serializes; parallel invocations thrash.
- **CPU is an F32-only world for weights in Fuel.** Mixed `[F32, BF16, F32]` matmul is CUDA-only. All CPU work casts weights to F32 at load.
- **Do not modify files under `C:\Projects\fuel`.** Fuel-side changes are requested from peer sessions (`2eymo83p` for dispatch/allocator, `trpe1mc5` for contract/seam), never made directly.
- **Evidence standard:** a claim is `[verified]` only when it was executed or read from source. "The file exists" is never evidence that "it works."

---

### Task 1: Remove dead-debug realize sites

Vestigial `to_vec1` calls whose consuming statistic was deleted when a debug print was removed. Each forces a full tensor realize and discards the result. Under Candle they are wasteful copies; under Fuel they would break the graph and make capture-shaped decode impossible. **This task has standalone value independent of the port** — `mlp_wrapper.rs:156` runs on every MLP call, every layer, every token.

**Files:**
- Modify: `src/model/mlp_wrapper.rs:155-159`
- Modify: `src/model/custom_transformer.rs:262-266`, `:548-552`
- Modify: `src/model/custom_attention.rs:673-677`
- Modify: `src/cache/parallel_cache_builder.rs:2459`
- Test: existing suites — `tests/batched_transformer_correctness.rs`, `tests/model_correctness.rs`, `tests/fused_rmsnorm_parity.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: no API change. Behaviour must be bit-identical — every deleted binding was already unused, so this is provably a no-op on output.

- [ ] **Step 1: Establish the green baseline**

Record the current pass/fail state so any later failure is attributable.

```bash
cd C:/Projects/lightbulb
cargo test --test batched_transformer_correctness --test model_correctness 2>&1 | tail -30
```

Expected: note the exact pass/fail counts. If anything already fails, record it — do **not** fix it in this task.

- [ ] **Step 2: Delete the `mlp_wrapper.rs` site**

In `src/model/mlp_wrapper.rs`, inside `MlpWrapper::forward`, delete these four lines:

```rust
        // DEBUG: Check input stats
        let input_vec = x.flatten_all()?.to_vec1::<f32>()?;
        let _input_max = input_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let _input_mean: f32 = input_vec.iter().sum::<f32>() / input_vec.len() as f32;
```

Both `_input_max` and `_input_mean` are unused. Nothing downstream reads `input_vec`.

- [ ] **Step 3: Delete the `custom_transformer.rs` sites**

At `:262-266` (load-time, in the embedding constructor) delete:

```rust
            // DEBUG: Verify weight shapes
            // DEBUG output removed
            let emb_w_vec = embedding_weight.flatten_all()?.to_vec1::<f32>()?;
            let _emb_w_mean: f32 = emb_w_vec.iter().sum::<f32>() / emb_w_vec.len() as f32;
```

At `:548-552` (inside the per-layer loop) delete the `if layer_idx < 3 || layer_idx >= self.blocks.len() - 3` block containing:

```rust
                let hs_vec = hidden_states.flatten_all()?.to_vec1::<f32>()?;
                let _mean: f32 = hs_vec.iter().sum::<f32>() / hs_vec.len() as f32;
```

Delete the enclosing `if` as well — with the body gone it has no effect.

**Do NOT touch `:593` or `:749`.** Those feed the H2O policy and are load-bearing; they are Plan 2's problem.

- [ ] **Step 4: Delete the `custom_attention.rs` site**

At `:673-677`, delete the `if layer_idx == 0` block containing:

```rust
            let _output_vec = output.flatten_all()?.to_vec1::<f32>()?;
            // DEBUG output removed
```

Delete the enclosing `if` too. **Do NOT touch `:919`** — that is H2O-related.

- [ ] **Step 5: Inspect and handle the `parallel_cache_builder.rs` site**

Read `src/cache/parallel_cache_builder.rs` around line 2459:

```bash
sed -n '2450,2470p' src/cache/parallel_cache_builder.rs
```

The site is `if let Ok(_idx_vec) = indices.to_vec2::<u32>() { … }` — the binding is unused. **If the `if let` body is empty or only contains further unused bindings, delete the whole block.** If the body does real work, leave it and record why in the commit message. Judgment is required here; do not delete blind.

- [ ] **Step 6: Verify the build and confirm behaviour is unchanged**

```bash
cargo build 2>&1 | tail -20
cargo test --test batched_transformer_correctness --test model_correctness 2>&1 | tail -30
```

Expected: builds clean; pass/fail counts **identical to Step 1**. Any new failure means a deleted binding was not actually dead — revert and investigate.

- [ ] **Step 7: Confirm no dead sites remain**

```bash
grep -n "DEBUG output removed\|DEBUG: Check input stats\|DEBUG: Verify weight shapes" src/ -r
```

Expected: no matches.

- [ ] **Step 8: Commit**

```bash
git add src/model/mlp_wrapper.rs src/model/custom_transformer.rs src/model/custom_attention.rs src/cache/parallel_cache_builder.rs
git commit -m "perf: Remove dead-debug tensor realizes from the decode hot path

Vestigial to_vec1 calls whose consuming statistics were deleted when
their debug prints were removed. Each forced a full tensor realize and
discarded the result.

mlp_wrapper.rs:156 was the worst — realizing the entire activation
tensor on every MLP call, every layer, every token, to compute two
unused floats. A wasteful copy under Candle; under Fuel's lazy graph it
would break fusion at every MLP and make capture-shaped decode
impossible.

Provably a no-op on output: every deleted binding was already unused."
```

---

### Task 2: Establish a buildable current Fuel checkout

Nothing Fuel-facing can be verified until there is a tree at current `origin/main` that is safe to build in. `C:\Projects\fuel` is a deliberately-stale shared mirror and is off-limits for git operations.

**Files:**
- Create: a git worktree at `C:\Projects\fuel-lightbulb-port` (path may change per Step 2's answer)
- Modify: none

**Interfaces:**
- Consumes: nothing.
- Produces: a filesystem path, referred to hereafter as `$FUEL_TREE`, containing current `origin/main` and safe to run `cargo` in. Tasks 3 and 4 depend on it.

- [ ] **Step 1: Confirm the shared tree is still stale**

```bash
cd C:/Projects/fuel && git log --oneline -1 && git log --oneline origin/main -1
```

If HEAD already equals `origin/main`, a peer fast-forwarded it — **skip to Step 4** and set `$FUEL_TREE=C:/Projects/fuel`.

- [ ] **Step 2: Ask the peer sessions for the correct convention**

Do not create a worktree in a shared repo unilaterally. Send via `mcp__claude-peers__send_message` to `2eymo83p`:

> I need a buildable tree at current origin/main to verify the dispatch fix and capture Fuel's API surface. `C:/Projects/fuel` is stale and read-only per your convention. Is the convention for me to `git worktree add C:/Projects/fuel-lightbulb-port origin/main` from the shared repo, or would you prefer a different path/branch, or to fast-forward the shared checkout? I don't want to add a worktree to a repo three sessions share without asking.

Wait for the answer. **This is a genuine block — do not guess.**

- [ ] **Step 3: Create the worktree as directed**

Using whatever the peer specifies. If they approve the proposed form:

```bash
cd C:/Projects/fuel && git worktree add C:/Projects/fuel-lightbulb-port origin/main
```

- [ ] **Step 4: Verify the tree is current and contains the expected fixes**

```bash
cd $FUEL_TREE
git log --oneline -1
grep -n "coverage_diagnostic" fuel-dispatch/src/plan.rs | head -3
ls fuel-core/src/kv_block_pool.rs
```

Expected: HEAD matches `origin/main`; `plan.rs` calls `coverage_diagnostic()`; `kv_block_pool.rs` exists (it is on `origin/main` per the allocator session).

- [ ] **Step 5: Confirm it builds, scoped**

```bash
cd $FUEL_TREE && cargo build --release -p fuel-lazy-examples --bin llama-lazy 2>&1 | tail -15
```

Expected: `Finished` with warnings. If it fails, that is a Fuel-side regression — report it to `2eymo83p` with the exact error rather than working around it.

- [ ] **Step 6: Record the tree location**

Append to `docs/superpowers/notes/fuel-environment.md` (create it if absent):

```markdown
# Fuel environment

- `$FUEL_TREE` = <the path>  — current origin/main, safe to build in.
- `C:\Projects\fuel` — shared stale mirror, READ-ONLY, never run git ops.
- Always `cargo -p <crate>`; never workspace-wide. One cargo at a time.
```

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/notes/fuel-environment.md
git commit -m "docs: Record the Fuel build tree convention for the port"
```

---

### Task 3: Capture Fuel's real API surface (spike)

**The deliverable is a document of verified signatures, not code.** Plan 2 cannot be written without it, because every prior structural claim about Fuel in this project has required correction on contact. This task exists to replace inference with evidence.

**Files:**
- Create: `docs/superpowers/notes/fuel-api-surface.md`
- Modify: none

**Interfaces:**
- Consumes: `$FUEL_TREE` from Task 2.
- Produces: verified signatures for weight loading, model construction, forward, realize, and KV cache — the exact names Plan 2's tasks will call.

- [ ] **Step 1: Confirm the diagnostic fix by reproducing the original failure**

```bash
cd $FUEL_TREE && ./target/release/llama-lazy.exe "TinyLlama/TinyLlama-1.1B-Chat-v1.0" "Once upon a time" 8 2>&1 | tail -20
```

Expected: it still **fails** (the capability gap is unfixed) but the message now names `Cpu` and lists real `(op, dtypes)` coverage showing `[F32,F32,F32]` present and `[F32,BF16,F32]` absent — rather than the old `available backends: []`. Record the exact output verbatim.

Report the confirmation to `2eymo83p`, who asked to have it validated against this repro.

- [ ] **Step 2: Extract the weight-loading surface**

```bash
cd $FUEL_TREE
grep -n "pub fn\|pub struct" fuel-core/src/lazy_nn_varbuilder.rs | head -40
grep -n "pub fn from_mmaped_safetensors\|pub fn from_tensors\|pub fn get\b\|pub fn to_dtype" fuel-core/src/lazy_nn_varbuilder.rs
```

Record every signature needed to load safetensors and fetch a named tensor, **including how to request F32** (the CPU constraint).

- [ ] **Step 3: Extract the model-construction and forward surface**

```bash
grep -n "pub struct LlamaModel\|pub struct LlamaConfig\|pub struct LlamaWeights" fuel-core/src/lazy.rs
grep -n "pub fn forward\|pub fn forward_hidden\|pub fn run_backbone_with_rope_tables" fuel-core/src/lazy.rs
grep -n "pub fn from_hf_json_str\|pub struct LlamaFullConfig\|pub struct Llama3Model" fuel-core/src/lazy_llama_full.rs
```

- [ ] **Step 4: Extract the realize and KV-cache surface**

```bash
grep -n "pub fn realize\|pub fn to_vec1\|pub fn to_scalar" fuel-core/src/lazy.rs | head -20
grep -n "pub fn with_capacity\|pub fn with_dims\|pub struct KvCache\|pub struct InferenceContext" fuel-core/src/inference_context.rs
grep -n "pub fn evict_blocks\|pub fn evict_range\|pub fn restore\|pub fn blocks_required_batch\|pub struct KvGeometry\|pub struct EvictReport" fuel-core/src/kv_block_pool.rs
```

- [ ] **Step 5: Write the API surface document**

Create `docs/superpowers/notes/fuel-api-surface.md` recording, for each of the four areas above: the **exact** signature, the **file:line** it came from, and a `[verified: read from source]` or `[verified: executed]` marker. Where a needed capability appears absent, record it as an explicit **gap** rather than inventing a plausible name.

Include a header stating the tree's commit hash, so the document's staleness is checkable later.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/notes/fuel-api-surface.md
git commit -m "docs: Record Fuel's verified API surface for the port

Signatures read from source at a known commit, each marked with its
file:line and evidence level. Written because every prior structural
claim about Fuel in this project needed correction on contact — Plan 2
is written against this document, not against inference."
```

---

### Task 4: Capture-vs-accumulate experiment

The cheapest decisive test of the attention-reduction route (`docs/superpowers/specs/2026-07-29-attention-reduction-in-graph.md`). Tests falsifier #2 — whether a runtime-offset accumulate composes with `CapturedRun` — **independently** of the arm question. If capture breaks here, the route dies before any H2O work is done.

**Files:**
- Create: `docs/superpowers/notes/capture-accumulate-experiment.md`
- Create: a scratch binary or test under `$FUEL_TREE` **only if** the peer sessions approve adding one; otherwise drive it from a Lightbulb-side test.

**Interfaces:**
- Consumes: `$FUEL_TREE` (Task 2), the realize/KV signatures from Task 3.
- Produces: a yes/no on whether runtime-offset accumulate survives capture. Gates Plan 3 (the attention reduction).

- [ ] **Step 1: Locate the CapturedRun entry points**

```bash
cd $FUEL_TREE
grep -rn "CapturedRun" fuel-core/src/lazy.rs fuel-core/src/inference_context.rs | head -20
```

Record how a run is captured and replayed, and what invalidates a capture.

- [ ] **Step 2: Find an existing capture test to model the experiment on**

```bash
grep -rln "CapturedRun" fuel-core/tests/ | head
```

Read the closest existing test. **Model the experiment on a known-good pattern rather than inventing a harness** — the goal is to test one variable, not to debug scaffolding.

- [ ] **Step 3: Build the minimal graph**

A graph containing: a matmul, a reduce-sum over one axis, and a runtime-offset accumulate (`c = c * decay + a`) into a persistent buffer. Use the `Op::WriteSlice` / `inplace_affine` shapes confirmed to exist. **Write the exact code only after Task 3 has recorded the real signatures** — this step is deliberately not pre-written, because pre-writing it against unverified signatures is the failure mode this plan exists to avoid.

- [ ] **Step 4: Capture, replay, and compare**

Capture the step, replay it, and assert the accumulator advances identically across replays and matches an uncaptured reference run.

- [ ] **Step 5: Record the result**

Write `docs/superpowers/notes/capture-accumulate-experiment.md` with the verdict, the code used, and the exact output. **If capture breaks, say so plainly and state the observed failure mode** — a negative result is the valuable outcome here, and it redirects to §15's preference #4.

- [ ] **Step 6: Report to the seam owner**

Send the result to `trpe1mc5`, who is tracking the falsifiers in §15 and Annex A.3. This is the first of the three to be settled by execution rather than reading.

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/notes/capture-accumulate-experiment.md
git commit -m "test: Capture-vs-accumulate experiment result

Falsifier #2 from the attention-reduction spec, tested independently of
the arm question."
```

---

## What this plan deliberately does NOT cover

Each becomes its own plan, written **after** Task 3 delivers verified signatures:

- **Plan 2 — `model_fuel/` vertical slice.** Load weights (F32-cast), build a lazy Llama forward, realize logits, greedy-sample one token, parity-test against the frozen `candlelight` path.
- **Plan 3 — attention reduction**, gated on Task 4's result.
- **Plan 4 — batched decode on the KV allocator** (`evict_blocks`/`evict_range`/`EvictReport`), the first end-to-end target.
- **Plan 5 — upstreaming** to Fuel: stateful H2O accumulation, KIVI granularity control, relationship-aware compression, routing observability.
- **Plan 6 — deletion**: `multi_gpu/`, Marlin FFI, `awq_qwen3.rs`, the frozen `model/`, `candlelight`, `mlmf`.

**Why the split:** Plans 2 and 4 require exact Fuel signatures that do not yet exist in verified form. Writing them now would mean inventing plausible names — the exact failure this project has corrected repeatedly. Task 3 exists to make them writable.
