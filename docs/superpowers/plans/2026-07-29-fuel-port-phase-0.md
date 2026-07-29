# Fuel Port Phase 0 (reissued): Cleanup, Environment, and Oracle Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Supersedes** `2026-07-29-fuel-port-phase-0.md` (the pre-oracle version). That plan gated Task 1 on `batched_transformer_correctness` / `model_correctness`, which **do not compile**, and assumed D1's freeze-the-Candle-path oracle, which was replaced. See [the spec](../specs/2026-07-28-fuel-port-design.md#the-oracle-three-tiers).

**Goal:** Reach the state where `model_fuel/` work can start safely — dead-debug realizes gone, a buildable Fuel tree, Fuel's real API surface recorded, and a *working* tier-1 oracle mechanism proven on a simple fragment.

**Architecture:** Five tasks in dependency order. Task 1 is pure Lightbulb cleanup with standalone value. Tasks 2–3 establish the Fuel environment and replace inference with recorded fact. Tasks 4–5 build the oracle bottom-up — a single-op differential first to prove the harness, then a subgraph recipe to prove the mechanism that actually matters.

**Tech Stack:** Rust 2024, `candlelight` (outgoing), Fuel, `kiss-ref-core` 0.1.0 (dev-dep), `cargo`, Git Bash / PowerShell on Windows.

## Global Constraints

- **`C:\Projects\fuel` is a shared working tree across three sessions — read-only.** No mutating git operations. Read-only queries (`git log`, `git show`, `git diff`, `git worktree list`) are fine.
- **Never run workspace-wide `cargo check`/`test` in the Fuel repo** — `tensor-tools` has a standing break and is a default member. Always `-p <crate>`. One cargo invocation at a time.
- **CPU is an F32-only world for weights in Fuel** (no `[F32, BF16, F32]` kernel). All CPU work casts weights to F32 at load. This is also required for kiss-ref differentials — see below.
- **Never trust a background task's exit code.** Piping to `tail` masks failures; three runs this session reported exit 0 while failing or doing nothing. **Read the output.**
- **Evidence standard.** `[verified]` means executed or read from source. "The file exists" is never evidence that "it works." Before reporting a `head`/`grep` result as a total, confirm it is one. Before reporting your own past actions, check the transcript, not your recollection.
- **kiss-ref is a differential target, never a verdict source.** A disagreement means "determine which of us is wrong."
- **Do not modify files under `C:\Projects\fuel`.** Fuel-side changes are requested from peers: `2eymo83p` (dispatch/allocator), `trpe1mc5` (contract/seam), `3vgwagtz` (kiss-ref).

---

### Task 1: Remove dead-debug realize sites

Vestigial `to_vec1` calls whose consuming statistic was deleted when a debug print was removed. Each forces a full tensor realize and discards the result — a wasteful copy under Candle, fusion-and-capture-destroying under Fuel. **Standalone value:** `mlp_wrapper.rs:156` realizes the entire activation tensor on every MLP call, every layer, every token.

**Files:**
- Modify: `src/model/mlp_wrapper.rs`, `src/model/custom_transformer.rs`, `src/model/custom_attention.rs`, `src/cache/parallel_cache_builder.rs`
- Test: the 11 test targets that compile (see Step 1)

**Interfaces:**
- Consumes: nothing.
- Produces: no API change. Every deleted binding was already unused, so this is provably a no-op on output.

- [ ] **Step 1: Establish the baseline from the suites that actually compile**

The three suites the old plan named do not compile. Use the 11 that do:

```bash
cd C:/Projects/lightbulb
cargo test --test contracts_integration --test decomposition \
  --test enhanced_correctness_tests --test fused_rmsnorm_parity \
  --test m4_integration --test m5_integration_tests \
  --test multi_gpu_validation --test parallel_model_manager_integration \
  2>&1 | tee /tmp/baseline.txt | tail -40
```

Omitted deliberately: `contracts_live`, `fused_rmsnorm_perf`, `integration_local_model` — they plausibly need a live server, a model file, or are perf-only. Run them separately if curious; do not gate on them.

**Record the exact pass/fail counts per suite.** If anything already fails, record it and do **not** fix it here.

- [ ] **Step 2: Delete the `mlp_wrapper.rs` site**

In `src/model/mlp_wrapper.rs`, inside `MlpWrapper::forward`, delete these four lines:

```rust
        // DEBUG: Check input stats
        let input_vec = x.flatten_all()?.to_vec1::<f32>()?;
        let _input_max = input_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let _input_mean: f32 = input_vec.iter().sum::<f32>() / input_vec.len() as f32;
```

Both consumers are unused; nothing downstream reads `input_vec`.

- [ ] **Step 3: Delete the `custom_transformer.rs` sites**

Delete this block (embedding constructor, load-time):

```rust
            // DEBUG: Verify weight shapes
            // DEBUG output removed
            let emb_w_vec = embedding_weight.flatten_all()?.to_vec1::<f32>()?;
            let _emb_w_mean: f32 = emb_w_vec.iter().sum::<f32>() / emb_w_vec.len() as f32;
```

And delete the `if layer_idx < 3 || layer_idx >= self.blocks.len() - 3 {` block containing:

```rust
                let hs_vec = hidden_states.flatten_all()?.to_vec1::<f32>()?;
                let _mean: f32 = hs_vec.iter().sum::<f32>() / hs_vec.len() as f32;
```

Delete the enclosing `if` too — with the body gone it has no effect.

**Do NOT touch the `agg`/`attn_weights` extractions further down.** Those feed the H2O policy and are load-bearing; they belong to the attention-reduction work.

- [ ] **Step 4: Delete the `custom_attention.rs` site**

Delete the `if layer_idx == 0 {` block containing:

```rust
            let _output_vec = output.flatten_all()?.to_vec1::<f32>()?;
            // DEBUG output removed
```

Delete the enclosing `if`. **Do NOT touch the `scores.to_vec2` site** — H2O-related.

- [ ] **Step 5: Inspect and handle the `parallel_cache_builder.rs` site**

```bash
sed -n '2450,2470p' src/cache/parallel_cache_builder.rs
```

The site is `if let Ok(_idx_vec) = indices.to_vec2::<u32>() { … }` with an unused binding. **If the body is empty or contains only further unused bindings, delete the whole block.** If it does real work, leave it and say why in the commit message. Judgment required — do not delete blind.

- [ ] **Step 6: Verify build and unchanged behaviour**

```bash
cargo check --lib 2>&1 | tail -20
cargo test --test contracts_integration --test decomposition \
  --test enhanced_correctness_tests --test fused_rmsnorm_parity \
  --test m4_integration --test m5_integration_tests \
  --test multi_gpu_validation --test parallel_model_manager_integration \
  2>&1 | tail -40
```

Expected: builds clean; counts **identical to Step 1**. Any new failure means a deleted binding was not dead — revert and investigate.

- [ ] **Step 7: Confirm no dead sites remain**

```bash
grep -rn "DEBUG output removed\|DEBUG: Check input stats\|DEBUG: Verify weight shapes" src/
```

Expected: no matches.

- [ ] **Step 8: Commit**

```bash
git add src/model/mlp_wrapper.rs src/model/custom_transformer.rs src/model/custom_attention.rs src/cache/parallel_cache_builder.rs
git commit -m "perf: Remove dead-debug tensor realizes from the decode hot path"
```

---

### Task 2: Establish a buildable Fuel checkout

**Files:** creates a git worktree; path recorded as `$FUEL_TREE`.

**Interfaces:** Produces `$FUEL_TREE` — a tree at current `origin/main`, safe to run `cargo` in. Tasks 3 and 5 depend on it.

- [ ] **Step 1: Check whether the shared tree was fast-forwarded**

```bash
cd C:/Projects/fuel && git log --oneline -1 && git log --oneline origin/main -1
```

If HEAD equals `origin/main`, set `$FUEL_TREE=C:/Projects/fuel` and skip to Step 4.

- [ ] **Step 2: Ask before creating a worktree in a shared repo**

Message `2eymo83p` via `mcp__claude-peers__send_message`:

> I need a buildable tree at current origin/main to verify the dispatch + dtype fixes and record Fuel's API surface. `C:/Projects/fuel` is stale and read-only per your convention. Is it fine for me to `git worktree add C:/Projects/fuel-lightbulb-port origin/main` from the shared repo, or would you prefer a different path, or to fast-forward the shared checkout?

**Genuine block — do not guess.**

- [ ] **Step 3: Create the worktree as directed**

- [ ] **Step 4: Verify currency and the landed fixes**

```bash
cd $FUEL_TREE && git log --oneline -1
grep -n "coverage_diagnostic" fuel-dispatch/src/plan.rs | head -3
ls fuel-core/src/kv_block_pool.rs
```

- [ ] **Step 5: Confirm `llama-lazy` now runs**

The dtype-reconciliation pass should have unblocked it.

```bash
cd $FUEL_TREE && cargo build --release -p fuel-lazy-examples --bin llama-lazy 2>&1 | tail -10
./target/release/llama-lazy.exe "TinyLlama/TinyLlama-1.1B-Chat-v1.0" "Once upon a time" 8 2>&1 | tail -20
```

Expected: tokens. **Report the result to `2eymo83p` either way** — they asked to have the fix confirmed against this exact repro. If it still fails, that is a real finding, not a setback.

- [ ] **Step 6: Record the environment and commit**

Write `docs/superpowers/notes/fuel-environment.md` with `$FUEL_TREE`, the read-only rule, and the `-p`/one-at-a-time cargo rules. Commit.

---

### Task 3: Record Fuel's real API surface

**The deliverable is a document of verified signatures, not code.** Every prior structural claim about Fuel in this project required correction on contact.

**Files:** Create `docs/superpowers/notes/fuel-api-surface.md`.

**Interfaces:** Consumes `$FUEL_TREE`. Produces the exact signatures later plans call.

- [ ] **Step 1: Weight loading**

```bash
cd $FUEL_TREE
grep -n "pub fn\|pub struct" fuel-core/src/lazy_nn_varbuilder.rs | head -40
```

Record how to load safetensors and fetch a named tensor, **including how to request F32** (the CPU constraint).

- [ ] **Step 2: Model construction and forward**

```bash
grep -n "pub struct LlamaModel\|pub struct LlamaConfig\|pub struct LlamaWeights" fuel-core/src/lazy.rs
grep -n "pub fn forward\|pub fn forward_hidden\|pub fn run_backbone_with_rope_tables" fuel-core/src/lazy.rs
grep -n "pub fn from_hf_json_str\|pub struct LlamaFullConfig\|pub struct Llama3Model" fuel-core/src/lazy_llama_full.rs
```

- [ ] **Step 3: Realize, KV cache, allocator**

```bash
grep -n "pub fn realize" fuel-core/src/lazy.rs | head -20
grep -n "pub fn with_capacity\|pub struct KvCache\|pub struct InferenceContext" fuel-core/src/inference_context.rs
grep -n "pub fn evict_blocks\|pub fn evict_range\|pub fn restore\|pub fn blocks_required_batch\|pub struct KvGeometry\|pub struct EvictReport" fuel-core/src/kv_block_pool.rs
```

Note `KvGeometry` now carries `n_layers` (shared block-table model: one physical block addresses the same slot in every layer's K/V).

- [ ] **Step 4: Read the graph-affinity doc**

```bash
grep -n -A 30 "graph-affine" fuel-core/src/lazy.rs | head -40
```

**Constructors mint new graphs; use `const_*_like` to build on an existing one.** Record the rule explicitly — it is the constraint that shapes all weight loading.

- [ ] **Step 5: Write and commit the document**

For each area: exact signature, `file:line`, and `[verified: read from source]` or `[verified: executed]`. Record absent capabilities as **gaps**, never as invented names. Header records the tree's commit hash so staleness is checkable.

---

### Task 4: Prove the kiss-ref differential harness on one op

Smallest thing that proves the tier-2 mechanism. If this doesn't work, tiers 1 and 2 both fail and it is better to know now.

**Files:**
- Modify: `Cargo.toml` (dev-dependency)
- Create: `tests/kissref_differential.rs`

**Interfaces:** Produces a working pattern for "compute X in Lightbulb, compute X in kiss-ref, compare within tolerance" that Task 5 generalizes.

- [ ] **Step 1: Read kiss-ref's API before writing anything**

```bash
cargo add kiss-ref-core@0.1.0 --dev --dry-run
```

Then read the published docs for `reference_*`, `diff_*`, `ulp_distance`, and the `DetClass` type. **Record the actual signatures before writing code** — do not infer them. If the crate's own docs are thin, `cargo doc --open -p kiss-ref-core` after adding it.

- [ ] **Step 2: Add the dev-dependency**

```toml
[dev-dependencies]
kiss-ref-core = "0.1.0"
```

Note: `ConstBits` and `ScalarFloat::from_bits` ship in **0.2.0**, not 0.1.0. Pathological bit patterns need a git rev at/after `721d03b`. Not required for this task.

- [ ] **Step 3: Write the failing test**

Pick **RMSNorm** — small, deterministic, and Lightbulb has `fused_rmsnorm.rs` plus a `fused_rmsnorm_parity` suite that already compiles and passes.

Compute an RMSNorm over a small F32 tensor two ways: Lightbulb's implementation, and kiss-ref's reference. Assert agreement within tolerance.

**Write the exact code only after Step 1 records the real signatures.** Deliberately not pre-written — inventing kiss-ref API names is the failure mode this plan exists to avoid.

**Constraints:** diff in **f32** (kiss-ref accumulates at storage precision; narrow lanes diverge structurally, not in ULPs). Honor `DetClass` — tolerance-compare anything nondeterministic, never bit-compare.

- [ ] **Step 4: Run it, expect failure, then make it pass**

```bash
cargo test --test kissref_differential 2>&1 | tail -20
```

- [ ] **Step 5: Commit**

---

### Task 5: Tier-1 toy-scale subgraph recipe

Proves the mechanism that actually matters: an independent reference for a **graph fragment**, which is where construction bugs live.

**Files:** Extend `tests/kissref_differential.rs`.

**Interfaces:** Produces the pattern the attention block will follow in Phase 1.

- [ ] **Step 1: Read the recipe API**

`eval_recipe(FlatDag, inputs, params, indices)` returns per-node `DetClass`. Record the exact construction of a `FlatDag` from source or docs before writing.

- [ ] **Step 2: Build the simplest multi-op fragment**

**RMSNorm followed by a matmul** — two ops, one data dependency. Toy dimensions (hidden 64). Express as a kiss-ref recipe; compute the same fragment in Lightbulb; diff.

This proves composition, which single-op differentials cannot.

- [ ] **Step 3: Add numeric edge cases**

Per kiss-ref's owner: graph-construction bugs and numeric-handling bugs surface on **different inputs**. Include a −0.0, a subnormal, and a large-magnitude row. (True bit-pattern pinning needs `ConstBits` from 0.2.0; approximate with constructible values for now and record the limitation.)

- [ ] **Step 4: Run, fix, commit**

- [ ] **Step 5: Report friction to `3vgwagtz`**

They explicitly asked for where the recipe grammar fights the shape rather than a workaround. Report it even if everything works — a clean run is also signal.

---

## What Phase 0 deliberately does NOT cover

Each becomes its own plan, written **after** Task 3 records verified signatures:

- **Phase 1 — the attention-block recipe.** The interesting fragment: softmax-over-scores, causal mask (prefer the additive −inf form), KV gather (`Node::Gather` with an `IndexRef`; a computed index escalates `DetClass` to `OrderInvariantNondeterministic`, so tolerance-compare downstream).
- **Phase 2 — falsifier #1**, the gating experiment: is the attention column-sum obtainable on the fused arm? Decides whether attention-driven eviction survives the port.
- **Phase 3 — `model_fuel/` vertical slice**, capture-shaped from the first commit.
- **Phase 4 — batched decode on the KV allocator.**
- **Phase 5 — upstreaming** to Fuel: stateful H2O, KIVI granularity, relationship-aware compression, routing observability.
- **Phase 6 — deletion**: `multi_gpu/`, Marlin FFI, `awq_qwen3.rs`, frozen `model/`, `candlelight`.
- **Tier 3 goldens** — the Candle-captured regression net. Cheap once the library compiles (it does), but explicitly *not* independent, so it is not a Phase 0 gate.
