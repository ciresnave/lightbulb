# Attention-driven eviction as an in-graph reduction — consumer specification

**Status**: draft, 2026-07-29. Written for Fuel's C-6 hot-path regime (§15 v0.3).
**Evidence status**: derived from Lightbulb's current implementation. **Not validated by a
running port.** The claim that these reductions are expressible in-graph is a conjecture
this document exists to make testable, not a result.

**Context**: §15 v0.3 added a hot-path regime to C-6 after Lightbulb's audit found that its
only load-bearing mid-forward value extractions are attention-score observations at
per-token, per-layer cadence — which is incompatible with `CapturedRun`'s stable-graph
requirement. The leading resolution is **ask for the reduction, not the intermediate**.
This document specifies the reductions.

---

## REVISED 2026-07-29 — the reduction is moving into the kernel

**This document was written assuming Lightbulb would express the column-sum itself as an
in-graph reduction over a materialised `probs`. That is no longer the plan for the fused arm.**

Baracuda is adding it as a **second output of the FlashDecoding kernel**, opt-in, deterministic:

| Output | Shape | Notes |
| --- | --- | --- |
| `a` | `Option<[B, H_q, Sk]>` | per-head; **free** — each Q-head is its own thread-block (grid.y = H_q), so writes are disjoint, no atomics. Built, 10/10 correct incl. multi-split + GQA |
| `a_mean` | `Option<[B, Sk]>` | separate deterministic head-mean kernel (fixed-order sum over H ÷ H). **Drop-in for `custom_transformer.rs:565`, which already does `.mean(1)`** |

`None` = zero cost. **Mean, not sum** (÷H in-kernel) — do not re-divide by H on the consumer
side. Batch-mean stays host-side (B=1 at decode makes `.mean(0)` a no-op).

**Determinism is why the head-mean is a second kernel rather than an atomicAdd.** A statistic
that varies between identical runs is not a reference: it would make eviction trajectories
unreproducible and undebuggable, and break tier-3 goldens. Fixed-order summation preserves
reproducibility; atomics would have destroyed it silently.

### What this changes in the design below

**The reduction half is no longer ours.** `a_t` arrives as a kernel output. What remains on the
Lightbulb side is **only the accumulation**:

```
c_t = (c_{t-1} · decay + a_t) ⊙ occ_t
```

— a fixed-shape `[max_slots]` elementwise in-place affine plus an occupancy mask, which is the
`inplace_affine` + persistent-buffer shape §"What this asks of Fuel" already identified. **No
`probs` observation, no fusion break, no arm dependence for the accumulator itself.**

**So the falsifier-#1 conclusion narrows to a cost question.** Whether attention-driven eviction
is affordable on the fused arm is now three measured configs on a 4070 (baseline / per-head
a-store / + head-mean), not an argument. If the combined cost is small, H2O stops being
arm-gated and the C-5 constraint softens; if large, deployments choose.

**What is still correct below**: the H2O recurrence, the per-step-vs-on-eviction split, the
occupancy-mask requirement for slot reuse, `insertion_step` unifying `n_t` and `occ_t`, and the
whole in-graph-reduction route **for the decomposed arm**, where `probs` is a real node and no
kernel change is needed.

---

## The central observation

**The accumulation runs every step. The scoring runs only when evicting.**

Lightbulb currently conflates them — it realizes attention weights every step *and* keeps
all the policy on the host. Only the accumulation is hot. Separating them is what makes
this tractable: the per-step half is a small tensor recurrence that can live in the graph;
the on-demand half is host-side policy over a vector of at most `num_slots` floats, run
rarely, and stays with the consumer where §15 puts it.

---

## H2O — the exact recurrence

From `src/cache/h2o_policy.rs::update_attention_scores`. Per decode step *t*, for key
position *k*:

```
a_t[k] = Σ_q  attn[q][k]              # column sum: attention received, summed over queries
c_t[k] = decay · c_{t-1}[k] + a_t[k]  # decayed running sum        (decay = 0.95)
n_t[k] = n_{t-1}[k] + 1               # steps present
```

`c` and `n` are the **only** state carried across steps. Everything else is derived.

### What must be in-graph (per step, hot)

| Quantity | Shape | Op |
| --- | --- | --- |
| `a_t` | `[key_len]` | reduce-sum of the attention matrix over the query axis |
| `c_t` | `[max_slots]` persistent | `c * decay + a` — read-modify-write at runtime offsets |
| `n_t` | `[max_slots]` persistent | increment at runtime offsets (or derive: `t - insertion_step[k]`, removing the tensor entirely) |

**`c_t`'s update is structurally a KV write** — a runtime-offset accumulate into a
persistent buffer that survives across realize calls. That is the machinery
`InferenceContext` + `Op::WriteSlice` already provide for K/V, pointed at a statistics
tensor instead of a cache. The arithmetic `c·decay + a` looks like Fuel's existing
`inplace_affine` registry entry.

**Note `n_t` may not need to exist.** `steps_present` is just `t − insertion_step[k]`, and
insertion step is already implied by slot position. If so, the per-step state reduces to a
single `[max_slots]` f32 tensor and one fused update. That is the smallest version of this
and worth confirming before building the larger one.

### What stays on the host (on eviction only, cold)

From `compute_eviction_scores`:

```
avg[k]   = c[k] / n[k]
score[k] = 1 / avg[k]        (+∞ when avg == 0 — never attended)
score[k] = −∞                when position > max_position − num_recent_to_keep
```

Sink protection, the reciprocal inversion, ranking, and the eviction decision itself are
**policy** and stay with the consumer per §15. They read `c` once per eviction — an
occasional C-6 observation of a `[max_slots]` vector, which is regime 1 and fine.

---

## R-KV — the reduction

From `src/cache/kv_compression.rs:595`:

```
summed = attention_scores.sum(1)     # sum over the head axis → [batch,1,1,seq_len]
importance = summed[..seq_len]
```

Structurally the same shape of ask: a reduce-sum over one axis, no raw scores needed.
Redundancy scoring (cosine similarity between keys) and `budget_fraction` selection are
host-side policy over already-reduced vectors.

**Open question, stated rather than assumed:** whether R-KV's attention input is captured
per-step (hot, regime 2) or only at compression time (occasional, regime 1). Lightbulb's
current call path does not make this obvious, and it changes whether R-KV needs the in-graph
treatment at all. To be resolved during the port.

---

## What this asks of Fuel

Nothing new, if the conjecture holds:

1. A reduce-sum over one axis of the attention tensor — already expressible.
2. A persistent `[max_slots]` f32 buffer surviving across realize calls — the
   `InferenceContext` pattern.
3. A runtime-offset read-modify-write `c·decay + a` — `Op::WriteSlice` shape, and
   possibly `inplace_affine` directly.

If all three hold, **nothing is observed, nothing breaks fusion, and the decode step stays
capture-shaped.** H2O's statefulness stops being consumer bookkeeping to upstream and
becomes a graph construct — a better outcome than either "upstream our stateful H2O" or
"accept the deoptimization."

## How this could fail

Stated because the conjecture deserves its falsifiers:

- **~~The attention matrix may not be materialized at all.~~ CONFIRMED, conditionally —
  2026-07-29, and this is the important one.** **[verified by Fuel]**
  `registry/flash_attn.rs:235` builds `scores = scale · (q · kᵀ)` as a real `MatMul` node
  feeding `softmax(mask(alibi(softcap(·))))`. So the matrix **is** materialized on the
  **decomposed** arm and **is not** on the **fused** arm (same file, `:31`, notes the tiled
  form even produces different numerics).

  **Consequence: observability is arm-dependent, which makes this partly a C-5 question,
  not purely C-6.** The in-graph reduction works *today* on the decomposed arm — but
  choosing it may silently cost the fused arm. That is a legitimate, *measurable* trade:
  benchmark decomposed-plus-reduction against fused-without-observation and pick. C-5 makes
  the choice explicit rather than accidental; C-4 makes it measurable. **Nothing blocks
  trying it.**

  The general rule, now in §15 v0.4: *before promising an in-graph reduction, check the
  reduced value survives the arm the consumer wants to run on.*

  **The second-output route is also less exotic than assumed.** `flash_attn` already carries
  an optional **`softmax_lse`** in its input signature (`:67` — "takes 4 or 5 inputs (q, k,
  v, [softmax_lse], [alibi])"). Auxiliary attention statistics are an *established* shape
  for these kernels, so a column-sum alongside the output is a backend ask with precedent,
  not an invention.
- **~~Runtime-offset accumulate may not compose with capture.~~ DISSOLVED on paper —
  2026-07-29.** **[verified by Fuel, by reading; not executed]** Three legs:
  (1) the `CapturedRun` invariant is **zero `cuMemAlloc` on *replay***, not "no
  runtime-varying values" (`baracuda/attention.rs:1386`) — allocating on first launch is
  fine; (2) `Op::WriteSliceDoff` takes its write start from a **device-resident rank-0 `I64`
  operand**, so a captured graph replays at the host-updated position — this is how KV append
  already works inside captured decode; (3) `affine_inplace_{f32,f64}` is bound to baracuda
  single-pointer kernels (`baracuda_dispatch.rs:1959–1963`).

  **Stronger: the accumulator needs no offset machinery at all.** Against the
  **capacity-shaped** KV buffer (fixed capacity + runtime valid length), the column-sum is
  naturally `[max_slots]`-shaped, with masked positions contributing zero post-softmax. So
  `a_t` is fixed-shape, `c_t = c_{t-1}·decay + a_t` is a fixed-shape in-place elementwise
  affine, and there is **no dynamic extent, no slice, no offset** in the recurrence. `c` is
  allocated once and reused forever, satisfying the invariant trivially.

  **Still unexecuted.** The decisive test — build the accumulator, launch twice, assert
  `allocation_count == 1` — needs a **live CUDA device**, because the invariant is about
  `cuMemAlloc` and CPU cannot answer it.

- **Slot reuse breaks the naive fixed-shape form — a correctness requirement, found
  2026-07-29 and then corrected.** The fixed-shape form has no removal, so a recycled slot's
  new occupant inherits the previous occupant's decayed history. Worse for `n_t`: it would
  report the prior tenure, inflating the denominator in `avg = c/n` so a brand-new token
  looks long-lived and low-attention — exactly the profile H2O evicts first, risking
  immediate re-eviction of freshly-admitted tokens. That is a correctness failure, not drift.

  **Correction — this repairs a semantic that does NOT exist today, in either form.**
  I originally cited `h2o_policy.rs:209`'s `clear_slot`, called from
  `parallel_cache_builder.rs:1909`, as evidence that token-level slot reuse is live. It is
  not. **[verified]** That call sits inside `reset_batch_index`, whose doc reads *"Request in
  slot 0 finished, starting new request"* over `0..batch_size` — **sequence-level** reuse of
  a batch row. The token-level path is `should_clear_slot` → KV `clear_slot()`, and `:2048`
  states *"CURRENT STATUS: Not actually used yet since `clear_slot()` is a stub."*

  **Consequence: Lightbulb's current implementation is not an oracle for this.** The mask
  needs its own test, not a differential comparison against present behaviour.

  **Open question, not a claim:** `update_attention_scores` keys `slot_metadata` by
  `slot_id` from `cache_positions` (slot → seq_position), while `reset_batch_index` calls
  `clear_slot(batch_index)` keyed by batch row. If those index spaces differ, `clear_slot`
  clears the wrong entries. They may coincide — the `[max_batch, heads, seq, head_dim]`
  cache shape hints at one row per batch slot. **Unresolved; resolve during the port.** If
  they diverge, the existing reset behaviour is wrong rather than merely incomplete, and the
  in-graph form must not inherit its structure.

  **The generalization (Fuel's, and larger than the mask):** token-level slot reuse goes
  live *precisely when the block-pool allocator lands* — refcount-aware evict plus block
  reuse **is** that mechanism. So **any per-slot side buffer above the allocator inherits
  stale state unless slot-recycle is observable at the allocator boundary.** That is a
  mechanism obligation; "which slots recycled" is information only the allocator has, and it
  composes with `{freed, still_shared}`. Prefix-cache hit accounting and per-slot C-4
  attribution have the same exposure. Routed to the allocator session as part-2 input.

  **Fix, preserving everything above** — an occupancy mask, elementwise and fixed-shape:

  ```
  c_t = (c_{t-1} · decay + a_t) ⊙ occ_t
  ```

  `occ_t` is `[max_slots]` 0/1, zeroed at a slot on admission. No offsets, no dynamic extent,
  so the zero-`cuMemAlloc` argument is undisturbed. **This also sharpens the `n_t` collapse**:
  if `insertion_step` is a `[max_slots]` tensor written on admission, then `n_t` is derived
  *and* `occ_t` is `insertion_step ≥ 0` — one buffer solves both. **The mask term is itself
  unvalidated**; admission may already zero the region, making it redundant.

- **Decay may need to be per-slot rather than uniform**, if slots age independently — that
  changes `inplace_affine` into something with a per-element scale vector. **Downgraded, not
  resolved**: it changes *which op*, not whether capture holds, since it stays fixed-shape,
  elementwise, and in-place-able.

## Next step — revised 2026-07-29

Falsifier #1 is answered, so the ordering changes. **Do not start with H2O.**

**Experiment 1 — SUPERSEDED 2026-07-29.** Falsifier #2 was answered on paper (see above),
so the capture question is no longer the gate. It still wants executing on a live CUDA
device eventually — build the `[max_slots]` accumulator, launch twice, assert
`allocation_count == 1`, following the existing
`fused_rope_is_capture_safe_zero_alloc_on_reuse` pattern — but the analysis says it will
pass, and it is no longer the cheapest way to kill the route.

**Experiment 1 (revised) — the arm question. This is now the gate.** Determine whether
`a_t` is obtainable on the arm we actually intend to run. The matrix is materialized on the
decomposed arm and not on the fused arm, so this decides whether attention-driven eviction
survives the port at all. **Run this before any `model_fuel/` code exists.**

**Consequence to design for if the answer is "decomposed only":** Lightbulb must choose
between attention-driven eviction and the fused arm *per deployment*. That makes it a **C-5
constraint the consumer sets**, not an arm Fuel picks — which is a configuration surface
this project would need to design, and is much cheaper to know now than to discover after
the rewrite.

**Experiment 2.** Express the H2O recurrence as graph nodes. Confirm the
`n_t = t − insertion_step[k]` collapse early — it is the difference between one fused update
and two.

**Experiment 3.** Benchmark decomposed-plus-reduction against fused-without-observation.
This is the C-5 trade, and it is a measurement, not an argument.

**Verified available** (so "plausibly zero new primitives" holds for the *accumulate* half):
`Op::WriteSlice` at `fuel-ir/src/dispatch.rs:398`, and `registry/inplace_affine.rs`. The
open question is entirely the **source** of `a_t` — i.e. the arm question.

**Fallback** if all routes fail: §15's preference #4 — state the incompatibility plainly so
consumers choose deliberately rather than discovering it late.
