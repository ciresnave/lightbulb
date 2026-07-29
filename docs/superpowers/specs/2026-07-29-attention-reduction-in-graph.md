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
- **Runtime-offset accumulate may not compose with capture.** If the write offsets vary per
  step in a way that forces a graph rebuild, the capture requirement is violated by the fix.
- **Decay may need to be per-slot rather than uniform**, if slots age independently — that
  changes `inplace_affine` into something with a per-element scale vector.

## Next step — revised 2026-07-29

Falsifier #1 is answered, so the ordering changes. **Do not start with H2O.**

**Experiment 1 (cheapest, decisive, no backend work).** Take the **decomposed** attention
arm, add the column-sum, and check whether `CapturedRun` still captures the step. This tests
falsifier #2 — runtime-offset accumulate vs. capture — *independently* of the arm question.
If capture breaks on a runtime-offset accumulate, the arm question is moot and the whole
route dies early and cheaply. One variable at a time. (Experiment design credit: Fuel's seam
owner; it is better than the ordering this document originally proposed.)

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
