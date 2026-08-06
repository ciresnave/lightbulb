# Falsifier #1 — is the attention column-sum obtainable on the fused arm?

**Date**: 2026-07-29. **Tree**: `C:\Projects\fuel-lightbulb-port` @ `13279179`.
**Question**: H2O and R-KV need `a_t[k] = Σ_q probs[q][k]` — the per-key column-sum of the
post-softmax attention weights, every step, every layer. Is that obtainable on the arm the
port actually wants to run?

**Evidence level**: `[read]` from source. **Nothing here was executed.** The cost argument in
§3 is an algorithmic claim about FlashAttention's structure, not a measurement, and the
kernel-feasibility question in §4 belongs to whoever owns the CUDA kernel.

---

## 1. Verdict: CONFIRMED — not obtainable on the fused arm, by design

`fuel-graph/src/registry/flash_attn.rs`, module header, verbatim:

> *"Attention does have a primitive decomposition (`matmul → softmax → matmul`, with masking
> + scaling), but FlashAttn's value is specifically that it **avoids** materializing the
> `[B, Hq, Sq, Sk]` attention matrix — a primitive lowering would defeat the purpose... A
> graph-level `decompose` to a primitive subgraph would be a footgun: it would either
> reproduce the very memory blowup FlashAttn exists to avoid, or pretend the primitive form
> is equivalent when it isn't (the tiled softmax in the kernel produces different numerics
> than the naive form)."*

So on the FlashAttn arm there is **no `probs` node to attach a reduction to**. This is not an
oversight to be fixed; it is the entire reason the op exists.

**On the decomposed arm the matrix does exist.** `:230–285` builds
`scores = scale·(q·kᵀ)` → softcap → alibi → mask → `probs = softmax(...)` as real nodes.
`probs` (`:285`) is exactly the tensor H2O needs to reduce. Note it is `probs`, *post*-softmax
— not `scores`. An earlier framing of this question said "scores", which would have been the
wrong tensor.

---

## 2. What this means for Lightbulb today

**Attention-driven eviction and the fused arm are mutually exclusive, absent a kernel change.**
That is a **C-5 constraint the consumer sets per deployment**, not an arm Fuel picks:

| Deployment wants | Arm | Consequence |
| --- | --- | --- |
| H2O / R-KV attention-driven eviction | **decomposed** | pays `[B, Hq, Sq, Sk]` materialization; loses FlashAttn's memory profile |
| Maximum decode throughput | **fused** | no attention observability; eviction must use a policy that doesn't need it (recency, streaming sinks, segmented) |

This is a real configuration surface the port must design, and it is cheaper to know now
than after `model_fuel/` is written. It also means **Lightbulb's eviction policy set is not
uniformly available** — `h2o_policy` and R-KV are arm-gated, while `streaming_policy`,
`eviction_policy`'s recency arm, and `segmented_eviction_policy` are not.

---

## 3. The route that is *not* "state the incompatibility"

§15's preference #4 (state it plainly, let consumers choose) is the fallback. But preference
#2 — **a second output from the producing op** — is better-supported than expected.

**The mechanism exists and is in use.** `output_views` is Fuel's multi-output hook
(`12-multi-output`). `Graph::set_output_views` enforces five documented invariants (non-empty
slot list, slot 0's dtype and shape equal the node's, each slot's layout shape matches, and
idempotent replacement). **Two registry entries already declare it**:
`registry/selective_scan.rs:130` and `registry/ssd_chunk_scan.rs:119`, each via an
`output_views(input_shapes, input_dtypes, params) -> Vec<OutputViewSpec>` function.
`flash_attn.rs` declares `output_views: None`.

**The cost argument — why this does not defeat FlashAttn's purpose.** This is the crux:

- H2O wants a reduction over the **query** axis, per key.
- FlashAttention tiles over **keys**, maintaining per-query running max/sum (that is what
  `softmax_lse` is — flash_attn's signature already carries it, `:67`).
- So the wanted reduction is along the axis flash *streams* over — orthogonal to what it
  naturally accumulates, which is why it is not simply free.
- **But** within each K/V tile the `[Sq, tile]` probs block exists transiently in fast memory.
  Summing over `q` within the tile yields partial column-sums for those keys; accumulating
  into an `[Sk]` output costs **O(Sk)** extra memory, not **O(Sq·Sk)**.

**That is ~1/Sq the size of the attention matrix.** The blowup FlashAttn exists to avoid is
not reintroduced. So the ask is bounded: *a second output slot carrying an `[Sk]` per-key
column-sum, accumulated per tile.*

---

## 4. What this does NOT establish

Stated explicitly, because the temptation is to treat §3 as a plan:

- **I have not verified that the CUDA kernel can be modified this way, or what it would cost
  in practice.** §3 reasons about FlashAttention's algorithmic structure. Register pressure,
  occupancy, and whether the tile-local reduction serializes anything are questions for
  whoever owns the kernel (Baracuda, routed through Fuel).
- **I have not executed anything here.** No benchmark of decomposed-vs-fused, no measurement
  of the materialization cost at our shapes.
- **`softmax_lse` appears in flash_attn's *input* signature, not its output.** It shows
  auxiliary attention statistics are an established concept for this op; it is not itself
  evidence that a second *output* is easy.

---

## 5. Recommended sequence

1. **Design the C-5 arm-selection surface now.** Whatever happens upstream, Lightbulb needs
   to express "this deployment requires attention observability" and have that select the
   decomposed arm. Without it, a deployment silently gets one or the other.
2. **Benchmark decomposed-vs-fused at our shapes** before deciding the default. The trade is
   measurable, not arguable — §15's C-4 makes measured cost the record.
3. **Raise the second-output ask with Fuel** as a bounded, mechanism-backed request rather
   than a wish: `output_views` exists, two ops use it, and the cost is O(Sk).
4. **Keep the in-graph reduction design** (`2026-07-29-attention-reduction-in-graph.md`) — it
   is correct *for the decomposed arm*, where `probs` is a real node. Nothing in this finding
   invalidates it; it bounds where it applies.
