//! Batched paged decode — **B sequences, one graph, one realize**, over Fuel's
//! `DeviceKvPool`.
//!
//! # What Fuel gives us and what it does not
//!
//! Fuel ships the *storage* half of paged decode: [`DeviceKvPool`] owns the
//! `[num_blocks, block_size, Hkv, D]` pool buffers, the block tables, and
//! `Op::PagedAttn`. It also ships `LlamaModel::forward_paged_step` — but that is
//! **one token, one session**, and Fuel's own `PagedSessionScheduler` drives it
//! in a serial per-session loop. There is no batched paged decode in Fuel at the
//! HEAD this was written against (`build_decode_attn` is single-session; there is
//! no `build_decode_attn_batched`). This module is that missing piece.
//!
//! The split is unchanged from `generate.rs`:
//!
//! | | |
//! | --- | --- |
//! | **Fuel** | the forward pass, the pool, the graph, `Op::PagedAttn` |
//! | **Lightbulb** | which token to pick, when to stop, **whether a request is admitted** |
//!
//! [`BatchedPagedDecoder::step`] returns **realized `Vec<f32>` logits, one row per
//! sequence**. Sampling never enters this file — per Fuel's consumer contract
//! §15 (and its own `DecodeModel` trait, which pointedly omits
//! `SamplingStrategy`), token selection is consumer policy. Likewise admission:
//! [`BatchedPagedDecoder::can_admit`] *asks the pool* via
//! `blocks_required_batch`/`free_blocks`, and the accept/reject decision is the
//! caller's.
//!
//! # Why prefill is one token at a time (and batching still pays)
//!
//! `Op::PagedAttn` is decode-only, `Sq == 1`. **Nothing validates that** —
//! `Tensor::paged_attn` reads `sq` and never checks it — so it is a silent
//! wrong-answer trap rather than an error. The reason it is real:
//!
//! - the CPU fast kernel **is** causal at `Sq>1`
//!   (`fuel-cpu-backend byte_kernels.rs`: `q_pos_abs = ctx_len + qi - sq`, then
//!   `if kj > q_pos_abs { continue; }`), but
//! - the registry DECOMPOSE recipe (`fuel-graph/src/registry/paged_attn.rs`
//!   ~229-238) builds **only** the variable-length mask
//!   `Ge(key_pos, context_len) → −inf`. There is no `q_pos` term at all.
//!
//! At `Sq == 1` the two coincide exactly. At `Sq > 1` the decomposed form lets
//! every query attend to every key — the answer depends on which lowering ran.
//! So this module asserts `Sq == 1` defensively ([`BatchedPagedDecoder::step`])
//! and feeds prompts token by token.
//!
//! The batched win is therefore **across sequences, not within one**: a batch of
//! `B` prompts costs `max(P_i)` steps at width `B` instead of `sum(P_i)` steps at
//! width 1. That matters more here than on the contiguous path, because the
//! paged path has no plan-once `DecodeSession` (Fuel's `inference_context.rs`
//! contains no paged arm at all) and re-optimises the graph every step.
//! Batching was expected to amortise that one optimise over `B` tokens.
//! **It does not — see the sweep below.**
//!
//! **This is a correctness deliverable, not a speedup. Measured, it is a large
//! slowdown that batching does not fix.** TinyLlama-1.1B, CPU, release, all-f32
//! weights, both arms in one process off one checkpoint load
//! (`b_sweep_per_sequence_cost`):
//!
//! | path | per step | per sequence-token | vs baseline |
//! | --- | --- | --- | --- |
//! | contiguous + persistent `DecodeSession`, B=1 | 977.420 ms | **977.420 ms** | 1.00× |
//! | batched paged, B=1 | 5.452 s | **5.452 s** | 5.58× |
//! | batched paged, B=2 | 21.724 s | **10.862 s** | 11.11× |
//! | batched paged, B=4 | 37.555 s | **9.389 s** | 9.61× |
//!
//! **Per-sequence cost does not fall with `B`.** It nearly doubles from B=1 to
//! B=2 and only partly recovers at B=4. A *fixed* re-plan cost `c` amortised
//! over `B` rows would give `c/B + marginal` — monotonically decreasing. The
//! observed shape refutes that: per-step cost rises 3.99× for 2× the rows
//! (B=1→2), then 1.73× for 2× the rows (B=2→4). **The re-plan cost grows with
//! `B`**, so batching enlarges the very thing it was supposed to amortise.
//! Under fixed `c` a large enough batch eventually wins; under `c(B)`, none does.
//!
//! B=8 is unmeasured (harness timeout), and cannot change the conclusion: per-step
//! cost is non-decreasing in `B`, so `per_seq(8) = step(8)/8 ≥ 37.555/8 = 4.69 s`
//! — still 4.8× the baseline in the impossible best case where four extra rows
//! cost nothing.
//!
//! A second run on a quiet machine reproduced every arm within 2–3 % (5.452 →
//! 5.344 s, 21.724 → 21.212 s, 37.555 → 37.124 s, baseline 977 → 950 ms). The
//! harness is stable to a couple of percent; the 74–99 % swings seen earlier
//! were machine contention and nothing else.
//!
//! **The warm-up/steady split identifies *which* cost this is.** Each arm
//! reports its first step separately: a plan-once path pays for the plan on
//! step 0 and replays afterwards, a re-planning path pays every step.
//!
//! | arm | warm-up | steady | ratio |
//! | --- | --- | --- | --- |
//! | contiguous persistent, B=1 | 4.956 s | 949.994 ms | **5.22×** |
//! | batched paged, B=1 | 5.295 s | 5.344 s | 0.99× |
//! | batched paged, B=2 | 21.041 s | 21.212 s | 0.99× |
//! | batched paged, B=4 | 37.442 s | 37.124 s | 1.01× |
//!
//! Two things are measured here, and it is worth separating them from what is
//! merely inferred, because the inference is load-bearing and easy to overstate.
//!
//! **Measured.** The paged path has **no first-step premium at any `B`**
//! (0.99/0.99/1.01), so it re-plans every token. The contiguous path has a
//! 5.22× premium, so it plans once and replays. On the contiguous arm both
//! terms are directly observable — step 0 is plan+execute and the steady steps
//! are execute-only — giving **planning ≈ 4.0 s and execution ≈ 0.95 s**.
//! Independently, paged B=1 (5.344 s) costs within 8 % of the contiguous
//! *first* step (4.956 s): one plan plus one row's execution, every token.
//!
//! **Inferred, and only under an assumption.** Splitting the *paged* step into
//! plan and execute needs an assumption, because nothing on this path isolates
//! them — that is precisely the missing mechanism. Assuming paged execution
//! costs the same ~0.95 s per row as contiguous execution:
//!
//! ```text
//! c(B=1) ≈ 4.4 s      c(B=2) ≈ 19.2 s      c(B=4) ≈ 33.1 s
//! ```
//!
//! **Do not treat that assumption as confirmed by the numbers reproducing.**
//! `c(B) ≡ step(B) − B·0.95 s` by construction, so `c(B)/B + 0.95 s` is
//! identically `step(B)/B` — an algebraic identity, not a check. It would
//! "reproduce" the data for *any* assumed execution cost. Likewise, dividing
//! `c(B)` back out to conclude that paged execution is at parity with
//! contiguous merely recovers the input assumption.
//!
//! What *is* safe: the planner dominates. Even bounding paged execution
//! generously, step 0 of the contiguous arm shows a ~4 s plan against a ~0.95 s
//! execute on the same model and machine, and the paged arm pays a
//! contiguous-sized first step on every token.
//!
//! **`c(B)`'s shape is not characterised.** Per-row planning runs 4.4 → 9.6 →
//! 8.3 s at B=1/2/4 — non-monotonic, worst at B=2. A cost that quadruples on
//! the first doubling and then grows 1.7× on the second could be a threshold, a
//! fallback engaging at B≥2, or an artifact at a single point. B=3 and B=8 would
//! settle it. None of this affects the verdict, which rests on the measured
//! warm-up ratios, not on `c(B)`.
//!
//! **Read this as a property of *this* paged path, not of paging.** The
//! **Cause of the measured re-planning, separately from the blocker on fixing
//! it.** These are two different facts and it is easy to credit the wrong one.
//!
//! *Cause, and it is sufficient on its own:* `forward_paged_step_batched` mints
//! a fresh graph root every call (`Tensor::from_f32`, `fuel-core/src/
//! lazy.rs:7599` in the `fuel-lightbulb-port` worktree), and by rule 1 every
//! `from_*` starts a NEW graph. Position enters concretely too (`tok_pos`, a
//! plain `usize`, :7540). Note this sweep's geometry: prompt 8 + 5 steps = final
//! position 13 against `BLOCK = 16`, so `max_blocks == 1` for every session at
//! every `B` and the block-table shape was `[B, 1]` for the entire run. **The
//! shape never varied and it re-planned anyway.** Nothing about block tables is
//! needed to explain the 0.99/0.99/1.01 ratios.
//!
//! *Blocker on the fix, which did not fire here:* `DeviceKvPool::
//! block_table_shape` is `[batch, max_blocks]`, and `max_blocks` grows as
//! sessions cross block boundaries. So once `L` crosses a boundary the graph is
//! a different *shape*, and plan reuse stops being merely unimplemented and
//! becomes **ill-defined** — there is no single plan to cache. The contiguous
//! arm escapes this by carrying KV extents symbolically (`cached_len_sym` /
//! `attended_len_sym`, `fuel_ir::SymId`), which is why its shape is stable
//! enough to replay. A paged `DecodeSession` therefore needs symbolic
//! block-table extents *first* — rule 3(a) below, with a working precedent in
//! the same file.
//!
//! **The next measurement, and its scope fixed in advance.** Within one block
//! window the paged graph is not merely shape-stable but *structure*-stable —
//! same nodes, same ops, with `context_lens` carrying length as data rather than
//! shape — so planning should be near-flat in `L`, stepping only at boundaries.
//! Attention execution meanwhile grows with KV span. Writing
//! `step(L) = P + a·L + b`, where `a·L` is attention over the KV span and `b` is
//! the `L`-invariant remainder (FFN, projections, embed, logits, norms):
//!
//! An `L`-sweep recovers the **slope `a`** and the **intercept `P + b`**. It
//! **cannot split that intercept** — planning and `L`-invariant execution are
//! both constant in `L` and so perfectly confounded by this method. At this
//! module's geometry the confound is not marginal but dominant: `L` runs 8→13,
//! and attention over ~13 KV entries is a rounding error against a full
//! TinyLlama FFN and four projections per layer, so `b ≫ a·L` and the intercept
//! is nearly all `b`.
//!
//! The tempting repair is to take `b` from the contiguous arm, where `P` and `E`
//! *are* separately observable, and subtract. **Refuse it.** That imports an
//! unmeasured cross-arm assumption — that paged and contiguous share
//! `L`-invariant execution cost — which is the same move that produced the
//! circular 0.95 s/row result above. It may well be true; it would not be
//! measured.
//!
//! So: an `L`-sweep is worth running to *confirm the mechanism* (planning flat,
//! execution growing — corroborating fresh-graph-per-call independently of the
//! warm-up ratios) and to get `a` cleanly. It is **not** a route to the absolute
//! plan/execute split. The better measurement is the **boundary step**: step
//! height at a block crossing gives the marginal planning cost of one additional
//! block *directly*, as a difference between two measurements at nearly equal
//! `L` — no intercept, no decomposition, no cross-arm assumption. If only one
//! thing gets run, run that.
//!
//! **Read this as a property of *this* paged path, not of paging.** The
//! contiguous arm holds a plan-once `DecodeSession`; the paged arm has none and
//! re-plans every step (rule 3 below). So the comparison is plan-once-versus-
//! re-plan at least as much as it is contiguous-versus-paged — which is the
//! intended measurement, because no paged plan-once path exists to measure. It
//! bounds what paged decode costs *today*; what it argues for is a paged
//! `DecodeSession` upstream in Fuel, not abandoning paging. Fuel's own
//! contiguous tiers show the size of the prize: re-plan-per-step to captured
//! replay was originally cited at ~10.4× on TinyLlama/4070. That figure did
//! NOT survive isolated measurement: with persistence held constant, capture is
//! ~4× (Lightbulb 3.69× median, paired n=3; Fuel 4.28× independently — see
//! model_fuel/mod.rs rule 3 for the full account). The 10.4× almost certainly
//! bundled persistence with capture.
//!
//! Note this supersedes an earlier figure of 4.52 s/token for the contiguous
//! baseline, which compared **bf16 projections measured on another day** against
//! this module's f32 numbers and understated the gap ~4.6×. The paged side
//! reproduced almost exactly across that gap (21.73 → 21.724 s/step at B=2);
//! it was the baseline that was wrong. Hold dtype, process and machine fixed,
//! or do not compare.
//!
//! # The three rules, as they land here
//!
//! 1. **Graph affinity.** The token-embedding table is the root
//!    (`Tensor::from_f32`) and *everything* — weights, RoPE tables, the u32
//!    block table, and the pool placeholders — is `const_*_like` off it.
//! 2. **F32.** [`BatchedPagedDecoder::new`] rejects any non-`F32` projection at
//!    construction — but **not for the reason this doc used to give.** It said
//!    `forward_paged_step` is "hard-gated to f32". That was true once and is now
//!    false: all three paged forwards accept `F32 | BF16 | F16`
//!    (`fuel-core/src/lazy.rs` :7530, :7670, :7983). Fuel is not gating us.
//!
//!    Two real reasons remain, both verified against Fuel's code rather than its
//!    comments:
//!
//!    * **`Op::PagedAttn` has no CUDA and no Vulkan kernel** — it is CPU-only in
//!      Fuel's binding table, and candidate enumeration filters by that table, so
//!      a `PagedAttn` node has no GPU candidate to be placed on. Paged attention
//!      therefore runs on CPU whatever backends are compiled in, and f32 avoids
//!      paying a promoting cast on that CPU fallback for nothing. (Wiring a CUDA
//!      `PagedAttn` kernel into the binding table is Fuel's PC-3.)
//!    * **Our tiering uses the f32-typed pool API.** `write_block` / `read_block`
//!      reject a non-F32 pool and direct callers to `write_block_bytes` /
//!      `read_block_bytes`. So a bf16 pool is possible, but [`crate::model_fuel::policies::BlockTierMover`]
//!      would have to move bytes instead of `f32`s — a real change, not a flag.
//!
//!    Note this rule is **specific to the paged path**. The contiguous path's
//!    f32 choice is a CPU artifact and is separately relaxable: Fuel's
//!    `insert_dtype_fixups` keeps bf16 where the kernel table serves the mixed
//!    key natively and inserts a cast only where it does not.
//! 3. **Capture-shaped decode.** *Not achieved here, and said plainly.* Each
//!    `step` builds a fresh graph. Plan-once needs (a) a fixed-width block table
//!    (`materialize_block_table`'s `max_blocks` grows when any session crosses a
//!    block boundary, which changes a tensor *shape*), and (b) dynamic
//!    `(phys, slot)` write offsets, which needs `write_slice_dyn` on **two** axes
//!    or a reshape-of-placeholder trick that is not verified to preserve in-place
//!    adoption. Batching is the amortisation available today.
//!
//! # The one piece of math reimplemented here, and why
//!
//! `Tensor::rope_with_tables_decomposed` requires `cos`/`sin` to be **exactly
//! `[seq, d]`** where `seq = dims[rank-2]`. At `q = [B, Hq, 1, D]` that is
//! `[1, D]` — *one shared position for the whole batch*. Batched decode has `B`
//! sequences at `B` different absolute positions, so that call is unusable at
//! `B > 1`.
//!
//! [`rope_batched`] reimplements the rotate-half application over a
//! `[B, 1, 1, D]` table. It does **not** reimplement the table math: the per-row
//! cos/sin come from Fuel's own `Tensor::rope_tables_const` (which delegates
//! to `fuel_graph::build_rope_tables`), concatenated along the batch axis. So the
//! only new code is `x*cos + rotate_half(x)*sin`, and it is gated by a test that
//! at `B == 1` it is bit-identical to `rope_with_tables_decomposed` — plus a
//! control that a *shared* position is caught, so the per-row table is proven
//! load-bearing rather than accidentally masked by short sequences.
//!
//! ## A trap worth knowing before you write a test for any of this
//!
//! **RoPE is relative**: `q(m)·k(n)` depends only on `m − n`. Adding a constant
//! to *every* position of a sequence — queries and cached keys alike — changes
//! nothing observable in the logits. That is what RoPE is *for*, and it means an
//! end-to-end parity test can be **completely blind** to per-row positions.
//!
//! Concretely (measured, not reasoned): with sequences decoded in lockstep,
//! "give every batch row row 0's position" is exactly a constant per-sequence
//! offset, and a batched-vs-serial logits oracle passes it with **max abs diff
//! 0.0**. It only becomes visible when a sequence's position error is
//! *non-uniform* — which happens when sequences arrive at different steps *and*
//! one of them is descheduled for a step. The oracle in this file replays such a
//! schedule for that reason, and a companion test asserts the oracle still
//! catches the shared-position bug, so the sensitivity cannot silently rot away.

use std::sync::Arc;

use anyhow::{Context, Result, bail};

use fuel::inference_context::InferenceContext;
use fuel::kv_block_pool::{KvGeometry, PhysBlockId, SessionHandle};
use fuel::kv_block_pool_device::DeviceKvPool;
use fuel::lazy::{LayerWeights, LlamaConfig, LlamaModel, WeightStorage};
use fuel::{DType, Device, Shape};

use fuel::lazy::Tensor;

/// Rotary position embedding at a **per-batch-row** position.
///
/// `x` is `[B, H, 1, D]`; `cos`/`sin` are `[B, 1, 1, D]` (row `b` holds the table
/// for sequence `b`'s own absolute position). Broadcasting over the head axis is
/// what makes one table row serve all of that sequence's heads.
///
/// This is `fuel_graph::NodeHandle::rope_with_tables_decomposed`'s body
/// (`fuel-graph/src/lib.rs` ~6931-6955) with the table shape generalised from
/// "one position" to "one position per row" — same ops, same order, same
/// associativity:
///
/// ```text
/// y = x * cos + concat(-x[..., D/2..], x[..., ..D/2]) * sin
/// ```
///
/// It exists only because the `Tensor` wrapper hard-requires `cos.dims() ==
/// [seq, d]` with `seq = x.dims[rank-2]`, which at decode is `1`. See the module
/// docs.
pub fn rope_batched(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dims: Vec<usize> = x.shape().dims().to_vec();
    let rank = dims.len();
    if rank < 2 {
        bail!("rope_batched: expected rank >= 2, got {dims:?}");
    }
    let d = dims[rank - 1];
    if !d.is_multiple_of(2) {
        bail!("rope_batched: head_dim {d} must be even");
    }
    let half = d / 2;
    let target = Shape::from_dims(&dims);

    let cos_b = cos
        .broadcast_to(target.clone())
        .map_err(|e| anyhow::anyhow!("rope_batched: broadcasting cos to {dims:?}: {e:?}"))?;
    let sin_b = sin
        .broadcast_to(target)
        .map_err(|e| anyhow::anyhow!("rope_batched: broadcasting sin to {dims:?}: {e:?}"))?;

    let first = x
        .slice(rank - 1, 0, half)
        .map_err(|e| anyhow::anyhow!("rope_batched: slice first half: {e:?}"))?;
    let second = x
        .slice(rank - 1, half, half)
        .map_err(|e| anyhow::anyhow!("rope_batched: slice second half: {e:?}"))?;
    let rotated = second
        .neg()
        .concat(&first, rank - 1)
        .map_err(|e| anyhow::anyhow!("rope_batched: concat rotated half: {e:?}"))?;

    let left = x
        .mul(&cos_b)
        .map_err(|e| anyhow::anyhow!("rope_batched: x * cos: {e:?}"))?;
    let right = rotated
        .mul(&sin_b)
        .map_err(|e| anyhow::anyhow!("rope_batched: rotate_half(x) * sin: {e:?}"))?;
    left.add(&right)
        .map_err(|e| anyhow::anyhow!("rope_batched: final add: {e:?}"))
}

/// `[B, 1, 1, head_dim]` RoPE cos/sin tables, row `b` at absolute position
/// `positions[b]`.
///
/// Built by concatenating `B` calls to Fuel's own
/// [`Tensor::rope_tables_const`] along the batch axis — the
/// `(theta, position) → (cos, sin)` math stays Fuel's canonical
/// `fuel_graph::build_rope_tables` rather than being duplicated here. At `B == 1`
/// this is bit-for-bit the same const the single-sequence path emits.
///
/// `anchor` supplies the graph (rule 1).
fn batched_rope_tables(
    anchor: &Tensor,
    rope_base: f64,
    positions: &[usize],
    head_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let row_shape = Shape::from_dims(&[1usize, 1, 1, head_dim]);
    let mut cos_acc: Option<Tensor> = None;
    let mut sin_acc: Option<Tensor> = None;
    for &pos in positions {
        // `[1, head_dim]` from Fuel's canonical table builder, viewed as one
        // batch row.
        let (c, s) = anchor.rope_tables_const(rope_base, pos, 1, head_dim);
        let c = c
            .reshape(row_shape.clone())
            .map_err(|e| anyhow::anyhow!("batched_rope_tables: cos reshape: {e:?}"))?;
        let s = s
            .reshape(row_shape.clone())
            .map_err(|e| anyhow::anyhow!("batched_rope_tables: sin reshape: {e:?}"))?;
        cos_acc = Some(match cos_acc {
            None => c,
            Some(acc) => acc
                .concat(&c, 0usize)
                .map_err(|e| anyhow::anyhow!("batched_rope_tables: cos concat: {e:?}"))?,
        });
        sin_acc = Some(match sin_acc {
            None => s,
            Some(acc) => acc
                .concat(&s, 0usize)
                .map_err(|e| anyhow::anyhow!("batched_rope_tables: sin concat: {e:?}"))?,
        });
    }
    match (cos_acc, sin_acc) {
        (Some(c), Some(s)) => Ok((c, s)),
        _ => bail!("batched_rope_tables: empty position list"),
    }
}

/// `rms_norm(x) * gain` — Fuel's `apply_affine_rms_norm` is private, so this is
/// the same three lines (`lazy.rs` ~9042) rebuilt on the public API.
fn affine_rms_norm(x: &Tensor, gain: &Arc<[f32]>, dim: usize, eps: f64) -> Result<Tensor> {
    if gain.len() != dim {
        bail!("affine_rms_norm: gain len {} != dim {dim}", gain.len());
    }
    let normed = x
        .rms_norm_last_dim(eps)
        .map_err(|e| anyhow::anyhow!("affine_rms_norm: rms_norm_last_dim: {e:?}"))?;
    let gain_t = x
        .const_like_dtype(gain, Shape::from_dims(&[dim]), x.dtype())
        .map_err(|e| anyhow::anyhow!("affine_rms_norm: gain const: {e:?}"))?;
    normed
        .broadcast_mul(&gain_t)
        .map_err(|e| anyhow::anyhow!("affine_rms_norm: broadcast_mul: {e:?}"))
}

/// Multi-sequence paged decode over a [`DeviceKvPool`].
///
/// Borrows the model rather than owning it — the weights are the large thing
/// (4.4 GB f32 for TinyLlama) and a serving process wants one copy shared by
/// every decoder/scheduler that touches it. Mirrors the shape of Fuel's own
/// `PagedSessionScheduler<'m, M>`.
pub struct BatchedPagedDecoder<'m> {
    model: &'m LlamaModel,
    pool: DeviceKvPool,
    device: Device,
}

impl<'m> BatchedPagedDecoder<'m> {
    /// Build a decoder over a fresh pool of `num_blocks` blocks of `block_size`
    /// tokens each.
    ///
    /// Rejects (typed, at construction) any weight that is not `F32`.
    ///
    /// **Not because Fuel's paged forwards require it** — they accept
    /// `F32 | BF16 | F16`. Because `Op::PagedAttn` has no CUDA or Vulkan kernel,
    /// so paged attention is placed on CPU regardless of which backends are
    /// compiled in, and f32 avoids a promoting cast on that CPU fallback that
    /// would buy nothing. And because the pool's `f32`-typed byte movement
    /// (`write_block` / `read_block`, hence `evict` / `restore`) rejects a
    /// non-F32 pool — a bf16 pool would need the `_bytes` variants throughout
    /// [`crate::model_fuel::policies::BlockTierMover`].
    ///
    /// Revisit when Fuel's PC-3 lands a CUDA `PagedAttn` kernel: at that point
    /// bf16 becomes a real question rather than a pointless one.
    ///
    /// Use `loader_f32::load_llama_f32_from_dir`, not the stock loader.
    pub fn new(model: &'m LlamaModel, num_blocks: usize, block_size: usize) -> Result<Self> {
        let cfg = &model.config;
        if block_size == 0 || num_blocks == 0 {
            bail!("BatchedPagedDecoder::new: num_blocks and block_size must both be > 0");
        }
        assert_all_f32(cfg, &model.weights.layers, &model.weights.output)?;

        let geom = KvGeometry {
            // MUST equal the model's layer count — a physical block is the same
            // slot in every layer's K and V buffer (the vLLM shared-block-table
            // model), so the pool is sized per-layer.
            n_layers: cfg.n_layers,
            num_blocks,
            block_size,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            elem_size: DType::F32.size_in_bytes(),
        };
        // **`Device::cpu()` here is a STARTING LOCATION, not a pin — keep it.**
        // Verified against Fuel's dispatch rather than assumed: the decision
        // device is `graph.placement(id)` (a hard pin, scheduler-only) else
        // `options.pinned_device` (soft, despite the name) else the target
        // backend. A leaf's `Device` is in neither chain — it only allocates the
        // leaf's storage and is consumed as input *residency*, which the cost
        // model prices and the optimizer is free to move.
        //
        // So "start on CPU and let the optimizer move what is worth moving" is
        // the correct consumer pattern, and `Device::custom` is not needed.
        //
        // It is still a performance LEVER, though: starting weights on CPU makes
        // every GPU candidate pay an inbound transfer, so a transfer-aware
        // planner may keep an op on CPU correctly, but for a reason we created
        // by starting it there. Co-resident inputs price at ~0. If we ever want
        // specific weights on a GPU, the move is to START them there — which
        // zeroes their transfer term — not to pin anything.
        let device = Device::cpu();
        let pool = DeviceKvPool::new(geom, DType::F32, &device)
            .map_err(|e| anyhow::anyhow!("allocating DeviceKvPool: {e:?}"))?;
        Ok(Self {
            model,
            pool,
            device,
        })
    }

    /// The pool, for capacity questions and lifecycle verbs the caller owns
    /// (`evict`, `restore`, `splice`, …).
    pub fn pool(&self) -> &DeviceKvPool {
        &self.pool
    }
    /// Mutable pool access — `core_mut().evict_blocks(..)` and friends.
    pub fn pool_mut(&mut self) -> &mut DeviceKvPool {
        &mut self.pool
    }
    /// The model this decoder drives.
    pub fn config(&self) -> &LlamaConfig {
        &self.model.config
    }

    /// Open a new sequence. Returns the pool's handle; the caller keys its own
    /// request state by it (`SessionHandle`'s field is private and it has no
    /// public constructor, so it must be *carried*, never reconstructed).
    pub fn open_session(&mut self) -> SessionHandle {
        self.pool.core_mut().open()
    }

    /// Release a sequence's blocks back to the pool. This is how headroom comes
    /// back; a scheduler that never calls it will report a full pool forever.
    pub fn close_session(&mut self, s: SessionHandle) {
        self.pool.core_mut().discard(s);
    }

    /// A sequence's absolute position — the number of tokens whose K/V is
    /// resident, which is exactly `Op::PagedAttn`'s `context_len`.
    ///
    /// Never `blocks * block_size`: that over-counts the partial last block.
    pub fn position(&self, s: SessionHandle) -> Option<usize> {
        self.pool.core().filled_tokens(s)
    }

    /// Free physical blocks.
    pub fn free_blocks(&self) -> usize {
        self.pool.core().free_blocks()
    }

    /// Blocks this batch of `(current_filled_tokens, tokens_to_add)` pairs would
    /// consume. Delegates to the pool's own arithmetic
    /// (`KvBlockPool::blocks_required_batch`) so a consumer's block math can
    /// never drift from the allocator's.
    pub fn blocks_required_batch(&self, seqs: &[(usize, usize)]) -> usize {
        self.pool.core().blocks_required_batch(seqs)
    }

    /// **Lightbulb's admission question, asked of Fuel's pool.** Returns whether
    /// the pool *could* seat `seqs` — the accept/reject/queue/preempt decision is
    /// the caller's policy, not this method's.
    pub fn can_admit(&self, seqs: &[(usize, usize)]) -> bool {
        self.blocks_required_batch(seqs) <= self.free_blocks()
    }

    /// Convenience: can these prompts be admitted from scratch?
    pub fn can_admit_prompts(&self, prompt_lens: &[usize]) -> bool {
        let seqs: Vec<(usize, usize)> = prompt_lens.iter().map(|&n| (0usize, n)).collect();
        self.can_admit(&seqs)
    }

    /// **One decode step for `B` sequences: one graph, one realize.**
    ///
    /// `batch` is `(session, token_to_feed)` pairs. Returns `B` rows of
    /// `[vocab_size]` logits **in `batch` order** — the same order that defines
    /// `Op::PagedAttn`'s batch axis, so a caller can zip results back to requests
    /// positionally.
    ///
    /// Admission is checked **before any `append`**, and the whole step is
    /// all-or-nothing: `KvBlockPool::append` is per-session and would otherwise
    /// let sequence 5 fail after 0..4 already grew, leaving the batch
    /// inconsistent with no way to tell how far it got.
    ///
    /// Sampling is not done here. These are realized host values; what to do with
    /// them is Lightbulb's.
    pub fn step(&mut self, batch: &[(SessionHandle, u32)]) -> Result<Vec<Vec<f32>>> {
        self.step_inner(batch, None)
    }

    /// **Test-only fault injection.** Runs [`Self::step`] but takes the RoPE
    /// positions from `forced` instead of each sequence's own `filled_tokens`.
    ///
    /// It exists so the batched-vs-serial oracle can have a *real* control:
    /// force every row to share row 0's position (the failure a naive batched
    /// implementation has) and require the parity check to catch it. Without a
    /// hook like this the control would have to be a source mutation, which no
    /// test can assert on.
    #[cfg(test)]
    fn step_forcing_rope_positions(
        &mut self,
        batch: &[(SessionHandle, u32)],
        forced: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        self.step_inner(batch, Some(forced))
    }

    fn step_inner(
        &mut self,
        batch: &[(SessionHandle, u32)],
        rope_override: Option<&[usize]>,
    ) -> Result<Vec<Vec<f32>>> {
        let b = batch.len();
        if b == 0 {
            bail!("BatchedPagedDecoder::step: empty batch");
        }
        // A session appearing twice would be appended twice for one token and
        // would occupy two rows of one block table — silently wrong positions,
        // not an error anywhere downstream. Cheap to rule out (B is small).
        for i in 0..b {
            for j in (i + 1)..b {
                if batch[i].0 == batch[j].0 {
                    bail!(
                        "BatchedPagedDecoder::step: session appears twice in one batch (rows {i} and {j})"
                    );
                }
            }
        }

        let cfg = self.model.config.clone();
        let geom = self.pool.geometry();
        let block_size = geom.block_size;

        // --- positions, admission, growth -------------------------------
        let mut positions: Vec<usize> = Vec::with_capacity(b);
        for (i, &(s, _)) in batch.iter().enumerate() {
            let p = self
                .pool
                .core()
                .filled_tokens(s)
                .with_context(|| format!("step: row {i} references an unknown session"))?;
            positions.push(p);
        }
        let seqs: Vec<(usize, usize)> = positions.iter().map(|&p| (p, 1usize)).collect();
        // Capacity, counted the way the step actually spends it. `append`
        // allocates for a row that starts a new block; copy-on-write allocates
        // for a row whose frontier block is SHARED, because the share has to be
        // broken before the write. Missing the second term is not a rounding
        // error — it lets the pre-check pass and then leaves the batch
        // partially advanced when a mid-batch allocation fails, which wedges
        // the sessions at inconsistent positions and is not retryable.
        //
        // This mirrors the accounting in Fuel's own `forward_paged_step_batched`,
        // generalized to ragged slots: our rows are at different positions, so
        // "starts a new block" is per row rather than shared across the batch.
        let mut cow_splits = 0usize;
        for (&(s, _), &pos) in batch.iter().zip(positions.iter()) {
            if pos % block_size != 0 {
                if let Some(frontier) = self.pool.core().resident_block(s, pos / block_size) {
                    if self.pool.core().block_refcount(frontier) > 1 {
                        cow_splits += 1;
                    }
                }
            }
        }
        let need = self.pool.core().blocks_required_batch(&seqs) + cow_splits;
        let have = self.pool.core().free_blocks();
        if need > have {
            bail!(
                "BatchedPagedDecoder::step: pool exhausted — this batch needs {need} more \
                 block(s) ({cow_splits} of them to break shared frontier blocks), {have} free. \
                 Nothing was appended; evict or close a session and retry, or admit a smaller \
                 batch."
            );
        }
        for (i, &(s, _)) in batch.iter().enumerate() {
            self.pool
                .core_mut()
                .append(s, 1)
                .map_err(|e| anyhow::anyhow!("step: row {i} append failed after the capacity pre-check passed (this is a pool-accounting bug, not a capacity one): {e:?}"))?;
        }

        // Where each row's new K/V lands — via Fuel's copy-on-write guard.
        //
        // `ensure_writable_block` is a CONTRACT, not an optimisation: its own
        // doc says it "must be called after `append` and before writing the
        // token — otherwise decoding a spliced session silently corrupts its
        // co-sharers". If the frontier block is exclusive it returns it
        // unchanged; if it is shared it breaks the share AND copies the block's
        // bytes (all layers, K and V) into the fresh one, so the session keeps
        // its shared-prefix content while its write stops mutating a block
        // another session still references.
        //
        // This replaces a hand-rolled guard that compared `(phys, slot)` pairs
        // across rows and bailed on a collision. That guard was blind to the
        // real hazard: two sessions sharing a physical block at DIFFERENT fill
        // levels have different slots, so it stayed silent while the write
        // corrupted the co-sharer — measured at 0.0296 max abs error against an
        // exclusive-cache replay, on a module whose parity bar is 1e-4. It was
        // also blind for a lone session: sharing is a property of the block, not
        // of co-batching, so a single session decoding into a spliced block
        // corrupts its donor with no batch involved at all.
        //
        // Copying the bytes is the part a bare refcount check would have missed.
        let mut phys: Vec<PhysBlockId> = Vec::with_capacity(b);
        let mut slot: Vec<usize> = Vec::with_capacity(b);
        for (i, (&(s, _), &pos)) in batch.iter().zip(positions.iter()).enumerate() {
            let p = self
                .pool
                .ensure_writable_block(s, pos / block_size)
                .map_err(|e| {
                    anyhow::anyhow!("step: row {i}: making the frontier block writable: {e:?}")
                })?;
            phys.push(p);
            slot.push(pos % block_size);
        }
        // With every frontier block now exclusively owned, two rows cannot
        // address the same (block, slot): distinct sessions hold distinct
        // physical blocks. Kept as a debug assertion rather than a runtime bail
        // because it is now an invariant of the loop above, not a caller error.
        debug_assert!(
            {
                let mut pairs: Vec<(PhysBlockId, usize)> =
                    phys.iter().copied().zip(slot.iter().copied()).collect();
                pairs.sort_unstable();
                let n = pairs.len();
                pairs.dedup();
                pairs.len() == n
            },
            "two rows target the same (phys, slot) after ensure_writable_block — \
             copy-on-write did not make the frontier blocks exclusive: phys={phys:?} slot={slot:?}"
        );

        let handles: Vec<SessionHandle> = batch.iter().map(|&(s, _)| s).collect();
        let pt = self
            .pool
            .materialize_block_table(&handles)
            .map_err(|e| anyhow::anyhow!("step: materializing the block table: {e:?}"))?;
        debug_assert_eq!(pt.batch, b);

        // --- graph ------------------------------------------------------
        let w = &self.model.weights;
        let dim = cfg.dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let scale = (1.0f64 / (cfg.head_dim as f64).sqrt()) as f32;

        // THE ROOT (rule 1). `from_f32` takes `impl Into<Arc<[f32]>>`, so cloning
        // the embedding table is a refcount bump, not a 262 MB copy.
        let embed = Tensor::from_f32(
            w.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, dim]),
            &self.device,
        );
        let tokens: Vec<u32> = batch.iter().map(|&(_, t)| t).collect();
        let token_ids = embed.const_u32_like(tokens, Shape::from_dims(&[b]));
        let mut h = embed
            .index_select(0usize, &token_ids)
            .map_err(|e| anyhow::anyhow!("step: embedding lookup: {e:?}"))?
            .reshape(Shape::from_dims(&[b, 1, dim]))
            .map_err(|e| anyhow::anyhow!("step: embedding reshape: {e:?}"))?;

        // Hoisted once for the whole step, like the contiguous path hoists its
        // mask: the per-row RoPE tables and the page table do not vary by layer.
        let rope_positions: &[usize] = match rope_override {
            None => &positions,
            Some(forced) => {
                if forced.len() != b {
                    bail!(
                        "step: forced RoPE positions have length {} but the batch is {b}",
                        forced.len()
                    );
                }
                forced
            }
        };
        let (rope_cos, rope_sin) =
            batched_rope_tables(&h, cfg.rope_base, rope_positions, cfg.head_dim)?;
        let block_table = h.const_u32_like(pt.block_table.clone(), pt.block_table_shape());
        let context_lens = h.const_u32_like(pt.context_lens.clone(), pt.context_lens_shape());

        // One `InferenceContext` per step: the graph is rebuilt each step, so
        // NodeIds are not stable and a longer-lived persistent map would
        // accumulate stale entries. The *pool buffers* persist — they are the
        // same Arcs every step, mutated in place by `Op::WriteSlice`.
        let mut ctx = InferenceContext::new(self.device.clone());

        for (li, layer) in w.layers.iter().enumerate() {
            let k_ph = h.const_placeholder_like(self.pool.pool_shape().clone(), DType::F32);
            let v_ph = h.const_placeholder_like(self.pool.pool_shape().clone(), DType::F32);
            let k_arc = self
                .pool
                .k_pool(li)
                .with_context(|| format!("step: no K pool buffer for layer {li}"))?;
            let v_arc = self
                .pool
                .v_pool(li)
                .with_context(|| format!("step: no V pool buffer for layer {li}"))?;
            ctx.insert(k_ph.node_id(), Arc::clone(k_arc));
            ctx.insert(v_ph.node_id(), Arc::clone(v_arc));

            h = self.layer_batched(
                &h,
                layer,
                &k_ph,
                &v_ph,
                &rope_cos,
                &rope_sin,
                &block_table,
                &context_lens,
                &phys,
                &slot,
                scale,
                b,
                dim,
                kv_dim,
            )?;
        }

        let h_norm = affine_rms_norm(&h, &w.final_norm_gain, dim, cfg.norm_eps)?;
        // `apply_linear` became fallible in Fuel (7ed43541-era); it used to
        // return `Tensor` directly.
        let logits = w
            .output
            .apply_linear(&h_norm, dim, cfg.vocab_size)
            .map_err(|e| anyhow::anyhow!("step: logits projection: {e:?}"))?;
        let logits_root = logits
            .reshape(Shape::from_dims(&[b, cfg.vocab_size]))
            .map_err(|e| anyhow::anyhow!("step: logits reshape: {e:?}"))?;

        // ONE realize for the whole batch and every layer.
        let flat = ctx
            .realize_one_as::<f32>(logits_root.graph_handle(), logits_root.node_id())
            .map_err(|e| anyhow::anyhow!("step: realize: {e:?}"))?;
        if flat.len() != b * cfg.vocab_size {
            bail!(
                "step: realized {} values, expected {} ({b} x {})",
                flat.len(),
                b * cfg.vocab_size,
                cfg.vocab_size
            );
        }
        Ok(flat.chunks(cfg.vocab_size).map(|c| c.to_vec()).collect())
    }

    /// One transformer layer at batch width `B`.
    ///
    /// Fuel's `apply_layer_paged` / `project_qkv_roped` / `ffn_block` are all
    /// private, so this is the same layer rebuilt on the public API, with two
    /// changes: RoPE is per-row ([`rope_batched`]), and the KV write is `B`
    /// **chained** `write_slice` calls instead of one.
    #[allow(clippy::too_many_arguments)]
    fn layer_batched(
        &self,
        x: &Tensor,
        layer: &LayerWeights,
        k_ph: &Tensor,
        v_ph: &Tensor,
        rope_cos: &Tensor,
        rope_sin: &Tensor,
        block_table: &Tensor,
        context_lens: &Tensor,
        phys: &[PhysBlockId],
        slot: &[usize],
        scale: f32,
        b: usize,
        dim: usize,
        kv_dim: usize,
    ) -> Result<Tensor> {
        let cfg = &self.model.config;
        let (hkv, hd) = (cfg.n_kv_heads, cfg.head_dim);

        let x_norm = affine_rms_norm(x, &layer.attn_norm_gain, dim, cfg.norm_eps)?;
        let q = layer
            .attn_q
            .apply_linear(&x_norm, dim, dim)
            .map_err(|e| anyhow::anyhow!("layer: q projection: {e:?}"))?
            .add_optional_trailing_bias(layer.attn_q_bias.as_ref())
            .map_err(|e| anyhow::anyhow!("layer: q bias: {e:?}"))?;
        let k = layer
            .attn_k
            .apply_linear(&x_norm, dim, kv_dim)
            .map_err(|e| anyhow::anyhow!("layer: k projection: {e:?}"))?
            .add_optional_trailing_bias(layer.attn_k_bias.as_ref())
            .map_err(|e| anyhow::anyhow!("layer: k bias: {e:?}"))?;
        let v = layer
            .attn_v
            .apply_linear(&x_norm, dim, kv_dim)
            .map_err(|e| anyhow::anyhow!("layer: v projection: {e:?}"))?
            .add_optional_trailing_bias(layer.attn_v_bias.as_ref())
            .map_err(|e| anyhow::anyhow!("layer: v bias: {e:?}"))?;

        let to_heads = |t: &Tensor, heads: usize, what: &str| -> Result<Tensor> {
            t.reshape(Shape::from_dims(&[b, 1, heads, hd]))
                .and_then(|r| r.permute([0usize, 2, 1, 3]))
                .map_err(|e| anyhow::anyhow!("layer: {what} head reshape: {e:?}"))
        };
        let q_h = to_heads(&q, cfg.n_heads, "q")?;
        let k_h = to_heads(&k, hkv, "k")?;
        let v_h = to_heads(&v, hkv, "v")?;

        let q_r = rope_batched(&q_h, rope_cos, rope_sin)?;
        let k_r = rope_batched(&k_h, rope_cos, rope_sin)?;

        // `Op::PagedAttn` is decode-only and NOTHING validates it (the fast
        // kernel is causal at Sq>1, the decompose recipe is not — they disagree).
        // Assert it here so a future "batched prefill" edit fails loudly instead
        // of returning plausible wrong numbers on one lowering path.
        let q_dims = q_r.shape().dims().to_vec();
        if q_dims.len() != 4 || q_dims[2] != 1 {
            bail!(
                "layer: Op::PagedAttn is decode-only (Sq must be 1); q is {q_dims:?}. \
                 Feed prompts one token at a time — see this module's docs."
            );
        }

        // --- the batched write+attend (the piece Fuel does not provide) ---
        //
        // `write_slice` is DESTRUCTIVE on its destination: the returned tensor's
        // Storage Arc *is* the destination's, and the executor removes the
        // destroyed input from the storage cache once the node has run. So `B`
        // writes into one pool buffer must be CHAINED — each taken off the
        // previous write's result — not fanned out in parallel off `k_ph`. Two
        // writes both rooted at `k_ph` would be an unordered WAW *and* the second
        // would find its input already evicted.
        let mut post_k = k_ph.clone();
        let mut post_v = v_ph.clone();
        for row in 0..b {
            let p = phys[row] as usize;
            let ranges = vec![(p, p + 1), (slot[row], slot[row] + 1), (0, hkv), (0, hd)];
            // `[1, Hkv, 1, D]` -> `[1, 1, Hkv, D]` is a pure reshape (h-major
            // order is identical) — the same alignment `build_decode_attn` does
            // for the single-sequence case.
            let k_slab = k_r
                .slice(0usize, row, 1)
                .and_then(|t| t.reshape(Shape::from_dims(&[1usize, 1, hkv, hd])))
                .map_err(|e| anyhow::anyhow!("layer: row {row} K slab: {e:?}"))?;
            let v_slab = v_h
                .slice(0usize, row, 1)
                .and_then(|t| t.reshape(Shape::from_dims(&[1usize, 1, hkv, hd])))
                .map_err(|e| anyhow::anyhow!("layer: row {row} V slab: {e:?}"))?;
            post_k = post_k
                .write_slice(&k_slab, ranges.clone())
                .map_err(|e| anyhow::anyhow!("layer: row {row} K write_slice: {e:?}"))?;
            post_v = post_v
                .write_slice(&v_slab, ranges)
                .map_err(|e| anyhow::anyhow!("layer: row {row} V write_slice: {e:?}"))?;
        }

        let attn = q_r
            .paged_attn(
                &post_k,
                &post_v,
                block_table,
                context_lens,
                None,
                scale,
                self.pool.geometry().block_size,
                None,
            )
            .map_err(|e| anyhow::anyhow!("layer: paged_attn: {e:?}"))?;

        let merged = attn
            .permute([0usize, 2, 1, 3])
            .and_then(|t| t.reshape(Shape::from_dims(&[b, 1, dim])))
            .map_err(|e| anyhow::anyhow!("layer: merge heads: {e:?}"))?;
        let attn_out = layer
            .attn_o
            .apply_linear(&merged, dim, dim)
            .map_err(|e| anyhow::anyhow!("layer: attn output projection: {e:?}"))?;
        let h1 = x
            .add(&attn_out)
            .map_err(|e| anyhow::anyhow!("layer: attn residual: {e:?}"))?;

        let h1_norm = affine_rms_norm(&h1, &layer.ffn_norm_gain, dim, cfg.norm_eps)?;
        let gate = layer
            .ffn_gate
            .apply_linear(&h1_norm, dim, cfg.ffn_dim)
            .map_err(|e| anyhow::anyhow!("layer: ffn gate projection: {e:?}"))?;
        let up = layer
            .ffn_up
            .apply_linear(&h1_norm, dim, cfg.ffn_dim)
            .map_err(|e| anyhow::anyhow!("layer: ffn up projection: {e:?}"))?;
        let swiglu = gate
            .silu()
            .mul(&up)
            .map_err(|e| anyhow::anyhow!("layer: swiglu: {e:?}"))?;
        let ffn_out = layer
            .ffn_down
            .apply_linear(&swiglu, cfg.ffn_dim, dim)
            .map_err(|e| anyhow::anyhow!("layer: ffn down projection: {e:?}"))?;
        h1.add(&ffn_out)
            .map_err(|e| anyhow::anyhow!("layer: ffn residual: {e:?}"))
    }

    /// Prefill `B` prompts **across** sequences: `max(prompt_len)` steps at width
    /// `B`, rather than `sum(prompt_len)` steps at width 1.
    ///
    /// Each step feeds the sub-batch of sequences that still have a token at that
    /// index, so the batch narrows as short prompts finish. Returns each
    /// sequence's logits **after its own last prompt token** — the row a caller
    /// samples its first generated token from — in `sessions` order.
    ///
    /// Prompts are fed one token at a time because `Op::PagedAttn` is `Sq == 1`;
    /// see the module docs for why that is a correctness constraint and not a
    /// performance one.
    pub fn prefill_batch(
        &mut self,
        sessions: &[SessionHandle],
        prompts: &[&[u32]],
    ) -> Result<Vec<Vec<f32>>> {
        if sessions.len() != prompts.len() {
            bail!(
                "prefill_batch: {} sessions but {} prompts",
                sessions.len(),
                prompts.len()
            );
        }
        if sessions.is_empty() {
            bail!("prefill_batch: nothing to prefill");
        }
        if let Some(i) = prompts.iter().position(|p| p.is_empty()) {
            bail!("prefill_batch: prompt {i} is empty — there is no token to feed");
        }

        // Whole-prefill admission, asked of the pool once. A per-step check would
        // let a batch get half way and strand.
        let seqs: Vec<(usize, usize)> = sessions
            .iter()
            .zip(prompts.iter())
            .map(|(&s, p)| (self.position(s).unwrap_or(0), p.len()))
            .collect();
        if !self.can_admit(&seqs) {
            bail!(
                "prefill_batch: pool cannot seat these prompts — need {} block(s), {} free",
                self.blocks_required_batch(&seqs),
                self.free_blocks()
            );
        }

        let longest = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
        let mut last: Vec<Option<Vec<f32>>> = vec![None; sessions.len()];
        for t in 0..longest {
            let mut sub: Vec<(SessionHandle, u32)> = Vec::new();
            let mut origin: Vec<usize> = Vec::new();
            for (i, (&s, p)) in sessions.iter().zip(prompts.iter()).enumerate() {
                if let Some(&tok) = p.get(t) {
                    sub.push((s, tok));
                    origin.push(i);
                }
            }
            let rows = self.step(&sub)?;
            for (row, i) in rows.into_iter().zip(origin.into_iter()) {
                last[i] = Some(row);
            }
        }
        last.into_iter()
            .enumerate()
            .map(|(i, o)| {
                o.with_context(|| format!("prefill_batch: sequence {i} produced no logits"))
            })
            .collect()
    }
}

/// Rule 2, enforced. Returns a typed error naming the offending weight rather
/// than letting a bf16 projection quietly re-enter the promoting-cast regime.
fn assert_all_f32(
    cfg: &LlamaConfig,
    layers: &[LayerWeights],
    output: &WeightStorage,
) -> Result<()> {
    if layers.len() != cfg.n_layers {
        bail!(
            "BatchedPagedDecoder: config says {} layers, weights have {}",
            cfg.n_layers,
            layers.len()
        );
    }
    for (i, l) in layers.iter().enumerate() {
        for (name, ws) in [
            ("attn_q", &l.attn_q),
            ("attn_k", &l.attn_k),
            ("attn_v", &l.attn_v),
            ("attn_o", &l.attn_o),
            ("ffn_gate", &l.ffn_gate),
            ("ffn_up", &l.ffn_up),
            ("ffn_down", &l.ffn_down),
        ] {
            if !matches!(ws, WeightStorage::F32(_)) {
                bail!(
                    "BatchedPagedDecoder: layer {i} {name} is {:?}, not F32. The paged path is \
                     f32-only in Fuel (write_block / read_block / forward_paged_step all reject \
                     other dtypes) and rule 2 wants [F32,F32,F32] matmul keys. Load with \
                     loader_f32::load_llama_f32_from_dir.",
                    ws.dtype()
                );
            }
        }
    }
    if !matches!(output, WeightStorage::F32(_)) {
        bail!(
            "BatchedPagedDecoder: lm_head is {:?}, not F32 (see the per-layer message)",
            output.dtype()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel::kv_block_pool_device::BlockKind;
    use fuel::lazy::LlamaWeights;
    use std::path::PathBuf;

    // ---- fixtures --------------------------------------------------------

    /// Deterministic pseudo-random f32 (mirrors Fuel's own pool tests).
    fn rand_f32(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.5
            })
            .collect()
    }

    /// Tiny deterministic weights sized to `cfg` — copied from Fuel's
    /// `tests/paged_decode_parity.rs` so the oracle here runs in milliseconds
    /// and in CI, not behind a 2.2 GB download.
    fn tiny_weights(cfg: &LlamaConfig, seed: u32) -> LlamaWeights {
        let mut s: u32 = seed;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            let v: Vec<f32> = (0..n).map(|_| next()).collect();
            Arc::from(v)
        };
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        LlamaWeights {
            // Fresh id per call, deliberately: two `tiny_weights(cfg, seed)`
            // calls with the SAME seed produce byte-identical weights but are
            // distinct weight sets, and Fuel's plan-reuse predicate
            // (`c4718467`) must not let one's held plan serve the other.
            instance: fuel::decode_shape::ModelInstanceId::next(),
            token_embedding: vec_of(cfg.vocab_size * cfg.dim),
            layers: (0..cfg.n_layers)
                .map(|_| LayerWeights {
                    attn_q: WeightStorage::F32(vec_of(cfg.dim * cfg.dim)),
                    attn_q_bias: None,
                    attn_k: WeightStorage::F32(vec_of(cfg.dim * kv_dim)),
                    attn_k_bias: None,
                    attn_v: WeightStorage::F32(vec_of(cfg.dim * kv_dim)),
                    attn_v_bias: None,
                    attn_o: WeightStorage::F32(vec_of(cfg.dim * cfg.dim)),
                    ffn_gate: WeightStorage::F32(vec_of(cfg.dim * cfg.ffn_dim)),
                    ffn_up: WeightStorage::F32(vec_of(cfg.dim * cfg.ffn_dim)),
                    ffn_down: WeightStorage::F32(vec_of(cfg.ffn_dim * cfg.dim)),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output: WeightStorage::F32(vec_of(cfg.dim * cfg.vocab_size)),
        }
    }

    fn toy_cfg(n_kv_heads: usize) -> LlamaConfig {
        LlamaConfig {
            vocab_size: 16,
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }

    /// `true` iff every pair is within `tol` (relative-or-absolute), the same
    /// bar Fuel's `paged_decode_parity.rs` uses.
    fn all_close(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len()
            && a.iter().zip(b.iter()).all(|(&x, &y)| {
                let d = (x - y).abs();
                let den = x.abs().max(y.abs()).max(f32::MIN_POSITIVE);
                d < tol || d / den < tol
            })
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn tinyllama_dir() -> Option<PathBuf> {
        // one locator, not twelve copies of an absolute path into one home directory
        crate::test_models::tinyllama_dir()
    }

    /// All-f32 checkpoint load, for the `#[ignore]`d TinyLlama test only.
    ///
    /// Mirrors `loader_f32.rs` (same `load_transposed_matrix` /
    /// `load_tensor_as_f32` calls, same reason: `[F32,F32,F32]` matmul keys so no
    /// promoting cast runs, and `DeviceKvPool` is f32-gated anyway). It is
    /// duplicated here rather than imported so `batched.rs` depends on nothing
    /// inside Lightbulb — only on `fuel` — which is what lets it be compiled and
    /// tested before the orchestrator wires it into `mod.rs`. Untied `lm_head`
    /// only (TinyLlama ships one); a tied checkpoint gets a clear error rather
    /// than a silent wrong head.
    fn load_f32_checkpoint(dir: &std::path::Path) -> Result<LlamaModel> {
        use fuel::lazy::{load_tensor_as_f32, load_transposed_matrix};
        use fuel::lazy_llama2c::Llama2cConfig;
        use fuel::safetensors::MmapedSafetensors;

        let config_str = std::fs::read_to_string(dir.join("config.json"))?;
        let config: LlamaConfig = Llama2cConfig::from_hf_json_str(&config_str)
            .map_err(|e| anyhow::anyhow!("config.json: {e:?}"))?
            .to_llama_config();
        let st = unsafe { MmapedSafetensors::multi(&[dir.join("model.safetensors")]) }
            .map_err(|e| anyhow::anyhow!("mmap: {e:?}"))?;

        let (dim, ffn_dim) = (config.dim, config.ffn_dim);
        let kv_dim = config.n_kv_heads * config.head_dim;
        let mat = |name: &str, out: usize, inp: usize| -> Result<WeightStorage> {
            let v = load_transposed_matrix(&st, name, out, inp)
                .map_err(|e| anyhow::anyhow!("loading {name}: {e:?}"))?;
            Ok(WeightStorage::F32(Arc::from(v)))
        };
        let vecf = |name: &str| -> Result<Arc<[f32]>> {
            let v = load_tensor_as_f32(&st, name)
                .map_err(|e| anyhow::anyhow!("loading {name}: {e:?}"))?;
            Ok(Arc::from(v))
        };

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(LayerWeights {
                attn_q: mat(
                    &format!("model.layers.{i}.self_attn.q_proj.weight"),
                    dim,
                    dim,
                )?,
                attn_q_bias: None,
                attn_k: mat(
                    &format!("model.layers.{i}.self_attn.k_proj.weight"),
                    kv_dim,
                    dim,
                )?,
                attn_k_bias: None,
                attn_v: mat(
                    &format!("model.layers.{i}.self_attn.v_proj.weight"),
                    kv_dim,
                    dim,
                )?,
                attn_v_bias: None,
                attn_o: mat(
                    &format!("model.layers.{i}.self_attn.o_proj.weight"),
                    dim,
                    dim,
                )?,
                ffn_gate: mat(
                    &format!("model.layers.{i}.mlp.gate_proj.weight"),
                    ffn_dim,
                    dim,
                )?,
                ffn_up: mat(
                    &format!("model.layers.{i}.mlp.up_proj.weight"),
                    ffn_dim,
                    dim,
                )?,
                ffn_down: mat(
                    &format!("model.layers.{i}.mlp.down_proj.weight"),
                    dim,
                    ffn_dim,
                )?,
                attn_norm_gain: vecf(&format!("model.layers.{i}.input_layernorm.weight"))?,
                ffn_norm_gain: vecf(&format!("model.layers.{i}.post_attention_layernorm.weight"))?,
            });
        }
        let weights = LlamaWeights {
            // See `loader_f32.rs`: a freshly loaded checkpoint is a new weight
            // set, so it mints its own identity (Fuel `c4718467`).
            instance: fuel::decode_shape::ModelInstanceId::next(),
            token_embedding: vecf("model.embed_tokens.weight")?,
            layers,
            final_norm_gain: vecf("model.norm.weight")?,
            output: mat("lm_head.weight", config.vocab_size, dim)
                .context("this helper handles untied heads only (TinyLlama ships one)")?,
        };
        Ok(LlamaModel { config, weights })
    }

    // ---- PROBE 0: chained WriteSlice into ONE bound pool buffer ----------

    /// **The load-bearing unknown this module rests on.** `B` batch rows put
    /// their new K/V in `B` different `(phys, slot)` pairs, which is `B`
    /// `Op::WriteSlice` nodes into ONE bound pool buffer inside ONE graph. They
    /// are chained (each off the previous write's result) because `write_slice`
    /// is destructive and the executor evicts the destroyed input from the
    /// storage cache once the node has run.
    ///
    /// No test in Fuel chains more than one. If chaining silently produced a
    /// fresh buffer instead of mutating the pool, the KV would never persist and
    /// every step after the first would be wrong — **quietly**, because
    /// attention over a zero tail still returns finite numbers.
    ///
    /// So: three chained writes at three different `(phys, slot)` pairs, one
    /// realize, then read the blocks back **through the pool** (a separate
    /// realize on the same Arcs) and check all three landed.
    #[test]
    fn chained_write_slice_lands_every_batch_row_in_the_pool() -> Result<()> {
        let (hkv, hd, block_size, num_blocks) = (2usize, 4usize, 4usize, 6usize);
        let dev = Device::cpu();
        let geom = KvGeometry {
            n_layers: 1,
            num_blocks,
            block_size,
            n_kv_heads: hkv,
            head_dim: hd,
            elem_size: 4,
        };
        let pool =
            DeviceKvPool::new(geom, DType::F32, &dev).map_err(|e| anyhow::anyhow!("{e:?}"))?;

        // Three "batch rows" of new K, shaped exactly as the decoder shapes them:
        // [B, Hkv, 1, D], sliced per row and reshaped to the block's slot layout.
        let b = 3usize;
        let src_data = rand_f32(b * hkv * hd, 7);
        let src = Tensor::from_f32(src_data.clone(), Shape::from_dims(&[b, hkv, 1, hd]), &dev);
        // Deliberately non-monotonic, non-identity targets.
        let targets: [(usize, usize); 3] = [(4, 2), (1, 0), (3, 3)];

        let ph = src.const_placeholder_like(pool.pool_shape().clone(), DType::F32);
        let mut post = ph.clone();
        for (row, &(p, s)) in targets.iter().enumerate() {
            let slab = src
                .slice(0usize, row, 1)
                .and_then(|t| t.reshape(Shape::from_dims(&[1usize, 1, hkv, hd])))
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            post = post
                .write_slice(&slab, vec![(p, p + 1), (s, s + 1), (0, hkv), (0, hd)])
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        }

        let mut ctx = InferenceContext::new(dev.clone());
        ctx.insert(
            ph.node_id(),
            Arc::clone(pool.k_pool(0).expect("layer 0 K buffer")),
        );
        let _ = ctx
            .realize_one_as::<f32>(post.graph_handle(), post.node_id())
            .map_err(|e| anyhow::anyhow!("realize: {e:?}"))?;

        // Read back THROUGH THE POOL — a separate graph over the same Arcs. This
        // is what proves the write landed in the pool rather than in some
        // graph-local copy that died with the realize.
        let per_slot = hkv * hd;
        for (row, &(p, s)) in targets.iter().enumerate() {
            let blk = pool
                .read_block(0, BlockKind::K, p as u32)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let got = &blk[s * per_slot..(s + 1) * per_slot];
            let want = &src_data[row * per_slot..(row + 1) * per_slot];
            assert_eq!(
                got, want,
                "row {row} did not land at block {p} slot {s} — chained write_slice \
                 into one bound pool buffer does not persist, and batched paged decode \
                 cannot be built this way"
            );
        }

        // CONTROL 1: a block nobody wrote is still zero. Without this, a bug that
        // splattered the source across the whole buffer would pass above.
        let untouched = pool
            .read_block(0, BlockKind::K, 5)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        assert!(
            untouched.iter().all(|&v| v == 0.0),
            "block 5 was never written but is not zero — the writes are not \
             confined to their ranges"
        );

        // CONTROL 2: the comparison above actually discriminates. Row 0's data
        // must NOT match block (1,0) — if it did, every assertion above would be
        // vacuous.
        let blk1 = pool
            .read_block(0, BlockKind::K, 1)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        assert_ne!(
            &blk1[0..per_slot],
            &src_data[0..per_slot],
            "CONTROL: row 0's data was found where row 1's belongs — the equality \
             check above is not discriminating between rows"
        );

        // CONTROL 3: the slot within a written block is respected — block 4's
        // slot 0 was not written (row 0 went to slot 2), so it is still zero.
        let blk4 = pool
            .read_block(0, BlockKind::K, 4)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        assert!(
            blk4[0..per_slot].iter().all(|&v| v == 0.0),
            "block 4 slot 0 was never written but is not zero — the slot axis of \
             the write range is being ignored"
        );
        Ok(())
    }

    // ---- PROBE 1: per-row RoPE is exact -----------------------------------

    /// [`rope_batched`] at `B == 1` must be **bit-identical** to Fuel's
    /// `rope_with_tables_decomposed`, and each row at `B == 3` must equal the
    /// `B == 1` result at that row's own position.
    ///
    /// This is the exactness proof for the one piece of math this module
    /// reimplements. A subtly wrong rotate-half (wrong half split, wrong sign,
    /// wrong concat order) still produces plausible tokens, which is the failure
    /// mode that costs a day.
    #[test]
    fn rope_batched_matches_single_position_rope_row_by_row() -> Result<()> {
        let (b, heads, hd) = (3usize, 4usize, 8usize);
        let base = 10000.0f64;
        let positions = [0usize, 7, 3];
        let dev = Device::cpu();

        let x_data = rand_f32(b * heads * hd, 11);
        let x = Tensor::from_f32(x_data.clone(), Shape::from_dims(&[b, heads, 1, hd]), &dev);
        let (cos, sin) = batched_rope_tables(&x, base, &positions, hd)?;
        let got = rope_batched(&x, &cos, &sin)?.realize_f32();

        let per_row = heads * hd;
        for (row, &pos) in positions.iter().enumerate() {
            let row_data = x_data[row * per_row..(row + 1) * per_row].to_vec();
            let xr = Tensor::from_f32(row_data, Shape::from_dims(&[1usize, heads, 1, hd]), &dev);
            let (c, s) = xr.rope_tables_const(base, pos, 1, hd);
            let want = xr
                .rope_with_tables_decomposed(&c, &s)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?
                .realize_f32();
            assert_eq!(
                &got[row * per_row..(row + 1) * per_row],
                &want[..],
                "row {row} (position {pos}) does not match Fuel's own decomposed \
                 RoPE at that position"
            );
        }

        // CONTROL: prove the per-row table is load-bearing. Give EVERY row
        // position 0 (row 0's position) and the rows at other positions must now
        // DISAGREE with their references. If they still agreed, the comparison
        // above would be blind to the whole per-row-position mechanism.
        let shared = [positions[0]; 3];
        let (c0, s0) = batched_rope_tables(&x, base, &shared, hd)?;
        let wrong = rope_batched(&x, &c0, &s0)?.realize_f32();
        for (row, &pos) in positions.iter().enumerate().skip(1) {
            assert_ne!(
                &wrong[row * per_row..(row + 1) * per_row],
                &got[row * per_row..(row + 1) * per_row],
                "CONTROL: row {row} at a SHARED position 0 produced the same values \
                 as at its own position {pos} — per-row RoPE is not actually being \
                 applied, or the comparison cannot see it"
            );
        }
        Ok(())
    }

    // ---- PROBE 2: batched Op::PagedAttn, heterogeneous context lengths ----

    /// `Op::PagedAttn` at `B > 1` with **different `context_len` per row**,
    /// against a hand-written dense softmax per row. Every paged test in Fuel is
    /// `B == 1`; this is the capability the whole module rests on and nothing
    /// upstream covers it.
    #[test]
    fn batched_paged_attn_matches_dense_reference_with_heterogeneous_context_lens() -> Result<()> {
        let (hq, hkv, d, block_size, num_blocks) = (4usize, 2usize, 4usize, 4usize, 24usize);
        let lens = [12usize, 5, 7];
        let b = lens.len();
        let scale = 1.0f32 / (d as f32).sqrt();
        let dev = Device::cpu();

        let geom = KvGeometry {
            n_layers: 1,
            num_blocks,
            block_size,
            n_kv_heads: hkv,
            head_dim: d,
            elem_size: 4,
        };
        let mut pool =
            DeviceKvPool::new(geom, DType::F32, &dev).map_err(|e| anyhow::anyhow!("{e:?}"))?;

        // A filler first, so no session gets an identity physical layout.
        let filler = pool.core_mut().open();
        pool.core_mut()
            .append(filler, 6)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;

        let mut sessions = Vec::new();
        let mut k_logical: Vec<Vec<f32>> = Vec::new();
        let mut v_logical: Vec<Vec<f32>> = Vec::new();
        for (i, &n) in lens.iter().enumerate() {
            let s = pool.core_mut().open();
            pool.core_mut()
                .append(s, n)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let phys = pool
                .core()
                .session_block_table(s)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let per_block = block_size * hkv * d;
            // One block's worth of data per assigned block; the tail beyond `n`
            // tokens is padding the mask must kill.
            let kdat = rand_f32(phys.len() * per_block, 100 + i as u32);
            let vdat = rand_f32(phys.len() * per_block, 200 + i as u32);
            for (bi, &p) in phys.iter().enumerate() {
                pool.write_block(
                    0,
                    BlockKind::K,
                    p,
                    &kdat[bi * per_block..(bi + 1) * per_block],
                )
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                pool.write_block(
                    0,
                    BlockKind::V,
                    p,
                    &vdat[bi * per_block..(bi + 1) * per_block],
                )
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            }
            sessions.push(s);
            k_logical.push(kdat);
            v_logical.push(vdat);
        }

        let pt = pool
            .materialize_block_table(&sessions)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        assert_eq!(pt.context_lens, vec![12u32, 5, 7]);
        assert_eq!(pt.batch, b);

        let q_data = rand_f32(b * hq * d, 42);
        let run = |block_table: Vec<u32>| -> Result<Vec<f32>> {
            let q = Tensor::from_f32(q_data.clone(), Shape::from_dims(&[b, hq, 1, d]), &dev);
            let kc = q.const_placeholder_like(pool.pool_shape().clone(), DType::F32);
            let vc = q.const_placeholder_like(pool.pool_shape().clone(), DType::F32);
            let bt = q.const_u32_like(block_table, pt.block_table_shape());
            let cl = q.const_u32_like(pt.context_lens.clone(), pt.context_lens_shape());
            let out = q
                .paged_attn(&kc, &vc, &bt, &cl, None, scale, block_size, None)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let mut ctx = InferenceContext::new(dev.clone());
            ctx.insert(kc.node_id(), Arc::clone(pool.k_pool(0).unwrap()));
            ctx.insert(vc.node_id(), Arc::clone(pool.v_pool(0).unwrap()));
            ctx.realize_one_as::<f32>(out.graph_handle(), out.node_id())
                .map_err(|e| anyhow::anyhow!("{e:?}"))
        };

        let got = run(pt.block_table.clone())?;

        // Dense reference, per row over that row's OWN context_len, with GQA
        // head mapping (Hq=4, Hkv=2 → n_rep=2).
        let n_rep = hq / hkv;
        let mut want = vec![0.0f32; b * hq * d];
        for row in 0..b {
            let n = lens[row];
            for h in 0..hq {
                let kvh = h / n_rep;
                let mut scores = vec![0.0f32; n];
                for t in 0..n {
                    let mut dot = 0.0f32;
                    for dd in 0..d {
                        dot += q_data[(row * hq + h) * d + dd]
                            * k_logical[row][(t * hkv + kvh) * d + dd];
                    }
                    scores[t] = dot * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - m).exp();
                    denom += *sc;
                }
                for (t, &sc) in scores.iter().enumerate() {
                    let p = sc / denom;
                    for dd in 0..d {
                        want[(row * hq + h) * d + dd] +=
                            p * v_logical[row][(t * hkv + kvh) * d + dd];
                    }
                }
            }
        }

        assert!(
            all_close(&got, &want, 1e-5),
            "batched paged_attn disagrees with the dense reference (max abs diff {})",
            max_abs_diff(&got, &want)
        );

        // CONTROL: swap block-table rows 0 and 1. Row 0 now reads row 1's blocks,
        // so the comparison MUST fail. If it still passed, the block table is
        // being ignored and the test above proves nothing about paging.
        let mut swapped = pt.block_table.clone();
        for c in 0..pt.max_blocks {
            swapped.swap(c, pt.max_blocks + c);
        }
        let corrupted = run(swapped)?;
        assert!(
            !all_close(&corrupted, &want, 1e-5),
            "CONTROL: permuting two rows of the block table changed nothing — \
             Op::PagedAttn is not routing through the block table, so the parity \
             check above is vacuous"
        );
        Ok(())
    }

    // ---- THE ORACLE: batched vs serial ------------------------------------

    /// The three sequences and the **continuous-batching schedule** the oracle
    /// replays: `SCHEDULE[g]` is the set of sequence indices active at global
    /// step `g`, each consuming its next token.
    ///
    /// The shape of this schedule is load-bearing and was arrived at the hard
    /// way (see [`oracle_for`]'s doc):
    ///
    /// - sequences **arrive at different steps** (0, 2, 1), so rows in one batch
    ///   sit at different absolute positions;
    /// - sequence 1 **pauses** at step 3 — that is what makes a shared-position
    ///   bug detectable at all.
    const SCHEDULE: &[&[usize]] = &[
        &[0],       // g0
        &[0, 2],    // g1
        &[0, 1, 2], // g2
        &[0, 2],    // g3 — sequence 1 is descheduled for a step
        &[0, 1, 2], // g4
        &[1],       // g5
    ];

    fn oracle_seqs() -> Vec<Vec<u32>> {
        vec![vec![1, 2, 3, 4, 5], vec![6, 7, 8], vec![9, 10, 11, 12]]
    }

    /// Fuel's own single-sequence `forward_paged_step`, one sequence at a time —
    /// the independent arm the batched decoder is measured against.
    fn serial_reference(model: &LlamaModel, seqs: &[Vec<u32>]) -> Result<Vec<Vec<Vec<f32>>>> {
        let cfg = &model.config;
        let dev = Device::cpu();
        let geom = KvGeometry {
            n_layers: cfg.n_layers,
            num_blocks: 32,
            block_size: 4,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            elem_size: 4,
        };
        let mut pool =
            DeviceKvPool::new(geom, DType::F32, &dev).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let mut out = Vec::new();
        for toks in seqs {
            let s = pool.core_mut().open();
            let mut rows = Vec::new();
            for &t in toks {
                rows.push(
                    model
                        .forward_paged_step(t, &mut pool, s)
                        .map_err(|e| anyhow::anyhow!("forward_paged_step: {e:?}"))?,
                );
            }
            out.push(rows);
        }
        Ok(out)
    }

    /// Replay [`SCHEDULE`] through the batched decoder. When `force_shared_rope`
    /// is set, every row in a step is given row 0's RoPE position — the exact
    /// mistake a naive batched implementation makes.
    ///
    /// Returns per-sequence, per-own-position logits.
    fn batched_run(
        model: &LlamaModel,
        seqs: &[Vec<u32>],
        force_shared_rope: bool,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let mut dec = BatchedPagedDecoder::new(model, 32, 4)?;
        let handles: Vec<SessionHandle> = (0..seqs.len()).map(|_| dec.open_session()).collect();
        let mut out: Vec<Vec<Vec<f32>>> = seqs.iter().map(|_| Vec::new()).collect();
        let mut next: Vec<usize> = vec![0; seqs.len()];
        for active in SCHEDULE {
            let mut sub = Vec::new();
            for &i in active.iter() {
                sub.push((handles[i], seqs[i][next[i]]));
                next[i] += 1;
            }
            let rows = if force_shared_rope {
                let shared = dec
                    .position(handles[active[0]])
                    .expect("row 0 session is open");
                let forced = vec![shared; sub.len()];
                dec.step_forcing_rope_positions(&sub, &forced)?
            } else {
                dec.step(&sub)?
            };
            for (row, &i) in rows.into_iter().zip(active.iter()) {
                out[i].push(row);
            }
        }
        for (i, toks) in seqs.iter().enumerate() {
            assert_eq!(
                next[i],
                toks.len(),
                "SCHEDULE does not consume sequence {i}"
            );
        }
        Ok(out)
    }

    fn parity_failures(a: &[Vec<Vec<f32>>], b: &[Vec<Vec<f32>>]) -> Vec<(usize, usize, f32)> {
        let mut bad = Vec::new();
        for (i, (xs, ys)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(xs.len(), ys.len(), "sequence {i}: step count");
            for (t, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
                if !all_close(x, y, 1e-4) {
                    bad.push((i, t, max_abs_diff(x, y)));
                }
            }
        }
        bad
    }

    /// **THE ORACLE.** Drive a toy model through the batched decoder under a
    /// continuous-batching schedule and, independently, through Fuel's own
    /// single-sequence `forward_paged_step`, and require the logits to agree at
    /// every position of every sequence. The two arms use **separate pools**, so
    /// the physical block layouts differ — parity across physical layouts is a
    /// property worth having, not a coincidence to be avoided.
    ///
    /// # What this oracle can and cannot see (learned by mutation-testing it)
    ///
    /// The first version of this test ran three different-length sequences in
    /// lockstep from step 0. Mutating `batched_rope_tables` to use `positions[0]`
    /// for **every** row left it fully green. So did shifting every position by
    /// +100. Both mutations were caught only by
    /// `rope_batched_matches_single_position_rope_row_by_row`.
    ///
    /// The reason is **RoPE's relative invariance**, which is the whole point of
    /// RoPE and not a bug: `q(m)·k(n)` depends only on `m − n`, so adding a
    /// constant to *every* position of a sequence — queries and cached keys
    /// alike — changes nothing observable in the logits. In a lockstep schedule
    /// "use row 0's position" *is* a constant per-sequence offset, so it is
    /// invisible here by construction.
    ///
    /// [`SCHEDULE`] therefore staggers arrival **and pauses sequence 1 for a
    /// step**, which makes the shared-position error non-uniform within that
    /// sequence and hence observable. `oracle_catches_a_shared_rope_position`
    /// asserts exactly that, so the sensitivity is a checked property rather
    /// than a claim in a comment.
    ///
    /// Stated plainly: this oracle pins **relative** position handling. Absolute
    /// positions are pinned by the bit-exact
    /// `rope_batched_matches_single_position_rope_row_by_row`, which compares
    /// against Fuel's own table builder at each row's own position.
    ///
    /// # Mutation results (run against this suite, recorded so the coverage
    /// claim is checkable rather than asserted)
    ///
    /// | mutation to `batched.rs` | caught by |
    /// | --- | --- |
    /// | every row uses `positions[0]` | this oracle (via `SCHEDULE`'s pause) + the RoPE unit test |
    /// | every position shifted `+100` | RoPE unit test ONLY — RoPE relative invariance, see above |
    /// | writes fan out off `k_ph` instead of chaining | this oracle + `prefill_batch_agrees_with_serial_paged_prefill` |
    /// | every row writes row 0's K slab | same two |
    ///
    /// Toy weights, so this runs in CI in milliseconds.
    fn oracle_for(cfg: LlamaConfig) -> Result<()> {
        let model = LlamaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg, 9999),
        };
        let seqs = oracle_seqs();

        let serial = serial_reference(&model, &seqs)?;
        let batched = batched_run(&model, &seqs, false)?;

        let bad = parity_failures(&batched, &serial);
        assert!(
            bad.is_empty(),
            "batched vs serial paged decode differ at (sequence, position, max abs diff): {bad:?}"
        );

        // CONTROL: the comparison must be able to fail. Re-run sequence 0 with
        // ONE token changed and require the check to catch it — two paths
        // agreeing because both are degenerate is a harness smell, not a result.
        let mut dec2 = BatchedPagedDecoder::new(&model, 32, 4)?;
        let h2 = dec2.open_session();
        let mut corrupted = Vec::new();
        for (t, &tok) in seqs[0].iter().enumerate() {
            let fed = if t == 1 {
                tok.wrapping_add(1) % cfg.vocab_size as u32
            } else {
                tok
            };
            corrupted.push(dec2.step(&[(h2, fed)])?.remove(0));
        }
        let differs = corrupted
            .iter()
            .zip(serial[0].iter())
            .any(|(a, b)| !all_close(a, b, 1e-4));
        assert!(
            differs,
            "CONTROL: feeding a different token produced identical logits — the \
             1e-4 comparison is not sensitive to the input, so the parity above is \
             vacuous"
        );
        Ok(())
    }

    /// **Regression test for the shared-frontier-block corruption.**
    ///
    /// This is the hazard the old hand-rolled `(phys, slot)` guard was blind to,
    /// and which had ZERO coverage: deleting that guard entirely changed no test
    /// result. Two sessions sharing a physical block at different fill levels
    /// have different slots, so the pairwise comparison never fired while the
    /// write corrupted the co-sharer.
    ///
    /// Three things are asserted, and the third is the one that matters most:
    ///   1. copy-on-write fires — the writer's frontier block id CHANGES;
    ///   2. the donor's bytes are untouched;
    ///   3. the writer's NEW block carries a COPY of the shared prefix.
    ///
    /// (3) is what separates Fuel's `ensure_writable_block` from the "just check
    /// `block_refcount > 1`" fix that suggests itself: a bare refcount check
    /// detects the share but leaves the session pointing at a *blank* fresh
    /// block, silently discarding the prefix it spliced in. Detecting is not
    /// enough; the bytes have to move.
    #[test]
    fn a_spliced_session_cows_its_frontier_instead_of_corrupting_the_donor() -> Result<()> {
        let cfg = toy_cfg(2);
        let model = LlamaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg, 9999),
        };
        let block_size = 4;
        let mut dec = BatchedPagedDecoder::new(&model, 32, block_size)?;

        // Donor: 6 tokens => block 0 full, block 1 holding 2.
        let a = dec.open_session();
        for t in 0..6u32 {
            dec.step(&[(a, t % cfg.vocab_size as u32)])?;
        }
        assert_eq!(dec.position(a), Some(6));

        // Sharer: splice the donor's two logical blocks in.
        let b = dec.open_session();
        dec.pool_mut()
            .core_mut()
            .splice(a, b, 0, 2)
            .map_err(|e| anyhow::anyhow!("splice: {e:?}"))?;

        let donor_frontier = dec
            .pool()
            .core()
            .resident_block(b, 1)
            .context("b's frontier block is not resident after the splice")?;
        assert_eq!(
            dec.pool().core().resident_block(a, 1),
            Some(donor_frontier),
            "the splice did not actually share logical block 1 — nothing below tests sharing"
        );
        let rc = dec.pool().core().block_refcount(donor_frontier);
        assert!(
            rc > 1,
            "refcount of the frontier block is {rc}, so it is not shared and this test is vacuous"
        );

        // Everything the donor holds, before the sharer writes anything.
        let n_layers = cfg.n_layers;
        let donor_before: Vec<(Vec<f32>, Vec<f32>)> = (0..n_layers)
            .map(|l| -> Result<(Vec<f32>, Vec<f32>)> {
                Ok((
                    dec.pool()
                        .read_block(l, BlockKind::K, donor_frontier)
                        .map_err(|e| anyhow::anyhow!("read K: {e:?}"))?,
                    dec.pool()
                        .read_block(l, BlockKind::V, donor_frontier)
                        .map_err(|e| anyhow::anyhow!("read V: {e:?}"))?,
                ))
            })
            .collect::<Result<_>>()?;

        // The sharer decodes one token into what is currently a SHARED block.
        dec.step(&[(b, 1u32)])?;

        // 1. Copy-on-write fired.
        let new_frontier = dec
            .pool()
            .core()
            .resident_block(b, 1)
            .context("b's frontier vanished")?;
        assert_ne!(
            new_frontier, donor_frontier,
            "b decoded straight into the block it shares with a — copy-on-write did not fire, so \
             this write lands in the donor's KV"
        );

        // 2. The donor is untouched, every layer, K and V.
        for (l, (k_before, v_before)) in donor_before.iter().enumerate() {
            let k_after = dec
                .pool()
                .read_block(l, BlockKind::K, donor_frontier)
                .map_err(|e| anyhow::anyhow!("read K: {e:?}"))?;
            let v_after = dec
                .pool()
                .read_block(l, BlockKind::V, donor_frontier)
                .map_err(|e| anyhow::anyhow!("read V: {e:?}"))?;
            assert_eq!(
                *k_before, k_after,
                "layer {l}: b's decode mutated the donor's K block — this is the silent \
                 cross-session corruption, and it produces wrong logits rather than an error"
            );
            assert_eq!(
                *v_before, v_after,
                "layer {l}: b's decode mutated the donor's V block"
            );
        }

        // 3. The prefix survived the break: slots 0..2 of the new block match
        //    the donor's. Slot 2 is where b just wrote, so it is excluded.
        let per_slot = cfg.n_kv_heads * cfg.head_dim;
        let shared_prefix = 2 * per_slot;
        for (l, (k_before, v_before)) in donor_before.iter().enumerate() {
            let k_new = dec
                .pool()
                .read_block(l, BlockKind::K, new_frontier)
                .map_err(|e| anyhow::anyhow!("read K: {e:?}"))?;
            let v_new = dec
                .pool()
                .read_block(l, BlockKind::V, new_frontier)
                .map_err(|e| anyhow::anyhow!("read V: {e:?}"))?;
            assert_eq!(
                k_before[..shared_prefix],
                k_new[..shared_prefix],
                "layer {l}: the copy-on-write block does NOT carry the spliced prefix's K — the \
                 share was broken but the bytes were not copied, so b silently lost its prefix"
            );
            assert_eq!(
                v_before[..shared_prefix],
                v_new[..shared_prefix],
                "layer {l}: the copy-on-write block does NOT carry the spliced prefix's V"
            );
        }

        Ok(())
    }

    #[test]
    fn batched_decode_matches_serial_paged_decode_no_gqa() -> Result<()> {
        oracle_for(toy_cfg(4))
    }

    #[test]
    fn batched_decode_matches_serial_paged_decode_gqa() -> Result<()> {
        oracle_for(toy_cfg(2))
    }

    /// **CONTROL for the oracle's most important sensitivity.** Replay the same
    /// schedule but give every row in a step row 0's RoPE position — the mistake
    /// a batched decoder makes if it reaches for
    /// `Tensor::rope_with_tables_decomposed` (which only accepts one shared
    /// position at `[B, H, 1, D]`) instead of building per-row tables.
    ///
    /// The oracle MUST catch it. If this test ever starts failing, the oracle has
    /// gone blind to per-row positions and its green is worth nothing — most
    /// likely because someone made the schedule lockstep again (see
    /// [`oracle_for`]'s doc for why that hides the bug entirely).
    #[test]
    fn oracle_catches_a_shared_rope_position() -> Result<()> {
        let cfg = toy_cfg(2);
        let model = LlamaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg, 9999),
        };
        let seqs = oracle_seqs();
        let serial = serial_reference(&model, &seqs)?;
        let shared = batched_run(&model, &seqs, true)?;
        let bad = parity_failures(&shared, &serial);
        assert!(
            !bad.is_empty(),
            "CONTROL: forcing every row to share row 0's RoPE position produced \
             logits the oracle accepts — the oracle cannot see per-row positions, \
             so `batched_decode_matches_serial_paged_decode_*` proves nothing about \
             them"
        );
        // Name where it bit, so a future reader can see the mechanism rather than
        // trusting the assertion: sequence 1 is the one that pauses, so its
        // position error is non-uniform and therefore visible.
        assert!(
            bad.iter().any(|&(seq, _, _)| seq == 1),
            "expected the paused sequence (1) to be among the failures, got {bad:?}"
        );
        Ok(())
    }

    /// A batch of one must reproduce Fuel's own `forward_paged_step` — the
    /// narrowest possible check that the reimplemented layer (RoPE, chained
    /// write, norm, FFN) is the same math, with batching taken out of the
    /// picture entirely.
    #[test]
    fn batch_of_one_matches_forward_paged_step() -> Result<()> {
        let cfg = toy_cfg(2);
        let model = LlamaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg, 4242),
        };
        let toks = [3u32, 1, 4, 1, 5, 9, 2, 6];

        let dev = Device::cpu();
        let geom = KvGeometry {
            n_layers: cfg.n_layers,
            num_blocks: 16,
            block_size: 4,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            elem_size: 4,
        };
        let mut pool =
            DeviceKvPool::new(geom, DType::F32, &dev).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let s = pool.core_mut().open();

        let mut dec = BatchedPagedDecoder::new(&model, 16, 4)?;
        let h = dec.open_session();

        for (t, &tok) in toks.iter().enumerate() {
            let want = model
                .forward_paged_step(tok, &mut pool, s)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let got = dec.step(&[(h, tok)])?.remove(0);
            assert!(
                all_close(&got, &want, 1e-5),
                "position {t}: B=1 batched != forward_paged_step (max abs diff {})",
                max_abs_diff(&got, &want)
            );
        }
        Ok(())
    }

    /// `prefill_batch` must land each sequence on the same logits the
    /// step-by-step drive produces at its own last prompt token — i.e. the
    /// convenience verb is not a second implementation.
    #[test]
    fn prefill_batch_agrees_with_serial_paged_prefill() -> Result<()> {
        let cfg = toy_cfg(2);
        let model = LlamaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg, 777),
        };
        let p0: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7];
        let p1: Vec<u32> = vec![8, 9];
        let prompts: Vec<&[u32]> = vec![&p0, &p1];

        let mut dec = BatchedPagedDecoder::new(&model, 32, 4)?;
        let sessions: Vec<SessionHandle> = (0..2).map(|_| dec.open_session()).collect();
        let got = dec.prefill_batch(&sessions, &prompts)?;

        let dev = Device::cpu();
        let geom = KvGeometry {
            n_layers: cfg.n_layers,
            num_blocks: 32,
            block_size: 4,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            elem_size: 4,
        };
        let mut pool =
            DeviceKvPool::new(geom, DType::F32, &dev).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        for (i, p) in prompts.iter().enumerate() {
            let s = pool.core_mut().open();
            let mut last = Vec::new();
            for &t in p.iter() {
                last = model
                    .forward_paged_step(t, &mut pool, s)
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            }
            assert!(
                all_close(&got[i], &last, 1e-4),
                "prefill_batch sequence {i} != serial paged prefill (max abs diff {})",
                max_abs_diff(&got[i], &last)
            );
        }
        // Positions advanced by exactly the prompt lengths.
        assert_eq!(dec.position(sessions[0]), Some(p0.len()));
        assert_eq!(dec.position(sessions[1]), Some(p1.len()));
        Ok(())
    }

    // ---- admission is Lightbulb's, and all-or-nothing ---------------------

    /// `can_admit` must agree with what actually happens, and a step that cannot
    /// fit must leave **every** sequence's position untouched — `append` is
    /// per-session, so a naive loop would grow rows 0..k-1 and then fail on row
    /// k, and nothing downstream could tell how far it got.
    #[test]
    fn admission_is_all_or_nothing_and_leaves_no_partial_growth() -> Result<()> {
        let cfg = toy_cfg(2);
        let model = LlamaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg, 5),
        };
        // 3 blocks of 1 token: block boundaries every step, so the third row's
        // append is exactly the one that cannot be served.
        let mut dec = BatchedPagedDecoder::new(&model, 3, 1)?;
        let a = dec.open_session();
        let b = dec.open_session();
        let c = dec.open_session();

        assert!(
            dec.can_admit(&[(0, 1), (0, 1), (0, 1)]),
            "3 blocks, 3 tokens"
        );
        dec.step(&[(a, 1), (b, 2), (c, 3)])?;
        assert_eq!(dec.free_blocks(), 0);

        // Now nothing fits.
        assert!(!dec.can_admit(&[(1, 1), (1, 1), (1, 1)]));
        let before: Vec<Option<usize>> = [a, b, c].iter().map(|&s| dec.position(s)).collect();
        let err = dec
            .step(&[(a, 1), (b, 2), (c, 3)])
            .expect_err("a full pool must refuse the step");
        assert!(
            err.to_string().contains("pool exhausted"),
            "expected a capacity error, got: {err}"
        );
        let after: Vec<Option<usize>> = [a, b, c].iter().map(|&s| dec.position(s)).collect();
        assert_eq!(
            before, after,
            "the refused step moved some sequence's position — admission is not \
             all-or-nothing and the batch is now inconsistent"
        );

        // CONTROL: the check is not just "always refuse". Free one block and a
        // one-row batch must succeed again.
        dec.close_session(c);
        assert_eq!(dec.free_blocks(), 1);
        assert!(dec.can_admit(&[(1, 1)]));
        dec.step(&[(a, 1)])?;
        assert_eq!(dec.position(a), Some(2));
        Ok(())
    }

    /// A session listed twice in one batch would be appended twice for one token
    /// and would occupy two rows of the block table — wrong, and silently so.
    #[test]
    fn a_duplicated_session_in_one_batch_is_rejected() -> Result<()> {
        let cfg = toy_cfg(2);
        let model = LlamaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg, 6),
        };
        let mut dec = BatchedPagedDecoder::new(&model, 8, 4)?;
        let s = dec.open_session();
        let err = dec
            .step(&[(s, 1), (s, 2)])
            .expect_err("the same session twice in one batch must be refused");
        assert!(err.to_string().contains("twice"), "got: {err}");
        assert_eq!(dec.position(s), Some(0), "the refused step still appended");
        Ok(())
    }

    /// Rule 2, as an executable gate: a non-F32 projection must be refused at
    /// construction, naming the weight and saying what to do.
    ///
    /// Uses `Q4_0` rather than `BF16` only because Lightbulb does not depend on
    /// `half` and Fuel does not re-export it; the gate is `!matches!(_,
    /// WeightStorage::F32(_))`, so both variants take the same arm.
    #[test]
    fn non_f32_weights_are_refused_at_construction() {
        let cfg = toy_cfg(2);
        let mut w = tiny_weights(&cfg, 8);
        w.layers[0].attn_q = WeightStorage::Q4_0 {
            words: Arc::from(vec![0u32; 4]),
            bytes_len: 16,
            in_features: cfg.dim,
            out_features: cfg.dim,
        };
        let model = LlamaModel {
            config: cfg,
            weights: w,
        };
        // `expect_err` would need `Debug` on the decoder; match instead.
        let msg = match BatchedPagedDecoder::new(&model, 8, 4) {
            Ok(_) => panic!("a non-F32 projection must be refused at construction"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("attn_q"),
            "the error must name the weight: {msg}"
        );
        assert!(
            msg.contains("loader_f32"),
            "the error must say what to do: {msg}"
        );
    }

    // ---- TinyLlama, behind #[ignore] --------------------------------------

    /// **End to end on a real checkpoint**: two prompts of different lengths
    /// decoded together in one batch, sampled by Lightbulb (argmax over realized
    /// logits), detokenized, and checked for *content* — a mis-wired batched path
    /// still emits tokens, it just emits nonsense.
    ///
    /// `#[ignore]`: needs the 2.2 GB TinyLlama checkpoint, loads it as f32
    /// (~4.4 GB), and the paged path re-optimises the graph every step. Run in
    /// release:
    ///
    /// `cargo test --release --lib model_fuel::batched -- --ignored --nocapture`
    ///
    /// Observed on this machine (6-token prompts, 4 generated tokens, B=2
    /// throughout):
    ///
    /// ```text
    /// seq 0: "Paris.\n\n"
    /// seq 1: "Tokyo.\n\n"
    /// 10 batched steps (B<=2) in 217.3s (21.73s/step)
    /// ```
    ///
    /// That is ~10.9 s per sequence-token against `generate.rs`'s 4.52 s/token
    /// single-sequence contiguous-persistent baseline — see this module's docs
    /// for why, and for why that is not a like-for-like comparison.
    #[test]
    #[ignore = "needs the TinyLlama checkpoint (~4.4 GB as f32); slow on CPU"]
    fn tinyllama_two_prompts_in_one_batch() -> Result<()> {
        let Some(dir) = tinyllama_dir() else {
            crate::test_notice::skip_unless_required(
                "LIGHTBULB_REQUIRE_MODEL",
                "no TinyLlama snapshot",
            );
            return Ok(());
        };
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let encode = |s: &str| -> Result<Vec<u32>> {
            Ok(tok
                .encode(s, true)
                .map_err(|e| anyhow::anyhow!("encode: {e}"))?
                .get_ids()
                .to_vec())
        };

        // The f32 loader, per rule 2 and the pool's f32 gate.
        let model = load_f32_checkpoint(&dir)?;
        let model = &model;

        let p0 = encode("The capital of France is")?;
        let p1 = encode("The capital of Japan is")?;
        let max_new = 4usize;

        // 64 blocks of 16 tokens = 1024 token-slots, plenty for two short seqs.
        let mut dec = BatchedPagedDecoder::new(model, 64, 16)?;
        assert!(
            dec.can_admit_prompts(&[p0.len() + max_new, p1.len() + max_new]),
            "pool too small for both prompts plus generation"
        );
        let s0 = dec.open_session();
        let s1 = dec.open_session();

        let started = std::time::Instant::now();
        let prompts: Vec<&[u32]> = vec![&p0, &p1];
        let mut logits = dec.prefill_batch(&[s0, s1], &prompts)?;
        let mut steps = p0.len().max(p1.len());

        let argmax = |row: &[f32]| -> u32 {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            best as u32
        };

        let mut out: Vec<Vec<u32>> = vec![Vec::new(), Vec::new()];
        for _ in 0..max_new {
            // Sampling: LIGHTBULB'S, host-side, over realized logits.
            let t0 = argmax(&logits[0]);
            let t1 = argmax(&logits[1]);
            out[0].push(t0);
            out[1].push(t1);
            logits = dec.step(&[(s0, t0), (s1, t1)])?;
            steps += 1;
        }
        let elapsed = started.elapsed();

        let d0 = tok
            .decode(&out[0], true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let d1 = tok
            .decode(&out[1], true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        eprintln!("seq 0: {d0:?}");
        eprintln!("seq 1: {d1:?}");
        eprintln!(
            "{steps} batched steps (B<=2) in {:.1?} ({:.2?}/step)",
            elapsed,
            elapsed / steps.max(1) as u32
        );

        assert!(
            d0.to_lowercase().contains("paris"),
            "sequence 0 should name Paris, got {d0:?} — the batched path runs but \
             produces nonsense, which points at the wiring (per-row RoPE, the \
             chained KV write, the block table row order) rather than the plumbing"
        );
        assert!(
            d1.to_lowercase().contains("tokyo"),
            "sequence 1 should name Tokyo, got {d1:?}. If sequence 0 is right and \
             this one is wrong, suspect the batch axis: row 1 may be reading row \
             0's blocks or row 0's RoPE position"
        );
        Ok(())
    }

    /// **The B-sweep** — does per-sequence cost actually fall as `B` grows?
    ///
    /// An earlier comparison put 4.52 s (contiguous, bf16, B=1) against ~10.9 s
    /// (paged, f32, B=2) and said plainly it was not apples to apples. Three
    /// of those four confounds are removable and this test removes them: both
    /// arms run **in one process, off one checkpoint load, with f32 weights on
    /// both sides**, so dtype, machine and day are held fixed. The fourth — no
    /// `DecodeSession` on the paged path, so every step re-optimises the whole
    /// graph — is structural (rule 3 is not achieved here, see the module
    /// header) and is not a confound but the very thing under measurement.
    ///
    /// **Uniform context lengths are deliberate and they bias the result.**
    /// Every row gets the same prompt length and all rows advance in lockstep,
    /// which is the *best case* for batching: no ragged padding waste, maximum
    /// shared work per step. Real continuous batching is ragged — sessions
    /// arrive and finish at different steps and never realign. So a **negative**
    /// result here is conclusive (it cannot get better with ragged input), while
    /// a **positive** result is an upper bound that ragged serving will not
    /// reach. Read the two directions asymmetrically.
    ///
    /// Reports per-sequence-token cost: wall-clock per step ÷ B. This is the
    /// number the decision turns on — a batched step that costs B times a serial
    /// step has bought nothing.
    ///
    /// **Each arm also reports its first step separately, and that ratio is the
    /// load-bearing evidence.** A plan-once path pays for the plan on step 0 and
    /// replays afterwards, so warm-up ≫ steady. A path that re-plans every token
    /// pays the same price every step, so warm-up ≈ steady. That distinguishes
    /// *"paging is expensive"* from *"this path re-plans every token"* without
    /// instrumenting Fuel at all — and the two have entirely different fixes.
    /// Without it the headline ratio is ambiguous between the two, which would
    /// make it easy to write off paging on the strength of a missing optimiser.
    ///
    /// Asserts only non-vacuity (that every arm really decoded, full-width
    /// logits, positions actually advanced), never a latency threshold — a
    /// timing assert would be flaky and this is a measurement harness, not a
    /// regression gate.
    ///
    /// Env overrides: `LB_SWEEP_B` (default `1,2,4,8`), `LB_SWEEP_PROMPT`
    /// (default 8), `LB_SWEEP_STEPS` (default 4).
    #[test]
    #[ignore = "needs the TinyLlama checkpoint; minutes per B in release"]
    fn b_sweep_per_sequence_cost() -> Result<()> {
        use fuel::inference_context::KvCache;
        use std::time::{Duration, Instant};

        // Deliberately NOT the `return Ok(())` skip the other tests here use.
        // An early return from a `#[test]` is a **pass**: a missing or moved
        // checkpoint would print green having measured nothing, and every
        // non-vacuity guard below — logits width, positions advanced,
        // `session.is_some()` — sits downstream of this gate and never runs.
        // This test is `#[ignore]`d and only ever invoked by name, so an absent
        // checkpoint is a broken invocation, not a routine skip.
        let dir = tinyllama_dir().expect(
            "no TinyLlama snapshot: this is a measurement harness, so it fails \
             rather than skipping — a green with no numbers behind it is worse \
             than a red",
        );

        let env_usize = |k: &str, d: usize| -> usize {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let prompt_len = env_usize("LB_SWEEP_PROMPT", 8);
        let steps = env_usize("LB_SWEEP_STEPS", 4);
        let bs: Vec<usize> = std::env::var("LB_SWEEP_B")
            .unwrap_or_else(|_| "1,2,4,8".to_string())
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .filter(|&b| b > 0)
            .collect();
        assert!(prompt_len > 0, "LB_SWEEP_PROMPT must be > 0");
        assert!(steps > 0, "LB_SWEEP_STEPS must be > 0");
        assert!(!bs.is_empty(), "LB_SWEEP_B parsed to nothing");

        const BLOCK: usize = 16;
        // Distinct token ids per row. Cost is shape-driven, not value-driven, so
        // synthetic ids measure the same thing as real ones while giving exact
        // control over context length — which is what keeps the rows uniform.
        let make_prompt = |row: usize| -> Vec<u32> {
            (0..prompt_len)
                .map(|t| ((row * 131 + t * 17) % 1000 + 1) as u32)
                .collect()
        };
        let argmax = |row: &[f32]| -> u32 {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    best = i;
                }
            }
            best as u32
        };
        let median = |mut v: Vec<Duration>| -> Duration {
            v.sort_unstable();
            v[v.len() / 2]
        };

        let t_load = Instant::now();
        let model = load_f32_checkpoint(&dir)?;
        let model = &model;
        let vocab = model.config.vocab_size;
        eprintln!(
            "f32 checkpoint loaded in {:.1?} | prompt_len={prompt_len} steps={steps} \
             (+1 warm-up, dropped) | B set = {bs:?}",
            t_load.elapsed()
        );

        // ---- Arm A: contiguous + persistent DecodeSession, B=1 ----------------
        // The current default decode path, exactly as `generate.rs` shapes it:
        // `with_capacity` (not `with_dims`) plus `forward_with_kv_context_
        // PERSISTENT` (not the plain variant). Both choices are load-bearing and
        // documented there; getting either wrong here would flatter the paged arm.
        let contiguous = {
            let device = Device::cpu();
            let c = &model.config;
            let mut cache = KvCache::with_capacity(
                c.n_layers,
                c.n_kv_heads,
                c.head_dim,
                prompt_len + steps + 3,
                fuel::DType::F32,
                &device,
            )
            .map_err(|e| anyhow::anyhow!("baseline KV cache: {e:?}"))?;
            let mut ctx = InferenceContext::new(device);
            let mut session: Option<fuel::inference_context::DecodeSession> = None;

            let p = make_prompt(0);
            let mut logits = model
                .forward_with_kv_context_persistent(&p, &mut cache, &mut ctx, &mut session)
                .map_err(|e| anyhow::anyhow!("baseline prefill: {e:?}"))?;
            assert_eq!(logits.len(), vocab, "baseline prefill logits width");

            let mut per = Vec::with_capacity(steps);
            let mut warmup = Duration::ZERO;
            let mut next = argmax(&logits);
            for i in 0..=steps {
                let t0 = Instant::now();
                logits = model
                    .forward_with_kv_context_persistent(&[next], &mut cache, &mut ctx, &mut session)
                    .map_err(|e| anyhow::anyhow!("baseline decode: {e:?}"))?;
                let dt = t0.elapsed();
                // Step 0 is warm-up: the persistent session is built on first
                // use, so including it would charge plan-once to every step.
                // It is reported rather than discarded — see the note below on
                // why the warm-up/steady ratio is the load-bearing evidence.
                if i > 0 {
                    per.push(dt);
                } else {
                    warmup = dt;
                }
                next = argmax(&logits);
            }
            assert_eq!(logits.len(), vocab, "baseline decode logits width");
            assert_eq!(per.len(), steps, "baseline measured step count");
            assert!(
                session.is_some(),
                "baseline never built a DecodeSession — the persistent path did \
                 not engage, so this measured the re-optimising loop and would \
                 understate the contiguous arm by ~2.5x"
            );
            let m = median(per.clone());
            eprintln!("  contiguous persistent B=1  {m:>10.3?}/step  {m:>10.3?}/seq-token");
            eprintln!("      warm-up {warmup:?} then {per:?}");
            m
        };

        // ---- Arm B: batched paged, B in bs ------------------------------------
        let mut rows: Vec<(usize, Duration, Duration)> = Vec::new();
        for &b in &bs {
            let per_seq_blocks = (prompt_len + steps + 2).div_ceil(BLOCK);
            let num_blocks = per_seq_blocks * b + 4;
            let mut dec = BatchedPagedDecoder::new(model, num_blocks, BLOCK)?;

            let sessions: Vec<SessionHandle> = (0..b).map(|_| dec.open_session()).collect();
            let owned: Vec<Vec<u32>> = (0..b).map(make_prompt).collect();
            let prompts: Vec<&[u32]> = owned.iter().map(|p| p.as_slice()).collect();

            let mut logits = dec.prefill_batch(&sessions, &prompts)?;
            assert_eq!(logits.len(), b, "prefill returned one row per session");

            let mut per = Vec::with_capacity(steps);
            let mut warmup = Duration::ZERO;
            for i in 0..=steps {
                let batch: Vec<(SessionHandle, u32)> = sessions
                    .iter()
                    .zip(logits.iter())
                    .map(|(&s, row)| (s, argmax(row)))
                    .collect();
                let t0 = Instant::now();
                logits = dec.step(&batch)?;
                let dt = t0.elapsed();
                if i > 0 {
                    per.push(dt);
                } else {
                    warmup = dt;
                }
            }

            // Non-vacuity: every row really decoded and really advanced. Without
            // this a silently-empty batch would post a spectacular per-token cost.
            assert_eq!(logits.len(), b, "step returned one row per session");
            for (i, row) in logits.iter().enumerate() {
                assert_eq!(row.len(), vocab, "row {i} logits width");
            }
            for (i, &s) in sessions.iter().enumerate() {
                assert_eq!(
                    dec.position(s),
                    Some(prompt_len + steps + 1),
                    "row {i} did not advance one position per step"
                );
            }
            assert_eq!(per.len(), steps, "measured step count at B={b}");

            let m = median(per.clone());
            let per_seq = m / (b as u32);
            eprintln!("  batched paged      B={b:<2} {m:>10.3?}/step  {per_seq:>10.3?}/seq-token");
            eprintln!("      warm-up {warmup:?} then {per:?}");
            rows.push((b, m, per_seq));
        }

        // ---- The comparison the decision turns on -----------------------------
        eprintln!();
        eprintln!("  B    per-step      per-seq-token   vs contiguous");
        eprintln!("  ---  ------------  --------------  -------------");
        for &(b, step, per_seq) in &rows {
            let ratio = per_seq.as_secs_f64() / contiguous.as_secs_f64();
            let verdict = if ratio < 1.0 { "FASTER" } else { "slower" };
            eprintln!("  {b:<3}  {step:>12.3?}  {per_seq:>14.3?}  {ratio:>8.2}x {verdict}");
        }
        let best = rows
            .iter()
            .min_by(|a, c| a.2.cmp(&c.2))
            .expect("bs is non-empty");
        eprintln!();
        eprintln!(
            "  best paged per-seq-token: {:.3?} at B={} | contiguous baseline: {:.3?}",
            best.2, best.0, contiguous
        );
        eprintln!(
            "  => paged {} the contiguous baseline at TinyLlama scale on CPU, \
             under lockstep-uniform contexts (the batched path's best case).",
            if best.2 < contiguous {
                "BEATS"
            } else {
                "does NOT beat"
            }
        );
        Ok(())
    }
}
