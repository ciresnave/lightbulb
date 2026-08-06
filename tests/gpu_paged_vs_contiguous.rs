//! **GPU paged decode vs GPU contiguous decode — the placement question, measured.**
//!
//! # Why this exists
//!
//! Three sessions produced three mutually exclusive accounts of where
//! `Op::PagedAttn` executes on a CUDA target, and six retractions between them.
//! In order:
//!
//! 1. *CPU-only.* `Op::PagedAttn` has no fused CUDA kernel in Fuel's binding
//!    table, so it falls back to CPU with H2D/D2H copies stitched around it.
//! 2. *GPU via decompose.* The op's lowering rule is total, so it decomposes to
//!    gather+SDPA primitives that do have CUDA kernels.
//! 3. *GPU via capability gating.* `opt.rs`'s `capability_gated_rules` installs
//!    a lowering rule for every fused op and a fusion rule only where a kernel
//!    exists, so a kernel-absent node lowers and never re-fuses.
//!
//! Each was argued from real code. **None of the three cited machinery that
//! runs.** `RuleRegistry::{lowering_only, default_rules, capability_gated_rules}`
//! have no production caller — every call site sits below `#[cfg(test)]` at
//! `fuel-core/src/lazy.rs:1776`, or in `opt.rs`'s own tests, or in a doc
//! comment. `lazy.rs:3827` states the reason outright: *"optimizer is
//! caller-composed."* No production code composes one.
//!
//! So a fused `PagedAttn` node reaches the executor unlowered — and **what the
//! executor then does with it is still unknown.** That is the question here.
//!
//! # Why reading cannot answer it, and this can
//!
//! The oscillation is the diagnostic: three positions and six retractions means
//! the method failed, not that the next read will succeed. Reading found
//! mechanisms; it could not establish which mechanisms *run*.
//!
//! The CUDA test offered as proof (`fuel-core/src/lazy.rs:14268`,
//! `bf16_paged_decode_matches_contiguous_on_cuda`) asserts only
//! `argmax(paged[i]) == argmax(contig[i])`. **Numerical parity is invariant to
//! placement** — a CPU fallback with copies around it passes identically, and
//! arguably more exactly. Its "runs on GPU via decompose" claim lives in the doc
//! comment, asserted by nothing executable. Evidence that would look the same
//! whether the claim is true or false carries no information about the claim.
//!
//! # The instrument
//!
//! Wall clock alone **cannot** answer this, and must not be reported as if it
//! could. A small gap is ambiguous between "the round-trip is cheap" and "no
//! round-trip happened because neither arm reached the GPU at all". So timing is
//! the secondary output. The primary instrument is **`nsys`**, run outside this
//! harness against one arm at a time:
//!
//! ```text
//! nsys profile --trace=cuda --stats=true -o paged \
//!   cargo test --release --features fuel-cuda --test gpu_paged_vs_contiguous \
//!   -- --ignored --nocapture
//! ```
//!
//! with `LB_ARM=paged` / `LB_ARM=contiguous` selecting a single arm so transfer
//! counts attribute cleanly to one code path. `nsys` observes `cudaMemcpyAsync`
//! DtoH/HtoD counts and kernel launches **at the driver level, beneath Fuel** —
//! it cannot be misled by a stale doc comment, a rule that is never installed,
//! or a placement-invariant assertion.
//!
//! Fuel's own `FUEL_TRACE=1` was the first candidate and is **useless here**:
//! `fuel-core` does not depend on `tracing` at all, `fuel-dispatch` emits two
//! non-span macros repo-wide, and `fuel-cuda-backend` declares `tracing` but
//! emits **zero** spans. Only `fuel-vulkan-backend` is instrumented. A CUDA run
//! under `FUEL_TRACE` shows no transfers because none are recorded — which
//! would have produced a confident *wrong* answer, not a null one.
//!
//! # Registered before the first run
//!
//! * **If `PagedAttn` falls back to CPU:** the paged arm shows a per-token burst
//!   of DtoH+HtoD that the contiguous arm does not, scaling with token count.
//! * **If it executes on device:** both arms transfer only at const upload and
//!   final readback, and the paged arm's kernel launches include the gather/SDPA
//!   primitives — which would also settle the lowering question directly.
//! * **The control that must not be skipped:** if *both* arms show near-zero
//!   kernel launches, neither reached the GPU (weights start on CPU, and a
//!   transfer-aware planner may decline to move them), the comparison is
//!   CPU-vs-CPU, and **no conclusion about paged decode may be drawn.** A
//!   control arm silently running the treatment is the failure mode this
//!   harness exists to avoid.
//! * **If paged is faster than contiguous:** suspect the harness before
//!   believing it.
//!
//! # Result, 2026-08-01 (DEBUG build, RTX 4070 Laptop 8 GB, TinyLlama-1.1B f32, k=1)
//!
//! ```text
//!                   PAGED           PAGED      CONTIGUOUS   CONTIGUOUS
//!                  Replan        PlanOnce           plain     captured
//!                 (DEFAULT)                       (DEFAULT)
//!   ms/token       8,192.0           275.8         5,901.0        26.47
//!   HtoD copies      9,992           4,304           3,290          491
//!   HtoD MB        202,627          79,422          70,617        9,015
//!   DtoH copies      6,335           3,446              26           26
//!   DtoH MB        101,691          40,088             216          216
//!   kernels         17,802          17,802          36,718        9,214
//! ```
//!
//! **BOTH SHIPPED DEFAULTS WERE THE WRONG ONE.** Fuel exposes a persistent seam on
//! each path — `PagedDecodePlan::PlanOnce` (`PagedDecodeSession.base_cache`) and
//! `forward_with_kv_context_captured` (caller-held `Option<DecodeSession>`) — and
//! both were off unless the consumer opted in. Turning them on:
//!
//! * paged **8,192 ms -> 275.8 ms** per token (**29.7x**)
//! * contiguous **5,901 ms -> 26.47 ms** per token (**223x**)
//!
//! **The answer to the question this harness was built for**, with both arms
//! correctly configured and differing only in paging:
//!
//! * **paged costs 10.4x contiguous** (275.8 ms vs 26.47 ms)
//! * **paged moves 186x the bytes back to host** (40,088 MB vs 216 MB) —
//!   2.5 GB per token against a 13.5 MB logits-readback floor
//! * paged runs **~2x the kernels** of captured contiguous, while doing the same
//!   model's work
//!
//! That DtoH gap **survives the plan flag**, which is what distinguishes it from
//! a planning artifact: if the round-trip were a `Replan` side effect it would
//! have converged toward the contiguous floor once persistence was on. It fell
//! 61% and stayed 186x away.
//!
//! # Three pathologies, and only one of them is paging
//!
//! 1. **Paged round-trip to host.** Survives every fix above. Fuel-side: no fused
//!    CUDA `Op::PagedAttn` kernel.
//! 2. **`Replan` as the paged default.** Consumer-side, one line. **Fixed
//!    upstream:** `PagedSessionScheduler::new` now sets
//!    `plan: PagedDecodePlan::PlanOnce` (`fuel-inference/src/multi_session.rs`).
//! 3. **Non-persistent contiguous decode.** Consumer-side, one seam. This was
//!    first attributed to a *loader/anchor* placement bug on the theory that
//!    weights sat on host and were copied per token. Wrong: threading
//!    `DecodeSession` cut HtoD 70,617 -> 9,015 MB (-87%) with no loader change.
//!    The arithmetic that *looked* like confirmation — 70,617 MB / 16 tokens
//!    = 4.4 GB/token, almost exactly one 4.1 GB model — is produced identically
//!    by both causes and discriminates neither.
//!
//! # Two instruments that pointed the wrong way
//!
//! **Wall clock:** 1.39x looks like a clean, quotable result. Both arms were
//! transfer-bound, so it measured PCIe volume, not paging.
//!
//! **`nvidia-smi`:** reported 0% utilization and 153 MiB against a 4.1 GB model,
//! from which the author concluded — and broadcast — that nothing ran on the GPU.
//! `nsys` showed ~17,800 kernel launches. Both readings were sampling artifacts:
//! utilization was 0% because the GPU *stalls on PCIe*, and residency read near-
//! empty because memory is allocated and freed per-operation. An instantaneous
//! sample of a thrashing system looks identical to an idle one.
//!
//! Only the instrument that watched the **bus** rather than the clock or the
//! gauge produced the answer.
//!
//! ---------------------------------------------------------------------------
//!
//! # 2026-08-05/06 rework: the batch-size sweep (Task 10) — RUN
//!
//! **`FUEL_BASELINE = 8771997eac86b2a786bebcdcadda16fdcf2d28dc`**
//! (`C:\Projects\fuel-lightbulb-port`), release build, RTX 4070 Laptop 8 GB,
//! TinyLlama-1.1B **f32**, `LB_MAX_NEW=16`.
//!
//! The previous attempt was blocked: `fuel-core` did not build under
//! `--features cuda` at the old `12df216b` baseline (three `E0061`s inside
//! `#[cfg(feature = "cuda")] forward_with_kv_context_captured`, introduced by
//! `3364e8a9` and invisible to a CPU CI matrix). Fixed upstream in `8771997e`.
//! Full diagnosis in `.superpowers/sdd/2026-08-05-engine-wiring/task-10-report.md`.
//!
//! ## Steady-state decode, ms/token (`--release`)
//!
//! ```text
//!                                          k=1                     k=2
//!                                  (n=4/5 runs, median)      (single run)
//!   paged        PlanOnce               151.1                    247.7
//!   paged        Replan               13305.2                      —
//!   contiguous   PlanOnce, capture off   97.5                    149.4
//!   contiguous   PlanOnce, capture on    53.2                     85.9
//!   contiguous   Replan                7023.0                      —
//!   k >= 4       BOTH ARMS: CUDA_ERROR_OUT_OF_MEMORY (see below)
//! ```
//!
//! ## The three results this was built to produce
//!
//! **1. CUDA-graph capture is worth ~2x, and it is pure launch overhead.**
//! Persistence held CONSTANT (`PlanOnce` both sides, `optimize_calls_steady == 0`
//! both sides), the only difference being the entry point:
//!
//! ```text
//!                       forward_decode_step   forward_with_kv_context_captured
//!   steady ms/token           97.5                        53.2
//!   HtoD MB / copies      9014.573 / 491              9014.573 / 491
//!   DtoH MB / copies       216.220 /  26               216.220 /  26
//!   kernel launches         36,718                       9,214
//! ```
//!
//! The byte counts are **identical to the digit**. Capture moves not one extra
//! or fewer byte; it collapses 36,718 kernel launches into 9,214 (-75%). Four
//! paired runs give 2.32x / 2.75x / 1.81x / 1.53x — median **2.06x**, and the
//! sign never flips. This is the number Fuel was holding the default-on decision
//! against, and it is now isolated rather than bounded.
//!
//! Persistence and capture turn out to be **orthogonal seams**: persistence cuts
//! HtoD bytes (70,617 -> 9,015 MB, from the 2026-08-01 table) and leaves launches
//! at 36,718; capture cuts launches and leaves bytes untouched.
//!
//! **2. The release k=1 paged penalty is ~2.6x, NOT 10.4x.** The 10.4x came from
//! a **debug** build (the prior session's `build_gpu_harness.bat` had no
//! `--release`) and is not a valid reference for a release sweep. Paired,
//! same-block ratios at k=1: paged / captured-contiguous = 4.43 / 2.59 / 2.09
//! (median **2.59x**); paged / uncaptured-contiguous = 1.61 / 1.43 / 1.37
//! (median **1.43x** — a much tighter estimate, because both sides then pay the
//! same launch overhead). **No inversion point exists in the measurable range:**
//! the ratio does not fall from k=1 to k=2 (2.59 -> 2.88 against captured
//! contiguous), and k >= 4 does not run.
//!
//! **3. The host round-trip does NOT amortize across a batch.** DtoH totals,
//! from `nsys --trace=cuda --stats=true`:
//!
//! ```text
//!                       k=1 (16 tok, 23 fwd)   k=2 (32 tok, 49 fwd)
//!   paged  DtoH MB            40,021.974            93,198.532
//!          per token           2,501.4                2,912.5
//!          per forward         1,740.1                1,902.0
//!   contig DtoH MB               216.220               218.268
//!          per token              13.51                  6.82
//! ```
//!
//! Paged DtoH is **flat per forward step** and therefore scales linearly with k:
//! doubling the batch doubles the round-trip. Contiguous's DtoH is a fixed
//! ~216 MB plus a 0.128 MB logits readback per token — it is already free.
//!
//! **This slope does not discriminate kernel shapes.** An earlier framing said a
//! flat-vs-scaling DtoH slope chose between a monolithic paged CUDA kernel and
//! small primitives; Fuel retracted that, and it is wrong: *any* GPU
//! implementation removes the host round-trip equally. The slope characterises
//! the CURRENT placement, and what it says is that batching will not rescue it.
//!
//! ## k >= 4 does not run on this card, on either path
//!
//! ```text
//!   contiguous k=4   CUDA_ERROR_OUT_OF_MEMORY  Node#20 Copy [11534336] F32 (44 MiB, an FFN matrix)
//!   paged      k=4   CUDA_ERROR_OUT_OF_MEMORY  Node#20 Copy [11534336] F32   (after 592 s)
//!   contiguous k=8   host abort: "memory allocation of 46137344 bytes failed" (the same 44 MiB)
//!   contiguous k=16  CUDA_ERROR_OUT_OF_MEMORY  Node#24 Copy [4194304] F32 (16 MiB)
//! ```
//!
//! The registered prediction was that per-session weight residency would kill
//! k=2 on an 8 GB card. It survives k=2 and dies at k=4 — so residency does
//! scale with k, less steeply than k x 4.4 GB. **Largest k that completed: 2.**
//!
//! ## `build_batched_decode_logits` rejects a realistic ragged batch — confirmed
//!
//! ```text
//!   BATCHED_ADMISSION k=2 REJECTED cached_lens=[23, 26]
//!     err=build_batched_decode_logits: non-uniform caches (cached_len/max_seq_len)
//! ```
//!
//! Every cache was allocated at the SAME capacity on purpose, so this isolates
//! *ragged positions* as the cause. Contiguous "batching" at k>1 is therefore k
//! serial decodes — which is exactly what the timings above measured, and what a
//! server would actually get. No uniform batch was synthesized to obtain a
//! prettier number.
//!
//! ## `LB_ARM=paged_sched` validity control: PASSES
//!
//! Run A (paged first): paged 78.5, scheduler 130.3. Run B (scheduler first):
//! scheduler 113.6, paged 157.3. **The sign of the difference flips with run
//! order**, so run position dominates the arm effect; medians agree within 3%
//! (117.9 vs 122.0) and both report identical witnesses (`optimize_calls` 9
//! total / 0 steady, `realize_count` 14). The direct drive measures the shipped
//! path.
//!
//! That same ordering effect is why every ratio above is quoted **paired within
//! a block of consecutive runs**: the absolute paged k=1 number ranges 78.5 to
//! 173.9 ms/token across five runs of identical configuration, while the paired
//! paged/uncaptured-contiguous ratio stays inside 1.37-1.61.
//!
//! ## (a) `LB_PLAN` no longer has a default
//!
//! It used to map `""` to `Replan`. Fuel flipped the driver default to
//! `PlanOnce`, so a bare run measured the **opposite** of what ships while
//! looking current. An arm that does not name its mode is not a control, so
//! unset is now a panic.
//!
//! ## (b) Three timing windows, not two — and the boundary is load-bearing
//!
//! **[verified in Fuel, in code, 2026-08-05]** The decode plan is built on the
//! **first decode token**, on both routes — never during prefill:
//!
//! * Paged: `fuel-inference/src/multi_session.rs` `prefill_pass` calls
//!   `forward_paged_step` (the non-persistent, re-planning entry) once per
//!   prompt token and builds no session; `decode_one` calls
//!   `forward_paged_step_persistent`, which is where the plan is built.
//! * Contiguous: `forward_with_kv_context_persistent` given the whole prompt
//!   sees `seq != 1`, drops any session, and falls back to the rebuild path
//!   *without* building one. The build lands on the first decode token here too.
//!
//! So the first decode token carries a one-time cost that no later token pays.
//! Folding it into a steady-state mean charges `PlanOnce` a one-time cost as if
//! it recurred, and at 16 tokens that is enough to move the ratio. Reported
//! separately: `prefill_ms | first_decode_ms | steady_ms_per_token`.
//!
//! **Token 0 (the token sampled from the prefill logits) sits INSIDE the prefill
//! window.** The `first_decode` window is the first token produced by a
//! `forward_*_persistent` call — i.e. the first token that can build a plan.
//! `steady` is every decode token after that one.
//!
//! ## Why the paged arm drives the model directly instead of `PagedSessionScheduler`
//!
//! `PagedSessionScheduler::step()` is `prefill_pass()` followed immediately by
//! `decode_one()` over the now-Decode-ready set, so **one `step()` call spans
//! prefill AND the first decode token**. The three-window split is unobtainable
//! through it. This harness therefore issues the scheduler's own two calls
//! itself, in the scheduler's own order:
//!
//! ```text
//!   prefill:      for tok in prompt { model.forward_paged_step(tok, pool, h) }  -> sample
//!   decode token: model.forward_paged_step_persistent(last, pool, h, cap, plan, &mut sess) -> sample
//! ```
//!
//! with `max_blocks_cap = ceil((prompt.len() + max_new) / block_size).max(1)`,
//! copied from `add_session`. `LB_ARM=paged_sched` runs the real scheduler at
//! the same `k` as a validity control: if the direct drive's steady ms/token
//! disagrees with the scheduler's, the direct drive is not measuring the
//! shipped path and nothing else here should be believed.
//!
//! ## (c) A k-sweep
//!
//! `LB_BATCH` sets k concurrent sequences. Prompts are **ragged on purpose**:
//! session *i* gets `PROMPT_LEN + i * RAGGED_STRIDE` tokens. A uniform batch is
//! not a workload anybody runs, and — see (d) — uniformity is exactly the
//! precondition Fuel's batched contiguous entry requires, so synthesizing it
//! would produce a number that describes nothing.
//!
//! ## (d) Capture isolation — correcting this harness's own headline
//!
//! The 5,901 -> 26.47 ms/token figure above toggled persistence **and** CUDA-graph
//! capture together: the fast arm called `forward_with_kv_context_captured`, the
//! slow arm `forward_with_kv_context`. 223x bounds each and measures neither.
//! Three contiguous configurations now exist, and persistence is held constant
//! across the last two:
//!
//! | `LB_PLAN` | `LB_CAPTURE` | entry point                         |
//! | --------- | ------------ | ----------------------------------- |
//! | `replan`  | `off`        | `forward_with_kv_context`           |
//! | `once`    | `off`        | `forward_decode_step`               |
//! | `once`    | `on`         | `forward_with_kv_context_captured`  |
//!
//! The last two differ only in capture. Their delta is capture's actual
//! contribution — the number Fuel is holding a default-on decision against.
//! `LB_CAPTURE` has no default either, for the same reason `LB_PLAN` no longer
//! does.
//!
//! ## Registered before the k>1 runs — and how each came out
//!
//! Both registered predictions were **confirmed in substance**; the first was
//! wrong about the exact k (it said 2, the card died at 4). Resolutions are in
//! the results section above; the predictions are kept verbatim below so the
//! record shows what was claimed before the numbers existed.
//!
//! * **Per-session weight residency is the OOM risk.** `load_llama_f32_from_dir`
//!   builds `WeightStorage::F32(Arc<[f32]>)` — **host** memory. Every realize
//!   ships the weights across PCIe (that is the 4.4 GB/token HtoD above). A
//!   persistent session holds its realized `base_cache`, so if that cache holds
//!   a *per-session* device copy of the weights, k sessions need k x 4.4 GB and
//!   an 8 GB card dies at **k = 2**. If instead the device buffers are shared
//!   Arcs, VRAM stays flat in k. This harness does not know which; the run does.
//!   Either outcome is a reportable finding about whether the plan-once seam
//!   scales to concurrent sequences at all.
//! * **`build_batched_decode_logits` should reject a ragged batch.** It requires
//!   `k >= 2` and uniform `cached_len`/`max_seq_len`/dtype across caches. Every
//!   cache here is allocated at the SAME capacity, so a rejection isolates
//!   *ragged positions* as the cause rather than mismatched capacity. The probe
//!   runs LAST, after all timing, so a surprise success cannot perturb the
//!   measurement.
//! * **`forward_decode_step` holds ONE session in the `InferenceContext`.**
//!   k concurrent sequences therefore need k contexts, against the "one
//!   `InferenceContext` per model" invariant Fuel's own docs state. All three
//!   contiguous configurations use one context per sequence so the contrast
//!   between them stays on capture/persistence and not on context count.
//!
//! ## Evidence that the persistent path actually ran
//!
//! "The field says `PlanOnce`" and "the persistent path ran" are different
//! claims. Two independent witnesses are printed at every point:
//!
//! * `fuel::pipelined_bridge::optimize_calls_thread_local()` delta — the number
//!   of `optimize_graph` invocations. Plan reuse means this stops growing.
//! * paged only: `PagedDecodeSession::realize_count()` per session — how many
//!   times the held graph was **rebound**. A positive count is direct evidence.
//!
//! Both arms run in ONE process off ONE checkpoint load when `LB_ARM` is unset,
//! so timing differences are not load variance. Under `nsys`, one arm per
//! process, because transfer attribution needs the isolation more than the
//! timing needs the shared load.
//!
//! Run (all GPU runs go through the machine-wide mutex):
//!
//! ```text
//! $env:LB_PLAN='once'; $env:LB_CAPTURE='off'; $env:LB_BATCH='4'
//! pwsh C:\Projects\fuel-crash-vmm\scripts\gpu-run.ps1 -Project lightbulb -- \
//!   cargo test --release --features fuel-cuda --test gpu_paged_vs_contiguous -- --ignored --nocapture
//! ```

#![cfg(feature = "fuel-cuda")]

use std::path::PathBuf;
use std::time::{Duration, Instant};

// `KvCache` lives in `inference_context`, NOT `lazy` — `lazy` re-exports plenty
// of neighbouring types, which makes the wrong path look plausible.
use fuel::inference_context::{
    InferenceContext, KvCache, PagedDecodePlan, PagedDecodeSession,
};
use fuel::kv_block_pool::KvGeometry;
use fuel::kv_block_pool_device::DeviceKvPool;
use fuel::lazy::{SamplingStrategy, sample_logits};
use fuel::pipelined_bridge::optimize_calls_thread_local;
use fuel::{DType, Device};
use fuel_inference::multi_session::{KvBudget, PagedSessionScheduler};
use lightbulb::model_fuel::loader_f32::load_llama_f32_from_dir;

/// Sequence 0's prompt length. Kept at 8 so sequence 0 is byte-identical to the
/// single sequence the 2026-08-01 numbers were taken on.
const PROMPT_LEN: usize = 8;
/// Each additional sequence's prompt grows by this much, making the batch
/// ragged. Uniform batches are not a workload; see the module docs, (c).
const RAGGED_STRIDE: usize = 3;
const DEFAULT_MAX_NEW: usize = 16;
const BLOCK_SIZE: usize = 16;

fn tinyllama_dir() -> Option<PathBuf> {
    let p = PathBuf::from(
        "C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/\
         snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    );
    p.join("config.json").is_file().then_some(p)
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

/// Sequence `i`'s prompt. `i == 0` reproduces the original single-sequence
/// prompt exactly; later sequences are longer AND differ in content, so no two
/// sessions share a prefix that a cache could accidentally serve.
fn prompt_tokens(i: usize) -> Vec<u32> {
    let len = PROMPT_LEN + i * RAGGED_STRIDE;
    (0..len).map(|t| (((t + i * 7) * 17) % 1000 + 1) as u32).collect()
}

fn max_new() -> usize {
    match std::env::var("LB_MAX_NEW") {
        Ok(s) if !s.is_empty() => s.parse().expect("LB_MAX_NEW must be a positive integer"),
        _ => DEFAULT_MAX_NEW,
    }
}

fn batch_k() -> usize {
    match std::env::var("LB_BATCH") {
        Ok(s) if !s.is_empty() => {
            let k: usize = s.parse().expect("LB_BATCH must be a positive integer");
            assert!(k >= 1, "LB_BATCH must be >= 1");
            k
        }
        _ => 1,
    }
}

/// `LB_PLAN`, with **no default**.
///
/// It used to map `""` to `Replan`. Fuel's driver default is now `PlanOnce`
/// (`PagedSessionScheduler::new`), so mapping unset to `Replan` would silently
/// measure a configuration that no longer ships — and a stale number that looks
/// current is worse than a wrong one. An arm that does not name its mode is not
/// a control.
fn plan_from_env() -> PagedDecodePlan {
    match std::env::var("LB_PLAN").unwrap_or_default().as_str() {
        "once" => PagedDecodePlan::PlanOnce,
        "replan" => PagedDecodePlan::Replan,
        "" => panic!(
            "LB_PLAN must be set explicitly to \"once\" or \"replan\"; an arm that does not \
             name its mode is not a control"
        ),
        other => panic!("LB_PLAN must be \"once\" or \"replan\"; got {other:?}"),
    }
}

/// `LB_CAPTURE`, contiguous arm only, **no default** — same reasoning as
/// `LB_PLAN`. `on` is only meaningful with `LB_PLAN=once`: the non-persistent
/// entry has no capture variant, so `replan` + `on` names a configuration that
/// does not exist and is rejected rather than silently downgraded.
fn capture_from_env(plan: PagedDecodePlan) -> bool {
    let capture = match std::env::var("LB_CAPTURE").unwrap_or_default().as_str() {
        "on" => true,
        "off" => false,
        "" => panic!(
            "LB_CAPTURE must be set explicitly to \"on\" or \"off\" for the contiguous arm; \
             the 2026-08-01 headline toggled persistence and capture together and so measured \
             neither"
        ),
        other => panic!("LB_CAPTURE must be \"on\" or \"off\"; got {other:?}"),
    };
    assert!(
        !(capture && matches!(plan, PagedDecodePlan::Replan)),
        "LB_CAPTURE=on requires LB_PLAN=once: forward_with_kv_context has no capture variant, \
         so replan+capture is not a configuration Fuel offers"
    );
    capture
}

/// Three timing windows plus the two witnesses that the persistent path ran.
///
/// `steady` holds one entry per decode ROUND after the first; each round
/// advances all `k` sequences by one token, so a round costs `k` tokens.
struct Windows {
    k: usize,
    /// Every sequence's prompt, plus the token sampled from its prefill logits.
    prefill: Duration,
    /// One token for each of the `k` sequences, through the persistent entry.
    /// This is where the plan is built.
    first_decode: Duration,
    steady: Vec<Duration>,
    /// `optimize_graph` invocations over the whole run (this thread).
    optimize_calls: usize,
    /// `optimize_graph` invocations attributable to the steady window alone.
    /// Under `PlanOnce` this should be 0 — the plan is reused, not rebuilt.
    optimize_calls_steady: usize,
    /// Paged only: per-session held-graph rebind counts. Empty for contiguous
    /// (`DecodeSession` exposes no equivalent; `optimize_calls` is the witness
    /// there).
    realize_counts: Vec<usize>,
}

impl Windows {
    fn steady_total(&self) -> Duration {
        self.steady.iter().copied().sum()
    }
    fn steady_ms_per_token(&self) -> f64 {
        let tokens = self.steady.len() * self.k;
        if tokens == 0 {
            return f64::NAN;
        }
        self.steady_total().as_secs_f64() * 1000.0 / tokens as f64
    }
    fn first_decode_ms_per_token(&self) -> f64 {
        self.first_decode.as_secs_f64() * 1000.0 / self.k as f64
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Paged arm, driven through the model API in `PagedSessionScheduler`'s own call
/// order (see the module docs for why the scheduler itself cannot be used here).
fn run_paged(
    loaded: &lightbulb::model_fuel::loader::LoadedLlama,
    dev: &Device,
    k: usize,
    n_new: usize,
    plan: PagedDecodePlan,
) -> fuel::Result<Windows> {
    let cfg = &loaded.config;
    let prompts: Vec<Vec<u32>> = (0..k).map(prompt_tokens).collect();
    let caps: Vec<usize> = prompts
        .iter()
        .map(|p| (p.len() + n_new).div_ceil(BLOCK_SIZE).max(1))
        .collect();
    // Surplus so admission is never the thing being measured, but proportional
    // to k rather than a fixed 16 — a fixed pool cannot hold k=16 sessions and
    // the OOM would be the harness's, not the system's.
    let num_blocks = caps.iter().sum::<usize>() * 2 + 8;

    let mut pool = DeviceKvPool::new(
        KvGeometry {
            n_layers: cfg.n_layers,
            num_blocks,
            block_size: BLOCK_SIZE,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            elem_size: DType::F32.size_in_bytes(),
        },
        DType::F32,
        dev,
    )?;
    eprintln!(
        "paged arm: k={k} plan={plan:?} num_blocks={num_blocks} block_size={BLOCK_SIZE} \
         prompt_lens={:?}",
        prompts.iter().map(|p| p.len()).collect::<Vec<_>>()
    );

    let handles: Vec<_> = (0..k).map(|_| pool.core_mut().open()).collect();
    let mut sessions: Vec<Option<PagedDecodeSession>> = (0..k).map(|_| None).collect();
    let mut last_tok: Vec<u32> = vec![0; k];
    let mut rng: Vec<u64> = vec![0; k];

    let opt0 = optimize_calls_thread_local();

    // --- Window 1: prefill. Mirrors `prefill_pass`: whole prompt one token at a
    // time through the NON-persistent entry, then sample. Token 0 is sampled
    // here, so it sits inside this window by construction.
    let t = Instant::now();
    for i in 0..k {
        let mut logits = Vec::new();
        for &tok in &prompts[i] {
            logits = loaded.model.forward_paged_step(tok, &mut pool, handles[i])?;
        }
        last_tok[i] = sample_logits(&logits, SamplingStrategy::Greedy, &mut rng[i]);
    }
    let prefill = t.elapsed();

    // --- Window 2: the first decode token. Mirrors `decode_one`. This is where
    // `PlanOnce` builds the held plan; no later token pays it.
    let t = Instant::now();
    for i in 0..k {
        let logits = loaded.model.forward_paged_step_persistent(
            last_tok[i],
            &mut pool,
            handles[i],
            caps[i],
            plan,
            &mut sessions[i],
        )?;
        last_tok[i] = sample_logits(&logits, SamplingStrategy::Greedy, &mut rng[i]);
    }
    let first_decode = t.elapsed();
    let opt_after_first = optimize_calls_thread_local();

    // --- Window 3: steady state. `n_new` counts sampled tokens: token 0 came
    // from prefill, token 1 from the window above, so `n_new - 2` rounds remain.
    let rounds = n_new.saturating_sub(2);
    let mut steady = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t = Instant::now();
        for i in 0..k {
            let logits = loaded.model.forward_paged_step_persistent(
                last_tok[i],
                &mut pool,
                handles[i],
                caps[i],
                plan,
                &mut sessions[i],
            )?;
            last_tok[i] = sample_logits(&logits, SamplingStrategy::Greedy, &mut rng[i]);
        }
        steady.push(t.elapsed());
    }

    let opt1 = optimize_calls_thread_local();
    Ok(Windows {
        k,
        prefill,
        first_decode,
        steady,
        optimize_calls: opt1 - opt0,
        optimize_calls_steady: opt1 - opt_after_first,
        realize_counts: sessions
            .iter()
            .map(|s| s.as_ref().map_or(0, PagedDecodeSession::realize_count))
            .collect(),
    })
}

/// Validity control for `run_paged`: the SHIPPED driver at the same `k`.
///
/// `step()` fuses prefill and the first decode token, so this reports two
/// windows, not three — that is the whole reason `run_paged` exists. What it is
/// for is the steady number: if the direct drive's steady ms/token disagrees
/// with this, the direct drive is not measuring the shipped path.
fn run_paged_scheduler(
    loaded: &lightbulb::model_fuel::loader::LoadedLlama,
    dev: &Device,
    k: usize,
    n_new: usize,
    plan: PagedDecodePlan,
) -> fuel::Result<Windows> {
    let prompts: Vec<Vec<u32>> = (0..k).map(prompt_tokens).collect();
    let num_blocks = prompts
        .iter()
        .map(|p| (p.len() + n_new).div_ceil(BLOCK_SIZE).max(1))
        .sum::<usize>()
        * 2
        + 8;
    let mut sched = PagedSessionScheduler::new(
        &loaded.model,
        KvBudget { block_size: BLOCK_SIZE, num_blocks },
        DType::F32,
        dev,
    )?;
    sched.set_plan(plan);
    assert_eq!(sched.plan(), plan, "set_plan did not take");
    let ids: Vec<_> = prompts
        .iter()
        .map(|p| sched.add_session(p, SamplingStrategy::Greedy, None, n_new))
        .collect::<fuel::Result<Vec<_>>>()?;

    let opt0 = optimize_calls_thread_local();
    // Step 0 = prefill + token 0 + the first decode token, all in one call.
    let t = Instant::now();
    let _ = sched.step();
    let prefill_and_first = t.elapsed();
    let opt_after_first = optimize_calls_thread_local();

    let rounds = n_new.saturating_sub(2);
    let mut steady = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t = Instant::now();
        let _ = sched.step();
        steady.push(t.elapsed());
    }
    let opt1 = optimize_calls_thread_local();
    let realize_counts = ids
        .iter()
        .map(|&id| sched.session_realize_count(id).unwrap_or(0))
        .collect();
    Ok(Windows {
        k,
        prefill: prefill_and_first,
        first_decode: Duration::ZERO,
        steady,
        optimize_calls: opt1 - opt0,
        optimize_calls_steady: opt1 - opt_after_first,
        realize_counts,
    })
}

/// Contiguous arm: `KvCache` + one of three entry points (see the module docs,
/// (d)). Greedy sampling matches the paged arm's `SamplingStrategy::Greedy`, so
/// both arms do the same amount of work per token.
///
/// One `InferenceContext` PER SEQUENCE, in all three configurations.
/// `forward_decode_step` stores its `DecodeSession` in the context, so a shared
/// context would hand sequence *i*'s held plan — bound to sequence *i*'s KV
/// Arcs — to sequence *i+1*. Using one context per sequence everywhere keeps
/// the only difference between the three configurations the entry point itself.
fn run_contiguous(
    loaded: &lightbulb::model_fuel::loader::LoadedLlama,
    dev: &Device,
    k: usize,
    n_new: usize,
    plan: PagedDecodePlan,
    capture: bool,
) -> fuel::Result<Windows> {
    let cfg = &loaded.config;
    let prompts: Vec<Vec<u32>> = (0..k).map(prompt_tokens).collect();
    // ONE capacity for every cache, deliberately: the batched-admission probe at
    // the end must reject on RAGGED POSITIONS, not on mismatched capacity, or it
    // proves nothing about ragged workloads.
    let cap = prompts.iter().map(Vec::len).max().unwrap_or(PROMPT_LEN) + n_new + 1;
    let persistent = matches!(plan, PagedDecodePlan::PlanOnce);
    eprintln!(
        "contiguous arm: k={k} persistent={persistent} capture={capture} cache_cap={cap} \
         prompt_lens={:?}",
        prompts.iter().map(Vec::len).collect::<Vec<_>>()
    );

    let mut caches: Vec<KvCache> = (0..k)
        .map(|_| {
            KvCache::with_capacity(
                cfg.n_layers,
                cfg.n_kv_heads,
                cfg.head_dim,
                cap,
                DType::F32,
                dev,
            )
        })
        .collect::<fuel::Result<Vec<_>>>()?;
    let mut ctxs: Vec<InferenceContext> =
        (0..k).map(|_| InferenceContext::new(dev.clone())).collect();
    // Only used by the capture configuration; `None` infers
    // `Option<CapturedDecodeSession>` from the signature, which is the only way
    // to name it from here (`fuel-core` re-exports `fuel_dispatch::topology`
    // alone).
    let mut captured: Vec<_> = (0..k).map(|_| None).collect();
    let mut sessions: Vec<_> = (0..k).map(|_| None).collect();
    let mut last_tok: Vec<u32> = vec![0; k];
    let mut rng: Vec<u64> = vec![0; k];

    let opt0 = optimize_calls_thread_local();

    // --- Window 1: prefill. The whole prompt in one call. Every configuration
    // takes the SAME path here: `seq != 1` makes the persistent and captured
    // entries drop any session and fall back to the rebuild path without
    // building a plan.
    let t = Instant::now();
    for i in 0..k {
        let logits =
            loaded.model.forward_with_kv_context(&prompts[i], &mut caches[i], &mut ctxs[i])?;
        last_tok[i] = sample_logits(&logits, SamplingStrategy::Greedy, &mut rng[i]);
    }
    let prefill = t.elapsed();

    // Closure would need three simultaneous &mut borrows of the Vecs; a fn
    // taking indices keeps borrowck happy without cloning anything.
    macro_rules! decode_one {
        ($i:expr) => {{
            let i = $i;
            let toks = [last_tok[i]];
            let logits = if !persistent {
                loaded.model.forward_with_kv_context(&toks, &mut caches[i], &mut ctxs[i])?
            } else if capture {
                loaded.model.forward_with_kv_context_captured(
                    &toks,
                    &mut caches[i],
                    &mut ctxs[i],
                    &mut sessions[i],
                    &mut captured[i],
                )?
            } else {
                loaded.model.forward_decode_step(&toks, &mut caches[i], &mut ctxs[i])?
            };
            last_tok[i] = sample_logits(&logits, SamplingStrategy::Greedy, &mut rng[i]);
        }};
    }

    // --- Window 2: the first decode token — where the plan is built.
    let t = Instant::now();
    for i in 0..k {
        decode_one!(i);
    }
    let first_decode = t.elapsed();
    let opt_after_first = optimize_calls_thread_local();

    // --- Window 3: steady state.
    let rounds = n_new.saturating_sub(2);
    let mut steady = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let t = Instant::now();
        for i in 0..k {
            decode_one!(i);
        }
        steady.push(t.elapsed());
    }
    let opt1 = optimize_calls_thread_local();

    // --- Admission probe, AFTER all timing so a surprise success cannot
    // perturb it. `build_batched_decode_logits` is the ONLY contiguous entry
    // that batches; if it will not admit a ragged batch then contiguous
    // "batching" at k>1 is k serial decodes, which is what the timing above
    // measured and what a server would actually get.
    if k >= 2 {
        let mut refs: Vec<&mut KvCache> = caches.iter_mut().collect();
        let lens: Vec<usize> = refs.iter().map(|c| c.cached_len).collect();
        match loaded.model.build_batched_decode_logits(&mut refs, &last_tok, dev, DType::F32) {
            Ok(rows) => eprintln!(
                "BATCHED_ADMISSION k={k} ACCEPTED cached_lens={lens:?} rows={}",
                rows.len()
            ),
            Err(e) => eprintln!("BATCHED_ADMISSION k={k} REJECTED cached_lens={lens:?} err={e}"),
        }
    } else {
        eprintln!(
            "BATCHED_ADMISSION k=1 NOT_APPLICABLE (build_batched_decode_logits requires k >= 2)"
        );
    }

    Ok(Windows {
        k,
        prefill,
        first_decode,
        steady,
        optimize_calls: opt1 - opt0,
        optimize_calls_steady: opt1 - opt_after_first,
        realize_counts: Vec::new(),
    })
}

/// One machine-readable line per point, plus the human view. The `RESULT` line
/// is what the results table is built from — grep it out of the run log rather
/// than re-typing numbers.
fn summarise(label: &str, cfg: &str, w: &Windows) {
    eprintln!("\n=== {label} ===");
    eprintln!("  prefill      {:>10.2} ms   ({} sequences, token 0 sampled inside)", ms(w.prefill), w.k);
    eprintln!(
        "  first decode {:>10.2} ms   ({:.2} ms/token over {} sequences) <- plan built here",
        ms(w.first_decode),
        w.first_decode_ms_per_token(),
        w.k
    );
    eprintln!(
        "  steady       {:>10.2} ms   over {} rounds x {} sequences = {} tokens",
        ms(w.steady_total()),
        w.steady.len(),
        w.k,
        w.steady.len() * w.k
    );
    eprintln!("  steady ms/token {:.3}", w.steady_ms_per_token());
    if !w.steady.is_empty() {
        eprintln!("  steady round median {:?}", median(w.steady.clone()));
    }
    eprintln!(
        "  optimize_calls total={} steady_window={} (PlanOnce reuse => steady_window == 0)",
        w.optimize_calls, w.optimize_calls_steady
    );
    if !w.realize_counts.is_empty() {
        eprintln!("  realize_count per session {:?}", w.realize_counts);
    }
    eprintln!(
        "RESULT {cfg} k={} prefill_ms={:.3} first_decode_ms={:.3} first_decode_ms_per_tok={:.3} \
         steady_ms_per_tok={:.3} steady_rounds={} optimize_calls={} optimize_calls_steady={} \
         realize={:?}",
        w.k,
        ms(w.prefill),
        ms(w.first_decode),
        w.first_decode_ms_per_token(),
        w.steady_ms_per_token(),
        w.steady.len(),
        w.optimize_calls,
        w.optimize_calls_steady,
        w.realize_counts
    );
}

#[test]
#[ignore = "needs a CUDA device and the TinyLlama checkpoint"]
fn gpu_paged_vs_contiguous_per_token() {
    // Deliberately NOT an early `return` on missing prerequisites: an early
    // return from a #[test] is a PASS, and a measurement harness that reports
    // success having measured nothing is the exact artifact this whole
    // investigation is trying to stop producing.
    let dir = tinyllama_dir()
        .expect("no TinyLlama snapshot — this harness measures, so it fails rather than skipping");

    let cuda = fuel_cuda_backend::CudaDevice::new(0)
        .expect("no CUDA device — this harness measures GPU placement, so absence is a failure");
    let dev: Device = cuda.into();

    let loaded = load_llama_f32_from_dir(&dir).expect("load the f32 checkpoint");

    let k = batch_k();
    let n_new = max_new();
    assert!(n_new >= 3, "LB_MAX_NEW must be >= 3 to leave a non-empty steady window");
    let plan = plan_from_env();
    // `LB_ARM` selects a single arm so `nsys` can attribute CUDA transfers to
    // one code path. Unset runs both in one process off one load, which is the
    // right shape for the timing A/B but the wrong shape for transfer counting.
    let arm = std::env::var("LB_ARM").unwrap_or_default();
    eprintln!("CONFIG arm={arm:?} plan={plan:?} k={k} max_new={n_new}");

    match arm.as_str() {
        "paged" => {
            let w = run_paged(&loaded, &dev, k, n_new, plan).expect("paged arm");
            summarise("PAGED (CUDA), isolated for nsys", &format!("arm=paged plan={plan:?}"), &w);
        }
        "paged_sched" => {
            let w =
                run_paged_scheduler(&loaded, &dev, k, n_new, plan).expect("paged scheduler arm");
            summarise(
                "PAGED via PagedSessionScheduler (validity control; two windows, not three)",
                &format!("arm=paged_sched plan={plan:?}"),
                &w,
            );
        }
        "contiguous" => {
            let capture = capture_from_env(plan);
            let w = run_contiguous(&loaded, &dev, k, n_new, plan, capture)
                .expect("contiguous arm");
            summarise(
                "CONTIGUOUS (CUDA), isolated for nsys",
                &format!("arm=contiguous plan={plan:?} capture={capture}"),
                &w,
            );
        }
        "" => {
            let capture = capture_from_env(plan);
            let paged = run_paged(&loaded, &dev, k, n_new, plan).expect("paged arm");
            let contig = run_contiguous(&loaded, &dev, k, n_new, plan, capture)
                .expect("contiguous arm");
            summarise("PAGED (CUDA)", &format!("arm=paged plan={plan:?}"), &paged);
            summarise(
                "CONTIGUOUS (CUDA)",
                &format!("arm=contiguous plan={plan:?} capture={capture}"),
                &contig,
            );

            let p = paged.steady_ms_per_token();
            let c = contig.steady_ms_per_token();
            eprintln!("\n=== steady-state per-token decode, GPU, k={k} ===");
            eprintln!("  paged      {p:.3} ms/token");
            eprintln!("  contiguous {c:.3} ms/token");
            eprintln!("  ratio      {:.3}x", p / c);
            eprintln!("RATIO k={k} plan={plan:?} capture={capture} paged_over_contiguous={:.4}", p / c);
            eprintln!(
                "\nNOTE: this ratio is NOT the answer on its own. It cannot distinguish\n\
                 \"no round-trip\" from \"neither arm reached the GPU at all\". Read it only\n\
                 alongside the nsys kernel-launch and memcpy counts."
            );
        }
        other => panic!(
            "LB_ARM must be \"paged\", \"paged_sched\", \"contiguous\", or unset; got {other:?}"
        ),
    }
}
