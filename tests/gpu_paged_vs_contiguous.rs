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
//! # Result, 2026-08-01 (debug build, RTX 4070 Laptop 8 GB, TinyLlama-1.1B f32)
//!
//! ```text
//!                    PAGED      CONTIGUOUS     ratio
//!   DtoH copies      6,335             26      244x
//!   DtoH bytes  101,691 MB        216 MB       470x
//!   HtoD copies      9,992          3,290      3.0x
//!   HtoD bytes  202,627 MB     70,617 MB       2.9x
//!   kernels         17,802         36,718      0.48x
//!   per-token       8.192 s        5.901 s     1.39x
//! ```
//!
//! **Two independent pathologies, which the wall clock blended into one
//! uninformative 1.39x ratio:**
//!
//! 1. **Paged-specific round-trip.** 6.2 GB pulled back to the host *per token*
//!    versus 13.5 MB contiguous — and contiguous's 13.5 MB is just the logits
//!    readback, the irreducible floor. The paged arm also runs **less than half**
//!    the GPU kernels; a path doing the same attention on-device would run more,
//!    not fewer. Work is leaving the device.
//! 2. **Weight re-upload, BOTH arms.** 70-202 GB uploaded for a 4.1 GB model over
//!    16 tokens, with thousands of matched `cuMemAlloc_v2`/`cuMemFree_v2` pairs.
//!    Weights never stay resident. Separate bug, separate fix.
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
//! empty because memory is allocated and freed per-operation. A instantaneous
//! sample of a thrashing system looks identical to an idle one.
//!
//! Only the instrument that watched the **bus** rather than the clock or the
//! gauge produced the answer.
//!
//! Both arms run in ONE process off ONE checkpoint load when `LB_ARM` is unset,
//! so timing differences are not load variance. Under `nsys`, one arm per
//! process, because transfer attribution needs the isolation more than the
//! timing needs the shared load.
//!
//! Run: `cargo test --release --features fuel-cuda --test gpu_paged_vs_contiguous -- --ignored --nocapture`

#![cfg(feature = "fuel-cuda")]

use std::path::PathBuf;
use std::time::{Duration, Instant};

// `KvCache` lives in `inference_context`, NOT `lazy` — `lazy` re-exports plenty
// of neighbouring types, which makes the wrong path look plausible.
use fuel::inference_context::{InferenceContext, KvCache};
use fuel::lazy::SamplingStrategy;
use fuel::{DType, Device};
use fuel_inference::multi_session::{KvBudget, PagedSessionScheduler};
use lightbulb::model_fuel::loader_f32::load_llama_f32_from_dir;

const PROMPT_LEN: usize = 8;
const MAX_NEW: usize = 16;
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

fn prompt_tokens() -> Vec<u32> {
    (0..PROMPT_LEN).map(|t| ((t * 17) % 1000 + 1) as u32).collect()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("logits contain NaN"))
        .expect("empty logits")
        .0 as u32
}

/// Paged arm: Fuel's `PagedSessionScheduler` driven on a CUDA device.
///
/// Returns every step's wall clock including step 0. Step 0 is prefill plus the
/// first sampled token and is reported but excluded from the steady statistic —
/// a previous harness summarised away the very step that carried the effect and
/// nearly reported a 5.55x win as a null result.
fn run_paged(
    loaded: &lightbulb::model_fuel::loader::LoadedLlama,
    dev: &Device,
) -> Vec<Duration> {
    let budget = KvBudget {
        block_size: BLOCK_SIZE,
        // Surplus so admission is never the thing being measured.
        num_blocks: 16,
    };
    let mut sched = PagedSessionScheduler::new(&loaded.model, budget, DType::F32, dev)
        .expect("build the paged scheduler on CUDA");

    let prompt = prompt_tokens();
    sched
        .add_session(&prompt, SamplingStrategy::Greedy, None, MAX_NEW)
        .expect("admit the session");

    let mut steps = Vec::with_capacity(MAX_NEW);
    for _ in 0..MAX_NEW {
        let t0 = Instant::now();
        let _ = sched.step();
        steps.push(t0.elapsed());
    }
    steps
}

/// Contiguous arm: `KvCache` + `forward_with_kv_context` on the same device.
///
/// Greedy argmax feeds the next token, matching the paged arm's
/// `SamplingStrategy::Greedy`, so both arms do the same amount of work per step.
fn run_contiguous(
    loaded: &lightbulb::model_fuel::loader::LoadedLlama,
    dev: &Device,
) -> Vec<Duration> {
    let cfg = &loaded.config;
    let mut cache = KvCache::with_capacity(
        cfg.n_layers,
        cfg.n_kv_heads,
        cfg.head_dim,
        PROMPT_LEN + MAX_NEW + 1,
        DType::F32,
        dev,
    )
    .expect("f32 KvCache on CUDA");
    let mut ctx = InferenceContext::new(dev.clone());

    let prompt = prompt_tokens();
    let mut steps = Vec::with_capacity(MAX_NEW);

    // Step 0 mirrors the paged arm: prefill plus the first sampled token.
    let t0 = Instant::now();
    let logits = loaded
        .model
        .forward_with_kv_context(&prompt, &mut cache, &mut ctx)
        .expect("contiguous prefill on CUDA");
    let mut next = argmax(&logits);
    steps.push(t0.elapsed());

    for _ in 1..MAX_NEW {
        let t = Instant::now();
        let logits = loaded
            .model
            .forward_with_kv_context(&[next], &mut cache, &mut ctx)
            .expect("contiguous decode on CUDA");
        next = argmax(&logits);
        steps.push(t.elapsed());
    }
    steps
}

fn summarise(label: &str, steps: &[Duration]) {
    eprintln!("\n=== {label} ===");
    for (i, d) in steps.iter().enumerate() {
        let tag = match i {
            0 => " (prefill + first token)",
            1 => " (first decode token)",
            _ => "",
        };
        eprintln!("  step {i:>2}: {d:?}{tag}");
    }
    let decode: Vec<Duration> = steps.iter().skip(1).copied().collect();
    if !decode.is_empty() {
        eprintln!("  decode median: {:?}  over {} steps", median(decode.clone()), decode.len());
    }
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

    // `LB_ARM` selects a single arm so `nsys` can attribute CUDA transfers to
    // one code path. Unset runs both in one process off one load, which is the
    // right shape for the timing A/B but the wrong shape for transfer counting.
    let arm = std::env::var("LB_ARM").unwrap_or_default();

    match arm.as_str() {
        "paged" => summarise("PAGED (CUDA), isolated for nsys", &run_paged(&loaded, &dev)),
        "contiguous" => {
            summarise("CONTIGUOUS (CUDA), isolated for nsys", &run_contiguous(&loaded, &dev))
        }
        "" => {
            let paged = run_paged(&loaded, &dev);
            let contig = run_contiguous(&loaded, &dev);
            summarise("PAGED (CUDA)", &paged);
            summarise("CONTIGUOUS (CUDA)", &contig);

            let p = median(paged.iter().skip(1).copied().collect());
            let c = median(contig.iter().skip(1).copied().collect());
            eprintln!("\n=== per-token decode, GPU ===");
            eprintln!("  paged      {p:?}");
            eprintln!("  contiguous {c:?}");
            eprintln!("  ratio      {:.3}x", p.as_secs_f64() / c.as_secs_f64());
            eprintln!(
                "\nNOTE: this ratio is NOT the answer on its own. It cannot distinguish\n\
                 \"no round-trip\" from \"neither arm reached the GPU\". Read it only\n\
                 alongside the nsys kernel-launch and memcpy counts."
            );
        }
        other => panic!("LB_ARM must be \"paged\", \"contiguous\", or unset; got {other:?}"),
    }
}
