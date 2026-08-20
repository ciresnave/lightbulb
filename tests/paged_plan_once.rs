//! **Plan-once paged decode: the A/B.**
//!
//! Fuel landed plan-once paged decode at `7ed43541`, flag-gated and default off
//! (`PagedDecodePlan::Replan`). This measures whether it does what it claims.
//!
//! # What is being tested, and why the arms are shaped this way
//!
//! Earlier measurement found paged decode costing **5.6x** the contiguous path,
//! and the warm-up/steady split identified the cause: the paged path had **no
//! first-step premium at any batch size** (0.99/0.99/1.01 at B=1/2/4), meaning
//! it re-planned the graph on every token, while the contiguous path paid its
//! plan once and replayed (5.22x premium). Plan-once is the fix for exactly
//! that.
//!
//! Both arms run **at the same commit**, flipping only the flag — not
//! parent-vs-commit across a checkout, which would fold in every unrelated
//! change between the two (ragged batched decode, the never-panic sweeps,
//! vulkane 0.9.0). Same binary, same process, same checkpoint load.
//!
//! # The instrument
//!
//! Per-token wall clock, split into **warm-up** (the first decode token, where
//! a plan-once path builds its plan) and **steady** (the rest, where it should
//! replay). The ratio is the signal:
//!
//! * `Replan`   — warm-up ≈ steady. Every token pays the planner.
//! * `PlanOnce` — warm-up ≫ steady. Step one pays; the rest replay.
//!
//! Plan and execute costs are reported as a **difference of two direct
//! measurements** (warm-up minus steady), never as a residual after subtracting
//! an assumed value. A previous analysis defined `c(B) = step(B) - B*0.95s` and
//! then "confirmed" it by observing that `c(B)/B + 0.95s` reproduced the data —
//! which is `step(B)/B` identically, an algebraic tautology that would have
//! reproduced any dataset under any assumption.
//!
//! # `session_realize_count` is a WIRING witness, not a PERF witness
//!
//! It answers "did the plan-once path execute?" and says nothing about whether
//! replay is cheaper. It exists here only to disambiguate a null result, per the
//! table agreed with Fuel:
//!
//! | arm | count | ratio | reading |
//! |---|---|---|---|
//! | `PlanOnce` | `0`/`None` | flat | the flag did not wire or take — a Fuel-side bug, **not** a perf finding |
//! | `PlanOnce` | `> 0` | flat | the flag took and replay does not beat re-planning here — a **real negative** result |
//! | `Replan` | `None` | — | the control holds no plan, so it is a true control |
//!
//! Without it a 1.0 ratio is ambiguous between "mis-wired" and "genuinely
//! doesn't help", and picking the flattering reading is how a null result gets
//! debugged out of existence.
//!
//! Run: `cargo test --release --test paged_plan_once -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::{Duration, Instant};

use fuel::inference_context::PagedDecodePlan;
use fuel::lazy::SamplingStrategy;
use fuel::{DType, Device};
use fuel_inference::multi_session::{KvBudget, PagedSessionScheduler};
use lightbulb::model_fuel::loader_f32::load_llama_f32_from_dir;

const PROMPT_LEN: usize = 8;
const MAX_NEW: usize = 7;
const BLOCK_SIZE: usize = 16;

/// Locate the TinyLlama checkpoint.
///
/// `TINYLLAMA_DIR` overrides the built-in default, which is one developer's
/// HuggingFace cache including a specific snapshot hash. A measurement harness
/// that only runs on one machine cannot be reproduced by whoever is asked to
/// believe its numbers.
fn tinyllama_dir() -> Option<PathBuf> {
    let p = match std::env::var_os("TINYLLAMA_DIR") {
        Some(v) => PathBuf::from(v),
        None => PathBuf::from(
            "C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/\
             snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        ),
    };
    p.join("config.json").is_file().then_some(p)
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

/// One arm: build a scheduler, set the flag, decode `MAX_NEW` tokens, and time
/// every step.
///
/// Returns `(warm_up, steady_median, realize_count, total_tokens, all_steps)`.
///
/// `all_steps` is every step's wall clock **including step 0**, and it is
/// printed rather than only summarised. That is not decoration: the first run of
/// this harness reported a flat warm-up/steady ratio and I nearly read it as
/// "plan-once does not help", when in fact the plan is built during step 0 —
/// which the summary statistics deliberately exclude. A verdict that reports
/// only its conclusion cannot show you that it was aimed at the wrong window;
/// one that reports what it actually saw can.
fn run_arm(
    loaded: &lightbulb::model_fuel::loader::LoadedLlama,
    plan: PagedDecodePlan,
) -> (Duration, Duration, Option<usize>, usize, Vec<Duration>) {
    let budget = KvBudget {
        block_size: BLOCK_SIZE,
        // prompt + generated fits one block at this geometry; the surplus keeps
        // admission from being the thing under measurement.
        num_blocks: 8,
    };
    let mut sched = PagedSessionScheduler::new(&loaded.model, budget, DType::F32, &Device::cpu())
        .expect("build the paged scheduler");
    sched.set_plan(plan);
    assert_eq!(sched.plan(), plan, "set_plan did not take");

    let prompt: Vec<u32> = (0..PROMPT_LEN)
        .map(|t| ((t * 17) % 1000 + 1) as u32)
        .collect();
    let id = sched
        .add_session(&prompt, SamplingStrategy::Greedy, None, MAX_NEW)
        .expect("admit the session");

    // Step 0 is prefill + the first sampled token, so it is not a decode step
    // and is excluded from both statistics. Step 1 is the first DECODE token —
    // the one a plan-once path pays its build on. Steps 2.. are replay.
    let mut warm_up = Duration::ZERO;
    let mut steady: Vec<Duration> = Vec::new();
    let mut all_steps: Vec<Duration> = Vec::with_capacity(MAX_NEW);
    for i in 0..MAX_NEW {
        let t0 = Instant::now();
        let _report = sched.step();
        let dt = t0.elapsed();
        all_steps.push(dt);
        match i {
            0 => {}            // prefill + first token
            1 => warm_up = dt, // first decode token
            _ => steady.push(dt),
        }
    }

    let count = sched.session_realize_count(id);
    let tokens = sched
        .run_to_completion()
        .into_iter()
        .find(|(sid, _)| *sid == id)
        .map(|(_, toks)| toks.len())
        .unwrap_or(0);

    assert!(
        !steady.is_empty(),
        "no steady-state samples — MAX_NEW is too small"
    );
    (warm_up, median(steady), count, tokens, all_steps)
}

#[test]
#[ignore = "needs the TinyLlama checkpoint; minutes in release"]
fn plan_once_vs_replan() {
    // Deliberately NOT the `return` skip other suites use: an early return from
    // a #[test] is a PASS, so a missing checkpoint would report success having
    // measured nothing. This is a measurement harness; absent data is a failure.
    let dir = tinyllama_dir()
        .expect("no TinyLlama snapshot — this harness measures, so it fails rather than skipping");
    let loaded = load_llama_f32_from_dir(&dir).expect("load the f32 checkpoint");

    // Both arms, one process, one checkpoint load, same commit.
    let (replan_warm, replan_steady, replan_count, replan_tokens, replan_steps) =
        run_arm(&loaded, PagedDecodePlan::Replan);
    let (once_warm, once_steady, once_count, once_tokens, once_steps) =
        run_arm(&loaded, PagedDecodePlan::PlanOnce);

    let ratio = |w: Duration, s: Duration| w.as_secs_f64() / s.as_secs_f64();
    let replan_ratio = ratio(replan_warm, replan_steady);
    let once_ratio = ratio(once_warm, once_steady);

    // Every step, both arms, before any summary. Step 0 is prefill + the first
    // sampled token and is excluded from the statistics below — printing it
    // anyway is what makes a mis-aimed measurement window visible instead of
    // silently producing a confident wrong verdict.
    eprintln!();
    eprintln!("  per-step wall clock (step 0 = prefill + first token, excluded from stats):");
    eprintln!("    Replan   {replan_steps:.3?}");
    eprintln!("    PlanOnce {once_steps:.3?}");

    eprintln!();
    eprintln!("  arm       warm-up      steady      ratio   realize_count");
    eprintln!("  --------  -----------  ----------  ------  -------------");
    eprintln!(
        "  Replan    {replan_warm:>11.3?}  {replan_steady:>10.3?}  {replan_ratio:>6.2}  {replan_count:?}"
    );
    eprintln!(
        "  PlanOnce  {once_warm:>11.3?}  {once_steady:>10.3?}  {once_ratio:>6.2}  {once_count:?}"
    );
    eprintln!();
    let speedup = replan_steady.as_secs_f64() / once_steady.as_secs_f64();
    eprintln!(
        "  steady-state per-token: {replan_steady:.3?} -> {once_steady:.3?}  ({speedup:.2}x)"
    );
    // Plan cost as a DIFFERENCE of two direct measurements, not a residual.
    if once_warm > once_steady {
        eprintln!(
            "  one-off plan build (warm-up minus steady): {:.3?}",
            once_warm - once_steady
        );
    }

    // Non-vacuity: both arms actually decoded.
    let want = PROMPT_LEN + MAX_NEW;
    assert_eq!(
        replan_tokens, want,
        "Replan arm did not produce {want} tokens"
    );
    assert_eq!(
        once_tokens, want,
        "PlanOnce arm did not produce {want} tokens"
    );

    // The control must be a true control: Replan holds no plan.
    assert!(
        replan_count.is_none() || replan_count == Some(0),
        "the Replan arm reports a realize count of {replan_count:?} — it is holding a plan, so it \
         is not a control and the A/B compares plan-once against itself"
    );

    // Wiring witness. This does NOT establish a speedup; it establishes which
    // question a flat ratio would put us in.
    assert!(
        matches!(once_count, Some(n) if n > 0),
        "the PlanOnce arm reports realize_count {once_count:?} — the flag did not wire or did not \
         take, so any timing below is not a measurement of plan-once. This is a Fuel-side bug, \
         not a performance finding"
    );
}
