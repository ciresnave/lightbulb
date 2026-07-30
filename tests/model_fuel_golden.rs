//! # Tier 3: golden-vector regression fixtures for `model_fuel`.
//!
//! ## READ THIS FIRST — what tier 3 is, and what it is not
//!
//! > **Tier 3 is a REGRESSION NET, NOT A VERDICT. These numbers are Lightbulb's
//! > own past output, not an independent reference. A mismatch means "Lightbulb's
//! > output moved" — it does NOT mean the new output is wrong, and a match does
//! > NOT mean the output is right. Both directions are settled by tiers 1/2
//! > (`tests/kissref_differential.rs`), never here.**
//!
//! Tier 3 shares every bug the implementation has, *by construction*: the fixture
//! was produced by running the very code it is now used to check. If a projection
//! is transposed wrongly, the golden faithfully records the wrong answer and will
//! defend it forever. That is the price of a fixed point captured from yourself,
//! and it is why the disclaimer above appears in three places that are hard to
//! miss: this doc, the `_README` field *inside* the JSON fixture (so it survives
//! being read out of context), and the text of every assertion failure.
//!
//! Tiers 1 and 2 (kiss-ref, an independent spec-exact implementation) answer
//! "is this right?". Tier 3 answers only "is this the same?".
//!
//! ## Status of the fixture, stated plainly
//!
//! At the time this file was written the fixture at
//! `tests/fixtures/model_fuel_golden/v1/tinyllama_f32_greedy.json`
//! **had not yet been generated**. The machinery is complete and its controls
//! pass; the golden run itself (`capture_golden_fixture`, `#[ignore]`) still
//! needs to be executed in release against the 2.2 GB checkpoint. Nothing in
//! this repository fabricates that file — an empty-but-correct mechanism is
//! worth more than an invented fixture, because an invented fixture is a lie
//! that passes.
//!
//! Everything below that does **not** need the checkpoint runs in ordinary CI in
//! milliseconds. That is deliberate: the expensive test is `#[ignore]`d, so the
//! only reason to trust the harness is that its comparison function is
//! exercised continuously by controls C1–C7 against a synthetic fixture built in
//! code.
//!
//! ## Why capture MUST use `load_llama_f32_from_dir`
//!
//! The stock loader (`load_llama_from_dir`) preserves **bf16 projections**, which
//! gives every matmul the key `[F32, BF16, F32]`. No CPU kernel serves that key,
//! so Fuel's optimizer inserts a promoting cast — 155 of them per realize on
//! TinyLlama-1.1B, measured by Fuel. That cast is *value-lossless but not
//! accumulation-preserving*: the promoted op accumulates in higher precision, so
//! the same graph produces different numbers depending on which arm ran.
//!
//! A golden captured on that path is **arm-dependent**, and an arm-dependent
//! golden is worthless as a fixed point: it would drift whenever Fuel's dispatch
//! changed its mind, and every such drift would look exactly like a Lightbulb
//! regression. The all-f32 loader keeps the key at `[F32, F32, F32]`, which is
//! natively served, so no fixup pass runs and the numbers are a property of the
//! kernel rather than of the dispatcher's choice.
//!
//! `provenance.loader` records the loader name and the check refuses to compare
//! against a fixture captured through anything else.
//!
//! ## What is captured, and why three layers rather than one
//!
//! `forward_with_kv_context_persistent` returns logits already sliced to the last
//! position, so one decode step is exactly `vocab_size` f32 — 32 000 for
//! TinyLlama, 128 KB raw. Storing all of it per step is unreviewable; storing
//! only the token id is nearly blind. So the fixture is layered, and the layers
//! are not redundant — read together they *triage* the failure:
//!
//! | layer | content | catches |
//! | --- | --- | --- |
//! | **L1** | token ids per step (+ decoded text) | the behavioural contract |
//! | **L2** | probe logit values, with a tolerance | material numeric movement |
//! | **L3** | sha256 of the full logits bit pattern | drift at or below tolerance |
//!
//! Token ids alone were rejected for two independent reasons that pull in
//! opposite directions. `argmax` is invariant to any drift smaller than the
//! top1/top2 margin, so a 1e-3 numerics regression is *invisible*; and `argmax`
//! breaks ties with a strict `>` (lowest index wins), so at a near-tie a 1-ULP
//! drift *flips the token* and the test fails loudly for a non-bug. Coarse and
//! spuriously flaky at once is the worst of both.
//!
//! The L2 probe set is fixed in code so it can never be re-picked to make a
//! capture pass: the argmax index, the runner-up index, both their values, and a
//! prime-stride sample across the vocabulary ([`PROBE_STRIDE`] = 997, giving 33
//! indices on a 32 000-token vocab). Per-step summary statistics (max, min, sum,
//! non-finite count) are accumulated in **f64** — a 32 000-term f32 sum carries
//! ~2e-5 relative error of its own, which is looser than the L2 tolerance and
//! would make the summary the least trustworthy number in the file.
//!
//! The single most valuable number in the fixture is `margin_abs` /`margin_rel`:
//! it converts "did this token flip because of a bug or because it was a coin
//! flip?" from an argument into a recorded fact.
//!
//! Every captured float is stored **twice**: `bits` (the exact u32 pattern, as
//! hex) and `approx` (a shortest-round-trip decimal string, for humans). The
//! comparison reads `bits` only. Decimal-only storage is the classic way a golden
//! silently loses a ULP and can then never assert bit-exactness again; control C5
//! pins the round trip.
//!
//! ## Determinism, and the honest scope of the bit-exact layer
//!
//! Source evidence says the default CPU f32 path *should* be run-to-run
//! bit-reproducible on a fixed binary and machine: `matmul_f32_capacity`
//! (`fuel-cpu-backend/src/byte_kernels.rs:4214`) is a strictly serial `i/kk/j`
//! nest with ascending-order accumulation and no rayon anywhere in that file, and
//! `optimize` runs once per `DecodeSession` so fusion and arm choices are fixed
//! for the rest of the loop.
//!
//! **That is a reading of the source, not a measurement.** The measurement is
//! [`determinism_probe`], and it is a *gate*: L3 (bit-exactness) may only be
//! believed once that probe has been run twice — once within a process and once
//! across processes — and reported zero drift.
//!
//! Against cross-machine reproducibility there is one identified mechanism:
//! `softmax_last_dim_f32` calls `.exp()`, i.e. the system libm, which is not
//! bit-identical across platforms or necessarily across patch versions. So L3 is
//! a **separate, explicitly named test** ([`golden_is_bit_exact`]) rather than a
//! warning tier inside the main check — a warning nobody reads is worse than no
//! check, whereas a test that is expected to fail off the capture machine and
//! says so in its own failure message is honest.
//!
//! ## The tolerance, and where its number comes from
//!
//! Recorded in the fixture rather than hidden in the code, so it is auditable
//! instead of folklore. See [`TOLERANCE_DERIVATION`] and [`derive_tolerance`]:
//!
//! ```text
//! tol_rel = clamp(measured_run_to_run_max_rel_delta * 100,
//!                 lower = 4 ULP relative (4.76837158203125e-7),
//!                 upper = min_recorded_margin_rel / 10)
//! tol_abs = 1e-6
//! ```
//!
//! The lower bound is not invented: 4 ULP is the §6.8 ceiling kiss-ref declares
//! for `Exp` (`Op::Exp.ulp_ceiling() == Some(4.0)`, already pinned by
//! `tests/kissref_differential.rs:372`), and `exp` is the transcendental this
//! path actually goes through in softmax. The upper bound exists so the tolerance
//! can never be loose enough to mask a drift that could flip a token; control C7
//! re-checks it on every run. Control C6 recomputes the whole rule from the
//! numbers stored in the fixture, so a silent loosening is caught.
//!
//! ## Regenerating (blessing) the fixture
//!
//! The failure mode being designed against is a golden that rots into a rubber
//! stamp because re-blessing is one reflex command away. Four mechanisms:
//!
//! 1. Capture is a **separate `#[ignore]`d test**, never the assertion itself.
//! 2. Overwriting an existing fixture needs **two keys**: `LIGHTBULB_BLESS_GOLDEN=1`
//!    *and* a non-empty `LIGHTBULB_BLESS_REASON`. First-time creation needs
//!    neither — bootstrapping is not a re-blessing.
//! 3. The reason is written into `provenance.reason`, so the git diff shows a
//!    human sentence explaining why the numbers moved, immediately next to the
//!    numbers that moved.
//! 4. Full provenance (commits, rustc, profile, host, checkpoint hash, loader)
//!    makes a stale fixture self-identifying, and the check reports
//!    "FIXTURE STALE / WRONG INPUTS" rather than "regression" when the inputs
//!    disagree.
//!
//! ```text
//! # first capture (no fixture on disk yet)
//! cargo test --release --test model_fuel_golden -- --ignored --nocapture capture_golden_fixture
//!
//! # re-blessing an existing fixture (both keys required)
//! $env:LIGHTBULB_BLESS_GOLDEN = "1"
//! $env:LIGHTBULB_BLESS_REASON = "RoPE base fixed in generate.rs; tiers 1/2 confirm the new values"
//! cargo test --release --test model_fuel_golden -- --ignored --nocapture capture_golden_fixture
//! ```
//!
//! ## Honest limitations
//!
//! - Shares all implementation bugs. Not a correctness verdict. Ever.
//! - Valid only for the **default, CPU-only feature set**. Verified with
//!   `cargo tree -e features -i fuel-cpu-backend` / `-i fuel-core`: both resolve
//!   with feature `default` only, and both crates declare `default = []`, so
//!   mkl / accelerate / aocl / onemkl / cuda / telemetry / jit are all off.
//!   Enabling any BLAS feature swaps the matmul kernel outright and invalidates
//!   every golden; a future agent adding one for speed would see the failure and
//!   might read it as a regression. [`default_feature_set_assumptions_hold`]
//!   pins the part that is detectable from inside Lightbulb.
//! - L3 (bit-exact) is expected to fail on different hardware or a different
//!   toolchain, and says so in its own message.
//! - Greedy, single sequence, one checkpoint. Says nothing about batching,
//!   sampling, contracts, or GQA-less models.
//! - `load_llama_f32_from_dir` has a tied-`lm_head` branch
//!   (`src/model_fuel/loader_f32.rs:112-125`) that TinyLlama does **not**
//!   exercise, so these goldens cannot regress-test it.
//! - Format precedent is Fuel's `fuel-correctness-fixtures` crate, deliberately
//!   *not* reused: its `CorrectnessFixture` carries a multi-backend-consensus
//!   provenance claim that tier 3 cannot make, so the types here are named
//!   differently to keep the distinction visible.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ─────────────────────────────────────────────────────────────────────────────
// The disclaimer, as data. It goes *inside* the fixture so a reader who opens
// the JSON without ever seeing this file still gets told what it is.
// ─────────────────────────────────────────────────────────────────────────────

const TIER3_DISCLAIMER: &str = "Tier 3 is a REGRESSION NET, NOT A VERDICT. These numbers are Lightbulb's \
own past output, not an independent reference. A mismatch means 'Lightbulb's output moved' — it does NOT \
mean the new output is wrong, and a match does NOT mean the output is right. Both directions are settled \
by tiers 1/2 (tests/kissref_differential.rs), never here.";

const FORMAT_VERSION: u32 = 1;

/// The loader the fixture must have been captured through. See the module doc.
const REQUIRED_LOADER: &str = "load_llama_f32_from_dir";

/// Prime stride over the vocabulary. Fixed in code, never hand-picked per
/// capture: a probe set chosen after seeing the values is a probe set chosen to
/// pass.
const PROBE_STRIDE: usize = 997;

/// 4 ULP relative for f32. Grounded in kiss-ref's declared §6.8 ceiling for
/// `Exp` (`Op::Exp.ulp_ceiling() == Some(4.0)`), which is the transcendental the
/// softmax on this path actually goes through — not a round number picked
/// because it felt safe.
const F32_4ULP_REL: f64 = 4.0 * (f32::EPSILON as f64); // 4.76837158203125e-7

/// Absolute floor, so a near-zero logit does not blow up the relative
/// denominator and manufacture a failure out of nothing.
const TOL_ABS: f64 = 1e-6;

const TOLERANCE_DERIVATION: &str = "tol_rel = clamp(measured_run_to_run_max_rel_delta * 100, \
lower = 4 ULP relative (4.0 * f32::EPSILON = 4.76837158203125e-7, grounded in kiss-ref's declared \
Op::Exp ULP ceiling of 4.0), upper = min_recorded_margin_rel / 10); tol_abs = 1e-6. The lower bound \
keeps libm patch-version variation from causing false alarms; the upper bound keeps the tolerance from \
ever being loose enough to mask a drift that could flip a token.";

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("model_fuel_golden")
        .join("v1")
        .join("tinyllama_f32_greedy.json")
}

fn tinyllama_dir() -> Option<PathBuf> {
    let p = PathBuf::from(
        "C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    );
    p.join("model.safetensors").is_file().then_some(p)
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema
//
// Names deliberately distinct from Fuel's CorrectnessFixture / FixtureFile /
// ToleranceBand: the shape is borrowed, the provenance claim is not.
// ─────────────────────────────────────────────────────────────────────────────

/// A single f32, stored twice.
///
/// `bits` is the exact IEEE-754 pattern and is **the only field the comparison
/// reads**. `approx` is a shortest-round-trip decimal *string* for human review;
/// it is a string rather than a JSON number because JSON cannot represent NaN or
/// infinity, and control C5 covers exactly those cases.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct F32Val {
    bits: String,
    approx: String,
}

impl F32Val {
    fn from_f32(v: f32) -> Self {
        F32Val {
            bits: format!("0x{:08x}", v.to_bits()),
            // `{:?}` on f32 is shortest-round-trip, and renders NaN / inf / -inf
            // as words rather than exploding.
            approx: format!("{v:?}"),
        }
    }

    fn to_f32(&self) -> f32 {
        let hex = self.bits.strip_prefix("0x").unwrap_or(&self.bits);
        let bits = u32::from_str_radix(hex, 16)
            .unwrap_or_else(|e| panic!("fixture float {:?} is not a hex u32: {e}", self.bits));
        f32::from_bits(bits)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Provenance {
    captured_utc: String,
    lightbulb_commit: String,
    fuel_worktree_commit: String,
    rustc: String,
    cargo_profile: String,
    host_os: String,
    host_arch: String,
    host_cpu: String,
    /// Not machine-detectable from inside Lightbulb; recorded as the assumption
    /// it is, and pinned externally by `cargo tree -e features` (see module doc).
    fuel_feature_assumption: String,
    lightbulb_features_enabled: Vec<String>,
    checkpoint_dir: String,
    config_json_sha256: String,
    model_safetensors_len: u64,
    loader: String,
    reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToleranceBlock {
    rel: f64,
    abs: f64,
    derivation: String,
    measured_run_to_run_max_rel_delta: f64,
    lower_bound_4ulp: f64,
    upper_bound_min_margin_over_10: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Stats {
    max: F32Val,
    min: F32Val,
    /// Accumulated in f64 on purpose — see the module doc.
    sum_f64: f64,
    nonfinite_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoldenStep {
    step: usize,
    logits_len: usize,
    argmax_index: u32,
    runner_up_index: u32,
    top1_value: F32Val,
    top2_value: F32Val,
    /// Derived (f64) from `top1_value`/`top2_value`; present so a reviewer can
    /// see at a glance how close this step was to flipping.
    margin_abs: f64,
    margin_rel: f64,
    /// Set at capture time when `margin_rel <= 10 * tol_rel`. Such a step is a
    /// coin flip, and a golden that asserts on it without saying so is
    /// presenting a coin flip as a regression signal.
    #[serde(default)]
    unstable: bool,
    probe_stride: usize,
    probe_indices: Vec<usize>,
    probe_values: Vec<F32Val>,
    stats: Stats,
    logits_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoldenCase {
    name: String,
    prompt: String,
    prompt_token_ids: Vec<u32>,
    max_new: usize,
    eos: Option<u32>,
    generated_token_ids: Vec<u32>,
    generated_text: String,
    steps: Vec<GoldenStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoldenFile {
    #[serde(rename = "_README")]
    readme: String,
    format_version: u32,
    provenance: Provenance,
    tolerance: ToleranceBlock,
    cases: Vec<GoldenCase>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Drift reporting
//
// The split between "the fixture is invalid" and "the output changed" is the
// distinction Fuel's own `CorrectnessDrift` draws, and it matters: they call for
// completely different responses. One is fixed by re-capturing; the other must
// never be.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DriftClass {
    /// The fixture does not describe this run's inputs at all. Re-capture.
    FixtureInvalid,
    /// The inputs match and the output moved. Do NOT re-capture to make it green.
    OutputChanged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layer {
    L0Provenance,
    L1Tokens,
    L2Probes,
    L3Digest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DriftKind {
    CheckpointMismatch,
    LoaderMismatch,
    FormatVersionMismatch,
    PromptMismatch,
    LogitsLenMismatch,
    ProbeIndexMismatch,
    StepCountMismatch,
    TokenSequenceMismatch,
    TokenMismatch,
    TopValueOutOfTolerance,
    ProbeOutOfTolerance,
    DigestMismatch,
}

impl DriftKind {
    fn class(self) -> DriftClass {
        match self {
            DriftKind::CheckpointMismatch
            | DriftKind::LoaderMismatch
            | DriftKind::FormatVersionMismatch
            | DriftKind::PromptMismatch
            | DriftKind::LogitsLenMismatch
            | DriftKind::ProbeIndexMismatch => DriftClass::FixtureInvalid,
            DriftKind::StepCountMismatch
            | DriftKind::TokenSequenceMismatch
            | DriftKind::TokenMismatch
            | DriftKind::TopValueOutOfTolerance
            | DriftKind::ProbeOutOfTolerance
            | DriftKind::DigestMismatch => DriftClass::OutputChanged,
        }
    }

    fn layer(self) -> Layer {
        match self {
            DriftKind::CheckpointMismatch
            | DriftKind::LoaderMismatch
            | DriftKind::FormatVersionMismatch
            | DriftKind::PromptMismatch
            | DriftKind::LogitsLenMismatch
            | DriftKind::ProbeIndexMismatch => Layer::L0Provenance,
            DriftKind::StepCountMismatch
            | DriftKind::TokenSequenceMismatch
            | DriftKind::TokenMismatch => Layer::L1Tokens,
            DriftKind::TopValueOutOfTolerance | DriftKind::ProbeOutOfTolerance => Layer::L2Probes,
            DriftKind::DigestMismatch => Layer::L3Digest,
        }
    }
}

#[derive(Debug, Clone)]
struct GoldenDrift {
    kind: DriftKind,
    location: String,
    detail: String,
}

impl GoldenDrift {
    fn new(kind: DriftKind, location: impl Into<String>, detail: impl Into<String>) -> Self {
        GoldenDrift {
            kind,
            location: location.into(),
            detail: detail.into(),
        }
    }
}

const TRIAGE_TABLE: &str = "\
TRIAGE — read the layers together, they are not redundant:
  L1 differs                             -> behaviour moved. Check the RECORDED MARGIN for that step
                                            before assuming a bug: a near-tie flip and a real
                                            regression are different events with the same symptom.
  L1 same, L2 out of tolerance           -> numerics moved materially.
  L1 same, L2 in tolerance, L3 differs   -> drift at or below tolerance; bit-level only. Expected on
                                            different hardware or a different toolchain (libm exp).
  any L0                                 -> FIXTURE STALE / WRONG INPUTS, not a regression. The
                                            fixture does not describe this run. Re-capture.";

fn render_drifts(drifts: &[GoldenDrift]) -> String {
    let mut by_layer: BTreeMap<&str, Vec<&GoldenDrift>> = BTreeMap::new();
    for d in drifts {
        let key = match d.kind.layer() {
            Layer::L0Provenance => "L0 provenance (FIXTURE INVALID)",
            Layer::L1Tokens => "L1 tokens",
            Layer::L2Probes => "L2 probes",
            Layer::L3Digest => "L3 digest",
        };
        by_layer.entry(key).or_default().push(d);
    }

    let mut s = String::new();
    s.push_str(TIER3_DISCLAIMER);
    s.push_str("\n\n");
    for (layer, ds) in &by_layer {
        s.push_str(&format!("  [{layer}]  {} finding(s)\n", ds.len()));
        for d in ds.iter().take(12) {
            s.push_str(&format!(
                "    {:?} at {}: {}\n",
                d.kind, d.location, d.detail
            ));
        }
        if ds.len() > 12 {
            s.push_str(&format!("    ... and {} more\n", ds.len() - 12));
        }
    }
    s.push('\n');
    s.push_str(TRIAGE_TABLE);
    s.push_str(
        "\n\nDo NOT re-bless to make this green. Establish via tiers 1/2 \
         (tests/kissref_differential.rs) whether the new output is more or less correct, then bless \
         with a reason:\n  LIGHTBULB_BLESS_GOLDEN=1 LIGHTBULB_BLESS_REASON=\"...\" cargo test --release \
         --test model_fuel_golden -- --ignored capture_golden_fixture\n\nAlso consider, before \
         concluding 'regression': did the build's FEATURE SET change? Enabling mkl/aocl/onemkl/\
         accelerate swaps the matmul kernel outright and invalidates every golden.",
    );
    s
}

// ─────────────────────────────────────────────────────────────────────────────
// Derivation: logits -> GoldenStep. Pure, so the controls can drive it without
// the model.
// ─────────────────────────────────────────────────────────────────────────────

/// Reimplementation of `src/model_fuel/generate.rs::argmax`, which is private.
///
/// The tie-break must match exactly: strict `>` seeded with `NEG_INFINITY`, so
/// the **lowest** index wins a tie. [`harness_argmax_agrees_with_generate_greedy`]
/// is the control that proves this reimplementation has not drifted from the
/// shipped one.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Second-highest by the same rule, excluding `top`.
fn runner_up(logits: &[f32], top: usize) -> u32 {
    let mut best = usize::MAX;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if i == top {
            continue;
        }
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    if best == usize::MAX { top as u32 } else { best as u32 }
}

fn probe_indices(vocab: usize) -> Vec<usize> {
    (0..vocab).step_by(PROBE_STRIDE).collect()
}

/// sha256 over the logits' exact f32 patterns, **little-endian, in index
/// order**. Both of those are fixed here on purpose: a future platform or a
/// reordered serialization would otherwise produce a false regression that looks
/// identical to a real one. Control C4 covers the comparison; the endianness is
/// pinned by `to_le_bytes` rather than by `bytemuck`, which would inherit the
/// host's.
fn logits_digest(logits: &[f32]) -> String {
    let mut h = Sha256::new();
    for &v in logits {
        h.update(v.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn derive_step(step: usize, logits: &[f32]) -> GoldenStep {
    assert!(!logits.is_empty(), "step {step}: empty logits");

    let top1 = argmax(logits) as usize;
    let top2 = runner_up(logits, top1) as usize;
    let t1 = logits[top1];
    let t2 = logits[top2];

    let margin_abs = (t1 as f64) - (t2 as f64);
    let margin_rel = margin_abs / (t1 as f64).abs().max(1e-30);

    let idxs = probe_indices(logits.len());
    let probe_values: Vec<F32Val> = idxs.iter().map(|&i| F32Val::from_f32(logits[i])).collect();

    let mut max = f32::NEG_INFINITY;
    let mut min = f32::INFINITY;
    let mut sum = 0f64;
    let mut nonfinite = 0usize;
    for &v in logits {
        if !v.is_finite() {
            nonfinite += 1;
            continue;
        }
        if v > max {
            max = v;
        }
        if v < min {
            min = v;
        }
        sum += v as f64;
    }

    GoldenStep {
        step,
        logits_len: logits.len(),
        argmax_index: top1 as u32,
        runner_up_index: top2 as u32,
        top1_value: F32Val::from_f32(t1),
        top2_value: F32Val::from_f32(t2),
        margin_abs,
        margin_rel,
        unstable: false, // set by finalize_unstable once the tolerance is known
        probe_stride: PROBE_STRIDE,
        probe_indices: idxs,
        probe_values,
        stats: Stats {
            max: F32Val::from_f32(max),
            min: F32Val::from_f32(min),
            sum_f64: sum,
            nonfinite_count: nonfinite,
        },
        logits_sha256: logits_digest(logits),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tolerance derivation — the rule, executable, so C6 can recompute it.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
struct DerivedTolerance {
    rel: f64,
    abs: f64,
    lower_bound_4ulp: f64,
    upper_bound_min_margin_over_10: f64,
}

/// The rule from the module doc, as code.
///
/// Returns `Err` rather than clamping when the bounds cross — that is, when the
/// tightest margin in the fixture is so small that no tolerance can be both
/// above the libm noise floor and below margin/10. Silently clamping there would
/// produce a tolerance that is guaranteed to mask token flips, which is the one
/// outcome this rule exists to prevent. (`f64::clamp` would also panic on
/// `lo > hi`; failing with a sentence beats failing with a panic message about
/// argument ordering.)
fn derive_tolerance(
    measured_run_to_run_max_rel_delta: f64,
    min_margin_rel: f64,
) -> Result<DerivedTolerance, String> {
    let lower = F32_4ULP_REL;
    let upper = min_margin_rel / 10.0;
    if upper.is_nan() || upper < lower {
        return Err(format!(
            "no admissible tolerance: the tightest recorded margin_rel is {min_margin_rel:e}, so the \
             upper bound margin/10 = {upper:e} is below the 4-ULP floor {lower:e}. That step is a coin \
             flip; flag it as unstable rather than asserting on it."
        ));
    }
    let candidate = measured_run_to_run_max_rel_delta * 100.0;
    let rel = candidate.max(lower).min(upper);
    Ok(DerivedTolerance {
        rel,
        abs: TOL_ABS,
        lower_bound_4ulp: lower,
        upper_bound_min_margin_over_10: upper,
    })
}

/// C7's rule, as a function so it can be applied at capture time *and* on every
/// check run.
fn margin_guard_violations(tol: &ToleranceBlock, cases: &[GoldenCase]) -> Vec<String> {
    let mut out = Vec::new();
    for c in cases {
        for s in &c.steps {
            if s.margin_rel <= 10.0 * tol.rel {
                out.push(format!(
                    "case {:?} step {}: margin_rel {:e} <= 10 * tol_rel {:e} — this step is a near-tie \
                     and a 1-ULP drift can flip its token. It is a coin flip, not a regression signal.",
                    c.name,
                    s.step,
                    s.margin_rel,
                    10.0 * tol.rel
                ));
            }
        }
    }
    out
}

fn finalize_unstable(tol: &ToleranceBlock, cases: &mut [GoldenCase]) {
    for c in cases.iter_mut() {
        for s in c.steps.iter_mut() {
            s.unstable = s.margin_rel <= 10.0 * tol.rel;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The comparison. Pure over data — no model, no filesystem — which is precisely
// what lets controls C1..C7 run in ordinary CI while the golden run is
// `#[ignore]`d.
// ─────────────────────────────────────────────────────────────────────────────

/// `None` when within tolerance; `Some((abs, rel))` otherwise.
fn value_drift(expected: &F32Val, got: &F32Val, tol: &ToleranceBlock) -> Option<(f64, f64)> {
    let e = expected.to_f32();
    let g = got.to_f32();
    if e.to_bits() == g.to_bits() {
        return None;
    }
    if e.is_nan() && g.is_nan() {
        return None;
    }
    if !e.is_finite() || !g.is_finite() {
        return Some((f64::INFINITY, f64::INFINITY));
    }
    let (ef, gf) = (e as f64, g as f64);
    let abs = (ef - gf).abs();
    let rel = abs / ef.abs().max(f64::MIN_POSITIVE);
    if abs <= tol.abs || rel <= tol.rel {
        None
    } else {
        Some((abs, rel))
    }
}

/// Inputs-level check, separated so a stale fixture reports as stale rather than
/// as a regression.
fn compare_inputs(
    fixture: &GoldenFile,
    observed_config_sha: &str,
    observed_loader: &str,
) -> Vec<GoldenDrift> {
    let mut d = Vec::new();
    if fixture.format_version != FORMAT_VERSION {
        d.push(GoldenDrift::new(
            DriftKind::FormatVersionMismatch,
            "file",
            format!(
                "fixture format_version {} but this harness speaks {FORMAT_VERSION}",
                fixture.format_version
            ),
        ));
    }
    if fixture.provenance.loader != observed_loader {
        d.push(GoldenDrift::new(
            DriftKind::LoaderMismatch,
            "provenance.loader",
            format!(
                "fixture captured through {:?}, this run used {:?} — goldens are only meaningful on \
                 the all-f32 path (bf16 projections trigger promoting casts that change accumulation)",
                fixture.provenance.loader, observed_loader
            ),
        ));
    }
    if fixture.provenance.config_json_sha256 != observed_config_sha {
        d.push(GoldenDrift::new(
            DriftKind::CheckpointMismatch,
            "provenance.config_json_sha256",
            format!(
                "fixture {} vs this checkpoint {} — different model, so nothing below is comparable",
                fixture.provenance.config_json_sha256, observed_config_sha
            ),
        ));
    }
    d
}

/// The whole per-case comparison.
///
/// `strict_digest` gates L3. It is off for the default check because
/// bit-exactness is only claimed for (same binary, same machine, same feature
/// set); [`golden_is_bit_exact`] turns it on and names that scope in its own
/// failure text.
fn compare_case(
    tol: &ToleranceBlock,
    expected: &GoldenCase,
    observed: &GoldenCase,
    strict_digest: bool,
) -> Vec<GoldenDrift> {
    let mut d = Vec::new();
    let case = expected.name.clone();

    // ---- inputs ----
    if expected.prompt_token_ids != observed.prompt_token_ids
        || expected.max_new != observed.max_new
        || expected.eos != observed.eos
    {
        d.push(GoldenDrift::new(
            DriftKind::PromptMismatch,
            format!("case {case}"),
            format!(
                "fixture ran ids={:?} max_new={} eos={:?}; this run ran ids={:?} max_new={} eos={:?}",
                expected.prompt_token_ids,
                expected.max_new,
                expected.eos,
                observed.prompt_token_ids,
                observed.max_new,
                observed.eos
            ),
        ));
        // Different inputs make every downstream comparison meaningless.
        return d;
    }

    // ---- L1: tokens ----
    if expected.generated_token_ids.len() != observed.generated_token_ids.len() {
        d.push(GoldenDrift::new(
            DriftKind::TokenSequenceMismatch,
            format!("case {case}"),
            format!(
                "fixture generated {} tokens {:?} ({:?}); this run generated {} tokens {:?} ({:?})",
                expected.generated_token_ids.len(),
                expected.generated_token_ids,
                expected.generated_text,
                observed.generated_token_ids.len(),
                observed.generated_token_ids,
                observed.generated_text
            ),
        ));
    } else {
        for (i, (e, o)) in expected
            .generated_token_ids
            .iter()
            .zip(&observed.generated_token_ids)
            .enumerate()
        {
            if e != o {
                let margin = expected
                    .steps
                    .get(i)
                    .map(|s| {
                        format!(
                            "recorded margin_abs {:e} margin_rel {:e}{}",
                            s.margin_abs,
                            s.margin_rel,
                            if s.unstable {
                                " [FLAGGED UNSTABLE AT CAPTURE — this step was already a coin flip]"
                            } else {
                                ""
                            }
                        )
                    })
                    .unwrap_or_else(|| "no recorded margin".into());
                d.push(GoldenDrift::new(
                    DriftKind::TokenMismatch,
                    format!("case {case} step {i}"),
                    format!("fixture token {e}, this run {o}; {margin}"),
                ));
            }
        }
    }

    if expected.steps.len() != observed.steps.len() {
        d.push(GoldenDrift::new(
            DriftKind::StepCountMismatch,
            format!("case {case}"),
            format!(
                "fixture has {} steps, this run produced {}",
                expected.steps.len(),
                observed.steps.len()
            ),
        ));
    }

    // ---- L2 / L3, per step ----
    for (e, o) in expected.steps.iter().zip(&observed.steps) {
        let at = format!("case {case} step {}", e.step);

        if e.logits_len != o.logits_len {
            d.push(GoldenDrift::new(
                DriftKind::LogitsLenMismatch,
                at.clone(),
                format!(
                    "fixture logits_len {} vs {} — different vocabulary, nothing else compares",
                    e.logits_len, o.logits_len
                ),
            ));
            continue;
        }
        if e.probe_indices != o.probe_indices || e.probe_stride != o.probe_stride {
            d.push(GoldenDrift::new(
                DriftKind::ProbeIndexMismatch,
                at.clone(),
                format!(
                    "probe set moved (fixture stride {} / {} indices; this run {} / {}) — the harness \
                     changed, so the fixture no longer describes it",
                    e.probe_stride,
                    e.probe_indices.len(),
                    o.probe_stride,
                    o.probe_indices.len()
                ),
            ));
            continue;
        }

        for (label, ev, ov) in [
            ("top1_value", &e.top1_value, &o.top1_value),
            ("top2_value", &e.top2_value, &o.top2_value),
        ] {
            if let Some((abs, rel)) = value_drift(ev, ov, tol) {
                d.push(GoldenDrift::new(
                    DriftKind::TopValueOutOfTolerance,
                    format!("{at} {label}"),
                    format!(
                        "fixture {} ({}), this run {} ({}); abs {:e} rel {:e} vs tol abs {:e} rel {:e}",
                        ev.approx, ev.bits, ov.approx, ov.bits, abs, rel, tol.abs, tol.rel
                    ),
                ));
            }
        }

        for (k, (ev, ov)) in e.probe_values.iter().zip(&o.probe_values).enumerate() {
            if let Some((abs, rel)) = value_drift(ev, ov, tol) {
                d.push(GoldenDrift::new(
                    DriftKind::ProbeOutOfTolerance,
                    format!("{at} probe[{k}] (vocab index {})", e.probe_indices[k]),
                    format!(
                        "fixture {} ({}), this run {} ({}); abs {:e} rel {:e} vs tol abs {:e} rel {:e}",
                        ev.approx, ev.bits, ov.approx, ov.bits, abs, rel, tol.abs, tol.rel
                    ),
                ));
            }
        }

        if strict_digest && e.logits_sha256 != o.logits_sha256 {
            d.push(GoldenDrift::new(
                DriftKind::DigestMismatch,
                at.clone(),
                format!(
                    "fixture sha256 {} vs this run {} — bit-level drift. EXPECTED on different \
                     hardware or a different toolchain (softmax goes through the system libm's exp, \
                     which is not bit-identical across platforms). On its own this is NOT a regression.",
                    e.logits_sha256, o.logits_sha256
                ),
            ));
        }
    }

    d
}

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic fixture for the controls.
//
// Built in code, never committed as JSON, and loudly labelled: a hand-written
// JSON file sitting next to a real golden is an invitation to mistake one for
// the other. These numbers are arbitrary and mean nothing about the model.
// ─────────────────────────────────────────────────────────────────────────────

const SYNTHETIC_VOCAB: usize = 32_000;
const SYNTHETIC_TOP1_IDX: usize = 1234;
const SYNTHETIC_TOP2_IDX: usize = 5678;
/// A probe index (997 * 1) whose value is set large enough that a 1-ULP nudge
/// exceeds `TOL_ABS`. C3 asserts that, so it provably exercises the *relative*
/// tolerance rather than sliding through on the absolute floor.
const SYNTHETIC_BIG_PROBE_IDX: usize = 997;

/// Deterministic pseudo-logits. An LCG, not `rand`, so this is reproducible
/// forever without pinning a crate's RNG algorithm.
fn synthetic_logits(seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut v = Vec::with_capacity(SYNTHETIC_VOCAB);
    for _ in 0..SYNTHETIC_VOCAB {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        // map to roughly [-10, 10]
        let u = (x >> 8) as f32 / (1u32 << 24) as f32;
        v.push(u * 20.0 - 10.0);
    }
    // A clear top-1 and runner-up, so margin_rel is comfortable (0.4) and the
    // tolerance's lower bound binds rather than its upper bound.
    v[SYNTHETIC_TOP1_IDX] = 25.0;
    v[SYNTHETIC_TOP2_IDX] = 15.0;
    v[SYNTHETIC_BIG_PROBE_IDX] = 20.0;
    v
}

fn synthetic_fixture() -> GoldenFile {
    let mut steps: Vec<GoldenStep> = (0..3)
        .map(|i| derive_step(i, &synthetic_logits(i as u32 + 7)))
        .collect();
    // derive_step picks the same argmax for every synthetic step by construction;
    // the token ids below just mirror it.
    let tokens: Vec<u32> = steps.iter().map(|s| s.argmax_index).collect();

    let min_margin_rel = steps
        .iter()
        .map(|s| s.margin_rel)
        .fold(f64::INFINITY, f64::min);
    let derived = derive_tolerance(0.0, min_margin_rel).expect("synthetic margins are comfortable");
    let tol = ToleranceBlock {
        rel: derived.rel,
        abs: derived.abs,
        derivation: TOLERANCE_DERIVATION.to_string(),
        measured_run_to_run_max_rel_delta: 0.0,
        lower_bound_4ulp: derived.lower_bound_4ulp,
        upper_bound_min_margin_over_10: derived.upper_bound_min_margin_over_10,
    };
    for s in steps.iter_mut() {
        s.unstable = s.margin_rel <= 10.0 * tol.rel;
    }

    GoldenFile {
        readme: format!("SYNTHETIC CONTROL FIXTURE — NOT A GOLDEN. {TIER3_DISCLAIMER}"),
        format_version: FORMAT_VERSION,
        provenance: Provenance {
            captured_utc: "1970-01-01T00:00:00Z".into(),
            lightbulb_commit: "synthetic".into(),
            fuel_worktree_commit: "synthetic".into(),
            rustc: "synthetic".into(),
            cargo_profile: "synthetic".into(),
            host_os: "synthetic".into(),
            host_arch: "synthetic".into(),
            host_cpu: "synthetic".into(),
            fuel_feature_assumption: "synthetic".into(),
            lightbulb_features_enabled: vec![],
            checkpoint_dir: "synthetic".into(),
            config_json_sha256: "synthetic-config-sha".into(),
            model_safetensors_len: 0,
            loader: REQUIRED_LOADER.into(),
            reason: "synthetic control fixture built in code; never written to disk".into(),
        },
        tolerance: tol,
        cases: vec![GoldenCase {
            name: "synthetic".into(),
            prompt: "synthetic".into(),
            prompt_token_ids: vec![1, 2, 3],
            max_new: 3,
            eos: Some(2),
            generated_token_ids: tokens,
            generated_text: "synthetic".into(),
            steps,
        }],
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// CONTROLS — no model, no checkpoint, ordinary CI, milliseconds.
//
// These are the only reason to trust a harness whose real test is `#[ignore]`d.
// A golden check that cannot fail is worse than none, so each control proves the
// comparison FIRES on a specific corruption — and C3 proves it does NOT fire on
// noise, because a tolerance that catches everything means nothing.
// ═════════════════════════════════════════════════════════════════════════════

/// C5 — the float codec round-trips exactly.
///
/// Runs first in spirit: a golden that loses a ULP in serialization can never
/// assert bit-exactness, and would do so silently.
#[test]
fn c5_float_codec_round_trips_bit_exactly() {
    let probes: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE / 2.0, // subnormal
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        25.0,
        20.0,
        -13.372_1,
        1e-30,
        3.402_823_4e38,
    ];

    for &v in &probes {
        let enc = F32Val::from_f32(v);
        let dec = enc.to_f32();
        assert_eq!(
            v.to_bits(),
            dec.to_bits(),
            "bits->f32->bits is not the identity for {v:?} (encoded {enc:?})"
        );

        // The decimal is display-only, but if it does not itself round-trip for
        // finite values then a human reading the fixture is reading something
        // other than the number the test compares.
        if v.is_finite() {
            let reparsed: f32 = enc
                .approx
                .parse()
                .unwrap_or_else(|e| panic!("approx {:?} does not parse: {e}", enc.approx));
            assert_eq!(
                v.to_bits(),
                reparsed.to_bits(),
                "approx {:?} is not a shortest-round-trip rendering of {v:?}",
                enc.approx
            );
        }
    }

    // And the signed zeros must be distinguishable, which is exactly the case a
    // naive decimal-only encoding loses.
    assert_ne!(
        F32Val::from_f32(0.0).bits,
        F32Val::from_f32(-0.0).bits,
        "the codec collapses +0.0 and -0.0"
    );
}

/// The synthetic fixture must survive a real JSON round trip, or every control
/// below is testing an in-memory structure that the on-disk format cannot
/// represent.
#[test]
fn fixture_survives_a_json_round_trip() {
    let f = synthetic_fixture();
    let json = serde_json::to_string_pretty(&f).expect("serialize");
    let back: GoldenFile = serde_json::from_str(&json).expect("deserialize");

    assert_eq!(back.format_version, f.format_version);
    assert_eq!(back.cases.len(), f.cases.len());
    for (a, b) in f.cases[0].steps.iter().zip(&back.cases[0].steps) {
        assert_eq!(a.logits_sha256, b.logits_sha256);
        assert_eq!(a.probe_values, b.probe_values, "probe bits changed on disk");
        assert_eq!(a.top1_value, b.top1_value);
    }
    // f64 fields must survive too — the tolerance is one of them.
    assert_eq!(back.tolerance.rel.to_bits(), f.tolerance.rel.to_bits());
    assert_eq!(back.tolerance.abs.to_bits(), f.tolerance.abs.to_bits());

    // The disclaimer must be in the serialized bytes, not just in the struct.
    assert!(
        json.contains("REGRESSION NET, NOT A VERDICT"),
        "the tier-3 disclaimer is missing from the serialized fixture"
    );
}

/// Baseline: an uncorrupted comparison must report **nothing**.
///
/// Without this, C1–C4 could all "pass" because the comparison reports drift
/// unconditionally — two paths failing identically is a harness smell.
#[test]
fn control_baseline_identical_input_reports_no_drift() {
    let f = synthetic_fixture();
    let observed = f.cases[0].clone();
    let drifts = compare_case(&f.tolerance, &f.cases[0], &observed, true);
    assert!(
        drifts.is_empty(),
        "an identical comparison reported drift, so every other control below is meaningless:\n{}",
        render_drifts(&drifts)
    );
    let inputs = compare_inputs(&f, "synthetic-config-sha", REQUIRED_LOADER);
    assert!(inputs.is_empty(), "identical inputs reported drift: {inputs:?}");
}

/// C1 — a corrupted token id is caught at L1.
#[test]
fn c1_corrupted_token_is_caught() {
    let f = synthetic_fixture();
    let mut observed = f.cases[0].clone();
    observed.generated_token_ids[1] = observed.generated_token_ids[1].wrapping_add(1);

    let drifts = compare_case(&f.tolerance, &f.cases[0], &observed, false);
    assert!(
        drifts.iter().any(|d| d.kind == DriftKind::TokenMismatch),
        "a corrupted token id was NOT caught — the golden check cannot fail, which is worse than \
         having no golden at all. Got: {drifts:?}"
    );
    assert_eq!(
        drifts[0].kind.class(),
        DriftClass::OutputChanged,
        "a token change must classify as OutputChanged, not as a stale fixture"
    );
    assert_eq!(drifts[0].kind.layer(), Layer::L1Tokens);
}

/// C2 — a probe value nudged by 10x the tolerance is caught at L2.
#[test]
fn c2_probe_beyond_tolerance_is_caught() {
    let f = synthetic_fixture();
    let tol = &f.tolerance;
    let mut observed = f.cases[0].clone();

    // The probe holding SYNTHETIC_BIG_PROBE_IDX's value (20.0).
    let k = f.cases[0]
        .steps[0]
        .probe_indices
        .iter()
        .position(|&i| i == SYNTHETIC_BIG_PROBE_IDX)
        .expect("the big probe index is in the probe set");

    let base = f.cases[0].steps[0].probe_values[k].to_f32();
    let nudged = ((base as f64) * (1.0 + 10.0 * tol.rel)) as f32;
    assert_ne!(
        base.to_bits(),
        nudged.to_bits(),
        "the 10x nudge was not representable in f32 — the control would be vacuous"
    );
    observed.steps[0].probe_values[k] = F32Val::from_f32(nudged);

    let drifts = compare_case(tol, &f.cases[0], &observed, false);
    assert!(
        drifts
            .iter()
            .any(|d| d.kind == DriftKind::ProbeOutOfTolerance),
        "a probe nudged by 10x the tolerance was NOT caught. tol_rel={:e} tol_abs={:e} base={base} \
         nudged={nudged}. Got: {drifts:?}",
        tol.rel,
        tol.abs
    );
}

/// C3 — a 1-ULP nudge must NOT be reported.
///
/// The point is not leniency. A tolerance that catches everything is a tolerance
/// that means nothing, and a golden that fires on every ULP would be re-blessed
/// weekly until nobody read it. The assertions inside prove this control
/// exercises the *relative* path rather than sliding through on `tol_abs`.
#[test]
fn c3_sub_tolerance_nudge_is_not_reported() {
    let f = synthetic_fixture();
    let tol = &f.tolerance;
    let mut observed = f.cases[0].clone();

    let k = f.cases[0]
        .steps[0]
        .probe_indices
        .iter()
        .position(|&i| i == SYNTHETIC_BIG_PROBE_IDX)
        .expect("the big probe index is in the probe set");

    let base = f.cases[0].steps[0].probe_values[k].to_f32();
    let nudged = f32::from_bits(base.to_bits() + 1); // exactly 1 ULP up
    let abs = ((base as f64) - (nudged as f64)).abs();
    let rel = abs / (base as f64).abs();

    // Control on the control: if the 1-ULP delta were under tol_abs, this test
    // would pass via the absolute floor and prove nothing about tol_rel.
    assert!(
        abs > tol.abs,
        "1 ULP at {base} is {abs:e}, which is under tol_abs {:e} — this control would be vacuous. \
         Raise SYNTHETIC_BIG_PROBE_IDX's magnitude.",
        tol.abs
    );
    assert!(
        rel < tol.rel,
        "1 ULP at {base} is rel {rel:e}, which is already over tol_rel {:e}",
        tol.rel
    );

    observed.steps[0].probe_values[k] = F32Val::from_f32(nudged);

    let drifts = compare_case(tol, &f.cases[0], &observed, false);
    assert!(
        drifts.is_empty(),
        "a 1-ULP nudge (abs {abs:e}, rel {rel:e}) was reported as drift against tol rel {:e} / abs \
         {:e} — the tolerance is not actually being applied:\n{}",
        tol.rel,
        tol.abs,
        render_drifts(&drifts)
    );

    // ... but the SAME nudge must still move the digest, or L3 is inert.
    let mut strict_observed = observed.clone();
    strict_observed.steps[0].logits_sha256 = "0".repeat(64);
    let strict = compare_case(tol, &f.cases[0], &strict_observed, true);
    assert!(
        strict.iter().any(|d| d.kind == DriftKind::DigestMismatch),
        "L3 did not fire on a changed digest"
    );
}

/// C4 — a corrupted digest is caught at L3, and **only** in strict mode.
///
/// Both halves matter. If the digest fired in the default check, every run on
/// different hardware would look like a regression; if it never fired, the
/// bit-level tripwire would be decorative.
#[test]
fn c4_corrupted_digest_is_caught_only_in_strict_mode() {
    let f = synthetic_fixture();
    let mut observed = f.cases[0].clone();

    let mut bytes: Vec<char> = observed.steps[1].logits_sha256.chars().collect();
    bytes[0] = if bytes[0] == 'a' { 'b' } else { 'a' };
    observed.steps[1].logits_sha256 = bytes.into_iter().collect();

    let lenient = compare_case(&f.tolerance, &f.cases[0], &observed, false);
    assert!(
        lenient.is_empty(),
        "the default (non-strict) check fired on a digest-only change — bit-exactness is not claimed \
         off the capture machine, so this would be a permanent false alarm:\n{}",
        render_drifts(&lenient)
    );

    let strict = compare_case(&f.tolerance, &f.cases[0], &observed, true);
    assert!(
        strict.iter().any(|d| d.kind == DriftKind::DigestMismatch),
        "a corrupted digest was NOT caught in strict mode. Got: {strict:?}"
    );
    assert_eq!(strict[0].kind.layer(), Layer::L3Digest);
}

/// C6 — the stored tolerance is exactly what the documented rule produces from
/// the stored inputs.
///
/// This is the anti-loosening control: someone editing `tolerance.rel` in the
/// JSON to silence a failure changes a number that no longer matches its own
/// recorded derivation, and this fires.
#[test]
fn c6_stored_tolerance_matches_its_declared_derivation() {
    let f = synthetic_fixture();
    let min_margin_rel = f.cases[0]
        .steps
        .iter()
        .map(|s| s.margin_rel)
        .fold(f64::INFINITY, f64::min);

    let recomputed = derive_tolerance(
        f.tolerance.measured_run_to_run_max_rel_delta,
        min_margin_rel,
    )
    .expect("derivation must succeed for a fixture that exists");

    assert_eq!(
        f.tolerance.rel.to_bits(),
        recomputed.rel.to_bits(),
        "stored tol_rel {:e} does not match the rule's output {:e} — the tolerance has been edited \
         away from its own derivation",
        f.tolerance.rel,
        recomputed.rel
    );
    assert_eq!(f.tolerance.abs.to_bits(), recomputed.abs.to_bits());
    assert_eq!(
        f.tolerance.lower_bound_4ulp.to_bits(),
        recomputed.lower_bound_4ulp.to_bits()
    );
    assert_eq!(
        f.tolerance.upper_bound_min_margin_over_10.to_bits(),
        recomputed.upper_bound_min_margin_over_10.to_bits()
    );

    // The rule itself, pinned. If someone changes F32_4ULP_REL the change is
    // visible here rather than diffused through every fixture.
    assert_eq!(F32_4ULP_REL, 4.768_371_582_031_25e-7);
    assert_eq!(TOL_ABS, 1e-6);
    // Comfortable margins => the 4-ULP floor binds, not the margin ceiling.
    assert_eq!(f.tolerance.rel, F32_4ULP_REL);

    // And the derivation must REFUSE rather than clamp when the bounds cross.
    let err = derive_tolerance(0.0, 1e-9).unwrap_err();
    assert!(
        err.contains("no admissible tolerance"),
        "a crossed-bounds derivation should refuse, got: {err}"
    );
}

/// C7 — the margin guard fires on a near-tie step, and stays quiet on healthy
/// ones.
///
/// A step whose top-1/top-2 margin is comparable to the tolerance is a coin
/// flip. Asserting on it produces loud failures for non-bugs, which is how a
/// regression net gets disabled.
#[test]
fn c7_margin_guard_flags_near_ties() {
    let f = synthetic_fixture();

    let healthy = margin_guard_violations(&f.tolerance, &f.cases);
    assert!(
        healthy.is_empty(),
        "the synthetic fixture has comfortable margins but the guard fired: {healthy:?}"
    );

    // Now build a genuine near-tie: top-2 one ULP below top-1.
    let mut logits = synthetic_logits(7);
    logits[SYNTHETIC_TOP2_IDX] = f32::from_bits(logits[SYNTHETIC_TOP1_IDX].to_bits() - 1);
    let step = derive_step(0, &logits);
    assert_eq!(step.argmax_index as usize, SYNTHETIC_TOP1_IDX);
    assert_eq!(step.runner_up_index as usize, SYNTHETIC_TOP2_IDX);
    assert!(
        step.margin_rel > 0.0 && step.margin_rel < 1e-6,
        "expected a hair-thin margin, got {:e}",
        step.margin_rel
    );

    let mut near_tie = f.cases.clone();
    near_tie[0].steps = vec![step];
    let violations = margin_guard_violations(&f.tolerance, &near_tie);
    assert_eq!(
        violations.len(),
        1,
        "the margin guard did not flag a 1-ULP near-tie, so the fixture would silently bake in a \
         coin flip. Got: {violations:?}"
    );

    // And `finalize_unstable` must mark it, so the flag survives into the file.
    finalize_unstable(&f.tolerance, &mut near_tie);
    assert!(near_tie[0].steps[0].unstable);
}

/// The stale-fixture path must be distinguishable from the regression path.
///
/// If a different checkpoint reported as "TokenMismatch", the reflex fix would
/// be to re-bless — exactly the wrong action.
#[test]
fn control_wrong_inputs_report_as_fixture_invalid_not_regression() {
    let f = synthetic_fixture();

    let wrong_ckpt = compare_inputs(&f, "some-other-config-sha", REQUIRED_LOADER);
    assert!(
        wrong_ckpt
            .iter()
            .any(|d| d.kind == DriftKind::CheckpointMismatch),
        "a different checkpoint was not detected: {wrong_ckpt:?}"
    );
    assert!(
        wrong_ckpt
            .iter()
            .all(|d| d.kind.class() == DriftClass::FixtureInvalid),
        "a different checkpoint must classify as FixtureInvalid"
    );

    let wrong_loader = compare_inputs(&f, "synthetic-config-sha", "load_llama_from_dir");
    assert!(
        wrong_loader
            .iter()
            .any(|d| d.kind == DriftKind::LoaderMismatch),
        "the bf16 loader was not rejected: {wrong_loader:?} — goldens captured on the bf16 path are \
         arm-dependent and must never be compared against"
    );

    // A changed prompt must short-circuit, not cascade into token noise.
    let mut observed = f.cases[0].clone();
    observed.prompt_token_ids.push(99);
    let drifts = compare_case(&f.tolerance, &f.cases[0], &observed, true);
    assert_eq!(drifts.len(), 1, "a prompt change should short-circuit: {drifts:?}");
    assert_eq!(drifts[0].kind, DriftKind::PromptMismatch);
    assert_eq!(drifts[0].kind.class(), DriftClass::FixtureInvalid);
}

/// The failure text a human will actually read must carry the disclaimer, the
/// triage table and the do-not-re-bless instruction. This is the *social*
/// mitigation, and it is the one most likely to rot silently.
///
/// The first assertion is not ceremony. A mutation that made `compare_case`
/// return an empty vector left this test **green**, because `render_drifts`
/// emits the disclaimer and the triage table even with nothing to report — so
/// without pinning that a drift was actually produced, this control was checking
/// three string constants against themselves.
#[test]
fn control_failure_message_carries_the_disclaimer_and_triage() {
    let f = synthetic_fixture();
    let mut observed = f.cases[0].clone();
    observed.generated_token_ids[0] = observed.generated_token_ids[0].wrapping_add(7);
    let drifts = compare_case(&f.tolerance, &f.cases[0], &observed, false);
    assert!(
        drifts.iter().any(|d| d.kind == DriftKind::TokenMismatch),
        "no drift was produced, so the rendered message below proves nothing about a real failure"
    );
    let msg = render_drifts(&drifts);

    // The specific finding must be in the text, not merely the boilerplate.
    assert!(
        msg.contains("TokenMismatch") && msg.contains("recorded margin_abs"),
        "the message omits the actual finding and its recorded margin, which is the one number that \
         distinguishes a near-tie flip from a regression:\n---\n{msg}"
    );

    for needle in [
        "REGRESSION NET, NOT A VERDICT",
        "TRIAGE",
        "Do NOT re-bless",
        "kissref_differential",
        "FEATURE SET",
    ] {
        assert!(
            msg.contains(needle),
            "the failure message is missing {needle:?}; a tier-3 failure that does not explain itself \
             will be read as a correctness verdict.\n---\n{msg}"
        );
    }
}

/// Digest sanity: the digest must actually depend on the values and on order.
#[test]
fn control_digest_is_order_and_value_sensitive() {
    let a = vec![1.0f32, 2.0, 3.0];
    let mut b = a.clone();
    b.swap(0, 2);
    let mut c = a.clone();
    c[1] = f32::from_bits(c[1].to_bits() + 1);

    assert_eq!(logits_digest(&a), logits_digest(&a.clone()));
    assert_ne!(logits_digest(&a), logits_digest(&b), "digest ignores order");
    assert_ne!(
        logits_digest(&a),
        logits_digest(&c),
        "digest ignores a 1-ULP value change"
    );
    // Signed zeros are distinct bit patterns and must be distinct digests.
    assert_ne!(logits_digest(&[0.0f32]), logits_digest(&[-0.0f32]));
}

/// The probe set is a fixed, documented function of the vocabulary — never
/// re-picked per capture.
#[test]
fn control_probe_set_is_fixed_and_covers_the_vocab() {
    let idx = probe_indices(32_000);
    assert_eq!(PROBE_STRIDE, 997, "the probe stride is part of the format");
    assert_eq!(idx.len(), 33, "expected 33 probes over a 32000 vocab");
    assert_eq!(idx[0], 0);
    assert_eq!(*idx.last().unwrap(), 997 * 32);
    assert!(
        idx.windows(2).all(|w| w[1] - w[0] == PROBE_STRIDE),
        "probe indices are not evenly strided"
    );
}

/// What we believe about the build, asserted for the part that is detectable
/// from inside Lightbulb.
///
/// The Fuel side is not detectable here; it is pinned externally and recorded in
/// the fixture's `fuel_feature_assumption`. Verified at authoring time with
/// `cargo tree -e features -i fuel-cpu-backend` and `-i fuel-core`: both resolve
/// with feature `default` only, and both declare `default = []`.
#[test]
fn default_feature_set_assumptions_hold() {
    assert!(
        !cfg!(feature = "cuda"),
        "goldens are captured and checked on the CPU path only"
    );
    assert!(!cfg!(feature = "metal"), "goldens are CPU-only");
    assert!(!cfg!(feature = "flash-attn"), "goldens are CPU-only");
    assert!(!cfg!(feature = "cuda-full"), "goldens are CPU-only");
}

/// If the fixture has been captured, validate everything about it that does not
/// need the model: codec, derivation, margin guard, loader provenance.
///
/// This is what keeps a committed fixture honest in ordinary CI. It **skips
/// loudly** while the fixture does not exist yet, which at the time of writing
/// is the case.
#[test]
fn committed_fixture_is_self_consistent_if_present() {
    let path = fixture_path();
    if !path.is_file() {
        eprintln!(
            "SKIPPING: no golden fixture at {}.\n\
             The tier-3 machinery is present and its controls pass, but the fixture has NOT been \
             generated. Generate it with:\n  cargo test --release --test model_fuel_golden -- \
             --ignored --nocapture capture_golden_fixture",
            path.display()
        );
        return;
    }

    let text = fs::read_to_string(&path).expect("reading the fixture");
    let f: GoldenFile = serde_json::from_str(&text).expect("parsing the fixture");

    assert_eq!(f.format_version, FORMAT_VERSION);
    assert!(
        f.readme.contains("REGRESSION NET, NOT A VERDICT"),
        "the committed fixture does not carry the tier-3 disclaimer"
    );
    assert_eq!(
        f.provenance.loader, REQUIRED_LOADER,
        "the fixture was captured through {:?}, not the all-f32 loader — bf16 projections make the \
         numbers arm-dependent and worthless as a fixed point",
        f.provenance.loader
    );
    assert!(!f.cases.is_empty(), "the fixture has no cases");

    // C5 over every float actually stored.
    for c in &f.cases {
        for s in &c.steps {
            assert_eq!(s.probe_stride, PROBE_STRIDE);
            assert_eq!(s.probe_indices, probe_indices(s.logits_len));
            assert_eq!(s.probe_values.len(), s.probe_indices.len());
            for v in s
                .probe_values
                .iter()
                .chain([&s.top1_value, &s.top2_value, &s.stats.max, &s.stats.min])
            {
                let round = F32Val::from_f32(v.to_f32());
                assert_eq!(
                    round.bits, v.bits,
                    "stored float {v:?} does not round-trip through the codec"
                );
            }
            assert_eq!(
                s.logits_sha256.len(),
                64,
                "sha256 field is not 64 hex chars: {:?}",
                s.logits_sha256
            );
        }
        assert_eq!(
            c.steps.len(),
            c.generated_token_ids.len(),
            "every generated token must have exactly one recorded step"
        );
        for (i, s) in c.steps.iter().enumerate() {
            assert_eq!(
                s.argmax_index, c.generated_token_ids[i],
                "step {i}'s recorded argmax disagrees with the recorded token — the capture harness \
                 and the shipped generate_greedy disagreed, so the fixture pins the wrong thing"
            );
        }
    }

    // C6 against the real numbers.
    let min_margin_rel = f
        .cases
        .iter()
        .flat_map(|c| c.steps.iter())
        .map(|s| s.margin_rel)
        .fold(f64::INFINITY, f64::min);
    let recomputed = derive_tolerance(f.tolerance.measured_run_to_run_max_rel_delta, min_margin_rel)
        .expect("the committed fixture's tolerance must be derivable");
    assert_eq!(
        f.tolerance.rel.to_bits(),
        recomputed.rel.to_bits(),
        "the committed tolerance {:e} does not match its own recorded derivation ({:e}) — it has been \
         hand-edited",
        f.tolerance.rel,
        recomputed.rel
    );

    // C7 against the real numbers.
    let violations = margin_guard_violations(&f.tolerance, &f.cases);
    assert!(
        violations.is_empty(),
        "the committed fixture contains steps that are coin flips:\n{}",
        violations.join("\n")
    );

    eprintln!(
        "fixture OK: {} cases, {} steps, tol_rel {:e} tol_abs {:e}, captured {} on {} ({})",
        f.cases.len(),
        f.cases.iter().map(|c| c.steps.len()).sum::<usize>(),
        f.tolerance.rel,
        f.tolerance.abs,
        f.provenance.captured_utc,
        f.provenance.host_cpu,
        f.provenance.cargo_profile
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Model-driving machinery. Everything below needs the 2.2 GB checkpoint and is
// `#[ignore]`d.
// ═════════════════════════════════════════════════════════════════════════════

use lightbulb::model_fuel::generate::generate_greedy;
use lightbulb::model_fuel::loader::LoadedLlama;
use lightbulb::model_fuel::loader_f32::load_llama_f32_from_dir;

/// Drive `forward_with_kv_context_persistent` with the **identical call shape**
/// to `src/model_fuel/generate.rs`, capturing the per-step logits that
/// `generate_greedy` throws away.
///
/// This is the one place tier 3 duplicates shipped code, and it is a real risk:
/// a golden pinned to a divergent copy pins the wrong thing. Two mitigations,
/// both stated rather than assumed:
///
/// 1. [`capture_case`] runs the real `generate_greedy` as well and asserts the
///    token sequences agree, so a divergence in *behaviour* is caught in the
///    same run rather than baked into the fixture.
/// 2. That cross-check would **not** catch a divergence that leaves the tokens
///    identical while changing the logits (a different KV capacity, different
///    eos handling). Keep this function textually identical to `generate.rs` and
///    re-read it whenever `generate.rs` changes.
fn capture_logits(
    loaded: &LoadedLlama,
    prompt_tokens: &[u32],
    max_new: usize,
    eos: Option<u32>,
) -> anyhow::Result<(Vec<Vec<f32>>, Vec<u32>)> {
    use fuel::inference_context::{InferenceContext, KvCache};
    use fuel::Device;

    // Identical to generate.rs: capacity = prompt + max_new + 1.
    let max_seq_len = prompt_tokens.len() + max_new + 1;
    let c = &loaded.config;
    let device = Device::cpu();

    let mut cache = KvCache::with_capacity(
        c.n_layers,
        c.n_kv_heads,
        c.head_dim,
        max_seq_len,
        fuel::DType::F32,
        &device,
    )
    .map_err(|e| anyhow::anyhow!("allocating KV cache: {e:?}"))?;
    let mut ctx = InferenceContext::new(device);
    let mut session: Option<fuel::inference_context::DecodeSession> = None;

    let mut logits = loaded
        .model
        .forward_with_kv_context_persistent(prompt_tokens, &mut cache, &mut ctx, &mut session)
        .map_err(|e| anyhow::anyhow!("prefill forward: {e:?}"))?;

    let mut all = Vec::with_capacity(max_new);
    let mut generated = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        all.push(logits.clone());
        let next = argmax(&logits);
        generated.push(next);
        if Some(next) == eos {
            break;
        }
        logits = loaded
            .model
            .forward_with_kv_context_persistent(&[next], &mut cache, &mut ctx, &mut session)
            .map_err(|e| anyhow::anyhow!("decode forward: {e:?}"))?;
    }

    Ok((all, generated))
}

struct CaseSpec {
    name: &'static str,
    prompt: &'static str,
    max_new: usize,
    eos: Option<u32>,
    /// When set, the prompt is truncated to its first token. See
    /// [`case_specs`] for why that is a distinct code path rather than variety.
    first_token_only: bool,
}

/// Three cases, chosen for **path coverage**, not variety.
fn case_specs() -> Vec<CaseSpec> {
    vec![
        // The known-good vertical-slice prompt: ties tier 3 to the existing
        // proof (`generate.rs::generates_coherent_text_end_to_end`).
        CaseSpec {
            name: "capital_of_france",
            prompt: "The capital of France is",
            max_new: 6,
            eos: Some(2),
            first_token_only: false,
        },
        // Longer: more decode steps, hence more chances of a near-tie, which is
        // what populates the margin data the triage depends on.
        CaseSpec {
            name: "longer_continuation",
            prompt: "In a distant galaxy, a small crew of engineers discovered that",
            max_new: 12,
            eos: Some(2),
            first_token_only: false,
        },
        // A SINGLE-token prompt. `forward_with_kv_context_persistent` falls back
        // to the rebuild path when `seq != 1` (fuel-core/src/lazy.rs:7864-7868),
        // so a 1-token prompt makes PREFILL take the session-building path on
        // step 0. Nothing else in the suite pins that. The token is taken from
        // case 1's encoding (i.e. BOS) rather than hardcoded, so it stays
        // correct if the tokenizer changes.
        CaseSpec {
            name: "single_token_prefill",
            prompt: "The capital of France is",
            max_new: 4,
            eos: Some(2),
            first_token_only: true,
        },
    ]
}

fn capture_case(
    loaded: &LoadedLlama,
    tok: &tokenizers::Tokenizer,
    spec: &CaseSpec,
) -> anyhow::Result<GoldenCase> {
    let mut ids: Vec<u32> = tok
        .encode(spec.prompt, true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    if spec.first_token_only {
        ids.truncate(1);
    }
    assert!(!ids.is_empty(), "case {} encoded to nothing", spec.name);
    if spec.first_token_only {
        assert_eq!(ids.len(), 1, "single-token case is not single-token");
    }

    let started = std::time::Instant::now();
    let (logits, tokens_from_capture) = capture_logits(loaded, &ids, spec.max_new, spec.eos)?;

    // STEP 8: confirm the last-position slice at RUNTIME rather than trusting a
    // static reading of lazy.rs:7794-7800.
    for (i, l) in logits.iter().enumerate() {
        assert_eq!(
            l.len(),
            loaded.config.vocab_size,
            "case {} step {i}: forward returned {} values, expected vocab_size {} — the persistent \
             forward is not returning last-position-only logits, and every derived number below \
             would be measuring the wrong thing",
            spec.name,
            l.len(),
            loaded.config.vocab_size
        );
    }

    // THE HARNESS-FIDELITY CONTROL. The capture drives the forward directly; the
    // shipped loop is `generate_greedy`. If they disagree, the fixture is pinning
    // a duplicate rather than the implementation, and that must surface now.
    let tokens_from_impl = generate_greedy(loaded, &ids, spec.max_new, spec.eos)?;
    assert_eq!(
        tokens_from_capture, tokens_from_impl,
        "case {}: the capture harness and the shipped generate_greedy produced DIFFERENT tokens \
         ({tokens_from_capture:?} vs {tokens_from_impl:?}). The harness has drifted from \
         src/model_fuel/generate.rs, so anything captured here would pin the wrong code.",
        spec.name
    );

    let text = tok
        .decode(&tokens_from_impl, true)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;

    eprintln!(
        "  case {:<22} {} prompt tok -> {} new tok in {:.1?} : {text:?}",
        spec.name,
        ids.len(),
        tokens_from_impl.len(),
        started.elapsed()
    );

    let steps: Vec<GoldenStep> = logits
        .iter()
        .enumerate()
        .map(|(i, l)| derive_step(i, l))
        .collect();

    Ok(GoldenCase {
        name: spec.name.to_string(),
        prompt: spec.prompt.to_string(),
        prompt_token_ids: ids,
        max_new: spec.max_new,
        eos: spec.eos,
        generated_token_ids: tokens_from_impl,
        generated_text: text,
        steps,
    })
}

// ── provenance helpers ───────────────────────────────────────────────────────

/// Resolve a repository's HEAD **by reading the filesystem**, never by shelling
/// out to git.
///
/// This is not squeamishness. `C:/Projects/fuel-lightbulb-port/.git` is a
/// worktree pointer into `C:/Projects/fuel/.git/worktrees/...`, and that mirror
/// is shared read-only across sessions where this project forbids running any
/// git command. Reading a file there is a read; running git there is not.
fn git_head(repo: &Path) -> String {
    let dot = repo.join(".git");
    let gitdir = if dot.is_file() {
        match fs::read_to_string(&dot) {
            Ok(s) => PathBuf::from(
                s.trim()
                    .strip_prefix("gitdir:")
                    .unwrap_or(s.trim())
                    .trim()
                    .to_string(),
            ),
            Err(_) => return "unknown".into(),
        }
    } else {
        dot
    };
    let head = match fs::read_to_string(gitdir.join("HEAD")) {
        Ok(s) => s.trim().to_string(),
        Err(_) => return "unknown".into(),
    };
    let Some(refname) = head.strip_prefix("ref:").map(str::trim) else {
        return head; // detached: already a sha
    };
    if let Ok(s) = fs::read_to_string(gitdir.join(refname)) {
        return s.trim().to_string();
    }
    if let Ok(packed) = fs::read_to_string(gitdir.join("packed-refs")) {
        for line in packed.lines() {
            if let Some((sha, name)) = line.split_once(' ') {
                if name.trim() == refname {
                    return sha.trim().to_string();
                }
            }
        }
    }
    format!("unresolved {refname}")
}

fn sha256_file(p: &Path) -> String {
    match fs::read(p) {
        Ok(bytes) => {
            let mut h = Sha256::new();
            h.update(&bytes);
            format!("{:x}", h.finalize())
        }
        Err(e) => format!("unreadable: {e}"),
    }
}

fn rustc_version() -> String {
    match std::process::Command::new("rustc").arg("-vV").output() {
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout).into_owned();
            text.lines()
                .filter(|l| {
                    l.starts_with("rustc ") || l.starts_with("host:") || l.starts_with("release:")
                })
                .collect::<Vec<_>>()
                .join("; ")
        }
        Err(_) => "unknown".into(),
    }
}

fn cargo_profile() -> &'static str {
    if cfg!(debug_assertions) { "debug" } else { "release" }
}

fn build_provenance(dir: &Path, reason: String) -> Provenance {
    Provenance {
        captured_utc: chrono::Utc::now().to_rfc3339(),
        lightbulb_commit: git_head(Path::new(env!("CARGO_MANIFEST_DIR"))),
        fuel_worktree_commit: git_head(Path::new("C:/Projects/fuel-lightbulb-port")),
        rustc: rustc_version(),
        cargo_profile: cargo_profile().to_string(),
        host_os: format!("{} {}", std::env::consts::OS, std::env::consts::FAMILY),
        host_arch: std::env::consts::ARCH.to_string(),
        host_cpu: std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".into()),
        fuel_feature_assumption:
            "fuel-core and fuel-cpu-backend taken as PATH deps with default features; both declare \
             default = [], so mkl / accelerate / aocl / onemkl / cuda / cudnn / nccl / telemetry / \
             jit are OFF. Not detectable from inside Lightbulb at runtime — verified externally with \
             `cargo tree -e features -i fuel-cpu-backend` and `-i fuel-core`, both of which resolve \
             to feature \"default\" only. Enabling any BLAS feature swaps the matmul kernel outright \
             and invalidates every number in this file."
                .into(),
        lightbulb_features_enabled: {
            let mut v = Vec::new();
            if cfg!(feature = "cuda") {
                v.push("cuda".to_string());
            }
            if cfg!(feature = "metal") {
                v.push("metal".to_string());
            }
            if cfg!(feature = "flash-attn") {
                v.push("flash-attn".to_string());
            }
            v
        },
        checkpoint_dir: dir.display().to_string(),
        config_json_sha256: sha256_file(&dir.join("config.json")),
        model_safetensors_len: fs::metadata(dir.join("model.safetensors"))
            .map(|m| m.len())
            .unwrap_or(0),
        loader: REQUIRED_LOADER.to_string(),
        reason,
    }
}

// ── the bless gate ───────────────────────────────────────────────────────────

/// Two keys to overwrite; none to bootstrap.
///
/// The reason string is not decoration: it lands in `provenance.reason` and
/// therefore in the git diff, immediately next to the numbers that moved. A
/// reviewer reads the sentence before the floats, which is the single most
/// effective thing standing between this fixture and rubber-stamp rot.
fn bless_reason_or_refuse(exists: bool) -> Result<String, String> {
    if !exists {
        return Ok("initial capture (no fixture existed; bootstrapping is not a re-blessing)".into());
    }
    let key = std::env::var("LIGHTBULB_BLESS_GOLDEN").unwrap_or_default();
    let reason = std::env::var("LIGHTBULB_BLESS_REASON").unwrap_or_default();
    if key != "1" || reason.trim().is_empty() {
        return Err(format!(
            "REFUSING to overwrite the existing golden fixture.\n\n{TIER3_DISCLAIMER}\n\n\
             Overwriting requires BOTH:\n  \
             LIGHTBULB_BLESS_GOLDEN=1        (currently {key:?})\n  \
             LIGHTBULB_BLESS_REASON=\"...\"    (currently {reason:?})\n\n\
             The gate exists because a golden that can be regenerated by reflex stops being a \
             regression net and becomes a rubber stamp. The reason is written into \
             provenance.reason, so the diff shows a human sentence explaining why the numbers moved, \
             right next to the numbers that moved.\n\n\
             Before blessing: establish via tiers 1/2 (tests/kissref_differential.rs) whether the new \
             output is more or less correct. Tier 3 cannot answer that and must not be used to."
        ));
    }
    Ok(reason)
}

// ── the tests that run the model ─────────────────────────────────────────────

/// **STEP 1 GATE.** Measure run-to-run reproducibility before anything claims
/// bit-exactness.
///
/// Two comparisons, and they catch different things:
///
/// - **Within one process** — catches state accumulating across sessions
///   (`KvCache`, `DecodeSession`, `InferenceContext`).
/// - **Across processes** — catches `HashMap` `RandomState` reseeding, allocator
///   and ASLR effects, anything process-scoped. Rust reseeds `RandomState` per
///   process, so an optimizer decision that depended on hash iteration order
///   would show up here and nowhere else.
///
/// The cross-process half needs the test **run twice**: the first run writes the
/// digests to `target/model_fuel_golden_determinism_probe.json`, the second
/// compares against it. That is stated rather than automated because a test that
/// silently re-execs itself is a test nobody can reason about.
///
/// ```text
/// cargo test --release --test model_fuel_golden -- --ignored --nocapture determinism_probe
/// cargo test --release --test model_fuel_golden -- --ignored --nocapture determinism_probe
/// ```
///
/// **L3 (bit-exactness) may not be believed until this has reported zero drift
/// in both halves.** If it reports nonzero drift, the digest layer drops to
/// informational and `measured_run_to_run_max_rel_delta` (which feeds the
/// tolerance) becomes nonzero.
#[test]
#[ignore = "needs the TinyLlama checkpoint; loads ~4.4 GB f32 and runs two generations"]
fn determinism_probe() -> anyhow::Result<()> {
    let Some(dir) = tinyllama_dir() else {
        eprintln!("skipping: no TinyLlama snapshot");
        return Ok(());
    };
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let ids: Vec<u32> = tok
        .encode("The capital of France is", true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();

    eprintln!("loading all-f32 weights (~4.4 GB) ...");
    let loaded = load_llama_f32_from_dir(&dir)?;

    let (a, ta) = capture_logits(&loaded, &ids, 4, Some(2))?;
    let (b, tb) = capture_logits(&loaded, &ids, 4, Some(2))?;
    assert_eq!(ta, tb, "same-process runs produced different tokens");
    assert_eq!(a.len(), b.len(), "same-process runs produced different step counts");

    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    let mut bit_identical = true;
    for (sa, sb) in a.iter().zip(&b) {
        assert_eq!(sa.len(), sb.len());
        for (&x, &y) in sa.iter().zip(sb) {
            if x.to_bits() != y.to_bits() {
                bit_identical = false;
            }
            let d = ((x as f64) - (y as f64)).abs();
            if d > max_abs {
                max_abs = d;
            }
            let r = d / (x as f64).abs().max(f64::MIN_POSITIVE);
            if r > max_rel {
                max_rel = r;
            }
        }
    }

    eprintln!("SAME-PROCESS: bit_identical={bit_identical} max_abs={max_abs:e} max_rel={max_rel:e}");

    let digests: Vec<String> = a.iter().map(|l| logits_digest(l)).collect();
    let probe_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("model_fuel_golden_determinism_probe.json");

    if probe_file.is_file() {
        let prev: Vec<String> =
            serde_json::from_str(&fs::read_to_string(&probe_file)?).unwrap_or_default();
        let cross_ok = prev == digests;
        eprintln!(
            "CROSS-PROCESS: {} (compared against {})",
            if cross_ok { "BIT-IDENTICAL" } else { "DIFFERENT" },
            probe_file.display()
        );
        assert!(
            cross_ok,
            "cross-process digests differ:\n  previous {prev:?}\n  now      {digests:?}\n\
             Something process-scoped (hash iteration order, allocator, ASLR) is reaching the \
             numbers. L3 bit-exactness must NOT be claimed; drop it to informational and feed the \
             measured relative delta into the tolerance."
        );
    } else {
        fs::create_dir_all(probe_file.parent().unwrap())?;
        fs::write(&probe_file, serde_json::to_string_pretty(&digests)?)?;
        eprintln!(
            "CROSS-PROCESS: baseline written to {}. RUN THIS TEST AGAIN to complete the probe — \
             one run only measures within a process.",
            probe_file.display()
        );
    }

    assert!(
        bit_identical,
        "same-process runs are NOT bit-identical (max_abs {max_abs:e}, max_rel {max_rel:e}). \
         State is leaking across sessions, or a kernel is not deterministic. Do not proceed to a \
         bit-exact golden layer; use max_rel as measured_run_to_run_max_rel_delta in the tolerance \
         derivation."
    );
    Ok(())
}

/// **The capture.** Writes the fixture. Never asserts against it.
///
/// Cost: one ~4.4 GB f32 load (shared across all cases — the weights are
/// immutable and each generation builds its own `KvCache`/session) plus **two**
/// generations per case, because the harness-fidelity control runs the shipped
/// `generate_greedy` alongside the logits capture.
#[test]
#[ignore = "needs the TinyLlama checkpoint; ~4.5 s/token in release, ~6 generations"]
fn capture_golden_fixture() -> anyhow::Result<()> {
    let Some(dir) = tinyllama_dir() else {
        eprintln!("skipping: no TinyLlama snapshot");
        return Ok(());
    };

    assert!(
        !cfg!(debug_assertions),
        "REFUSING to capture a golden from a debug build. Debug and release have NOT been shown to \
         produce identical f32 here (the argument that they do — no FP contraction without an \
         explicit mul_add — is an argument, not a measurement), and the check runs in release. \
         Re-run with --release."
    );

    let path = fixture_path();
    let reason = bless_reason_or_refuse(path.is_file()).map_err(|m| anyhow::anyhow!("{m}"))?;

    // The measured determinism figure that feeds the tolerance. 0.0 is only
    // legitimate once `determinism_probe` has reported bit-identical in BOTH
    // halves; override it here if it reported drift.
    let measured_delta: f64 = std::env::var("LIGHTBULB_MEASURED_RUN_DELTA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    eprintln!(
        "measured_run_to_run_max_rel_delta = {measured_delta:e} (override with \
         LIGHTBULB_MEASURED_RUN_DELTA; 0.0 asserts determinism_probe reported bit-identical)"
    );

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    eprintln!("loading all-f32 weights (~4.4 GB) from {} ...", dir.display());
    let t0 = std::time::Instant::now();
    let loaded = load_llama_f32_from_dir(&dir)?;
    eprintln!("loaded in {:.1?}", t0.elapsed());

    let mut cases = Vec::new();
    for spec in case_specs() {
        cases.push(capture_case(&loaded, &tok, &spec)?);
    }

    let min_margin_rel = cases
        .iter()
        .flat_map(|c| c.steps.iter())
        .map(|s| s.margin_rel)
        .fold(f64::INFINITY, f64::min);
    let derived = derive_tolerance(measured_delta, min_margin_rel).map_err(|e| {
        anyhow::anyhow!(
            "{e}\nThe tightest margin across the captured cases is too small for any admissible \
             tolerance. Do not widen the rule; drop or replace the offending case."
        )
    })?;
    let tol = ToleranceBlock {
        rel: derived.rel,
        abs: derived.abs,
        derivation: TOLERANCE_DERIVATION.to_string(),
        measured_run_to_run_max_rel_delta: measured_delta,
        lower_bound_4ulp: derived.lower_bound_4ulp,
        upper_bound_min_margin_over_10: derived.upper_bound_min_margin_over_10,
    };
    finalize_unstable(&tol, &mut cases);

    // Refuse to write a fixture whose steps are coin flips without saying so.
    let violations = margin_guard_violations(&tol, &cases);
    for v in &violations {
        eprintln!("UNSTABLE STEP: {v}");
    }
    assert!(
        violations.is_empty(),
        "{} captured step(s) have a margin comparable to the tolerance. They are flagged \
         `unstable: true` in the struct, but writing them into a golden presents a coin flip as a \
         regression signal. Replace the prompt or shorten max_new.",
        violations.len()
    );

    let file = GoldenFile {
        readme: TIER3_DISCLAIMER.to_string(),
        format_version: FORMAT_VERSION,
        provenance: build_provenance(&dir, reason),
        tolerance: tol,
        cases,
    };

    fs::create_dir_all(path.parent().unwrap())?;
    let mut json = serde_json::to_string_pretty(&file)?;
    json.push('\n');
    fs::write(&path, &json)?;

    eprintln!(
        "wrote {} ({} bytes, {} cases, {} steps, tol_rel {:e})",
        path.display(),
        json.len(),
        file.cases.len(),
        file.cases.iter().map(|c| c.steps.len()).sum::<usize>(),
        file.tolerance.rel
    );
    eprintln!("\n{TIER3_DISCLAIMER}\n");
    Ok(())
}

/// **The check.** L1 then L2. Not L3 — see [`golden_is_bit_exact`].
#[test]
#[ignore = "needs the TinyLlama checkpoint; ~4.5 s/token in release"]
fn golden_matches_fixture() -> anyhow::Result<()> {
    let Some(dir) = tinyllama_dir() else {
        eprintln!("skipping: no TinyLlama snapshot");
        return Ok(());
    };
    let path = fixture_path();
    assert!(
        path.is_file(),
        "no golden fixture at {}. The tier-3 machinery exists but the fixture has NOT been \
         generated. Create it with:\n  cargo test --release --test model_fuel_golden -- --ignored \
         --nocapture capture_golden_fixture",
        path.display()
    );

    let fixture: GoldenFile = serde_json::from_str(&fs::read_to_string(&path)?)?;
    let observed_sha = sha256_file(&dir.join("config.json"));

    let input_drifts = compare_inputs(&fixture, &observed_sha, REQUIRED_LOADER);
    assert!(
        input_drifts.is_empty(),
        "FIXTURE STALE / WRONG INPUTS — this is NOT a regression:\n{}",
        render_drifts(&input_drifts)
    );

    if fixture.provenance.cargo_profile != cargo_profile() {
        eprintln!(
            "NOTE: fixture captured in {} profile, checking in {}. Debug/release f32 equality is \
             argued but not measured; treat any drift below with that in mind.",
            fixture.provenance.cargo_profile,
            cargo_profile()
        );
    }

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let loaded = load_llama_f32_from_dir(&dir)?;

    let mut drifts = Vec::new();
    for spec in case_specs() {
        let Some(expected) = fixture.cases.iter().find(|c| c.name == spec.name) else {
            eprintln!("fixture has no case {:?}; skipping", spec.name);
            continue;
        };
        let observed = capture_case(&loaded, &tok, &spec)?;
        drifts.extend(compare_case(&fixture.tolerance, expected, &observed, false));
    }

    // C7 on this run's numbers, not just the captured ones.
    for v in margin_guard_violations(&fixture.tolerance, &fixture.cases) {
        eprintln!("UNSTABLE STEP (fixture): {v}");
    }

    assert!(
        drifts.is_empty(),
        "GOLDEN DRIFT — {} finding(s).\n{}",
        drifts.len(),
        render_drifts(&drifts)
    );
    Ok(())
}

/// **L3, on its own, deliberately.**
///
/// Bit-exactness is claimed only for (same binary, same machine, same feature
/// set). This test is expected to fail elsewhere and says so rather than
/// pretending otherwise, which is why it is not folded into
/// [`golden_matches_fixture`] as a warning: a warning nobody reads is worse than
/// no check.
#[test]
#[ignore = "needs the checkpoint; bit-exactness only holds on the capture machine"]
fn golden_is_bit_exact() -> anyhow::Result<()> {
    let Some(dir) = tinyllama_dir() else {
        eprintln!("skipping: no TinyLlama snapshot");
        return Ok(());
    };
    let path = fixture_path();
    assert!(path.is_file(), "no golden fixture at {}", path.display());
    let fixture: GoldenFile = serde_json::from_str(&fs::read_to_string(&path)?)?;

    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let loaded = load_llama_f32_from_dir(&dir)?;

    let mut drifts = Vec::new();
    for spec in case_specs() {
        let Some(expected) = fixture.cases.iter().find(|c| c.name == spec.name) else {
            continue;
        };
        let observed = capture_case(&loaded, &tok, &spec)?;
        drifts.extend(
            compare_case(&fixture.tolerance, expected, &observed, true)
                .into_iter()
                .filter(|d| d.kind == DriftKind::DigestMismatch),
        );
    }

    assert!(
        drifts.is_empty(),
        "BIT-LEVEL DRIFT ({} step(s)). This is EXPECTED on different hardware, a different rustc, \
         or a different feature set — softmax goes through the system libm's `exp`, which is not \
         bit-identical across platforms. On its own this is NOT a regression: check \
         `golden_matches_fixture` (L1/L2) first. Fixture host: {} / {} / {}.\n{}",
        drifts.len(),
        fixture.provenance.host_cpu,
        fixture.provenance.rustc,
        fixture.provenance.cargo_profile,
        render_drifts(&drifts)
    );
    Ok(())
}

/// **C8 — cache-size invariance.**
///
/// `generate_greedy` sizes the `KvCache` as `prompt.len() + max_new + 1`, so the
/// cache capacity is an *input* to every run. Run the same prompt with `max_new`
/// and `max_new + 3` and the first `max_new` tokens — and step 0's logits, bit
/// for bit — must be identical.
///
/// This catches mask / `WriteSlice` boundary leakage: a class of bug where the
/// answer depends on how much room was allocated. A golden alone would silently
/// freeze such a bug in place, because it only ever captures one capacity.
#[test]
#[ignore = "needs the TinyLlama checkpoint; two generations in release"]
fn c8_cache_capacity_does_not_leak_into_values() -> anyhow::Result<()> {
    let Some(dir) = tinyllama_dir() else {
        eprintln!("skipping: no TinyLlama snapshot");
        return Ok(());
    };
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let ids: Vec<u32> = tok
        .encode("The capital of France is", true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();

    let loaded = load_llama_f32_from_dir(&dir)?;

    let n = 4usize;
    let (small_logits, small_tokens) = capture_logits(&loaded, &ids, n, Some(2))?;
    let (big_logits, big_tokens) = capture_logits(&loaded, &ids, n + 3, Some(2))?;

    let k = small_tokens.len().min(big_tokens.len());
    assert_eq!(
        &small_tokens[..k],
        &big_tokens[..k],
        "the first {k} tokens changed when only the KV cache CAPACITY changed \
         ({} vs {} slots) — the mask or the WriteSlice offset depends on capacity, and a golden \
         would freeze that bug in place",
        ids.len() + n + 1,
        ids.len() + n + 4
    );

    let a = &small_logits[0];
    let b = &big_logits[0];
    assert_eq!(a.len(), b.len());
    let differing = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert_eq!(
        differing, 0,
        "{differing} of {} step-0 logits differ bit-for-bit between cache capacities {} and {}. \
         Prefill reads no cache beyond what it writes, so capacity must not reach the values.",
        a.len(),
        ids.len() + n + 1,
        ids.len() + n + 4
    );

    eprintln!("cache-capacity invariance OK: {k} tokens and step-0 logits bit-identical");
    Ok(())
}

/// The reimplemented [`argmax`] must agree with the shipped private one.
///
/// `src/model_fuel/generate.rs::argmax` is private, so the harness has to carry
/// a copy — and a copy that has drifted would make L1 pin the wrong tie-break.
/// This checks the agreement through the only observable surface there is:
/// `generate_greedy`'s returned tokens versus argmax over the captured logits.
#[test]
#[ignore = "needs the TinyLlama checkpoint"]
fn harness_argmax_agrees_with_generate_greedy() -> anyhow::Result<()> {
    let Some(dir) = tinyllama_dir() else {
        eprintln!("skipping: no TinyLlama snapshot");
        return Ok(());
    };
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let ids: Vec<u32> = tok
        .encode("The capital of France is", true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();

    let loaded = load_llama_f32_from_dir(&dir)?;
    let (logits, harness_tokens) = capture_logits(&loaded, &ids, 4, Some(2))?;
    let impl_tokens = generate_greedy(&loaded, &ids, 4, Some(2))?;

    assert_eq!(
        harness_tokens, impl_tokens,
        "the harness's argmax disagrees with the shipped generate_greedy"
    );
    let derived: Vec<u32> = logits.iter().map(|l| argmax(l)).collect();
    assert_eq!(
        derived, impl_tokens,
        "argmax over the captured logits does not reproduce generate_greedy's tokens — the capture \
         is measuring something other than what the implementation decides on"
    );
    Ok(())
}

/// The tie-break itself, checkable without the model: strict `>` means the
/// LOWEST index wins a tie. If `generate.rs` ever changes to `>=`, tokens shift
/// at ties and this states the contract that would break.
#[test]
fn control_argmax_tie_break_is_lowest_index() {
    assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1, "ties must go to the lowest index");
    assert_eq!(argmax(&[f32::NEG_INFINITY; 4]), 0, "all -inf must yield index 0");
    // NaN never compares `>`, so it can never win — matching generate.rs.
    assert_eq!(argmax(&[f32::NAN, 1.0]), 1);
    assert_eq!(runner_up(&[1.0, 3.0, 3.0, 2.0], 1), 2);
    assert_eq!(runner_up(&[5.0], 0), 0, "a one-element row has no runner-up");
}
