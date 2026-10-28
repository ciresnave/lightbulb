// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tier-2 oracle: differential tests against kiss-ref.
//!
//! kiss-ref is a **spec-exact reference implementation** of the KISS base-op
//! vocabulary — deliberately naive, correctness-only. It is an *independent*
//! target: it shares no code, and therefore no bugs, with Candle, Fuel, or
//! Lightbulb. That independence is what the port's oracle rests on.
//!
//! **kiss-ref is a differential TARGET, never a verdict source.** A disagreement
//! means "determine which of us is wrong" — not "kiss-ref is right by definition".
//!
//! ## Why f32 only
//!
//! kiss-ref's float reductions accumulate at *storage* precision in ascending
//! index order. A GPU kernel accumulating in f32 over bf16 storage will differ
//! **structurally, not in ULPs** (33% measured on an FP8 reduce upstream). So all
//! differentials here run in f32, which is also the dtype the port loads weights
//! at. For a narrow lane, use `reference_reduce_acc` / `reference_matmul_acc`,
//! which take the accumulator dtype explicitly.
//!
//! ## Scope of this file today
//!
//! Scalar/elementwise ops only — `reference_f32` takes one argument-tuple row per
//! evaluation and errors on structural or tensor ops. Tensor and subgraph
//! (`eval_recipe`) differentials are the tier-1 work and land separately.

use kiss_ops_vocab::Op;
use kiss_ref_core::{
    DetClass, FlatDag, Monoid, Node, Tolerance, diff_f32, eval_recipe, reference_f32,
};

/// Inputs chosen to span the interesting regions of an activation, not just the
/// easy middle: signed zeros, a subnormal, values either side of the SiLU
/// minimum (~-1.278), and magnitudes large enough to saturate the sigmoid.
fn activation_probe_values() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        f32::MIN_POSITIVE / 2.0, // subnormal
        1e-7,
        -1e-7,
        0.5,
        -0.5,
        1.0,
        -1.0,
        -1.278_464_5, // near SiLU's minimum
        2.0,
        -2.0,
        10.0,
        -10.0,
        88.0,  // sigmoid saturates to 1
        -88.0, // sigmoid saturates to 0
    ]
}

/// The tolerance kiss-ref *declares* for `op`, rather than one we guessed.
///
/// `Op::ulp_ceiling()` returns the §6.8 bound where the spec defines one
/// (`Sqrt` 2.0, `Exp`/`Log`/`Sin`/`Cos`/`Atan`/`Erf`/`Atan2` 4.0, `Lgamma` 8.0)
/// and `None` otherwise.
///
/// **`None` does not always mean "exact".** For an exact atom it does. But a
/// *transcendental non-primitive* — `Silu`, `Sigmoid` — also returns `None`,
/// because the ceiling is a per-op spec bound and **does not compose**: the
/// spec declares no ceiling for an op it defines by decomposition. Demanding
/// bit-equality there would be asserting something the spec never promised.
///
/// So there are two honest cases, and callers must say which they are in:
///
/// - the op declares a ceiling → use it ([`declared_tolerance`]);
/// - the op decomposes through a transcendental atom → use *that atom's*
///   declared ceiling ([`tolerance_via`]), which is derived rather than guessed.
///
/// A bare `unwrap_or(Exact)` would be wrong for the second case: it asserts
/// bit-equality the spec never promised.
fn declared_tolerance(op: Op) -> Tolerance {
    match op.ulp_ceiling() {
        Some(ulp) => Tolerance::Ulp(ulp as u64),
        // Genuinely exact atoms have no declared ceiling because none is needed.
        None => Tolerance::Exact,
    }
}

/// Tolerance for an op with no declared ceiling, derived from the transcendental
/// atom it decomposes through. `atom` must itself declare one, or the derivation
/// is meaningless and this panics rather than silently guessing.
fn tolerance_via(atom: Op) -> Tolerance {
    match atom.ulp_ceiling() {
        Some(ulp) => Tolerance::Ulp(ulp as u64),
        None => panic!(
            "{atom:?} declares no ULP ceiling, so it cannot ground a derived bound — \
             pick an atom that does, or state a judgment bound explicitly"
        ),
    }
}

/// Run `op` through kiss-ref and through Lightbulb's Candle path, and compare.
///
/// `candidate` is produced by the caller so each test states plainly which
/// Lightbulb code path it is exercising.
fn assert_matches_reference(op: Op, inputs: &[f32], candidate: &[f32], tol: Tolerance) {
    // kiss-ref wants one argument-tuple per evaluation; these are all unary.
    let rows: Vec<[f32; 1]> = inputs.iter().map(|&x| [x]).collect();
    let row_refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();

    let report = diff_f32(op, &row_refs, candidate, tol)
        .unwrap_or_else(|e| panic!("kiss-ref diff for {op:?} failed: {e:?}"));

    if report.mismatches != 0 {
        // Surface the reference values too — a bare "3 mismatches" is not
        // actionable, and which side is wrong is genuinely open.
        let reference = reference_f32(op, &row_refs).expect("reference evaluation");
        let mut detail = String::new();
        for (i, ((&x, &r), &c)) in inputs.iter().zip(&reference).zip(candidate).enumerate() {
            if r.to_bits() != c.to_bits() {
                detail.push_str(&format!(
                    "\n  [{i}] input={x:e}  kiss-ref={r:e}  lightbulb={c:e}"
                ));
            }
        }
        panic!(
            "{op:?}: {} of {} values disagree (max {} ULP, tolerance {tol:?}).\n\
             kiss-ref is a differential target, not a verdict — investigate which side is wrong.{detail}",
            report.mismatches, report.n, report.max_ulp
        );
    }
}

// `silu_matches_kiss_ref`/`sigmoid_matches_kiss_ref` removed 2026-10-22
// (candlelight removal, CireSnave: "Remove all dependencies on Candle and
// Candlelight..."): both claimed to test "the real code path" Lightbulb's
// MLP layers call, via candlelight::nn::ops::{silu,sigmoid}. That stopped
// being true once ParallelModelManager (candlelight's serving path) was
// removed in the same pass -- the real path is now fuel::lazy::Tensor's
// `.silu()` (src/model_fuel/batched.rs:975), a lazy-eval API this test
// would need to learn to drive. Re-deriving these against Fuel's real
// activation path is real porting work, not a mechanical swap -- see
// docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md.

// ─────────────────────────────────────────────────────────────────────────────
// Tier 1: subgraph recipes.
//
// The tier that actually matters. Single-op differentials prove primitives;
// they cannot prove *composition*, and graph-construction bugs live in the
// composition. `eval_recipe` evaluates a DAG of KISS ops exactly, so it gives an
// independent reference for a graph *fragment*.
//
// Toy dimensions on purpose: graph-construction correctness is
// dimension-independent, so a 2x8 fragment proves the wiring while staying
// inside kiss-ref's performance envelope (it is scalar, per-element, and a
// production-sized matmul would take minutes).
// ─────────────────────────────────────────────────────────────────────────────

/// RMSNorm as a kiss-ref recipe:
///
/// ```text
///  0  Bind(0)                             x           [rows, d]
///  1  Apply{Mul, [0,0]}                   x²
///  2  Reduce{Sum, axes=[1], keepdim}      Σx²         [rows, 1]
///  3  ReducedCount([1])                   d
///  4  Apply{Div, [2,3]}                   mean(x²)
///  5  RuntimeScalar(0)                    eps
///  6  Apply{Add, [4,5]}                   mean(x²)+eps
///  7  Apply{Sqrt, [6]}                    rms         [rows, 1]
///  8  Apply{Div, [0,7]}                   x/rms       (broadcast over d)
///  9  Bind(1)                             weight      [d]
/// 10  Apply{Mul, [8,9]}                   x/rms * w   (broadcast over rows)
/// ```
///
/// Eleven nodes, two broadcasts, a keepdim reduction and a runtime scalar —
/// enough structure for a wiring bug to hide in, which is the point.
fn rmsnorm_recipe() -> FlatDag {
    FlatDag::new(
        vec![
            Node::Bind(0),
            Node::Apply {
                op: Op::Mul,
                children: vec![0, 0],
            },
            Node::Reduce {
                monoid: Monoid::Sum,
                axes: vec![1],
                keepdim: true,
                child: 1,
            },
            Node::ReducedCount(vec![1]),
            Node::Apply {
                op: Op::Div,
                children: vec![2, 3],
            },
            Node::RuntimeScalar(0),
            Node::Apply {
                op: Op::Add,
                children: vec![4, 5],
            },
            Node::Apply {
                op: Op::Sqrt,
                children: vec![6],
            },
            Node::Apply {
                op: Op::Div,
                children: vec![0, 7],
            },
            Node::Bind(1),
            Node::Apply {
                op: Op::Mul,
                children: vec![8, 9],
            },
        ],
        vec![10],
    )
}

/// Toy-dimension input with the numeric texture that matters: a signed zero, a
/// subnormal, and values spanning several orders of magnitude. Construction
/// bugs and numeric-handling bugs surface on *different* inputs, and toy scale
/// makes it affordable to carry both.
///
/// (Inputs arrive via `Tensor::from_vec(Vec<f32>)` on both sides, so they are
/// already bit-exact — `Node::ConstBits` matters for pathological *constant
/// leaves inside* a DAG, which this recipe has none of. An earlier note here
/// claimed ConstBits was needed for these inputs; that was wrong.)
fn rmsnorm_probe_input() -> (Vec<f32>, usize, usize) {
    let rows = 2;
    let d = 8;
    let x = vec![
        // row 0 — ordinary magnitudes plus a signed zero
        1.0,
        -1.0,
        0.5,
        -0.0,
        2.0,
        -3.5,
        0.25,
        4.0,
        // row 1 — a subnormal and a wide dynamic range
        f32::MIN_POSITIVE / 2.0,
        1e-6,
        1e3,
        -1e3,
        0.125,
        -0.125,
        7.5,
        -0.75,
    ];
    (x, rows, d)
}

// `rmsnorm_fragment_matches_kiss_ref_recipe` removed 2026-10-22
// (candlelight removal): same reason, same file. See
// docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md.

/// Control for the recipe path. A recipe that silently evaluated to the wrong
/// shape, or a comparison that never actually compared, would make a
/// recipe-vs-candlelight comparison pass for the wrong reason -- that
/// comparison (`rmsnorm_fragment_matches_kiss_ref_recipe`) was removed
/// 2026-10-22 (candlelight removal), but this control, which checks the
/// kiss-ref recipe path on its own, still has independent value.
#[test]
fn recipe_harness_detects_a_wrong_candidate() -> anyhow::Result<()> {
    let (x_data, rows, d) = rmsnorm_probe_input();
    let w_data: Vec<f32> = vec![1.0; 8];

    let dag = rmsnorm_recipe();
    let kx = kiss_ref_core::Tensor::from_vec(x_data, &[rows, d])
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let kw = kiss_ref_core::Tensor::from_vec(w_data, &[d]).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let evaluated =
        eval_recipe::<f32>(&dag, &[kx, kw], &[1e-5], &[]).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let reference = evaluated.outputs[0].as_slice();

    assert_eq!(reference.len(), rows * d, "recipe produced the wrong shape");

    // A deliberately wrong candidate must be caught by the same comparison the
    // real test uses.
    let mut wrong = reference.to_vec();
    wrong[3] += 1.0;
    let disagreements = reference
        .iter()
        .zip(&wrong)
        .filter(|&(&r, &c)| (r - c).abs() / r.abs().max(1e-30) > 1e-5)
        .count();

    assert!(
        disagreements >= 1,
        "recipe comparison failed to detect a corrupted candidate — \
         a passing recipe differential would be meaningless"
    );
    Ok(())
}

/// The harness itself must fail when the candidate is wrong, or a green run
/// proves nothing. Deliberately corrupts one value and asserts the diff catches
/// it — the control for this test file.
#[test]
fn harness_detects_a_wrong_candidate() -> anyhow::Result<()> {
    let inputs = activation_probe_values();

    // A plain f32 computation of SiLU stands in for "a real candidate" here --
    // this test's whole point is that the HARNESS catches corruption, not that
    // this value matches anything Lightbulb runs. silu(x) = x * sigmoid(x).
    let mut candidate: Vec<f32> = inputs.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    // Perturb one entry well beyond any plausible ULP tolerance.
    candidate[5] += 1.0;

    let rows: Vec<[f32; 1]> = inputs.iter().map(|&x| [x]).collect();
    let row_refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
    let report = diff_f32(Op::Silu, &row_refs, &candidate, Tolerance::Ulp(4))
        .expect("diff should evaluate even when values disagree");

    assert!(
        report.mismatches >= 1,
        "harness failed to detect a deliberately corrupted candidate — \
         a passing differential would be meaningless"
    );
    Ok(())
}

/// What does the spec actually declare for the ops this file tests?
///
/// Not an assertion about Lightbulb — an assertion about our *understanding* of
/// kiss-ref's tolerance model, so a change in it is caught here rather than
/// silently loosening or tightening every differential above.
#[test]
fn declared_ceilings_are_what_we_think() {
    // Declared §6.8 bounds.
    assert_eq!(Op::Sqrt.ulp_ceiling(), Some(2.0));
    assert_eq!(Op::Exp.ulp_ceiling(), Some(4.0));

    // Transcendental NON-primitives: no declared ceiling, because the bound is
    // per-op and does not compose. `None` here means "the spec makes no promise",
    // NOT "bit-exact".
    assert_eq!(Op::Silu.ulp_ceiling(), None);
    assert_eq!(Op::Sigmoid.ulp_ceiling(), None);

    // declared_tolerance() is for ops that declare a ceiling.
    assert_eq!(declared_tolerance(Op::Sqrt), Tolerance::Ulp(2));
    // For the non-primitives the bound is derived from the atom they decompose
    // through (e.g. SiLU/sigmoid via exp).
    assert_eq!(tolerance_via(Op::Exp), Tolerance::Ulp(4));
}

// `nan_contamination_pattern_matches_kiss_ref` (NaN as a structural tracer: which inputs influence which outputs) removed 2026-10-22
// (candlelight removal): used the now-deleted `lightbulb::model::
// fused_rmsnorm::FusedRmsNorm`. Re-derive against Fuel's RMSNorm
// (`model_fuel::batched::affine_rms_norm`) if this NaN-propagation
// structural check is still wanted -- see
// docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md.
