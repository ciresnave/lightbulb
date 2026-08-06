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

use candlelight::core::{Device, Tensor};
use kiss_ops_vocab::Op;
use kiss_ref_core::{
    diff_f32, eval_recipe, reference_f32, DetClass, FlatDag, Monoid, Node, Tolerance,
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

/// SiLU (x * sigmoid(x)) is the activation in Lightbulb's MLP path
/// (`candlelight::nn::ops::silu`), so this is a real code path rather than a
/// synthetic one.
#[test]
fn silu_matches_kiss_ref() -> anyhow::Result<()> {
    let inputs = activation_probe_values();
    let device = Device::Cpu;

    // Lightbulb's actual path: the same `silu` its MLP layers call.
    let t = Tensor::new(inputs.as_slice(), &device)?;
    let candidate = candlelight::nn::ops::silu(&t)?.to_vec1::<f32>()?;

    assert_eq!(candidate.len(), inputs.len(), "silu changed element count");

    // SiLU declares no §6.8 ceiling (it is a non-primitive), so the bound is
    // DERIVED from the transcendental it decomposes through: sigmoid → exp.
    // Not a guessed 4 — exp's own declared ceiling.
    assert_matches_reference(Op::Silu, &inputs, &candidate, tolerance_via(Op::Exp));
    Ok(())
}

/// Sigmoid on its own — isolates whether any SiLU disagreement comes from the
/// sigmoid or from the multiply.
#[test]
fn sigmoid_matches_kiss_ref() -> anyhow::Result<()> {
    let inputs = activation_probe_values();
    let device = Device::Cpu;

    let t = Tensor::new(inputs.as_slice(), &device)?;
    // sigmoid(x) = 1 / (1 + exp(-x)), via Candle's own ops.
    let candidate = candlelight::nn::ops::sigmoid(&t)?.to_vec1::<f32>()?;

    // Same derivation: sigmoid(x) = 1/(1+exp(-x)), so exp grounds the bound.
    assert_matches_reference(Op::Sigmoid, &inputs, &candidate, tolerance_via(Op::Exp));
    Ok(())
}

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
            Node::Apply { op: Op::Mul, children: vec![0, 0] },
            Node::Reduce { monoid: Monoid::Sum, axes: vec![1], keepdim: true, child: 1 },
            Node::ReducedCount(vec![1]),
            Node::Apply { op: Op::Div, children: vec![2, 3] },
            Node::RuntimeScalar(0),
            Node::Apply { op: Op::Add, children: vec![4, 5] },
            Node::Apply { op: Op::Sqrt, children: vec![6] },
            Node::Apply { op: Op::Div, children: vec![0, 7] },
            Node::Bind(1),
            Node::Apply { op: Op::Mul, children: vec![8, 9] },
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
        1.0, -1.0, 0.5, -0.0, 2.0, -3.5, 0.25, 4.0,
        // row 1 — a subnormal and a wide dynamic range
        f32::MIN_POSITIVE / 2.0, 1e-6, 1e3, -1e3, 0.125, -0.125, 7.5, -0.75,
    ];
    (x, rows, d)
}

#[test]
fn rmsnorm_fragment_matches_kiss_ref_recipe() -> anyhow::Result<()> {
    use lightbulb::model::fused_rmsnorm::FusedRmsNorm;

    let (x_data, rows, d) = rmsnorm_probe_input();
    let w_data: Vec<f32> = vec![1.0, 0.5, 2.0, -1.0, 0.25, 1.5, -0.5, 3.0];
    let eps = 1e-5_f64;
    let device = Device::Cpu;

    // ---- Lightbulb: its actual RMSNorm path ----
    let x = Tensor::from_vec(x_data.clone(), (rows, d), &device)?;
    let w = Tensor::from_vec(w_data.clone(), d, &device)?;
    let norm = FusedRmsNorm::new_with_weight(w, eps);
    let candidate = norm.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;

    // ---- kiss-ref: the same fragment as a recipe ----
    let dag = rmsnorm_recipe();
    let kx = kiss_ref_core::Tensor::from_vec(x_data, &[rows, d])
        .map_err(|e| anyhow::anyhow!("kiss-ref input tensor: {e:?}"))?;
    let kw = kiss_ref_core::Tensor::from_vec(w_data, &[d])
        .map_err(|e| anyhow::anyhow!("kiss-ref weight tensor: {e:?}"))?;
    let evaluated = eval_recipe::<f32>(&dag, &[kx, kw], &[eps as f32], &[])
        .map_err(|e| anyhow::anyhow!("eval_recipe: {e:?}"))?;
    let reference = evaluated.outputs[0].as_slice();

    assert_eq!(
        candidate.len(),
        reference.len(),
        "element count differs — the recipe's shape does not match Lightbulb's output"
    );

    // The root carries a Sum reduction, so it is nondeterministic by kiss-ref's
    // own classification: tolerance-compare, never bit-compare.
    let root_det = evaluated.dets[10];
    assert!(
        !matches!(root_det, DetClass::ExactByte),
        "root unexpectedly ExactByte ({root_det:?}) — if kiss-ref now classifies \
         this fragment as exact, the comparison below is weaker than it could be"
    );

    let mut worst = 0.0_f32;
    let mut detail = String::new();
    for (i, (&r, &c)) in reference.iter().zip(&candidate).enumerate() {
        let denom = r.abs().max(1e-30);
        let rel = (r - c).abs() / denom;
        if rel > worst {
            worst = rel;
        }
        if rel > 1e-5 {
            detail.push_str(&format!("\n  [{i}] kiss-ref={r:e}  lightbulb={c:e}  rel={rel:e}"));
        }
    }

    assert!(
        detail.is_empty(),
        "RMSNorm fragment disagrees with the kiss-ref recipe (worst rel {worst:e}).\n\
         kiss-ref is a differential target, not a verdict — determine which side is wrong.{detail}"
    );
    Ok(())
}

/// Control for the recipe path. A recipe that silently evaluated to the wrong
/// shape, or a comparison that never actually compared, would make the test
/// above pass for the wrong reason.
#[test]
fn recipe_harness_detects_a_wrong_candidate() -> anyhow::Result<()> {
    let (x_data, rows, d) = rmsnorm_probe_input();
    let w_data: Vec<f32> = vec![1.0; 8];

    let dag = rmsnorm_recipe();
    let kx = kiss_ref_core::Tensor::from_vec(x_data, &[rows, d])
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let kw = kiss_ref_core::Tensor::from_vec(w_data, &[d])
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let evaluated = eval_recipe::<f32>(&dag, &[kx, kw], &[1e-5], &[])
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
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
    let device = Device::Cpu;

    let t = Tensor::new(inputs.as_slice(), &device)?;
    let mut candidate = candlelight::nn::ops::silu(&t)?.to_vec1::<f32>()?;

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
    // through, which is what the SiLU/sigmoid tests above actually use.
    assert_eq!(tolerance_via(Op::Exp), Tolerance::Ulp(4));
}

/// NaN as a **structural tracer**: which inputs influence which outputs?
///
/// This is the sharpest tier-1 test in the file, and it checks something the
/// value comparison cannot. RMSNorm reduces over the feature axis *within a
/// row*, so a NaN in row 0 must contaminate **exactly row 0's outputs** and
/// leave row 1 untouched. If either implementation reduced over the wrong axis,
/// or broadcast the reciprocal-RMS across the wrong dimension, the NaN would
/// spread to a different set of positions — while ordinary finite inputs would
/// still produce plausible-looking numbers that a tolerance check might pass.
///
/// A wrong-axis reduction is precisely the graph-construction bug tier 1 exists
/// to catch, and NaN propagation makes it visible as a *pattern* rather than a
/// magnitude.
#[test]
fn nan_contamination_pattern_matches_kiss_ref() -> anyhow::Result<()> {
    use lightbulb::model::fused_rmsnorm::FusedRmsNorm;

    let (mut x_data, rows, d) = rmsnorm_probe_input();
    // One NaN, in row 0 only.
    x_data[2] = f32::NAN;

    let w_data: Vec<f32> = vec![1.0, 0.5, 2.0, -1.0, 0.25, 1.5, -0.5, 3.0];
    let eps = 1e-5_f64;
    let device = Device::Cpu;

    let x = Tensor::from_vec(x_data.clone(), (rows, d), &device)?;
    let w = Tensor::from_vec(w_data.clone(), d, &device)?;
    let candidate = FusedRmsNorm::new_with_weight(w, eps)
        .forward(&x)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let dag = rmsnorm_recipe();
    let kx = kiss_ref_core::Tensor::from_vec(x_data, &[rows, d])
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let kw = kiss_ref_core::Tensor::from_vec(w_data, &[d])
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let evaluated = eval_recipe::<f32>(&dag, &[kx, kw], &[eps as f32], &[])
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let reference = evaluated.outputs[0].as_slice();

    let cand_nan: Vec<bool> = candidate.iter().map(|v| v.is_nan()).collect();
    let ref_nan: Vec<bool> = reference.iter().map(|v| v.is_nan()).collect();

    assert_eq!(
        cand_nan, ref_nan,
        "NaN contamination pattern differs — one side reduces or broadcasts over a          different axis.
  lightbulb: {cand_nan:?}
  kiss-ref:  {ref_nan:?}"
    );

    // And pin the expectation itself, so agreeing-but-both-wrong is caught:
    // row 0 fully contaminated, row 1 entirely clean.
    let expected: Vec<bool> = (0..rows * d).map(|i| i < d).collect();
    assert_eq!(
        cand_nan, expected,
        "both implementations agree, but not with RMSNorm's semantics — a NaN in          row 0 must contaminate exactly row 0 (per-row feature reduction)"
    );
    Ok(())
}
