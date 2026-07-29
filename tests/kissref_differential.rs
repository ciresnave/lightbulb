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
use kiss_ref_core::{diff_f32, reference_f32, Tolerance};

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

    // SiLU is a transcendental (sigmoid), so exact bit-equality is not the right
    // bar across two independent implementations. A tight ULP bound still
    // catches a wrong formula, a wrong branch, or a sign error.
    assert_matches_reference(Op::Silu, &inputs, &candidate, Tolerance::Ulp(4));
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

    assert_matches_reference(Op::Sigmoid, &inputs, &candidate, Tolerance::Ulp(4));
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
