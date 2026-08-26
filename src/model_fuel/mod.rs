//! The Fuel-backed model path — Lightbulb's tensor core, ported off Candle.
//!
//! **This module is under construction and coexists with `crate::model`**, which
//! stays frozen on `candlelight` until this path reaches parity. See
//! `docs/superpowers/specs/2026-07-28-fuel-port-design.md` (decision D1).
//!
//! # The three rules this module is built to
//!
//! **1. Graph affinity.** Every `Tensor::from_*` constructor mints a *new*
//! graph, and tensors from different graphs cannot be combined — every binary op
//! asserts it. So there is exactly one root per graph and everything else is
//! built on it:
//!
//! ```text
//! let x = Tensor::from_f32(...);      // the root — activations
//! let w = x.const_f32_like(...);          // weights, on x's graph
//! let y = x.matmul(&w);                   // ✓
//! ```
//!
//! In model terms: **the activation tensor is the root and the weights are
//! `const_*_like` off it.** Getting this wrong is a runtime panic, not a compile
//! error, so it is a rule rather than a type.
//!
//! **2. F32 weights.** Fuel's CPU backend has no `[F32, BF16, F32]` matmul
//! kernel. The optimizer will now insert a promoting cast automatically, but that
//! cast is *value-lossless and not accumulation-preserving* — it changes numerics
//! versus a native mixed kernel. Loading F32 keeps the key at `[F32, F32, F32]`,
//! which is natively supported, so **no fixup runs and the oracle's goldens are
//! captured on a path the pass never touches.**
//!
//! **Candle→Fuel migration hazard, hit immediately:** Fuel's `Device` is a
//! **struct with `Device::cpu()`**, not an enum with a `Device::Cpu` variant.
//! Candle code ports across almost verbatim right up to this, and the error is
//! `E0599 no associated function or constant named 'Cpu'`. Fuel's own
//! `fuel-inference` carried this same breakage (in `prefix_cache.rs` /
//! `speculative.rs`) unnoticed since the candle→fuel rename, so it is a
//! well-trodden hole rather than a one-off.
//!
//! **3. Capture-shaped decode.** Requires a stable graph: no per-token graph
//! rebuilding, no host-side branching inside the step, runtime-offset KV writes.
//! Cheap to design in, painful to retrofit — so it is designed in from the first
//! commit.
//!
//! **Measured: capture is worth ~4×.** Lightbulb measured capture's
//! contribution directly on 2026-08-06 (RTX 4070 Laptop, TinyLlama-1.1B f32,
//! release, Fuel `8771997e`, k=1) with **persistence held constant on both
//! arms** — `forward_with_kv_context_captured` against `forward_decode_step`:
//!
//! | | |
//! | --- | --- |
//! | steady ms/token | 95.46 → **25.89** |
//! | speedup | **~3.7×** here (paired n=3: 4.34 / 3.69 / 3.05); **4.28×** measured independently by Fuel on the same card |
//! | one-time capture build | 132–178 ms, on token 2 — **excluded from the steady window** |
//! | kernel launches | 36,718 → 9,214 |
//! | host↔device bytes | **unchanged, identical to the digit** |
//! | device memsets | 14,368 → 2,811 (363.7 → 280.2 MB) |
//!
//! **Two independent instruments agree, which is why this number is trusted.**
//! Fuel measured the captured replay cost at 25.87 ms/token; Lightbulb's median
//! is 25.891 — **0.08% apart**, on the same card, from separate harnesses. The
//! residual spread between 3.7× and 4.28× lives entirely in the noisier
//! *no-capture* baseline (ours 79.7–111.2, Fuel's 111.77), not in the quantity
//! capture actually changes.
//!
//! **An earlier revision of this rule said ~2×. That was a measurement error,
//! not a re-interpretation.** `forward_with_kv_context_captured`
//! (`fuel-core/src/lazy.rs:9768`) has *two* one-time build steps: case 2 builds
//! the decode session on token 1, and case 3 — *"Second decode token: build the
//! capture"* — builds the CUDA graph on token 2. The harness's steady window
//! started at token 2, so a 132–178 ms one-time build was averaged into thirteen
//! ~26 ms tokens. It inflated the **captured arm only**, so it did not cancel in
//! the ratio; it roughly halved the apparent speedup. Fuel found it, the
//! arithmetic confirmed it before it was accepted, and the harness now measures
//! that token in its own fourth window. The stale figures (1.83× ratio-of-
//! medians, 2.06× median-of-paired-ratios) are in git history and are wrong.
//!
//! Capture is therefore launch-overhead plus memset elimination worth ~4×, still
//! short of the 10.4× that originally justified this rule. The likely remaining
//! gap: a 10.4× figure almost certainly compares capture against a **re-planning**
//! baseline, bundling persistence with capture — the same confound that made
//! Lightbulb's own 223× figure uninterpretable. Persistence is now the default on
//! both Fuel decode routes, so that portion is not capture's to claim.
//!
//! **The rule stands, and more comfortably than the previous revision claimed.**
//! ~4× is worth having outright, it costs nothing to design in, and retrofitting
//! it is expensive. The justification shrank from 10.4× — it did not evaporate.
//!
//! # What this is not yet
//!
//! A working model. This module currently proves only that Fuel is usable from
//! inside Lightbulb alongside `candlelight`. Everything else is ahead.

pub mod batched;
pub mod decoder;
pub mod device;
pub mod engine_model;
pub mod generate;
pub mod loader;
pub mod loader_f32;
pub mod policies;
pub mod session;

/// Smoke check: build a two-tensor graph on Fuel and realize it, from inside
/// Lightbulb's own crate.
///
/// Deliberately the smallest possible thing. It answers one question — *can this
/// crate link and drive Fuel at all, with candlelight also present?* — before any
/// model code depends on the answer. Two large ML frameworks in one binary is the
/// kind of arrangement that fails at link time or dependency resolution rather
/// than at runtime, and finding that out here is cheap.
///
/// Returns the realized product so a caller can assert on real values rather than
/// on the absence of a panic.
pub fn smoke_matmul() -> Vec<f32> {
    use fuel::Device;
    use fuel::lazy::Tensor;

    let dev = Device::cpu();

    // a = [[1, 2, 3],
    //      [4, 5, 6]]                    — the ROOT; owns the graph.
    let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2usize, 3usize), &dev);

    // w = [[1, 0],
    //      [0, 1],
    //      [1, 1]]                       — built on a's graph, per rule 1.
    // NOT `Tensor::from_f32(..)`, which would mint a second graph and panic
    // at the matmul.
    let w = a.const_f32_like(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], (3usize, 2usize));

    // a @ w = [[1+3, 2+3], [4+6, 5+6]] = [[4, 5], [10, 11]]
    let y = a.matmul(&w).expect("matmul: graph build");
    y.realize_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fuel links, builds a graph, and realizes it from inside Lightbulb —
    /// with candlelight in the same binary.
    #[test]
    fn fuel_is_usable_from_lightbulb() {
        let got = smoke_matmul();
        assert_eq!(
            got,
            vec![4.0, 5.0, 10.0, 11.0],
            "Fuel realized the wrong values — the graph is wrong, not the linkage"
        );
    }

    /// Pins rule 1 as an executable fact rather than a comment, so a future
    /// refactor that "simplifies" `const_f32_like` into `from_f32` fails here
    /// with an explanation instead of panicking somewhere downstream.
    #[test]
    #[should_panic(expected = "same graph")]
    fn independently_constructed_tensors_cannot_be_combined() {
        use fuel::Device;
        use fuel::lazy::Tensor;

        let dev = Device::cpu();
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2usize, 3usize), &dev);
        // A SECOND graph — this is the mistake the module doc warns about.
        let w = Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], (3usize, 2usize), &dev);
        let _ = a.matmul(&w);
    }
}
