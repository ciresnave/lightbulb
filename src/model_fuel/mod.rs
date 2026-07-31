//! The Fuel-backed model path — Lightbulb's tensor core, ported off Candle.
//!
//! **This module is under construction and coexists with `crate::model`**, which
//! stays frozen on `candlelight` until this path reaches parity. See
//! `docs/superpowers/specs/2026-07-28-fuel-port-design.md` (decision D1).
//!
//! # The three rules this module is built to
//!
//! **1. Graph affinity.** Every `LazyTensor::from_*` constructor mints a *new*
//! graph, and tensors from different graphs cannot be combined — every binary op
//! asserts it. So there is exactly one root per graph and everything else is
//! built on it:
//!
//! ```text
//! let x = LazyTensor::from_f32(...);      // the root — activations
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
//! **3. Capture-shaped decode.** `CapturedRun` measured 10.4× on TinyLlama decode,
//! byte-exact, and requires a stable graph: no per-token graph rebuilding, no
//! host-side branching inside the step, runtime-offset KV writes. Cheap to design
//! in, painful to retrofit — so it is designed in from the first commit.
//!
//! # What this is not yet
//!
//! A working model. This module currently proves only that Fuel is usable from
//! inside Lightbulb alongside `candlelight`. Everything else is ahead.

pub mod generate;
pub mod loader;
pub mod loader_f32;
pub mod policies;

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
    use fuel::lazy::LazyTensor;
    use fuel::Device;

    let dev = Device::cpu();

    // a = [[1, 2, 3],
    //      [4, 5, 6]]                    — the ROOT; owns the graph.
    let a = LazyTensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2usize, 3usize), &dev);

    // w = [[1, 0],
    //      [0, 1],
    //      [1, 1]]                       — built on a's graph, per rule 1.
    // NOT `LazyTensor::from_f32(..)`, which would mint a second graph and panic
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
        use fuel::lazy::LazyTensor;
        use fuel::Device;

        let dev = Device::cpu();
        let a = LazyTensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2usize, 3usize), &dev);
        // A SECOND graph — this is the mistake the module doc warns about.
        let w = LazyTensor::from_f32(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], (3usize, 2usize), &dev);
        let _ = a.matmul(&w);
    }
}
