// Rust FFI bindings for Marlin CUDA kernels (AWQ quantization)
//
// PROVENANCE RESOLVED 2026-08-20 — ruled by CireSnave. Do not delete this
// note; it exists so the question is not re-derived from scratch.
//
// RULING: this `extern "C"` block is INTERFACE, not expression. Its form is
// dictated by the CUDA kernel ABI, so two authors binding the same kernel
// independently produce nearly the same file, and similarity is evidence of a
// shared constraint rather than of copying. No third-party copyright is owed
// and this crate's `MIT OR Apache-2.0` is truthful for this file.
//
// The measurements are kept because a future reader grepping history will find
// the old header's claims and re-derive all of this otherwise:
//
//   * The old header said "Based on candle-vllm's Marlin implementation" and
//     "License: Apache-2.0 OR MIT (dual-licensed like candle-vllm)". BOTH were
//     false. candle-vllm is MIT ONLY (spdx_id = MIT, single LICENSE file), and
//     it does not declare these symbols at all — its src/backend/gptq.rs:3-4
//     IMPORTS all six from `attention_rs`. The header named a PEER, not a
//     parent: another consumer of the same upstream, which is provenance no
//     "does the URL resolve" check could ever falsify.
//
//   * The actual declaring project is guoqingbao/attention.rs,
//     src/kernels/src/ffi.rs — `license = "MIT"` in Cargo.toml, no LICENSE
//     file in the repo (hence GitHub reporting `license: null`).
//
//   * ONE MEASUREMENT SITS IN TENSION WITH THE RULING'S STATED REASON, and is
//     recorded rather than dropped. Parameter NAMES are not fixed by the ABI.
//     The CUDA entry point in marlin_matmul_f16.cu names them
//     `A, B, scales, zeros, g_idx, C, prob_m, prob_k, prob_n, workspace,
//     groupsize, stream`. attention.rs's Rust renames six of those
//     (A→inputs, B→weight, C→out, prob_m/k/n→m/k/n), and this file carries all
//     six renames identically. So the TYPES and ORDER are ABI-mandated, as the
//     ruling says; the NAMES were an upstream author's choice and match.
//     TWO SEPARATE QUESTIONS, and they are not in conflict — read them apart
//     or this note looks self-contradictory:
//       (1) WAS it derived?  Yes. Six matching renames is not convergence.
//       (2) Was what was derived PROTECTABLE?  The ruling says no: the
//           signature is ABI-dictated, and the identifiers are functionally
//           inert labels on a dictated signature — they affect neither
//           linkage, calling convention, nor codegen, which puts them nearer
//           to comments than to code.
//     "We can show we copied it" and "what we copied was protectable" are
//     different claims. The evidence answers the first; the ruling answers the
//     second. So the ruling stands — but "can only be built one way" is exactly
//     true of the signature and NOT of the identifiers, and that distinction
//     must not be lost if this is ever cited as precedent for a file that is
//     not an ABI binding.
//
// SCOPE: this ruling covers ABI-mandated bindings ONLY. It does NOT extend to
// the callers — src/backend/marlin.rs, src/model/awq_qwen3.rs,
// src/loaders/awq.rs — which choose their own error handling, dispatch and
// naming, and which were audited separately (see the commit that added this).

// ⚠️ NOTHING IN THIS TREE CAN PROVIDE THESE SYMBOLS.
//
// The CUDA sources that defined them lived in `kernels/` and were deleted on
// 2026-08-20 as unattributed verbatim third-party code (see that commit, and
// build.rs). Enabling `feature = "cuda"` and reaching this path would now fail
// at LINK time rather than at compile time.
//
// In practice nothing changes: `build.rs`'s `compile_cuda_kernels()` was always
// a stub, so these symbols were never provided even when the sources were here.
// Stated explicitly so the next reader meets the fact in the file instead of in
// a linker error.
//
// Fuel does NOT yet cover this. Measured 2026-08-20 by Fuel's architect at
// their `origin/main`: `fuel-cuda-backend/src/baracuda/quant_w4a16.rs` is 747
// lines with ZERO tests and ZERO call sites — never invoked. The `marlin` and
// `awq` cargo features its doc comment names do not exist, and `enum
// QuantFormat` has no definition anywhere in their workspace. Their summary:
// "the kernels are dead code with a `pub mod` in front of them." It is not
// scheduled. And GAP-190 (`fuel-ir/src/stype.rs:340`) is a design blocker
// upstream of the wiring: `AffineBlock` cannot distinguish a CODEBOOK quant
// family from a LINEAR one, so Marlin and NF4 collapse onto one descriptor —
// "there is no scale for which a linear dequant reproduces NF4."
//
// So this scaffolding is kept deliberately, not by neglect: it is our own code,
// it carries no licence exposure now that `kernels/` is gone, and it retires
// with candlelight along with the rest of that path.

use core::ffi::{c_int, c_void};

#[allow(dead_code)]
extern "C" {
    /// Marlin 4-bit FP16 matrix multiplication (GPTQ)
    /// 
    /// Performs quantized matrix multiplication: out = inputs @ weight
    /// where weight is in 4-bit GPTQ format.
    ///
    /// # Arguments
    /// * `inputs` - Input tensor (FP16), shape [m, k]
    /// * `weight` - Quantized weight tensor (4-bit), shape [k, n]
    /// * `scales` - Quantization scales (FP16), shape [k/groupsize, n]
    /// * `zeros` - Zero-points (FP16), shape [k/groupsize, n]
    /// * `g_idx` - Group indices for each column (optional, can be null)
    /// * `out` - Output tensor (FP16), shape [m, n]
    /// * `m` - Number of rows in inputs
    /// * `k` - Number of columns in inputs (rows in weight)
    /// * `n` - Number of columns in weight
    /// * `workspace` - Temporary workspace buffer
    /// * `groupsize` - Quantization group size (typically 128)
    /// * `stream` - CUDA stream handle
    pub fn marlin_4bit_f16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );

    /// Marlin 4-bit BF16 matrix multiplication (GPTQ)
    /// 
    /// Same as marlin_4bit_f16 but with BF16 precision.
    pub fn marlin_4bit_bf16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );

    /// Marlin AWQ 4-bit FP16 matrix multiplication
    /// 
    /// Performs quantized matrix multiplication optimized for AWQ format.
    /// AWQ uses activation-aware weight quantization for better accuracy.
    ///
    /// # Arguments
    /// Same as marlin_4bit_f16 but optimized for AWQ quantization scheme.
    pub fn marlin_awq_4bit_f16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );

    /// Marlin AWQ 4-bit BF16 matrix multiplication
    /// 
    /// Same as marlin_awq_4bit_f16 but with BF16 precision.
    pub fn marlin_awq_4bit_bf16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );

    /// Repack weights from GPTQ format to Marlin format
    /// 
    /// Converts GPTQ-quantized weights into Marlin's internal format
    /// for efficient computation.
    ///
    /// # Arguments
    /// * `weight` - Input weight tensor (GPTQ format)
    /// * `result` - Output weight tensor (Marlin format)
    /// * `m` - Number of rows
    /// * `n` - Number of columns
    /// * `stream` - CUDA stream handle
    pub fn gptq_repack(
        weight: *const c_void,
        result: *const c_void,
        m: c_int,
        n: c_int,
        stream: i64,
    );

    /// Repack weights from AWQ format to Marlin format
    /// 
    /// Converts AWQ-quantized weights into Marlin's internal format.
    ///
    /// # Arguments
    /// * `weight` - Input weight tensor (AWQ format)
    /// * `result` - Output weight tensor (Marlin format)
    /// * `k` - Input dimension
    /// * `n` - Output dimension
    /// * `bits` - Bit precision (typically 4)
    /// * `stream` - CUDA stream handle
    pub fn awq_repack(
        weight: *const c_void,
        result: *const c_void,
        k: c_int,
        n: c_int,
        bits: c_int,
        stream: i64,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires CUDA runtime
    fn test_ffi_declarations_compile() {
        // This test just ensures FFI declarations compile correctly
        // Actual kernel testing requires CUDA device
    }
}
