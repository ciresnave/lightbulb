// Rust FFI bindings for Marlin CUDA kernels (AWQ quantization)
// Based on candle-vllm's Marlin implementation
// License: Apache-2.0 OR MIT (dual-licensed like candle-vllm)

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
