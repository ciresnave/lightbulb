// Marlin matrix multiplication CustomOp for AWQ quantization
//
// This module provides Candle CustomOp wrappers around the Marlin CUDA kernels
// for efficient 4-bit quantized inference.

use candlelight::core::{CpuStorage, CustomOp1, CustomOp3, DType, Result, Shape, WithDType};

#[cfg(feature = "cuda")]
use candlelight::core::cuda::{CudaStorage, CudaStorageSlice};

/// Quantization format for weight compression
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    /// GPTQ: General-purpose quantization
    GPTQ,
    /// AWQ: Activation-aware weight quantization (better accuracy)
    AWQ,
}

/// Precision for matrix multiplication
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// FP16 (half precision)
    F16,
    /// BF16 (bfloat16)
    BF16,
}

impl Precision {
    pub fn dtype(&self) -> DType {
        match self {
            Precision::F16 => DType::F16,
            Precision::BF16 => DType::BF16,
        }
    }
}

/// Marlin matrix multiplication CustomOp
///
/// Performs quantized matrix multiplication: out = inputs @ weights
/// where weights are in 4-bit GPTQ/AWQ format.
///
/// # Arguments
/// * `inputs` - Input activations (FP16 or BF16), shape [batch, in_features]
/// * `weights` - Quantized weights (4-bit), shape [in_features, out_features]
/// * `scales` - Quantization scales, shape [in_features/group_size, out_features]
///
/// # Returns
/// Output tensor (FP16 or BF16), shape [batch, out_features]
pub struct MarlinMatMul {
    pub precision: Precision,
    pub quant_format: QuantFormat,
    pub group_size: usize,
}

impl MarlinMatMul {
    pub fn new(precision: Precision, quant_format: QuantFormat, group_size: usize) -> Self {
        Self {
            precision,
            quant_format,
            group_size,
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd_impl(
        &self,
        inputs: &CudaStorage,
        weights: &CudaStorage,
        scales: &CudaStorage,
        inputs_shape: &Shape,
        weights_shape: &Shape,
        scales_shape: &Shape,
    ) -> Result<(CudaStorage, Shape)> {
        use super::marlin_ffi;
        use std::ffi::c_int;

        // Validate shapes
        let (m, k) = match inputs_shape.dims() {
            [m, k] => (*m, *k),
            _ => candlelight::core::bail!("inputs must be 2D, got shape {:?}", inputs_shape),
        };

        let (k2, n) = match weights_shape.dims() {
            [k2, n] => (*k2, *n),
            _ => candlelight::core::bail!("weights must be 2D, got shape {:?}", weights_shape),
        };

        if k != k2 {
            candlelight::core::bail!("dimension mismatch: inputs k={}, weights k={}", k, k2);
        }

        // Validate scales shape: [k/group_size, n]
        let expected_scale_rows = k / self.group_size;
        match scales_shape.dims() {
            [scale_rows, scale_cols] => {
                if *scale_rows != expected_scale_rows {
                    candlelight::core::bail!(
                        "scales shape mismatch: expected [{}, {}], got [{}, {}]",
                        expected_scale_rows,
                        n,
                        scale_rows,
                        scale_cols
                    );
                }
                if *scale_cols != n {
                    candlelight::core::bail!(
                        "scales columns mismatch: expected {}, got {}",
                        n,
                        scale_cols
                    );
                }
            }
            _ => candlelight::core::bail!("scales must be 2D, got shape {:?}", scales_shape),
        }

        // Allocate output storage
        let out_shape = Shape::from_dims(&[m, n]);
        let elem_count = m * n;
        let out = unsafe {
            CudaStorage::alloc(self.precision.dtype(), elem_count, inputs.device().clone())?
        };

        // Allocate workspace (Marlin requires temporary workspace)
        // Size: typically 16 * n elements
        let workspace_size = 16 * n;
        let workspace =
            unsafe { CudaStorage::alloc(DType::U8, workspace_size, inputs.device().clone())? };

        // Get raw pointers
        let inputs_ptr = inputs.as_cuda_slice::<f16>()?.device_ptr() as *const std::ffi::c_void;
        let weights_ptr = weights.as_cuda_slice::<i32>()?.device_ptr() as *const c_int;
        let scales_ptr = scales.as_cuda_slice::<f16>()?.device_ptr() as *const std::ffi::c_void;
        let out_ptr = out.as_cuda_slice::<f16>()?.device_ptr() as *mut std::ffi::c_void;
        let workspace_ptr =
            workspace.as_cuda_slice::<u8>()?.device_ptr() as *const std::ffi::c_void;

        // Get CUDA stream
        let stream = inputs.device().cuda_stream() as i64;

        // Call appropriate kernel based on format and precision
        unsafe {
            match (self.quant_format, self.precision) {
                (QuantFormat::GPTQ, Precision::F16) => {
                    marlin_ffi::marlin_4bit_f16(
                        inputs_ptr,
                        weights_ptr,
                        scales_ptr,
                        std::ptr::null(), // zeros (optional for GPTQ)
                        std::ptr::null(), // g_idx (optional)
                        out_ptr,
                        m as c_int,
                        k as c_int,
                        n as c_int,
                        workspace_ptr,
                        self.group_size as c_int,
                        stream,
                    );
                }
                (QuantFormat::GPTQ, Precision::BF16) => {
                    marlin_ffi::marlin_4bit_bf16(
                        inputs_ptr,
                        weights_ptr,
                        scales_ptr,
                        std::ptr::null(),
                        std::ptr::null(),
                        out_ptr,
                        m as c_int,
                        k as c_int,
                        n as c_int,
                        workspace_ptr,
                        self.group_size as c_int,
                        stream,
                    );
                }
                (QuantFormat::AWQ, Precision::F16) => {
                    marlin_ffi::marlin_awq_4bit_f16(
                        inputs_ptr,
                        weights_ptr,
                        scales_ptr,
                        std::ptr::null(),
                        std::ptr::null(),
                        out_ptr,
                        m as c_int,
                        k as c_int,
                        n as c_int,
                        workspace_ptr,
                        self.group_size as c_int,
                        stream,
                    );
                }
                (QuantFormat::AWQ, Precision::BF16) => {
                    marlin_ffi::marlin_awq_4bit_bf16(
                        inputs_ptr,
                        weights_ptr,
                        scales_ptr,
                        std::ptr::null(),
                        std::ptr::null(),
                        out_ptr,
                        m as c_int,
                        k as c_int,
                        n as c_int,
                        workspace_ptr,
                        self.group_size as c_int,
                        stream,
                    );
                }
            }
        }

        Ok((out, out_shape))
    }
}

impl CustomOp3 for MarlinMatMul {
    fn name(&self) -> &'static str {
        match self.quant_format {
            QuantFormat::GPTQ => "marlin_gptq_matmul",
            QuantFormat::AWQ => "marlin_awq_matmul",
        }
    }

    fn cpu_fwd(
        &self,
        _inputs: &CpuStorage,
        _inputs_shape: &Shape,
        _weights: &CpuStorage,
        _weights_shape: &Shape,
        _scales: &CpuStorage,
        _scales_shape: &Shape,
    ) -> Result<(CpuStorage, Shape)> {
        candlelight::core::bail!("Marlin matmul only supported on CUDA")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        inputs: &CudaStorage,
        inputs_shape: &Shape,
        weights: &CudaStorage,
        weights_shape: &Shape,
        scales: &CudaStorage,
        scales_shape: &Shape,
    ) -> Result<(CudaStorage, Shape)> {
        self.cuda_fwd_impl(
            inputs,
            weights,
            scales,
            inputs_shape,
            weights_shape,
            scales_shape,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marlin_matmul_construction() {
        let op = MarlinMatMul::new(Precision::F16, QuantFormat::AWQ, 128);
        assert_eq!(op.precision, Precision::F16);
        assert_eq!(op.quant_format, QuantFormat::AWQ);
        assert_eq!(op.group_size, 128);
        assert_eq!(op.name(), "marlin_awq_matmul");
    }

    #[test]
    fn test_precision_dtype() {
        assert_eq!(Precision::F16.dtype(), DType::F16);
        assert_eq!(Precision::BF16.dtype(), DType::BF16);
    }

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore] // Requires CUDA runtime and compiled kernels
    fn test_marlin_matmul_cuda() {
        // This test would require:
        // 1. CUDA device
        // 2. Compiled Marlin kernels
        // 3. Quantized weight tensors
        // Placeholder for future integration testing
    }
}
