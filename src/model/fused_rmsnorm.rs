//! Fused RMSNorm layer with graceful CPU fallback
//!
//! This module provides a unified API for RMSNorm operations:
//! - On CUDA with the cuda feature: uses fused GPU kernels from candle-layer-norm
//! - On CPU or without cuda feature: falls back to standard candlelight::nn::RmsNorm
//!
//! The fused GPU kernels provide 20-30% speedup on normalization operations by
//! combining normalization + residual addition into a single kernel pass.

use candlelight::core::{Result, Tensor};
use candlelight::nn::{Module, RmsNorm as StandardRmsNorm};

/// Wrapper for RMSNorm that automatically uses fused kernels on CUDA
#[derive(Debug)]
pub struct FusedRmsNorm {
    /// Standard RMSNorm (always available for CPU fallback)
    standard: StandardRmsNorm,
    /// Weight tensor (stored separately for CUDA kernel access)
    weight: Tensor,
    /// Epsilon value for numerical stability
    eps: f64,
}

impl FusedRmsNorm {
    /// Create a new FusedRmsNorm layer
    pub fn new(size: usize, eps: f64, vb: candlelight::nn::VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        let standard = StandardRmsNorm::new(weight.clone(), eps);
        Ok(Self {
            standard,
            weight,
            eps,
        })
    }

    /// Create from existing weight tensor
    pub fn new_with_weight(weight: Tensor, eps: f64) -> Self {
        let standard = StandardRmsNorm::new(weight.clone(), eps);
        Self {
            standard,
            weight,
            eps,
        }
    }

    /// Forward pass - uses fused kernel on CUDA, standard on CPU
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            // Check if tensor is on CUDA device
            if matches!(x.device(), Device::Cuda(_)) {
                return self.forward_cuda(x);
            }
        }

        // Fallback to standard implementation (Module trait)
        Ok(self.standard.forward(x)?)
    }

    /// Forward pass with residual addition - fuses norm + residual on CUDA
    ///
    /// # ⚠️ THE TWO ARMS DO NOT COMPUTE THE SAME THING — DO NOT CALL THIS
    ///
    /// **This has no callers, which is the only reason it is latent rather
    /// than a live defect.** Measured 2026-09-02:
    ///
    /// - **CPU**: `rms_norm(x)` then `+ residual` — NORMALIZE, THEN ADD.
    /// - **CUDA**: `fused_add_rms_norm(x, residual, ...)`, which adds first and
    ///   normalizes the sum — ADD, THEN NORMALIZE.
    ///
    /// Those are different operations and they give different results. The
    /// doc comment that stood here claimed the CPU path was "equivalent to
    /// `norm.forward(x)? + residual`", which described the CPU arm accurately
    /// and asserted an equivalence to the CUDA arm that does not hold.
    ///
    /// A helper named as though its arms are interchangeable is a trap armed
    /// for whoever calls it first. **Fix the divergence before adding a
    /// caller**; pre-norm transformers want the CUDA ordering (add, then
    /// normalize).
    pub fn forward_with_residual(&self, x: &Tensor, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        {
            // Check if tensors are on CUDA device
            if matches!(x.device(), Device::Cuda(_)) && matches!(residual.device(), Device::Cuda(_))
            {
                return self.forward_cuda_with_residual(x, residual);
            }
        }

        // Fallback: separate operations (Module trait)
        let normalized = self.standard.forward(x)?;
        let output = (&normalized + residual)?;
        Ok((output, normalized))
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda(&self, x: &Tensor) -> Result<Tensor> {
        // Use fused kernel with stored weight
        Ok(candle_layer_norm::rms_norm(
            x,
            &self.weight,
            None,
            self.eps as f32,
        )?)
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda_with_residual(
        &self,
        x: &Tensor,
        residual: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // Use fused add + rms_norm kernel with stored weight
        // Returns (normalized, residual_add)
        let (normalized, residual_sum) = candle_layer_norm::fused_add_rms_norm(
            x,
            residual,
            &self.weight,
            None,
            self.eps as f32,
        )?;

        // Return in expected order: (output_with_residual, normalized)
        Ok((residual_sum, normalized))
    }

    /// Get the underlying weight tensor
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get the epsilon value
    pub fn eps(&self) -> f64 {
        self.eps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candlelight::core::{DType, Device};

    #[test]
    fn test_fused_rmsnorm_cpu() -> Result<()> {
        let device = Device::Cpu;
        let size = 128;
        let eps = 1e-5;

        // Create test data
        let vb = candlelight::nn::VarBuilder::zeros(DType::F32, &device);
        let norm = FusedRmsNorm::new(size, eps, vb.pp("norm"))?;

        let x = Tensor::randn(0f32, 1.0, (4, size), &device)?;

        // Should work on CPU
        let _output = norm.forward(&x)?;

        Ok(())
    }

    #[test]
    fn test_fused_rmsnorm_with_residual_cpu() -> Result<()> {
        let device = Device::Cpu;
        let size = 128;
        let eps = 1e-5;

        let vb = candlelight::nn::VarBuilder::zeros(DType::F32, &device);
        let norm = FusedRmsNorm::new(size, eps, vb.pp("norm"))?;

        let x = Tensor::randn(0f32, 1.0, (4, size), &device)?;
        let residual = Tensor::randn(0f32, 1.0, (4, size), &device)?;

        // Should work on CPU
        let (output, _normalized) = norm.forward_with_residual(&x, &residual)?;

        // Check shape matches
        assert_eq!(output.dims(), x.dims());

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore] // Requires CUDA device
    fn test_fused_rmsnorm_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let size = 128;
        let eps = 1e-5;

        let vb = candlelight::nn::VarBuilder::zeros(DType::F32, &device);
        let norm = FusedRmsNorm::new(size, eps, vb.pp("norm"))?;

        let x = Tensor::randn(0f32, 1.0, (4, size), &device)?;

        // Should use fused kernel on CUDA
        let _output = norm.forward(&x)?;

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore] // Requires CUDA device
    fn test_fused_rmsnorm_numerical_parity() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let size = 128;
        let eps = 1e-5;

        // Create same weights for both
        let weight = Tensor::ones((size,), DType::F32, &device)?;

        let fused = FusedRmsNorm::new_with_weight(weight.clone(), eps);
        let standard = StandardRmsNorm::new(weight, eps);

        let x = Tensor::randn(0f32, 1.0, (4, size), &device)?;

        // Compare outputs
        let fused_out = fused.forward(&x)?;
        let standard_out = standard.forward(&x)?;

        // Should be numerically close (within tolerance)
        let diff = (fused_out - standard_out)?.abs()?.max(0)?;
        let max_diff: f32 = diff.to_scalar()?;

        assert!(max_diff < 1e-4, "Max difference: {}", max_diff);

        Ok(())
    }
}
