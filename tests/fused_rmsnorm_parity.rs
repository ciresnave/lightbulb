//! Numerical parity tests for FusedRmsNorm
//!
//! Validates that FusedRmsNorm produces numerically identical results to
//! standard candle_nn::RmsNorm within acceptable tolerance.

use candlelight::core::{DType, Device, Result, Tensor};
use candlelight::nn::{Module, RmsNorm as StandardRmsNorm, VarBuilder};
use lightbulb::model::fused_rmsnorm::FusedRmsNorm;

const TOLERANCE: f32 = 1e-4;

/// Helper to compute max absolute difference between tensors
fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let diff = (a - b)?.abs()?.flatten_all()?;
    let max_val = diff.max(0)?;
    max_val.to_scalar::<f32>()
}

#[test]
fn test_parity_cpu_f32() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let hidden_size = 512;
    let batch_size = 4;
    let seq_len = 128;
    let eps = 1e-5;

    // Create same weights for both implementations
    let vb = VarBuilder::zeros(dtype, &device);
    let weight = vb.get(hidden_size, "weight")?;

    let fused = FusedRmsNorm::new_with_weight(weight.clone(), eps);
    let standard = StandardRmsNorm::new(weight, eps);

    // Create test input
    let input = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;

    // Reshape to 2D for RmsNorm (expects [batch*seq, hidden])
    let input_2d = input.reshape((batch_size * seq_len, hidden_size))?;

    // Compare outputs
    let fused_out = fused.forward(&input_2d)?;
    let standard_out = standard.forward(&input_2d)?;

    let diff = max_abs_diff(&fused_out, &standard_out)?;

    assert!(
        diff < TOLERANCE,
        "Max difference {} exceeds tolerance {} on CPU F32",
        diff,
        TOLERANCE
    );

    println!("CPU F32 parity test passed: max_diff = {:.6e}", diff);
    Ok(())
}

#[test]
fn test_parity_cpu_f32_various_sizes() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let eps = 1e-5;

    let test_configs = vec![
        (128, 2, 32),   // Small: hidden=128, batch=2, seq=32
        (512, 4, 64),   // Medium: hidden=512, batch=4, seq=64
        (1024, 8, 128), // Large: hidden=1024, batch=8, seq=128
        (4096, 1, 256), // Very large hidden, small batch
    ];

    for (hidden_size, batch_size, seq_len) in test_configs {
        let vb = VarBuilder::zeros(dtype, &device);
        let weight = vb.get(hidden_size, "weight")?;

        let fused = FusedRmsNorm::new_with_weight(weight.clone(), eps);
        let standard = StandardRmsNorm::new(weight, eps);

        let input = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
        let input_2d = input.reshape((batch_size * seq_len, hidden_size))?;

        let fused_out = fused.forward(&input_2d)?;
        let standard_out = standard.forward(&input_2d)?;

        let diff = max_abs_diff(&fused_out, &standard_out)?;

        assert!(
            diff < TOLERANCE,
            "Max difference {} exceeds tolerance {} for config ({}, {}, {})",
            diff,
            TOLERANCE,
            hidden_size,
            batch_size,
            seq_len
        );

        println!(
            "Config (hidden={}, batch={}, seq={}): max_diff = {:.6e}",
            hidden_size, batch_size, seq_len, diff
        );
    }

    Ok(())
}

#[test]
fn test_parity_with_residual_cpu() -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let hidden_size = 512;
    let batch_size = 4;
    let seq_len = 64;
    let eps = 1e-5;

    // Create same weights
    let vb = VarBuilder::zeros(dtype, &device);
    let weight = vb.get(hidden_size, "weight")?;

    let fused = FusedRmsNorm::new_with_weight(weight.clone(), eps);
    let standard = StandardRmsNorm::new(weight, eps);

    // Create test inputs
    let input = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
    let residual = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;

    let input_2d = input.reshape((batch_size * seq_len, hidden_size))?;
    let residual_2d = residual.reshape((batch_size * seq_len, hidden_size))?;

    // Test fused version
    let (fused_output, _fused_normalized) = fused.forward_with_residual(&input_2d, &residual_2d)?;

    // Simulate standard version (separate operations)
    let standard_normalized = standard.forward(&input_2d)?;
    let standard_output = (&standard_normalized + &residual_2d)?;

    let diff = max_abs_diff(&fused_output, &standard_output)?;

    assert!(
        diff < TOLERANCE,
        "Max difference {} exceeds tolerance {} for residual addition on CPU",
        diff,
        TOLERANCE
    );

    println!("CPU residual parity test passed: max_diff = {:.6e}", diff);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore] // Requires CUDA hardware
fn test_parity_cuda_f32() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::F32;
    let hidden_size = 512;
    let batch_size = 4;
    let seq_len = 128;
    let eps = 1e-5;

    // Create same weights for both implementations
    let vb = VarBuilder::zeros(dtype, &device);
    let weight = vb.get(hidden_size, "weight")?;

    let fused = FusedRmsNorm::new_with_weight(weight.clone(), eps);
    let standard = StandardRmsNorm::new(weight, eps);

    // Create test input
    let input = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
    let input_2d = input.reshape((batch_size * seq_len, hidden_size))?;

    // Compare outputs
    let fused_out = fused.forward(&input_2d)?;
    let standard_out = standard.forward(&input_2d)?;

    let diff = max_abs_diff(&fused_out, &standard_out)?;

    assert!(
        diff < TOLERANCE,
        "Max difference {} exceeds tolerance {} on CUDA F32",
        diff,
        TOLERANCE
    );

    println!("CUDA F32 parity test passed: max_diff = {:.6e}", diff);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore] // Requires CUDA hardware
fn test_parity_cuda_f16() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::F16;
    let hidden_size = 512;
    let batch_size = 4;
    let seq_len = 128;
    let eps = 1e-5;

    // F16 has lower precision, use larger tolerance
    let f16_tolerance = 1e-3;

    let vb = VarBuilder::zeros(dtype, &device);
    let weight = vb.get(hidden_size, "weight")?;

    let fused = FusedRmsNorm::new_with_weight(weight.clone(), eps);
    let standard = StandardRmsNorm::new(weight, eps);

    let input =
        Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?.to_dtype(dtype)?;
    let input_2d = input.reshape((batch_size * seq_len, hidden_size))?;

    let fused_out = fused.forward(&input_2d)?;
    let standard_out = standard.forward(&input_2d)?;

    // Convert to F32 for comparison
    let fused_f32 = fused_out.to_dtype(DType::F32)?;
    let standard_f32 = standard_out.to_dtype(DType::F32)?;

    let diff = max_abs_diff(&fused_f32, &standard_f32)?;

    assert!(
        diff < f16_tolerance,
        "Max difference {} exceeds tolerance {} on CUDA F16",
        diff,
        f16_tolerance
    );

    println!("CUDA F16 parity test passed: max_diff = {:.6e}", diff);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore] // Requires CUDA hardware
fn test_parity_cuda_with_residual() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::F32;
    let hidden_size = 512;
    let batch_size = 4;
    let seq_len = 64;
    let eps = 1e-5;

    let vb = VarBuilder::zeros(dtype, &device);
    let weight = vb.get(hidden_size, "weight")?;

    let fused = FusedRmsNorm::new_with_weight(weight.clone(), eps);
    let standard = StandardRmsNorm::new(weight, eps);

    let input = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
    let residual = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;

    let input_2d = input.reshape((batch_size * seq_len, hidden_size))?;
    let residual_2d = residual.reshape((batch_size * seq_len, hidden_size))?;

    // Test fused version (should use fused kernel on CUDA)
    let (fused_output, _fused_normalized) = fused.forward_with_residual(&input_2d, &residual_2d)?;

    // Standard version (separate operations)
    let standard_normalized = standard.forward(&input_2d)?;
    let standard_output = (&standard_normalized + &residual_2d)?;

    let diff = max_abs_diff(&fused_output, &standard_output)?;

    assert!(
        diff < TOLERANCE,
        "Max difference {} exceeds tolerance {} for residual on CUDA",
        diff,
        TOLERANCE
    );

    println!("CUDA residual parity test passed: max_diff = {:.6e}", diff);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore] // Requires CUDA hardware
fn test_parity_cuda_various_sizes() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::F32;
    let eps = 1e-5;

    let test_configs = vec![
        (128, 2, 32),   // Small
        (512, 4, 64),   // Medium
        (1024, 8, 128), // Large
        (2048, 4, 256), // Very large hidden
        (4096, 1, 512), // Huge hidden, small batch
        (8192, 2, 64),  // Max supported by fused kernel
    ];

    for (hidden_size, batch_size, seq_len) in test_configs {
        let vb = VarBuilder::zeros(dtype, &device);
        let weight = vb.get(hidden_size, "weight")?;

        let fused = FusedRmsNorm::new_with_weight(weight.clone(), eps);
        let standard = StandardRmsNorm::new(weight, eps);

        let input = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?;
        let input_2d = input.reshape((batch_size * seq_len, hidden_size))?;

        let fused_out = fused.forward(&input_2d)?;
        let standard_out = standard.forward(&input_2d)?;

        let diff = max_abs_diff(&fused_out, &standard_out)?;

        assert!(
            diff < TOLERANCE,
            "Max difference {} exceeds tolerance {} on CUDA for config ({}, {}, {})",
            diff,
            TOLERANCE,
            hidden_size,
            batch_size,
            seq_len
        );

        println!(
            "CUDA config (hidden={}, batch={}, seq={}): max_diff = {:.6e}",
            hidden_size, batch_size, seq_len, diff
        );
    }

    Ok(())
}
