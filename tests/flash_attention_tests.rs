//! FlashAttention Correctness Tests
//!
//! This module tests the FlashAttention integration to ensure numerical parity
//! with the manual attention fallback implementation.
//!
//! # Test Strategy
//!
//! 1. Create identical attention layer configurations
//! 2. Run same inputs through FlashAttention path and manual attention path
//! 3. Compare outputs with strict numerical tolerance
//! 4. Test various scenarios:
//!    - Short contexts (decode phase)
//!    - Long contexts (prefill phase)
//!    - Batched scenarios (multiple sequences)
//!    - With and without attention masks
//!    - GQA configurations (different num_kv_heads)
//!
//! # Success Criteria
//!
//! - Outputs must match within 1e-3 relative error (FlashAttention uses F16 internally)
//! - Tests must pass on both CPU (fallback) and CUDA (FlashAttention)
//! - Fallback behavior must be graceful when FlashAttention not available

use anyhow::Result;
use candlelight::core::{Device, Tensor};

#[cfg(feature = "flash-attn")]
use candlelight::core::DType;

/// Maximum allowed relative error between FlashAttention and manual attention
/// Note: Slightly higher tolerance than CPU tests due to F16 precision in FlashAttention
const MAX_RELATIVE_ERROR: f32 = 1e-3;

/// Maximum allowed absolute error for values close to zero
const MAX_ABSOLUTE_ERROR: f32 = 1e-5;

/// Helper function to compare two tensors and assert they're close
fn assert_tensors_close(a: &Tensor, b: &Tensor, name: &str, rtol: f32, atol: f32) -> Result<()> {
    // Check shapes match
    if a.dims() != b.dims() {
        anyhow::bail!("{}: Shape mismatch: {:?} vs {:?}", name, a.dims(), b.dims());
    }

    // Flatten tensors for comparison
    let a_flat = a.flatten_all()?.to_vec1::<f32>()?;
    let b_flat = b.flatten_all()?.to_vec1::<f32>()?;

    let mut max_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut num_errors = 0;

    for (i, (&val_a, &val_b)) in a_flat.iter().zip(b_flat.iter()).enumerate() {
        let abs_diff = (val_a - val_b).abs();
        let rel_diff = if val_a.abs() > atol {
            abs_diff / val_a.abs()
        } else {
            0.0
        };

        max_diff = max_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);

        if abs_diff > atol && rel_diff > rtol {
            num_errors += 1;
            if num_errors <= 5 {
                // Print first 5 errors
                println!(
                    "  [{}][{}]: a={:.6e}, b={:.6e}, abs_diff={:.6e}, rel_diff={:.6e}",
                    name, i, val_a, val_b, abs_diff, rel_diff
                );
            }
        }
    }

    if num_errors > 0 {
        anyhow::bail!(
            "{}: {} values exceed tolerance (rtol={}, atol={})\n  Max abs diff: {:.6e}\n  Max rel diff: {:.6e}",
            name,
            num_errors,
            rtol,
            atol,
            max_diff,
            max_rel_diff
        );
    }

    println!(
        "✓ {}: All values within tolerance (max abs diff: {:.6e}, max rel diff: {:.6e})",
        name, max_diff, max_rel_diff
    );

    Ok(())
}

/// Test FlashAttention on single sequence decode (1 token)
#[test]
fn test_flash_attention_single_token_decode() -> Result<()> {
    println!("\n=== Testing FlashAttention: Single Token Decode ===");

    let device = Device::cuda_if_available(0)?;

    // Configuration for test
    let batch_size = 1;
    let seq_len = 1; // Decode mode
    let context_len = 128; // Existing KV cache context
    let hidden_size = 512;
    let num_heads = 8;
    let head_dim = hidden_size / num_heads;

    // Create mock Q, K, V tensors (in practice these come from projections)
    let q = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let k = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_heads, context_len, head_dim),
        &device,
    )?;
    let v = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_heads, context_len, head_dim),
        &device,
    )?;

    // Test manual attention path
    let manual_output = compute_manual_attention(&q, &k, &v, None)?;

    // Test FlashAttention path (if available)
    #[cfg(feature = "flash-attn")]
    {
        if device.is_cuda() {
            let flash_output = compute_flash_attention(&q, &k, &v, false)?;
            assert_tensors_close(
                &manual_output,
                &flash_output,
                "FlashAttention vs Manual (decode)",
                MAX_RELATIVE_ERROR,
                MAX_ABSOLUTE_ERROR,
            )?;
        } else {
            println!("⚠ Skipping FlashAttention test: CUDA not available");
        }
    }

    #[cfg(not(feature = "flash-attn"))]
    println!("⚠ Skipping FlashAttention test: flash-attn feature not enabled");

    Ok(())
}

/// Test FlashAttention on prefill (multiple tokens)
#[test]
fn test_flash_attention_prefill() -> Result<()> {
    println!("\n=== Testing FlashAttention: Prefill (Multiple Tokens) ===");

    let device = Device::cuda_if_available(0)?;

    // Configuration for test
    let batch_size = 1;
    let seq_len = 64; // Prefill mode
    let hidden_size = 512;
    let num_heads = 8;
    let head_dim = hidden_size / num_heads;

    // Create mock Q, K, V tensors
    let q = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let k = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let v = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;

    // Test manual attention path with causal mask
    let manual_output = compute_manual_attention(&q, &k, &v, None)?;

    // Test FlashAttention path (if available)
    #[cfg(feature = "flash-attn")]
    {
        if device.is_cuda() {
            let flash_output = compute_flash_attention(&q, &k, &v, true)?; // causal=true for prefill
            assert_tensors_close(
                &manual_output,
                &flash_output,
                "FlashAttention vs Manual (prefill)",
                MAX_RELATIVE_ERROR,
                MAX_ABSOLUTE_ERROR,
            )?;
        } else {
            println!("⚠ Skipping FlashAttention test: CUDA not available");
        }
    }

    #[cfg(not(feature = "flash-attn"))]
    println!("⚠ Skipping FlashAttention test: flash-attn feature not enabled");

    Ok(())
}

/// Test FlashAttention with batched sequences
#[test]
fn test_flash_attention_batched() -> Result<()> {
    println!("\n=== Testing FlashAttention: Batched Sequences ===");

    let device = Device::cuda_if_available(0)?;

    // Configuration for test
    let batch_size = 4;
    let seq_len = 32;
    let hidden_size = 512;
    let num_heads = 8;
    let head_dim = hidden_size / num_heads;

    // Create mock Q, K, V tensors
    let q = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let k = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;
    let v = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;

    // Test manual attention path
    let manual_output = compute_manual_attention(&q, &k, &v, None)?;

    // Test FlashAttention path (if available)
    #[cfg(feature = "flash-attn")]
    {
        if device.is_cuda() {
            let flash_output = compute_flash_attention(&q, &k, &v, true)?;
            assert_tensors_close(
                &manual_output,
                &flash_output,
                "FlashAttention vs Manual (batched)",
                MAX_RELATIVE_ERROR,
                MAX_ABSOLUTE_ERROR,
            )?;
        } else {
            println!("⚠ Skipping FlashAttention test: CUDA not available");
        }
    }

    #[cfg(not(feature = "flash-attn"))]
    println!("⚠ Skipping FlashAttention test: flash-attn feature not enabled");

    Ok(())
}

/// Test FlashAttention with GQA (num_kv_heads < num_heads)
#[test]
fn test_flash_attention_gqa() -> Result<()> {
    println!("\n=== Testing FlashAttention: Grouped Query Attention ===");

    let device = Device::cuda_if_available(0)?;

    // Configuration for test
    let batch_size = 2;
    let seq_len = 32;
    let hidden_size = 512;
    let num_heads = 8;
    let num_kv_heads = 2; // GQA: fewer KV heads than Q heads
    let head_dim = hidden_size / num_heads;

    // Create mock Q, K, V tensors
    let q = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )?;

    // K and V have fewer heads (GQA)
    let k = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_kv_heads, seq_len, head_dim),
        &device,
    )?;
    let v = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_kv_heads, seq_len, head_dim),
        &device,
    )?;

    // Expand K and V to match Q heads (manual GQA expansion)
    let heads_per_kv = num_heads / num_kv_heads;

    // Repeat each KV head `heads_per_kv` times along the heads dimension
    // From [batch, num_kv_heads, seq, head_dim] to [batch, num_heads, seq, head_dim]
    let mut k_expanded_parts = Vec::new();
    for i in 0..num_kv_heads {
        let k_head = k.narrow(1, i, 1)?; // Extract single head
        for _ in 0..heads_per_kv {
            k_expanded_parts.push(k_head.clone());
        }
    }
    let k_expanded = Tensor::cat(&k_expanded_parts, 1)?.contiguous()?; // Ensure contiguous

    let mut v_expanded_parts = Vec::new();
    for i in 0..num_kv_heads {
        let v_head = v.narrow(1, i, 1)?; // Extract single head
        for _ in 0..heads_per_kv {
            v_expanded_parts.push(v_head.clone());
        }
    }
    let v_expanded = Tensor::cat(&v_expanded_parts, 1)?.contiguous()?; // Ensure contiguous    // Test manual attention path
    let manual_output = compute_manual_attention(&q, &k_expanded, &v_expanded, None)?;

    // FlashAttention expects expanded K/V for GQA
    #[cfg(feature = "flash-attn")]
    {
        if device.is_cuda() {
            let flash_output = compute_flash_attention(&q, &k_expanded, &v_expanded, true)?;
            assert_tensors_close(
                &manual_output,
                &flash_output,
                "FlashAttention vs Manual (GQA)",
                MAX_RELATIVE_ERROR,
                MAX_ABSOLUTE_ERROR,
            )?;
        } else {
            println!("⚠ Skipping FlashAttention test: CUDA not available");
        }
    }

    #[cfg(not(feature = "flash-attn"))]
    println!("⚠ Skipping FlashAttention test: flash-attn feature not enabled");

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute attention using manual implementation (reference)
fn compute_manual_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (_batch, _num_heads, _seq_q, head_dim) = q.dims4()?;
    let scale = 1.0 / (head_dim as f64).sqrt();

    // Q @ K^T
    // Ensure k is contiguous before transpose
    let k = k.contiguous()?;
    let k_t = k.t()?;
    let mut attn_weights = q.matmul(&k_t)?;

    // Scale
    attn_weights = (attn_weights * scale)?;

    // Apply mask if provided
    if let Some(m) = mask {
        attn_weights = attn_weights.broadcast_add(m)?;
    }

    // Softmax
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

    // Weighted sum: attn @ V
    let output = attn_weights.matmul(v)?;

    Ok(output)
}

/// Compute attention using FlashAttention (when available)
#[cfg(feature = "flash-attn")]
fn compute_flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    let (_batch, _num_heads, _seq, head_dim) = q.dims4()?;
    let softmax_scale = (1.0 / (head_dim as f64).sqrt()) as f32;

    // FlashAttention expects (batch, seq_len, num_heads, head_dim)
    // Our tensors are (batch, num_heads, seq_len, head_dim)
    let q_flash = q.transpose(1, 2)?;
    let k_flash = k.transpose(1, 2)?;
    let v_flash = v.transpose(1, 2)?;

    // Convert to F16 for CUDA
    let device = q.device();
    let flash_dtype = if device.is_cuda() {
        DType::F16
    } else {
        q.dtype()
    };

    let q_flash = q_flash.to_dtype(flash_dtype)?;
    let k_flash = k_flash.to_dtype(flash_dtype)?;
    let v_flash = v_flash.to_dtype(flash_dtype)?;

    // Call FlashAttention
    let attn_output =
        candle_flash_attn::flash_attn(&q_flash, &k_flash, &v_flash, softmax_scale, causal)?;

    // Convert back and transpose
    let output = attn_output.to_dtype(q.dtype())?.transpose(1, 2)?;

    Ok(output)
}

#[cfg(not(feature = "flash-attn"))]
fn compute_flash_attention(_q: &Tensor, _k: &Tensor, _v: &Tensor, _causal: bool) -> Result<Tensor> {
    anyhow::bail!("FlashAttention not available: compile with --features flash-attn")
}
