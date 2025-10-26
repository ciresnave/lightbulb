//! FlashAttention Performance Benchmark
//!
//! This benchmark compares FlashAttention performance against manual attention implementation.
//!
//! ## Usage
//!
//! ```bash
//! # CPU baseline (manual attention only)
//! cargo run --release --example benchmark_flashattention
//!
//! # GPU with FlashAttention (requires CUDA)
//! cargo run --release --features cuda,flash-attn --example benchmark_flashattention
//! ```
//!
//! ## Benchmark Scenarios
//!
//! 1. **Decode mode** (seq_len=1): Single token generation
//! 2. **Short prefill** (seq_len=64): Small prompt processing
//! 3. **Medium prefill** (seq_len=512): Typical prompt processing
//! 4. **Long prefill** (seq_len=2048): Long context processing
//! 5. **Batched decode** (batch=8, seq=1): Multi-sequence generation
//!
//! ## Expected Results
//!
//! On NVIDIA A100 with FlashAttention:
//! - Decode: ~10-20% faster than manual (memory bound)
//! - Short prefill: ~1.5-2× faster (moderate speedup)
//! - Medium prefill: ~2-3× faster (significant speedup)
//! - Long prefill: ~3-5× faster (best speedup)
//!
//! On CPU:
//! - FlashAttention not available, falls back to manual attention

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::time::Instant;

#[cfg(feature = "flash-attn")]
use candle_core::DType;

/// Benchmark configuration
struct BenchmarkConfig {
    name: &'static str,
    batch_size: usize,
    seq_len: usize,
    context_len: usize,
    hidden_size: usize,
    num_heads: usize,
    num_iterations: usize,
}

impl BenchmarkConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

fn main() -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║          FlashAttention Performance Benchmark               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let device = Device::cuda_if_available(0)?;
    println!("Device: {:?}", device);

    #[cfg(feature = "flash-attn")]
    {
        if device.is_cuda() {
            println!("FlashAttention: ENABLED ✓");
        } else {
            println!("FlashAttention: Disabled (CPU mode)");
        }
    }

    #[cfg(not(feature = "flash-attn"))]
    println!("FlashAttention: Disabled (feature not enabled)");

    println!();

    // Benchmark scenarios
    let configs = vec![
        BenchmarkConfig {
            name: "Decode (single token)",
            batch_size: 1,
            seq_len: 1,
            context_len: 128,
            hidden_size: 4096,
            num_heads: 32,
            num_iterations: 100,
        },
        BenchmarkConfig {
            name: "Short prefill",
            batch_size: 1,
            seq_len: 64,
            context_len: 64,
            hidden_size: 4096,
            num_heads: 32,
            num_iterations: 50,
        },
        BenchmarkConfig {
            name: "Medium prefill",
            batch_size: 1,
            seq_len: 512,
            context_len: 512,
            hidden_size: 4096,
            num_heads: 32,
            num_iterations: 20,
        },
        BenchmarkConfig {
            name: "Long prefill",
            batch_size: 1,
            seq_len: 2048,
            context_len: 2048,
            hidden_size: 4096,
            num_heads: 32,
            num_iterations: 10,
        },
        BenchmarkConfig {
            name: "Batched decode (8 sequences)",
            batch_size: 8,
            seq_len: 1,
            context_len: 128,
            hidden_size: 4096,
            num_heads: 32,
            num_iterations: 50,
        },
    ];

    for config in configs {
        run_benchmark(&config, &device)?;
    }

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Benchmark Complete                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

fn run_benchmark(config: &BenchmarkConfig, device: &Device) -> Result<()> {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark: {}", config.name);
    println!(
        "  Config: batch={}, seq={}, context={}, heads={}, hidden={}",
        config.batch_size, config.seq_len, config.context_len, config.num_heads, config.hidden_size
    );
    println!("  Iterations: {}", config.num_iterations);

    // Create tensors
    let q = Tensor::randn(
        0f32,
        1f32,
        (
            config.batch_size,
            config.num_heads,
            config.seq_len,
            config.head_dim(),
        ),
        device,
    )?;
    let k = Tensor::randn(
        0f32,
        1f32,
        (
            config.batch_size,
            config.num_heads,
            config.context_len,
            config.head_dim(),
        ),
        device,
    )?;
    let v = Tensor::randn(
        0f32,
        1f32,
        (
            config.batch_size,
            config.num_heads,
            config.context_len,
            config.head_dim(),
        ),
        device,
    )?;

    // Benchmark manual attention
    let manual_time = benchmark_manual_attention(&q, &k, &v, config.num_iterations)?;
    println!(
        "  Manual attention: {:.2}ms per iteration",
        manual_time * 1000.0
    );

    // Benchmark FlashAttention (if available)
    #[cfg(feature = "flash-attn")]
    {
        if device.is_cuda() {
            let causal = config.seq_len > 1; // Causal for prefill, non-causal for decode
            let flash_time = benchmark_flash_attention(&q, &k, &v, causal, config.num_iterations)?;
            println!(
                "  FlashAttention:   {:.2}ms per iteration",
                flash_time * 1000.0
            );

            let speedup = manual_time / flash_time;
            println!("  ⚡ Speedup: {:.2}× faster", speedup);

            if speedup > 1.1 {
                println!("  ✓ FlashAttention provides significant speedup!");
            } else {
                println!("  ℹ Marginal difference (expected for small sequences)");
            }
        }
    }

    println!();
    Ok(())
}

fn benchmark_manual_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_iterations: usize,
) -> Result<f64> {
    // Warmup
    for _ in 0..5 {
        let _ = compute_manual_attention(q, k, v)?;
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = compute_manual_attention(q, k, v)?;
    }
    let elapsed = start.elapsed().as_secs_f64();

    Ok(elapsed / num_iterations as f64)
}

#[cfg(feature = "flash-attn")]
fn benchmark_flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    num_iterations: usize,
) -> Result<f64> {
    // Warmup
    for _ in 0..5 {
        let _ = compute_flash_attention(q, k, v, causal)?;
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..num_iterations {
        let _ = compute_flash_attention(q, k, v, causal)?;
    }
    let elapsed = start.elapsed().as_secs_f64();

    Ok(elapsed / num_iterations as f64)
}

// ============================================================================
// Attention Implementations
// ============================================================================

fn compute_manual_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (_batch, _num_heads, _seq_q, head_dim) = q.dims4()?;
    let scale = 1.0 / (head_dim as f64).sqrt();

    // Q @ K^T
    let k = k.contiguous()?;
    let k_t = k.t()?;
    let attn_weights = q.matmul(&k_t)?;

    // Scale
    let attn_weights = (attn_weights * scale)?;

    // Softmax
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

    // Weighted sum
    let output = attn_weights.matmul(v)?;

    Ok(output)
}

#[cfg(feature = "flash-attn")]
fn compute_flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    let (_batch, _num_heads, _seq, head_dim) = q.dims4()?;
    let softmax_scale = (1.0 / (head_dim as f64).sqrt()) as f32;

    // FlashAttention expects (batch, seq_len, num_heads, head_dim)
    let q_flash = q.transpose(1, 2)?;
    let k_flash = k.transpose(1, 2)?;
    let v_flash = v.transpose(1, 2)?;

    // Convert to F16 for CUDA
    let q_flash = q_flash.to_dtype(DType::F16)?;
    let k_flash = k_flash.to_dtype(DType::F16)?;
    let v_flash = v_flash.to_dtype(DType::F16)?;

    // Call FlashAttention
    let attn_output =
        candle_flash_attn::flash_attn(&q_flash, &k_flash, &v_flash, softmax_scale, causal)?;

    // Convert back and transpose
    let output = attn_output.to_dtype(q.dtype())?.transpose(1, 2)?;

    Ok(output)
}
