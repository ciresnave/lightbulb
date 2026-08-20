//! Performance benchmarks for FusedRmsNorm
//!
//! Measures the performance improvement of fused GPU kernels compared to
//! standard RmsNorm implementation.
//!
//! Run with: cargo bench --bench fused_rmsnorm_perf --features cuda
//!
//! Note: Requires CUDA hardware to see actual speedup.

use candlelight::core::{DType, Device, Result, Tensor};
use candlelight::nn::{Module, RmsNorm as StandardRmsNorm, VarBuilder};
use lightbulb::model::fused_rmsnorm::FusedRmsNorm;
use std::time::Instant;

/// Benchmark configuration
struct BenchConfig {
    name: &'static str,
    hidden_size: usize,
    batch_size: usize,
    seq_len: usize,
    warmup_iters: usize,
    bench_iters: usize,
}

impl BenchConfig {
    fn total_elements(&self) -> usize {
        self.batch_size * self.seq_len * self.hidden_size
    }
}

/// Benchmark a forward pass
fn benchmark_forward<F>(name: &str, config: &BenchConfig, mut forward_fn: F) -> Result<f64>
where
    F: FnMut() -> Result<Tensor>,
{
    // Warmup
    for _ in 0..config.warmup_iters {
        let _ = forward_fn()?;
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..config.bench_iters {
        let _ = forward_fn()?;
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_secs_f64() * 1000.0 / config.bench_iters as f64;

    println!(
        "  {} ({}): {:.3} ms/iter ({} elements)",
        name,
        config.name,
        avg_ms,
        config.total_elements()
    );

    Ok(avg_ms)
}

#[test]
#[ignore] // Manual benchmark - run with --ignored
fn benchmark_cpu() -> Result<()> {
    println!("\n=== CPU Benchmarks ===\n");

    let configs = vec![
        BenchConfig {
            name: "Small (Llama-7B layer)",
            hidden_size: 4096,
            batch_size: 1,
            seq_len: 128,
            warmup_iters: 5,
            bench_iters: 50,
        },
        BenchConfig {
            name: "Medium (Llama-7B layer, batch=4)",
            hidden_size: 4096,
            batch_size: 4,
            seq_len: 128,
            warmup_iters: 5,
            bench_iters: 50,
        },
        BenchConfig {
            name: "Large (Llama-13B layer)",
            hidden_size: 5120,
            batch_size: 4,
            seq_len: 256,
            warmup_iters: 3,
            bench_iters: 30,
        },
    ];

    let device = Device::Cpu;
    let dtype = DType::F32;
    let eps = 1e-5;

    for config in configs {
        println!("\nConfig: {}", config.name);

        let vb = VarBuilder::zeros(dtype, &device);
        let weight = vb.get(config.hidden_size, "weight")?;

        let fused = FusedRmsNorm::new_with_weight(weight.clone(), eps);
        let standard = StandardRmsNorm::new(weight, eps);

        let input = Tensor::randn(
            0f32,
            1.0,
            (config.batch_size, config.seq_len, config.hidden_size),
            &device,
        )?;
        let input_2d = input.reshape((config.batch_size * config.seq_len, config.hidden_size))?;

        // Benchmark standard
        let standard_time =
            benchmark_forward("Standard RmsNorm", &config, || standard.forward(&input_2d))?;

        // Benchmark fused (on CPU, should be same as standard)
        let fused_time =
            benchmark_forward("Fused RmsNorm  ", &config, || fused.forward(&input_2d))?;

        let speedup = standard_time / fused_time;
        println!("  Speedup: {:.2}x", speedup);
        println!("  Note: On CPU, fused uses standard implementation (no speedup expected)");
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore] // Requires CUDA hardware
fn benchmark_cuda() -> Result<()> {
    println!("\n=== CUDA Benchmarks ===\n");

    let configs = vec![
        BenchConfig {
            name: "Small (Llama-7B layer, batch=1)",
            hidden_size: 4096,
            batch_size: 1,
            seq_len: 128,
            warmup_iters: 10,
            bench_iters: 100,
        },
        BenchConfig {
            name: "Medium (Llama-7B layer, batch=4)",
            hidden_size: 4096,
            batch_size: 4,
            seq_len: 128,
            warmup_iters: 10,
            bench_iters: 100,
        },
        BenchConfig {
            name: "Large (Llama-13B layer, batch=4)",
            hidden_size: 5120,
            batch_size: 4,
            seq_len: 256,
            warmup_iters: 10,
            bench_iters: 100,
        },
        BenchConfig {
            name: "Very Large (Llama-70B layer, batch=2)",
            hidden_size: 8192,
            batch_size: 2,
            seq_len: 512,
            warmup_iters: 5,
            bench_iters: 50,
        },
    ];

    let device = Device::new_cuda(0)?;
    let dtype = DType::F32;
    let eps = 1e-5;

    println!("Device: {:?}", device);
    println!("Expected speedup: 20-30% (1.2-1.3x)\n");

    for config in configs {
        println!("\nConfig: {}", config.name);

        let vb = VarBuilder::zeros(dtype, &device);
        let weight = vb.get(config.hidden_size, "weight")?;

        let fused = FusedRmsNorm::new_with_weight(weight.clone(), eps);
        let standard = StandardRmsNorm::new(weight, eps);

        let input = Tensor::randn(
            0f32,
            1.0,
            (config.batch_size, config.seq_len, config.hidden_size),
            &device,
        )?;
        let input_2d = input.reshape((config.batch_size * config.seq_len, config.hidden_size))?;

        // Benchmark standard
        let standard_time =
            benchmark_forward("Standard RmsNorm", &config, || standard.forward(&input_2d))?;

        // Benchmark fused (should use fused kernel on CUDA)
        let fused_time =
            benchmark_forward("Fused RmsNorm  ", &config, || fused.forward(&input_2d))?;

        let speedup = standard_time / fused_time;
        let improvement = (speedup - 1.0) * 100.0;

        println!("  Speedup: {:.2}x ({:.1}% faster)", speedup, improvement);

        if speedup >= 1.2 {
            println!("  ✅ Meets 20% improvement target");
        } else if speedup >= 1.1 {
            println!("  ⚠️  Below 20% target but shows improvement");
        } else {
            println!("  ❌ No significant speedup detected");
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore] // Requires CUDA hardware
fn benchmark_cuda_with_residual() -> Result<()> {
    println!("\n=== CUDA Benchmarks with Residual Addition ===\n");

    let configs = vec![
        BenchConfig {
            name: "Llama-7B layer (batch=4)",
            hidden_size: 4096,
            batch_size: 4,
            seq_len: 128,
            warmup_iters: 10,
            bench_iters: 100,
        },
        BenchConfig {
            name: "Llama-13B layer (batch=4)",
            hidden_size: 5120,
            batch_size: 4,
            seq_len: 256,
            warmup_iters: 10,
            bench_iters: 100,
        },
    ];

    let device = Device::new_cuda(0)?;
    let dtype = DType::F32;
    let eps = 1e-5;

    println!("Testing fused norm + residual addition");
    println!("Expected speedup: Higher than norm alone (fewer kernel launches)\n");

    for config in configs {
        println!("\nConfig: {}", config.name);

        let vb = VarBuilder::zeros(dtype, &device);
        let weight = vb.get(config.hidden_size, "weight")?;

        let fused = FusedRmsNorm::new_with_weight(weight.clone(), eps);
        let standard = StandardRmsNorm::new(weight, eps);

        let input = Tensor::randn(
            0f32,
            1.0,
            (config.batch_size, config.seq_len, config.hidden_size),
            &device,
        )?;
        let residual = Tensor::randn(
            0f32,
            1.0,
            (config.batch_size, config.seq_len, config.hidden_size),
            &device,
        )?;

        let input_2d = input.reshape((config.batch_size * config.seq_len, config.hidden_size))?;
        let residual_2d =
            residual.reshape((config.batch_size * config.seq_len, config.hidden_size))?;

        // Benchmark standard (separate operations)
        let standard_time = benchmark_forward("Standard (norm + add)", &config, || {
            let normalized = standard.forward(&input_2d)?;
            let output = (&normalized + &residual_2d)?;
            Ok(output)
        })?;

        // Benchmark fused (single kernel)
        let fused_time = benchmark_forward("Fused (norm+add)   ", &config, || {
            let (output, _) = fused.forward_with_residual(&input_2d, &residual_2d)?;
            Ok(output)
        })?;

        let speedup = standard_time / fused_time;
        let improvement = (speedup - 1.0) * 100.0;

        println!("  Speedup: {:.2}x ({:.1}% faster)", speedup, improvement);
        println!("  Note: Fused version saves one kernel launch");
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore] // Requires CUDA hardware
fn benchmark_cuda_fp16() -> Result<()> {
    println!("\n=== CUDA FP16 Benchmarks ===\n");

    let config = BenchConfig {
        name: "Llama-7B layer (FP16, batch=4)",
        hidden_size: 4096,
        batch_size: 4,
        seq_len: 128,
        warmup_iters: 10,
        bench_iters: 100,
    };

    let device = Device::new_cuda(0)?;
    let dtype = DType::F16;
    let eps = 1e-5;

    println!("Testing FP16 performance (lower precision, higher throughput)");
    println!("Config: {}\n", config.name);

    let vb = VarBuilder::zeros(dtype, &device);
    let weight = vb.get(config.hidden_size, "weight")?;

    let fused = FusedRmsNorm::new_with_weight(weight.clone(), eps);
    let standard = StandardRmsNorm::new(weight, eps);

    let input = Tensor::randn(
        0f32,
        1.0,
        (config.batch_size, config.seq_len, config.hidden_size),
        &device,
    )?
    .to_dtype(dtype)?;
    let input_2d = input.reshape((config.batch_size * config.seq_len, config.hidden_size))?;

    // Benchmark standard
    let standard_time = benchmark_forward("Standard RmsNorm (FP16)", &config, || {
        standard.forward(&input_2d)
    })?;

    // Benchmark fused
    let fused_time = benchmark_forward("Fused RmsNorm (FP16)  ", &config, || {
        fused.forward(&input_2d)
    })?;

    let speedup = standard_time / fused_time;
    let improvement = (speedup - 1.0) * 100.0;

    println!("  Speedup: {:.2}x ({:.1}% faster)", speedup, improvement);

    Ok(())
}
