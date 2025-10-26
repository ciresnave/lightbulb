//! Simple MLP benchmark to measure CPU kernel fusion performance (M3.3)
//!
//! This benchmark directly measures MLP forward pass with and without fusion
//! to isolate the performance improvement from CPU kernel fusion.

use lightbulb::model::mlp_wrapper::Mlp;
use candle_core::{Device, DType, Result, Tensor};
use candle_nn::VarBuilder;
use std::time::Instant;

// Benchmark configuration
const NUM_ITERATIONS: usize = 1000;
const BATCH_SIZE: usize = 1;
const SEQ_LEN: usize = 1; // Decode mode (critical path)
const HIDDEN_SIZE: usize = 4096;
const INTERMEDIATE_SIZE: usize = 11008; // Llama-style MLP

fn main() -> Result<()> {
    println!("=== M3.3 MLP CPU Kernel Fusion Benchmark ===\n");
    println!("Configuration:");
    println!("  Hidden size: {}", HIDDEN_SIZE);
    println!("  Intermediate size: {}", INTERMEDIATE_SIZE);
    println!("  Batch size: {}", BATCH_SIZE);
    println!("  Sequence length: {} (decode mode)", SEQ_LEN);
    println!("  Iterations: {}\n", NUM_ITERATIONS);

    let device = Device::Cpu;
    let dtype = DType::F32;

    // Create input tensor
    let input = Tensor::randn(0f32, 1f32, &[BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE], &device)?;

    // ============================================================
    // Benchmark 1: Unfused MLP (baseline)
    // ============================================================
    println!("[1/2] Creating UNFUSED MLP...");
    let vb_unfused = VarBuilder::zeros(dtype, &device);
    let mlp_unfused = Mlp::new(
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        vb_unfused.pp("mlp"),
        false, // use_fused_kernels = false
    )?;

    println!("[1/2] Benchmarking UNFUSED MLP...");
    let unfused_latency = benchmark_mlp(&mlp_unfused, &input, NUM_ITERATIONS, "UNFUSED")?;

    // ============================================================
    // Benchmark 2: Fused MLP
    // ============================================================
    println!("\n[2/2] Creating FUSED MLP...");
    let vb_fused = VarBuilder::zeros(dtype, &device);
    let mlp_fused = Mlp::new(
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        vb_fused.pp("mlp"),
        true, // use_fused_kernels = true
    )?;

    println!("[2/2] Benchmarking FUSED MLP...");
    let fused_latency = benchmark_mlp(&mlp_fused, &input, NUM_ITERATIONS, "FUSED")?;

    // ============================================================
    // Results comparison
    // ============================================================
    println!("\n=== RESULTS ===\n");
    
    let unfused_throughput = 1000.0 / unfused_latency; // tokens/sec
    let fused_throughput = 1000.0 / fused_latency;

    println!("Unfused MLP (baseline):");
    println!("  Mean latency: {:.3} ms/forward", unfused_latency);
    println!("  Throughput: {:.1} tokens/sec\n", unfused_throughput);

    println!("Fused MLP:");
    println!("  Mean latency: {:.3} ms/forward", fused_latency);
    println!("  Throughput: {:.1} tokens/sec\n", fused_throughput);

    // Calculate improvement
    let latency_improvement = ((unfused_latency - fused_latency) / unfused_latency) * 100.0;
    let throughput_improvement = ((fused_throughput - unfused_throughput) / unfused_throughput) * 100.0;

    println!("Improvement:");
    println!("  Latency: {:.1}% faster", latency_improvement);
    println!("  Throughput: {:.1}% higher", throughput_improvement);

    // Validation
    println!("\nAnalysis:");
    if throughput_improvement >= 10.0 {
        println!("  ✅ SUCCESS: Achieved target >10% throughput improvement!");
    } else if throughput_improvement >= 5.0 {
        println!("  ⚠️  Moderate improvement: {:.1}% (target: >10%)", throughput_improvement);
        println!("  Note: Fusion benefits may be limited by compiler optimizations.");
    } else if throughput_improvement >= 0.0 {
        println!("  ⚠️  Minimal improvement: {:.1}% (target: >10%)", throughput_improvement);
        println!("  Note: Candle may already be optimizing these operations internally.");
    } else {
        println!("  ❌ Regression: {:.1}% slower", throughput_improvement.abs());
        println!("  Note: Fusion overhead may exceed benefits in this configuration.");
    }

    println!("\nNotes:");
    println!("  - Run in --release mode for accurate performance measurement");
    println!("  - Results vary by CPU, system load, and compiler optimizations");
    println!("  - Fusion reduces memory traffic (11.3% theoretical bandwidth reduction)");
    println!("  - Benefits are most visible with large batch sizes and memory-bound workloads");

    Ok(())
}

fn benchmark_mlp(mlp: &Mlp, input: &Tensor, iterations: usize, label: &str) -> Result<f64> {
    // Warmup
    for _ in 0..10 {
        let _ = mlp.forward(input)?;
    }

    // Benchmark
    let mut latencies = Vec::with_capacity(iterations);
    
    for _ in 0..iterations {
        let start = Instant::now();
        let _output = mlp.forward(input)?;
        let elapsed = start.elapsed();
        latencies.push(elapsed.as_secs_f64() * 1000.0); // Convert to ms
    }

    // Calculate statistics
    let mean_latency: f64 = latencies.iter().sum::<f64>() / latencies.len() as f64;
    
    let variance: f64 = latencies.iter()
        .map(|&x| {
            let diff = x - mean_latency;
            diff * diff
        })
        .sum::<f64>() / latencies.len() as f64;
    let std_dev = variance.sqrt();
    let cov = (std_dev / mean_latency) * 100.0;

    println!("  ✓ {} iterations completed", iterations);
    println!("  Mean: {:.3} ms, Std dev: {:.3} ms ({:.1}% CoV)", mean_latency, std_dev, cov);

    Ok(mean_latency)
}
