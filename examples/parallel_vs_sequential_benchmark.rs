//! Performance benchmark: ParallelModelManager vs ModelManager
//!
//! Compares the parallel batched implementation against the sequential implementation
//! to demonstrate the speedup achieved through true parallel batching and chunked prefill.

use lightbulb::engine::{Request, RequestContext};
use lightbulb::model::{ModelManager, ParallelModelManager};
use std::time::Instant;

const MODEL_PATH: &str = "../models/llama-3b";

fn main() -> anyhow::Result<()> {
    if !std::path::Path::new(MODEL_PATH).exists() {
        eprintln!("Error: Model not found at {}", MODEL_PATH);
        eprintln!("Please ensure llama-3b model is available");
        return Ok(());
    }

    println!("\n{}", "=".repeat(80));
    println!("Performance Benchmark: Parallel vs Sequential Batched Inference");
    println!("{}", "=".repeat(80));

    // Test configurations
    let batch_sizes = vec![1, 2, 4, 6];
    let tokens_per_request = 10;

    println!("\nConfiguration:");
    println!("  Model: llama-3b (f32, CPU)");
    println!("  Tokens per request: {}", tokens_per_request);
    println!("  Batch sizes: {:?}", batch_sizes);

    let mut results = Vec::new();

    for &batch_size in &batch_sizes {
        println!("\n{}", "-".repeat(80));
        println!("Testing batch size: {}", batch_size);
        println!("{}", "-".repeat(80));

        // Create test requests
        let requests: Vec<Request> = (0..batch_size)
            .map(|i| Request {
                id: format!("req-{}", i),
                prompt: match i % 3 {
                    0 => "The capital of France is".to_string(),
                    1 => "Rust programming language".to_string(),
                    _ => "Machine learning is".to_string(),
                },
                max_new_tokens: tokens_per_request,
            })
            .collect();

        // Benchmark Sequential (ModelManager)
        println!("\n[1/2] Running Sequential (ModelManager)...");
        let sequential_result = benchmark_sequential(requests.clone())?;

        // Benchmark Parallel (ParallelModelManager)
        println!("\n[2/2] Running Parallel (ParallelModelManager)...");
        let parallel_result = benchmark_parallel(requests.clone())?;

        // Calculate speedup
        let speedup = sequential_result.tokens_per_sec / parallel_result.tokens_per_sec;

        println!("\n--- Results for Batch Size {} ---", batch_size);
        println!("\nSequential:");
        println!("  Total time: {:.2}s", sequential_result.total_time);
        println!("  Forward time: {:.2}ms", sequential_result.forward_time_ms);
        println!("  Tokens generated: {}", sequential_result.tokens_generated);
        println!(
            "  Throughput: {:.2} tokens/sec",
            sequential_result.tokens_per_sec
        );

        println!("\nParallel:");
        println!("  Total time: {:.2}s", parallel_result.total_time);
        println!("  Forward time: {:.2}ms", parallel_result.forward_time_ms);
        println!("  Tokens generated: {}", parallel_result.tokens_generated);
        println!(
            "  Throughput: {:.2} tokens/sec",
            parallel_result.tokens_per_sec
        );
        println!("  Prefill batches: {}", parallel_result.prefill_batches);
        println!("  Decode batches: {}", parallel_result.decode_batches);
        println!(
            "  Chunked prefill batches: {}",
            parallel_result.chunked_prefill_batches
        );
        println!(
            "  Padding efficiency: {:.1}%",
            parallel_result.padding_efficiency
        );

        println!(
            "\nSpeedup: {:.2}x {}",
            speedup,
            if speedup > 1.0 { "🚀" } else { "⚠️" }
        );

        results.push((batch_size, sequential_result, parallel_result, speedup));
    }

    // Print summary table
    print_summary_table(&results);

    Ok(())
}

struct BenchmarkResult {
    total_time: f64,
    forward_time_ms: f64,
    tokens_generated: usize,
    tokens_per_sec: f64,
    prefill_batches: usize,
    decode_batches: usize,
    chunked_prefill_batches: usize,
    padding_efficiency: f64,
}

fn benchmark_sequential(requests: Vec<Request>) -> anyhow::Result<BenchmarkResult> {
    let mut model = ModelManager::load(MODEL_PATH, 8, 512, Some("f32"))?;

    let mut batch: Vec<RequestContext> = requests.into_iter().map(RequestContext::new).collect();

    let start_time = Instant::now();

    // Run until all complete
    for _ in 0..50 {
        let active = batch.iter().filter(|ctx| ctx.should_continue()).count();
        if active == 0 {
            break;
        }
        model.forward_batch(&mut batch)?;
    }

    let total_time = start_time.elapsed().as_secs_f64();
    let stats = model.stats();

    Ok(BenchmarkResult {
        total_time,
        forward_time_ms: stats.total_forward_time_ms,
        tokens_generated: stats.total_tokens_generated,
        tokens_per_sec: stats.tokens_per_second(),
        prefill_batches: stats.prefill_requests,
        decode_batches: stats.decode_requests,
        chunked_prefill_batches: 0,
        padding_efficiency: 100.0,
    })
}

fn benchmark_parallel(requests: Vec<Request>) -> anyhow::Result<BenchmarkResult> {
    let mut model = ParallelModelManager::load(MODEL_PATH, 8, 512, Some("f32"), None)?;

    let mut batch: Vec<RequestContext> = requests.into_iter().map(RequestContext::new).collect();

    let start_time = Instant::now();

    // Run until all complete
    for _ in 0..50 {
        let active = batch.iter().filter(|ctx| ctx.should_continue()).count();
        if active == 0 {
            break;
        }
        model.forward_batch(&mut batch)?;
    }

    let total_time = start_time.elapsed().as_secs_f64();
    let stats = model.stats();

    Ok(BenchmarkResult {
        total_time,
        forward_time_ms: stats.total_forward_time_ms,
        tokens_generated: stats.total_tokens_generated,
        tokens_per_sec: stats.tokens_per_second(),
        prefill_batches: stats.prefill_batches,
        decode_batches: stats.decode_batches,
        chunked_prefill_batches: stats.chunked_prefill_batches,
        padding_efficiency: stats.padding_efficiency() * 100.0,
    })
}

fn print_summary_table(results: &[(usize, BenchmarkResult, BenchmarkResult, f64)]) {
    println!("\n{}", "=".repeat(80));
    println!("Summary: Performance Comparison");
    println!("{}", "=".repeat(80));

    println!(
        "\n{:<12} {:<15} {:<15} {:<15} {:<10}",
        "Batch Size", "Sequential", "Parallel", "Speedup", "Efficiency"
    );
    println!("{}", "-".repeat(80));

    for &(batch_size, ref seq, ref par, speedup) in results {
        println!(
            "{:<12} {:<15.2} {:<15.2} {:<15.2}x {:<10}",
            batch_size,
            seq.tokens_per_sec,
            par.tokens_per_sec,
            speedup,
            if speedup > 1.0 { "✓" } else { "✗" }
        );
    }

    println!("\n{}", "=".repeat(80));
    println!("Key Insights:");
    println!("{}", "=".repeat(80));

    // Calculate average speedup
    let avg_speedup: f64 = results.iter().map(|(_, _, _, s)| s).sum::<f64>() / results.len() as f64;

    println!("\n✓ Average speedup: {:.2}x", avg_speedup);

    if avg_speedup > 1.0 {
        println!("✓ Parallel batching is FASTER than sequential");
        println!("  → True parallel forward passes are working!");
        println!("  → Chunked prefill with padding is effective");
    } else {
        println!("⚠ Parallel is currently slower - this may be due to:");
        println!("  → CPU-bound operations (try GPU for better speedup)");
        println!("  → Small batch sizes (parallelism overhead)");
        println!("  → Implementation details needing optimization");
    }

    // Print parallel-specific insights
    if let Some((_, _, par, _)) = results.last() {
        println!("\nParallel Implementation Details:");
        println!("  Chunked prefill batches: {}", par.chunked_prefill_batches);
        println!("  Decode batches: {}", par.decode_batches);
        println!("  Padding efficiency: {:.1}%", par.padding_efficiency);
    }

    println!("\n{}", "=".repeat(80));
}
