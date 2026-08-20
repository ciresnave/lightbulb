//! Performance Benchmark: Batched vs Sequential Inference
//!
//! This benchmark compares:
//! 1. Sequential processing (1 request at a time)
//! 2. Batched processing (multiple requests in parallel)
//!
//! Measures:
//! - Total time to completion
//! - Tokens per second (throughput)
//! - Average latency per request
//! - Time to first token
//!
//! Run with:
//! ```bash
//! cargo run --example performance_benchmark --release
//! ```

use anyhow::Result;
use lightbulb::engine::{Request, RequestContext, RequestState};
use lightbulb::model::ParallelModelManager;
use std::time::Instant;

#[derive(Debug)]
struct BenchmarkResult {
    mode: String,
    num_requests: usize,
    total_time_ms: u128,
    total_tokens: usize,
    tokens_per_second: f64,
    avg_latency_ms: f64,
    time_to_first_token_ms: u128,
}

impl BenchmarkResult {
    fn print(&self) {
        println!("\n📊 {} Results:", self.mode);
        println!("  Requests: {}", self.num_requests);
        println!("  Total time: {}ms", self.total_time_ms);
        println!("  Total tokens: {}", self.total_tokens);
        println!("  Throughput: {:.2} tokens/sec", self.tokens_per_second);
        println!("  Avg latency: {:.2}ms per request", self.avg_latency_ms);
        println!("  Time to first token: {}ms", self.time_to_first_token_ms);
    }
}

fn benchmark_sequential(
    model_manager: &mut ParallelModelManager,
    requests: Vec<Request>,
) -> Result<BenchmarkResult> {
    println!("\n🔄 Running SEQUENTIAL benchmark...");

    let num_requests = requests.len();
    let mut total_tokens = 0;
    let start = Instant::now();
    let mut time_to_first_token_ms = 0;
    let mut first_token_recorded = false;

    // Process each request one at a time
    for (i, req) in requests.into_iter().enumerate() {
        let mut batch = vec![RequestContext::new(req)];

        loop {
            // Check if completed
            if batch[0].state == RequestState::Completed {
                break;
            }

            let step_start = Instant::now();
            model_manager.forward_batch(&mut batch)?;

            // Record time to first token
            if !first_token_recorded && !batch[0].generated_tokens.is_empty() {
                time_to_first_token_ms = step_start.elapsed().as_millis();
                first_token_recorded = true;
            }
        }

        total_tokens += batch[0].generated_tokens.len();
        print!(".");
        if (i + 1) % 10 == 0 {
            print!(" {}/{}", i + 1, num_requests);
        }
    }
    println!();

    let total_time_ms = start.elapsed().as_millis();
    let tokens_per_second = (total_tokens as f64) / (total_time_ms as f64 / 1000.0);
    let avg_latency_ms = total_time_ms as f64 / num_requests as f64;

    Ok(BenchmarkResult {
        mode: "SEQUENTIAL".to_string(),
        num_requests,
        total_time_ms,
        total_tokens,
        tokens_per_second,
        avg_latency_ms,
        time_to_first_token_ms,
    })
}

fn benchmark_batched(
    model_manager: &mut ParallelModelManager,
    requests: Vec<Request>,
    batch_size: usize,
) -> Result<BenchmarkResult> {
    println!(
        "\n⚡ Running BATCHED benchmark (batch_size={})...",
        batch_size
    );

    let num_requests = requests.len();
    let mut total_tokens = 0;
    let start = Instant::now();
    let mut time_to_first_token_ms = 0;
    let mut first_token_recorded = false;

    // Process requests in batches
    for chunk_start in (0..num_requests).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(num_requests);
        let chunk = &requests[chunk_start..chunk_end];

        let mut batch: Vec<RequestContext> =
            chunk.iter().cloned().map(RequestContext::new).collect();

        // Process this batch until all requests complete
        loop {
            // Check if all completed
            if batch.iter().all(|ctx| ctx.state == RequestState::Completed) {
                break;
            }

            let step_start = Instant::now();
            model_manager.forward_batch(&mut batch)?;

            // Record time to first token
            if !first_token_recorded && batch.iter().any(|ctx| !ctx.generated_tokens.is_empty()) {
                time_to_first_token_ms = step_start.elapsed().as_millis();
                first_token_recorded = true;
            }
        }

        // Count tokens
        for ctx in &batch {
            total_tokens += ctx.generated_tokens.len();
        }

        print!(".");
        if (chunk_end) % (batch_size * 5) == 0 {
            print!(" {}/{}", chunk_end, num_requests);
        }
    }
    println!();

    let total_time_ms = start.elapsed().as_millis();
    let tokens_per_second = (total_tokens as f64) / (total_time_ms as f64 / 1000.0);
    let avg_latency_ms = total_time_ms as f64 / num_requests as f64;

    Ok(BenchmarkResult {
        mode: format!("BATCHED (batch_size={})", batch_size),
        num_requests,
        total_time_ms,
        total_tokens,
        tokens_per_second,
        avg_latency_ms,
        time_to_first_token_ms,
    })
}

fn create_test_requests(count: usize, tokens_per_request: usize) -> Vec<Request> {
    let prompts = vec![
        "The capital of France is",
        "Artificial intelligence will",
        "The quick brown fox",
        "In the year 2025",
        "Machine learning models",
        "The future of technology",
        "Climate change is",
        "Space exploration has",
        "Renewable energy sources",
        "Quantum computing will",
    ];

    (0..count)
        .map(|i| Request {
            id: format!("req-{}", i),
            prompt: prompts[i % prompts.len()].to_string(),
            max_new_tokens: tokens_per_request,
        })
        .collect()
}

fn main() -> Result<()> {
    println!("\n🔦 Lightbulb Performance Benchmark");
    println!("==================================\n");

    // Configuration
    let num_requests = 20;
    let tokens_per_request = 10;
    let batch_sizes = vec![1, 2, 4];

    println!("Configuration:");
    println!("  Model: llama-3b");
    println!("  Requests: {}", num_requests);
    println!("  Tokens per request: {}", tokens_per_request);
    println!(
        "  Expected total tokens: ~{}",
        num_requests * tokens_per_request
    );

    // Load model once
    println!("\n📂 Loading model...");
    let model_path = "../models/llama-3b";
    let max_batch_size = *batch_sizes.iter().max().unwrap();
    let mut model_manager =
        ParallelModelManager::load(model_path, max_batch_size, 512, Some("f32"), None)?;
    println!("✓ Model loaded\n");

    let mut results = Vec::new();

    // Benchmark 1: Sequential processing
    {
        let requests = create_test_requests(num_requests, tokens_per_request);
        model_manager.reset_stats();
        let result = benchmark_sequential(&mut model_manager, requests)?;
        result.print();
        results.push(result);
    }

    // Benchmark 2-N: Batched processing with different batch sizes
    for &batch_size in &batch_sizes {
        let requests = create_test_requests(num_requests, tokens_per_request);
        model_manager.reset_stats();
        let result = benchmark_batched(&mut model_manager, requests, batch_size)?;
        result.print();

        // Print batch statistics
        let stats = model_manager.stats();
        println!("  Batch stats:");
        // decode_batch_opportunities and max_concurrent_decodes were removed from
        // ParallelBatchStats; these are the surviving batch-shape counters.
        println!("    Total batches: {}", stats.total_batches);
        println!("    Decode batches: {}", stats.decode_batches);
        println!("    Max batch size: {}", stats.max_batch_size);

        results.push(result);
    }

    // Comparative analysis
    println!("\n{}", "=".repeat(60));
    println!("📈 COMPARATIVE ANALYSIS");
    println!("{}", "=".repeat(60));

    let sequential = &results[0];

    println!(
        "\n{:<25} {:>12} {:>15} {:>12}",
        "Mode", "Time (ms)", "Throughput", "Speedup"
    );
    println!("{}", "-".repeat(65));

    for result in &results {
        let speedup = sequential.total_time_ms as f64 / result.total_time_ms as f64;
        println!(
            "{:<25} {:>12} {:>12.2} t/s {:>11.2}x",
            result.mode, result.total_time_ms, result.tokens_per_second, speedup
        );
    }

    // Best performer
    let best = results
        .iter()
        .max_by(|a, b| {
            a.tokens_per_second
                .partial_cmp(&b.tokens_per_second)
                .unwrap()
        })
        .unwrap();

    let speedup = best.tokens_per_second / sequential.tokens_per_second;

    println!("\n🏆 Best Configuration: {}", best.mode);
    println!("   Speedup: {:.2}x faster than sequential", speedup);
    println!(
        "   Throughput improvement: {:.2} → {:.2} tokens/sec",
        sequential.tokens_per_second, best.tokens_per_second
    );

    // Efficiency analysis
    println!("\n💡 Efficiency Analysis:");
    for result in &results[1..] {
        // Skip sequential
        let efficiency = (result.tokens_per_second / sequential.tokens_per_second)
            / (result
                .mode
                .chars()
                .filter(|c| c.is_numeric())
                .collect::<String>()
                .parse::<f64>()
                .unwrap_or(1.0));
        println!("   {}: {:.1}% efficiency", result.mode, efficiency * 100.0);
    }

    println!("\n✨ Benchmark complete!\n");

    Ok(())
}
