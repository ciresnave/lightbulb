//! Benchmark: Optimal Chunk Size for Multi-Chunk Prefill
//!
//! This benchmark tests different chunk sizes to find the optimal value
//! that balances padding waste vs transfer/launch overhead.
//!
//! Measures:
//! - Throughput (tokens/second)
//! - Padding efficiency (actual_tokens / total_tokens)
//! - Time to first token (latency)
//! - Memory usage
//!
//! Run with:
//! ```bash
//! cargo run --example benchmark_chunk_sizes --release
//! ```

use anyhow::Result;
use lightbulb::engine::{Request, RequestContext};
use lightbulb::model::{ChunkedPrefillConfig, ParallelModelManager};
use std::path::Path;
use std::time::Instant;

#[derive(Debug)]
struct ChunkSizeBenchmark {
    chunk_size: usize,
    alignment: usize,
    throughput_tokens_per_sec: f64,
    padding_efficiency: f64,
    avg_time_to_first_token_ms: f64,
    total_time_ms: u128,
}

impl ChunkSizeBenchmark {
    fn print(&self) {
        println!(
            "\n📊 Chunk Size: {} (align: {})",
            self.chunk_size, self.alignment
        );
        println!(
            "  Throughput:        {:.2} tokens/sec",
            self.throughput_tokens_per_sec
        );
        println!(
            "  Padding Efficiency: {:.1}%",
            self.padding_efficiency * 100.0
        );
        println!(
            "  Avg TTFT:          {:.2}ms",
            self.avg_time_to_first_token_ms
        );
        println!("  Total Time:        {}ms", self.total_time_ms);
    }
}

fn benchmark_chunk_size(
    model_path: &Path,
    chunk_size: usize,
    alignment: usize,
    test_prompts: &[String],
    max_tokens: usize,
) -> Result<ChunkSizeBenchmark> {
    // Create config with specific chunk size
    let config = ChunkedPrefillConfig {
        chunk_size,
        max_batch_size: 4,
        // `pad_last_chunk` was removed from ChunkedPrefillConfig; padding of the
        // final chunk is no longer configurable here.
        pad_token_id: 0,
    };

    let mut manager = ParallelModelManager::load(
        model_path,
        4,           // max_batch_size
        512,         // context_length
        Some("f32"), // dtype
        Some(config),
    )?;

    // Create requests
    let mut batch: Vec<RequestContext> = test_prompts
        .iter()
        .enumerate()
        .map(|(i, prompt)| {
            RequestContext::new(Request {
                id: format!("req-{}", i),
                prompt: prompt.clone(),
                max_new_tokens: max_tokens,
            })
        })
        .collect();

    let start = Instant::now();
    let mut total_tokens = 0;
    let mut total_ttft_ms = 0.0;
    let mut ttft_count = 0;

    // Run until all requests complete
    loop {
        // Check if all completed
        if batch.iter().all(|ctx| !ctx.should_continue()) {
            break;
        }

        let step_start = Instant::now();
        manager.forward_batch(&mut batch)?;

        // Track time to first token for each request
        for ctx in &batch {
            if ctx.generated_tokens.len() == 1 {
                total_ttft_ms += step_start.elapsed().as_secs_f64() * 1000.0;
                ttft_count += 1;
            }
        }
    }

    let total_time_ms = start.elapsed().as_millis();

    // Calculate metrics
    for ctx in &batch {
        total_tokens += ctx.generated_tokens.len();
    }

    let stats = manager.stats();
    let throughput = total_tokens as f64 / (total_time_ms as f64 / 1000.0);
    let avg_ttft = if ttft_count > 0 {
        total_ttft_ms / ttft_count as f64
    } else {
        0.0
    };

    // Calculate padding efficiency
    let actual_tokens = total_tokens + test_prompts.iter().map(|p| p.len()).sum::<usize>();
    let total_with_padding = actual_tokens + stats.total_padding_tokens;
    let padding_efficiency = actual_tokens as f64 / total_with_padding as f64;

    Ok(ChunkSizeBenchmark {
        chunk_size,
        alignment,
        throughput_tokens_per_sec: throughput,
        padding_efficiency,
        avg_time_to_first_token_ms: avg_ttft,
        total_time_ms,
    })
}

fn main() -> Result<()> {
    println!("🔬 Chunk Size Optimization Benchmark");
    println!("=====================================\n");

    let model_path = Path::new("models/llama-3b");

    // Test prompts with varying lengths
    let test_prompts = vec![
        "Explain quantum computing in simple terms.".to_string(),
        "Write a short story about a robot learning to paint. Include vivid descriptions and dialogue between the robot and its teacher.".to_string(),
        "What are the key differences between machine learning and deep learning?".to_string(),
        "Describe the process of photosynthesis step by step, including the light-dependent and light-independent reactions.".to_string(),
    ];

    println!("Test Configuration:");
    println!("  Prompts: {} varying lengths", test_prompts.len());
    println!("  Max tokens per request: 50");
    println!("  Device: CPU\n");

    // Test different chunk sizes
    let chunk_configs = vec![
        (64, 32),   // Small chunks, 32-aligned
        (128, 32),  // Medium-small, 32-aligned
        (256, 32),  // Medium, 32-aligned
        (512, 32),  // Current default
        (1024, 32), // Large chunks
        (128, 64),  // Test 64-alignment (AMD GPU)
        (256, 64),
        (128, 16), // Test 16-alignment (Tensor cores)
        (256, 16),
    ];

    let mut results = Vec::new();

    for (chunk_size, alignment) in chunk_configs {
        print!("Testing chunk_size={} align={} ... ", chunk_size, alignment);

        match benchmark_chunk_size(&model_path, chunk_size, alignment, &test_prompts, 50) {
            Ok(result) => {
                println!("✓");
                result.print();
                results.push(result);
            }
            Err(e) => {
                println!("✗ Error: {}", e);
            }
        }
    }

    // Find optimal configuration
    if !results.is_empty() {
        println!("\n🏆 Optimal Configuration:");

        let best_throughput = results
            .iter()
            .max_by(|a, b| {
                a.throughput_tokens_per_sec
                    .partial_cmp(&b.throughput_tokens_per_sec)
                    .unwrap()
            })
            .unwrap();

        let best_efficiency = results
            .iter()
            .max_by(|a, b| {
                a.padding_efficiency
                    .partial_cmp(&b.padding_efficiency)
                    .unwrap()
            })
            .unwrap();

        let best_latency = results
            .iter()
            .min_by(|a, b| {
                a.avg_time_to_first_token_ms
                    .partial_cmp(&b.avg_time_to_first_token_ms)
                    .unwrap()
            })
            .unwrap();

        println!(
            "  Best Throughput: chunk_size={} ({:.2} tok/s)",
            best_throughput.chunk_size, best_throughput.throughput_tokens_per_sec
        );
        println!(
            "  Best Efficiency: chunk_size={} ({:.1}% efficient)",
            best_efficiency.chunk_size,
            best_efficiency.padding_efficiency * 100.0
        );
        println!(
            "  Best Latency:    chunk_size={} ({:.2}ms TTFT)",
            best_latency.chunk_size, best_latency.avg_time_to_first_token_ms
        );
    }

    Ok(())
}
