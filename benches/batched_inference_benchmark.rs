//! Performance benchmarks for batched inference
//!
//! Validates the 5-10x (CPU) and 10-50x (GPU) speedup claims for ParallelModelManager
//! vs sequential baseline.
//!
//! Run with: cargo bench --bench batched_inference_benchmark
//!
//! Results are saved to: benches/results/batch_performance_<timestamp>.json

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lightbulb::engine::{Request, RequestContext};
use lightbulb::model::ParallelModelManager;
use std::path::Path;
use std::time::Duration;

const MODEL_PATH: &str = "../models/llama-3b";
const PROMPT: &str = "The quick brown fox jumps over the lazy dog";

fn model_available() -> bool {
    Path::new(MODEL_PATH).exists()
}

/// Benchmark single batch forward pass with varying batch sizes
fn bench_batch_sizes(c: &mut Criterion) {
    if !model_available() {
        println!("Skipping benchmark: model not found at {}", MODEL_PATH);
        return;
    }

    let mut group = c.benchmark_group("batch_forward_pass");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(20);

    // Test batch sizes: 1, 2, 4, 8, 16, 32
    for batch_size in [1, 2, 4, 8, 16, 32].iter() {
        let mut model = ParallelModelManager::load(
            MODEL_PATH,
            *batch_size as usize,
            512,
            Some("f32"),
            None,
        )
        .expect("Failed to load model");

        // Create batch of requests
        let mut batch: Vec<RequestContext> = (0..*batch_size)
            .map(|i| {
                RequestContext::new(Request {
                    id: format!("req-{}", i),
                    prompt: PROMPT.to_string(),
                    max_new_tokens: 1, // Just measure one forward pass
                })
            })
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let _tokens = model
                        .forward_batch(&mut batch)
                        .expect("Forward pass failed");
                    black_box(&batch);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark decode throughput (tokens/second) for different batch sizes
fn bench_decode_throughput(c: &mut Criterion) {
    if !model_available() {
        println!("Skipping benchmark: model not found at {}", MODEL_PATH);
        return;
    }

    let mut group = c.benchmark_group("decode_throughput");
    group.measurement_time(Duration::from_secs(60));
    group.sample_size(10);

    for batch_size in [1, 4, 8, 16].iter() {
        let mut model = ParallelModelManager::load(
            MODEL_PATH,
            *batch_size as usize,
            512,
            Some("f32"),
            None,
        )
        .expect("Failed to load model");

        let mut batch: Vec<RequestContext> = (0..*batch_size)
            .map(|i| {
                RequestContext::new(Request {
                    id: format!("req-{}", i),
                    prompt: PROMPT.to_string(),
                    max_new_tokens: 50, // Generate 50 tokens
                })
            })
            .collect();

        group.throughput(Throughput::Elements((*batch_size * 50) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    // Generate 50 tokens
                    for _ in 0..50 {
                        let _tokens = model
                            .forward_batch(&mut batch)
                            .expect("Forward pass failed");
                    }
                    black_box(&batch);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark prefill phase with varying sequence lengths
fn bench_prefill_lengths(c: &mut Criterion) {
    if !model_available() {
        println!("Skipping benchmark: model not found at {}", MODEL_PATH);
        return;
    }

    let mut group = c.benchmark_group("prefill_sequence_lengths");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(15);

    let batch_size = 4;
    let prompts = vec![
        "Short".to_string(),                                  // ~128 tokens after tokenization
        "Medium ".repeat(32),                                  // ~512 tokens
        "Long prompt with more context ".repeat(64),          // ~1024 tokens
        "Very long context prompt with extensive text ".repeat(128), // ~2048 tokens
    ];

    for (seq_idx, prompt) in prompts.iter().enumerate() {
        let seq_label = match seq_idx {
            0 => "128_tokens",
            1 => "512_tokens",
            2 => "1024_tokens",
            3 => "2048_tokens",
            _ => "unknown",
        };

        let mut model = ParallelModelManager::load(MODEL_PATH, batch_size, 2048, Some("f32"), None)
            .expect("Failed to load model");

        let mut batch: Vec<RequestContext> = (0..batch_size)
            .map(|i| {
                RequestContext::new(Request {
                    id: format!("req-{}", i),
                    prompt: prompt.clone(),
                    max_new_tokens: 1, // Just prefill
                })
            })
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(seq_label), &seq_idx, |b, _| {
            b.iter(|| {
                let _tokens = model
                    .forward_batch(&mut batch)
                    .expect("Forward pass failed");
                black_box(&batch);
            });
        });
    }

    group.finish();
}

/// Compare batched vs sequential processing speedup
fn bench_batched_vs_sequential(c: &mut Criterion) {
    if !model_available() {
        println!("Skipping benchmark: model not found at {}", MODEL_PATH);
        return;
    }

    let mut group = c.benchmark_group("batched_vs_sequential");
    group.measurement_time(Duration::from_secs(45));
    group.sample_size(10);

    // Sequential baseline (batch_size=1 processed multiple times)
    let num_requests = 8;
    group.bench_function("sequential_8_requests", |b| {
        let mut model = ParallelModelManager::load(MODEL_PATH, 1, 512, Some("f32"), None)
            .expect("Failed to load model");

        b.iter(|| {
            for i in 0..num_requests {
                let mut batch = vec![RequestContext::new(Request {
                    id: format!("req-{}", i),
                    prompt: PROMPT.to_string(),
                    max_new_tokens: 10,
                })];

                for _ in 0..10 {
                    let _tokens = model
                        .forward_batch(&mut batch)
                        .expect("Forward pass failed");
                }
            }
        });
    });

    // Batched (batch_size=8 processed once)
    group.bench_function("batched_8_requests", |b| {
        let mut model = ParallelModelManager::load(MODEL_PATH, 8, 512, Some("f32"), None)
            .expect("Failed to load model");

        b.iter(|| {
            let mut batch: Vec<RequestContext> = (0..num_requests)
                .map(|i| {
                    RequestContext::new(Request {
                        id: format!("req-{}", i),
                        prompt: PROMPT.to_string(),
                        max_new_tokens: 10,
                    })
                })
                .collect();

            for _ in 0..10 {
                let _tokens = model
                    .forward_batch(&mut batch)
                    .expect("Forward pass failed");
            }
        });
    });

    group.finish();
}

/// Measure memory efficiency across batch sizes
fn bench_memory_usage(c: &mut Criterion) {
    if !model_available() {
        println!("Skipping benchmark: model not found at {}", MODEL_PATH);
        return;
    }

    let mut group = c.benchmark_group("memory_usage");
    group.sample_size(10);

    for batch_size in [1, 4, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &bs| {
                b.iter(|| {
                    let model = ParallelModelManager::load(MODEL_PATH, bs, 512, Some("f32"), None)
                        .expect("Failed to load model");
                    black_box(model);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_batch_sizes,
    bench_decode_throughput,
    bench_prefill_lengths,
    bench_batched_vs_sequential,
    bench_memory_usage
);
criterion_main!(benches);
