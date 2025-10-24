//! Integration tests for ParallelModelManager
//!
//! Tests the production-ready parallel batched inference system with:
//! - Real llama-3b model
//! - Chunked prefill with padding
//! - True parallel batched forward passes
//! - Scattered KV cache management

use lightbulb::engine::{Request, RequestContext};
use lightbulb::model::{ChunkedPrefillConfig, ParallelModelManager};
use std::path::Path;

const MODEL_PATH: &str = "../models/llama-3b";

fn model_available() -> bool {
    Path::new(MODEL_PATH).exists()
}

#[test]
#[ignore]// Run with: cargo test --test parallel_model_manager_integration -- --ignored
fn test_parallel_model_loading() {
    if !model_available() {
        println!("Skipping test: model not found at {}", MODEL_PATH);
        return;
    }

    let result = ParallelModelManager::load(
        MODEL_PATH,
        4,    // max_batch_size
        512,  // context_length
        Some("f32"),
        None, // use default chunked prefill config
    );

    assert!(result.is_ok(), "Model loading failed: {:?}", result.err());

    let model = result.unwrap();
    println!("✓ Model loaded successfully");
    println!("  Hidden size: {}", model.config().hidden_size);
    println!("  Layers: {}", model.config().num_hidden_layers);
    println!("  Heads: {}", model.config().num_attention_heads);
}

#[test]
#[ignore]
fn test_parallel_single_request_generation() {
    if !model_available() {
        println!("Skipping test: model not found at {}", MODEL_PATH);
        return;
    }

    let mut model = ParallelModelManager::load(MODEL_PATH, 4, 512, Some("f32"), None)
        .expect("Failed to load model");

    let req = Request {
        id: "test-1".to_string(),
        prompt: "The capital of France is".to_string(),
        max_new_tokens: 5,
    };

    let mut batch = vec![RequestContext::new(req)];

    println!("\n=== Single Request Test ===");
    println!("Prompt: {}", batch[0].request.prompt);

    for step in 0..10 {
        let tokens = model.forward_batch(&mut batch).expect("Forward failed");

        if let Some(Some(token)) = tokens.get(0) {
            let decoded = model.decode(&batch[0].generated_tokens, false).unwrap();
            println!("Step {}: token={} text=\"{}\"", step, token, decoded);
        }

        if !batch[0].should_continue() {
            println!("✓ Generation complete at step {}", step);
            break;
        }
    }

    assert!(!batch[0].generated_tokens.is_empty(), "Should generate tokens");
    
    let final_text = model.decode(&batch[0].generated_tokens, false).unwrap();
    println!("\n✓ Final output: \"{}\"", final_text);
    println!("✓ Tokens generated: {}", batch[0].tokens_generated);
}

#[test]
#[ignore]
fn test_parallel_multi_request_batching() {
    if !model_available() {
        println!("Skipping test: model not found at {}", MODEL_PATH);
        return;
    }

    let mut model = ParallelModelManager::load(MODEL_PATH, 8, 512, Some("f32"), None)
        .expect("Failed to load model");

    let requests = vec![
        Request {
            id: "req-1".to_string(),
            prompt: "The capital of France is".to_string(),
            max_new_tokens: 8,
        },
        Request {
            id: "req-2".to_string(),
            prompt: "Rust is a programming".to_string(),
            max_new_tokens: 8,
        },
        Request {
            id: "req-3".to_string(),
            prompt: "Machine learning".to_string(),
            max_new_tokens: 8,
        },
    ];

    let mut batch: Vec<RequestContext> = requests.into_iter().map(RequestContext::new).collect();

    println!("\n=== Multi-Request Batching Test ===");
    println!("Batch size: {}", batch.len());

    for step in 0..15 {
        let active = batch.iter().filter(|ctx| ctx.should_continue()).count();
        if active == 0 {
            println!("\n✓ All requests completed at step {}", step);
            break;
        }

        model.forward_batch(&mut batch).expect("Forward failed");

        // Print progress
        for (i, ctx) in batch.iter().enumerate() {
            if ctx.tokens_generated > 0 && step % 3 == 0 {
                let text = model.decode(&ctx.generated_tokens, false).unwrap();
                println!("  Request {}: \"{}\"", i, text);
            }
        }
    }

    // Verify all completed
    for (i, ctx) in batch.iter().enumerate() {
        assert!(!ctx.generated_tokens.is_empty(), "Request {} should generate tokens", i);
        let text = model.decode(&ctx.generated_tokens, false).unwrap();
        println!("\n✓ Request {} final: \"{}\" (prompt: \"{}\")", 
                 i, text, ctx.request.prompt);
    }

    // Print statistics
    let stats = model.stats();
    println!("\n=== Statistics ===");
    println!("Total batches: {}", stats.total_batches);
    println!("Prefill batches: {}", stats.prefill_batches);
    println!("Decode batches: {}", stats.decode_batches);
    println!("Chunked prefill batches: {}", stats.chunked_prefill_batches);
    println!("Max batch size seen: {}", stats.max_batch_size);
    println!("Tokens generated: {}", stats.total_tokens_generated);
    println!("Padding efficiency: {:.1}%", stats.padding_efficiency() * 100.0);
    println!("Throughput: {:.2} tokens/sec", stats.tokens_per_second());
}

#[test]
#[ignore]
fn test_chunked_prefill_with_variable_lengths() {
    if !model_available() {
        println!("Skipping test: model not found at {}", MODEL_PATH);
        return;
    }

    // Configure chunked prefill with small chunk size for testing
    let chunked_config = ChunkedPrefillConfig {
        chunk_size: 64,  // Small chunks to test batching
        max_batch_size: 4,
        pad_token_id: 0,
    };

    let mut model = ParallelModelManager::load(
        MODEL_PATH,
        4,
        512,
        Some("f32"),
        Some(chunked_config),
    )
    .expect("Failed to load model");

    // Create requests with very different prompt lengths
    let requests = vec![
        Request {
            id: "short".to_string(),
            prompt: "Hello".to_string(),  // Very short
            max_new_tokens: 5,
        },
        Request {
            id: "medium".to_string(),
            prompt: "The quick brown fox jumps over the lazy dog. This is a test.".to_string(),
            max_new_tokens: 5,
        },
        Request {
            id: "long".to_string(),
            prompt: "Artificial intelligence and machine learning are transforming the world. \
                    Deep learning models have achieved remarkable success in natural language processing, \
                    computer vision, and many other domains.".to_string(),
            max_new_tokens: 5,
        },
    ];

    let mut batch: Vec<RequestContext> = requests.into_iter().map(RequestContext::new).collect();

    println!("\n=== Chunked Prefill Test ===");
    println!("Chunk size: 64");
    
    // Tokenize to see lengths
    for ctx in &batch {
        let tokens = model.tokenize(&ctx.request.prompt, true).unwrap();
        println!("  {} tokens: {} tokens", ctx.request.id, tokens.len());
    }

    // Process through completion
    for step in 0..20 {
        let active = batch.iter().filter(|ctx| ctx.should_continue()).count();
        if active == 0 {
            break;
        }
        model.forward_batch(&mut batch).expect("Forward failed");
    }

    // Verify results
    for ctx in &batch {
        assert!(!ctx.generated_tokens.is_empty());
        let text = model.decode(&ctx.generated_tokens, false).unwrap();
        println!("\n✓ {}: \"{}\"", ctx.request.id, text);
    }

    // Check chunked prefill was used
    let stats = model.stats();
    println!("\n=== Chunked Prefill Statistics ===");
    println!("Chunked prefill batches: {}", stats.chunked_prefill_batches);
    println!("Total padding tokens: {}", stats.total_padding_tokens);
    println!("Padding efficiency: {:.1}%", stats.padding_efficiency() * 100.0);
    
    assert!(stats.chunked_prefill_batches > 0, "Should use chunked prefill");
}

#[test]
#[ignore]
fn test_parallel_correctness_vs_sequential() {
    if !model_available() {
        println!("Skipping test: model not found at {}", MODEL_PATH);
        return;
    }

    use lightbulb::model::ModelManager;

    let prompt = "The capital of France is";
    let max_tokens = 5;

    // Run with ParallelModelManager
    let mut parallel_model = ParallelModelManager::load(MODEL_PATH, 4, 512, Some("f32"), None)
        .expect("Failed to load parallel model");
    
    let req1 = Request {
        id: "test-parallel".to_string(),
        prompt: prompt.to_string(),
        max_new_tokens: max_tokens,
    };
    let mut batch1 = vec![RequestContext::new(req1)];

    for _ in 0..10 {
        parallel_model.forward_batch(&mut batch1).expect("Forward failed");
        if !batch1[0].should_continue() {
            break;
        }
    }

    let parallel_tokens = batch1[0].generated_tokens.clone();
    let parallel_text = parallel_model.decode(&parallel_tokens, false).unwrap();

    // Run with sequential ModelManager
    let mut sequential_model = ModelManager::load(MODEL_PATH, 4, 512, Some("f32"))
        .expect("Failed to load sequential model");
    
    let req2 = Request {
        id: "test-sequential".to_string(),
        prompt: prompt.to_string(),
        max_new_tokens: max_tokens,
    };
    let mut batch2 = vec![RequestContext::new(req2)];

    for _ in 0..10 {
        sequential_model.forward_batch(&mut batch2).expect("Forward failed");
        if !batch2[0].should_continue() {
            break;
        }
    }

    let sequential_tokens = batch2[0].generated_tokens.clone();
    let sequential_text = sequential_model.decode(&sequential_tokens, false).unwrap();

    println!("\n=== Correctness Comparison ===");
    println!("Prompt: \"{}\"", prompt);
    println!("\nParallel output:   \"{}\"", parallel_text);
    println!("Sequential output: \"{}\"", sequential_text);
    println!("\nParallel tokens:   {:?}", parallel_tokens);
    println!("Sequential tokens: {:?}", sequential_tokens);

    // Tokens should match exactly (deterministic generation with same seed)
    assert_eq!(
        parallel_tokens, sequential_tokens,
        "Parallel and sequential should produce identical results"
    );
    
    println!("\n✓ Outputs match perfectly!");
}

#[test]
#[ignore]
fn test_parallel_performance_metrics() {
    if !model_available() {
        println!("Skipping test: model not found at {}", MODEL_PATH);
        return;
    }

    let mut model = ParallelModelManager::load(MODEL_PATH, 8, 512, Some("f32"), None)
        .expect("Failed to load model");

    // Create a substantial batch
    let requests: Vec<Request> = (0..6)
        .map(|i| Request {
            id: format!("perf-{}", i),
            prompt: match i % 3 {
                0 => "The capital of France is".to_string(),
                1 => "Rust programming language".to_string(),
                _ => "Machine learning".to_string(),
            },
            max_new_tokens: 10,
        })
        .collect();

    let mut batch: Vec<RequestContext> = requests.into_iter().map(RequestContext::new).collect();

    println!("\n=== Performance Metrics Test ===");
    println!("Batch size: {}", batch.len());

    let start_time = std::time::Instant::now();

    for step in 0..25 {
        let active = batch.iter().filter(|ctx| ctx.should_continue()).count();
        if active == 0 {
            println!("All requests completed at step {}", step);
            break;
        }
        model.forward_batch(&mut batch).expect("Forward failed");
    }

    let total_time = start_time.elapsed();

    // Print comprehensive statistics
    let stats = model.stats();
    println!("\n=== Performance Statistics ===");
    println!("Total wall time: {:.2}s", total_time.as_secs_f64());
    println!("Total batches: {}", stats.total_batches);
    println!("  Prefill batches: {}", stats.prefill_batches);
    println!("  Decode batches: {}", stats.decode_batches);
    println!("  Chunked prefill batches: {}", stats.chunked_prefill_batches);
    println!("Max concurrent batch size: {}", stats.max_batch_size);
    println!("Total tokens generated: {}", stats.total_tokens_generated);
    println!("Total requests processed: {}", stats.total_requests_processed);
    println!("Average batch size: {:.2}", stats.average_batch_size());
    println!("Forward time: {:.2}ms", stats.total_forward_time_ms);
    println!("Throughput: {:.2} tokens/sec", stats.tokens_per_second());
    println!("Padding efficiency: {:.1}%", stats.padding_efficiency() * 100.0);

    // Verify all completed successfully
    for ctx in &batch {
        assert!(!ctx.generated_tokens.is_empty());
    }

    println!("\n✓ All performance metrics collected successfully");
}
