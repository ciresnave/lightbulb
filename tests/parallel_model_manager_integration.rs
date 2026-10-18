//! Integration tests for ParallelModelManager
//!
//! Tests the production-ready parallel batched inference system with:
//! - Real llama-3b model
//! - Chunked prefill with padding
//! - True parallel batched forward passes
//! - Scattered KV cache management

use lightbulb::engine::{Request, RequestContext, RequestState};
use lightbulb::model::{ChunkedPrefillConfig, ParallelModelManager};
use std::path::Path;

const MODEL_PATH: &str = "../models/llama-3b";

/// The checkpoint these tests need, or a LOUD FAILURE.
///
/// # Why this panics instead of returning a bool
///
/// Every test here opened with a skip that RETURNED SUCCESS:
///
/// ```text
/// if !model_available() {
///     println!("Skipping test: model not found at {}", MODEL_PATH);
///     return;
/// }
/// ```
///
/// **Measured 2026-09-02**, and the worst instance in the repo:
/// `test_parallel_multi_request_batching` was NOT `#[ignore]`d, so it ran in
/// the ordinary `cargo test`, printed "Skipping test: model not found", and
/// reported `ok`. It was counted as a passing test in every green suite report
/// while executing no assertion at all.
///
/// A test that cannot run must say so by NOT RUNNING, never by passing.
/// `#[ignore]` is the honest gate; a skip that returns success is not, and it
/// survives removing the honest one.
///
/// `LIGHTBULB_LLAMA3B_MODEL` overrides the default path, matching how the GGUF
/// gates take their fixtures.
fn require_model() -> String {
    let path = std::env::var("LIGHTBULB_LLAMA3B_MODEL").unwrap_or_else(|_| MODEL_PATH.to_string());
    assert!(
        std::path::Path::new(&path).exists(),
        "checkpoint not found at {path:?}. These tests are #[ignore]d, so running them is deliberate and a missing checkpoint is a setup error rather than a reason to pass. Set LIGHTBULB_LLAMA3B_MODEL to a model directory."
    );
    path
}

#[test]
#[ignore] // Run with: cargo test --test parallel_model_manager_integration -- --ignored
fn test_parallel_model_loading() {
    let model_path = require_model();

    let result = ParallelModelManager::load(
        &model_path,
        4,   // max_batch_size
        512, // context_length
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
    let model_path = require_model();

    let mut model = ParallelModelManager::load(&model_path, 4, 512, Some("f32"), None)
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

    assert!(
        !batch[0].generated_tokens.is_empty(),
        "Should generate tokens"
    );

    let final_text = model.decode(&batch[0].generated_tokens, false).unwrap();
    println!("\n✓ Final output: \"{}\"", final_text);
    println!("✓ Tokens generated: {}", batch[0].tokens_generated);
}

#[test]
// Joined its five siblings behind `#[ignore]`: it needs the same checkpoint they
// do, and without it it was a vacuous pass counted in every green suite report.
#[ignore] // Run with: cargo test --test parallel_model_manager_integration -- --ignored
fn test_parallel_multi_request_batching() {
    let model_path = require_model();

    let mut model = ParallelModelManager::load(&model_path, 8, 512, Some("f32"), None)
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

    // Continuous batching requires fixed-size batch matching slot pool size (8)
    // Pad with dummy completed contexts for unused slots
    let mut batch: Vec<RequestContext> = requests.into_iter().map(RequestContext::new).collect();

    // Pad to 8 slots (slot pool size)
    while batch.len() < 8 {
        let dummy = RequestContext {
            request: Request {
                id: format!("dummy-{}", batch.len()),
                prompt: String::new(),
                max_new_tokens: 0,
            },
            state: RequestState::Completed, // Mark as completed so they're ignored
            tokens_generated: 0,
            position: 0,
            cache_index: None,
            generated_tokens: Vec::new(),
            temperature: 0.0,
            prompt_tokens: 0,
            stopped_on_eos: false,
            // Matches `RequestContext::new`'s default. These pads are
            // `Completed`, so nothing ever tokenizes their empty prompt.
            add_special_tokens: true,
        };
        batch.push(dummy);
    }

    let num_real_requests = 3; // Track how many are real vs padding

    // DON'T manually transition to Decoding!
    // Requests start as Pending, and forward_batch() handles:
    //   1. Slot allocation from pool
    //   2. Prefill (tokenize + process prompt)
    //   3. Automatic transition to Decoding
    //   4. Subsequent decode passes

    println!("\n=== Multi-Request Batching Test ===");
    println!(
        "Batch size: {} ({} real requests, {} padding)",
        batch.len(),
        num_real_requests,
        batch.len() - num_real_requests
    );
    println!("\nInitial prompts:");
    for (i, ctx) in batch.iter().enumerate().take(num_real_requests) {
        println!("  Request {}: \"{}\"", i, ctx.request.prompt);
    }

    // Process through completion
    for step in 0..20 {
        // Run forward pass - handles Pending → Decoding → Completed transitions
        model.forward_batch(&mut batch).expect("Forward failed");

        // Debug: print state after each step
        println!("\nAfter step {}:", step + 1);
        for (i, ctx) in batch.iter().enumerate() {
            println!(
                "  Request {}: state={:?}, tokens_generated={}, generated_tokens.len()={}",
                i,
                ctx.state,
                ctx.tokens_generated,
                ctx.generated_tokens.len()
            );

            // Also print generated text every few steps
            if ctx.tokens_generated > 0 && step % 3 == 0 {
                let text = model.decode(&ctx.generated_tokens, false).unwrap();
                println!("    Generated text: \"{}\"", text);
            }
        }

        // Check if any requests are still active (Pending or Decoding)
        let active = batch
            .iter()
            .filter(|ctx| ctx.state != RequestState::Completed)
            .count();

        if active == 0 {
            println!("\n✓ All requests completed at step {}", step + 1);
            break;
        }
    }

    // Verify all real requests completed successfully
    for (i, ctx) in batch.iter().enumerate().take(num_real_requests) {
        assert!(
            !ctx.generated_tokens.is_empty(),
            "Request {} should generate tokens",
            i
        );
        let text = model.decode(&ctx.generated_tokens, false).unwrap();
        println!(
            "\n✓ Request {} final: \"{}\" (prompt: \"{}\")",
            i, text, ctx.request.prompt
        );
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
    println!(
        "Padding efficiency: {:.1}%",
        stats.padding_efficiency() * 100.0
    );
    println!("Throughput: {:.2} tokens/sec", stats.tokens_per_second());
}

#[test]
#[ignore]
fn test_chunked_prefill_with_variable_lengths() {
    let model_path = require_model();

    // Configure chunked prefill with small chunk size for testing
    let chunked_config = ChunkedPrefillConfig {
        chunk_size: 64, // Small chunks to test batching
        max_batch_size: 4,
        pad_token_id: 0,
    };

    let mut model =
        ParallelModelManager::load(&model_path, 4, 512, Some("f32"), Some(chunked_config))
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
    println!(
        "Padding efficiency: {:.1}%",
        stats.padding_efficiency() * 100.0
    );

    assert!(
        stats.chunked_prefill_batches > 0,
        "Should use chunked prefill"
    );
}

/// M4 slice 5.1: does a cache index freed by one completed request corrupt
/// the NEXT, unrelated request that reuses it?
///
/// `EngineModel::step_batch`'s trait doc (`src/engine/model_runner.rs`) warns
/// that a scheduler stepping a changing set of live requests per call can
/// silently corrupt cache state across requests -- citing a *now-deleted*
/// implementor as the one that broke this way. `ParallelModelManager::
/// forward_batch` looks built to avoid it (`release_cache_index` runs the
/// same call a request completes, `src/model/parallel_model_manager.rs:1395`
/// and `:1736`) -- but that is a reading of the code, not a measurement. This
/// test is the measurement, before any of slice 5.2+ depends on the answer.
///
/// Method: run request A alone to completion (freeing its cache index), then
/// submit an unrelated request B into a FRESH `Vec` -- the same shape
/// `run_jobs` will produce once 5.2+ lands (a job leaves the active set, a
/// later one arrives, nothing carries the old `RequestContext` forward). B
/// almost certainly reuses A's just-freed cache index (`max_batch_size: 1`
/// forces it -- there is no other index to allocate). The control is a
/// SEPARATE, freshly loaded model instance given a batch of ONLY B, so
/// nothing about A ever touches it. At `temperature: 0.0` (greedy, the
/// default), decoding is deterministic: if cache reuse leaked any of A's KV
/// state into B's slot, B's tokens diverge from the control. If the
/// bookkeeping is correct, they must match bit-for-bit.
///
/// Needs the TinyLlama checkpoint (see `test_models::tinyllama_dir`). **Not**
/// `#[ignore]`d, per CireSnave's standing rule (2026-10-16, EXPECTATIONS
/// §5.1g/5.1h): a test that can't run on a GitHub-hosted runner but can run
/// on the self-hosted one runs there, under its own job, rather than being
/// silently excluded from every run via `#[ignore]`. Hosted CI jobs that run
/// this whole test file must `--skip` it by name until that job exists (the
/// PM is routing the self-hosted job spec and the checkpoint's provenance;
/// see the PR this landed in). Run by hand meanwhile:
/// `cargo test --test parallel_model_manager_integration \
///   cache_index_reused_by_a_later_unrelated_request_is_not_corrupted`
///
/// Measured (not merely written), 2026-10-17, CPU, `f32`: **passes.**
/// `B after A (reused slot): [27441, 1524, 29889, 13] "Jupiter.\n"` ==
/// `B alone (control): [27441, 1524, 29889, 13] "Jupiter.\n"`, bit-for-bit.
/// 520.95s wall time for the one test (model load x2 + ~8 decode steps on
/// CPU f32) -- slow, but this runs on its own self-hosted job, not inline
/// with the rest of the suite.
const SLOT_REUSE_TEST_MAX_BATCH_SIZE: usize = 1; // forces B onto the exact index A just freed
const SLOT_REUSE_TEST_CONTEXT_LENGTH: usize = 256;
const SLOT_REUSE_TEST_STEP_BUDGET: usize = 32;

fn slot_reuse_test_model(model_dir: &std::path::Path) -> ParallelModelManager {
    ParallelModelManager::load(
        model_dir,
        SLOT_REUSE_TEST_MAX_BATCH_SIZE,
        SLOT_REUSE_TEST_CONTEXT_LENGTH,
        Some("f32"),
        None,
    )
    .expect("failed to load TinyLlama")
}

/// Steps `model` on the single request in `batch` until it stops continuing.
/// A budget exhausted without completion is a test setup failure (loud), not
/// a silent return -- every call site's assertions depend on actually
/// reaching completion.
fn step_single_request_to_completion(
    model: &mut ParallelModelManager,
    batch: &mut [RequestContext],
) {
    for _ in 0..SLOT_REUSE_TEST_STEP_BUDGET {
        model.forward_batch(batch).expect("forward_batch failed");
        if !batch[0].should_continue() {
            return;
        }
    }
    panic!(
        "request {} did not reach completion within {SLOT_REUSE_TEST_STEP_BUDGET} steps",
        batch[0].request.id
    );
}

fn b_request() -> Request {
    Request {
        id: "req-b".to_string(),
        prompt: "The largest planet in the solar system is".to_string(),
        max_new_tokens: 4,
    }
}

#[test]
fn cache_index_reused_by_a_later_unrelated_request_is_not_corrupted() {
    let model_dir = lightbulb::test_models::tinyllama_dir().expect(
        "TinyLlama checkpoint not found -- set TINYLLAMA_DIR, or see test_models::describe() \
         for the other resolution paths this test tries",
    );

    // --- Run A to completion, then B into a fresh Vec on the SAME model. ---
    let mut model = slot_reuse_test_model(&model_dir);

    let a_request = Request {
        id: "req-a".to_string(),
        prompt: "Rust is a systems programming language that".to_string(),
        max_new_tokens: 4,
    };
    let mut a_batch = vec![RequestContext::new(a_request)];
    step_single_request_to_completion(&mut model, &mut a_batch);
    // `a_batch[0].cache_index` itself is NOT cleared by `complete()` -- it only
    // flips `state` (`src/engine/mod.rs::RequestContext::complete`). The real
    // release is pool-internal (`self.cache_index_pool[cache_idx] = false`,
    // `parallel_model_manager.rs:1396`), invisible from the caller's
    // `RequestContext`. A completed context is discarded, never read again, so
    // that stale field is harmless -- this test does not assert on it; it
    // asserts on what the POOL does with the freed slot, below.
    drop(a_batch); // nothing carries forward into B's batch, matching run_jobs's future shape

    let mut b_batch = vec![RequestContext::new(b_request())];
    model
        .forward_batch(&mut b_batch)
        .expect("forward_batch failed for B's first step (after A)");
    assert_eq!(
        b_batch[0].cache_index,
        Some(0),
        "with max_batch_size=1, B's prefill must reuse the only index the pool has -- the exact \
         index A just freed. If this is anything else, the scenario this test exists to check \
         (slot reuse) never actually happened"
    );
    if b_batch[0].should_continue() {
        step_single_request_to_completion(&mut model, &mut b_batch);
    }
    let b_after_a_tokens = b_batch[0].generated_tokens.clone();
    let b_after_a_text = model.decode(&b_after_a_tokens, false).unwrap();

    // --- Control: B alone, on a SEPARATE model instance A never touched. ---
    let mut control_model = slot_reuse_test_model(&model_dir);
    let mut control_batch = vec![RequestContext::new(b_request())];
    step_single_request_to_completion(&mut control_model, &mut control_batch);
    let b_control_tokens = control_batch[0].generated_tokens.clone();
    let b_control_text = control_model.decode(&b_control_tokens, false).unwrap();

    println!("B after A (reused slot): {b_after_a_tokens:?} \"{b_after_a_text}\"");
    println!("B alone   (control):     {b_control_tokens:?} \"{b_control_text}\"");

    assert_eq!(
        b_after_a_tokens, b_control_tokens,
        "request B produced different tokens when it reused A's freed cache index than when it \
         ran alone on a fresh model instance -- A's KV state leaked into B's slot. Both runs are \
         greedy (temperature 0.0, the default), so this is not sampling noise."
    );
}

#[test]
#[ignore]
fn test_parallel_deterministic_consistency() {
    let model_path = require_model();

    let prompt = "The capital of France is";
    let max_tokens = 5;

    // Run twice with ParallelModelManager to verify deterministic output
    let mut model1 = ParallelModelManager::load(&model_path, 4, 512, Some("f32"), None)
        .expect("Failed to load model (run 1)");

    let req1 = Request {
        id: "test-run1".to_string(),
        prompt: prompt.to_string(),
        max_new_tokens: max_tokens,
    };
    let mut batch1 = vec![RequestContext::new(req1)];

    for _ in 0..10 {
        model1.forward_batch(&mut batch1).expect("Forward failed");
        if !batch1[0].should_continue() {
            break;
        }
    }

    let tokens1 = batch1[0].generated_tokens.clone();
    let text1 = model1.decode(&tokens1, false).unwrap();

    let mut model2 = ParallelModelManager::load(&model_path, 4, 512, Some("f32"), None)
        .expect("Failed to load model (run 2)");

    let req2 = Request {
        id: "test-run2".to_string(),
        prompt: prompt.to_string(),
        max_new_tokens: max_tokens,
    };
    let mut batch2 = vec![RequestContext::new(req2)];

    for _ in 0..10 {
        model2.forward_batch(&mut batch2).expect("Forward failed");
        if !batch2[0].should_continue() {
            break;
        }
    }

    let tokens2 = batch2[0].generated_tokens.clone();
    let text2 = model2.decode(&tokens2, false).unwrap();

    println!("\n=== Deterministic Consistency ===");
    println!("Prompt: \"{}\"", prompt);
    println!("\nRun 1: \"{}\" ({:?})", text1, tokens1);
    println!("Run 2: \"{}\" ({:?})", text2, tokens2);

    assert_eq!(
        tokens1, tokens2,
        "Two runs should produce identical results"
    );

    println!("\n✓ Outputs match!");
}

#[test]
#[ignore]
fn test_parallel_performance_metrics() {
    let model_path = require_model();

    // Use load_adaptive() to let the hardware detection pick the optimal batch size
    let mut model = ParallelModelManager::load_adaptive(MODEL_PATH, 512, Some("f32"), None)
        .expect("Failed to load model");

    let adaptive_batch_size = model.get_adaptive_batch_size().unwrap_or(8);

    // Test with 100 requests to measure maximum throughput
    let num_requests = 100;

    // Use RequestQueue to properly manage requests and avoid cache index exhaustion
    use lightbulb::engine::{BatchAssembler, BatchConfig, Request, RequestQueue};
    let queue = RequestQueue::new();
    let batch_config = BatchConfig::new(adaptive_batch_size, 2048);
    let assembler = BatchAssembler::new(batch_config);

    // Submit all requests to the queue
    for i in 0..num_requests {
        let req = Request {
            id: format!("perf-{}", i),
            prompt: match i % 3 {
                0 => "The capital of France is".to_string(),
                1 => "Rust programming language".to_string(),
                _ => "Machine learning".to_string(),
            },
            max_new_tokens: 10,
        };
        queue.submit(req).unwrap();
    }

    println!("\n=== Performance Metrics Test with Adaptive Batch Sizing ===");
    println!("Requests to process: {}", num_requests);
    println!("Adaptive batch size: {}", adaptive_batch_size);

    let start_time = std::time::Instant::now();
    let mut active_batch: Vec<RequestContext> = Vec::new();

    // Process until queue is empty and all active requests are complete
    for step in 0..500 {
        // Remove completed requests from active batch
        active_batch.retain(|ctx| ctx.state != RequestState::Completed);

        // Try to fill the batch from the queue
        while active_batch.len() < adaptive_batch_size && !queue.is_empty() {
            if let Some(ctx) = queue.pop() {
                active_batch.push(ctx);
            } else {
                break;
            }
        }

        // If no active requests, we're done
        if active_batch.is_empty() {
            println!("All requests completed at step {}", step);
            break;
        }

        // Process the batch
        model
            .forward_batch(&mut active_batch)
            .expect("Forward failed");
    }

    let total_time = start_time.elapsed();

    // Print comprehensive statistics
    let stats = model.stats();
    println!("\n=== Performance Statistics ===");
    println!("Total wall time: {:.2}s", total_time.as_secs_f64());
    println!("Total batches: {}", stats.total_batches);
    println!("  Prefill batches: {}", stats.prefill_batches);
    println!("  Decode batches: {}", stats.decode_batches);
    println!(
        "  Chunked prefill batches: {}",
        stats.chunked_prefill_batches
    );
    println!("Max concurrent batch size: {}", stats.max_batch_size);
    println!("Total tokens generated: {}", stats.total_tokens_generated);
    println!(
        "Total requests processed: {}",
        stats.total_requests_processed
    );
    println!("Average batch size: {:.2}", stats.average_batch_size());
    println!("Forward time: {:.2}ms", stats.total_forward_time_ms);
    println!("Throughput: {:.2} tokens/sec", stats.tokens_per_second());
    println!(
        "Padding efficiency: {:.1}%",
        stats.padding_efficiency() * 100.0
    );

    // Get prefix cache statistics
    let cache_stats = model.prefix_cache_stats();
    println!("\n=== Prefix Cache Statistics ===");
    println!("Cache hits: {}", cache_stats.hits);
    println!("Cache misses: {}", cache_stats.misses);
    println!("Hit rate: {:.1}%", cache_stats.hit_rate() * 100.0);
    println!(
        "Tokens saved: {} ({:.1} avg per hit)",
        cache_stats.total_saved_tokens,
        cache_stats.avg_saved_tokens()
    );
    println!(
        "Potential TTFT improvement: {:.1}% (if KV caching enabled)",
        cache_stats.hit_rate() * 100.0 * 0.5 // Conservative estimate
    );

    // Verify we processed all requests
    assert_eq!(
        stats.total_requests_processed, num_requests,
        "Should have processed all {} requests",
        num_requests
    );

    println!(
        "\n✓ Performance test complete with {:.2} tokens/sec",
        stats.tokens_per_second()
    );
}
