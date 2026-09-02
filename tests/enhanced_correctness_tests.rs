//! Enhanced correctness tests for batched inference
//!
//! Verifies that BatchedTransformer produces identical outputs to sequential processing
//! across various scenarios:
//! - Variable sequence lengths
//! - Different batch sizes  
//! - KV cache consistency
//! - Attention masking
//! - Edge cases
//!
//! Run with: cargo test --test enhanced_correctness_tests -- --ignored

use candlelight::core::{DType, Device, Tensor};
use lightbulb::engine::{Request, RequestContext};
use lightbulb::model::ParallelModelManager;
use std::path::Path;

type Result<T> = anyhow::Result<T>;

const MODEL_PATH: &str = "../models/Qwen3-32B-AWQ";
const TOLERANCE_RTOL: f32 = 1e-4; // Relative tolerance
const TOLERANCE_ATOL: f32 = 1e-5; // Absolute tolerance

/// The checkpoint these tests need, or a LOUD FAILURE.
///
/// # Why this panics instead of returning a bool
///
/// Every test here used to open with a skip that RETURNED SUCCESS:
///
/// ```text
/// if !model_available() {
///     println!("Skipping test: model not found");
///     return Ok(());
/// }
/// ```
///
/// **Measured 2026-09-02.** Run exactly as this file's header documents —
/// `cargo test --test enhanced_correctness_tests -- --ignored` — that produced
/// `test result: ok. 8 passed; 0 failed` with ZERO assertions executed. The
/// `println!` is swallowed without `--nocapture`, so an operator following the
/// documented invocation saw eight green tests and nothing else. The only tell
/// was `finished in 0.00s`.
///
/// These tests are already `#[ignore]`d, so running them is a deliberate act.
/// A missing checkpoint is then a SETUP ERROR, not a reason to report success.
/// The `#[ignore]` is an honest gate — it does not run and does not claim to.
/// A skip that passes is the dishonest half, and it SURVIVES removing the
/// honest one: delete the `#[ignore]` and these go green while asserting
/// nothing.
///
/// `LIGHTBULB_AWQ_MODEL` overrides the default path, matching how
/// `gguf_serving_e2e` and the tokenizer fidelity gates take their fixtures.
fn require_model() -> String {
    let path = std::env::var("LIGHTBULB_AWQ_MODEL").unwrap_or_else(|_| MODEL_PATH.to_string());
    assert!(
        Path::new(&path).exists(),
        "checkpoint not found at {path:?}. These tests are #[ignore]d and were run explicitly, so a missing checkpoint is a setup error rather than a reason to pass. Set LIGHTBULB_AWQ_MODEL to an AWQ checkpoint directory."
    );
    path
}

/// Helper to compare generated tokens between two runs
fn tokens_match(a: &[u32], b: &[u32]) -> bool {
    if a.len() != b.len() {
        println!("Length mismatch: {} vs {}", a.len(), b.len());
        return false;
    }

    let mismatches: Vec<_> = a
        .iter()
        .zip(b.iter())
        .enumerate()
        .filter(|(_, (x, y))| x != y)
        .collect();

    if !mismatches.is_empty() {
        println!("Token mismatches:");
        for (idx, (x, y)) in mismatches.iter().take(10) {
            println!("  Position {}: {} vs {}", idx, x, y);
        }
        return false;
    }

    true
}

#[test]
#[ignore] // Requires model files
fn test_batch_vs_sequential_single_token() -> Result<()> {
    let model_path = require_model();

    println!("\n=== Test: Batch vs Sequential (Single Token) ===\n");

    let prompts = vec![
        "The capital of France is",
        "Rust is a programming language that",
        "Machine learning models",
    ];

    // Sequential processing (batch_size=1)
    let mut sequential_results = Vec::new();
    for prompt in &prompts {
        let mut model = ParallelModelManager::load(&model_path, 1, 512, None, None)?;
        let mut batch = vec![RequestContext::new(Request {
            id: "seq".to_string(),
            prompt: prompt.to_string(),
            max_new_tokens: 1,
        })];

        model.forward_batch(&mut batch)?;
        sequential_results.push(batch[0].generated_tokens.clone());
    }

    // Batched processing (batch_size=3)
    let mut model = ParallelModelManager::load(&model_path, 3, 512, None, None)?;
    let mut batch: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            RequestContext::new(Request {
                id: format!("batch-{}", i),
                prompt: p.to_string(),
                max_new_tokens: 1,
            })
        })
        .collect();

    model.forward_batch(&mut batch)?;

    // Compare results
    for (i, (seq_tokens, batch_ctx)) in sequential_results.iter().zip(batch.iter()).enumerate() {
        println!("Prompt {}: \"{}\"", i, prompts[i]);
        println!("  Sequential: {:?}", seq_tokens);
        println!("  Batched:    {:?}", batch_ctx.generated_tokens);

        assert!(
            tokens_match(seq_tokens, &batch_ctx.generated_tokens),
            "Token mismatch for prompt {}",
            i
        );
    }

    println!("\n✓ All tokens match between batch and sequential processing");
    Ok(())
}

#[test]
#[ignore]
fn test_batch_vs_sequential_multi_token() -> Result<()> {
    let model_path = require_model();

    println!("\n=== Test: Batch vs Sequential (10 Tokens) ===\n");

    let prompts = vec!["Once upon a time", "In a galaxy far away"];
    let num_tokens = 10;

    // Sequential processing
    let mut sequential_results = Vec::new();
    for prompt in &prompts {
        let mut model = ParallelModelManager::load(&model_path, 1, 512, None, None)?;
        let mut batch = vec![RequestContext::new(Request {
            id: "seq".to_string(),
            prompt: prompt.to_string(),
            max_new_tokens: num_tokens,
        })];

        for _ in 0..num_tokens {
            model.forward_batch(&mut batch)?;
            if !batch[0].should_continue() {
                break;
            }
        }

        sequential_results.push(batch[0].generated_tokens.clone());
    }

    // Batched processing
    let mut model = ParallelModelManager::load(&model_path, 2, 512, None, None)?;
    let mut batch: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            RequestContext::new(Request {
                id: format!("batch-{}", i),
                prompt: p.to_string(),
                max_new_tokens: num_tokens,
            })
        })
        .collect();

    for _ in 0..num_tokens {
        model.forward_batch(&mut batch)?;
        if batch.iter().all(|ctx| !ctx.should_continue()) {
            break;
        }
    }

    // Compare results
    for (i, (seq_tokens, batch_ctx)) in sequential_results.iter().zip(batch.iter()).enumerate() {
        let seq_text = model.decode(seq_tokens, false)?;
        let batch_text = model.decode(&batch_ctx.generated_tokens, false)?;

        println!("\nPrompt {}: \"{}\"", i, prompts[i]);
        println!("  Sequential: \"{}\"", seq_text);
        println!("  Batched:    \"{}\"", batch_text);

        assert!(
            tokens_match(seq_tokens, &batch_ctx.generated_tokens),
            "Token mismatch for prompt {}",
            i
        );
    }

    println!("\n✓ Multi-token generation matches");
    Ok(())
}

#[test]
#[ignore]
fn test_variable_sequence_lengths() -> Result<()> {
    let model_path = require_model();

    println!("\n=== Test: Variable Sequence Lengths ===\n");

    let prompts: Vec<String> = vec![
        "Short".to_string(),                                // ~5 tokens
        "Medium length prompt with more words".to_string(), // ~20 tokens
        "Long prompt with significantly more context and many words to process".repeat(4), // ~100 tokens
    ];

    let mut model = ParallelModelManager::load(&model_path, 3, 512, None, None)?;
    let mut batch: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            RequestContext::new(Request {
                id: format!("req-{}", i),
                prompt: p.clone(),
                max_new_tokens: 5,
            })
        })
        .collect();

    for step in 0..5 {
        let tokens = model.forward_batch(&mut batch)?;
        println!("\nStep {}: {:?}", step, tokens);
    }

    println!("\n✓ Variable length sequences processed successfully");
    Ok(())
}

#[test]
#[ignore]
fn test_kv_cache_consistency() -> Result<()> {
    let model_path = require_model();

    println!("\n=== Test: KV Cache Consistency ===\n");

    // Run same prompt twice to verify cache behavior
    let prompt = "The meaning of life is";

    // First run
    let mut model1 = ParallelModelManager::load(&model_path, 1, 512, None, None)?;
    let mut batch1 = vec![RequestContext::new(Request {
        id: "run1".to_string(),
        prompt: prompt.to_string(),
        max_new_tokens: 5,
    })];

    for _ in 0..5 {
        model1.forward_batch(&mut batch1)?;
    }
    let result1 = batch1[0].generated_tokens.clone();

    // Second run (should be identical)
    let mut model2 = ParallelModelManager::load(&model_path, 1, 512, None, None)?;
    let mut batch2 = vec![RequestContext::new(Request {
        id: "run2".to_string(),
        prompt: prompt.to_string(),
        max_new_tokens: 5,
    })];

    for _ in 0..5 {
        model2.forward_batch(&mut batch2)?;
    }
    let result2 = batch2[0].generated_tokens.clone();

    println!("Run 1: {:?}", result1);
    println!("Run 2: {:?}", result2);

    assert!(
        tokens_match(&result1, &result2),
        "Cache inconsistency detected"
    );

    println!("\n✓ KV cache produces consistent results");
    Ok(())
}

#[test]
#[ignore]
fn test_edge_case_empty_prompt() -> Result<()> {
    let model_path = require_model();

    println!("\n=== Test: Edge Case - Empty Prompt ===\n");

    let mut model = ParallelModelManager::load(&model_path, 1, 512, None, None)?;
    let mut batch = vec![RequestContext::new(Request {
        id: "empty".to_string(),
        prompt: "".to_string(),
        max_new_tokens: 5,
    })];

    // Should handle empty prompt gracefully
    let result = model.forward_batch(&mut batch);

    match result {
        Ok(_) => println!("✓ Empty prompt handled successfully"),
        Err(e) => println!("✓ Empty prompt rejected with error: {}", e),
    }

    Ok(())
}

#[test]
#[ignore]
fn test_edge_case_max_length() -> Result<()> {
    let model_path = require_model();

    println!("\n=== Test: Edge Case - Max Length ===\n");

    let context_len = 128; // Small context for faster testing
    let prompt = "Word ".repeat(100); // Prompt near max length

    let mut model = ParallelModelManager::load(&model_path, 1, context_len, None, None)?;
    let mut batch = vec![RequestContext::new(Request {
        id: "maxlen".to_string(),
        prompt: prompt.clone(),
        max_new_tokens: 50, // Will hit context limit
    })];

    let mut generated = 0;
    for _ in 0..50 {
        match model.forward_batch(&mut batch) {
            Ok(_) => generated += 1,
            Err(e) => {
                println!(
                    "✓ Stopped at context limit after {} tokens: {}",
                    generated, e
                );
                return Ok(());
            }
        }

        if !batch[0].should_continue() {
            break;
        }
    }

    println!("✓ Generated {} tokens before stopping", generated);
    Ok(())
}

#[test]
#[ignore]
fn test_batch_dynamic_completion() -> Result<()> {
    let model_path = require_model();

    println!("\n=== Test: Dynamic Batch Completion ===\n");

    // Requests with different max_new_tokens
    let mut model = ParallelModelManager::load(&model_path, 3, 512, None, None)?;
    let mut batch = vec![
        RequestContext::new(Request {
            id: "short".to_string(),
            prompt: "Hello".to_string(),
            max_new_tokens: 2,
        }),
        RequestContext::new(Request {
            id: "medium".to_string(),
            prompt: "Hello world".to_string(),
            max_new_tokens: 5,
        }),
        RequestContext::new(Request {
            id: "long".to_string(),
            prompt: "Hello world, how are you".to_string(),
            max_new_tokens: 10,
        }),
    ];

    let mut step = 0;
    while batch.iter().any(|ctx| ctx.should_continue()) {
        model.forward_batch(&mut batch)?;
        step += 1;

        let active = batch.iter().filter(|ctx| ctx.should_continue()).count();
        println!("Step {}: {} requests still active", step, active);

        // Verify earlier requests stopped
        if step > 2 {
            assert!(
                !batch[0].should_continue(),
                "Short request should have stopped"
            );
        }
        if step > 5 {
            assert!(
                !batch[1].should_continue(),
                "Medium request should have stopped"
            );
        }
    }

    println!("\n✓ Requests completed at different times correctly");
    Ok(())
}

#[test]
#[ignore]
fn test_attention_masking() -> Result<()> {
    let model_path = require_model();

    println!("\n=== Test: Attention Masking ===\n");

    // Test that padding doesn't affect results
    let prompts = vec![
        "Test",           // Short (will be padded in batch)
        "Test with more", // Medium
    ];

    // Process in batch (padding will be applied)
    let mut model1 = ParallelModelManager::load(&model_path, 2, 512, None, None)?;
    let mut batch: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            RequestContext::new(Request {
                id: format!("batch-{}", i),
                prompt: p.to_string(),
                max_new_tokens: 3,
            })
        })
        .collect();

    for _ in 0..3 {
        model1.forward_batch(&mut batch)?;
    }
    let batch_result = batch[0].generated_tokens.clone();

    // Process first prompt alone (no padding)
    let mut model2 = ParallelModelManager::load(&model_path, 1, 512, None, None)?;
    let mut single = vec![RequestContext::new(Request {
        id: "single".to_string(),
        prompt: prompts[0].to_string(),
        max_new_tokens: 3,
    })];

    for _ in 0..3 {
        model2.forward_batch(&mut single)?;
    }
    let single_result = single[0].generated_tokens.clone();

    println!("Batched (with padding): {:?}", batch_result);
    println!("Single (no padding):    {:?}", single_result);

    assert!(
        tokens_match(&batch_result, &single_result),
        "Padding affected generation results!"
    );

    println!("\n✓ Attention masking correctly ignores padding");
    Ok(())
}
