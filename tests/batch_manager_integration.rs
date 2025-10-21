//! Integration tests for BatchManager
//!
//! These tests verify that BatchManager produces correct outputs when
//! wrapping Candle's standard Llama model, comparing against direct
//! Llama usage.
//!
//! Tests are designed to remain valid even after implementing parallel
//! batched processing - correctness should be the same whether sequential
//! or parallel.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::llama::{Cache, Config, Llama};
use lightbulb::engine::BatchExecutor;
use lightbulb::model::{BatchManager, BatchMetadata};
use std::path::Path;

/// Load Llama model from local directory for testing
fn load_test_model() -> Result<(Llama, Cache, Config, Device)> {
    let model_path = "../models/llama-3b";

    if !Path::new(model_path).exists() {
        anyhow::bail!(
            "Test model not found at {}. Run with --ignored to skip.",
            model_path
        );
    }

    // Use the project's load_local_llama utility
    lightbulb::loaders::load_local_llama(
        model_path,
        Some("f32"),
        true,  // use_kv_cache
        false, // use_flash_attn
    )
}

#[test]
#[ignore] // Requires real model weights
fn test_batch_manager_single_request_matches_direct_llama() -> Result<()> {
    println!("\n=== Test: Single Request - BatchManager vs Direct Llama ===\n");

    // Load model
    let (llama_model, _, config, device) = load_test_model()?;

    // Create BatchManager
    let batch_executor = BatchExecutor::new(
        4,   // max_batch_size
        512, // context_length
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size / config.num_attention_heads,
        DType::F32,
        &device,
    )?;

    let mut batch_manager = BatchManager::new(llama_model, batch_executor, device.clone());

    // Load separate model for direct comparison
    let (direct_llama, _, _, _) = load_test_model()?;
    let mut direct_cache = Cache::new(true, DType::F32, &config, &device)?;
    let batch_cache = Cache::new(true, DType::F32, &config, &device)?;

    // Test tokens (simple sequence)
    let test_tokens = vec![1u32, 2, 3, 4, 5]; // BOS + some tokens
    let token_tensor = Tensor::new(test_tokens.as_slice(), &device)?.unsqueeze(0)?; // [1, 5]

    // Direct Llama forward pass
    println!("Running direct Llama forward pass...");
    let mut direct_llama_mut = direct_llama;
    let direct_logits = direct_llama_mut.forward(&token_tensor, 0, &mut direct_cache)?;
    let direct_shape = direct_logits.dims();
    println!("Direct logits shape: {:?}", direct_shape);

    // Get last token logits
    let direct_last = direct_logits.i(direct_logits.dim(0)? - 1)?;
    let direct_top_token = direct_last.argmax(0)?.to_scalar::<u32>()?;
    println!("Direct model top token: {}", direct_top_token);

    // BatchManager forward pass (prefill mode)
    println!("\nRunning BatchManager prefill batch...");
    let metadata = BatchMetadata::from_prefill_batch(
        vec![0],                 // Single request
        vec![test_tokens.len()], // prompt_lengths
    );

    let batch_logits = batch_manager.forward_prefill_batch(
        &token_tensor.squeeze(0)?, // [5] tokens
        &metadata,
        &mut [batch_cache],
    )?;

    let batch_shape = batch_logits.dims();
    println!("BatchManager logits shape: {:?}", batch_shape);

    let batch_last = batch_logits.i(0)?; // First (only) request
    let batch_top_token = batch_last.argmax(0)?.to_scalar::<u32>()?;
    println!("BatchManager top token: {}", batch_top_token);

    // Compare outputs
    println!("\n=== Verification ===");
    assert_eq!(
        direct_top_token, batch_top_token,
        "Top tokens should match: direct={}, batch={}",
        direct_top_token, batch_top_token
    );

    // Compare full logit distributions (should be very close)
    let direct_vec = direct_last.to_vec1::<f32>()?;
    let batch_vec = batch_last.to_vec1::<f32>()?;

    assert_eq!(
        direct_vec.len(),
        batch_vec.len(),
        "Logit dimensions should match"
    );

    // Calculate max difference
    let max_diff: f32 = direct_vec
        .iter()
        .zip(batch_vec.iter())
        .map(|(d, b)| (d - b).abs())
        .fold(0.0f32, f32::max);

    println!("Max logit difference: {:.6e}", max_diff);

    // Should be very close (allowing for some FP differences due to sequential processing)
    assert!(
        max_diff < 1e-2,
        "Max logit difference too large: {:.6e}",
        max_diff
    );

    println!("✓ BatchManager output matches direct Llama!");
    Ok(())
}

#[test]
#[ignore] // Requires real model weights
fn test_batch_manager_decode_step() -> Result<()> {
    println!("\n=== Test: Decode Step - BatchManager vs Direct Llama ===\n");

    // Load model
    let (llama_model, _, config, device) = load_test_model()?;

    // Create BatchManager
    let batch_executor = BatchExecutor::new(
        4,
        512,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size / config.num_attention_heads,
        DType::F32,
        &device,
    )?;

    let mut batch_manager = BatchManager::new(llama_model, batch_executor, device.clone());

    // Load separate model for direct comparison
    let (mut direct_llama, _, _, _) = load_test_model()?;
    let mut direct_cache = Cache::new(true, DType::F32, &config, &device)?;

    // First, prefill with same prompt
    let prefill_tokens = vec![1u32, 2, 3];
    let prefill_tensor = Tensor::new(prefill_tokens.as_slice(), &device)?.unsqueeze(0)?;

    println!("Prefilling both models with tokens: {:?}", prefill_tokens);

    // Prefill direct model
    let _ = direct_llama.forward(&prefill_tensor, 0, &mut direct_cache)?;

    // Prefill batch manager (need fresh cache)
    let mut batch_cache = Cache::new(true, DType::F32, &config, &device)?;
    let prefill_metadata = BatchMetadata::from_prefill_batch(vec![0], vec![prefill_tokens.len()]);
    let _ = batch_manager.forward_prefill_batch(
        &prefill_tensor.squeeze(0)?,
        &prefill_metadata,
        &mut [batch_cache],
    )?;

    // Now test decode step
    println!("\nTesting decode step (single token generation)...");
    let decode_token = 42u32;
    let decode_tensor_2d = Tensor::new(&[decode_token], &device)?.unsqueeze(0)?; // [1, 1] for direct Llama
    let decode_tensor_1d = Tensor::new(&[decode_token], &device)?; // [1] for BatchManager

    // Direct decode
    let position = prefill_tokens.len();
    let direct_logits = direct_llama.forward(&decode_tensor_2d, position, &mut direct_cache)?;
    let direct_last = direct_logits.i(0)?;
    let direct_top = direct_last.argmax(0)?.to_scalar::<u32>()?;

    println!("Direct model next token: {}", direct_top);

    // BatchManager decode (create fresh cache and prefill again to match state)
    let decode_metadata = BatchMetadata::from_decode_batch(vec![0], vec![position]);

    let mut batch_caches2 = vec![Cache::new(true, DType::F32, &config, &device)?];
    let _ = batch_manager.forward_prefill_batch(
        &prefill_tensor.squeeze(0)?,
        &prefill_metadata,
        &mut batch_caches2,
    )?;

    let batch_logits = batch_manager.forward_decode_batch(
        &decode_tensor_1d,
        &decode_metadata,
        &mut batch_caches2,
    )?;

    let batch_last = batch_logits.i(0)?;
    let batch_top = batch_last.argmax(0)?.to_scalar::<u32>()?;

    println!("BatchManager next token: {}", batch_top);

    // Verify match
    println!("\n=== Verification ===");
    assert_eq!(
        direct_top, batch_top,
        "Decode tokens should match: direct={}, batch={}",
        direct_top, batch_top
    );

    println!("✓ Decode step matches!");
    Ok(())
}

#[test]
#[ignore] // Requires real model weights
fn test_batch_manager_multiple_requests() -> Result<()> {
    println!("\n=== Test: Multiple Requests - BatchManager ===\n");

    // Load model
    let (llama_model, _, config, device) = load_test_model()?;

    // Create BatchManager
    let batch_executor = BatchExecutor::new(
        8,
        512,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size / config.num_attention_heads,
        DType::F32,
        &device,
    )?;

    let mut batch_manager = BatchManager::new(llama_model, batch_executor, device.clone());

    // Create 3 different "requests" with different prompts
    let prompts = vec![
        vec![1u32, 2, 3],    // Request 0
        vec![1u32, 4, 5, 6], // Request 1
        vec![1u32, 7],       // Request 2
    ];

    println!("Testing batch of {} requests", prompts.len());
    for (i, prompt) in prompts.iter().enumerate() {
        println!("  Request {}: {} tokens", i, prompt.len());
    }

    // Create caches for each request
    let mut caches: Vec<Cache> = (0..prompts.len())
        .map(|_| {
            Cache::new(true, DType::F32, &config, &device).map_err(|e| anyhow::anyhow!("{}", e))
        })
        .collect::<Result<Vec<_>>>()?;

    // Concatenate all prompts
    let total_tokens: usize = prompts.iter().map(|p| p.len()).sum();
    let mut all_tokens: Vec<u32> = Vec::with_capacity(total_tokens);
    let prompt_lengths: Vec<usize> = prompts.iter().map(|p| p.len()).collect();

    for prompt in &prompts {
        all_tokens.extend(prompt);
    }

    println!("\nTotal tokens: {}", all_tokens.len());
    println!("Prompt lengths: {:?}", prompt_lengths);

    // Run prefill batch
    let tokens_tensor = Tensor::new(all_tokens.as_slice(), &device)?;
    let metadata = BatchMetadata::from_prefill_batch((0..prompts.len()).collect(), prompt_lengths);

    println!("\nRunning batched prefill...");
    let logits_batch =
        batch_manager.forward_prefill_batch(&tokens_tensor, &metadata, &mut caches)?;

    let batch_shape = logits_batch.dims();
    println!("Output shape: {:?}", batch_shape);
    assert_eq!(
        batch_shape[0],
        prompts.len(),
        "Should have logits for each request"
    );

    // Extract top token for each request
    println!("\n=== Generated Tokens ===");
    for i in 0..prompts.len() {
        let logits = logits_batch.i(i)?;
        let top_token = logits.argmax(0)?.to_scalar::<u32>()?;
        println!("Request {}: next token = {}", i, top_token);
    }

    // Now test decode batch - all at same decode step
    println!("\n=== Testing Decode Batch ===");
    let decode_tokens = vec![10u32, 20, 30]; // One token per request
    let decode_tensor = Tensor::new(decode_tokens.as_slice(), &device)?; // [3]

    let positions: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
    let decode_metadata =
        BatchMetadata::from_decode_batch((0..prompts.len()).collect(), positions.clone());

    println!("Decode positions: {:?}", positions);

    let decode_logits =
        batch_manager.forward_decode_batch(&decode_tensor, &decode_metadata, &mut caches)?;

    println!("Decode output shape: {:?}", decode_logits.dims());

    println!("\n=== Next Tokens After Decode ===");
    for i in 0..prompts.len() {
        let logits = decode_logits.i(i)?;
        let top_token = logits.argmax(0)?.to_scalar::<u32>()?;
        println!("Request {}: next token = {}", i, top_token);
    }

    println!("\n✓ Multiple request batch processing works!");
    Ok(())
}

#[test]
#[ignore] // Requires real model weights
fn test_batch_size_one_equals_direct_llama() -> Result<()> {
    println!("\n=== Test: Batch Size 1 === Direct Llama (Unified Code Path) ===\n");

    // This test validates that treating single requests as batch_size=1
    // produces identical results to direct model usage

    let (llama_model, _, config, device) = load_test_model()?;

    let batch_executor = BatchExecutor::new(
        1,
        512, // max_batch_size = 1
        config.num_hidden_layers,
        config.num_attention_heads,
        config.hidden_size / config.num_attention_heads,
        DType::F32,
        &device,
    )?;

    let mut batch_manager = BatchManager::new(llama_model, batch_executor, device.clone());

    // Load direct model
    let (mut direct_llama, _, _, _) = load_test_model()?;
    let mut direct_cache = Cache::new(true, DType::F32, &config, &device)?;

    // Store batch cache in a Vec so we can pass mutable slice repeatedly
    let mut batch_caches = vec![Cache::new(true, DType::F32, &config, &device)?];

    // Run multiple generation steps to ensure caches stay in sync
    let mut current_tokens = vec![1u32]; // Start with BOS

    println!("Generating 5 tokens, comparing at each step...\n");

    for step in 0..5 {
        // Direct forward
        let direct_logits = if step == 0 {
            // First step: prefill with all tokens
            let token_tensor = Tensor::new(current_tokens.as_slice(), &device)?.unsqueeze(0)?;
            direct_llama.forward(&token_tensor, step, &mut direct_cache)?
        } else {
            // Subsequent steps: decode with last token only
            let last_token = current_tokens[current_tokens.len() - 1];
            let token_tensor = Tensor::new(&[last_token], &device)?.unsqueeze(0)?;
            direct_llama.forward(&token_tensor, step, &mut direct_cache)?
        };

        let direct_next = direct_logits
            .i(direct_logits.dim(0)? - 1)?
            .argmax(0)?
            .to_scalar::<u32>()?;

        // Batch forward (batch_size=1)
        let batch_logits = if step == 0 {
            // First step is prefill
            let token_tensor = Tensor::new(current_tokens.as_slice(), &device)?;
            let metadata = BatchMetadata::from_prefill_batch(vec![0], vec![current_tokens.len()]);
            batch_manager.forward_prefill_batch(&token_tensor, &metadata, &mut batch_caches)?
        } else {
            // Subsequent steps are decode
            let last_token = Tensor::new(&[current_tokens[current_tokens.len() - 1]], &device)?; // [1]
            let metadata = BatchMetadata::from_decode_batch(vec![0], vec![step]);
            batch_manager.forward_decode_batch(&last_token, &metadata, &mut batch_caches)?
        };

        let batch_next = batch_logits.i(0)?.argmax(0)?.to_scalar::<u32>()?;

        println!(
            "Step {}: direct={}, batch={} {}",
            step,
            direct_next,
            batch_next,
            if direct_next == batch_next {
                "✓"
            } else {
                "✗"
            }
        );

        assert_eq!(
            direct_next, batch_next,
            "Tokens should match at step {}",
            step
        );

        current_tokens.push(direct_next);
    }

    println!("\n✓ Batch size 1 produces identical results to direct usage!");
    println!("Generated sequence: {:?}", current_tokens);
    Ok(())
}
