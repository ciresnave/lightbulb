/// Diagnostic test to compare single-request vs batched generation
/// This helps identify attention mask or batching issues
use anyhow::Result;
use candlelight::core::{DType, Device, Tensor};
use candlelight::transformers::models::llama::{Cache, Config};
use lightbulb::model::model_manager::ModelManager;
use lightbulb::request::{GenerationParams, Request};
use std::sync::Arc;
use tokenizers::Tokenizer;

fn load_test_model() -> Result<(ModelManager, Arc<Tokenizer>, Config, Device)> {
    let model_path = "models/TinyLlama-1.1B-Chat-v1.0";
    let device = Device::Cpu;

    let tokenizer = Arc::new(Tokenizer::from_file(format!(
        "{}/tokenizer.json",
        model_path
    ))?);
    let config = Config::tiny_llama_1_1b_chat();

    let mut manager = ModelManager::new(
        model_path.to_string(),
        512, // context
        4,   // max_batch
        &device,
        DType::F32,
    )?;

    Ok((manager, tokenizer, config, device))
}

#[test]
#[ignore]
fn test_single_vs_batch_generation() -> Result<()> {
    println!("\n=== Comparing Single vs Batch Generation ===\n");

    let (mut manager, tokenizer, _config, _device) = load_test_model()?;

    let test_prompts = vec![
        "The capital of France is",
        "Rust is a programming",
        "Machine learning is",
    ];

    // First: Generate each prompt individually (single-request mode)
    println!("--- Phase 1: Single-Request Generation ---\n");
    let mut single_results = Vec::new();

    for (idx, prompt) in test_prompts.iter().enumerate() {
        let tokens = tokenizer.encode(prompt.to_string(), true)?;
        let token_ids: Vec<u32> = tokens.get_ids().to_vec();

        let req = Request::new(
            format!("single-{}", idx),
            token_ids.clone(),
            GenerationParams {
                max_tokens: 8,
                temperature: 1.0,
                top_p: 0.9,
                ..Default::default()
            },
        );

        println!("Request {}: \"{}\"", idx, prompt);
        println!("Input tokens: {:?}", token_ids);

        let mut generated_tokens = Vec::new();
        manager.submit_request(req)?;

        loop {
            let (results, _batch_results) = manager.process_batch()?;

            if let Some(result) = results.first() {
                if let Some(token) = result.last_token {
                    generated_tokens.push(token);
                    print!("{} ", token);
                }

                if result.is_finished {
                    println!("\nFinished: {} tokens", generated_tokens.len());
                    break;
                }
            }
        }

        let generated_text = tokenizer.decode(&generated_tokens, true)?;
        println!("Generated text: {:?}\n", generated_text);

        single_results.push((prompt.to_string(), generated_tokens.clone(), generated_text));
    }

    // Second: Generate all prompts in batch
    println!("\n--- Phase 2: Batched Generation ---\n");
    let mut batch_results = Vec::new();

    // Submit all requests at once
    for (idx, prompt) in test_prompts.iter().enumerate() {
        let tokens = tokenizer.encode(prompt.to_string(), true)?;
        let token_ids: Vec<u32> = tokens.get_ids().to_vec();

        let req = Request::new(
            format!("batch-{}", idx),
            token_ids.clone(),
            GenerationParams {
                max_tokens: 8,
                temperature: 1.0,
                top_p: 0.9,
                ..Default::default()
            },
        );

        manager.submit_request(req)?;
    }

    let mut request_tokens: Vec<Vec<u32>> = vec![Vec::new(); test_prompts.len()];
    let mut completed = vec![false; test_prompts.len()];

    // Process batched requests
    while !completed.iter().all(|&c| c) {
        let (results, _batch_results) = manager.process_batch()?;

        for result in results {
            // Extract index from request ID
            let idx = if let Some(idx_str) = result.request_id.strip_prefix("batch-") {
                idx_str.parse::<usize>().unwrap()
            } else {
                continue;
            };

            if let Some(token) = result.last_token {
                request_tokens[idx].push(token);
            }

            if result.is_finished {
                completed[idx] = true;
            }
        }
    }

    for (idx, tokens) in request_tokens.iter().enumerate() {
        let generated_text = tokenizer.decode(tokens, true)?;
        println!("Request {}: \"{}\"", idx, test_prompts[idx]);
        println!("Tokens: {:?}", tokens);
        println!("Generated text: {:?}\n", generated_text);

        batch_results.push((
            test_prompts[idx].to_string(),
            tokens.clone(),
            generated_text,
        ));
    }

    // Third: Compare results
    println!("\n=== Comparison ===\n");

    let mut all_match = true;
    for i in 0..test_prompts.len() {
        let (prompt, single_tokens, single_text) = &single_results[i];
        let (_, batch_tokens, batch_text) = &batch_results[i];

        println!("Prompt {}: \"{}\"", i, prompt);
        println!("  Single tokens: {:?}", single_tokens);
        println!("  Batch tokens:  {:?}", batch_tokens);

        if single_tokens == batch_tokens {
            println!("  ✅ MATCH");
        } else {
            println!("  ❌ MISMATCH");
            all_match = false;

            // Find first divergence
            for (pos, (s, b)) in single_tokens.iter().zip(batch_tokens.iter()).enumerate() {
                if s != b {
                    println!(
                        "     First divergence at position {}: {} (single) vs {} (batch)",
                        pos, s, b
                    );
                    break;
                }
            }
        }

        println!("  Single text: {:?}", single_text);
        println!("  Batch text:  {:?}", batch_text);
        println!();
    }

    if all_match {
        println!("✅ All results match! Batching is working correctly.");
    } else {
        println!("❌ Results differ between single and batch mode!");
        println!("This indicates an issue with batched attention computation or masking.");
    }

    Ok(())
}

#[test]
#[ignore]
fn test_attention_mask_values() -> Result<()> {
    println!("\n=== Testing Attention Mask Values ===\n");

    use lightbulb::model::custom_attention::get_indices_and_mask_variable;

    // Test case: 3 requests with different lengths
    let seq_lengths = vec![11, 6, 4];
    let batch_mask = vec![true, true, true, false]; // 3 active, 1 inactive

    let iam = get_indices_and_mask_variable(&seq_lengths, &batch_mask, 512)?;

    println!("Sequence lengths: {:?}", seq_lengths);
    println!("Batch mask: {:?}", batch_mask);
    println!("Max sequence length: {}", seq_lengths.iter().max().unwrap());
    println!();

    // Check indices shape and values
    println!("Indices tensor shape: {:?}", iam.indices.shape());
    let indices_data = iam.indices.to_vec2::<u32>()?;
    println!("Indices values (first few rows):");
    for (i, row) in indices_data.iter().take(5).enumerate() {
        println!("  Row {}: {:?}", i, &row[..std::cmp::min(15, row.len())]);
    }
    println!();

    // Check mask shape and sample values
    println!("Mask tensor shape: {:?}", iam.mask.shape());

    // Sample mask values for each request at different positions
    for req_idx in 0..3 {
        let seq_len = seq_lengths[req_idx];
        println!("Request {} (seq_len={}):", req_idx, seq_len);

        // Check mask at actual token position (should be 0.0)
        let actual_token_mask = iam
            .mask
            .i((req_idx, 0, seq_len - 1, seq_len - 1))?
            .to_scalar::<f32>()?;
        println!(
            "  Mask at [req={}, head=0, pos={}, kv={}]: {}",
            req_idx,
            seq_len - 1,
            seq_len - 1,
            actual_token_mask
        );

        // Check mask at padding position (should be NEG_INFINITY)
        if seq_len < 11 {
            let padding_mask = iam
                .mask
                .i((req_idx, 0, seq_len, seq_len))?
                .to_scalar::<f32>()?;
            println!(
                "  Mask at padding [req={}, head=0, pos={}, kv={}]: {}",
                req_idx, seq_len, seq_len, padding_mask
            );
        }

        // Check future position mask (should be NEG_INFINITY for causal)
        if seq_len > 1 {
            let future_mask = iam
                .mask
                .i((req_idx, 0, seq_len - 2, seq_len - 1))?
                .to_scalar::<f32>()?;
            println!(
                "  Mask at future [req={}, head=0, pos={}, kv={}]: {}",
                req_idx,
                seq_len - 2,
                seq_len - 1,
                future_mask
            );
        }
        println!();
    }

    Ok(())
}
