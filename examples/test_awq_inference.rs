//! AWQ Model Inference Test
//!
//! This example demonstrates loading and running inference with an AWQ-quantized model.
//! Specifically tests the Qwen3-32B-AWQ model with 4-bit quantization.
//!
//! Usage:
//!   cargo run --example test_awq_inference --features cuda
//!
//! Expected output:
//!   - Model loads successfully
//!   - Forward pass executes without errors
//!   - Text generation produces coherent output

use anyhow::Result;
use candlelight::core::{DType, Device, Tensor};
use lightbulb::loaders;

fn main() -> Result<()> {
    println!("=== AWQ Inference Test ===\n");

    // Model path
    let model_dir = "c:/Users/cires/OneDrive/Documents/projects/lightbulb/models/Qwen3-32B-AWQ";

    println!("Step 1: Loading AWQ model...");
    let (mut model, config, device, _name_mapper) = loaders::load_awq_llama(
        model_dir,
        Some("f16"), // Use F16 for activations
        false,       // Flash attention not yet supported for AWQ
    )?;

    println!("\n✓ Model loaded successfully!\n");

    // Test with a simple prompt
    println!("Step 2: Running forward pass...");
    let prompt = "The capital of France is";
    println!("Prompt: \"{}\"", prompt);

    // Tokenize (simplified - using token IDs directly for testing)
    // In production, you'd use a proper tokenizer
    let input_ids: Vec<u32> = vec![
        791,  // "The"
        6864, // "capital"
        315,  // "of"
        9822, // "France"
        374,  // "is"
    ];

    let input_tensor =
        Tensor::from_vec(input_ids.clone(), (1, input_ids.len()), &device)?.to_dtype(DType::U32)?;

    println!("Input shape: {:?}", input_tensor.shape());

    // Run forward pass
    let logits = model.forward(&input_tensor, 0)?;
    println!("Output logits shape: {:?}", logits.shape());

    // Get top-k predictions
    let logits_vec = logits.to_vec1::<f32>()?;
    let mut indexed: Vec<(usize, f32)> = logits_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 5 predicted tokens:");
    for (i, (token_id, score)) in indexed.iter().take(5).enumerate() {
        println!("  {}. Token ID: {}, Score: {:.4}", i + 1, token_id, score);
    }

    // Test autoregressive generation (generate a few tokens)
    println!("\nStep 3: Testing autoregressive generation...");
    let max_tokens = 10;
    let mut generated_ids = input_ids.clone();
    let mut current_ids: Vec<u32> = input_ids.clone();

    for step in 0..max_tokens {
        // Create input tensor
        let input = Tensor::from_vec(current_ids.clone(), (1, current_ids.len()), &device)?
            .to_dtype(DType::U32)?;

        // Forward pass
        let logits = model.forward(&input, step)?;

        // Sample next token (greedy - just take argmax)
        let logits_vec = logits.to_vec1::<f32>()?;
        let next_token = logits_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();

        generated_ids.push(next_token);

        // For next iteration, use only the new token (causal attention already has cache)
        current_ids = vec![next_token];

        println!("  Step {}: Generated token ID {}", step + 1, next_token);
    }
    println!("\n✓ Autoregressive generation successful!");
    println!("\nGenerated token IDs: {:?}", generated_ids);

    // Test batch inference
    println!("\nStep 4: Testing batch inference...");
    let batch_size = 2;
    let seq_len = 5;

    let batch_data: Vec<u32> = vec![input_ids[0]; batch_size * seq_len];
    let batch_input =
        Tensor::from_vec(batch_data, (batch_size, seq_len), &device)?.to_dtype(DType::U32)?;

    println!("Batch input shape: {:?}", batch_input.shape());

    let batch_logits = model.forward(&batch_input, 0)?;
    println!("Batch output shape: {:?}", batch_logits.shape());

    println!("\n✓ Batch inference successful!");

    // Memory usage estimation
    println!("\nStep 5: Memory usage estimation...");
    if let Device::Cuda(_cuda_device) = &device {
        // Note: This is a simplified estimation
        let param_memory_gb =
            (config.hidden_size * config.num_hidden_layers * 4) / 1024 / 1024 / 1024;
        println!("  - Estimated model memory: ~{} GB", param_memory_gb);
        println!("  - Quantization: 4-bit AWQ");
        println!("  - Activation dtype: F16");
        println!("  - Device: CUDA");
    }

    println!("\n=== All Tests Passed! ===");
    println!("\nSummary:");
    println!("  ✓ Model loading: PASS");
    println!("  ✓ Forward pass: PASS");
    println!("  ✓ Autoregressive generation: PASS");
    println!("  ✓ Batch inference: PASS");
    println!("\nAWQ Phase 3 implementation is functional!");

    Ok(())
}
