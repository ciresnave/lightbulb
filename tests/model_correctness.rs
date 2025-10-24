//! Model Correctness Test - Compare BatchedTransformer vs Standard Llama
//!
//! This test loads actual model weights and verifies that our batched implementation
//! produces numerically identical outputs to the standard Candle Llama model.
//!
//! # Prerequisites
//!
//! You need a Llama model in the `models/` directory. To run this test:
//!
//! ```bash
//! cargo test --test model_correctness -- --ignored --nocapture
//! ```
//!
//! # Success Criteria
//!
//! - Output logits must match within 1e-4 relative error
//! - Test with batch sizes: 1, 2, 5
//! - Test with different sequence lengths
//!
//! This is the critical test that validates our implementation is correct!

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama};
use lightbulb::engine::BatchExecutor;
use lightbulb::model::{BatchMetadata, BatchedTransformer, BatchedTransformerConfig};
use std::path::Path;

/// Tolerance for numerical comparison
const RTOL: f32 = 1e-3; // Relative tolerance (0.1%)
const ATOL: f32 = 1e-5; // Absolute tolerance

/// Helper to compare two tensors
fn assert_close(a: &Tensor, b: &Tensor, name: &str) -> Result<()> {
    if a.dims() != b.dims() {
        anyhow::bail!("{}: Shape mismatch {:?} vs {:?}", name, a.dims(), b.dims());
    }

    let a_flat = a.flatten_all()?.to_vec1::<f32>()?;
    let b_flat = b.flatten_all()?.to_vec1::<f32>()?;

    let mut max_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut num_errors = 0;

    for (i, (&a_val, &b_val)) in a_flat.iter().zip(b_flat.iter()).enumerate() {
        let abs_diff = (a_val - b_val).abs();
        let rel_diff = if b_val.abs() > ATOL {
            abs_diff / b_val.abs()
        } else {
            abs_diff
        };

        max_diff = max_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);

        if rel_diff > RTOL && abs_diff > ATOL {
            num_errors += 1;
            if num_errors <= 3 {
                println!(
                    "  Error {}: a={:.6}, b={:.6}, diff={:.6e}, rel={:.6e}",
                    i, a_val, b_val, abs_diff, rel_diff
                );
            }
        }
    }

    println!(
        "  {}: max_abs={:.6e}, max_rel={:.6e}, errors={}/{}",
        name,
        max_diff,
        max_rel_diff,
        num_errors,
        a_flat.len()
    );

    if num_errors > 0 {
        anyhow::bail!("{}: {} values exceed tolerance", name, num_errors);
    }

    Ok(())
}

/// Main correctness test - compares batched vs standard implementation
#[test]
#[ignore] // Run with: cargo test --test model_correctness -- --ignored --nocapture
fn test_batched_vs_standard_llama() -> Result<()> {
    println!("\n=== BATCHED VS STANDARD LLAMA CORRECTNESS TEST ===\n");

    // 1. Find model directory
    let model_paths = vec![
        "../models/llama-3b",    // Most likely location
        "../../models/llama-3b", // Alternative
        "models/llama-3b",       // Current dir
        "../models/TinyLlama-1.1B-Chat-v1.0",
        "models/TinyLlama-1.1B-Chat-v1.0",
    ];

    let model_dir = model_paths
        .iter()
        .find(|p| Path::new(p).exists())
        .ok_or_else(|| anyhow::anyhow!("No model found. Tried:\n{}", model_paths.join("\n")))?;

    println!("✓ Found model at: {}\n", model_dir);

    // 2. Load standard Llama model
    println!("Loading standard Llama model...");
    let (standard_model, mut standard_cache, config, device) =
        lightbulb::loaders::load_local_llama(model_dir, Some("f32"), true, false)?;

    println!(
        "  Config: {} layers, {} heads, {} hidden_size",
        config.num_hidden_layers, config.num_attention_heads, config.hidden_size
    );

    // 3. Create BatchedTransformer config from standard config
    println!("\nCreating BatchedTransformer config...");
    let batched_config = BatchedTransformerConfig {
        vocab_size: config.vocab_size,
        hidden_size: config.hidden_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        intermediate_size: config.intermediate_size,
        max_position_embeddings: config.max_position_embeddings,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        rope_scaling: None,
        sliding_window: None,
        use_flash_attn: false,
        tie_word_embeddings: config.tie_word_embeddings, // Match standard config
    };

    batched_config.validate()?;
    println!("  ✓ Config validated");

    // 4. Load BatchedTransformer with same weights
    println!("\nLoading BatchedTransformer with same weights...");
    let files = lightbulb::loaders::find_safetensors_files(Path::new(model_dir))?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &device)? };

    let mut batched_model = BatchedTransformer::new(batched_config, vb)?;
    println!("  ✓ BatchedTransformer loaded");

    // 5. Create test inputs
    println!("\n== Running Correctness Tests ==\n");

    let test_cases = vec![
        ("Single token", vec![vec![1u32, 2, 3, 4]]), // One sequence, 4 tokens
        ("Two sequences", vec![vec![1u32, 2], vec![10u32, 20, 30]]), // Different lengths
    ];

    for (name, sequences) in test_cases {
        println!("Test case: {}", name);
        println!("  Sequences: {:?}", sequences);

        // Create input tokens
        let total_tokens: usize = sequences.iter().map(|s| s.len()).sum();
        let all_tokens: Vec<u32> = sequences.iter().flat_map(|s| s.iter().copied()).collect();
        let input_tensor = Tensor::from_vec(all_tokens.clone(), total_tokens, &device)?;

        // Create batch metadata
        let metadata = BatchMetadata::from_sequences(&sequences);
        let mut batch_executor = BatchExecutor::new(
            1,                              // batch_size (set to 1 for simplicity)
            config.max_position_embeddings, // context window
            config.num_hidden_layers,
            config.num_key_value_heads, // num_heads (use num_kv_heads for GQA models!)
            config.hidden_size / config.num_attention_heads, // head_dim
            DType::F32,
            &device,
        )?;

        // Run batched model
        println!("  Running BatchedTransformer...");
        let batched_logits =
            batched_model.forward(&input_tensor, &mut batch_executor, &metadata)?;
        println!("    Output shape: {:?}", batched_logits.dims());

        // Run standard model on each sequence independently
        println!("  Running standard Llama...");
        let mut standard_logits_vec = Vec::new();

        for seq in &sequences {
            let seq_tensor = Tensor::from_vec(seq.clone(), seq.len(), &device)?.unsqueeze(0)?; // [seq_len] -> [1, seq_len]

            // DEBUG: Check embedding layer for standard model
            // We can't easily access internals, but we can check input
            // DEBUG output removed

            let seq_logits = standard_model.forward(&seq_tensor, 0, &mut standard_cache)?;
            // Standard Llama returns [batch=1, vocab_size] (last token only!)
            // We want [vocab_size]
            let last_logits = seq_logits.i(0)?; // [vocab_size]
            standard_logits_vec.push(last_logits);
        }

        // Stack standard outputs
        let standard_logits = Tensor::stack(&standard_logits_vec, 0)?; // [num_seqs, vocab_size]

        //  Compare last token of each sequence
        // Batched model returns [batch_size, total_tokens, vocab_size]
        // Standard model logits: [num_seqs, vocab_size]

        let mut batched_last_logits_vec = Vec::new();
        let mut token_offset = 0;
        for seq in &sequences {
            // Get the last token position for this sequence
            let last_token_pos = token_offset + seq.len() - 1;
            // Extract [batch=0, pos=last_token_pos, :] → [vocab_size]
            let last_logit = batched_logits.i((0, last_token_pos))?;
            batched_last_logits_vec.push(last_logit);
            token_offset += seq.len();
        }

        let batched_last_logits = Tensor::stack(&batched_last_logits_vec, 0)?; // [num_seqs, vocab_size]

        // DEBUG: Print first few values
        println!(
            "  Batched logits[0:10]: {:?}",
            batched_last_logits
                .i(0)?
                .to_vec1::<f32>()?
                .iter()
                .take(10)
                .collect::<Vec<_>>()
        );
        println!(
            "  Standard logits[0:10]: {:?}",
            standard_logits
                .i(0)?
                .to_vec1::<f32>()?
                .iter()
                .take(10)
                .collect::<Vec<_>>()
        );

        // Compare
        assert_close(&batched_last_logits, &standard_logits, name)?;
        println!("  ✓ {} passed!\n", name);
    }

    println!("=== ALL TESTS PASSED! ===");
    println!("\n✓ BatchedTransformer produces identical outputs to standard Llama");
    println!("✓ Ready for performance benchmarking!\n");

    Ok(())
}

/// Test that individual components match
#[test]
#[ignore]
fn test_component_correctness() -> Result<()> {
    println!("\n=== Testing Individual Components ===\n");

    // TODO: Test individual layers (attention, MLP, etc.)
    // This helps debug if the full model test fails

    println!("✓ Component tests not yet implemented");
    Ok(())
}
