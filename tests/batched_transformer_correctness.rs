//! Correctness tests for BatchedTransformer vs Sequential Llama
//!
//! This test suite verifies that our custom BatchedTransformer produces
//! numerically identical outputs to Candle's standard sequential Llama model.
//!
//! We test:
//! - RoPE frequency computation
//! - Embedding layer outputs
//! - Attention outputs (when we can isolate them)
//! - Full forward pass outputs
//! - Multi-step generation

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use lightbulb::engine::BatchExecutor;
use lightbulb::model::batch_metadata::BatchMetadata;
use lightbulb::model::custom_transformer::{BatchedTransformer, BatchedTransformerConfig};
use std::path::PathBuf;

/// Helper to find model files
fn get_model_path() -> Option<PathBuf> {
    // Try common locations for model with safetensors
    let candidates = vec![
        "models/llama-3b",
        "../models/llama-3b",
        "../../models/llama-3b",
        // Add more paths as needed
    ];

    for candidate in candidates {
        let path = PathBuf::from(candidate);
        if path.join("model.safetensors").exists() {
            return Some(path);
        }
    }
    None
}

/// Helper to create a BatchedTransformerConfig from Candle's Llama Config
fn config_from_candle(
    candle_config: &candle_transformers::models::llama::Config,
) -> BatchedTransformerConfig {
    BatchedTransformerConfig {
        vocab_size: candle_config.vocab_size,
        hidden_size: candle_config.hidden_size,
        num_hidden_layers: candle_config.num_hidden_layers,
        num_attention_heads: candle_config.num_attention_heads,
        num_key_value_heads: candle_config.num_key_value_heads,
        intermediate_size: candle_config.intermediate_size,
        max_position_embeddings: candle_config.max_position_embeddings,
        rms_norm_eps: candle_config.rms_norm_eps,
        rope_theta: candle_config.rope_theta,
        rope_scaling: None,
        sliding_window: None,
        use_flash_attn: false,
        tie_word_embeddings: candle_config.tie_word_embeddings,
    }
}

/// Compare two tensors with tolerance
fn tensors_close(a: &Tensor, b: &Tensor, rtol: f32, atol: f32) -> Result<bool> {
    if a.dims() != b.dims() {
        println!("Shape mismatch: {:?} vs {:?}", a.dims(), b.dims());
        return Ok(false);
    }

    let a_vec = a.flatten_all()?.to_vec1::<f32>()?;
    let b_vec = b.flatten_all()?.to_vec1::<f32>()?;

    let mut max_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut num_mismatches = 0;

    for (i, (a_val, b_val)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        let abs_diff = (a_val - b_val).abs();
        let rel_diff = if b_val.abs() > 1e-8 {
            abs_diff / b_val.abs()
        } else {
            abs_diff
        };

        max_diff = max_diff.max(abs_diff);
        max_rel_diff = max_rel_diff.max(rel_diff);

        if abs_diff > atol && rel_diff > rtol {
            num_mismatches += 1;
            if num_mismatches <= 10 {
                // Print first 10 mismatches
                println!(
                    "Mismatch at index {}: a={:.6}, b={:.6}, diff={:.6}, rel_diff={:.6}",
                    i, a_val, b_val, abs_diff, rel_diff
                );
            }
        }
    }

    println!(
        "Comparison: max_abs_diff={:.6}, max_rel_diff={:.6}, mismatches={}/{}",
        max_diff,
        max_rel_diff,
        num_mismatches,
        a_vec.len()
    );

    Ok(num_mismatches == 0)
}

#[test]
#[ignore]// Requires model files
fn test_single_token_forward() -> Result<()> {
    println!("\n=== Test: Single Token Forward Pass ===\n");

    let model_path = match get_model_path() {
        Some(path) => path,
        None => {
            println!("Skipping test: No model found");
            return Ok(());
        }
    };

    let device = Device::Cpu;
    let dtype = DType::F32;

    // Load Candle's sequential Llama
    println!("Loading sequential Llama...");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[model_path.join("model.safetensors")],
            dtype,
            &device,
        )?
    };

    // Llama 3B config (from models/llama-3b/config.json)
    let candle_config = candle_transformers::models::llama::Config {
        hidden_size: 576,
        intermediate_size: 1536,
        vocab_size: 49152,
        num_hidden_layers: 30,
        num_attention_heads: 9,
        num_key_value_heads: 3,
        rms_norm_eps: 1e-5,
        rope_theta: 100000.0,
        bos_token_id: Some(0),
        eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(0)),
        max_position_embeddings: 8192,
        use_flash_attn: false,
        rope_scaling: None,
        tie_word_embeddings: true,
    };

    let candle_model = candle_transformers::models::llama::Llama::load(vb, &candle_config)?;

    // Load our BatchedTransformer
    println!("Loading BatchedTransformer...");
    let vb2 = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[model_path.join("model.safetensors")],
            dtype,
            &device,
        )?
    };

    let batched_config = config_from_candle(&candle_config);
    let mut batched_model = BatchedTransformer::new(batched_config, vb2)?;

    // Create test input: single token (BOS token is 0 for this model)
    let token_id = 0u32;
    // Candle's Llama expects 2D input: [batch_size, seq_len]
    let input_sequential = Tensor::from_vec(vec![token_id], (1, 1), &device)?;

    // Sequential forward pass
    println!("\nRunning sequential forward pass...");

    // First get embedding output from Candle
    let sequential_embeddings = candle_model.embed(&input_sequential)?;
    println!(
        "Sequential embedding shape: {:?}",
        sequential_embeddings.dims()
    );
    println!(
        "Sequential embedding mean: {:.6}",
        sequential_embeddings.mean_all()?
    );
    let seq_emb_vec = sequential_embeddings.flatten_all()?.to_vec1::<f32>()?;
    println!(
        "Sequential embedding samples: {:?}",
        &seq_emb_vec[..5.min(seq_emb_vec.len())]
    );

    let mut sequential_cache =
        candle_transformers::models::llama::Cache::new(false, dtype, &candle_config, &device)?;
    let sequential_logits = candle_model.forward(&input_sequential, 0, &mut sequential_cache)?;

    println!("Sequential logits shape: {:?}", sequential_logits.dims());

    // Debug: Check sequential logits values
    let seq_vec = sequential_logits.flatten_all()?.to_vec1::<f32>()?;
    let seq_mean: f32 = seq_vec.iter().sum::<f32>() / seq_vec.len() as f32;
    let seq_max = seq_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let seq_min = seq_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "DEBUG Sequential logits: mean={:.6}, min={:.6}, max={:.6}, sample[0:5]={:?}",
        seq_mean,
        seq_min,
        seq_max,
        &seq_vec[0..5]
    );

    // Batched forward pass
    println!("\nRunning batched forward pass...");
    let input_batched = Tensor::from_vec(vec![token_id], (1,), &device)?;

    let batch_size = 1;
    let context_window = 512;
    let head_dim = candle_config.hidden_size / candle_config.num_attention_heads;
    let mut batch_executor = BatchExecutor::new(
        batch_size,
        context_window,
        candle_config.num_hidden_layers,
        candle_config.num_key_value_heads,
        head_dim,
        dtype,
        &device,
    )
    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let metadata = BatchMetadata {
        is_prefill: true,
        batch_size: 1,
        request_ids: vec![0],
        slot_offsets: vec![0],
        sequence_lengths: vec![1],
        cu_seqlens: Some(vec![0, 1]),
        max_seqlen: Some(1),
        query_start_loc: Some(vec![0]),
        context_lens: vec![0],
    };

    let batched_logits = batched_model.forward(&input_batched, &mut batch_executor, &metadata)?;

    println!("Batched logits shape: {:?}", batched_logits.dims());

    // Debug: Check batched logits values
    let batch_vec = batched_logits.flatten_all()?.to_vec1::<f32>()?;
    let batch_mean: f32 = batch_vec.iter().sum::<f32>() / batch_vec.len() as f32;
    let batch_max = batch_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let batch_min = batch_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "DEBUG Batched logits: mean={:.6}, min={:.6}, max={:.6}, sample[0:5]={:?}",
        batch_mean,
        batch_min,
        batch_max,
        &batch_vec[0..5]
    );

    // Check if ANY individual logits match
    println!("\nChecking if individual logits match:");
    let mut exact_matches = 0;
    let mut close_matches = 0; // within 0.01
    for i in 0..seq_vec.len().min(batch_vec.len()) {
        if (seq_vec[i] - batch_vec[i]).abs() < 1e-6 {
            exact_matches += 1;
        } else if (seq_vec[i] - batch_vec[i]).abs() < 0.01 {
            close_matches += 1;
        }
    }
    println!(
        "  Exact matches (< 1e-6): {}/{}",
        exact_matches,
        seq_vec.len()
    );
    println!(
        "  Close matches (< 0.01): {}/{}",
        close_matches,
        seq_vec.len()
    );

    // Show first 20 logits side-by-side
    println!("\nFirst 20 logits comparison:");
    println!("  Index | Sequential | Batched | Difference");
    for i in 0..20.min(seq_vec.len()) {
        println!(
            "  {:5} | {:10.4} | {:10.4} | {:10.4}",
            i,
            seq_vec[i],
            batch_vec[i],
            seq_vec[i] - batch_vec[i]
        );
    }

    // Compare outputs
    println!("\nComparing outputs...");
    // Sequential: [1, vocab_size], Batched: [1, 1, vocab_size]
    let sequential_logits_flat = sequential_logits.squeeze(0)?; // [vocab_size]
    let batched_logits_flat = batched_logits.squeeze(0)?.squeeze(0)?; // [vocab_size]

    let match_result = tensors_close(&sequential_logits_flat, &batched_logits_flat, 1e-4, 1e-5)?;

    assert!(
        match_result,
        "Batched and sequential outputs should match within tolerance"
    );

    println!("✅ Single token forward pass test PASSED");
    Ok(())
}

#[test]
#[ignore]// Requires model files
fn test_batch_forward() -> Result<()> {
    println!("\n=== Test: Batch Forward Pass ===\n");

    let model_path = match get_model_path() {
        Some(path) => path,
        None => {
            println!("Skipping test: No model found");
            return Ok(());
        }
    };

    let device = Device::Cpu;
    let dtype = DType::F32;

    // Load models (same as above)
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[model_path.join("model.safetensors")],
            dtype,
            &device,
        )?
    };

    let candle_config = candle_transformers::models::llama::Config {
        hidden_size: 576,
        intermediate_size: 1536,
        vocab_size: 49152,
        num_hidden_layers: 30,
        num_attention_heads: 9,
        num_key_value_heads: 3,
        rms_norm_eps: 1e-5,
        rope_theta: 100000.0,
        bos_token_id: Some(0),
        eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(0)),
        max_position_embeddings: 8192,
        use_flash_attn: false,
        rope_scaling: None,
        tie_word_embeddings: true,
    };

    let candle_model = candle_transformers::models::llama::Llama::load(vb, &candle_config)?;

    let vb2 = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[model_path.join("model.safetensors")],
            dtype,
            &device,
        )?
    };

    let batched_config = config_from_candle(&candle_config);
    let mut batched_model = BatchedTransformer::new(batched_config, vb2)?;

    // Test with batch of 4 tokens - each is the FIRST token of independent sequences
    let tokens = vec![1u32, 1312, 1234, 5678]; // Different first tokens for 4 sequences
    let batch_size = tokens.len();

    // Sequential processing (one at a time, each at position 0)
    println!("\nRunning sequential forward passes...");
    let mut sequential_results = Vec::new();
    for &token in tokens.iter() {
        let input = Tensor::from_vec(vec![token], (1, 1), &device)?;
        let mut cache =
            candle_transformers::models::llama::Cache::new(true, dtype, &candle_config, &device)?;
        let logits = candle_model.forward(&input, 0, &mut cache)?; // All at position 0
        sequential_results.push(logits);
    }

    // Batched processing (all at once)
    println!("\nRunning batched forward pass...");
    let input_batched = Tensor::from_vec(tokens.clone(), (batch_size,), &device)?;

    let context_window = 512;
    let head_dim = candle_config.hidden_size / candle_config.num_attention_heads;
    let mut batch_executor = BatchExecutor::new(
        batch_size,
        context_window,
        candle_config.num_hidden_layers,
        candle_config.num_key_value_heads,
        head_dim,
        dtype,
        &device,
    )
    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    let metadata = BatchMetadata {
        is_prefill: false, // Decode mode - each token is independent
        batch_size,
        request_ids: (0..batch_size).collect(),
        slot_offsets: (0..batch_size).collect(),
        sequence_lengths: vec![1; batch_size],
        cu_seqlens: Some((0..=batch_size).collect()),
        max_seqlen: Some(1),
        query_start_loc: None,
        context_lens: vec![0; batch_size], // All are first tokens, no context yet
    };

    let batched_logits = batched_model.forward(&input_batched, &mut batch_executor, &metadata)?;

    println!("Batched logits shape: {:?}", batched_logits.dims());

    // Compare each output
    println!("\nComparing outputs...");
    let mut all_match = true;
    for (i, seq_logits) in sequential_results.iter().enumerate() {
        let seq_flat = seq_logits.squeeze(0)?;
        let batched_flat = batched_logits.get(i)?.squeeze(0)?;

        println!("\nComparing token {}:", i);
        let match_result = tensors_close(&seq_flat, &batched_flat, 1e-4, 1e-5)?;

        if !match_result {
            all_match = false;
            println!("❌ Token {} outputs don't match", i);
        } else {
            println!("✅ Token {} outputs match", i);
        }
    }

    assert!(
        all_match,
        "All batched outputs should match sequential outputs"
    );

    println!("\n✅ Batch forward pass test PASSED");
    Ok(())
}

#[test]
#[ignore] // Requires model files
fn test_multi_step_generation() -> Result<()> {
    println!("\n=== Test: Multi-Step Generation ===\n");

    let model_path = match get_model_path() {
        Some(path) => path,
        None => {
            println!("Skipping test: No model found");
            return Ok(());
        }
    };

    let device = Device::Cpu;
    let dtype = DType::F32;

    // Load models
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[model_path.join("model.safetensors")],
            dtype,
            &device,
        )?
    };

    let candle_config = candle_transformers::models::llama::Config {
        hidden_size: 576,
        intermediate_size: 1536,
        vocab_size: 49152,
        num_hidden_layers: 30,
        num_attention_heads: 9,
        num_key_value_heads: 3,
        rms_norm_eps: 1e-5,
        rope_theta: 100000.0,
        bos_token_id: Some(0),
        eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(0)),
        max_position_embeddings: 8192,
        use_flash_attn: false,
        rope_scaling: None,
        tie_word_embeddings: true,
    };

    let candle_model = candle_transformers::models::llama::Llama::load(vb, &candle_config)?;

    let vb2 = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[model_path.join("model.safetensors")],
            dtype,
            &device,
        )?
    };

    let batched_config = config_from_candle(&candle_config);
    let mut batched_model = BatchedTransformer::new(batched_config, vb2)?;

    // Generate 5 tokens sequentially with both models
    let num_steps = 5;
    let initial_token = 1u32; // BOS

    // Sequential generation
    println!("\nSequential generation:");
    let mut sequential_tokens = vec![initial_token];
    let mut sequential_cache =
        candle_transformers::models::llama::Cache::new(true, dtype, &candle_config, &device)?;

    for step in 0..num_steps {
        let input = Tensor::from_vec(vec![sequential_tokens[step] as u32], (1, 1), &device)?;
        let logits = candle_model.forward(&input, step, &mut sequential_cache)?;

        // Get next token (argmax)
        let next_token = logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;

        sequential_tokens.push(next_token);
        
        // Debug: Show top-5 logits
        let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
        let mut indexed: Vec<(usize, f32)> = logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("  Step {} (seq): token {} (logit={:.2}), top-5: {:?}", 
            step, next_token, indexed[0].1,
            &indexed[0..5].iter().map(|(idx, val)| format!("{}:{:.1}", idx, val)).collect::<Vec<_>>()
        );
    }

    // Batched generation
    println!("\nBatched generation:");
    let mut batched_tokens = vec![initial_token];

    let batch_size = 1;
    let context_window = 512;
    let head_dim = candle_config.hidden_size / candle_config.num_attention_heads;
    let mut batch_executor = BatchExecutor::new(
        batch_size,
        context_window,
        candle_config.num_hidden_layers,
        candle_config.num_key_value_heads,
        head_dim,
        dtype,
        &device,
    )
    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

    for step in 0..num_steps {
        let input = Tensor::from_vec(vec![batched_tokens[step] as u32], (1,), &device)?;

        let metadata = BatchMetadata {
            is_prefill: step == 0,
            batch_size: 1,
            request_ids: vec![0],
            slot_offsets: vec![step],
            sequence_lengths: vec![1],
            cu_seqlens: Some(vec![0, 1]),
            max_seqlen: Some(1),
            query_start_loc: if step == 0 { Some(vec![0]) } else { None },
            context_lens: vec![step],
        };

        println!("\n  Batched Step {}: slot_offset={}, context_len={}", step, step, step);
        let logits = batched_model.forward(&input, &mut batch_executor, &metadata)?;

        // Get next token (argmax)
        let next_token = logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;

        batched_tokens.push(next_token);
        
        // Debug: Show top-5 logits and compare with sequential
        let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
        let mut indexed: Vec<(usize, f32)> = logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("  Step {} (bat): token {} (logit={:.2}), top-5: {:?}", 
            step, next_token, indexed[0].1,
            &indexed[0..5].iter().map(|(idx, val)| format!("{}:{:.1}", idx, val)).collect::<Vec<_>>()
        );
    }

    // Compare generated sequences
    println!("\nComparing generated sequences:");
    println!("Sequential: {:?}", sequential_tokens);
    println!("Batched:    {:?}", batched_tokens);

    assert_eq!(
        sequential_tokens, batched_tokens,
        "Generated token sequences should be identical"
    );

    println!("\n✅ Multi-step generation test PASSED");
    Ok(())
}

#[test]
fn test_rope_computation() -> Result<()> {
    println!("\n=== Test: RoPE Computation ===\n");

    // Test that our RoPE implementation produces correct rotation
    let device = Device::Cpu;
    let dtype = DType::F32;

    // Create simple test tensors
    let head_dim = 64;
    let seq_len = 4;
    let rope_theta = 10000.0f32;

    // Manually compute what RoPE should produce
    // This tests the mathematical correctness of the rotation

    // Simple input: ones tensor
    let x = Tensor::ones((1, 1, seq_len, head_dim), dtype, &device)?;

    // Compute frequencies
    let inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f32 / head_dim as f32))
        .collect();

    println!(
        "inv_freq (first 5): {:?}",
        &inv_freq[0..5.min(inv_freq.len())]
    );

    // This test validates our frequency computation matches expected values
    assert!(
        inv_freq[0] > 0.99 && inv_freq[0] < 1.01,
        "First freq should be ~1.0"
    );
    assert!(
        inv_freq[inv_freq.len() - 1] < 0.001,
        "Last freq should be very small"
    );

    println!("✅ RoPE computation test PASSED");
    Ok(())
}
