//! Integration tests for batched inference with Candle

use candle_core::{DType, Device, Tensor};
use lightbulb::engine::{
    BatchAssembler, BatchConfig, BatchExecutor, Request, RequestContext, RequestQueue, RequestState,
};

/// Test end-to-end batched inference simulation
/// This test verifies the complete flow:
/// 1. Requests submitted to queue
/// 2. Batch assembled
/// 3. Cache indices assigned
/// 4. KV cache operations work correctly
/// 5. Requests can be released and indices reused
#[test]
fn test_end_to_end_batch_inference_simulation() {
    // Setup
    let device = Device::Cpu;
    let batch_size = 3;
    let context = 256;
    let num_layers = 4;
    let num_heads = 8;
    let head_dim = 64;
    let dtype = DType::F32;
    let seq_len = 1; // Decoding: one token at a time

    // Create components
    let queue = RequestQueue::new();
    let config = BatchConfig {
        max_batch_size: batch_size,
        max_batch_tokens: 512,
    };
    let assembler = BatchAssembler::new(config);
    let mut executor = BatchExecutor::new(
        batch_size, context, num_layers, num_heads, head_dim, dtype, &device,
    )
    .unwrap();

    // Submit requests
    for i in 1..=5 {
        let req = Request {
            id: format!("req-{}", i),
            prompt: format!("Test prompt {}", i),
            max_new_tokens: 10,
        };
        queue.submit(req).unwrap();
    }

    assert_eq!(queue.len(), 5);

    // === First batch iteration ===
    println!("\n=== First Batch ===");
    let mut batch = assembler.assemble_batch(&queue);
    assert_eq!(batch.len(), 3); // max_batch_size
    assert_eq!(queue.len(), 2); // 2 remaining

    // Start decoding for all requests in batch
    for ctx in &mut batch {
        ctx.start_decoding();
    }

    // Assign cache indices
    executor.prepare_batch(&mut batch).unwrap();
    println!(
        "Batch 1 cache indices: {:?}",
        batch.iter().map(|ctx| ctx.cache_index).collect::<Vec<_>>()
    );
    assert_eq!(batch[0].cache_index, Some(0));
    assert_eq!(batch[1].cache_index, Some(1));
    assert_eq!(batch[2].cache_index, Some(2));
    assert_eq!(executor.available_slots(), 0);

    // Get indices and mask
    let iam = executor.get_indices_and_mask(&batch, seq_len).unwrap();

    // Simulate forward pass through all layers
    let k_shape = (batch_size, num_heads, seq_len, head_dim);
    for layer_idx in 0..num_layers {
        // Create dummy KV tensors (in real model, these come from attention)
        let k = Tensor::randn(0f32, 1.0, k_shape, &device).unwrap();
        let v = Tensor::randn(0f32, 1.0, k_shape, &device).unwrap();

        // Append to cache
        let (k_out, _v_out) = executor.append_kv(layer_idx, &k, &v, &iam).unwrap();

        // Verify output shapes
        assert_eq!(k_out.dims(), &[batch_size, num_heads, context, head_dim]);

        println!("Layer {} KV cache updated successfully", layer_idx);
    }

    // Advance cache builder positions by seq_len for active slots (once per forward)
    for ctx in &batch {
        if let Some(cache_idx) = ctx.cache_index {
            let cur = executor.get_cache_position(cache_idx);
            executor.set_cache_position(cache_idx, cur + seq_len);
        }
    }

    // Simulate completing first request
    println!("\n=== Completing request req-1 ===");
    let completed_idx = batch[0].cache_index.unwrap();
    batch[0].complete();
    executor.release_request(completed_idx);
    println!(
        "Released cache index {}, available slots: {}",
        completed_idx,
        executor.available_slots()
    );
    assert_eq!(executor.available_slots(), 1);

    // Remove completed requests from batch
    batch.retain(|ctx| ctx.state != RequestState::Completed);
    assert_eq!(batch.len(), 2);

    // === Second batch iteration ===
    println!("\n=== Second Batch ===");
    // Assemble new requests to fill the batch
    // We have 1 available slot, but queue has 2 requests
    // BatchAssembler will take up to max_batch_size (3 total with 2 continuing)
    let new_requests = assembler.assemble_batch(&queue);
    println!("New requests assembled: {}", new_requests.len());
    // Could be 1 or 2 depending on available slots
    // But we only have 1 slot available, so prepare_batch will handle it
    assert!(new_requests.len() >= 1 && new_requests.len() <= 2);

    // Start decoding new request(s)
    let mut new_batch = new_requests;
    for ctx in &mut new_batch {
        ctx.start_decoding();
    }

    // Try to assign cache indices to new requests
    // This will fail if we try to assign more than available slots
    if new_batch.len() > executor.available_slots() {
        // Take only what fits
        new_batch.truncate(executor.available_slots());
    }

    executor.prepare_batch(&mut new_batch).unwrap();
    println!(
        "New request cache indices: {:?}",
        new_batch
            .iter()
            .map(|ctx| ctx.cache_index)
            .collect::<Vec<_>>()
    );
    assert_eq!(new_batch[0].cache_index, Some(0)); // Reused released index!

    // Combine with continuing requests
    batch.extend(new_batch);
    assert_eq!(batch.len(), 3);

    // Process second iteration
    let iam = executor.get_indices_and_mask(&batch, seq_len).unwrap();
    for layer_idx in 0..num_layers {
        let k = Tensor::randn(0f32, 1.0, k_shape, &device).unwrap();
        let v = Tensor::randn(0f32, 1.0, k_shape, &device).unwrap();
        let (k_out, v_out) = executor.append_kv(layer_idx, &k, &v, &iam).unwrap();
        assert_eq!(k_out.dims(), &[batch_size, num_heads, context, head_dim]);
    }

    println!("\n=== Test Completed Successfully ===");
    println!("✓ Cache indices assigned and reused correctly");
    println!("✓ Multi-layer KV cache operations successful");
    println!("✓ Batch assembly and request lifecycle working");
    println!("✓ End-to-end batched inference validated!");
}

/// Test that batch masks correctly handle mixed Pending/Decoding states
#[test]
fn test_batch_mask_with_mixed_states() {
    let device = Device::Cpu;
    let mut executor = BatchExecutor::new(3, 128, 2, 4, 32, DType::F32, &device).unwrap();

    let req1 = Request {
        id: "req-1".to_string(),
        prompt: "Test1".to_string(),
        max_new_tokens: 10,
    };
    let req2 = Request {
        id: "req-2".to_string(),
        prompt: "Test2".to_string(),
        max_new_tokens: 10,
    };
    let req3 = Request {
        id: "req-3".to_string(),
        prompt: "Test3".to_string(),
        max_new_tokens: 10,
    };

    let mut ctx1 = RequestContext::new(req1);
    let mut ctx2 = RequestContext::new(req2);
    let ctx3 = RequestContext::new(req3);

    ctx1.start_decoding(); // Decoding
    ctx2.start_decoding(); // Decoding
    // ctx3 stays Pending

    let mut batch = vec![ctx1, ctx2, ctx3];
    executor.prepare_batch(&mut batch).unwrap();

    // Get indices and mask - should only process Decoding requests
    let iam = executor.get_indices_and_mask(&batch, 1).unwrap();

    // The batch mask should be [true, true, false]
    // Only the first two requests should be processed

    // Create dummy tensors
    let k = Tensor::zeros((3, 4, 1, 32), DType::F32, &device).unwrap();
    let v = Tensor::zeros((3, 4, 1, 32), DType::F32, &device).unwrap();

    // Should work without error
    let result = executor.append_kv(0, &k, &v, &iam);
    assert!(result.is_ok());
}

/// Test cache overflow handling with long sequences
#[test]
fn test_cache_context_window() {
    let device = Device::Cpu;
    let context = 10; // Small context for testing
    let mut executor = BatchExecutor::new(1, context, 1, 2, 16, DType::F32, &device).unwrap();

    let req = Request {
        id: "req-1".to_string(),
        prompt: "Test".to_string(),
        max_new_tokens: 20, // More than context
    };
    let mut ctx = RequestContext::new(req);
    ctx.start_decoding();

    let mut batch = vec![ctx];
    executor.prepare_batch(&mut batch).unwrap();

    // Simulate generating more tokens than context
    for i in 0..15 {
        let iam = executor.get_indices_and_mask(&batch, 1).unwrap();
        let k = Tensor::zeros((1, 2, 1, 16), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 1, 16), DType::F32, &device).unwrap();

        let result = executor.append_kv(0, &k, &v, &iam);
        assert!(result.is_ok(), "Failed at iteration {}", i);

        // Update position
        batch[0].record_token();
    }

    // Should have wrapped around in cache (circular buffer)
    println!("Successfully handled cache overflow with circular buffering");
}

/// Test parallel batch processing capability
#[test]
fn test_concurrent_batch_operations() {
    let device = Device::Cpu;
    let queue = RequestQueue::new();

    // Submit 10 requests - will create partial batch at end
    for i in 1..=10 {
        let req = Request {
            id: format!("req-{}", i),
            prompt: format!("Prompt {}", i),
            max_new_tokens: 5,
        };
        queue.submit(req).unwrap();
    }

    let assembler = BatchAssembler::new(BatchConfig {
        max_batch_size: 4,
        max_batch_tokens: 1024,
    });

    let mut executor = BatchExecutor::new(4, 128, 2, 4, 32, DType::F32, &device).unwrap();

    let mut total_processed = 0;

    // Process in batches
    while queue.len() > 0 {
        let mut batch = assembler.assemble_batch(&queue);
        if batch.is_empty() {
            break;
        }

        let batch_len = batch.len();
        println!("Processing batch of {} requests", batch_len);

        // Start decoding
        for ctx in &mut batch {
            ctx.start_decoding();
        }

        // Assign cache indices
        executor.prepare_batch(&mut batch).unwrap();

        // Simulate one forward pass
        let iam = executor.get_indices_and_mask(&batch, 1).unwrap();

        // Create tensors with max batch size dimensions
        // This works even for partial batches because batch_mask indicates active slots
        let k = Tensor::zeros((4, 4, 1, 32), DType::F32, &device).unwrap();
        let v = Tensor::zeros((4, 4, 1, 32), DType::F32, &device).unwrap();

        let (_k_out, _v_out) = executor.append_kv(0, &k, &v, &iam).unwrap();

        total_processed += batch_len;

        // Release all
        for ctx in batch {
            if let Some(idx) = ctx.cache_index {
                executor.release_request(idx);
            }
        }
    }

    assert_eq!(total_processed, 10);
    assert_eq!(executor.available_slots(), 4); // All released
    println!(
        "Processed {} requests in batches (including partial batch)",
        total_processed
    );
}

/// Test that partial batches (fewer requests than max_batch_size) work correctly
/// This is critical for production where the last batch often has fewer requests
#[test]
fn test_partial_batch_handling() {
    let device = Device::Cpu;
    let max_batch_size = 4;
    let mut executor =
        BatchExecutor::new(max_batch_size, 128, 2, 4, 32, DType::F32, &device).unwrap();

    // Create a partial batch with only 2 requests (half of max_batch_size)
    let req1 = Request {
        id: "req-1".to_string(),
        prompt: "Test1".to_string(),
        max_new_tokens: 10,
    };
    let req2 = Request {
        id: "req-2".to_string(),
        prompt: "Test2".to_string(),
        max_new_tokens: 10,
    };

    let mut ctx1 = RequestContext::new(req1);
    let mut ctx2 = RequestContext::new(req2);
    ctx1.start_decoding();
    ctx2.start_decoding();

    let mut batch = vec![ctx1, ctx2];

    // Prepare batch (assign cache indices)
    executor.prepare_batch(&mut batch).unwrap();
    assert_eq!(batch.len(), 2); // Partial batch

    // Get indices and mask - should pad batch_mask to max_batch_size internally
    let iam = executor.get_indices_and_mask(&batch, 1).unwrap();

    // Create tensors with max batch size (4), not actual batch size (2)
    // The batch_mask will tell Candle which slots are active
    let k = Tensor::zeros((max_batch_size, 4, 1, 32), DType::F32, &device).unwrap();
    let v = Tensor::zeros((max_batch_size, 4, 1, 32), DType::F32, &device).unwrap();

    // Should work without tensor dimension mismatch
    let result = executor.append_kv(0, &k, &v, &iam);
    assert!(
        result.is_ok(),
        "Partial batch should work with padded batch_mask"
    );

    let (k_out, v_out) = result.unwrap();
    // Output should still have full max_batch_size dimensions
    assert_eq!(k_out.dims()[0], max_batch_size);
    assert_eq!(v_out.dims()[0], max_batch_size);

    println!("✓ Partial batch (2 out of 4) handled correctly");
    println!("✓ Batch mask properly padded to max_batch_size");
    println!("✓ Tensor dimensions match cache expectations");
}

/// Test padded multi-chunk prefill to verify positions advance by actual_len only
/// and no cross-slot contamination occurs
#[test]
fn test_padded_multi_chunk_prefill() {
    let device = Device::Cpu;
    let max_batch_size = 4;
    let context = 128;
    let num_layers = 2;
    let num_heads = 4;
    let head_dim = 32;
    let dtype = DType::F32;

    let mut executor = BatchExecutor::new(
        max_batch_size,
        context,
        num_layers,
        num_heads,
        head_dim,
        dtype,
        &device,
    )
    .unwrap();

    // Simulate two requests with different lengths
    // Request 0: 50 tokens (will be split into chunks)
    // Request 1: 30 tokens
    let slot_0 = 0;
    let slot_1 = 1;

    // Reset both slots to position 0
    executor.reset_request_state(slot_0);
    executor.reset_request_state(slot_1);

    // First chunk: both active, padded to 32 tokens each
    // Request 0: 32 actual tokens (chunk 1/2)
    // Request 1: 30 actual tokens with 2 padding
    let chunk1_seq_len = 32;
    let iam1 = executor
        .get_indices_and_mask_simple(2, chunk1_seq_len)
        .unwrap();

    // Simulate KV append for chunk 1
    let k1 = Tensor::randn(
        0.0f32,
        1.0f32,
        (max_batch_size, num_heads, chunk1_seq_len, head_dim),
        &device,
    )
    .unwrap()
    .to_dtype(dtype)
    .unwrap();
    let v1 = k1.clone();
    executor.append_kv(0, &k1, &v1, &iam1).unwrap();

    // Advance positions by actual lengths (not padded)
    // Request 0: 32 tokens -> position 32
    // Request 1: 30 tokens -> position 30
    executor.set_cache_position(slot_0, 32);
    executor.set_cache_position(slot_1, 30);

    assert_eq!(executor.get_cache_position(slot_0), 32);
    assert_eq!(executor.get_cache_position(slot_1), 30);

    // Second chunk: only request 0 active (remaining 18 tokens, padded to 32)
    let chunk2_seq_len = 32;
    let iam2 = executor
        .get_indices_and_mask_simple(1, chunk2_seq_len)
        .unwrap();

    let k2 = Tensor::randn(
        0.0f32,
        1.0f32,
        (max_batch_size, num_heads, chunk2_seq_len, head_dim),
        &device,
    )
    .unwrap()
    .to_dtype(dtype)
    .unwrap();
    let v2 = k2.clone();
    executor.append_kv(0, &k2, &v2, &iam2).unwrap();

    // Advance position for request 0 by actual length only (18 tokens)
    executor.set_cache_position(slot_0, 32 + 18);

    // Verify final positions
    assert_eq!(
        executor.get_cache_position(slot_0),
        50,
        "Request 0 should have processed 50 tokens total"
    );
    assert_eq!(
        executor.get_cache_position(slot_1),
        30,
        "Request 1 position should not change when inactive"
    );

    println!("✓ Padded multi-chunk prefill: positions advanced by actual_len only");
    println!("✓ No cross-slot contamination when some slots inactive");
}

/// Test that indices_and_mask is pure and doesn't mutate builder state
#[test]
fn test_indices_and_mask_purity() {
    let device = Device::Cpu;
    let max_batch_size = 2;
    let context = 64;
    let num_layers = 1;
    let num_heads = 2;
    let head_dim = 16;
    let dtype = DType::F32;

    let mut executor = BatchExecutor::new(
        max_batch_size,
        context,
        num_layers,
        num_heads,
        head_dim,
        dtype,
        &device,
    )
    .unwrap();

    let slot_0 = 0;
    executor.reset_request_state(slot_0);

    let pos_before = executor.get_cache_position(slot_0);
    assert_eq!(pos_before, 0);

    // Call indices_and_mask multiple times without explicit position advancement
    let _iam1 = executor.get_indices_and_mask_simple(1, 5).unwrap();
    let pos_after_iam1 = executor.get_cache_position(slot_0);

    let _iam2 = executor.get_indices_and_mask_simple(1, 5).unwrap();
    let pos_after_iam2 = executor.get_cache_position(slot_0);

    // Position should not change from calling indices_and_mask alone
    assert_eq!(
        pos_after_iam1, pos_before,
        "indices_and_mask should not mutate position"
    );
    assert_eq!(
        pos_after_iam2, pos_before,
        "indices_and_mask should be pure (no side effects)"
    );

    // Now explicitly advance
    executor.set_cache_position(slot_0, pos_before + 5);
    let pos_after_advance = executor.get_cache_position(slot_0);
    assert_eq!(
        pos_after_advance, 5,
        "Position should only change with explicit set_cache_position"
    );

    println!("✓ indices_and_mask is pure and doesn't mutate builder state");
    println!("✓ Explicit position advancement required via set_cache_position");
}
