/// Example demonstrating StreamingLLM-style KV cache eviction
///
/// This shows how attention sinks + sliding window enable constant memory
/// for long-context generation without quality degradation.
///
/// Run with: cargo run --example streaming_llm_demo
use anyhow::Result;
use candlelight::core::{DType, Device, IndexOp, Tensor};
use lightbulb::cache::{ParallelCacheBuilder, StreamingConfig};

fn main() -> Result<()> {
    println!("🌊 StreamingLLM Demonstration");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let device = Device::Cpu;

    // Setup: Small cache with StreamingLLM enabled
    // 4 attention sinks + 8 eviction zone + 16 window = 28 total cache size
    let streaming_config = StreamingConfig::new(4, 16);
    println!("Configuration:");
    println!("  Attention sinks:  {} tokens", streaming_config.sink_size);
    println!(
        "  Sliding window:   {} tokens",
        streaming_config.window_size
    );
    println!(
        "  Total cache size: {} tokens (4 sinks + 8 eviction + 16 window)",
        28
    );
    println!("  Enabled:          {}\n", streaming_config.enabled);

    let batch_size = 2;
    let cache_size = 28; // Larger than sink + window to create eviction zone
    let mut cache_builder = ParallelCacheBuilder::new(batch_size, cache_size, DType::F32, &device)?;
    cache_builder.set_streaming_config(Some(streaming_config.clone()));

    // Create a KV cache (simplified: 4 heads, 8 dim)
    let num_heads = 4;
    let head_dim = 8;
    let mut kv_cache = cache_builder.make_cache(num_heads, head_dim)?;

    println!("📝 Simulating long context generation...\n");

    // Simulate processing 50 tokens (much longer than cache size of 20)
    let num_tokens = 50;

    for token_idx in 0..num_tokens {
        // Process 1 token at a time
        let batch_mask = vec![true, false]; // Only process slot 0

        // Get cache indices for this token
        let iam = cache_builder.indices_and_mask(1, &batch_mask)?;

        // Extract the cache index for slot 0 (the active slot)
        let cache_idx = iam.indices.i((0, 0))?.to_scalar::<u32>()? as usize;

        // Simulate K/V data (would come from model in real usage)
        let k_shape = (batch_size, num_heads, 1, head_dim);
        let v_shape = (batch_size, num_heads, 1, head_dim);
        let k_data = (Tensor::ones(k_shape, DType::F32, &device)? * (token_idx as f64))?;
        let v_data = (Tensor::ones(v_shape, DType::F32, &device)? * (token_idx as f64))?;

        // Append to cache
        let (_k_cache, _v_cache) = kv_cache.append(&k_data, &v_data, &iam)?;

        // Update position
        cache_builder.set_position(0, token_idx + 1);

        // Log interesting milestones
        if token_idx < 5
            || token_idx == 19
            || token_idx >= 27 && token_idx <= 35
            || token_idx == num_tokens - 1
        {
            println!(
                "Token {:2}: cache_idx = {:2}  ({})",
                token_idx,
                cache_idx,
                describe_cache_position(token_idx, &streaming_config, cache_size)
            );
        } else if token_idx == 5 {
            println!("  ... (sequential filling continues)");
        } else if token_idx == 30 {
            println!("  ... (eviction continues in middle zone)");
        }
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✅ StreamingLLM Demo Complete!\n");

    println!("📊 Key Observations:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("• Tokens 0-3:   Attention sinks (never evicted)");
    println!("• Tokens 4-27:  Initial sequential fill");
    println!("• Token 28+:    Cache full! Evicting from middle zone [4..12)");
    println!("• Recent 16:    Always preserved in window [12..28)");
    println!(
        "• Memory:       Constant {} tokens (vs {} with full cache)",
        cache_size, num_tokens
    );
    println!(
        "\n💾 Memory savings: {:.1}%",
        (1.0 - cache_size as f64 / num_tokens as f64) * 100.0
    );
    println!("🎯 Quality:      Minimal degradation (sinks stabilize attention)\n");

    Ok(())
}

fn describe_cache_position(
    token_idx: usize,
    config: &StreamingConfig,
    cache_size: usize,
) -> String {
    if token_idx < config.sink_size {
        format!("Attention sink #{}", token_idx)
    } else if token_idx < cache_size {
        "Sequential fill".to_string()
    } else {
        let recent_start = cache_size - config.window_size;
        format!(
            "Evicting middle, protecting recent window [{}..]",
            recent_start
        )
    }
}
