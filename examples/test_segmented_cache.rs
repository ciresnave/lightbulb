//! Shadow mode validation for segmented KV cache with attention-driven eviction
//!
//! This test validates the full segmented cache pipeline by:
//! - Loading a quantized GGUF model via ParallelModelManager
//! - Enabling segmented cache in observation (shadow) mode
//! - Processing multiple prompts to create UserInput + ModelGeneration spans
//! - Examining attention distributions per span
//! - Logging eviction candidates (without evicting)
//! - Verifying span lifecycle (creation, tracking, completion)
//!
//! Run with: cargo run --example test_segmented_cache --release
//!
//! Requires: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

use anyhow::Result;
use lightbulb::cache::SegmentedCacheConfig;
use lightbulb::engine::{Request, RequestContext, RequestState};
use lightbulb::model::ParallelModelManager;

fn main() -> Result<()> {
    println!("\n=== Segmented KV Cache Shadow Mode Validation ===\n");

    // Load the smallest available GGUF model
    let model_path = "../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

    if !std::path::Path::new(model_path).exists() {
        // Try alternate path (running from project root vs lightbulb/)
        let alt_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
        if !std::path::Path::new(alt_path).exists() {
            eprintln!("Model not found at {} or {}", model_path, alt_path);
            eprintln!("Please ensure the GGUF model file exists.");
            return Ok(());
        }
        return run_validation(alt_path);
    }

    run_validation(model_path)
}

fn run_validation(model_path: &str) -> Result<()> {
    println!("Model: {}", model_path);

    // Load model
    println!("Loading quantized model...");
    let start = std::time::Instant::now();

    let mut model = ParallelModelManager::load_gguf(
        model_path, 2,    // max_batch_size (small for testing)
        256,  // context_length (small to trigger pressure faster)
        None, // chunked_prefill_config
        None, // device (auto-detect)
    )?;

    let load_time = start.elapsed();
    println!(
        "Model loaded in {:.2}s (layers={}, heads={}, hidden={})\n",
        load_time.as_secs_f32(),
        model.config().num_hidden_layers,
        model.config().num_attention_heads,
        model.config().hidden_size,
    );

    // Enable segmented cache in observation mode
    let config = SegmentedCacheConfig {
        enabled: true,
        capture_attention: true,
        shadow_mode: true,
        log_span_attention: true,
        h2o_update_interval: 4, // Log every 4 decode steps for visibility
        attention_power: 2.0,
        base_decay_rate: 0.02,
        eviction_pressure_threshold: 0.80,
        ..Default::default()
    };

    println!("Enabling segmented cache (shadow mode)...");
    println!("  attention_power: {}", config.attention_power);
    println!("  base_decay_rate: {}", config.base_decay_rate);
    println!(
        "  eviction_pressure_threshold: {}",
        config.eviction_pressure_threshold
    );
    println!("  h2o_update_interval: {}", config.h2o_update_interval);
    println!();

    model.enable_segmented_cache(config);

    // === Turn 1: First user prompt ===
    println!("--- Turn 1: First user prompt ---");
    let req1 = Request {
        id: "turn1".to_string(),
        prompt: "What is the capital of France? Please explain briefly.".to_string(),
        max_new_tokens: 30,
    };

    let mut batch = vec![RequestContext::new(req1)];
    let gen_start = std::time::Instant::now();

    let mut step = 0;
    while !batch.is_empty() && step < 40 {
        step += 1;
        model.forward_batch(&mut batch)?;
        batch.retain(|ctx| ctx.state != RequestState::Completed);
    }

    let turn1_time = gen_start.elapsed();
    println!(
        "Turn 1 complete in {:.2}s ({} decode steps)\n",
        turn1_time.as_secs_f32(),
        step
    );

    // === Turn 2: Second user prompt (different topic) ===
    println!("--- Turn 2: Second user prompt ---");
    let req2 = Request {
        id: "turn2".to_string(),
        prompt: "Write a haiku about the ocean.".to_string(),
        max_new_tokens: 30,
    };

    let mut batch = vec![RequestContext::new(req2)];
    let gen_start = std::time::Instant::now();

    let mut step = 0;
    while !batch.is_empty() && step < 40 {
        step += 1;
        model.forward_batch(&mut batch)?;
        batch.retain(|ctx| ctx.state != RequestState::Completed);
    }

    let turn2_time = gen_start.elapsed();
    println!(
        "Turn 2 complete in {:.2}s ({} decode steps)\n",
        turn2_time.as_secs_f32(),
        step
    );

    // === Turn 3: Third prompt to accumulate more spans ===
    println!("--- Turn 3: Third user prompt ---");
    let req3 = Request {
        id: "turn3".to_string(),
        prompt: "Explain how a combustion engine works in simple terms.".to_string(),
        max_new_tokens: 40,
    };

    let mut batch = vec![RequestContext::new(req3)];
    let gen_start = std::time::Instant::now();

    let mut step = 0;
    while !batch.is_empty() && step < 50 {
        step += 1;
        model.forward_batch(&mut batch)?;
        batch.retain(|ctx| ctx.state != RequestState::Completed);
    }

    let turn3_time = gen_start.elapsed();
    println!(
        "Turn 3 complete in {:.2}s ({} decode steps)\n",
        turn3_time.as_secs_f32(),
        step
    );

    // === Final Report ===
    println!("=== Validation Summary ===\n");

    // Check cache utilization
    let max_util = model.cache_builder().get_max_utilization();
    println!("Cache utilization: {:.1}%", max_util * 100.0);

    // Check span attention summaries
    let summaries = model.cache_builder().compute_span_attention_summary();
    println!("Active spans: {}", summaries.len());

    if summaries.is_empty() {
        println!("WARNING: No span attention data collected!");
        println!("  This could mean:");
        println!("  - capture_attention is not reaching the attention layer");
        println!("  - H2O policy is not receiving attention weights");
        println!("  - Spans are being created but not scored");
    } else {
        println!("\nSpan Attention Rankings (most attended first):");
        println!(
            "{:>5} | {:>18} | {:>6} | {:>12} | {:>12}",
            "ID", "Tag", "Tokens", "Total Attn", "Avg Attn"
        );
        println!("{}", "-".repeat(65));
        for (span_id, tag, token_count, total_attn, avg_attn) in &summaries {
            println!(
                "{:>5} | {:>18?} | {:>6} | {:>12.4} | {:>12.6}",
                span_id, tag, token_count, total_attn, avg_attn
            );
        }
    }

    // Log eviction candidates
    println!("\nEviction candidates (shadow mode — no actual eviction):");
    let config = SegmentedCacheConfig::observation();
    model.cache_builder().log_eviction_candidates(&config, 10);

    println!("\nTiming:");
    println!("  Model load:  {:.2}s", load_time.as_secs_f32());
    println!("  Turn 1:      {:.2}s", turn1_time.as_secs_f32());
    println!("  Turn 2:      {:.2}s", turn2_time.as_secs_f32());
    println!("  Turn 3:      {:.2}s", turn3_time.as_secs_f32());
    println!(
        "  Total:       {:.2}s",
        (load_time + turn1_time + turn2_time + turn3_time).as_secs_f32()
    );

    println!("\nValidation complete.");
    Ok(())
}
