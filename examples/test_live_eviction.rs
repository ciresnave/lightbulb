//! Live eviction validation — forces context pressure to trigger segment demotion
//!
//! Uses a tiny context window (64 tokens) so that a few prompts fill the cache
//! past the eviction threshold, triggering the full pipeline:
//!   attention scoring → rank candidates → demote to RAM → evict from cache
//!
//! Run with: cargo run --example test_live_eviction --release
//!
//! Requires: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

use anyhow::Result;
use lightbulb::cache::SegmentedCacheConfig;
use lightbulb::engine::{Request, RequestContext, RequestState};
use lightbulb::model::ParallelModelManager;

fn main() -> Result<()> {
    println!("\n=== Live Eviction Validation ===\n");

    let model_path = find_model()?;
    println!("Model: {}", model_path);

    // Load model with TINY context window to force pressure quickly
    println!("Loading model (context=64 tokens)...");
    let start = std::time::Instant::now();

    let mut model = ParallelModelManager::load_gguf(
        &model_path,
        1,  // max_batch_size (single request at a time)
        64, // context_length — deliberately tiny to force eviction
        None,
        None,
    )?;

    println!("Model loaded in {:.2}s\n", start.elapsed().as_secs_f32());

    // Enable live segmented cache (NOT shadow mode)
    let config = SegmentedCacheConfig {
        enabled: true,
        capture_attention: true,
        shadow_mode: false,    // LIVE eviction
        tiered_demotion: true, // Demote to RAM, don't delete
        log_span_attention: true,
        h2o_update_interval: 2, // Frequent updates for small context
        attention_power: 2.0,
        base_decay_rate: 0.05,             // Faster decay for tiny context
        eviction_pressure_threshold: 0.70, // Lower threshold — evict at 70%
        per_token_eviction: false,
        max_ram_demoted_segments: 16,
        disk_storage_path: None, // RAM only, no disk
        ..Default::default()
    };

    println!("Enabling live segmented cache:");
    println!("  context_length: 64 tokens");
    println!("  eviction_pressure_threshold: 70%");
    println!("  tiered_demotion: true (RAM)");
    println!("  shadow_mode: false (LIVE)");
    println!();

    model.enable_segmented_cache(config);

    // Turn 1: ~20 tokens of prompt + ~10 generated = ~30 tokens (47% of 64)
    println!("--- Turn 1 ---");
    run_turn(&mut model, "turn1", "What is the capital of France?", 10)?;
    report_state(&model);

    // Turn 2: ~15 tokens + ~10 generated = ~25 more (total ~55, 86% — should trigger eviction)
    println!("--- Turn 2 ---");
    run_turn(&mut model, "turn2", "Write a short poem about cats.", 10)?;
    report_state(&model);

    // Turn 3: Should definitely trigger eviction of Turn 1's spans
    println!("--- Turn 3 ---");
    run_turn(&mut model, "turn3", "Explain gravity briefly.", 10)?;
    report_state(&model);

    println!("\n=== Live Eviction Validation Complete ===");
    Ok(())
}

fn run_turn(
    model: &mut ParallelModelManager,
    id: &str,
    prompt: &str,
    max_tokens: usize,
) -> Result<()> {
    let req = Request {
        id: id.to_string(),
        prompt: prompt.to_string(),
        max_new_tokens: max_tokens,
    };

    let mut batch = vec![RequestContext::new(req)];
    let start = std::time::Instant::now();
    let mut steps = 0;

    while !batch.is_empty() && steps < max_tokens + 5 {
        steps += 1;
        model.forward_batch(&mut batch)?;
        batch.retain(|ctx| ctx.state != RequestState::Completed);
    }

    println!(
        "  Completed in {:.2}s ({} steps)",
        start.elapsed().as_secs_f32(),
        steps
    );
    Ok(())
}

fn report_state(model: &ParallelModelManager) {
    let max_util = model.cache_builder().get_max_utilization();
    let summaries = model.cache_builder().compute_span_attention_summary();

    println!("  Cache utilization: {:.1}%", max_util * 100.0);
    println!("  Active spans: {}", summaries.len());

    if !summaries.is_empty() {
        for (span_id, tag, tokens, total_attn, _avg) in &summaries {
            let span_info = model
                .cache_builder()
                .get_span(*span_id)
                .map(|s| format!("[{}..{}]", s.start_pos, s.end_pos))
                .unwrap_or_else(|| "?".to_string());

            println!(
                "    Span {:2} | {:?} {} | {} tokens | attn: {:.4}",
                span_id, tag, span_info, tokens, total_attn
            );
        }
    }
    println!();
}

fn find_model() -> Result<String> {
    let candidates = [
        "../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    ];
    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Ok(path.to_string());
        }
    }
    anyhow::bail!("Model not found. Tried: {:?}", candidates)
}
