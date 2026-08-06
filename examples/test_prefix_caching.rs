// Test prefix KV caching with repeated prompts
//
// This example demonstrates that prefix caching is fully functional and provides
// significant TTFT (Time To First Token) reductions for repeated prompt prefixes.

use anyhow::Result;
use lightbulb::engine::Request;
use lightbulb::model::ParallelModelManager;
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== Prefix KV Caching Verification ===\n");

    // Path to a small model (adjust to your local path)
    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/TinyLlama-1.1B-Chat-v1.0".to_string());

    println!("Loading model from: {}", model_path);
    let start = Instant::now();
    let mut model = ParallelModelManager::load(
        &model_path,
        4,   // max_batch_size
        512, // context_length
        Some("f32"),
        None, // use default chunked prefill config
    )?;
    println!("Model loaded in {:.2}s\n", start.elapsed().as_secs_f64());

    // Shared system prompt (should be cached after first use)
    let system_prompt = "You are a helpful AI assistant. Please provide clear and concise answers.";

    // Test with 5 different questions but same system prompt
    let questions = vec![
        "What is the capital of France?",
        "Explain photosynthesis briefly.",
        "What year did World War 2 end?",
        "Name three programming languages.",
        "What is the speed of light?",
    ];

    println!("Testing prefix caching with repeated system prompt:\n");
    println!("System prompt: \"{}\"\n", system_prompt);

    let mut ttft_times = Vec::new();

    for (i, question) in questions.iter().enumerate() {
        let prompt = format!("{}\n\nUser: {}\n\nAssistant:", system_prompt, question);

        // `Request::new` was removed; it is a plain struct now.
        let request = Request {
            id: format!("test-{}", i),
            prompt: prompt.clone(),
            max_new_tokens: 50,
        };

        println!("Request {}: {}", i + 1, question);

        let start = Instant::now();
        let mut batch = vec![lightbulb::engine::RequestContext::new(request)];

        // Run one forward pass to get first token (TTFT measurement)
        let _results = model.forward_batch(&mut batch)?;
        let ttft = start.elapsed();
        ttft_times.push(ttft.as_secs_f64());

        println!("  TTFT: {:.3}s", ttft.as_secs_f64());

        // For cache testing, we don't need to generate full responses
        // The TTFT is what matters for measuring cache hit performance
    }

    // Print statistics
    println!("\n=== TTFT Statistics ===");
    println!("First request (cache miss): {:.3}s", ttft_times[0]);
    // `improvement` is bound OUTSIDE the `if`, because the verdict block near
    // the end of this function reads it. It used to be a `let` inside the
    // block, so those reads referred to nothing — the example did not compile.
    let improvement = if ttft_times.len() > 1 {
        let avg_subsequent = ttft_times[1..].iter().sum::<f64>() / (ttft_times.len() - 1) as f64;
        println!("Average subsequent (cache hits): {:.3}s", avg_subsequent);
        let speedup = ttft_times[0] / avg_subsequent;
        println!("Speedup from caching: {:.2}x", speedup);
        let improvement = (1.0 - avg_subsequent / ttft_times[0]) * 100.0;
        println!("TTFT improvement: {:.1}%", improvement);
        improvement
    } else {
        // One sample means no hit to compare against, so there is no measured
        // improvement to claim — not a 0% improvement, an absent measurement.
        // The verdict block treats this as "lower than expected", which is the
        // right reading of "we could not tell".
        0.0
    };

    // Print prefix cache statistics
    println!("\n=== Prefix Cache Statistics ===");
    let stats = model.prefix_cache_stats();
    println!("Cache hits: {}", stats.hits);
    println!("Cache misses: {}", stats.misses);
    println!(
        "Hit rate: {:.1}%",
        if stats.hits + stats.misses > 0 {
            (stats.hits as f64 / (stats.hits + stats.misses) as f64) * 100.0
        } else {
            0.0
        }
    );
    println!("Total tokens saved: {}", stats.total_saved_tokens);
    println!("Cache evictions: {}", stats.evictions);

    if stats.hits == 0 {
        println!("\n⚠️  WARNING: No cache hits detected!");
        println!("This could mean:");
        println!("  1. System prompt is too short (< min_prefix_len)");
        println!("  2. Prompts don't actually share a prefix");
        println!("  3. Cache configuration issue");
    } else if improvement > 15.0 {
        println!("\n✅ PREFIX CACHING IS WORKING!");
        println!(
            "   Achieved {:.1}% TTFT reduction on cache hits",
            improvement
        );
    } else {
        println!("\n⚠️  Cache is working but improvement is lower than expected");
        println!("   Expected: 15-50% improvement");
        println!("   Actual: {:.1}%", improvement);
    }

    Ok(())
}
