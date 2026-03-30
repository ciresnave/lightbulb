//! Demonstration of parallel batched inference
//!
//! This example shows how multiple requests are processed together
//! in a single forward pass, demonstrating the core batching capability.
//!
//! Run with:
//! ```bash
//! cargo run --example batch_inference_demo --release
//! ```

use anyhow::Result;
use lightbulb::engine::{Request, RequestContext, RequestState};
use lightbulb::model::ParallelModelManager;
use std::time::Instant;

fn main() -> Result<()> {
    println!("\n🔦 Lightbulb Parallel Batched Inference Demo\n");

    // Load model
    println!("📂 Loading llama-3b model...");
    let model_path = "../models/llama-3b";
    let mut model_manager = ParallelModelManager::load(
        model_path,
        4,           // max_batch_size
        512,         // context_length
        Some("f32"), // dtype
        None,        // chunked_prefill_config
    )?;
    println!("✓ Model loaded successfully\n");

    // Create multiple requests with different prompts
    let requests = vec![
        Request {
            id: "req-1".to_string(),
            prompt: "The capital of France is".to_string(),
            max_new_tokens: 5,
        },
        Request {
            id: "req-2".to_string(),
            prompt: "Artificial intelligence is".to_string(),
            max_new_tokens: 5,
        },
        Request {
            id: "req-3".to_string(),
            prompt: "The weather today is".to_string(),
            max_new_tokens: 5,
        },
    ];

    println!("📝 Processing {} requests in parallel:\n", requests.len());
    for req in &requests {
        println!("  • [{}] \"{}\"", req.id, req.prompt);
    }
    println!();

    // Create request contexts
    let mut batch: Vec<RequestContext> = requests.into_iter().map(RequestContext::new).collect();

    // Process batch through multiple generation steps
    let total_start = Instant::now();
    let max_steps = 10; // Generate up to 10 tokens

    for step in 0..max_steps {
        // Check if all requests are completed
        if batch.iter().all(|ctx| ctx.state == RequestState::Completed) {
            break;
        }

        let step_start = Instant::now();

        // Single batched forward pass for all active requests
        let results = model_manager.forward_batch(&mut batch)?;

        let step_ms = step_start.elapsed().as_millis();

        // Count active requests in this batch
        let active_count = batch
            .iter()
            .filter(|ctx| ctx.state != RequestState::Completed)
            .count();

        println!(
            "Step {}: Processed {} requests in {}ms",
            step + 1,
            active_count,
            step_ms
        );

        // Decode and print tokens for each request
        for (i, ctx) in batch.iter().enumerate() {
            if let Some(token) = results[i] {
                if let Ok(text) = model_manager.decode(&[token], false) {
                    print!("  [{}] +{:?} ", ctx.request.id, text);
                }
            }
        }
        println!();
    }

    let total_ms = total_start.elapsed().as_millis();

    // Print final results
    println!("\n📊 Results:\n");
    for ctx in &batch {
        let generated_text = model_manager.decode(&ctx.generated_tokens, false)?;
        println!(
            "  [{}] \"{}{}\"",
            ctx.request.id, ctx.request.prompt, generated_text
        );
    }

    // Print statistics
    println!("\n⏱️  Performance:");
    println!("  Total time: {}ms", total_ms);
    println!(
        "  Total tokens: {}",
        batch
            .iter()
            .map(|ctx| ctx.generated_tokens.len())
            .sum::<usize>()
    );

    let stats = model_manager.stats();
    println!("\n📈 Batch Statistics:");
    println!("  Total batches: {}", stats.total_batches);
    println!("  Prefill requests: {}", stats.prefill_requests);
    println!("  Decode requests: {}", stats.decode_requests);
    println!(
        "  Decode batch opportunities: {}",
        stats.decode_batch_opportunities
    );
    println!("  Max concurrent decodes: {}", stats.max_concurrent_decodes);
    println!(
        "  Avg forward time: {:.2}ms",
        stats.total_forward_time_ms / stats.total_batches as f64
    );

    println!("\n✨ Parallel batched inference complete!\n");

    Ok(())
}
