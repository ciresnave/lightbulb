//! Test multi-request parallel generation with separate prompts

use anyhow::Result;
use lightbulb::engine::{Request, RequestContext, RequestState};
use lightbulb::model::{ChunkedPrefillConfig, ParallelModelManager};

fn main() -> Result<()> {
    println!("\n🧪 Testing Multi-Request Parallel Generation\n");

    let model_path = "models/llama-3b";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Model not found at {}", model_path);
        return Ok(());
    }

    let config = ChunkedPrefillConfig {
        chunk_size: 512,
        ..Default::default()
    };

    let mut model = ParallelModelManager::load(model_path, 4, 512, Some("f32"), Some(config))?;
    println!("✓ Model loaded\n");

    // Create multiple requests with different prompts
    let requests = vec![
        Request {
            id: "math-1".to_string(),
            prompt: "What is 5 + 3? Answer: ".to_string(),
            max_new_tokens: 8,
        },
        Request {
            id: "greeting-1".to_string(),
            prompt: "Hello! My name is ".to_string(),
            max_new_tokens: 8,
        },
        Request {
            id: "color-1".to_string(),
            prompt: "The sky is ".to_string(),
            max_new_tokens: 8,
        },
    ];

    println!("🔹 Requests:");
    for req in &requests {
        println!("  {} → \"{}\"", req.id, req.prompt);
    }
    println!();

    // Create batch contexts
    let mut batch: Vec<RequestContext> = requests.into_iter().map(RequestContext::new).collect();

    // Generate tokens
    println!("🔸 Generating...\n");
    for step in 0..20 {
        let active = batch
            .iter()
            .filter(|c| c.state != RequestState::Completed)
            .count();
        if active == 0 {
            println!("✓ All requests completed at step {}", step);
            break;
        }

        model.forward_batch(&mut batch)?;

        if step < 3 {
            println!("Step {}:", step + 1);
            for ctx in &batch {
                if ctx.state != RequestState::Completed {
                    println!(
                        "  {} → tokens_generated={}, position={}",
                        ctx.request.id, ctx.tokens_generated, ctx.position
                    );
                }
            }
        }
    }

    // Print results
    println!("\n📝 Results:\n");
    for ctx in &batch {
        let tokens_u32: Vec<u32> = ctx.generated_tokens.clone();
        let output = model
            .tokenizer()
            .decode(&tokens_u32, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;
        println!(
            "  {} → \"{}{}\"",
            ctx.request.id, ctx.request.prompt, output
        );
    }

    println!("\n✨ Test complete");
    Ok(())
}
