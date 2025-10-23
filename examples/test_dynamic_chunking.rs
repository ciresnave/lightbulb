//! Test dynamic chunking with multiple requests of varying lengths
//!
//! This validates:
//! 1. Dynamic chunk sizing adapts to batch composition
//! 2. Generation quality is good across different prompts
//! 3. Padding efficiency improves with adaptive chunking

use anyhow::Result;
use lightbulb::engine::{Request, RequestContext};
use lightbulb::model::{ChunkedPrefillConfig, ParallelModelManager};

fn main() -> Result<()> {
    println!("\n🧪 Testing Dynamic Chunking with Variable-Length Requests\n");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let model_path = "models/llama-3b";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Model not found at {}", model_path);
        return Ok(());
    }

    let config = ChunkedPrefillConfig {
        chunk_size: 512, // Max chunk size, will be reduced dynamically
        ..Default::default()
    };

    let mut model = ParallelModelManager::load(model_path, 4, 512, Some("f32"), Some(config))?;
    println!("✓ Model loaded with dynamic chunk sizing\n");

    // Create requests with different lengths and topics
    let test_cases = vec![
        (
            "Very short",
            "Hello",
            "Simple greeting to test minimal padding",
        ),
        (
            "Short",
            "The capital of France is",
            "Geographic fact (original test case)",
        ),
        (
            "Medium",
            "Once upon a time, there was a brave knight who",
            "Story beginning to test creative generation",
        ),
        (
            "Longer",
            "Artificial intelligence and machine learning are transforming the way we think about computing and automation in",
            "Technical topic with longer context",
        ),
    ];

    println!("📝 Test Cases:\n");
    for (i, (label, prompt, description)) in test_cases.iter().enumerate() {
        println!(
            "  {}. {} ({} chars): \"{}\"",
            i + 1,
            label,
            prompt.len(),
            prompt
        );
        println!("     → {}\n", description);
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Test each case individually first
    println!("🔬 Phase 1: Individual Requests (testing quality)\n");
    for (label, prompt, _) in &test_cases {
        println!("Testing: {} - \"{}\"", label, prompt);

        let request = Request {
            id: format!("test-{}", label),
            prompt: prompt.to_string(),
            max_new_tokens: 15,
        };

        let mut batch = vec![RequestContext::new(request)];

        // Generate tokens
        for _ in 0..20 {
            let active = batch.iter().filter(|c| c.should_continue()).count();
            if active == 0 {
                break;
            }
            model.forward_batch(&mut batch)?;
        }

        // Decode result
        let tokens_u32: Vec<u32> = batch[0].generated_tokens.clone();
        let output = model
            .tokenizer()
            .decode(&tokens_u32, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;

        println!("  → Output: \"{}\"\n", output);
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Test with mixed batch (different lengths processed together)
    println!("🔬 Phase 2: Parallel Batch (testing dynamic chunking)\n");
    println!("Processing all requests together to see chunk size adaptation...\n");

    let mut mixed_batch: Vec<RequestContext> = test_cases
        .iter()
        .map(|(label, prompt, _)| {
            RequestContext::new(Request {
                id: format!("batch-{}", label),
                prompt: prompt.to_string(),
                max_new_tokens: 12,
            })
        })
        .collect();

    // Generate tokens for the batch
    for step in 0..20 {
        let active = mixed_batch.iter().filter(|c| c.should_continue()).count();
        if active == 0 {
            break;
        }

        println!("Step {}: {} active requests", step + 1, active);
        model.forward_batch(&mut mixed_batch)?;
    }

    println!("\n📊 Results from Parallel Batch:\n");
    for (i, ctx) in mixed_batch.iter().enumerate() {
        let tokens_u32: Vec<u32> = ctx.generated_tokens.clone();
        let output = model
            .tokenizer()
            .decode(&tokens_u32, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;

        println!(
            "  {}. {} ({} tokens generated)",
            i + 1,
            test_cases[i].0,
            tokens_u32.len()
        );
        println!("     Prompt: \"{}\"", test_cases[i].1);
        println!("     Output: \"{}\"\n", output);
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n✨ Test Complete!\n");
    println!("Key Observations:");
    println!("  • Watch for 'Dynamic chunk_size' debug output");
    println!("  • Chunk size should adapt to longest request in each batch");
    println!("  • All outputs should be coherent English (not gibberish)");
    println!("  • Parallel batch should handle mixed lengths efficiently\n");

    Ok(())
}
