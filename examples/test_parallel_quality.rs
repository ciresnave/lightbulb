//! Simple test to verify parallel batching generates coherent text

use anyhow::Result;
use lightbulb::engine::{Request, RequestContext};
use lightbulb::model::{ChunkedPrefillConfig, ParallelModelManager};

fn main() -> Result<()> {
    println!("\n🧪 Testing Parallel Batching Generation Quality\n");

    let model_path = "models/llama-3b";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Model not found at {}", model_path);
        return Ok(());
    }

    // Use default max chunk size - dynamic chunking will calculate optimal size per batch
    let config = ChunkedPrefillConfig {
        chunk_size: 512, // Max chunk size, will be reduced dynamically based on batch
        ..Default::default()
    };

    let mut model = ParallelModelManager::load(model_path, 4, 512, Some("f32"), Some(config))?;
    println!("✓ Model loaded (with dynamic chunk sizing)\n");

    // Single simple request to test quality
    let request = Request {
        id: "test-1".to_string(),
        prompt: "2 + 2 is ".to_string(),
        max_new_tokens: 10,
    };

    println!("Prompt: \"{}\"", request.prompt);

    let ctx = RequestContext::new(request);
    let mut batch = vec![ctx];

    // Generate tokens
    for step in 0..15 {
        // Check for active requests (Pending or Decoding, not Completed)
        let active = batch
            .iter()
            .filter(|c| c.state != lightbulb::engine::RequestState::Completed)
            .count();
        if active == 0 {
            break;
        }

        model.forward_batch(&mut batch)?;

        if step < 3 {
            println!("Step {}: {:?}", step + 1, batch[0].generated_tokens);
        }
    }

    // Print result
    let tokens_u32: Vec<u32> = batch[0].generated_tokens.clone();
    let output = model
        .tokenizer()
        .decode(&tokens_u32, true)
        .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;
    println!("\n📝 Generated: \"{}{}\"", batch[0].request.prompt, output);
    println!("\n✨ Test complete");

    Ok(())
}
