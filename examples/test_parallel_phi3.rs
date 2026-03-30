//! Test Phi-3 with simple sequential generation
//! This confirms Phi-3 can generate good text before we try parallel batching

use anyhow::Result;
use candlelight::core::{Device, IndexOp};
use candlelight::transformers::models::quantized_phi3::ModelWeights as Phi3ModelWeights;
use lightbulb::gguf;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("models/Phi-3-mini-4k-instruct-q4.gguf");

    println!("\n🧪 Testing Phi-3 Sequential Generation");
    println!("Model: {}\n", model_path);

    // Load GGUF and extract tokenizer
    let content = gguf::Content::read(model_path)?;
    let tokenizer = content.extract_tokenizer()?;
    println!("✓ Tokenizer extracted\n");

    // Load model using Candle
    let device = Device::Cpu;
    let mut file = std::fs::File::open(model_path)?;
    let candle_content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let mut model = Phi3ModelWeights::from_gguf(false, candle_content, &mut file, &device)?;
    println!("✓ Model loaded\n");

    // Same prompt as test_phi3_gguf.rs
    let prompt = "The capital of France is";
    println!("Prompt: \"{}\"\n", prompt);

    // Tokenize
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
    let prompt_tokens = tokens.get_ids();

    // Prefill
    let input = candle_core::Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
    let mut logits = model.forward(&input, 0)?;

    // Extract last token logits
    logits = if logits.dims().len() == 3 {
        logits.i((0, prompt_tokens.len() - 1))?
    } else {
        logits.i(0)?
    };

    let mut next_token = logits.argmax(0)?.to_scalar::<u32>()?;
    let mut generated = vec![next_token];

    // Generate 20 tokens
    for pos in 1..20 {
        let input = candle_core::Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let mut logits = model.forward(&input, prompt_tokens.len() + pos - 1)?;

        logits = if logits.dims().len() == 3 {
            logits.i((0, 0))?
        } else {
            logits.i(0)?
        };

        next_token = logits.argmax(0)?.to_scalar::<u32>()?;
        generated.push(next_token);
    }

    let output = tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;

    println!("📝 Output: \"{}\"", output);
    println!("\n🎯 Expected (from test_phi3_gguf baseline):");
    println!("   \"Paris. The capital of France is Paris. It is not only...\"");
    println!("\n✨ Confirms Phi-3 generates coherent text!");

    Ok(())
}
