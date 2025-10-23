//! Test TinyLlama with simple sequential generation
//! Compare quality with Phi-3 to validate if TinyLlama is the problem

use anyhow::Result;
use candle_core::{Device, IndexOp};
use candle_transformers::models::quantized_llama::ModelWeights as LlamaModelWeights;
use lightbulb::gguf;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

    println!("\n🧪 Testing TinyLlama Sequential Generation");
    println!("Model: {}\n", model_path);

    // Load GGUF and extract tokenizer
    let content = gguf::Content::read(model_path)?;
    let tokenizer = content.extract_tokenizer()?;
    println!("✓ Tokenizer extracted\n");

    // Load model using Candle
    let device = Device::Cpu;
    let mut file = std::fs::File::open(model_path)?;
    let candle_content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    let mut model = LlamaModelWeights::from_gguf(candle_content, &mut file, &device)?;
    println!("✓ Model loaded\n");

    // Same prompt as Phi-3
    let prompt = "The capital of France is";
    println!("Prompt: \"{}\"\n", prompt);

    // Tokenize
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
    let prompt_tokens = tokens.get_ids();

    println!("Prompt tokens: {:?}", prompt_tokens);

    // Prefill
    let input = candle_core::Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;

    println!("Logits shape: {:?}", logits.dims());

    // Extract last token logits - shape should be [batch, seq, vocab]
    let logits = if logits.dims().len() == 3 {
        logits.i((0, prompt_tokens.len() - 1))?
    } else if logits.dims().len() == 2 {
        logits.i(0)?
    } else {
        logits
    };

    println!("After extraction: {:?}", logits.dims());

    let mut next_token = logits.argmax(0)?.to_scalar::<u32>()?;
    let mut generated = vec![next_token];

    println!("First token: {}", next_token);

    // Generate 20 tokens
    for pos in 1..20 {
        let input = candle_core::Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens.len() + pos - 1)?;

        // Decode output might be [batch, vocab] or [batch, 1, vocab]
        let logits = if logits.dims().len() == 3 {
            logits.i((0, 0))?
        } else {
            logits.i(0)?
        };

        next_token = logits.argmax(0)?.to_scalar::<u32>()?;
        generated.push(next_token);
        println!("Token {}: {}", pos + 1, next_token);
    }

    let output = tokenizer
        .decode(&generated, true)
        .map_err(|e| anyhow::anyhow!("Decode error: {}", e))?;

    println!("\n📝 Output: \"{}\"", output);
    println!("\n🎯 Compare with Phi-3 output:");
    println!("   \"Paris. [response]: The capital of France is Paris. It is not only...\"");
    println!("\n💡 If TinyLlama output is poor, that confirms model quality is the issue!");

    Ok(())
}
