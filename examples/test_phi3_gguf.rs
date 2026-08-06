//! Test Phi-3 GGUF model - baseline test with integrated tokenizer extraction
//!
//! Uses lightning::gguf for fast memory-mapped loading with tokenizer extraction

use anyhow::Result;
use candlelight::core::{Device, IndexOp, Tensor};
use candlelight::transformers::models::quantized_phi3::ModelWeights;
use lightbulb::gguf;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let model_path = args
        .iter()
        .position(|x| x == "--model")
        .and_then(|i| args.get(i + 1))
        .ok_or_else(|| anyhow::anyhow!("Usage: --model <gguf-path>"))?;

    println!("\n🔦 Phi-3 GGUF Baseline Test");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("Model: {}\n", model_path);

    // Load model using lightning::gguf for fast memory-mapped access
    let load_start = Instant::now();
    let content = gguf::Content::read(model_path)?;
    let load_time = load_start.elapsed();

    println!(
        "⚡ GGUF loaded via memory mapping in {:.2}ms",
        load_time.as_secs_f64() * 1000.0
    );

    println!("\n📊 GGUF Metadata:");
    println!(
        "   Architecture: {:?}",
        content.metadata().get("general.architecture")
    );
    println!("   Name: {:?}", content.metadata().get("general.name"));

    // Show tokenizer metadata
    println!("\n🔍 Tokenizer metadata in GGUF:");
    let tokenizer_keys: Vec<_> = content
        .metadata()
        .keys()
        .filter(|k| k.starts_with("tokenizer"))
        .collect();

    if tokenizer_keys.is_empty() {
        println!("   ⚠ No tokenizer metadata found in GGUF!");
    } else {
        for key in tokenizer_keys.iter().take(10) {
            println!("   - {}", key);
        }
        if tokenizer_keys.len() > 10 {
            println!("   ... and {} more keys", tokenizer_keys.len() - 10);
        }
    }

    // Extract tokenizer from GGUF metadata
    println!("\n📝 Extracting tokenizer from GGUF...");
    let tokenizer_start = Instant::now();
    let tokenizer = content.extract_tokenizer()?;
    let tokenizer_time = tokenizer_start.elapsed();
    println!(
        "✓ Tokenizer extracted in {:.2}ms",
        tokenizer_time.as_secs_f64() * 1000.0
    );

    // TODO: Load model weights from our mmap content
    // For now, we'll use Candle's loader but we have the tokenizer ready!
    let device = Device::Cpu;
    let mut file = std::fs::File::open(model_path)?;
    let candle_content = candlelight::core::quantized::gguf_file::Content::read(&mut file)?;
    let mut model = ModelWeights::from_gguf(false, candle_content, &mut file, &device)?;
    println!("✓ Model loaded (using Candle for now)\n");

    // Test generation
    let test_prompts = vec![
        "The capital of France is",
        "Artificial intelligence is",
        "Once upon a time",
    ];

    for prompt in test_prompts {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📝 Prompt: \"{}\"", prompt);
        println!("───────────────────────────────────────────");

        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
        let prompt_tokens = tokens.get_ids();

        print!("🤖 Output: \"");
        std::io::Write::flush(&mut std::io::stdout())?;

        // Prefill
        let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;

        // Debug: Check logits shape
        // DEBUG output removed

        // Extract last token logits - Phi-3 might return [batch, seq, vocab] or [batch, vocab]
        let logits = if logits.dims().len() == 3 {
            logits.i((0, prompt_tokens.len() - 1))?
        } else if logits.dims().len() == 2 {
            logits.i(0)?
        } else {
            return Err(anyhow::anyhow!(
                "Unexpected logits shape: {:?}",
                logits.dims()
            ));
        };

        // DEBUG output removed
        let mut next_token = logits.argmax(0)?.to_scalar::<u32>()?;

        if let Ok(text) = tokenizer.decode(&[next_token], false) {
            print!("{}", text);
            std::io::Write::flush(&mut std::io::stdout())?;
        }

        // Decode (generate up to 30 tokens)
        for idx in 0..29 {
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, prompt_tokens.len() + idx)?;

            // Extract logits - handle both [batch, vocab] and [batch, seq, vocab]
            let logits = if logits.dims().len() == 3 {
                logits.i((0, 0))?
            } else if logits.dims().len() == 2 {
                logits.i(0)?
            } else {
                return Err(anyhow::anyhow!(
                    "Unexpected decode logits shape: {:?}",
                    logits.dims()
                ));
            };

            next_token = logits.argmax(0)?.to_scalar::<u32>()?;

            if let Ok(text) = tokenizer.decode(&[next_token], false) {
                print!("{}", text);
                std::io::Write::flush(&mut std::io::stdout())?;
            }

            // Check for EOS
            if next_token == 32000 || next_token == 2 || next_token == 32007 {
                break;
            }
        }

        println!("\"\n");
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✨ Baseline test complete!\n");
    println!("💡 If this generates good English, our custom");
    println!("   BatchedTransformer code has issues.");
    println!("   If not, the model is just too small.");

    Ok(())
}
