//! Test AWQ loader with Qwen3-32B-AWQ model

use anyhow::Result;
use std::path::Path;

fn main() -> Result<()> {
    let model_dir =
        Path::new("c:/Users/cires/OneDrive/Documents/projects/lightbulb/models/Qwen3-32B-AWQ");

    println!("Testing AWQ metadata loading...\n");

    // Load AWQ metadata
    let (config, awq_config) = lightbulb::loaders::load_awq_metadata(model_dir)?;

    println!("\n✓ Successfully loaded AWQ metadata!");
    println!("\nModel configuration:");
    println!("  - Architecture: {}", config["architectures"][0]);
    println!("  - Model type: {}", config["model_type"]);
    println!("  - Hidden size: {}", config["hidden_size"]);
    println!("  - Num layers: {}", config["num_hidden_layers"]);
    println!("  - Num heads: {}", config["num_attention_heads"]);
    println!("  - Num KV heads: {}", config["num_key_value_heads"]);
    println!("  - Vocab size: {}", config["vocab_size"]);
    println!(
        "  - Max position embeddings: {}",
        config["max_position_embeddings"]
    );

    println!("\nAWQ quantization:");
    println!("  - Bits: {}", awq_config.bits);
    println!("  - Group size: {}", awq_config.group_size);
    println!("  - Version: {}", awq_config.version);
    println!("  - Zero point: {}", awq_config.zero_point);

    println!("\n🎉 AWQ metadata test PASSED!");

    Ok(())
}
