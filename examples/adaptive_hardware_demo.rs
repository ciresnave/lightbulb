//! Hardware-Adaptive Inference Demo
//!
//! Demonstrates M1.5: Hardware Adaptivity
//! - Automatic hardware detection
//! - Dynamic batch size calculation
//! - Model recommendations based on available resources
//! - Runtime performance monitoring
//!
//! Usage:
//! ```
//! cargo run --example adaptive_hardware_demo -- <model_dir>
//! ```

use anyhow::Result;
use lightbulb::hardware::{
    batch_sizing::{BatchSizeConfig, ModelMemoryProfile},
    model_selection::recommend_model,
    HardwareProfile,
};
use lightbulb::model::{ChunkedPrefillConfig, ParallelModelManager};
use std::env;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    println!("=== Hardware-Adaptive Inference Demo (M1.5) ===\n");

    // === STEP 1: Detect Hardware ===
    println!("1. HARDWARE DETECTION");
    println!("   Scanning system capabilities...\n");

    let profile = HardwareProfile::detect()?;
    println!("{}", profile.summary());
    println!("   ML Suitability Score: {:.1}/10.0", profile.ml_score);
    println!("   Recommended Backend: {:?}\n", profile.recommended_backend());

    // === STEP 2: Model Recommendation ===
    println!("2. MODEL RECOMMENDATION");
    println!("   Analyzing optimal model for your hardware...\n");

    let recommendation = recommend_model(&profile)?;
    println!("   Recommended Model: {}", recommendation.model_name);
    println!("   Data Type: {:?}", recommendation.dtype);
    println!("   Backend: {:?}", recommendation.backend);
    println!("   Expected Throughput: {:.2} tokens/sec", recommendation.estimated_throughput);
    println!("   Confidence: {:.1}%\n", recommendation.confidence * 100.0);

    // === STEP 3: Batch Size Calculation ===
    println!("3. DYNAMIC BATCH SIZING");
    println!("   Computing optimal batch size...\n");

    // Example model profile (Llama 3B)
    let model_profile = ModelMemoryProfile {
        weights_bytes: 6 * 1024 * 1024 * 1024, // 6GB
        num_layers: 32,
        hidden_size: 4096,
        num_kv_heads: 32,
        context_window: 512,
    };

    let batch_config = BatchSizeConfig::default();
    let optimal_batch = lightbulb::hardware::batch_sizing::calculate_optimal_batch_size(
        &profile,
        &model_profile,
        2,
        Some(batch_config),
    )?;

    println!("   Optimal Batch Size: {}", optimal_batch);
    println!("   Memory Utilization Target: 70%");
    println!("   Safety Margin: 1.2x\n");

    // === STEP 4: Load Model (if path provided) ===
    if args.len() > 1 {
        let model_dir = &args[1];
        println!("4. LOADING MODEL");
        println!("   Using adaptive configuration...\n");

        let start = Instant::now();
        let _manager = ParallelModelManager::load_adaptive(
            model_dir,
            512,            // context_length
            Some("f16"),    // dtype
            Some(ChunkedPrefillConfig::default()),
        )?;
        let load_time = start.elapsed();

        println!("   ✓ Model loaded in {:.2}s", load_time.as_secs_f64());
        println!("   ✓ Batch size automatically configured");
        println!("   ✓ Ready for inference\n");

        println!("5. NEXT STEPS");
        println!("   • Use manager.generate_batch() for parallel inference");
        println!("   • Monitor with manager.stats() for performance metrics");
        println!("   • Batch size adapts automatically to system load\n");
    } else {
        println!("4. SKIPPING MODEL LOAD");
        println!("   (Provide model_dir as argument to load model)\n");
        println!("   Usage: cargo run --example adaptive_hardware_demo -- <model_dir>\n");
    }

    // === STEP 5: Alternative Models ===
    println!("ALTERNATIVE MODELS FOR YOUR HARDWARE:");
    let viable_models = lightbulb::hardware::model_selection::list_viable_models(&profile)?;

    for (i, rec) in viable_models.iter().take(5).enumerate() {
        println!(
            "   {}. {} ({:?}) - {:.1} tok/s",
            i + 1,
            rec.model_name,
            rec.dtype,
            rec.estimated_throughput
        );
    }

    println!("\n=== Demo Complete ===");
    println!("Hardware adaptivity enables optimal performance across diverse systems.");
    println!("M1.5 automatically configures batch sizes and recommends suitable models.\n");

    Ok(())
}
