//! Hardware-aware initialization example
//!
//! Demonstrates automatic configuration of inference engine based on
//! detected hardware capabilities.
//!
//! This example shows:
//! 1. Hardware detection (CPU, RAM, GPU)
//! 2. Automatic batch size calculation
//! 3. Optimal chunk size selection
//! 4. Memory utilization analysis
//! 5. Integration with SlotPool and SlotPoolMonitor
//!
//! Run: `cargo run --example hardware_aware_init --release`

use lightbulb::engine::slot_monitor::{AdjustmentConfig, SlotPoolMonitor};
use lightbulb::engine::slot_pool::SlotPool;
use lightbulb::hardware::batch_sizing::ModelMemoryProfile;
use lightbulb::init::SystemConfig;

fn main() -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("   HARDWARE-AWARE INFERENCE ENGINE INITIALIZATION");
    println!("═══════════════════════════════════════════════════════════\n");

    // Define model characteristics (example: 7B parameter Llama-style model)
    let model_profile = ModelMemoryProfile {
        weights_bytes: 6 * 1024 * 1024 * 1024, // 6GB (f16)
        num_layers: 32,
        hidden_size: 4096,
        num_kv_heads: 32,
        context_window: 512,
    };

    println!("📋 Model Profile:");
    println!("  Parameters: ~7B");
    println!(
        "  Weights: {:.1} GB (f16)",
        model_profile.weights_bytes as f64 / 1e9
    );
    println!("  Layers: {}", model_profile.num_layers);
    println!("  Hidden size: {}", model_profile.hidden_size);
    println!(
        "  Context window: {} tokens\n",
        model_profile.context_window
    );

    // Auto-detect hardware and calculate optimal configuration
    println!("🔍 Detecting hardware and calculating optimal configuration...\n");
    let config = SystemConfig::auto_detect(model_profile.clone(), 2)?;

    // Print detailed configuration summary
    config.print_summary();

    // Initialize slot pool with calculated size
    println!("\n🚀 Initializing inference engine...\n");
    let mut slot_pool = SlotPool::new(config.slot_pool_size);
    println!(
        "✓ SlotPool initialized with {} slots",
        config.slot_pool_size
    );

    // Initialize monitoring if enabled
    if config.enable_monitoring {
        let monitor_config = AdjustmentConfig {
            target_utilization: config.memory_utilization_target,
            ..AdjustmentConfig::default()
        };

        let _monitor =
            SlotPoolMonitor::with_config(model_profile.clone(), config.dtype_bytes, monitor_config);
        println!("✓ SlotPoolMonitor initialized");
        println!(
            "  - Target utilization: {:.0}%",
            config.memory_utilization_target * 100.0
        );
        println!("  - Runtime adjustment: enabled");
    }

    println!("\n📈 Chunk Size Configuration:");
    println!("  Chunk size: {} tokens", config.chunk_size);
    println!(
        "  Strategy: {}",
        if config.hardware.gpu.is_some() {
            "GPU-optimized (larger chunks for throughput)"
        } else {
            "CPU-optimized (conservative chunks from benchmarks)"
        }
    );

    // Simulate usage scenarios
    println!("\n🔬 Simulation: Load Scenarios\n");

    // Scenario 1: Light load
    println!("─── Scenario 1: Light Load (25% capacity) ───");
    let light_load_slots = config.slot_pool_size / 4;
    let stats_light = config.memory_stats();
    let light_memory = stats_light.model_weights_gb
        + (stats_light.kv_cache_per_slot_mb * light_load_slots as f64 / 1000.0);
    println!("  Active slots: {}", light_load_slots);
    println!("  Memory usage: {:.2} GB", light_memory);
    println!(
        "  Utilization: {:.1}%",
        (light_memory / stats_light.available_memory_gb) * 100.0
    );

    // Scenario 2: Nominal load
    println!("\n─── Scenario 2: Nominal Load (70% capacity) ───");
    let nominal_load_slots = (config.slot_pool_size as f64 * 0.7) as usize;
    let nominal_memory = stats_light.model_weights_gb
        + (stats_light.kv_cache_per_slot_mb * nominal_load_slots as f64 / 1000.0);
    println!("  Active slots: {}", nominal_load_slots);
    println!("  Memory usage: {:.2} GB", nominal_memory);
    println!(
        "  Utilization: {:.1}%",
        (nominal_memory / stats_light.available_memory_gb) * 100.0
    );

    // Scenario 3: Peak load
    println!("\n─── Scenario 3: Peak Load (100% capacity) ───");
    let stats = config.memory_stats();
    println!("  Active slots: {}", config.slot_pool_size);
    println!("  Memory usage: {:.2} GB", stats.total_memory_gb);
    println!("  Utilization: {:.1}%", stats.utilization_percent);

    // Performance projections
    println!("\n📊 Performance Projections:");

    let prefill_tokens_per_chunk = config.chunk_size;
    let decode_latency_ms = 50.0; // Approximate per-token decode latency

    println!(
        "  Prefill throughput: {} tokens/chunk",
        prefill_tokens_per_chunk
    );
    println!(
        "  Decode latency: ~{:.0} ms/token (estimated)",
        decode_latency_ms
    );
    println!("  Max concurrent: {} requests", config.slot_pool_size);

    let throughput_estimate = config.slot_pool_size as f64 / (decode_latency_ms / 1000.0);
    println!(
        "  Estimated throughput: {:.0} tokens/sec (decode phase)",
        throughput_estimate
    );

    // Comparison with naive configuration
    println!("\n📐 Comparison with Naive Configuration:");
    println!("  Naive (fixed batch=8):  8 slots");
    println!("  Hardware-aware:         {} slots", config.slot_pool_size);
    let improvement = ((config.slot_pool_size as f64 / 8.0) - 1.0) * 100.0;
    if improvement > 0.0 {
        println!("  Improvement:            +{:.0}% capacity", improvement);
        println!("\n✅ Hardware-aware configuration better utilizes available resources!");
    } else if improvement < 0.0 {
        println!(
            "  Adjustment:             {:.0}% (preventing OOM)",
            improvement
        );
        println!("\n✅ Hardware-aware configuration prevents memory issues!");
    } else {
        println!("  Match:                  Same as naive");
    }

    // Tips and recommendations
    println!("\n💡 Tips:");
    if config.hardware.gpu.is_some() {
        println!("  • GPU detected: Consider enabling FlashAttention for 2-4x speedup");
        println!(
            "  • Larger chunk sizes ({}+) work well on GPU",
            config.chunk_size
        );
    } else {
        println!(
            "  • CPU mode: Chunk size optimized for L3 cache ({})",
            config.chunk_size
        );
        println!("  • Consider quantization (Q4, Q8) for 2-4x memory savings");
    }

    if stats.utilization_percent < 60.0 {
        let headroom_slots = ((stats.available_memory_gb - stats.total_memory_gb)
            / (stats.kv_cache_per_slot_mb / 1000.0)) as usize;
        println!(
            "  • Memory headroom available: ~{} additional slots possible",
            headroom_slots
        );
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("✅ Initialization complete! Ready for inference.");
    println!("═══════════════════════════════════════════════════════════\n");

    Ok(())
}
