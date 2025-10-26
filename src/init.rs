//! Hardware-aware system initialization
//!
//! This module provides automatic configuration of batch size, chunk size,
//! and slot pool sizing based on detected hardware capabilities.
//!
//! # Usage
//!
//! ```ignore
//! use lightbulb::init::{SystemConfig, initialize_inference_engine};
//!
//! // Auto-detect hardware and configure system
//! let config = SystemConfig::auto_detect(model_profile)?;
//!
//! // Or customize
//! let config = SystemConfig {
//!     slot_pool_size: 64,
//!     chunk_size: 256,
//!     enable_monitoring: true,
//!     ..SystemConfig::auto_detect(model_profile)?
//! };
//!
//! println!("Optimized for: {} slots, {} chunk size",
//!          config.slot_pool_size, config.chunk_size);
//! ```

use crate::hardware::{HardwareProfile, batch_sizing::*};
use anyhow::{Context, Result};

/// System configuration for inference engine
#[derive(Debug, Clone)]
pub struct SystemConfig {
    /// Number of concurrent request slots (batch size)
    pub slot_pool_size: usize,

    /// Chunk size for prefill operations (tokens per chunk)
    pub chunk_size: usize,

    /// Enable runtime slot pool monitoring and adjustment
    pub enable_monitoring: bool,

    /// Memory utilization target (0.0-1.0)
    pub memory_utilization_target: f64,

    /// Hardware profile used for configuration
    pub hardware: HardwareProfile,

    /// Model memory profile
    pub model: ModelMemoryProfile,

    /// Bytes per tensor element (2 for f16, 4 for f32)
    pub dtype_bytes: usize,
}

impl SystemConfig {
    /// Auto-detect hardware and calculate optimal configuration
    ///
    /// # Arguments
    ///
    /// * `model` - Model memory profile (layers, hidden size, context window)
    /// * `dtype_bytes` - Bytes per tensor element (2 for f16, 4 for f32)
    ///
    /// # Returns
    ///
    /// Optimized configuration based on detected hardware
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = ModelMemoryProfile {
    ///     weights_bytes: 6 * 1024 * 1024 * 1024, // 6GB model
    ///     num_layers: 32,
    ///     hidden_size: 4096,
    ///     num_kv_heads: 32,
    ///     context_window: 512,
    /// };
    ///
    /// let config = SystemConfig::auto_detect(model, 2)?;
    /// println!("Auto-configured: {} slots", config.slot_pool_size);
    /// ```
    pub fn auto_detect(model: ModelMemoryProfile, dtype_bytes: usize) -> Result<Self> {
        // Detect hardware capabilities
        let hardware =
            HardwareProfile::detect().context("Failed to detect hardware capabilities")?;

        println!("🔍 Detected hardware:");
        println!(
            "  CPU: {} physical cores, {} logical cores",
            hardware.cpu.physical_cores, hardware.cpu.logical_cores
        );
        println!(
            "  RAM: {:.2} GB available",
            hardware.memory.available_bytes as f64 / 1e9
        );

        if let Some(gpu) = &hardware.gpu {
            println!("  GPU: {}", gpu.name);
            println!("  VRAM: {:.2} GB", gpu.vram_bytes as f64 / 1e9);
        } else {
            println!("  GPU: None (CPU mode)");
        }

        // Calculate optimal slot pool size using batch_sizing formulas
        let batch_config = BatchSizeConfig::default();
        let slot_pool_size =
            calculate_optimal_batch_size(&hardware, &model, dtype_bytes, Some(batch_config))
                .context("Failed to calculate optimal batch size")?;

        // Determine optimal chunk size based on device
        let chunk_size = Self::calculate_chunk_size(&hardware);

        println!("\n✅ Auto-configured:");
        println!("  Slot pool size: {} concurrent requests", slot_pool_size);
        println!("  Chunk size: {} tokens per prefill chunk", chunk_size);
        println!("  Monitoring: enabled");

        Ok(Self {
            slot_pool_size,
            chunk_size,
            enable_monitoring: true,
            memory_utilization_target: 0.7,
            hardware,
            model,
            dtype_bytes,
        })
    }

    /// Calculate optimal chunk size based on hardware
    ///
    /// Chunk size affects prefill memory usage and throughput.
    /// Larger chunks = faster prefill but higher memory spikes.
    ///
    /// # Strategy
    ///
    /// - **CPU**: 256 tokens (from benchmark results)
    /// - **GPU (mobile/integrated)**: 512 tokens
    /// - **GPU (discrete)**: 1024 tokens
    fn calculate_chunk_size(hardware: &HardwareProfile) -> usize {
        if let Some(gpu) = &hardware.gpu {
            // GPU: scale based on VRAM
            // < 4GB: 512 tokens (mobile/integrated)
            // >= 4GB: 1024 tokens (discrete GPU)
            if gpu.vram_bytes < 4 * 1024 * 1024 * 1024 {
                512
            } else {
                1024
            }
        } else {
            // CPU: conservative chunk size (from benchmark results)
            // benchmark_chunk_sizes.rs showed 256 is optimal for CPU
            256
        }
    }

    /// Create a custom configuration with specific settings
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = SystemConfig::custom(
    ///     hardware,
    ///     model,
    ///     32,   // slot_pool_size
    ///     512,  // chunk_size
    ///     2,    // dtype_bytes (f16)
    /// );
    /// ```
    pub fn custom(
        hardware: HardwareProfile,
        model: ModelMemoryProfile,
        slot_pool_size: usize,
        chunk_size: usize,
        dtype_bytes: usize,
    ) -> Self {
        Self {
            slot_pool_size,
            chunk_size,
            enable_monitoring: true,
            memory_utilization_target: 0.7,
            hardware,
            model,
            dtype_bytes,
        }
    }

    /// Get memory statistics for current configuration
    pub fn memory_stats(&self) -> MemoryStats {
        let kv_per_request = self.model.kv_cache_bytes_per_request(self.dtype_bytes);
        let total_kv_memory = kv_per_request * self.slot_pool_size as u64;
        let total_memory = self.model.weights_bytes + total_kv_memory;

        let available = if let Some(gpu) = &self.hardware.gpu {
            gpu.vram_bytes
        } else {
            self.hardware.memory.available_bytes
        };

        let utilization = total_memory as f64 / available as f64;

        MemoryStats {
            model_weights_gb: self.model.weights_bytes as f64 / 1e9,
            kv_cache_per_slot_mb: kv_per_request as f64 / 1e6,
            total_kv_cache_gb: total_kv_memory as f64 / 1e9,
            total_memory_gb: total_memory as f64 / 1e9,
            available_memory_gb: available as f64 / 1e9,
            utilization_percent: utilization * 100.0,
        }
    }

    /// Print detailed configuration summary
    pub fn print_summary(&self) {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║         INFERENCE ENGINE CONFIGURATION                  ║");
        println!("╚══════════════════════════════════════════════════════════╝");

        let stats = self.memory_stats();

        println!("\n📊 Resource Allocation:");
        println!("  Model weights:     {:.2} GB", stats.model_weights_gb);
        println!("  KV cache per slot: {:.1} MB", stats.kv_cache_per_slot_mb);
        println!(
            "  Total KV cache:    {:.2} GB ({} slots)",
            stats.total_kv_cache_gb, self.slot_pool_size
        );
        println!("  Total memory:      {:.2} GB", stats.total_memory_gb);
        println!("  Available:         {:.2} GB", stats.available_memory_gb);
        println!("  Utilization:       {:.1}%", stats.utilization_percent);

        println!("\n⚙️  Performance Settings:");
        println!(
            "  Slot pool size:    {} concurrent requests",
            self.slot_pool_size
        );
        println!("  Chunk size:        {} tokens", self.chunk_size);
        println!(
            "  Monitoring:        {}",
            if self.enable_monitoring {
                "enabled"
            } else {
                "disabled"
            }
        );

        if stats.utilization_percent > 90.0 {
            println!("\n⚠️  WARNING: Memory utilization >90%. Consider reducing slot_pool_size.");
        } else if stats.utilization_percent < 50.0 {
            println!(
                "\n💡 TIP: Memory utilization <50%. Consider increasing slot_pool_size for higher throughput."
            );
        }
    }
}

/// Memory statistics for current configuration
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub model_weights_gb: f64,
    pub kv_cache_per_slot_mb: f64,
    pub total_kv_cache_gb: f64,
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub utilization_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> ModelMemoryProfile {
        ModelMemoryProfile {
            weights_bytes: 6 * 1024 * 1024 * 1024, // 6GB
            num_layers: 32,
            hidden_size: 4096,
            num_kv_heads: 32,
            context_window: 512,
        }
    }

    #[test]
    fn test_auto_detect_config() {
        // This test requires actual hardware, so we'll skip it in CI
        // Run manually with: cargo test test_auto_detect_config -- --ignored
        if std::env::var("CI").is_ok() {
            return;
        }

        let model = test_model();
        let config = SystemConfig::auto_detect(model, 2).unwrap();

        assert!(config.slot_pool_size >= 2);
        assert!(config.slot_pool_size <= 128);
        assert!(config.chunk_size >= 128);
        assert!(config.chunk_size <= 2048);
        assert!(config.enable_monitoring);
    }

    #[test]
    fn test_memory_stats() {
        let hardware = HardwareProfile {
            cpu: crate::hardware::CpuInfo {
                physical_cores: 8,
                logical_cores: 16,
                architecture: "x86_64".to_string(),
                model_name: "Test CPU".to_string(),
            },
            memory: crate::hardware::MemoryInfo {
                total_bytes: 32 * 1024 * 1024 * 1024,     // 32GB
                available_bytes: 28 * 1024 * 1024 * 1024, // 28GB available
                bandwidth_gbs: None,
            },
            gpu: None,
            ml_score: 5.0,
        };

        let model = test_model();
        let config = SystemConfig::custom(hardware, model, 16, 256, 2);
        let stats = config.memory_stats();

        assert!((stats.model_weights_gb - 6.0).abs() < 0.5); // Within 0.5GB
        assert!(stats.kv_cache_per_slot_mb > 0.0);
        assert!(stats.total_memory_gb > stats.model_weights_gb);
        assert!(stats.utilization_percent > 0.0);
        assert!(stats.utilization_percent <= 100.0);
    }

    #[test]
    fn test_chunk_size_cpu() {
        let hardware = HardwareProfile {
            cpu: crate::hardware::CpuInfo {
                physical_cores: 4,
                logical_cores: 8,
                architecture: "x86_64".to_string(),
                model_name: "Test CPU".to_string(),
            },
            memory: crate::hardware::MemoryInfo {
                total_bytes: 16 * 1024 * 1024 * 1024,
                available_bytes: 14 * 1024 * 1024 * 1024,
                bandwidth_gbs: None,
            },
            gpu: None,
            ml_score: 4.0,
        };

        let chunk_size = SystemConfig::calculate_chunk_size(&hardware);
        assert_eq!(chunk_size, 256); // CPU default from benchmarks
    }

    #[test]
    fn test_chunk_size_gpu() {
        let hardware = HardwareProfile {
            cpu: crate::hardware::CpuInfo {
                physical_cores: 8,
                logical_cores: 16,
                architecture: "x86_64".to_string(),
                model_name: "Test CPU".to_string(),
            },
            memory: crate::hardware::MemoryInfo {
                total_bytes: 32 * 1024 * 1024 * 1024,
                available_bytes: 28 * 1024 * 1024 * 1024,
                bandwidth_gbs: None,
            },
            gpu: Some(crate::hardware::GpuInfo {
                name: "NVIDIA RTX 4090".to_string(),
                vram_bytes: 24 * 1024 * 1024 * 1024, // 24GB
                backend: crate::hardware::GpuBackend::Cuda,
                compute_capability: Some("8.9".to_string()),
            }),
            ml_score: 9.0,
        };

        let chunk_size = SystemConfig::calculate_chunk_size(&hardware);
        assert_eq!(chunk_size, 1024); // High-end GPU
    }
}
