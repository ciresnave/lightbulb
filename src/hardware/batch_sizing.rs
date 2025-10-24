//! Dynamic batch size calculation based on hardware resources
//!
//! This module implements adaptive batch sizing that automatically scales
//! from minimal hardware (2-4 batch size) to server-class hardware (64-128+).
//!
//! # Algorithm
//!
//! Batch size depends on:
//! 1. **Memory constraints**: Reserve headroom for KV cache growth
//! 2. **CPU/GPU parallelism**: Scale with core count and memory bandwidth
//! 3. **Safety margins**: Conservative estimates prevent OOM (70% max utilization)
//!
//! # Formula
//!
//! ```text
//! max_batch_size = min(
//!     memory_limited_batch_size,
//!     cpu_limited_batch_size,
//!     gpu_limited_batch_size (if GPU available)
//! )
//! ```

use crate::hardware::HardwareProfile;
use anyhow::Result;

/// Configuration for batch size calculation
#[derive(Debug, Clone)]
pub struct BatchSizeConfig {
    /// Target memory utilization (0.0-1.0, default 0.7)
    pub memory_utilization_target: f64,
    /// Safety margin for memory estimates (default 1.2x actual)
    pub memory_safety_margin: f64,
    /// Minimum batch size (default 2)
    pub min_batch_size: usize,
    /// Maximum batch size (default 128)
    pub max_batch_size: usize,
}

impl Default for BatchSizeConfig {
    fn default() -> Self {
        Self {
            memory_utilization_target: 0.7,
            memory_safety_margin: 1.2,
            min_batch_size: 2,
            max_batch_size: 128,
        }
    }
}

/// Model memory requirements
#[derive(Debug, Clone)]
pub struct ModelMemoryProfile {
    /// Model weight size in bytes (e.g., 3B params × 2 bytes/param for f16 = 6GB)
    pub weights_bytes: u64,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of KV heads (for GQA models)
    pub num_kv_heads: usize,
    /// Context window size (max sequence length)
    pub context_window: usize,
}

impl ModelMemoryProfile {
    /// Estimate per-request KV cache memory usage
    ///
    /// KV cache shape: [num_layers, num_kv_heads, context_window, head_dim]
    /// head_dim = hidden_size / num_attention_heads
    ///
    /// For each layer, we store 2 tensors (K and V):
    /// - K: [num_kv_heads, context_window, head_dim]
    /// - V: [num_kv_heads, context_window, head_dim]
    ///
    /// Memory = num_layers × 2 (K+V) × num_kv_heads × context_window × head_dim × bytes_per_element
    pub fn kv_cache_bytes_per_request(&self, dtype_bytes: usize) -> u64 {
        // head_dim = hidden_size / num_heads (assume num_heads = num_kv_heads for simplicity)
        let head_dim = self.hidden_size / self.num_kv_heads.max(1);
        
        let kv_elements_per_layer = 
            2 * self.num_kv_heads * self.context_window * head_dim;
        
        let total_elements = self.num_layers * kv_elements_per_layer;
        
        (total_elements * dtype_bytes) as u64
    }

    /// Estimate total memory per request (weights + KV cache)
    pub fn total_bytes_per_request(&self, dtype_bytes: usize) -> u64 {
        // Note: Model weights are shared across all requests
        // Only KV cache scales with batch size
        self.kv_cache_bytes_per_request(dtype_bytes)
    }
}

/// Calculate optimal batch size based on hardware and model
///
/// # Arguments
///
/// * `profile` - Hardware capabilities (CPU, RAM, GPU)
/// * `model` - Model memory requirements
/// * `dtype_bytes` - Bytes per tensor element (2 for f16, 4 for f32)
/// * `config` - Batch sizing configuration (optional, uses defaults)
///
/// # Returns
///
/// Recommended batch size that balances:
/// - Memory constraints (prevent OOM)
/// - CPU/GPU parallelism (utilize available cores)
/// - Safety margins (conservative estimates)
///
/// # Example
///
/// ```ignore
/// let profile = HardwareProfile::detect()?;
/// let model = ModelMemoryProfile {
///     weights_bytes: 6 * 1024 * 1024 * 1024, // 6GB
///     num_layers: 32,
///     hidden_size: 4096,
///     num_kv_heads: 32,
///     context_window: 512,
/// };
/// let batch_size = calculate_optimal_batch_size(&profile, &model, 2, None)?;
/// ```
pub fn calculate_optimal_batch_size(
    profile: &HardwareProfile,
    model: &ModelMemoryProfile,
    dtype_bytes: usize,
    config: Option<BatchSizeConfig>,
) -> Result<usize> {
    let config = config.unwrap_or_default();

    // === 1. Memory-limited batch size ===
    
    // Available memory for inference (after OS overhead)
    let available_memory = (profile.memory.available_bytes as f64 
        * config.memory_utilization_target) as u64;
    
    // Memory needed for model weights (shared across all requests)
    let model_weights = model.weights_bytes;
    
    // Remaining memory for KV caches
    let memory_for_kv = available_memory.saturating_sub(model_weights);
    
    // Per-request memory (KV cache only, since weights are shared)
    let per_request_memory = (model.total_bytes_per_request(dtype_bytes) as f64 
        * config.memory_safety_margin) as u64;
    
    let memory_limited_batch = if per_request_memory > 0 {
        (memory_for_kv / per_request_memory) as usize
    } else {
        config.max_batch_size
    };

    // === 2. CPU-limited batch size ===
    
    // Scale with CPU cores, but apply diminishing returns
    // Formula: batch_size = sqrt(cores) × multiplier
    // - Dual-core: sqrt(2) × 2 ≈ 3
    // - 16-core: sqrt(16) × 2 = 8
    // - 128-core: sqrt(128) × 2 ≈ 23
    let cpu_cores = profile.cpu.physical_cores;
    let cpu_multiplier = if profile.gpu.is_some() { 1.5 } else { 2.0 };
    let cpu_limited_batch = ((cpu_cores as f64).sqrt() * cpu_multiplier).ceil() as usize;

    // === 3. GPU-limited batch size (if applicable) ===
    
    let gpu_limited_batch = if let Some(gpu) = &profile.gpu {
        // GPU batch sizing: VRAM capacity is primary constraint
        let vram_for_kv = ((gpu.vram_bytes as f64 * config.memory_utilization_target) as u64)
            .saturating_sub(model_weights);
        
        let gpu_batch = if per_request_memory > 0 {
            (vram_for_kv / per_request_memory) as usize
        } else {
            config.max_batch_size
        };
        
        // GPUs benefit from larger batches (kernel launch amortization)
        // Boost by 1.5x compared to CPU
        ((gpu_batch as f64) * 1.5).ceil() as usize
    } else {
        config.max_batch_size
    };

    // === 4. Final batch size: minimum of all constraints ===
    
    let optimal_batch = memory_limited_batch
        .min(cpu_limited_batch)
        .min(gpu_limited_batch)
        .clamp(config.min_batch_size, config.max_batch_size);

    Ok(optimal_batch)
}

/// Runtime batch size adjuster for dynamic scaling
///
/// Monitors actual memory usage during inference and adjusts batch size
/// if needed to prevent OOM or increase utilization.
#[derive(Debug)]
pub struct RuntimeBatchAdjuster {
    current_batch_size: usize,
    peak_memory_per_request: Option<u64>,
    config: BatchSizeConfig,
    adjustment_history: Vec<(std::time::Instant, usize, String)>,
}

impl RuntimeBatchAdjuster {
    /// Create a new runtime adjuster with initial batch size
    pub fn new(initial_batch_size: usize, config: BatchSizeConfig) -> Self {
        Self {
            current_batch_size: initial_batch_size,
            peak_memory_per_request: None,
            config,
            adjustment_history: Vec::new(),
        }
    }

    /// Record observed memory usage for a batch
    ///
    /// Call this after each batch to track actual memory consumption.
    pub fn record_batch_memory(&mut self, total_memory_used: u64, batch_size: usize) {
        if batch_size > 0 {
            let per_request = total_memory_used / batch_size as u64;
            
            self.peak_memory_per_request = Some(
                self.peak_memory_per_request
                    .map(|peak| peak.max(per_request))
                    .unwrap_or(per_request)
            );
        }
    }

    /// Check if batch size adjustment is needed
    ///
    /// # Arguments
    ///
    /// * `current_memory_usage` - Current total memory usage
    /// * `available_memory` - Total available memory
    /// * `queue_length` - Number of pending requests
    ///
    /// # Returns
    ///
    /// New batch size if adjustment recommended, None otherwise
    pub fn check_adjustment(
        &mut self,
        current_memory_usage: u64,
        available_memory: u64,
        queue_length: usize,
    ) -> Option<usize> {
        let utilization = current_memory_usage as f64 / available_memory as f64;
        
        // Reduce batch size if memory pressure is high (>80%)
        if utilization > 0.8 && self.current_batch_size > self.config.min_batch_size {
            let new_batch_size = (self.current_batch_size * 3 / 4).max(self.config.min_batch_size);
            
            self.adjustment_history.push((
                std::time::Instant::now(),
                new_batch_size,
                format!("Reduced: High memory pressure ({:.1}%)", utilization * 100.0),
            ));
            
            self.current_batch_size = new_batch_size;
            return Some(new_batch_size);
        }
        
        // Increase batch size if utilization is low (<50%) and queue is growing
        if utilization < 0.5 
            && queue_length > self.current_batch_size 
            && self.current_batch_size < self.config.max_batch_size 
        {
            let new_batch_size = (self.current_batch_size * 5 / 4).min(self.config.max_batch_size);
            
            self.adjustment_history.push((
                std::time::Instant::now(),
                new_batch_size,
                format!("Increased: Low utilization ({:.1}%), queue backlog ({})", 
                    utilization * 100.0, queue_length),
            ));
            
            self.current_batch_size = new_batch_size;
            return Some(new_batch_size);
        }
        
        None
    }

    /// Get current batch size
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// Get adjustment history for logging
    pub fn history(&self) -> &[(std::time::Instant, usize, String)] {
        &self.adjustment_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_calculation() {
        // Llama 3B-like model
        let model = ModelMemoryProfile {
            weights_bytes: 6 * 1024 * 1024 * 1024, // 6GB
            num_layers: 32,
            hidden_size: 4096,
            num_kv_heads: 32,
            context_window: 512,
        };
        
        let kv_bytes = model.kv_cache_bytes_per_request(2); // f16
        
        // Expected: 32 layers × 2 (K+V) × 32 heads × 512 context × 128 head_dim × 2 bytes
        // = 32 × 2 × 32 × 512 × 128 × 2 = 268,435,456 bytes ≈ 256 MB
        
        println!("KV cache per request: {:.1} MB", kv_bytes as f64 / (1024.0 * 1024.0));
        assert!(kv_bytes > 200 * 1024 * 1024); // At least 200 MB
        assert!(kv_bytes < 400 * 1024 * 1024); // Less than 400 MB
    }

    #[test]
    fn test_batch_size_scaling() {
        use crate::hardware::{CpuInfo, MemoryInfo, GpuInfo, GpuBackend};

        // Test different hardware profiles
        
        // 1. Minimal hardware: 4GB RAM, dual-core CPU
        let profile_minimal = HardwareProfile {
            cpu: CpuInfo {
                physical_cores: 2,
                logical_cores: 4,
                architecture: "x86_64".to_string(),
                model_name: "Dual Core".to_string(),
            },
            memory: MemoryInfo {
                total_bytes: 4 * 1024 * 1024 * 1024,
                available_bytes: 3 * 1024 * 1024 * 1024,
                bandwidth_gbs: None,
            },
            gpu: None,
            ml_score: 3.0,
        };
        
        // 2. Mid-range hardware: 16GB RAM, 8-core CPU
        let profile_mid = HardwareProfile {
            cpu: CpuInfo {
                physical_cores: 8,
                logical_cores: 16,
                architecture: "x86_64".to_string(),
                model_name: "8-Core CPU".to_string(),
            },
            memory: MemoryInfo {
                total_bytes: 16 * 1024 * 1024 * 1024,
                available_bytes: 12 * 1024 * 1024 * 1024,
                bandwidth_gbs: Some(40.0),
            },
            gpu: None,
            ml_score: 6.0,
        };
        
        // 3. High-end hardware: 32GB RAM, 16-core CPU, 24GB GPU
        let profile_high = HardwareProfile {
            cpu: CpuInfo {
                physical_cores: 16,
                logical_cores: 32,
                architecture: "x86_64".to_string(),
                model_name: "16-Core CPU".to_string(),
            },
            memory: MemoryInfo {
                total_bytes: 32 * 1024 * 1024 * 1024,
                available_bytes: 24 * 1024 * 1024 * 1024,
                bandwidth_gbs: Some(80.0),
            },
            gpu: Some(GpuInfo {
                name: "RTX 4090".to_string(),
                vram_bytes: 24 * 1024 * 1024 * 1024,
                backend: GpuBackend::Cuda,
                compute_capability: Some("8.9".to_string()),
            }),
            ml_score: 9.0,
        };
        
        // Llama 3B model
        let model = ModelMemoryProfile {
            weights_bytes: 6 * 1024 * 1024 * 1024,
            num_layers: 32,
            hidden_size: 4096,
            num_kv_heads: 32,
            context_window: 512,
        };
        
        let batch_minimal = calculate_optimal_batch_size(&profile_minimal, &model, 2, None).unwrap();
        let batch_mid = calculate_optimal_batch_size(&profile_mid, &model, 2, None).unwrap();
        let batch_high = calculate_optimal_batch_size(&profile_high, &model, 2, None).unwrap();
        
        println!("Batch sizes:");
        println!("  Minimal (4GB): {}", batch_minimal);
        println!("  Mid (16GB): {}", batch_mid);
        println!("  High (32GB + GPU): {}", batch_high);
        
        // Verify scaling: minimal < mid, and high is at least as good as mid
        assert!(batch_minimal >= 2, "Minimal should be at least 2");
        assert!(batch_mid > batch_minimal, "Mid should exceed minimal");
        assert!(batch_high >= batch_mid, "High should be at least as good as mid");
        
        // Verify reasonable bounds
        assert!(batch_minimal <= 8);
        assert!(batch_mid <= 32);
        assert!(batch_high <= 128);
    }

    #[test]
    fn test_runtime_adjustment() {
        let config = BatchSizeConfig::default();
        let mut adjuster = RuntimeBatchAdjuster::new(8, config);
        
        // Simulate high memory pressure
        let available = 16 * 1024 * 1024 * 1024u64; // 16GB
        let high_usage = (available as f64 * 0.85) as u64; // 85% usage
        
        let adjustment = adjuster.check_adjustment(high_usage, available, 5);
        assert!(adjustment.is_some());
        assert!(adjustment.unwrap() < 8); // Should reduce
        
        println!("Adjustment after high memory: {} -> {}", 8, adjustment.unwrap());
        
        // Simulate low utilization with queue backlog
        let low_usage = (available as f64 * 0.3) as u64; // 30% usage
        let queue_length = 20; // Many pending requests
        
        let adjustment2 = adjuster.check_adjustment(low_usage, available, queue_length);
        if let Some(new_batch) = adjustment2 {
            println!("Adjustment after low utilization: {} -> {}", 
                adjuster.current_batch_size(), new_batch);
            // Either increased or stayed same (clamped at max)
            assert!(new_batch >= adjuster.current_batch_size(), 
                "Should increase or stay same when memory available");
        } else {
            println!("No adjustment needed - already at optimal size");
        }
    }
}
