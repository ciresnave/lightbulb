//! Adaptive Mixed-Precision Inference
//!
//! Dynamic per-layer precision selection based on activation statistics and
//! accuracy requirements. Enables optimal performance/accuracy tradeoffs.
//!
//! # Features
//!
//! - **Per-Layer Profiling**: Track activation statistics per layer
//! - **Dynamic Precision Selection**: Choose FP32/FP16/BF16/INT8 adaptively
//! - **Accuracy Monitoring**: Track accuracy degradation from quantization
//! - **Performance Benchmarking**: Per-layer latency profiling
//!
//! # Example
//!
//! ```rust,ignore
//! use lightbulb::engine::mixed_precision::{MixedPrecisionConfig, PrecisionProfiler};
//!
//! // Configure mixed precision
//! let config = MixedPrecisionConfig::new()
//!     .default_precision(Precision::FP16)
//!     .accuracy_threshold(0.95);
//!
//! // Profile activations
//! let mut profiler = PrecisionProfiler::new(config);
//! profiler.record_activation(layer_id, &activations);
//! let precision = profiler.select_precision(layer_id);
//! ```

use std::collections::HashMap;

/// Supported precision modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 32-bit floating point (full precision)
    FP32,
    /// 16-bit floating point
    FP16,
    /// Brain float 16 (mixed precision training format)
    BF16,
    /// 8-bit integer
    INT8,
}

impl Precision {
    /// Get bits per parameter
    pub fn bits(&self) -> usize {
        match self {
            Precision::FP32 => 32,
            Precision::FP16 => 16,
            Precision::BF16 => 16,
            Precision::INT8 => 8,
        }
    }

    /// Get relative memory usage (compared to FP32)
    pub fn memory_ratio(&self) -> f32 {
        self.bits() as f32 / 32.0
    }

    /// Get estimated speedup (compared to FP32)
    pub fn speedup_estimate(&self) -> f32 {
        match self {
            Precision::FP32 => 1.0,
            Precision::FP16 => 1.8,
            Precision::BF16 => 1.7,
            Precision::INT8 => 3.0,
        }
    }
}

/// Configuration for mixed-precision inference
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Default precision for unoptimized layers
    pub default_precision: Precision,

    /// Accuracy threshold (0-1) for precision selection
    pub accuracy_threshold: f32,

    /// Enable dynamic precision adjustment
    pub dynamic_adjustment: bool,

    /// Profiling window size (number of batches)
    pub profiling_window: usize,

    /// Per-layer precision overrides
    pub layer_overrides: HashMap<usize, Precision>,

    /// Sensitive layer patterns (always use high precision)
    pub sensitive_patterns: Vec<String>,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            default_precision: Precision::FP16,
            accuracy_threshold: 0.95,
            dynamic_adjustment: true,
            profiling_window: 10,
            layer_overrides: HashMap::new(),
            sensitive_patterns: vec![
                "lm_head".to_string(),
                "embed".to_string(),
                "layernorm".to_string(),
            ],
        }
    }
}

impl MixedPrecisionConfig {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set default precision
    pub fn default_precision(mut self, precision: Precision) -> Self {
        self.default_precision = precision;
        self
    }

    /// Set accuracy threshold
    pub fn accuracy_threshold(mut self, threshold: f32) -> Self {
        self.accuracy_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Enable/disable dynamic adjustment
    pub fn dynamic_adjustment(mut self, enabled: bool) -> Self {
        self.dynamic_adjustment = enabled;
        self
    }

    /// Set profiling window
    pub fn profiling_window(mut self, window: usize) -> Self {
        self.profiling_window = window;
        self
    }

    /// Add layer precision override
    pub fn set_layer_precision(mut self, layer_id: usize, precision: Precision) -> Self {
        self.layer_overrides.insert(layer_id, precision);
        self
    }

    /// Check if layer is sensitive (should use high precision)
    pub fn is_sensitive_layer(&self, layer_name: &str) -> bool {
        self.sensitive_patterns
            .iter()
            .any(|pattern| layer_name.contains(pattern))
    }
}

/// Activation statistics for a layer
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Mean absolute value
    pub mean_abs: f32,

    /// Standard deviation
    pub std_dev: f32,

    /// Min value
    pub min: f32,

    /// Max value
    pub max: f32,

    /// Dynamic range (max - min)
    pub dynamic_range: f32,

    /// Number of samples
    pub num_samples: usize,
}

impl ActivationStats {
    /// Create empty stats
    pub fn new() -> Self {
        Self {
            mean_abs: 0.0,
            std_dev: 0.0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            dynamic_range: 0.0,
            num_samples: 0,
        }
    }

    /// Update stats with new values
    pub fn update(&mut self, values: &[f32]) {
        if values.is_empty() {
            return;
        }

        // Update min/max
        for &v in values {
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        }
        self.dynamic_range = self.max - self.min;

        // Update mean
        let sum: f32 = values.iter().map(|v| v.abs()).sum();
        let new_mean = sum / values.len() as f32;
        self.mean_abs = if self.num_samples == 0 {
            new_mean
        } else {
            // Incremental mean
            (self.mean_abs * self.num_samples as f32 + sum)
                / (self.num_samples + values.len()) as f32
        };

        // Update std dev
        let variance: f32 = values
            .iter()
            .map(|v| (v - self.mean_abs).powi(2))
            .sum::<f32>()
            / values.len() as f32;
        self.std_dev = variance.sqrt();

        self.num_samples += values.len();
    }

    /// Recommend precision based on stats
    pub fn recommend_precision(&self) -> Precision {
        // FP32 for extreme dynamic range
        if self.dynamic_range > 1e6 {
            return Precision::FP32;
        }

        // INT8 for narrow range with low std dev
        if self.dynamic_range < 10.0 && self.std_dev < 1.0 {
            return Precision::INT8;
        }

        // BF16 for moderate range
        if self.dynamic_range < 1e4 {
            return Precision::BF16;
        }

        // FP16 default
        Precision::FP16
    }
}

impl Default for ActivationStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Precision profiler that tracks activation statistics
pub struct PrecisionProfiler {
    /// Configuration
    config: MixedPrecisionConfig,

    /// Per-layer activation statistics
    layer_stats: HashMap<usize, ActivationStats>,

    /// Per-layer precision selections
    layer_precisions: HashMap<usize, Precision>,

    /// Per-layer accuracy estimates
    layer_accuracies: HashMap<usize, f32>,

    /// Per-layer latencies (microseconds)
    layer_latencies: HashMap<usize, Vec<u64>>,

    /// Number of profiling batches processed
    batches_processed: usize,
}

impl PrecisionProfiler {
    /// Create new profiler
    pub fn new(config: MixedPrecisionConfig) -> Self {
        Self {
            config,
            layer_stats: HashMap::new(),
            layer_precisions: HashMap::new(),
            layer_accuracies: HashMap::new(),
            layer_latencies: HashMap::new(),
            batches_processed: 0,
        }
    }

    /// Record activation statistics for a layer
    pub fn record_activation(&mut self, layer_id: usize, values: &[f32]) {
        let stats = self.layer_stats.entry(layer_id).or_insert_with(ActivationStats::new);
        stats.update(values);
    }

    /// Record layer latency
    pub fn record_latency(&mut self, layer_id: usize, latency_us: u64) {
        let latencies = self.layer_latencies.entry(layer_id).or_insert_with(Vec::new);
        latencies.push(latency_us);

        // Keep only recent window
        if latencies.len() > self.config.profiling_window {
            latencies.remove(0);
        }
    }

    /// Record accuracy for a layer
    pub fn record_accuracy(&mut self, layer_id: usize, accuracy: f32) {
        self.layer_accuracies.insert(layer_id, accuracy);
    }

    /// Select precision for a layer
    pub fn select_precision(&mut self, layer_id: usize) -> Precision {
        // Check override first
        if let Some(&precision) = self.config.layer_overrides.get(&layer_id) {
            return precision;
        }

        // Check if already selected
        if let Some(&precision) = self.layer_precisions.get(&layer_id) {
            // Adjust dynamically if enabled
            if self.config.dynamic_adjustment && self.batches_processed > self.config.profiling_window {
                return self.adjust_precision(layer_id, precision);
            }
            return precision;
        }

        // Initial selection based on stats
        let precision = if let Some(stats) = self.layer_stats.get(&layer_id) {
            stats.recommend_precision()
        } else {
            self.config.default_precision
        };

        self.layer_precisions.insert(layer_id, precision);
        precision
    }

    /// Dynamically adjust precision based on accuracy
    fn adjust_precision(&mut self, layer_id: usize, current: Precision) -> Precision {
        let accuracy = self.layer_accuracies.get(&layer_id).copied().unwrap_or(1.0);

        // If accuracy is below threshold, increase precision
        if accuracy < self.config.accuracy_threshold {
            let new_precision = match current {
                Precision::INT8 => Precision::FP16,
                Precision::FP16 => Precision::BF16,
                Precision::BF16 => Precision::FP32,
                Precision::FP32 => Precision::FP32, // Already max
            };
            self.layer_precisions.insert(layer_id, new_precision);
            return new_precision;
        }

        // If accuracy is good, try lower precision for speedup
        if accuracy > 0.99 && self.batches_processed % 10 == 0 {
            let new_precision = match current {
                Precision::FP32 => Precision::BF16,
                Precision::BF16 => Precision::FP16,
                Precision::FP16 => Precision::INT8,
                Precision::INT8 => Precision::INT8, // Already min
            };
            self.layer_precisions.insert(layer_id, new_precision);
            return new_precision;
        }

        current
    }

    /// Finalize batch profiling
    pub fn end_batch(&mut self) {
        self.batches_processed += 1;
    }

    /// Get average latency for a layer
    pub fn get_avg_latency(&self, layer_id: usize) -> Option<f64> {
        self.layer_latencies.get(&layer_id).map(|latencies| {
            if latencies.is_empty() {
                return 0.0;
            }
            latencies.iter().sum::<u64>() as f64 / latencies.len() as f64
        })
    }

    /// Get statistics for a layer
    pub fn get_stats(&self, layer_id: usize) -> Option<&ActivationStats> {
        self.layer_stats.get(&layer_id)
    }

    /// Get precision for a layer
    pub fn get_precision(&self, layer_id: usize) -> Option<Precision> {
        self.layer_precisions.get(&layer_id).copied()
    }

    /// Get overall statistics
    pub fn overall_stats(&self) -> PrecisionStats {
        let total_layers = self.layer_precisions.len();
        if total_layers == 0 {
            return PrecisionStats::default();
        }

        let mut precision_counts = HashMap::new();
        for &precision in self.layer_precisions.values() {
            *precision_counts.entry(precision).or_insert(0) += 1;
        }

        let avg_memory_ratio = self
            .layer_precisions
            .values()
            .map(|p| p.memory_ratio())
            .sum::<f32>()
            / total_layers as f32;

        let avg_speedup = self
            .layer_precisions
            .values()
            .map(|p| p.speedup_estimate())
            .sum::<f32>()
            / total_layers as f32;

        PrecisionStats {
            total_layers,
            fp32_layers: *precision_counts.get(&Precision::FP32).unwrap_or(&0),
            fp16_layers: *precision_counts.get(&Precision::FP16).unwrap_or(&0),
            bf16_layers: *precision_counts.get(&Precision::BF16).unwrap_or(&0),
            int8_layers: *precision_counts.get(&Precision::INT8).unwrap_or(&0),
            avg_memory_ratio,
            avg_speedup_estimate: avg_speedup,
            batches_processed: self.batches_processed,
        }
    }

    /// Reset profiler
    pub fn reset(&mut self) {
        self.layer_stats.clear();
        self.layer_precisions.clear();
        self.layer_accuracies.clear();
        self.layer_latencies.clear();
        self.batches_processed = 0;
    }
}

/// Overall precision statistics
#[derive(Debug, Clone, Default)]
pub struct PrecisionStats {
    /// Total layers profiled
    pub total_layers: usize,

    /// Layers using FP32
    pub fp32_layers: usize,

    /// Layers using FP16
    pub fp16_layers: usize,

    /// Layers using BF16
    pub bf16_layers: usize,

    /// Layers using INT8
    pub int8_layers: usize,

    /// Average memory ratio (compared to full FP32)
    pub avg_memory_ratio: f32,

    /// Average speedup estimate
    pub avg_speedup_estimate: f32,

    /// Batches processed
    pub batches_processed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_properties() {
        assert_eq!(Precision::FP32.bits(), 32);
        assert_eq!(Precision::FP16.bits(), 16);
        assert_eq!(Precision::INT8.bits(), 8);

        assert_eq!(Precision::FP32.memory_ratio(), 1.0);
        assert_eq!(Precision::FP16.memory_ratio(), 0.5);
        assert_eq!(Precision::INT8.memory_ratio(), 0.25);

        assert!(Precision::INT8.speedup_estimate() > Precision::FP16.speedup_estimate());
    }

    #[test]
    fn test_config_defaults() {
        let config = MixedPrecisionConfig::default();
        assert_eq!(config.default_precision, Precision::FP16);
        assert_eq!(config.accuracy_threshold, 0.95);
        assert!(config.dynamic_adjustment);
    }

    #[test]
    fn test_config_builder() {
        let config = MixedPrecisionConfig::new()
            .default_precision(Precision::BF16)
            .accuracy_threshold(0.99)
            .dynamic_adjustment(false)
            .profiling_window(20)
            .set_layer_precision(0, Precision::FP32);

        assert_eq!(config.default_precision, Precision::BF16);
        assert_eq!(config.accuracy_threshold, 0.99);
        assert!(!config.dynamic_adjustment);
        assert_eq!(config.profiling_window, 20);
        assert_eq!(config.layer_overrides.get(&0), Some(&Precision::FP32));
    }

    #[test]
    fn test_sensitive_layer_detection() {
        let config = MixedPrecisionConfig::default();
        assert!(config.is_sensitive_layer("model.lm_head.weight"));
        assert!(config.is_sensitive_layer("embed_tokens"));
        assert!(config.is_sensitive_layer("final_layernorm"));
        assert!(!config.is_sensitive_layer("attention.query"));
    }

    #[test]
    fn test_activation_stats() {
        let mut stats = ActivationStats::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        stats.update(&values);

        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.dynamic_range, 4.0);
        assert_eq!(stats.num_samples, 5);
        assert!(stats.mean_abs > 2.5 && stats.mean_abs < 3.5);
    }

    #[test]
    fn test_stats_precision_recommendation() {
        // Narrow range -> INT8
        let mut stats1 = ActivationStats::new();
        stats1.update(&vec![1.0, 1.5, 2.0, 2.5, 3.0]);
        assert_eq!(stats1.recommend_precision(), Precision::INT8);

        // Moderate range -> BF16
        let mut stats2 = ActivationStats::new();
        stats2.update(&vec![-100.0, -50.0, 0.0, 50.0, 100.0]);
        assert_eq!(stats2.recommend_precision(), Precision::BF16);

        // Extreme range -> FP32
        let mut stats3 = ActivationStats::new();
        stats3.update(&vec![-1e7, 0.0, 1e7]);
        assert_eq!(stats3.recommend_precision(), Precision::FP32);
    }

    #[test]
    fn test_profiler_basic() {
        let config = MixedPrecisionConfig::new();
        let mut profiler = PrecisionProfiler::new(config);

        // Record activations
        profiler.record_activation(0, &vec![1.0, 2.0, 3.0]);
        profiler.record_activation(1, &vec![-1e6, 0.0, 1e6]);

        // Select precisions
        let precision0 = profiler.select_precision(0);
        let precision1 = profiler.select_precision(1);

        // Layer 0 should use lower precision (narrow range)
        assert!(precision0 == Precision::INT8 || precision0 == Precision::FP16);

        // Layer 1 should use higher precision (wide range)
        assert!(precision1 == Precision::FP32 || precision1 == Precision::BF16);
    }

    #[test]
    fn test_profiler_overrides() {
        let config = MixedPrecisionConfig::new().set_layer_precision(0, Precision::FP32);
        let mut profiler = PrecisionProfiler::new(config);

        // Even with narrow range, override should be respected
        profiler.record_activation(0, &vec![1.0, 2.0, 3.0]);
        assert_eq!(profiler.select_precision(0), Precision::FP32);
    }

    #[test]
    fn test_profiler_latency_tracking() {
        let config = MixedPrecisionConfig::new();
        let mut profiler = PrecisionProfiler::new(config);

        profiler.record_latency(0, 100);
        profiler.record_latency(0, 200);
        profiler.record_latency(0, 300);

        let avg = profiler.get_avg_latency(0).unwrap();
        assert!((avg - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_profiler_accuracy_tracking() {
        let config = MixedPrecisionConfig::new();
        let mut profiler = PrecisionProfiler::new(config);

        profiler.record_activation(0, &vec![1.0, 2.0, 3.0]);
        profiler.select_precision(0); // Initial selection

        profiler.record_accuracy(0, 0.90); // Below threshold
        profiler.end_batch();

        // Should upgrade precision due to low accuracy
        // (but only after profiling window)
        for _ in 0..10 {
            profiler.end_batch();
        }

        let new_precision = profiler.select_precision(0);
        // Precision should increase or stay same
        assert!(new_precision != Precision::INT8 || profiler.batches_processed <= 10);
    }

    #[test]
    fn test_overall_stats() {
        let config = MixedPrecisionConfig::new();
        let mut profiler = PrecisionProfiler::new(config);

        // Profile multiple layers
        profiler.record_activation(0, &vec![1.0, 2.0]);
        profiler.record_activation(1, &vec![1e6, 2e6]);
        profiler.record_activation(2, &vec![10.0, 20.0]);

        profiler.select_precision(0);
        profiler.select_precision(1);
        profiler.select_precision(2);

        let stats = profiler.overall_stats();
        assert_eq!(stats.total_layers, 3);
        assert!(stats.avg_memory_ratio < 1.0); // Should use less memory than FP32
        assert!(stats.avg_speedup_estimate > 1.0); // Should be faster
    }

    #[test]
    fn test_profiler_reset() {
        let config = MixedPrecisionConfig::new();
        let mut profiler = PrecisionProfiler::new(config);

        profiler.record_activation(0, &vec![1.0, 2.0]);
        profiler.select_precision(0);
        profiler.end_batch();

        assert_eq!(profiler.batches_processed, 1);
        assert!(!profiler.layer_stats.is_empty());

        profiler.reset();

        assert_eq!(profiler.batches_processed, 0);
        assert!(profiler.layer_stats.is_empty());
        assert!(profiler.layer_precisions.is_empty());
    }
}
