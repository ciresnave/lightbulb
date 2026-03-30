//! Configuration for segmented KV cache with attention-driven eviction
//!
//! Controls all aspects of the segmented cache system:
//! - Automatic CacheSpan creation during inference
//! - Attention weight capture and aggregation
//! - Segment-level relevance scoring with superlinear boost
//! - Pressure-sensitive eviction parameters
//! - Tiered demotion (VRAM → RAM → Disk)
//! - Hybrid per-token eviction on lowest-scoring segments

use serde::{Deserialize, Serialize};

/// Configuration for the segmented KV cache system.
///
/// The segmented cache extends Lightbulb's existing cache infrastructure
/// (CacheSpan, H2OPolicy, VotingAggregator) with model-informed eviction
/// decisions. Rather than using heuristics (recency, frequency), the model's
/// own attention weights determine what context is valuable.
///
/// # Phases of Operation
///
/// 1. **Observation** (`shadow_mode = true`): Spans are created and scored,
///    eviction candidates are logged, but no eviction occurs. Use this to
///    tune parameters.
///
/// 2. **Live eviction** (`shadow_mode = false`): Lowest-scoring segments
///    are evicted when context pressure exceeds `eviction_pressure_threshold`.
///
/// 3. **Tiered demotion** (`tiered_demotion = true`): Evicted segments are
///    demoted to RAM/disk rather than deleted, enabling retrieval.
///
/// # Example
///
/// ```ignore
/// let config = SegmentedCacheConfig {
///     enabled: true,
///     shadow_mode: true,  // Start in observation mode
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentedCacheConfig {
    /// Master switch for the segmented cache system.
    /// When false, no spans are created and no attention is captured.
    pub enabled: bool,

    /// Enable attention weight extraction during decode.
    /// Requires the manual attention path (not FlashAttention) for the
    /// single decode token. During decode (seq_q=1), this is a single
    /// vector-matrix multiply per head per layer — not expensive.
    pub capture_attention: bool,

    /// Exponent for superlinear attention boost (default: 2.0).
    ///
    /// Higher values create sharper distinction between "glanced at" and
    /// "relied upon" tokens:
    /// - 1.0: Linear (equivalent to H2O behavior)
    /// - 2.0: Quadratic — a token receiving 0.5 attention gets 4x the boost
    ///   of one receiving 0.25
    /// - 3.0: Cubic — even sharper discrimination
    ///
    /// Recommended: start at 2.0, tune using shadow mode logs.
    pub attention_power: f32,

    /// Base decay rate per inference pass (default: 0.02).
    ///
    /// Every segment's relevance score is decremented by this amount
    /// (modified by pressure) on each pass. A segment that receives no
    /// attention for `1.0 / base_decay_rate` passes (50 at default) will
    /// have its score reduced to zero.
    pub base_decay_rate: f32,

    /// Context utilization threshold to begin eviction (default: 0.80).
    ///
    /// When `current_position / context_size` exceeds this value, the
    /// system starts evaluating segments for eviction. Below this threshold,
    /// all segments are retained.
    pub eviction_pressure_threshold: f32,

    /// EMA smoothing factor for pressure changes (default: 0.3).
    ///
    /// Prevents sudden pressure spikes (e.g., from ingesting a large file)
    /// from causing cascade evictions. Lower values = more smoothing.
    /// - 0.1: Very smooth, slow to react
    /// - 0.3: Moderate smoothing (default)
    /// - 0.9: Almost no smoothing, fast reaction
    pub pressure_ema_alpha: f32,

    /// Enable tiered demotion instead of pure deletion (default: true).
    ///
    /// When true: evicted segments are stored in RAM (CPU tensors) with
    /// KnowledgeBase summaries, and can be further demoted to disk.
    /// The model can request retrieval via `<RETRIEVE:key>` tokens.
    ///
    /// When false: evicted segments are permanently deleted. Simpler but
    /// no recovery from bad eviction decisions.
    pub tiered_demotion: bool,

    /// Enable hybrid per-token eviction on lowest-scoring segment (default: false).
    ///
    /// When the lowest-scoring segment approaches eviction, activates
    /// per-token attention tracking within that segment only. High-value
    /// tokens survive while low-value tokens are culled individually.
    /// Only one segment is tracked at a time (bounded compute).
    pub per_token_eviction: bool,

    /// Shadow mode: log eviction candidates without evicting (default: true).
    ///
    /// Start with this enabled to observe scoring behavior and tune
    /// parameters before enabling live eviction.
    pub shadow_mode: bool,

    /// How often to update H2O attention scores during decode (default: 8).
    ///
    /// Attention capture involves a GPU→CPU tensor transfer. This interval
    /// controls how many decode steps occur between updates. Lower values
    /// give more accurate scoring but cost more in transfer overhead.
    /// - 1: Every decode step (most accurate, ~2-6ms overhead per step)
    /// - 8: Every 8th step (good balance)
    /// - 32: Infrequent updates (minimal overhead)
    pub h2o_update_interval: usize,

    /// Enable debug logging of per-span attention summaries (default: false).
    ///
    /// When enabled, logs span_id, tag, position range, and total/average
    /// attention for each active span at each H2O update. Useful for
    /// understanding attention distributions during development.
    pub log_span_attention: bool,

    /// Maximum number of segments to demote to RAM (default: 64).
    ///
    /// Limits RAM usage from demoted KV tensors. When exceeded, oldest
    /// RAM-tier segments are further demoted to disk (if available).
    pub max_ram_demoted_segments: usize,

    /// Disk storage backend path for cold-tier segments (default: None).
    ///
    /// When set, enables RAM→Disk demotion using RocksDB at this path.
    /// When None, segments demoted past RAM limit are permanently deleted.
    pub disk_storage_path: Option<String>,
}

impl Default for SegmentedCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            capture_attention: true,
            attention_power: 2.0,
            base_decay_rate: 0.02,
            eviction_pressure_threshold: 0.80,
            pressure_ema_alpha: 0.3,
            tiered_demotion: true,
            per_token_eviction: false,
            shadow_mode: true,
            h2o_update_interval: 8,
            log_span_attention: false,
            max_ram_demoted_segments: 64,
            disk_storage_path: None,
        }
    }
}

impl SegmentedCacheConfig {
    /// Create a config suitable for initial observation/tuning.
    ///
    /// Shadow mode enabled, attention logging on, no eviction.
    pub fn observation() -> Self {
        Self {
            enabled: true,
            shadow_mode: true,
            log_span_attention: true,
            ..Default::default()
        }
    }

    /// Create a config for live operation with tiered demotion.
    pub fn live() -> Self {
        Self {
            enabled: true,
            shadow_mode: false,
            tiered_demotion: true,
            ..Default::default()
        }
    }

    /// Create a config for live operation with simple deletion (no tiers).
    pub fn live_simple() -> Self {
        Self {
            enabled: true,
            shadow_mode: false,
            tiered_demotion: false,
            ..Default::default()
        }
    }

    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), String> {
        if self.attention_power < 1.0 {
            return Err(format!(
                "attention_power must be >= 1.0, got {}",
                self.attention_power
            ));
        }
        if self.base_decay_rate <= 0.0 || self.base_decay_rate >= 1.0 {
            return Err(format!(
                "base_decay_rate must be in (0.0, 1.0), got {}",
                self.base_decay_rate
            ));
        }
        if self.eviction_pressure_threshold <= 0.0 || self.eviction_pressure_threshold >= 1.0 {
            return Err(format!(
                "eviction_pressure_threshold must be in (0.0, 1.0), got {}",
                self.eviction_pressure_threshold
            ));
        }
        if self.pressure_ema_alpha <= 0.0 || self.pressure_ema_alpha >= 1.0 {
            return Err(format!(
                "pressure_ema_alpha must be in (0.0, 1.0), got {}",
                self.pressure_ema_alpha
            ));
        }
        if self.h2o_update_interval == 0 {
            return Err("h2o_update_interval must be > 0".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validates() {
        let config = SegmentedCacheConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_observation_config() {
        let config = SegmentedCacheConfig::observation();
        assert!(config.enabled);
        assert!(config.shadow_mode);
        assert!(config.log_span_attention);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_live_config() {
        let config = SegmentedCacheConfig::live();
        assert!(config.enabled);
        assert!(!config.shadow_mode);
        assert!(config.tiered_demotion);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_attention_power() {
        let config = SegmentedCacheConfig {
            attention_power: 0.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_decay_rate() {
        let config = SegmentedCacheConfig {
            base_decay_rate: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        let config = SegmentedCacheConfig {
            base_decay_rate: 1.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_pressure_threshold() {
        let config = SegmentedCacheConfig {
            eviction_pressure_threshold: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_h2o_interval() {
        let config = SegmentedCacheConfig {
            h2o_update_interval: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
