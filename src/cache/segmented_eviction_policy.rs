//! Segment-aware eviction policy with attention-driven relevance scoring
//!
//! This policy uses the model's own attention weights to determine which segments
//! of the KV cache are most valuable. Unlike per-token policies (H2O), it operates
//! at the segment level — groups of tokens that form coherent units (user prompts,
//! model generations, tool outputs, etc.).
//!
//! # Key Design Decisions
//!
//! - **Superlinear boost**: Attention weight is raised to a configurable power (default 2.0).
//!   This captures the asymmetry between "glanced at" (0.02 attention → 0.0004 boost)
//!   and "relied upon" (0.8 attention → 0.64 boost).
//!
//! - **Diminishing returns**: Within a single evaluation pass, subsequent attention hits
//!   on the same segment contribute progressively less. The first hit says "this is
//!   relevant now"; additional hits confirm the same thing.
//!
//! - **Pressure-sensitive parameters**: Decay rate and boost magnitudes adjust based on
//!   how full the context window is. This lets the model's behavior determine outcomes
//!   under pressure, rather than imposing external thresholds.
//!
//! - **Tag-based importance floors**: SystemPrompt segments have a high importance floor
//!   (effectively immortal until extreme pressure). Other tags have configurable weights.

use super::cache_span::SpanId;
use super::eviction_policy::EvictionPolicy;
use super::h2o_policy::H2OPolicy;
use super::segmented_cache_config::SegmentedCacheConfig;
use crate::cache::{CacheTag, SpanRegistry};
use std::collections::HashMap;

/// Per-span relevance score tracking.
#[derive(Debug, Clone)]
pub struct SpanScore {
    /// Cumulative boosted relevance score.
    /// Increases with attention, decreases with decay each pass.
    pub relevance: f32,

    /// Number of times this span was boosted in the current evaluation pass.
    /// Used for diminishing returns within a single pass.
    pub hits_this_pass: u32,

    /// Number of consecutive passes with zero attention received.
    /// Useful for diagnostics.
    pub passes_without_attention: u32,
}

impl SpanScore {
    fn new() -> Self {
        Self {
            relevance: 0.5, // Start at neutral importance
            hits_this_pass: 0,
            passes_without_attention: 0,
        }
    }
}

/// Segment-aware eviction policy.
///
/// Aggregates per-token attention data (from H2O) into per-span relevance scores,
/// applying superlinear boost, decay, and pressure-sensitive parameter adjustment.
#[derive(Debug, Clone)]
pub struct SegmentedEvictionPolicy {
    config: SegmentedCacheConfig,

    /// Per-span relevance scores, keyed by SpanId.
    span_scores: HashMap<SpanId, SpanScore>,

    /// EMA-smoothed cache utilization pressure (0.0 to 1.0+).
    smoothed_pressure: f32,

    /// Current evaluation pass counter.
    current_pass: u64,

    /// Cached mapping from cache position → SpanId.
    /// Rebuilt each time `update_scores()` is called.
    position_to_span: HashMap<usize, SpanId>,
}

impl SegmentedEvictionPolicy {
    /// Create a new segmented eviction policy.
    pub fn new(config: SegmentedCacheConfig) -> Self {
        Self {
            config,
            span_scores: HashMap::new(),
            smoothed_pressure: 0.0,
            current_pass: 0,
            position_to_span: HashMap::new(),
        }
    }

    /// Get the tag-based importance weight for a cache tag.
    ///
    /// Higher values mean the segment is harder to evict.
    /// SystemPrompt is nearly immortal; ModelGeneration is most expendable.
    fn tag_importance(tag: CacheTag) -> f32 {
        match tag {
            CacheTag::SystemPrompt => 1.0,
            CacheTag::ToolOutput => 0.8,
            CacheTag::LongTermMemory => 0.75,
            CacheTag::UserInput => 0.7,
            CacheTag::ModelGeneration => 0.5,
            CacheTag::Custom => 0.6,
        }
    }

    /// Update pressure using exponential moving average.
    ///
    /// Smoothing prevents sudden pressure spikes from causing cascade evictions.
    pub fn update_pressure(&mut self, current_utilization: f32) {
        let alpha = self.config.pressure_ema_alpha;
        self.smoothed_pressure =
            alpha * current_utilization + (1.0 - alpha) * self.smoothed_pressure;
    }

    /// Compute the pressure modifier that scales decay rate.
    ///
    /// Below threshold: modifier = 1.0 (normal decay).
    /// Above threshold: modifier increases linearly, up to 3.0 at 100% utilization.
    fn pressure_decay_modifier(&self) -> f32 {
        let threshold = self.config.eviction_pressure_threshold;
        if self.smoothed_pressure <= threshold {
            1.0
        } else {
            // Linear scale from 1.0 at threshold to 3.0 at 1.0 utilization
            let excess = (self.smoothed_pressure - threshold) / (1.0 - threshold).max(0.01);
            1.0 + excess * 2.0
        }
    }

    /// Update span relevance scores using H2O per-token attention data.
    ///
    /// This is the core scoring algorithm. For each active span:
    /// 1. Apply decay (modified by pressure)
    /// 2. Aggregate attention from H2O token metadata within the span's position range
    /// 3. Apply superlinear boost with diminishing returns
    ///
    /// # Arguments
    /// * `registry` - SpanRegistry containing active spans and their position ranges
    /// * `h2o` - H2OPolicy containing per-token cumulative attention data
    /// * `current_utilization` - Current cache utilization (0.0 to 1.0)
    pub fn update_scores(
        &mut self,
        registry: &SpanRegistry,
        h2o: &H2OPolicy,
        current_utilization: f32,
    ) {
        self.current_pass += 1;
        self.update_pressure(current_utilization);

        let decay_modifier = self.pressure_decay_modifier();
        let effective_decay = self.config.base_decay_rate * decay_modifier;
        let power = self.config.attention_power;

        // Rebuild position-to-span mapping and process each active span
        self.position_to_span.clear();

        // Collect span info first to avoid borrow issues
        let span_infos: Vec<(SpanId, usize, usize, usize, CacheTag)> = (0..usize::MAX)
            .map(|slot| {
                registry
                    .spans_for_slot(slot)
                    .into_iter()
                    .filter(|s| s.is_active())
                    .map(|s| (s.id, s.slot, s.start_pos, s.end_pos, s.tag))
                    .collect::<Vec<_>>()
            })
            // Stop when we get an empty slot (heuristic: no more slots)
            .take_while(|spans| !spans.is_empty())
            .flatten()
            .collect();

        // If no spans found with the take_while approach, try a direct approach
        // by iterating slots 0..reasonable_max
        let span_infos = if span_infos.is_empty() {
            let mut infos = Vec::new();
            for slot in 0..256 {
                // Reasonable max slots
                let spans = registry.spans_for_slot(slot);
                if spans.is_empty() && slot > 0 {
                    // Check a few more in case of gaps
                    let mut all_empty = true;
                    for s in slot + 1..slot + 4 {
                        if !registry.spans_for_slot(s).is_empty() {
                            all_empty = false;
                            break;
                        }
                    }
                    if all_empty {
                        break;
                    }
                }
                for span in spans {
                    if span.is_active() {
                        infos.push((span.id, span.slot, span.start_pos, span.end_pos, span.tag));
                    }
                }
            }
            infos
        } else {
            span_infos
        };

        // Track which spans we've seen this pass (to clean up stale entries)
        let mut seen_spans = Vec::with_capacity(span_infos.len());

        for (span_id, _slot, start_pos, end_pos, tag) in &span_infos {
            seen_spans.push(*span_id);

            // Build position-to-span mapping for this span
            for pos in *start_pos..*end_pos {
                self.position_to_span.insert(pos, *span_id);
            }

            // Get or create score entry
            let score = self
                .span_scores
                .entry(*span_id)
                .or_insert_with(SpanScore::new);

            // Reset per-pass hit counter
            score.hits_this_pass = 0;

            // Apply decay
            score.relevance = (score.relevance - effective_decay).max(0.0);

            // Compute max attention weight received by any token in this span
            let mut max_attention: f32 = 0.0;
            let mut any_attention = false;

            for pos in *start_pos..*end_pos {
                if let Some(metadata) = h2o.get_token_metadata(pos) {
                    if metadata.cumulative_attention > max_attention {
                        max_attention = metadata.cumulative_attention;
                    }
                    if metadata.cumulative_attention > 0.0 {
                        any_attention = true;
                    }
                }
            }

            if any_attention {
                // Superlinear boost: attention^power
                let raw_boost = max_attention.powf(power);

                // Diminishing returns: first hit is full, subsequent hits reduced
                let diminishing_factor = 0.9_f32.powi(score.hits_this_pass as i32);
                let effective_boost = raw_boost * diminishing_factor;

                // Tag-based importance scaling
                let tag_weight = Self::tag_importance(*tag);

                score.relevance += effective_boost * tag_weight;
                score.hits_this_pass += 1;
                score.passes_without_attention = 0;
            } else {
                score.passes_without_attention += 1;
            }
        }

        // Clean up scores for spans that no longer exist
        self.span_scores.retain(|id, _| seen_spans.contains(id));
    }

    /// Rank spans by eviction priority (lowest relevance first).
    ///
    /// Returns spans sorted from most-evictable to least-evictable.
    /// SystemPrompt spans are excluded (never evicted through normal scoring).
    ///
    /// # Returns
    ///
    /// Vector of `(SpanId, relevance_score)` sorted ascending by score.
    pub fn rank_spans_for_eviction(&self, registry: &SpanRegistry) -> Vec<(SpanId, f32)> {
        let mut candidates: Vec<(SpanId, f32)> = self
            .span_scores
            .iter()
            .filter_map(|(&span_id, score)| {
                // Check if span still exists and is active
                let span = registry.get(span_id)?;
                if !span.is_active() {
                    return None;
                }
                // SystemPrompt spans are protected
                if span.tag == CacheTag::SystemPrompt {
                    return None;
                }
                Some((span_id, score.relevance))
            })
            .collect();

        // Sort by relevance ascending (lowest = best eviction candidate)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates
    }

    /// Get the current smoothed pressure value.
    pub fn smoothed_pressure(&self) -> f32 {
        self.smoothed_pressure
    }

    /// Get the score for a specific span.
    pub fn get_span_score(&self, span_id: SpanId) -> Option<&SpanScore> {
        self.span_scores.get(&span_id)
    }

    /// Get the current pass count.
    pub fn current_pass(&self) -> u64 {
        self.current_pass
    }

    /// Get the position-to-span mapping (for EvictionPolicy trait bridge).
    pub fn position_to_span(&self) -> &HashMap<usize, SpanId> {
        &self.position_to_span
    }
}

impl EvictionPolicy for SegmentedEvictionPolicy {
    /// Bridge segment-level scores to per-token eviction scores.
    ///
    /// Each token inherits its parent span's eviction score.
    /// Tokens not belonging to any span get the highest eviction priority.
    /// Tokens in SystemPrompt spans get NEG_INFINITY (never evict).
    fn compute_eviction_scores(
        &self,
        _cache_size: usize,
        _num_used: usize,
        cache_positions: &HashMap<usize, usize>,
    ) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = cache_positions
            .iter()
            .map(|(&slot_id, &seq_position)| {
                // Look up which span this position belongs to
                if let Some(&span_id) = self.position_to_span.get(&seq_position) {
                    if let Some(span_score) = self.span_scores.get(&span_id) {
                        // Invert relevance for eviction score (lower relevance = higher eviction priority)
                        let eviction_score = if span_score.relevance > 0.0 {
                            1.0 / span_score.relevance
                        } else {
                            f32::INFINITY // Zero relevance = highest eviction priority
                        };
                        (slot_id, eviction_score)
                    } else {
                        // Span exists in position map but no score yet — high eviction priority
                        (slot_id, f32::INFINITY)
                    }
                } else {
                    // Position not in any span — highest eviction priority
                    (slot_id, f32::INFINITY)
                }
            })
            .collect();

        // Sort by eviction score descending (highest = evict first)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
    }

    fn name(&self) -> &str {
        "SegmentedEviction"
    }
}

/// Per-token attention tracker for hybrid eviction.
///
/// Activated on the lowest-scoring segment when it approaches eviction.
/// Tracks individual token attention scores within that segment, enabling
/// selective eviction of low-value tokens while preserving high-value ones.
///
/// Only one tracker can be active at a time (bounded compute budget).
#[derive(Debug, Clone)]
pub struct PerTokenTracker {
    /// The span being tracked at per-token granularity.
    pub target_span: SpanId,

    /// Per-position relevance scores within the span.
    /// Indexed by offset from `span_start_pos`.
    pub token_scores: Vec<f32>,

    /// Which positions have been evicted (true = evicted).
    /// Indexed by offset from `span_start_pos`.
    pub evicted_mask: Vec<bool>,

    /// Absolute start position of the tracked span.
    pub span_start_pos: usize,

    /// Configuration: superlinear boost exponent.
    attention_power: f32,

    /// Configuration: per-pass decay.
    decay_rate: f32,
}

impl PerTokenTracker {
    /// Create a new per-token tracker for a span.
    ///
    /// # Arguments
    ///
    /// * `span_id` - The span to track
    /// * `span_start_pos` - Absolute start position of the span
    /// * `span_len` - Number of tokens in the span
    /// * `attention_power` - Superlinear boost exponent
    /// * `decay_rate` - Per-pass decay rate
    pub fn new(
        span_id: SpanId,
        span_start_pos: usize,
        span_len: usize,
        attention_power: f32,
        decay_rate: f32,
    ) -> Self {
        Self {
            target_span: span_id,
            token_scores: vec![0.5; span_len], // Start neutral
            evicted_mask: vec![false; span_len],
            span_start_pos,
            attention_power,
            decay_rate,
        }
    }

    /// Update per-token scores using H2O attention data.
    ///
    /// Applies decay and superlinear boost to each token individually.
    pub fn update_scores(&mut self, h2o: &H2OPolicy, pressure_modifier: f32) {
        let effective_decay = self.decay_rate * pressure_modifier;

        for (offset, score) in self.token_scores.iter_mut().enumerate() {
            if self.evicted_mask[offset] {
                continue; // Skip already-evicted tokens
            }

            // Apply decay
            *score = (*score - effective_decay).max(0.0);

            // Check attention for this position
            let abs_pos = self.span_start_pos + offset;
            if let Some(metadata) = h2o.get_token_metadata(abs_pos) {
                if metadata.cumulative_attention > 0.0 {
                    let boost = metadata.cumulative_attention.powf(self.attention_power);
                    *score += boost;
                }
            }
        }
    }

    /// Select positions for eviction within the tracked span.
    ///
    /// Returns absolute positions (not offsets) of tokens to evict,
    /// sorted by score ascending (lowest-value first).
    ///
    /// # Arguments
    ///
    /// * `max_evictions` - Maximum number of tokens to evict
    /// * `score_threshold` - Only evict tokens with score below this value
    pub fn select_evictions(&self, max_evictions: usize, score_threshold: f32) -> Vec<usize> {
        let mut candidates: Vec<(usize, f32)> = self
            .token_scores
            .iter()
            .enumerate()
            .filter(|(offset, _)| !self.evicted_mask[*offset])
            .filter(|(_, score)| **score < score_threshold)
            .map(|(offset, &score)| (self.span_start_pos + offset, score))
            .collect();

        // Sort by score ascending (lowest first)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates
            .into_iter()
            .take(max_evictions)
            .map(|(pos, _)| pos)
            .collect()
    }

    /// Mark positions as evicted within this tracker.
    ///
    /// # Arguments
    ///
    /// * `absolute_positions` - Absolute positions to mark as evicted
    pub fn mark_evicted(&mut self, absolute_positions: &[usize]) {
        for &pos in absolute_positions {
            if pos >= self.span_start_pos {
                let offset = pos - self.span_start_pos;
                if offset < self.evicted_mask.len() {
                    self.evicted_mask[offset] = true;
                }
            }
        }
    }

    /// Count how many tokens are still active (not evicted).
    pub fn active_count(&self) -> usize {
        self.evicted_mask
            .iter()
            .filter(|&&evicted| !evicted)
            .count()
    }

    /// Count how many tokens have been evicted.
    pub fn evicted_count(&self) -> usize {
        self.evicted_mask.iter().filter(|&&evicted| evicted).count()
    }

    /// Get the remaining (non-evicted) position ranges.
    ///
    /// Used for span compaction — identifies contiguous runs of surviving tokens.
    pub fn remaining_ranges(&self) -> Vec<std::ops::Range<usize>> {
        let mut ranges = Vec::new();
        let mut in_range = false;
        let mut range_start = 0;

        for (offset, &evicted) in self.evicted_mask.iter().enumerate() {
            if !evicted && !in_range {
                range_start = self.span_start_pos + offset;
                in_range = true;
            } else if evicted && in_range {
                ranges.push(range_start..self.span_start_pos + offset);
                in_range = false;
            }
        }

        // Close final range if still open
        if in_range {
            ranges.push(range_start..self.span_start_pos + self.evicted_mask.len());
        }

        ranges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::h2o_policy::{H2OConfig, TokenMetadata};
    use crate::cache::{CacheSpan, SpanRegistry};

    fn make_config() -> SegmentedCacheConfig {
        SegmentedCacheConfig {
            enabled: true,
            attention_power: 2.0,
            base_decay_rate: 0.02,
            eviction_pressure_threshold: 0.80,
            pressure_ema_alpha: 0.3,
            ..Default::default()
        }
    }

    fn make_registry_with_spans() -> SpanRegistry {
        let mut registry = SpanRegistry::new();

        // System prompt: positions 0..10
        let span1 = CacheSpan::new(1, 0, 0, 10, CacheTag::SystemPrompt, Some("sys".into()));
        registry.register(span1).unwrap();

        // User input: positions 10..30
        let span2 = CacheSpan::new(2, 0, 10, 30, CacheTag::UserInput, None);
        registry.register(span2).unwrap();

        // Model generation: positions 30..50
        let span3 = CacheSpan::new(3, 0, 30, 50, CacheTag::ModelGeneration, None);
        registry.register(span3).unwrap();

        // Another user input: positions 50..70
        let span4 = CacheSpan::new(4, 0, 50, 70, CacheTag::UserInput, None);
        registry.register(span4).unwrap();

        registry
    }

    fn make_h2o_with_attention(attention_map: &[(usize, f32)]) -> H2OPolicy {
        let config = H2OConfig {
            enabled: true,
            num_recent_to_keep: 0,
            decay_factor: 1.0,
        };
        let mut h2o = H2OPolicy::new(config);

        // Simulate attention updates by feeding attention matrices
        // We'll use the internal metadata directly since H2O tracks by slot_id
        let mut cache_positions = HashMap::new();
        for &(pos, _) in attention_map {
            cache_positions.insert(pos, pos);
        }

        // Build attention weights: [1][max_pos+1] with specified values
        let max_pos = attention_map.iter().map(|&(p, _)| p).max().unwrap_or(0);
        let mut weights = vec![0.0_f32; max_pos + 1];
        for &(pos, attn) in attention_map {
            weights[pos] = attn;
        }

        h2o.update_attention_scores(&[weights], &cache_positions);
        h2o
    }

    #[test]
    fn test_new_policy_starts_neutral() {
        let policy = SegmentedEvictionPolicy::new(make_config());
        assert_eq!(policy.current_pass(), 0);
        assert_eq!(policy.smoothed_pressure(), 0.0);
        assert!(policy.span_scores.is_empty());
    }

    #[test]
    fn test_pressure_smoothing() {
        let mut policy = SegmentedEvictionPolicy::new(make_config());

        // Sudden jump to 0.9 utilization
        policy.update_pressure(0.9);
        // With alpha=0.3: smoothed = 0.3 * 0.9 + 0.7 * 0.0 = 0.27
        assert!((policy.smoothed_pressure() - 0.27).abs() < 0.001);

        // Another update at 0.9
        policy.update_pressure(0.9);
        // smoothed = 0.3 * 0.9 + 0.7 * 0.27 = 0.27 + 0.189 = 0.459
        assert!((policy.smoothed_pressure() - 0.459).abs() < 0.001);
    }

    #[test]
    fn test_pressure_modifier_below_threshold() {
        let mut policy = SegmentedEvictionPolicy::new(make_config());
        policy.smoothed_pressure = 0.5; // Below 0.80 threshold
        assert_eq!(policy.pressure_decay_modifier(), 1.0);
    }

    #[test]
    fn test_pressure_modifier_above_threshold() {
        let mut policy = SegmentedEvictionPolicy::new(make_config());
        policy.smoothed_pressure = 0.9; // Above 0.80 threshold
        let modifier = policy.pressure_decay_modifier();
        assert!(modifier > 1.0);
        assert!(modifier < 3.0);
    }

    #[test]
    fn test_superlinear_boost() {
        let config = make_config();
        let registry = make_registry_with_spans();

        // Give high attention to user input span (positions 10-30)
        // and low attention to model generation span (positions 30-50)
        let mut attention = Vec::new();
        for pos in 10..30 {
            attention.push((pos, 0.5)); // High attention
        }
        for pos in 30..50 {
            attention.push((pos, 0.05)); // Low attention
        }
        let h2o = make_h2o_with_attention(&attention);

        let mut policy = SegmentedEvictionPolicy::new(config);
        policy.update_scores(&registry, &h2o, 0.5);

        // User input span (id=2) should have much higher relevance than
        // model generation span (id=3) due to superlinear boost
        let score_user = policy.get_span_score(2).unwrap().relevance;
        let score_gen = policy.get_span_score(3).unwrap().relevance;

        assert!(
            score_user > score_gen,
            "User input score ({}) should exceed model generation score ({})",
            score_user,
            score_gen
        );
    }

    #[test]
    fn test_system_prompt_excluded_from_eviction() {
        let config = make_config();
        let registry = make_registry_with_spans();

        let h2o = make_h2o_with_attention(&[(15, 0.1)]); // Some attention

        let mut policy = SegmentedEvictionPolicy::new(config);
        policy.update_scores(&registry, &h2o, 0.5);

        let candidates = policy.rank_spans_for_eviction(&registry);

        // SystemPrompt span (id=1) should NOT appear in eviction candidates
        assert!(
            !candidates.iter().any(|(id, _)| *id == 1),
            "SystemPrompt should never be an eviction candidate"
        );
    }

    #[test]
    fn test_decay_reduces_relevance() {
        let config = make_config();
        let registry = make_registry_with_spans();

        // Give attention to span 2 on first pass
        let h2o1 = make_h2o_with_attention(&[(15, 0.5)]);
        let mut policy = SegmentedEvictionPolicy::new(config.clone());
        policy.update_scores(&registry, &h2o1, 0.5);

        let score_after_boost = policy.get_span_score(2).unwrap().relevance;

        // Now run several passes with zero attention
        let h2o_empty = make_h2o_with_attention(&[]);
        for _ in 0..10 {
            policy.update_scores(&registry, &h2o_empty, 0.5);
        }

        let score_after_decay = policy.get_span_score(2).unwrap().relevance;

        assert!(
            score_after_decay < score_after_boost,
            "Score should decrease after passes without attention: {} vs {}",
            score_after_decay,
            score_after_boost
        );
    }

    #[test]
    fn test_eviction_policy_trait() {
        let config = make_config();
        let registry = make_registry_with_spans();

        let h2o = make_h2o_with_attention(&[(15, 0.5), (35, 0.01)]);

        let mut policy = SegmentedEvictionPolicy::new(config);
        policy.update_scores(&registry, &h2o, 0.5);

        // Call EvictionPolicy trait method
        let mut cache_positions = HashMap::new();
        for pos in 0..70 {
            cache_positions.insert(pos, pos);
        }

        let scores = policy.compute_eviction_scores(70, 70, &cache_positions);
        assert!(!scores.is_empty());

        // First entry should be the highest eviction priority
        // (positions in the model generation span with low attention)
    }

    #[test]
    fn test_tag_importance_ordering() {
        // SystemPrompt should be hardest to evict
        assert!(
            SegmentedEvictionPolicy::tag_importance(CacheTag::SystemPrompt)
                > SegmentedEvictionPolicy::tag_importance(CacheTag::UserInput)
        );
        assert!(
            SegmentedEvictionPolicy::tag_importance(CacheTag::UserInput)
                > SegmentedEvictionPolicy::tag_importance(CacheTag::ModelGeneration)
        );
    }

    // === PerTokenTracker tests ===

    #[test]
    fn test_per_token_tracker_creation() {
        let tracker = PerTokenTracker::new(1, 10, 20, 2.0, 0.02);
        assert_eq!(tracker.target_span, 1);
        assert_eq!(tracker.span_start_pos, 10);
        assert_eq!(tracker.token_scores.len(), 20);
        assert_eq!(tracker.evicted_mask.len(), 20);
        assert_eq!(tracker.active_count(), 20);
        assert_eq!(tracker.evicted_count(), 0);
    }

    #[test]
    fn test_per_token_select_evictions() {
        let mut tracker = PerTokenTracker::new(1, 10, 10, 2.0, 0.02);

        // Manually set scores: some high, some low
        tracker.token_scores = vec![0.1, 0.9, 0.05, 0.8, 0.02, 0.7, 0.03, 0.6, 0.01, 0.5];

        // Select up to 3 tokens below threshold 0.1
        let evictions = tracker.select_evictions(3, 0.1);

        // Should pick positions 18 (0.01), 14 (0.02), 16 (0.03) — the 3 lowest
        assert_eq!(evictions.len(), 3);
        assert_eq!(evictions[0], 18); // score 0.01
        assert_eq!(evictions[1], 14); // score 0.02
        assert_eq!(evictions[2], 16); // score 0.03
    }

    #[test]
    fn test_per_token_mark_evicted() {
        let mut tracker = PerTokenTracker::new(1, 10, 10, 2.0, 0.02);

        tracker.mark_evicted(&[12, 15, 18]);

        assert_eq!(tracker.evicted_count(), 3);
        assert_eq!(tracker.active_count(), 7);
        assert!(tracker.evicted_mask[2]); // pos 12 - offset 2
        assert!(tracker.evicted_mask[5]); // pos 15 - offset 5
        assert!(tracker.evicted_mask[8]); // pos 18 - offset 8
    }

    #[test]
    fn test_per_token_remaining_ranges() {
        let mut tracker = PerTokenTracker::new(1, 10, 10, 2.0, 0.02);

        // Evict positions 12, 13, 16
        tracker.mark_evicted(&[12, 13, 16]);

        let ranges = tracker.remaining_ranges();

        // Expected: [10..12, 14..16, 17..20]
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], 10..12);
        assert_eq!(ranges[1], 14..16);
        assert_eq!(ranges[2], 17..20);
    }

    #[test]
    fn test_per_token_remaining_ranges_all_active() {
        let tracker = PerTokenTracker::new(1, 10, 5, 2.0, 0.02);
        let ranges = tracker.remaining_ranges();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], 10..15);
    }

    #[test]
    fn test_per_token_remaining_ranges_all_evicted() {
        let mut tracker = PerTokenTracker::new(1, 10, 3, 2.0, 0.02);
        tracker.mark_evicted(&[10, 11, 12]);
        let ranges = tracker.remaining_ranges();
        assert!(ranges.is_empty());
    }

    #[test]
    fn test_per_token_skips_evicted_in_selection() {
        let mut tracker = PerTokenTracker::new(1, 10, 5, 2.0, 0.02);
        tracker.token_scores = vec![0.01, 0.02, 0.03, 0.04, 0.05];

        // Evict the lowest first
        tracker.mark_evicted(&[10]); // offset 0, score 0.01

        // Now select — should skip the already-evicted position
        let evictions = tracker.select_evictions(2, 0.1);
        assert_eq!(evictions.len(), 2);
        assert_eq!(evictions[0], 11); // score 0.02 (not 10, which is evicted)
        assert_eq!(evictions[1], 12); // score 0.03
    }
}
