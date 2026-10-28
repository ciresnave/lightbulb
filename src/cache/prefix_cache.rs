// SPDX-License-Identifier: MIT OR Apache-2.0
//! Prefix KV Cache
//!
//! Implements hash-and-reuse caching for common prompt prefixes.
//! When multiple requests share the same prefix (e.g., system prompts,
//! instruction templates), we can cache the KV states from prefill and
//! reuse them, avoiding redundant computation.
//!
//! # Benefits
//! - Reduces TTFT (Time To First Token) by 15-50% for requests with shared prefixes
//! - Lower compute costs for batched requests with common instructions
//! - Enables efficient few-shot prompting with cached examples
//!
//! # Design
//! - Hash: SHA256 of tokenized prefix (u32 tokens -> deterministic hash)
//! - Storage: LRU cache of (hash -> PrefixKvEntry)
//! - Per-layer KV tensors stored as Candle tensors
//! - Thread-safe via Arc<Mutex<>> for concurrent access
//!
//! # Usage
//!
//! The Tensor-typed cache implementation (`PrefixKvEntry`/`PrefixKvCache`) was
//! removed 2026-10-22 -- see the note at the bottom of this file. Only the
//! plain config/stats types below survive.

#[derive(Debug, Clone)]
pub struct PrefixCacheConfig {
    /// Maximum cache size in megabytes
    pub max_size_mb: usize,
    /// Minimum prefix length to consider caching (avoid tiny prefixes)
    pub min_prefix_len: usize,
    /// Maximum prefix length to cache (avoid huge prefixes)
    pub max_prefix_len: usize,
    /// Whether to enable prefix caching (feature flag)
    pub enabled: bool,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_size_mb: 512,    // 512MB default cache
            min_prefix_len: 8,   // At least 8 tokens
            max_prefix_len: 256, // Up to 256 tokens
            enabled: true,
        }
    }
}
#[derive(Debug, Default, Clone)]
pub struct PrefixCacheStats {
    pub hits: usize,
    pub misses: usize,
    pub insertions: usize,
    pub evictions: usize,
    pub total_saved_tokens: usize,
    /// Entries currently held, and their total size in bytes. Was populated
    /// by the now-deleted `PrefixKvCache::stats` from `entry_count()`/
    /// `current_size_bytes()` at snapshot time — these two were tracked in
    /// separate locks from the hit/miss counters above, so this struct was a
    /// composed snapshot, not a single atomic read. Nothing writes this pair
    /// today; see `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`.
    pub entries: usize,
    pub size_bytes: usize,
}

impl PrefixCacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    pub fn avg_saved_tokens(&self) -> f64 {
        if self.hits == 0 {
            0.0
        } else {
            self.total_saved_tokens as f64 / self.hits as f64
        }
    }
}

// PrefixKvEntry/PrefixKvCache (the candlelight-Tensor-typed cache itself) removed
// 2026-10-22 (candlelight removal, CireSnave: "Remove all dependencies on Candle and
// Candlelight..."): model_fuel/policies.rs's own doc comment already called this
// implementation "candle-typed AND the wrong model" for Fuel's prefix-matching design.
// PrefixCacheConfig/PrefixCacheStats (plain, never Tensor-typed) survive --
// SharedCacheStats (engine/model_runner.rs) still holds a PrefixCacheStats.
// See docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md.
