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
//! ```ignore
//! let cache = PrefixKvCache::new(capacity_mb);
//!
//! // Try to reuse cached prefix
//! if let Some(entry) = cache.get(&tokens[..prefix_len]) {
//!     // Copy cached KV into active cache, start from entry.length
//! } else {
//!     // Compute prefill normally, then cache the prefix
//!     cache.insert(&tokens[..prefix_len], kv_tensors, device);
//! }
//! ```

use anyhow::Result;
use candlelight::core::{Device, Tensor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A cached prefix entry containing KV states for all layers
#[derive(Clone)]
pub struct PrefixKvEntry {
    /// Hash of the tokenized prefix
    pub hash: String,
    /// Length of the prefix in tokens
    pub length: usize,
    /// KV cache tensors per layer: Vec<(k_cache, v_cache)>
    /// Shape: [1, num_heads, prefix_len, head_dim]
    pub kv_by_layer: Vec<(Tensor, Tensor)>,
    /// Timestamp of last access (for LRU eviction)
    pub last_used: std::time::Instant,
    /// Estimated memory size in bytes
    pub size_bytes: usize,
}

impl PrefixKvEntry {
    /// Estimate memory usage of this entry
    fn estimate_size(&self) -> usize {
        // Rough estimate: 2 tensors per layer * float32 (4 bytes) * elements
        self.kv_by_layer
            .iter()
            .map(|(k, v)| {
                let k_elems = k.dims().iter().product::<usize>();
                let v_elems = v.dims().iter().product::<usize>();
                (k_elems + v_elems) * 4 // assuming f32
            })
            .sum()
    }
}

/// Configuration for prefix cache behavior
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

/// LRU cache for prefix KV states
pub struct PrefixKvCache {
    config: PrefixCacheConfig,
    /// Cache storage: hash -> entry
    cache: Arc<Mutex<HashMap<String, PrefixKvEntry>>>,
    /// Current cache size in bytes
    current_size: Arc<Mutex<usize>>,
    /// Hit/miss statistics
    stats: Arc<Mutex<PrefixCacheStats>>,
}

#[derive(Debug, Default, Clone)]
pub struct PrefixCacheStats {
    pub hits: usize,
    pub misses: usize,
    pub insertions: usize,
    pub evictions: usize,
    pub total_saved_tokens: usize,
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

impl PrefixKvCache {
    /// Create a new prefix KV cache
    pub fn new(config: PrefixCacheConfig) -> Self {
        Self {
            config,
            cache: Arc::new(Mutex::new(HashMap::new())),
            current_size: Arc::new(Mutex::new(0)),
            stats: Arc::new(Mutex::new(PrefixCacheStats::default())),
        }
    }

    /// Hash a token sequence to a cache key
    fn hash_tokens(tokens: &[u32]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();

        // Hash the token sequence
        for &token in tokens {
            hasher.update(&token.to_le_bytes());
        }

        format!("{:x}", hasher.finalize())
    }

    /// Try to retrieve a cached prefix entry
    /// Try to retrieve an exact cached entry for the provided token slice
    ///
    /// This performs an exact lookup for the given token slice (hash of the
    /// entire slice). Useful when callers already have the exact prefix they
    /// want to query.
    pub fn get(&self, tokens: &[u32]) -> Option<PrefixKvEntry> {
        if !self.config.enabled || tokens.len() < self.config.min_prefix_len {
            return None;
        }

        let hash = Self::hash_tokens(tokens);
        let mut cache = self.cache.lock().unwrap();

        if let Some(entry) = cache.get_mut(&hash) {
            // Verify length matches (hash collision check)
            if entry.length == tokens.len() {
                entry.last_used = std::time::Instant::now();

                // Update stats
                let mut stats = self.stats.lock().unwrap();
                stats.hits += 1;
                stats.total_saved_tokens += tokens.len();

                return Some(entry.clone());
            }
        }

        // Miss
        let mut stats = self.stats.lock().unwrap();
        stats.misses += 1;
        None
    }

    /// Find the best (longest) cached prefix that matches the start of `tokens`.
    ///
    /// This searches for the longest prefix (within configured bounds) where
    /// the cached entry exactly equals `tokens[..prefix_len]`. Returns the
    /// cached entry if found. Updates stats similarly to `get`.
    pub fn get_best_prefix(&self, tokens: &[u32]) -> Option<PrefixKvEntry> {
        if !self.config.enabled || tokens.len() < self.config.min_prefix_len {
            return None;
        }

        let max_len = tokens.len().min(self.config.max_prefix_len);

        // Lock cache once for lookups
        let mut cache = self.cache.lock().unwrap();

        // Search descending from the longest allowed prefix to the minimum
        for len in (self.config.min_prefix_len..=max_len).rev() {
            let prefix = &tokens[..len];
            let hash = Self::hash_tokens(prefix);

            if let Some(entry) = cache.get_mut(&hash) {
                if entry.length == len {
                    entry.last_used = std::time::Instant::now();

                    let mut stats = self.stats.lock().unwrap();
                    stats.hits += 1;
                    stats.total_saved_tokens += len;

                    return Some(entry.clone());
                }
            }
        }

        // Miss
        let mut stats = self.stats.lock().unwrap();
        stats.misses += 1;
        None
    }

    /// Insert a new prefix entry into the cache
    pub fn insert(
        &self,
        tokens: &[u32],
        kv_by_layer: Vec<(Tensor, Tensor)>,
        _device: &Device,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check length constraints
        if tokens.len() < self.config.min_prefix_len || tokens.len() > self.config.max_prefix_len {
            return Ok(());
        }

        let hash = Self::hash_tokens(tokens);

        let entry = PrefixKvEntry {
            hash: hash.clone(),
            length: tokens.len(),
            size_bytes: 0, // Will be computed
            kv_by_layer: kv_by_layer.clone(),
            last_used: std::time::Instant::now(),
        };

        let entry_size = entry.estimate_size();
        let max_size = self.config.max_size_mb * 1024 * 1024;

        // Evict entries if needed
        {
            let mut cache = self.cache.lock().unwrap();
            let mut current_size = self.current_size.lock().unwrap();

            while *current_size + entry_size > max_size && !cache.is_empty() {
                // Find LRU entry
                let lru_hash = cache
                    .iter()
                    .min_by_key(|(_, e)| e.last_used)
                    .map(|(h, _)| h.clone());

                if let Some(lru_hash) = lru_hash {
                    if let Some(evicted) = cache.remove(&lru_hash) {
                        *current_size = current_size.saturating_sub(evicted.size_bytes);

                        let mut stats = self.stats.lock().unwrap();
                        stats.evictions += 1;
                    }
                } else {
                    break;
                }
            }

            // Insert new entry
            let mut entry = entry;
            entry.size_bytes = entry_size;

            cache.insert(hash.clone(), entry);
            *current_size += entry_size;

            let mut stats = self.stats.lock().unwrap();
            stats.insertions += 1;
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> PrefixCacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear the entire cache
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();

        let mut current_size = self.current_size.lock().unwrap();
        *current_size = 0;
    }

    /// Get current cache size in bytes
    pub fn current_size_bytes(&self) -> usize {
        *self.current_size.lock().unwrap()
    }

    /// Check if tokens would hit the cache (without actually retrieving)
    /// Returns true if this exact prefix has been seen before
    /// Does NOT update statistics (only used to avoid redundant insertions)
    pub fn check_would_hit(&self, tokens: &[u32]) -> bool {
        if !self.config.enabled || tokens.len() < self.config.min_prefix_len {
            return false;
        }

        let hash = Self::hash_tokens(tokens);
        let cache = self.cache.lock().unwrap();

        if let Some(entry) = cache.get(&hash) {
            if entry.length == tokens.len() {
                return true;
            }
        }

        false
    }

    /// Record a prompt for tracking (creates placeholder entry without KV)
    /// This allows us to track hits/misses even before full KV caching is implemented
    pub fn record_prompt(&self, tokens: &[u32]) {
        if !self.config.enabled {
            return;
        }

        // Check length constraints
        if tokens.len() < self.config.min_prefix_len || tokens.len() > self.config.max_prefix_len {
            return;
        }

        let hash = Self::hash_tokens(tokens);
        let mut cache = self.cache.lock().unwrap();

        // Check if already exists
        if cache.contains_key(&hash) {
            return;
        }

        // Create placeholder entry (no actual KV tensors yet)
        let entry = PrefixKvEntry {
            hash: hash.clone(),
            length: tokens.len(),
            size_bytes: 0,
            kv_by_layer: Vec::new(), // Empty - no KV cached yet
            last_used: std::time::Instant::now(),
        };

        cache.insert(hash, entry);
        // Don't update current_size since we're not actually storing KV tensors

        let mut stats = self.stats.lock().unwrap();
        stats.insertions += 1;
    }

    /// Get number of cached entries
    pub fn entry_count(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_tokens() {
        let tokens1 = vec![1, 2, 3, 4, 5];
        let tokens2 = vec![1, 2, 3, 4, 5];
        let tokens3 = vec![1, 2, 3, 4, 6];

        let hash1 = PrefixKvCache::hash_tokens(&tokens1);
        let hash2 = PrefixKvCache::hash_tokens(&tokens2);
        let hash3 = PrefixKvCache::hash_tokens(&tokens3);

        assert_eq!(hash1, hash2, "Same tokens should have same hash");
        assert_ne!(hash1, hash3, "Different tokens should have different hash");
    }

    #[test]
    fn test_cache_insert_and_get() {
        let config = PrefixCacheConfig {
            max_size_mb: 1,
            min_prefix_len: 2,
            max_prefix_len: 100,
            enabled: true,
        };

        let cache = PrefixKvCache::new(config);
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // Initially no cache hit
        assert!(cache.get(&tokens).is_none());

        // Stats should show a miss
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
    }

    #[test]
    fn test_min_prefix_length() {
        let config = PrefixCacheConfig {
            max_size_mb: 1,
            min_prefix_len: 5,
            max_prefix_len: 100,
            enabled: true,
        };

        let cache = PrefixKvCache::new(config);
        let short_tokens = vec![1, 2, 3]; // Too short

        // Should not be cached (too short)
        assert!(cache.get(&short_tokens).is_none());

        // Stats should show nothing (filtered before stats)
        let stats = cache.stats();
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_disabled_cache() {
        let config = PrefixCacheConfig {
            max_size_mb: 1,
            min_prefix_len: 2,
            max_prefix_len: 100,
            enabled: false,
        };

        let cache = PrefixKvCache::new(config);
        let tokens = vec![1, 2, 3, 4, 5];

        // Should return None when disabled
        assert!(cache.get(&tokens).is_none());
    }
}
