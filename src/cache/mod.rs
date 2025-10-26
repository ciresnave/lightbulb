//! KV Cache implementations for parallel batching
//!
//! Forked from Candle's ScatteredCacheBuilder with modifications for:
//! - Independent per-slot position tracking (no global increment)
//! - Slot clearing capability
//! - Efficient position setting (O(1) instead of O(n))

pub mod cache_span;
pub mod eviction_policy;
pub mod h2o_policy;
pub mod parallel_cache_builder;
pub mod prefix_cache;
pub mod streaming_policy;

pub use cache_span::{
    CacheSpan, CacheTag, EvictionImpact, EvictionResult, SpanId, SpanRegistry, SpanState,
};
pub use eviction_policy::{EvictionPolicy, RecencyPolicy, VotingAggregator};
pub use h2o_policy::{H2OConfig, H2OPolicy, TokenMetadata};
pub use parallel_cache_builder::{
    CacheUsageInfo, IndicesAndMask, ParallelCacheBuilder, ParallelKvCache,
};
pub use prefix_cache::{PrefixCacheConfig, PrefixCacheStats, PrefixKvCache, PrefixKvEntry};
pub use streaming_policy::{StreamingConfig, compute_streaming_index, compute_streaming_indices};
