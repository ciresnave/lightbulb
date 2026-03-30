//! KV Cache implementations for parallel batching
//!
//! Forked from Candle's ScatteredCacheBuilder with modifications for:
//! - Independent per-slot position tracking (no global increment)
//! - Slot clearing capability
//! - Efficient position setting (O(1) instead of O(n))

pub mod cache_span;
pub mod eviction_policy;
pub mod h2o_policy;
pub mod kv_compression;
pub mod parallel_cache_builder;
pub mod prefix_cache;
pub mod segmented_cache_config;
pub mod segmented_eviction_policy;
pub mod streaming_policy;
pub mod tensor_codec;
pub mod tiered_storage;

pub use cache_span::{
    CacheSpan, CacheTag, EvictionImpact, EvictionResult, SpanId, SpanRegistry, SpanState,
};
pub use eviction_policy::{EvictionPolicy, RecencyPolicy, VotingAggregator};
pub use h2o_policy::{H2OConfig, H2OPolicy, TokenMetadata};
pub use kv_compression::{
    CompressionCtx, CompressionPolicy, KiviConfig, KiviQuantizer, KvCompressor, LowRankCompressor,
    LowRankConfig, QuantGranularity, QuantScales, RelationshipAwareConfig,
    RelationshipAwareEviction, RkvConfig, RkvScorer,
};
pub use parallel_cache_builder::{
    CacheUsageInfo, IndicesAndMask, ParallelCacheBuilder, ParallelKvCache,
};
pub use prefix_cache::{PrefixCacheConfig, PrefixCacheStats, PrefixKvCache, PrefixKvEntry};
pub use segmented_cache_config::SegmentedCacheConfig;
pub use segmented_eviction_policy::{PerTokenTracker, SegmentedEvictionPolicy, SpanScore};
pub use streaming_policy::{StreamingConfig, compute_streaming_index, compute_streaming_indices};
pub use tiered_storage::{DemotedSegment, StorageTier, TieredStorageManager};
