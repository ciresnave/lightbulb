//! KV Cache implementations for parallel batching
//!
//! This module itself contains no third-party code. The fork it used to
//! describe is `parallel_cache_builder`, which carried its own upstream
//! copyright and SPDX header — that file (and `kv_compression`/
//! `tensor_codec`, which existed only to support it) was removed 2026-10-22
//! (candlelight removal, CireSnave: "Remove all dependencies on Candle and
//! Candlelight..."). Its only production caller, `ParallelModelManager`, was
//! removed in the same pass; `tests/parallel_kv_cache.rs`'s own doc comment
//! already named Fuel's `kv_block_pool` (driven by `model_fuel::policies`)
//! as "a strict superset" of what it tested. See
//! `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`.

pub mod cache_span;
pub mod eviction_policy;
pub mod h2o_policy;
pub mod prefix_cache;
pub mod segmented_cache_config;
pub mod segmented_eviction_policy;
pub mod streaming_policy;
pub mod tiered_storage;

pub use cache_span::{
    CacheSpan, CacheTag, EvictionImpact, EvictionResult, SpanId, SpanRegistry, SpanState,
};
pub use eviction_policy::{EvictionPolicy, RecencyPolicy, VotingAggregator};
pub use h2o_policy::{H2OConfig, H2OPolicy, TokenMetadata};
pub use prefix_cache::{PrefixCacheConfig, PrefixCacheStats};
pub use segmented_cache_config::SegmentedCacheConfig;
pub use segmented_eviction_policy::{PerTokenTracker, SegmentedEvictionPolicy, SpanScore};
pub use streaming_policy::{StreamingConfig, compute_streaming_index, compute_streaming_indices};
