//! Batched model components for true batched inference
//!
//! This module contains custom wrappers and implementations that enable
//! true batched inference by bypassing Candle's standard sequential API.

pub mod batch_manager; // Model-agnostic batch manager (NEW)
pub mod batch_metadata;
pub mod batched_llama_wrapper; // Legacy - use batch_manager instead
pub mod custom_attention; // Phase 2D - Approach 2: Parallel batched attention
pub mod custom_transformer;
pub mod custom_transformer_block;
pub mod mlp_wrapper; // Thin wrapper around Candle's MLP components
pub mod model_manager; // Phase 2D - Approach 2: Generic transformer

pub use batch_manager::{BatchManager, TransformerModel};
pub use batch_metadata::BatchMetadata;
pub use batched_llama_wrapper::BatchedLlamaWrapper; // Legacy
pub use custom_attention::BatchedAttention;
pub use custom_transformer::{
    BatchedGemma, BatchedLlama, BatchedMistral, BatchedTransformer, BatchedTransformerConfig,
};
pub use custom_transformer_block::BatchedTransformerBlock;
pub use model_manager::{BatchStats, ModelManager};
