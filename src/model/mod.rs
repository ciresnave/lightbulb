//! Batched model components for true batched inference
//!
//! This module contains custom wrappers and implementations that enable
//! true batched inference by bypassing Candle's standard sequential API.

pub mod awq_qwen3; // AWQ-quantized Qwen3 model with Marlin kernels
pub mod batch_metadata;
pub mod batched_llama_wrapper; // Legacy reference (unused)
pub mod chunked_prefill; // Chunked prefill scheduler with padding
pub mod custom_attention; // Parallel batched attention
pub mod custom_transformer;
pub mod custom_transformer_block;
pub mod decode_state; // M3.2: Decode-loop optimization state
pub mod fused_kernels; // M3.3: CPU kernel fusion for performance
pub mod fused_rmsnorm; // M4.1: Fused RMSNorm with CUDA acceleration
pub mod kv_tensor; // Canonical KV tensor representation for ScatteredKvCache
pub mod mlp_wrapper; // Thin wrapper around Candle's MLP components
pub mod parallel_model_manager; // Production: Parallel batching with BatchedTransformer
pub mod quantizable_linear; // Unified linear layer supporting both regular and quantized weights
pub mod speculative_adapters; // Adapters for speculative decoding

pub use batch_metadata::BatchMetadata;
pub use batched_llama_wrapper::BatchedLlamaWrapper; // Legacy
pub use chunked_prefill::{
    ChunkedBatch, ChunkedPrefillConfig, ChunkedPrefillScheduler, PrefillRequest,
};
pub use custom_attention::BatchedAttention;
pub use custom_transformer::{
    BatchedGemma, BatchedLlama, BatchedMistral, BatchedTransformer, BatchedTransformerConfig,
};
pub use custom_transformer_block::BatchedTransformerBlock;
pub use kv_tensor::KVTensor;
pub use parallel_model_manager::{ParallelBatchStats, ParallelModelManager};
pub use speculative_adapters::BatchedTransformerAdapter;

// Re-export AWQ model
pub use awq_qwen3::{AwqQwen3, Qwen3Config};
