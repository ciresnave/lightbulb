/// Generic batched transformer model supporting multiple architectures (Llama, Mistral, Gemma, etc.)
///
/// This implementation provides a unified architecture that can be configured to support
/// various transformer-based language models through different configuration constructors.
use candle_core::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm, VarBuilder, embedding};
use std::collections::HashMap;

use crate::engine::BatchExecutor;
use crate::model::batch_metadata::BatchMetadata;
use crate::model::custom_transformer_block::BatchedTransformerBlock;

/// Configuration for the generic batched transformer model
#[derive(Debug, Clone)]
pub struct BatchedTransformerConfig {
    /// Size of the vocabulary
    pub vocab_size: usize,

    /// Hidden dimension size
    pub hidden_size: usize,

    /// Number of transformer layers
    pub num_hidden_layers: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of key-value heads (for Grouped Query Attention)
    /// If equal to num_attention_heads, uses Multi-Head Attention
    pub num_key_value_heads: usize,

    /// Size of the intermediate MLP layer
    pub intermediate_size: usize,

    /// Maximum sequence length for positional embeddings
    pub max_position_embeddings: usize,

    /// Epsilon for RMS normalization
    pub rms_norm_eps: f64,

    /// Base frequency for RoPE embeddings
    pub rope_theta: f32,

    /// Optional rope scaling configuration for extended context
    pub rope_scaling: Option<HashMap<String, f32>>,

    /// Optional sliding window size (for Mistral-style attention)
    pub sliding_window: Option<usize>,

    /// Whether to use flash attention (future feature)
    pub use_flash_attn: bool,

    /// Whether to tie word embeddings with output layer
    pub tie_word_embeddings: bool,
}

impl BatchedTransformerConfig {
    /// Create config from Llama model configuration
    pub fn from_llama(
        vocab_size: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        rms_norm_eps: f64,
        rope_theta: f32,
        max_position_embeddings: usize,
        tie_word_embeddings: bool,
    ) -> Self {
        Self {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            rope_scaling: None,
            sliding_window: None,
            use_flash_attn: false,
            tie_word_embeddings,
        }
    }

    /// Create config for Llama 7B v0.1
    pub fn llama_7b() -> Self {
        Self::from_llama(
            32000,   // vocab_size
            4096,    // hidden_size
            11008,   // intermediate_size
            32,      // num_hidden_layers
            32,      // num_attention_heads
            32,      // num_key_value_heads (MHA)
            1e-5,    // rms_norm_eps
            10000.0, // rope_theta
            4096,    // max_position_embeddings
            false,   // tie_word_embeddings
        )
    }

    /// Create config for Llama 2 7B
    pub fn llama2_7b() -> Self {
        Self::from_llama(32000, 4096, 11008, 32, 32, 32, 1e-5, 10000.0, 4096, false)
    }

    /// Create config for Llama 3 8B (with GQA)
    pub fn llama3_8b() -> Self {
        Self::from_llama(
            128256, // vocab_size (larger for Llama 3)
            4096, 14336, 32, 32, 8, // num_key_value_heads (GQA with 8 KV heads)
            1e-5, 500000.0, // rope_theta (increased for Llama 3)
            8192,     // max_position_embeddings (longer context)
            false,
        )
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            candle_core::bail!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size,
                self.num_attention_heads
            );
        }

        if self.num_attention_heads % self.num_key_value_heads != 0 {
            candle_core::bail!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads,
                self.num_key_value_heads
            );
        }

        if self.num_hidden_layers == 0 {
            candle_core::bail!("num_hidden_layers must be > 0");
        }
        Ok(())
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Generic batched transformer model
///
/// Supports various decoder-only transformer architectures through configuration:
/// - Llama, Llama2, Llama3
/// - Mistral (with sliding window attention)
/// - Gemma
/// - Qwen
/// - Phi
/// - And other similar architectures
pub struct BatchedTransformer {
    /// Token embedding layer
    embedding: Embedding,

    /// Stack of transformer blocks
    blocks: Vec<BatchedTransformerBlock>,

    /// Final normalization layer
    norm: RmsNorm,

    /// Language model head (vocabulary projection)
    lm_head: Linear,

    /// Precomputed cosine values for RoPE [max_seq_len, head_dim]
    cos: Tensor,

    /// Precomputed sine values for RoPE [max_seq_len, head_dim]
    sin: Tensor,

    /// Model configuration
    config: BatchedTransformerConfig,

    /// Device (CPU/CUDA)
    device: Device,

    /// Data type (F32/F16/BF16)
    dtype: DType,
}

impl BatchedTransformer {
    /// Create a new batched transformer model
    pub fn new(config: BatchedTransformerConfig, vb: VarBuilder) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        let device = vb.device().clone();
        let dtype = vb.dtype();

        // Create token embeddings
        let embedding = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        // Precompute RoPE frequencies
        let head_dim = config.head_dim();
        let (cos, sin) = Self::precompute_rope_frequencies(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            &device,
            dtype,
        )?;

        // Create transformer blocks
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        let vb_blocks = vb.pp("model.layers");

        for layer_idx in 0..config.num_hidden_layers {
            let block = BatchedTransformerBlock::new(
                layer_idx,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.intermediate_size,
                config.rms_norm_eps,
                vb_blocks.pp(layer_idx),
            )?;
            blocks.push(block);
        }

        // Final normalization
        let norm =
            candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;

        // Language model head
        let lm_head = if config.tie_word_embeddings {
            // Share weights with embedding layer - create a new Linear using the same weights
            let embedding_weight = vb.get(
                (config.vocab_size, config.hidden_size),
                "model.embed_tokens.weight",
            )?;

            // DEBUG: Verify weight shapes
            eprintln!(
                "DEBUG: LM head weight shape (tied): {:?}",
                embedding_weight.dims()
            );
            let emb_w_vec = embedding_weight.flatten_all()?.to_vec1::<f32>()?;
            let emb_w_mean: f32 = emb_w_vec.iter().sum::<f32>() / emb_w_vec.len() as f32;
            eprintln!(
                "DEBUG: LM head weight mean: {:.6}, sample[0:3]={:?}",
                emb_w_mean,
                &emb_w_vec[0..3]
            );

            candle_nn::Linear::new(embedding_weight.clone(), None)
        } else {
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embedding,
            blocks,
            norm,
            lm_head,
            cos,
            sin,
            config,
            device,
            dtype,
        })
    }

    /// Precompute RoPE frequency tensors
    ///
    /// Creates cos and sin tensors of shape [max_seq_len, head_dim] for rotary embeddings
    fn precompute_rope_frequencies(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f32,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        // Compute inverse frequencies: 1 / (theta^(2i/d)) for i in [0, d/2)
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / rope_theta.powf(i as f32 / head_dim as f32))
            .collect();

        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;

        // Create position indices [0, 1, 2, ..., max_seq_len-1]
        let positions = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;

        // Compute frequencies: position * inv_freq
        // Shape: [max_seq_len, head_dim/2]
        let freqs = positions.matmul(&inv_freq)?;

        // IMPORTANT: Unlike some models (Mixtral, Yi, etc), Llama does NOT duplicate
        // frequencies. The cos/sin tensors remain at half dimension [max_seq_len, head_dim/2].
        // See: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/llama.rs#L204

        // Compute cos and sin
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok((cos, sin))
    }

    /// Forward pass through the transformer
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs tensor of shape [total_tokens]
    /// * `batch_executor` - Manages KV cache across all layers
    /// * `metadata` - Batch metadata with sequence information
    ///
    /// # Returns
    /// Logits tensor of shape [total_tokens, vocab_size]
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        batch_executor: &mut BatchExecutor,
        metadata: &BatchMetadata,
    ) -> Result<Tensor> {
        // Clear the cached IndicesAndMask at the start of each forward pass
        // This ensures all layers get the same position indices for this token,
        // while the persistent builder tracks position advancement across tokens
        batch_executor.clear_iam_cache();

        // Get total number of tokens
        let total_tokens = input_ids.dims()[0];

        // Embed tokens: [total_tokens] -> [total_tokens, hidden_size]
        let hidden_states = self.embedding.forward(input_ids)?;

        // DEBUG: Print embedding output stats
        let emb_vec = hidden_states.flatten_all()?.to_vec1::<f32>()?;
        let emb_mean: f32 = emb_vec.iter().sum::<f32>() / emb_vec.len() as f32;
        let emb_max = emb_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let emb_min = emb_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!(
            "DEBUG Batched Embedding OUTPUT: mean={:.6}, min={:.6}, max={:.6}, sample[0:3]={:?}",
            emb_mean,
            emb_min,
            emb_max,
            &emb_vec[0..3.min(emb_vec.len())]
        );

        // DEBUG: Check what token ID we're embedding
        let token_ids = input_ids.to_vec1::<u32>()?;
        eprintln!("DEBUG Token IDs: {:?}", token_ids);

        // Reshape to 3D for transformer blocks: [total_tokens, hidden] -> [1, total_tokens, hidden]
        // For prefill: batch=1, seq=total_tokens
        // For decode: batch=total_tokens, seq=1
        let mut hidden_states = if metadata.is_prefill {
            hidden_states.unsqueeze(0)? // [total_tokens, hidden] -> [1, total_tokens, hidden]
        } else {
            hidden_states.reshape((total_tokens, 1, self.config.hidden_size))? // [total_tokens, hidden] -> [total_tokens, 1, hidden]
        };

        // Verify shape
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;
        if batch_size * seq_len != total_tokens {
            candle_core::bail!(
                "Expected {} total tokens, got batch={} * seq={}",
                total_tokens,
                batch_size,
                seq_len
            );
        }
        if hidden_size != self.config.hidden_size {
            candle_core::bail!(
                "Expected hidden size {}, got {}",
                self.config.hidden_size,
                hidden_size
            );
        }

        // Pass through all transformer blocks
        // For RoPE position:
        // - During prefill: use 0 (processing first token(s) of prompt)
        // - During decode: use context_lens (position in sequence)
        // Each request may be at a different position, but for single-request batches
        // or when all requests are at the same position, we can use the first one
        let index_pos = if metadata.is_prefill {
            0
        } else {
            // During decode, the position is how many tokens we've already processed
            // context_lens tells us this
            metadata.context_lens.get(0).copied().unwrap_or(0)
        };

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            hidden_states = block.forward(
                &hidden_states,
                index_pos, // RoPE starting position
                &self.cos,
                &self.sin,
                batch_executor,
                metadata,
            )?;
            // Note: layer_idx is used internally by block for KV cache access

            // DEBUG: Print stats for first and last few layers
            if layer_idx < 3 || layer_idx >= self.blocks.len() - 3 {
                let hs_vec = hidden_states.flatten_all()?.to_vec1::<f32>()?;
                let mean: f32 = hs_vec.iter().sum::<f32>() / hs_vec.len() as f32;
                let max = hs_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = hs_vec.iter().cloned().fold(f32::INFINITY, f32::min);
                eprintln!(
                    "DEBUG Batched Layer {}: mean={:.6}, min={:.6}, max={:.6}, sample[0:3]={:?}",
                    layer_idx,
                    mean,
                    min,
                    max,
                    &hs_vec[0..3.min(hs_vec.len())]
                );
            }
        }

        // Final normalization
        hidden_states = self.norm.forward(&hidden_states)?;

        // DEBUG: Check after final norm
        let hs_vec = hidden_states.flatten_all()?.to_vec1::<f32>()?;
        let hs_mean: f32 = hs_vec.iter().sum::<f32>() / hs_vec.len() as f32;
        let hs_max = hs_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let hs_min = hs_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!(
            "DEBUG After final norm: mean={:.6}, min={:.6}, max={:.6}, sample[0:3]={:?}",
            hs_mean,
            hs_min,
            hs_max,
            &hs_vec[0..3.min(hs_vec.len())]
        );

        // Extract last token's hidden state (same as Candle's Llama)
        // For prefill: [1, seq_len, hidden] -> [1, hidden]
        // For decode: [batch, 1, hidden] -> [batch, hidden]
        let (dim0, dim1, _) = hidden_states.dims3()?;
        let last_hidden = if metadata.is_prefill {
            // Prefill: batch=1, seq=seq_len, extract last token
            hidden_states.i((.., dim1 - 1, ..))?.contiguous()?
        } else {
            // Decode: batch=batch, seq=1, extract the single token
            hidden_states.i((.., 0, ..))?.contiguous()?
        };

        // Project to vocabulary: [batch_or_1, hidden_size] -> [batch_or_1, vocab_size]
        let logits = self.lm_head.forward(&last_hidden)?;

        // DEBUG: Check logits
        let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
        let logits_mean: f32 = logits_vec.iter().sum::<f32>() / logits_vec.len() as f32;
        let logits_max = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let logits_min = logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!(
            "DEBUG Final logits: mean={:.6}, min={:.6}, max={:.6}, sample[0:5]={:?}",
            logits_mean,
            logits_min,
            logits_max,
            &logits_vec[0..5.min(logits_vec.len())]
        );

        Ok(logits)
    }

    /// Get model configuration
    pub fn config(&self) -> &BatchedTransformerConfig {
        &self.config
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get dtype
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Type alias for Llama models using BatchedTransformer
pub type BatchedLlama = BatchedTransformer;

/// Type alias for Mistral models using BatchedTransformer
pub type BatchedMistral = BatchedTransformer;

/// Type alias for Gemma models using BatchedTransformer
pub type BatchedGemma = BatchedTransformer;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{IndexOp, test_utils::to_vec2_round};

    fn create_test_config() -> BatchedTransformerConfig {
        BatchedTransformerConfig {
            vocab_size: 1000,
            hidden_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 8,
            num_key_value_heads: 8,
            intermediate_size: 768,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            sliding_window: None,
            use_flash_attn: false,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn test_config_validation() {
        let mut config = create_test_config();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid: hidden_size not divisible by num_attention_heads
        config.hidden_size = 255;
        assert!(config.validate().is_err());
        config.hidden_size = 256;

        // Invalid: num_attention_heads not divisible by num_key_value_heads
        config.num_attention_heads = 9;
        config.num_key_value_heads = 4;
        assert!(config.validate().is_err());
        config.num_attention_heads = 8;
        config.num_key_value_heads = 8;

        // Invalid: zero layers
        config.num_hidden_layers = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_head_dim_calculation() {
        let config = create_test_config();
        assert_eq!(config.head_dim(), 32); // 256 / 8 = 32

        let llama_config = BatchedTransformerConfig::llama_7b();
        assert_eq!(llama_config.head_dim(), 128); // 4096 / 32 = 128
    }

    #[test]
    fn test_llama_configs() {
        // Test Llama 7B config
        let llama7b = BatchedTransformerConfig::llama_7b();
        assert_eq!(llama7b.vocab_size, 32000);
        assert_eq!(llama7b.hidden_size, 4096);
        assert_eq!(llama7b.num_hidden_layers, 32);
        assert_eq!(llama7b.num_attention_heads, 32);
        assert_eq!(llama7b.num_key_value_heads, 32); // MHA
        assert!(llama7b.validate().is_ok());

        // Test Llama 3 8B config
        let llama3_8b = BatchedTransformerConfig::llama3_8b();
        assert_eq!(llama3_8b.vocab_size, 128256);
        assert_eq!(llama3_8b.num_key_value_heads, 8); // GQA
        assert_eq!(llama3_8b.max_position_embeddings, 8192);
        assert!(llama3_8b.validate().is_ok());
    }

    #[test]
    fn test_rope_frequency_generation() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 32;
        let max_seq_len = 64;
        let rope_theta = 10000.0;

        let (cos, sin) = BatchedTransformer::precompute_rope_frequencies(
            head_dim,
            max_seq_len,
            rope_theta,
            &device,
            DType::F32,
        )?;

        // Check shapes
        assert_eq!(cos.dims(), &[max_seq_len, head_dim]);
        assert_eq!(sin.dims(), &[max_seq_len, head_dim]);

        // Check that cos^2 + sin^2 ≈ 1 (approximately, due to duplication)
        let cos_sq = cos.sqr()?;
        let sin_sq = sin.sqr()?;
        let sum = (cos_sq + sin_sq)?;

        // First position should have cos=1, sin=0 for low frequencies
        let cos_first: Vec<Vec<f32>> = to_vec2_round(&cos.i((0..1, ..))?, 4)?;
        assert!(
            (cos_first[0][0] - 1.0).abs() < 0.01,
            "First cos value should be ~1.0"
        );

        Ok(())
    }

    #[test]
    fn test_rope_properties() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 64;
        let max_seq_len = 128;

        let (cos, sin) = BatchedTransformer::precompute_rope_frequencies(
            head_dim,
            max_seq_len,
            10000.0,
            &device,
            DType::F32,
        )?;

        // Verify dimensions are correct
        assert_eq!(cos.dims(), &[max_seq_len, head_dim]);
        assert_eq!(sin.dims(), &[max_seq_len, head_dim]);

        // Verify values are in valid range [-1, 1]
        let cos_vals: Vec<Vec<f32>> = to_vec2_round(&cos, 4)?;
        let sin_vals: Vec<Vec<f32>> = to_vec2_round(&sin, 4)?;

        for row in cos_vals.iter() {
            for &val in row.iter() {
                assert!(val >= -1.0 && val <= 1.0, "Cos value out of range: {}", val);
            }
        }

        for row in sin_vals.iter() {
            for &val in row.iter() {
                assert!(val >= -1.0 && val <= 1.0, "Sin value out of range: {}", val);
            }
        }

        Ok(())
    }
}
