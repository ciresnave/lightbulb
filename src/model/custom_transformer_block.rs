//! Custom Batched Transformer Block
//!
//! This module implements a complete transformer block that combines self-attention
//! and feed-forward network (MLP) with layer normalization and residual connections.
//! Unlike standard Llama blocks which process requests sequentially, this implementation
//! handles entire batches in parallel for maximum throughput.
//!
//! # Architecture
//!
//! ```text
//! input [batch, seq, hidden]
//!   │
//!   ├──> RMSNorm ──> BatchedAttention ──> residual add ─┐
//!   │                                                     │
//!   └─────────────────────────────────────────────────>─┘
//!   │
//!   ├──> RMSNorm ──> BatchedMLP ──> residual add ─┐
//!   │                                              │
//!   └──────────────────────────────────────────>─┘
//!   │
//!   └──> output [batch, seq, hidden]
//! ```
//!
//! This follows the Pre-LN (Pre-Layer Normalization) pattern used in Llama,
//! where normalization is applied before the sub-layer rather than after.
//!
//! # Pre-LN vs Post-LN
//!
//! - **Pre-LN** (Llama, GPT-3, used here):
//!   ```text
//!   x = x + Attention(Norm(x))
//!   x = x + MLP(Norm(x))
//!   ```
//!   Benefits: Better training stability, no need for warmup
//!
//! - **Post-LN** (Original Transformer):
//!   ```text
//!   x = Norm(x + Attention(x))
//!   x = Norm(x + MLP(x))
//!   ```
//!
//! # Batching Strategy
//!
//! - **Input**: [batch_size, seq_len, hidden_size]
//! - **Processing**: All operations handle batches naturally
//! - **Output**: [batch_size, seq_len, hidden_size]
//!
//! The transformer block is where the magic happens:
//! 1. Self-attention allows tokens to attend to each other
//! 2. MLP processes each token independently
//! 3. Residual connections preserve information flow
//!
//! # Example
//!
//! ```rust,ignore
//! let block = BatchedTransformerBlock::new(
//!     layer_idx,
//!     hidden_size,
//!     num_heads,
//!     num_kv_heads,
//!     intermediate_size,
//!     rms_norm_eps,
//!     vb.pp(&format!("layers.{}", layer_idx)),
//! )?;
//!
//! // Forward pass
//! let output = block.forward(
//!     &hidden_states,
//!     index_pos,
//!     cos, sin,
//!     batch_executor,
//!     metadata,
//! )?;
//! ```

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{RmsNorm, VarBuilder, rms_norm};
use std::fs::File;
use std::io::{Read, Seek};

use super::custom_attention::BatchedAttention;
use super::mlp_wrapper::Mlp; // Thin wrapper using Candle's components
use crate::engine::BatchExecutor;
use crate::model::batch_metadata::BatchMetadata;

/// Batched Transformer Block
///
/// A complete transformer layer combining self-attention and feed-forward network
/// with Pre-LN (Pre-Layer Normalization) and residual connections.
///
/// # Components
///
/// - **input_layernorm**: RMSNorm before attention
/// - **self_attn**: Multi-head self-attention with KV cache
/// - **post_attention_layernorm**: RMSNorm before MLP
/// - **mlp**: Feed-forward network with SwiGLU
///
/// # Memory Layout
///
/// All operations maintain batch-first layout:
/// - Input: [batch_size, seq_len, hidden_size]
/// - Intermediate: [batch_size, seq_len, hidden_size] (after each sub-layer)
/// - Output: [batch_size, seq_len, hidden_size]
#[derive(Debug)]
pub struct BatchedTransformerBlock {
    /// Layer index (0 to num_layers-1)
    layer_idx: usize,

    /// RMSNorm before self-attention (Pre-LN pattern)
    input_layernorm: RmsNorm,

    /// Batched multi-head self-attention
    self_attn: BatchedAttention,

    /// RMSNorm before MLP (Pre-LN pattern)
    post_attention_layernorm: RmsNorm,

    /// Feed-forward network (using Candle's Mlp - no need for custom version!)
    mlp: Mlp,

    /// Hidden dimension size
    hidden_size: usize,

    /// Device for tensor operations
    device: Device,

    /// Data type (F32, F16, BF16)
    dtype: DType,
}

impl BatchedTransformerBlock {
    /// Create a new BatchedTransformerBlock
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of this layer (0 to num_layers-1)
    /// * `hidden_size` - Dimension of hidden states (e.g., 4096)
    /// * `num_heads` - Number of attention heads (e.g., 32)
    /// * `num_kv_heads` - Number of KV heads for GQA (e.g., 32 or 8)
    /// * `intermediate_size` - MLP intermediate dimension (e.g., 11008)
    /// * `rms_norm_eps` - Epsilon for RMSNorm stability (e.g., 1e-5)
    /// * `vb` - VarBuilder for loading weights
    ///
    /// # Returns
    ///
    /// BatchedTransformerBlock instance with initialized weights
    pub fn new(
        layer_idx: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_size: usize,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        // Load RMSNorm layers (Candle provides this!)
        let input_layernorm = rms_norm(hidden_size, rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            rms_norm(hidden_size, rms_norm_eps, vb.pp("post_attention_layernorm"))?;

        // Create batched attention layer
        let self_attn_vb = vb.pp("self_attn");
        let self_attn = BatchedAttention::new(hidden_size, num_heads, num_kv_heads, self_attn_vb)?;

        // Use our thin MLP wrapper (built on Candle's Linear + ops::silu)
        let mlp = Mlp::new(hidden_size, intermediate_size, vb.pp("mlp"))?;

        Ok(Self {
            layer_idx,
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
            hidden_size,
            device,
            dtype,
        })
    }

    /// Create a new BatchedTransformerBlock from GGUF quantized weights
    ///
    /// # Arguments
    /// * `layer_idx` - Index of this layer (0 to num_layers-1)
    /// * `hidden_size` - Dimension of hidden states
    /// * `num_heads` - Number of attention heads
    /// * `num_kv_heads` - Number of KV heads for GQA
    /// * `intermediate_size` - MLP intermediate dimension
    /// * `rms_norm_eps` - Epsilon for RMSNorm stability
    /// * `gguf_content` - GGUF file content with metadata and tensor info
    /// * `file` - Open file handle for reading tensor data
    /// * `device` - Device to load tensors on
    pub fn from_gguf<R: Read + Seek>(
        layer_idx: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_size: usize,
        rms_norm_eps: f64,
        gguf_content: &crate::gguf::Content,
        file: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{}", layer_idx);

        // Load RMSNorm weights (dequantize them since RmsNorm doesn't work with quantized weights directly)
        let input_norm_tensor =
            gguf_content.tensor(file, &format!("{}.attn_norm.weight", prefix), device)?;
        let post_attn_norm_tensor =
            gguf_content.tensor(file, &format!("{}.ffn_norm.weight", prefix), device)?;

        let input_layernorm = RmsNorm::new(input_norm_tensor.dequantize(device)?, rms_norm_eps);
        let post_attention_layernorm =
            RmsNorm::new(post_attn_norm_tensor.dequantize(device)?, rms_norm_eps);

        // Load attention with quantized weights
        let self_attn = BatchedAttention::from_gguf(
            hidden_size,
            num_heads,
            num_kv_heads,
            gguf_content,
            file,
            device,
            layer_idx,
        )?;

        // Load MLP with quantized weights
        let mlp = Mlp::from_gguf(
            hidden_size,
            intermediate_size,
            gguf_content,
            file,
            device,
            layer_idx,
        )?;

        // Use F32 for quantized models
        let dtype = DType::F32;

        Ok(Self {
            layer_idx,
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
            hidden_size,
            device: device.clone(),
            dtype,
        })
    }

    /// Forward pass through the transformer block
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Input tensor [batch_size, seq_len, hidden_size]
    /// * `index_pos` - Starting position for RoPE
    /// * `cos` - Pre-computed cosine tensor for RoPE
    /// * `sin` - Pre-computed sine tensor for RoPE
    /// * `batch_executor` - Contains ScatteredKvCache for K/V storage
    /// * `metadata` - Batch structure information
    ///
    /// # Returns
    ///
    /// Output tensor [batch_size, seq_len, hidden_size]
    ///
    /// # Algorithm (Pre-LN Pattern)
    ///
    /// 1. Attention block:
    ///    - Normalize: normed = input_layernorm(hidden_states)
    ///    - Attend: attn_output = self_attn(normed, ...)
    ///    - Residual: hidden_states = hidden_states + attn_output
    ///
    /// 2. MLP block:
    ///    - Normalize: normed = post_attention_layernorm(hidden_states)
    ///    - Transform: mlp_output = mlp(normed)
    ///    - Residual: hidden_states = hidden_states + mlp_output
    ///
    /// 3. Return: hidden_states
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        index_pos: usize,
        cos: &Tensor,
        sin: &Tensor,
        batch_executor: &mut BatchExecutor,
        metadata: &BatchMetadata,
    ) -> Result<Tensor> {
        let (_batch_size, _seq_len, hidden_dim) = hidden_states.dims3()?;

        // Validate input dimensions
        if hidden_dim != self.hidden_size {
            candle_core::bail!(
                "Input hidden dimension {} does not match expected {} for layer {}",
                hidden_dim,
                self.hidden_size,
                self.layer_idx
            );
        }

        // === Self-Attention Block (with Pre-LN) ===

        // DEBUG: Check RAW input to layer (before norm)
        if self.layer_idx < 3 {
            let raw_vec = hidden_states.flatten_all()?.to_vec1::<f32>()?;
            let raw_mean: f32 = raw_vec.iter().sum::<f32>() / raw_vec.len() as f32;
            let raw_max = raw_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            // DEBUG output removed
        }

        // 1. Normalize input
        let normed_input = self.input_layernorm.forward(hidden_states)?;

        // DEBUG: Check normalization output
        if self.layer_idx < 3 {
            let norm_vec = normed_input.flatten_all()?.to_vec1::<f32>()?;
            let norm_mean: f32 = norm_vec.iter().sum::<f32>() / norm_vec.len() as f32;
            let norm_max = norm_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            // DEBUG output removed
        }

        // 2. Apply batched self-attention
        let attn_output = self.self_attn.forward(
            &normed_input,
            index_pos,
            cos,
            sin,
            batch_executor,
            metadata,
            self.layer_idx,
        )?;

        // DEBUG: Check shapes and values
        if self.layer_idx == 0 {
            let attn_vec = attn_output.flatten_all()?.to_vec1::<f32>()?;
            let attn_mean: f32 = attn_vec.iter().sum::<f32>() / attn_vec.len() as f32;
            let attn_max = attn_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            // DEBUG output removed

            // Check input to residual
            let hs_vec = hidden_states.flatten_all()?.to_vec1::<f32>()?;
            let hs_mean: f32 = hs_vec.iter().sum::<f32>() / hs_vec.len() as f32;
            // DEBUG output removed
            // DEBUG output removed
        }

        // 3. Residual connection: x = x + attention(norm(x))
        let hidden_states = (hidden_states + attn_output)?;

        // DEBUG: Check result AFTER residual
        if self.layer_idx == 0 {
            let hs_vec = hidden_states.flatten_all()?.to_vec1::<f32>()?;
            let hs_mean: f32 = hs_vec.iter().sum::<f32>() / hs_vec.len() as f32;
            let hs_max = hs_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            // DEBUG output removed
        }

        // === MLP Block (with Pre-LN) ===
        // 1. Normalize input
        let normed_hidden = self.post_attention_layernorm.forward(&hidden_states)?;

        // DEBUG: Check MLP input normalization for first few layers
        if self.layer_idx < 3 {
            let norm_mlp_vec = normed_hidden.flatten_all()?.to_vec1::<f32>()?;
            let norm_mlp_mean: f32 = norm_mlp_vec.iter().sum::<f32>() / norm_mlp_vec.len() as f32;
            let norm_mlp_max = norm_mlp_vec
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            // DEBUG output removed
        }

        // 2. Apply batched MLP
        let mlp_output = self.mlp.forward(&normed_hidden)?;

        // DEBUG: Check MLP output
        if self.layer_idx == 0 {
            let mlp_vec = mlp_output.flatten_all()?.to_vec1::<f32>()?;
            let mlp_mean: f32 = mlp_vec.iter().sum::<f32>() / mlp_vec.len() as f32;
            let mlp_max = mlp_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mlp_min = mlp_vec.iter().cloned().fold(f32::INFINITY, f32::min);
            // DEBUG output removed
        }

        // 3. Residual connection: x = x + mlp(norm(x))
        let hidden_states = (hidden_states + mlp_output)?;

        // DEBUG: Check final Layer 0 output
        if self.layer_idx == 0 {
            let final_vec = hidden_states.flatten_all()?.to_vec1::<f32>()?;
            let final_mean: f32 = final_vec.iter().sum::<f32>() / final_vec.len() as f32;
            let final_max = final_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let final_min = final_vec.iter().cloned().fold(f32::INFINITY, f32::min);
            // DEBUG output removed
        }

        Ok(hidden_states)
    }

    /// Get the layer index
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Get the device this block operates on
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the data type used by this block
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{BatchExecutor, RequestContext, RequestState};
    use candle_core::{Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    fn create_test_batch_executor(
        max_batch_size: usize,
        max_seq_len: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> BatchExecutor {
        let device = Device::Cpu;
        let dtype = candle_core::DType::F32;
        BatchExecutor::new(
            max_batch_size,
            max_seq_len,
            num_layers,
            num_heads,
            head_dim,
            dtype,
            &device,
        )
        .unwrap()
    }

    fn create_dummy_cos_sin(max_seq_len: usize, head_dim: usize) -> (Tensor, Tensor) {
        let device = Device::Cpu;
        // RoPE cos/sin are generated at half the head dimension (freqs for pairs)
        let half_dim = head_dim / 2;
        let cos = Tensor::ones((max_seq_len, half_dim), candle_core::DType::F32, &device).unwrap();
        let sin = Tensor::zeros((max_seq_len, half_dim), candle_core::DType::F32, &device).unwrap();
        (cos, sin)
    }

    #[test]
    fn test_batched_transformer_block_shapes() {
        let device = Device::Cpu;
        let dtype = candle_core::DType::F32;
        let hidden_size = 64;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = hidden_size / num_heads;
        let intermediate_size = 128;
        let rms_norm_eps = 1e-5;
        let batch_size = 2;
        let seq_len = 1; // Decode mode: single token per request
        let max_seq_len = 16;
        let num_layers = 2;

        // Create VarMap and VarBuilder
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        // Create transformer block
        let block = BatchedTransformerBlock::new(
            0, // layer_idx
            hidden_size,
            num_heads,
            num_kv_heads,
            intermediate_size,
            rms_norm_eps,
            vb.pp("layer_0"),
        )
        .unwrap();

        // Create test inputs - decode mode expects [batch_size, 1, hidden_size]
        let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, hidden_size), &device).unwrap();
        let (cos, sin) = create_dummy_cos_sin(max_seq_len, head_dim);

        // Create batch executor and metadata
        let mut batch_executor =
            create_test_batch_executor(batch_size, max_seq_len, num_layers, num_heads, head_dim);

        let request_ids: Vec<usize> = (0..batch_size).collect();
        let positions: Vec<usize> = vec![0; batch_size];
        let metadata =
            BatchMetadata::from_decode_batch(request_ids.clone(), positions.clone(), positions);

        // Prepare batch (assign cache indices)
        let mut contexts: Vec<RequestContext> = request_ids
            .iter()
            .map(|&id| {
                let mut ctx = RequestContext::new(crate::engine::Request {
                    id: id.to_string(),
                    prompt: String::new(),
                    max_new_tokens: 10,
                });
                ctx.state = RequestState::Decoding;
                ctx
            })
            .collect();
        batch_executor.prepare_batch(&mut contexts).unwrap();

        // Forward pass
        let output = block
            .forward(&input, 0, &cos, &sin, &mut batch_executor, &metadata)
            .unwrap();

        // Check output shape
        assert_eq!(output.dims(), &[batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_batched_transformer_block_single_token() {
        let device = Device::Cpu;
        let dtype = candle_core::DType::F32;
        let hidden_size = 32;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = hidden_size / num_heads;
        let intermediate_size = 64;
        let rms_norm_eps = 1e-5;
        let max_seq_len = 8;
        let num_layers = 1;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let block = BatchedTransformerBlock::new(
            0,
            hidden_size,
            num_heads,
            num_kv_heads,
            intermediate_size,
            rms_norm_eps,
            vb.pp("layer_0"),
        )
        .unwrap();

        // Single token (batch_size=1, seq_len=1)
        let input = Tensor::randn(0f32, 1f32, (1, 1, hidden_size), &device).unwrap();
        let (cos, sin) = create_dummy_cos_sin(max_seq_len, head_dim);

        let mut batch_executor =
            create_test_batch_executor(1, max_seq_len, num_layers, num_heads, head_dim);

        let request_ids = vec![0];
        let positions = vec![0];
        let metadata =
            BatchMetadata::from_decode_batch(request_ids.clone(), positions.clone(), positions);

        let mut contexts = vec![{
            let mut ctx = RequestContext::new(crate::engine::Request {
                id: request_ids[0].to_string(),
                prompt: String::new(),
                max_new_tokens: 10,
            });
            ctx.state = RequestState::Decoding;
            ctx
        }];
        batch_executor.prepare_batch(&mut contexts).unwrap();

        let output = block
            .forward(&input, 0, &cos, &sin, &mut batch_executor, &metadata)
            .unwrap();

        assert_eq!(output.dims(), &[1, 1, hidden_size]);
    }

    #[test]
    fn test_batched_transformer_block_properties() {
        let device = Device::Cpu;
        let dtype = candle_core::DType::F32;
        let hidden_size = 64;
        let num_heads = 4;
        let num_kv_heads = 4;
        let intermediate_size = 128;
        let rms_norm_eps = 1e-5;
        let layer_idx = 5;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let block = BatchedTransformerBlock::new(
            layer_idx,
            hidden_size,
            num_heads,
            num_kv_heads,
            intermediate_size,
            rms_norm_eps,
            vb.pp("layer_5"),
        )
        .unwrap();

        assert_eq!(block.layer_idx(), layer_idx);
        assert_eq!(block.hidden_size(), hidden_size);
        assert_eq!(block.dtype(), dtype);
    }

    #[test]
    fn test_batched_transformer_block_dimension_validation() {
        let device = Device::Cpu;
        let dtype = candle_core::DType::F32;
        let hidden_size = 64;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_dim = hidden_size / num_heads;
        let intermediate_size = 128;
        let rms_norm_eps = 1e-5;
        let max_seq_len = 8;
        let num_layers = 1;

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let block = BatchedTransformerBlock::new(
            0,
            hidden_size,
            num_heads,
            num_kv_heads,
            intermediate_size,
            rms_norm_eps,
            vb.pp("layer_0"),
        )
        .unwrap();

        // Wrong hidden dimension (should fail)
        let wrong_input = Tensor::randn(0f32, 1f32, (2, 4, 32), &device).unwrap();
        let (cos, sin) = create_dummy_cos_sin(max_seq_len, head_dim);

        let mut batch_executor =
            create_test_batch_executor(2, max_seq_len, num_layers, num_heads, head_dim);

        let request_ids = vec![0, 1];
        let positions = vec![0, 0];
        let metadata =
            BatchMetadata::from_decode_batch(request_ids.clone(), positions.clone(), positions);

        let mut contexts: Vec<RequestContext> = request_ids
            .iter()
            .map(|&id| {
                let mut ctx = RequestContext::new(crate::engine::Request {
                    id: id.to_string(),
                    prompt: String::new(),
                    max_new_tokens: 10,
                });
                ctx.state = RequestState::Decoding;
                ctx
            })
            .collect();
        batch_executor.prepare_batch(&mut contexts).unwrap();

        let result = block.forward(&wrong_input, 0, &cos, &sin, &mut batch_executor, &metadata);

        assert!(result.is_err());
    }
}
