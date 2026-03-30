//! AWQ-quantized Qwen3 model implementation
//!
//! This module provides a Qwen3 implementation that uses AWQ 4-bit quantization
//! for weights, integrated with Marlin CUDA kernels for efficient inference.

use candlelight::core::{D, DType, Device, Module, Result, Tensor};
use candlelight::nn::{Activation, Embedding, VarBuilder, kv_cache::KvCache};
use std::sync::Arc;

use crate::loaders::awq::AwqLinear;

/// Qwen3 configuration (compatible with HF config.json)
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
}

fn default_rope_theta() -> f64 {
    1000000.0 // Qwen3 default
}

fn default_hidden_act() -> Activation {
    Activation::Silu
}

/// RmsNorm wrapper for Qwen3
#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candlelight::nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}

/// Helper function for repeat_kv (GQA)
fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

/// Rotary Position Embedding for Qwen3
#[derive(Debug, Clone)]
struct Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    head_dim: usize,
}

impl Qwen3RotaryEmbedding {
    fn new(dtype: DType, cfg: &Qwen3Config, device: &Device) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let theta = cfg.rope_theta;

        // Calculate inv_freq: 1.0 / (theta^(2i/d)) for i in [0, d/2)
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f64 / theta.powf(i as f64 / head_dim as f64))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;

        // Calculate position indices: [0, 1, 2, ..., max_seq_len-1]
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;

        // freqs = t @ inv_freq -> [max_seq_len, head_dim/2]
        let freqs = t.matmul(&inv_freq.to_dtype(DType::F32)?)?;

        // Concatenate [freqs, freqs] to get [max_seq_len, head_dim]
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;

        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self { sin, cos, head_dim })
    }

    /// Apply rotary embeddings to query/key tensors
    fn apply_rotary_emb(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;

        let sin = self.sin.narrow(0, offset, seq_len)?;
        let cos = self.cos.narrow(0, offset, seq_len)?;

        // Reshape for broadcasting: [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;

        let q_rot = apply_rope(q, &cos, &sin)?;
        let k_rot = apply_rope(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }
}

/// Apply RoPE rotation: x * cos + rotate_half(x) * sin
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(candlelight::core::D::Minus1)? / 2;

    // Split x into two halves along last dimension
    let x1 = x.narrow(candlelight::core::D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(candlelight::core::D::Minus1, half_dim, half_dim)?;

    // rotate_half(x) = [-x2, x1]
    let x_rotated = Tensor::cat(&[&x2.neg()?, &x1], candlelight::core::D::Minus1)?;

    // x * cos + rotate_half(x) * sin
    (x.broadcast_mul(cos)? + x_rotated.broadcast_mul(sin)?)
}

/// Qwen3 MLP with AWQ quantization
#[derive(Debug, Clone)]
struct Qwen3AwqMlp {
    gate_proj: AwqLinear,
    up_proj: AwqLinear,
    down_proj: AwqLinear,
    act_fn: Activation,
}

impl Qwen3AwqMlp {
    fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;

        let gate_proj = AwqLinear::new(hidden_size, intermediate_size, vb.pp("gate_proj"))
            .map_err(|e| candlelight::core::Error::Msg(format!("Failed to load gate_proj: {}", e)))?;
        let up_proj = AwqLinear::new(hidden_size, intermediate_size, vb.pp("up_proj"))
            .map_err(|e| candlelight::core::Error::Msg(format!("Failed to load up_proj: {}", e)))?;
        let down_proj = AwqLinear::new(intermediate_size, hidden_size, vb.pp("down_proj"))
            .map_err(|e| candlelight::core::Error::Msg(format!("Failed to load down_proj: {}", e)))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Qwen3AwqMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: gate(x) * up(x) -> down
        let gate = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let up = x.apply(&self.up_proj)?;
        let gated = (gate * up)?;
        gated.apply(&self.down_proj)
    }
}

/// Qwen3 Attention with AWQ quantization and per-head RMSNorm
#[derive(Debug, Clone)]
struct Qwen3AwqAttention {
    q_proj: AwqLinear,
    k_proj: AwqLinear,
    v_proj: AwqLinear,
    o_proj: AwqLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<Qwen3RotaryEmbedding>,
    kv_cache: KvCache,
}

impl Qwen3AwqAttention {
    fn new(
        cfg: &Qwen3Config,
        rotary_emb: Arc<Qwen3RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden_size / num_heads;

        let q_proj = AwqLinear::new(hidden_size, num_heads * head_dim, vb.pp("q_proj"))
            .map_err(|e| candlelight::core::Error::Msg(format!("Failed to load q_proj: {}", e)))?;
        let k_proj = AwqLinear::new(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))
            .map_err(|e| candlelight::core::Error::Msg(format!("Failed to load k_proj: {}", e)))?;
        let v_proj = AwqLinear::new(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))
            .map_err(|e| candlelight::core::Error::Msg(format!("Failed to load v_proj: {}", e)))?;
        let o_proj = AwqLinear::new(num_heads * head_dim, hidden_size, vb.pp("o_proj"))
            .map_err(|e| candlelight::core::Error::Msg(format!("Failed to load o_proj: {}", e)))?;

        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let kv_cache = KvCache::new(2, cfg.num_hidden_layers);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;

        // 1. Project Q, K, V
        let q = x.apply(&self.q_proj)?;
        let k = x.apply(&self.k_proj)?;
        let v = x.apply(&self.v_proj)?;

        // 2. Reshape: (B, L, H*D) -> (B, H, L, D)
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Per-head RMSNorm (Qwen3 specific)
        let q_flat = q.flatten(0, 2)?; // (B*H*L, D)
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, seq_len, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, seq_len, self.head_dim))?;

        // 4. Apply RoPE
        let (q, k) = self.rotary_emb.apply_rotary_emb(&q, &k, offset)?;

        // 5. Update KV cache
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // 6. Repeat K/V for GQA (if num_kv_heads < num_heads)
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        // 7. Attention: scores = Q @ K^T / sqrt(head_dim)
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // 8. Apply causal mask
        let scores = match attn_mask {
            Some(mask) => scores.broadcast_add(mask)?,
            None => scores,
        };

        // 9. Softmax and attend
        let probs = candlelight::nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)

        // 10. Output projection
        ctx.transpose(1, 2)?
            .reshape((b, seq_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

/// Qwen3 Decoder Layer
#[derive(Debug, Clone)]
struct Qwen3AwqDecoderLayer {
    self_attn: Qwen3AwqAttention,
    mlp: Qwen3AwqMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3AwqDecoderLayer {
    fn new(cfg: &Qwen3Config, rotary: Arc<Qwen3RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen3AwqAttention::new(cfg, rotary, vb.pp("self_attn"))?;
        let mlp = Qwen3AwqMlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        // Pre-norm attention
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;

        // Pre-norm MLP
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = h.apply(&self.mlp)?;
        x + h
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// AWQ-quantized Qwen3 model
#[derive(Debug)]
pub struct AwqQwen3 {
    embed_tokens: Embedding,
    layers: Vec<Qwen3AwqDecoderLayer>,
    norm: RmsNorm,
    lm_head: AwqLinear,
    device: Device,
    dtype: DType,
}

impl AwqQwen3 {
    pub fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candlelight::nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        let rotary = Arc::new(Qwen3RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3AwqDecoderLayer::new(cfg, rotary.clone(), vb_l.pp(i))?);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = AwqLinear::new(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))
            .map_err(|e| candlelight::core::Error::Msg(format!("Failed to load lm_head: {}", e)))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, seq_len) = input_ids.dims2()?;

        // Embedding
        let mut h = self.embed_tokens.forward(input_ids)?;

        // Causal mask (only needed for seq_len > 1)
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.causal_mask(b, seq_len, offset)?)
        };

        // Transformer layers
        for layer in &mut self.layers {
            h = layer.forward(&h, mask.as_ref(), offset)?;
        }

        // Final norm + LM head
        let h = self.norm.forward(&h)?;
        let logits = h.apply(&self.lm_head)?;

        // Return last token logits for generation
        logits.narrow(1, seq_len - 1, 1)?.squeeze(1)
    }

    /// Generate causal attention mask
    fn causal_mask(&self, batch_size: usize, seq_len: usize, offset: usize) -> Result<Tensor> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    if i + offset < j + offset {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();

        let mask =
            Tensor::from_vec(mask, (seq_len, seq_len), &self.device)?.to_dtype(self.dtype)?;

        // Broadcast to [batch, 1, seq_len, seq_len]
        mask.unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((batch_size, 1, seq_len, seq_len))
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_dimensions() {
        let device = Device::Cpu;
        let cfg = Qwen3Config {
            vocab_size: 151936,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 64,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            hidden_act: Activation::Silu,
        };

        let rope = Qwen3RotaryEmbedding::new(DType::F32, &cfg, &device).unwrap();
        assert_eq!(rope.head_dim, 80); // 5120 / 64
        assert_eq!(rope.sin.dims(), &[40960, 80]);
        assert_eq!(rope.cos.dims(), &[40960, 80]);
    }
}
