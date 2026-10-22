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
    /// Whether `lm_head` reuses `embed_tokens`' weight rather than carrying its own. HF's own
    /// default is `false` when the field is absent from `config.json`, so a missing field here
    /// must not be read as "tied" — see `LmHead::load` for why this flag alone is not trusted.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Per-head dimension of Q/K/V, read straight from `config.json`. Every real Qwen3 dense
    /// checkpoint (0.6B through ~4B, confirmed on `Qwen/Qwen3-0.6B`'s and
    /// `Orion-zhen/Qwen3-0.6B-AWQ`'s `config.json`) sets this to 128 **regardless of
    /// `hidden_size`/`num_attention_heads`** — HF `transformers`' own `Qwen3Config` keeps the
    /// two independent, and `fuel_transformers`' GGUF Qwen3 reader (`loader_gguf_qwen3.rs`)
    /// reads it from `qwen3.attention.key_length`, never derives it. `#[serde(default)]` so a
    /// `config.json` that omits the field (HF's own default, used when the two happen to
    /// agree) falls back to the formula in [`Qwen3Config::effective_head_dim`] rather than
    /// failing to parse.
    #[serde(default)]
    pub head_dim: Option<usize>,
}

impl Qwen3Config {
    /// The head dimension actually used by Q/K/V projections, rotary embeddings and the
    /// per-head norms. Prefers the explicit `head_dim`; falls back to the formula ONLY when
    /// the checkpoint's `config.json` omits the field, matching HF `transformers`' own
    /// default. Every call site that used to compute `hidden_size / num_attention_heads`
    /// directly must go through this instead — that formula is wrong whenever `head_dim` is
    /// set and disagrees with it, which is every real Qwen3 dense checkpoint.
    pub fn effective_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
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
        let head_dim = cfg.effective_head_dim();
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
    x.broadcast_mul(cos)? + x_rotated.broadcast_mul(sin)?
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
            .map_err(|e| {
                candlelight::core::Error::Msg(format!("Failed to load gate_proj: {}", e))
            })?;
        let up_proj = AwqLinear::new(hidden_size, intermediate_size, vb.pp("up_proj"))
            .map_err(|e| candlelight::core::Error::Msg(format!("Failed to load up_proj: {}", e)))?;
        let down_proj = AwqLinear::new(intermediate_size, hidden_size, vb.pp("down_proj"))
            .map_err(|e| {
                candlelight::core::Error::Msg(format!("Failed to load down_proj: {}", e))
            })?;

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
        let head_dim = cfg.effective_head_dim();

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

/// The output projection: either its own AWQ-quantized weight, or a tied reuse of
/// `embed_tokens`'s (non-quantized) embedding weight.
///
/// ⚠️ A SEPARATE TYPE, NOT A BOOL BRANCH INSIDE `AwqQwen3`, because the two cases load
/// different tensors under different names and the quantized path's weight is 4-bit while the
/// tied path's is not — collapsing them into one `AwqLinear` field was exactly the bug this
/// type exists to fix (it made the tied case structurally unrepresentable, not just unhandled).
#[derive(Debug, Clone)]
enum LmHead {
    Quantized(AwqLinear),
    Tied(candlelight::nn::Linear),
}

impl LmHead {
    /// Decide which case this checkpoint is, from what is actually on disk — not from
    /// `config.json`'s `tie_word_embeddings` alone.
    ///
    /// ⚠️ THE TENSOR'S PRESENCE IS THE GROUND TRUTH, THE CONFIG FLAG IS ONLY A FALLBACK CHECK.
    /// A checkpoint could in principle set `tie_word_embeddings: true` and still ship its own
    /// `lm_head.*` tensors (some conversion pipelines do this defensively), in which case using
    /// the real weight is strictly more correct than reusing the embedding one. Trusting the
    /// config flag over the tensor's presence would silently discard a real quantized lm_head
    /// the checkpoint actually shipped.
    fn load(cfg: &Qwen3Config, embed_tokens: &Embedding, vb: VarBuilder) -> Result<Self> {
        let lm_head_vb = vb.pp("lm_head");
        if lm_head_vb.contains_tensor("qweight") {
            let awq = AwqLinear::new(cfg.hidden_size, cfg.vocab_size, lm_head_vb).map_err(|e| {
                candlelight::core::Error::Msg(format!("Failed to load lm_head: {}", e))
            })?;
            return Ok(Self::Quantized(awq));
        }
        if cfg.tie_word_embeddings {
            return Ok(Self::Tied(candlelight::nn::Linear::new(
                embed_tokens.embeddings().clone(),
                None,
            )));
        }
        candlelight::core::bail!(
            "no lm_head.qweight tensor and config.json's tie_word_embeddings is false -- this \
             checkpoint names neither a quantized lm_head nor a tied one"
        )
    }
}

impl candlelight::core::Module for LmHead {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            // `AwqLinear` has both an inherent `forward` (`anyhow::Result`, used by its CUDA
            // Marlin path) and this `Module::forward` (`candle::Result`) -- `.forward(x)` here
            // would resolve to the inherent one and mismatch this method's return type, so the
            // trait method is named explicitly.
            Self::Quantized(awq) => candlelight::core::Module::forward(awq, x),
            Self::Tied(linear) => linear.forward(x),
        }
    }
}

/// AWQ-quantized Qwen3 model
#[derive(Debug)]
pub struct AwqQwen3 {
    embed_tokens: Embedding,
    layers: Vec<Qwen3AwqDecoderLayer>,
    norm: RmsNorm,
    lm_head: LmHead,
    device: Device,
    dtype: DType,
}

impl AwqQwen3 {
    pub fn new(cfg: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = candlelight::nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        let rotary = Arc::new(Qwen3RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(Qwen3AwqDecoderLayer::new(cfg, rotary.clone(), vb_l.pp(i))?);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = LmHead::load(cfg, &embed_tokens, vb.clone())?;

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
            tie_word_embeddings: false,
            head_dim: None,
        };

        let rope = Qwen3RotaryEmbedding::new(DType::F32, &cfg, &device).unwrap();
        assert_eq!(rope.head_dim, 80); // 5120 / 64
        assert_eq!(rope.sin.dims(), &[40960, 80]);
        assert_eq!(rope.cos.dims(), &[40960, 80]);
    }

    /// A tiny config whose every dimension still satisfies `AwqLinear`'s hard-coded
    /// `group_size = 128` divisibility (`in_features` a multiple of 128 and of 8,
    /// `out_features` a multiple of 8) -- smaller numbers would make `AwqLinear::new` itself
    /// fail with a shape error unrelated to what this test is checking.
    fn tiny_cfg(tie_word_embeddings: bool) -> Qwen3Config {
        Qwen3Config {
            vocab_size: 16,
            hidden_size: 128,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            hidden_act: Activation::Silu,
            tie_word_embeddings,
            head_dim: None,
        }
    }

    /// Like [`tiny_cfg`], but with an explicit `head_dim` that the formula
    /// (`hidden_size / num_attention_heads == 64`) would get wrong — the shape every real
    /// Qwen3 dense checkpoint is in. `256` rather than something formula-adjacent, so a
    /// transposed or halved head_dim could not accidentally satisfy both.
    fn tiny_cfg_with_head_dim_override(tie_word_embeddings: bool) -> Qwen3Config {
        Qwen3Config {
            head_dim: Some(256),
            ..tiny_cfg(tie_word_embeddings)
        }
    }

    /// The three tensors one `AwqLinear` needs, named under `prefix`, zero-filled (construction
    /// only reads shapes, never values).
    fn awq_linear_tensors(
        prefix: &str,
        in_features: usize,
        out_features: usize,
        dev: &Device,
    ) -> Vec<(String, Tensor)> {
        const GROUP_SIZE: usize = 128; // matches AwqLinear::new's own hard-coded value
        vec![
            (
                format!("{prefix}.qweight"),
                Tensor::zeros((in_features / 8, out_features), DType::U32, dev).unwrap(),
            ),
            (
                format!("{prefix}.qzeros"),
                Tensor::zeros(
                    (in_features / GROUP_SIZE, out_features / 8),
                    DType::U32,
                    dev,
                )
                .unwrap(),
            ),
            (
                format!("{prefix}.scales"),
                Tensor::zeros((in_features / GROUP_SIZE, out_features), DType::F16, dev).unwrap(),
            ),
        ]
    }

    /// The attention-block tensors for one decoder layer, named under `model.layers.{idx}`.
    /// Split out of `decoder_layer_tensors` to keep every helper under Codacy's 50-line limit.
    fn decoder_attn_tensors(p: &str, cfg: &Qwen3Config, dev: &Device) -> Vec<(String, Tensor)> {
        // The REAL shape a checkpoint's tensors carry, which is `cfg.head_dim` where the
        // checkpoint sets one -- never the formula, which the override test below relies on
        // disagreeing with.
        let head_dim = cfg.effective_head_dim();
        let mut t = Vec::new();
        t.extend(awq_linear_tensors(
            &format!("{p}.self_attn.q_proj"),
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            dev,
        ));
        t.extend(awq_linear_tensors(
            &format!("{p}.self_attn.k_proj"),
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            dev,
        ));
        t.extend(awq_linear_tensors(
            &format!("{p}.self_attn.v_proj"),
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            dev,
        ));
        t.extend(awq_linear_tensors(
            &format!("{p}.self_attn.o_proj"),
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            dev,
        ));
        t.push((
            format!("{p}.self_attn.q_norm.weight"),
            Tensor::zeros(head_dim, DType::F16, dev).unwrap(),
        ));
        t.push((
            format!("{p}.self_attn.k_norm.weight"),
            Tensor::zeros(head_dim, DType::F16, dev).unwrap(),
        ));
        t
    }

    /// The MLP-block and norm tensors for one decoder layer, named under `model.layers.{idx}`.
    /// Split out of `decoder_layer_tensors` to keep every helper under Codacy's 50-line limit.
    fn decoder_mlp_tensors(p: &str, cfg: &Qwen3Config, dev: &Device) -> Vec<(String, Tensor)> {
        let mut t = Vec::new();
        t.extend(awq_linear_tensors(
            &format!("{p}.mlp.gate_proj"),
            cfg.hidden_size,
            cfg.intermediate_size,
            dev,
        ));
        t.extend(awq_linear_tensors(
            &format!("{p}.mlp.up_proj"),
            cfg.hidden_size,
            cfg.intermediate_size,
            dev,
        ));
        t.extend(awq_linear_tensors(
            &format!("{p}.mlp.down_proj"),
            cfg.intermediate_size,
            cfg.hidden_size,
            dev,
        ));
        t.push((
            format!("{p}.input_layernorm.weight"),
            Tensor::zeros(cfg.hidden_size, DType::F16, dev).unwrap(),
        ));
        t.push((
            format!("{p}.post_attention_layernorm.weight"),
            Tensor::zeros(cfg.hidden_size, DType::F16, dev).unwrap(),
        ));
        t
    }

    /// The tensors one `Qwen3AwqDecoderLayer` needs, named under `model.layers.{idx}`.
    /// Factored out of `base_model_tensors` to keep that function under Codacy's 50-line
    /// limit, not because this half is reused anywhere else.
    fn decoder_layer_tensors(idx: usize, cfg: &Qwen3Config, dev: &Device) -> Vec<(String, Tensor)> {
        let p = format!("model.layers.{idx}");
        let mut t = decoder_attn_tensors(&p, cfg, dev);
        t.extend(decoder_mlp_tensors(&p, cfg, dev));
        t
    }

    /// Every tensor `AwqQwen3::new` needs EXCEPT `lm_head.*` -- the embedding, every decoder
    /// layer's attention/MLP/norms, and the final norm. Shared by the tied and untied tests so
    /// a divergence between them can only be the one tensor set each adds on top.
    fn base_model_tensors(
        cfg: &Qwen3Config,
        dev: &Device,
    ) -> std::collections::HashMap<String, Tensor> {
        let mut t: Vec<(String, Tensor)> = vec![(
            "model.embed_tokens.weight".to_string(),
            Tensor::zeros((cfg.vocab_size, cfg.hidden_size), DType::F16, dev).unwrap(),
        )];
        for i in 0..cfg.num_hidden_layers {
            t.extend(decoder_layer_tensors(i, cfg, dev));
        }
        t.push((
            "model.norm.weight".to_string(),
            Tensor::zeros(cfg.hidden_size, DType::F16, dev).unwrap(),
        ));
        t.into_iter().collect()
    }

    /// A checkpoint that ties `lm_head` to `embed_tokens` (no `lm_head.*` tensors at all, same
    /// shape `__metadata__` would record for a real such checkpoint) must still load.
    ///
    /// **This is the red-first case.** Before `LmHead`/`LmHead::load` existed,
    /// `AwqQwen3::new` always built `AwqLinear::new(..., vb.pp("lm_head"))` unconditionally,
    /// which fails with a `qweight` shape/missing-tensor error on exactly this fixture --
    /// demonstrated by temporarily reverting the fix and re-running this test, which errors on
    /// `"Failed to load lm_head"` rather than succeeding.
    #[test]
    fn awq_qwen3_loads_a_tied_embedding_checkpoint_with_no_lm_head_tensors() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg(true);
        let tensors = base_model_tensors(&cfg, &dev);
        assert!(
            !tensors.keys().any(|k| k.starts_with("lm_head")),
            "fixture bug: a tied-checkpoint fixture must not itself carry lm_head tensors"
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F16, &dev);

        let model = AwqQwen3::new(&cfg, vb)
            .expect("a tied checkpoint (tie_word_embeddings=true, no lm_head.* tensors) must load");

        assert!(
            matches!(model.lm_head, LmHead::Tied(_)),
            "loaded, but not through the tied path -- the fixture or the detection logic has drifted"
        );
    }

    /// The untied, AWQ-quantized `lm_head` path must keep working -- same fixture plus its own
    /// `lm_head.*` tensors, `tie_word_embeddings: false`.
    #[test]
    fn awq_qwen3_loads_an_untied_checkpoint_with_its_own_lm_head() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg(false);
        let mut tensors = base_model_tensors(&cfg, &dev);
        tensors.extend(awq_linear_tensors(
            "lm_head",
            cfg.hidden_size,
            cfg.vocab_size,
            &dev,
        ));
        let vb = VarBuilder::from_tensors(tensors, DType::F16, &dev);

        let model = AwqQwen3::new(&cfg, vb)
            .expect("an untied checkpoint with its own lm_head.* tensors must load");

        assert!(
            matches!(model.lm_head, LmHead::Quantized(_)),
            "loaded, but not through the quantized path -- the fixture or the detection logic has drifted"
        );
    }

    /// Neither a quantized `lm_head` nor `tie_word_embeddings: true` names a genuine gap, not
    /// something to silently paper over with a guess.
    #[test]
    fn awq_qwen3_refuses_a_checkpoint_naming_neither_a_quantized_nor_a_tied_lm_head() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg(false);
        let tensors = base_model_tensors(&cfg, &dev);
        let vb = VarBuilder::from_tensors(tensors, DType::F16, &dev);

        let err = AwqQwen3::new(&cfg, vb).expect_err(
            "neither lm_head.qweight nor tie_word_embeddings=true is present; this must refuse, not guess",
        );
        assert!(
            err.to_string().contains("tie_word_embeddings"),
            "expected the refusal to name the missing signal, got: {err}"
        );
    }

    /// Every real Qwen3 dense checkpoint sets `head_dim` independently of
    /// `hidden_size`/`num_attention_heads` -- `Qwen/Qwen3-0.6B`'s `config.json` sets
    /// `hidden_size: 1024`, `num_attention_heads: 16` (formula: 64) and `head_dim: 128`
    /// explicitly; `Orion-zhen/Qwen3-0.6B-AWQ` (the AWQ slice-2 target, same base model)
    /// carries the identical override.
    ///
    /// **Red-first**: before `Qwen3Config::effective_head_dim` existed, `Qwen3RotaryEmbedding::
    /// new` and `Qwen3AwqAttention::new` each recomputed `hidden_size / num_attention_heads`
    /// directly, ignoring any `head_dim` override. On this fixture (`head_dim: Some(256)`,
    /// formula gives 64) that sized `q_proj`/`k_proj`/`v_proj`/`o_proj` for the wrong shape and
    /// `AwqQwen3::new` failed inside `Failed to load q_proj` with a `qweight` shape mismatch --
    /// demonstrated by reverting those two call sites (and `decoder_attn_tensors` above) back
    /// to the formula and re-running this test. This is the deeper reason slice 1's real
    /// target checkpoint still could not load once tied embeddings were handled.
    #[test]
    fn awq_qwen3_loads_a_checkpoint_whose_head_dim_disagrees_with_the_formula() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg_with_head_dim_override(true);
        assert_ne!(
            cfg.effective_head_dim(),
            cfg.hidden_size / cfg.num_attention_heads,
            "fixture bug: the override must disagree with the formula, or this test cannot \
             tell the fixed code apart from the bug"
        );
        let tensors = base_model_tensors(&cfg, &dev);
        let vb = VarBuilder::from_tensors(tensors, DType::F16, &dev);

        // Construction is the whole claim here: it is the one step that reads q_proj/k_proj/
        // v_proj/o_proj's shapes against `cfg`, which is exactly where the formula and the
        // override disagree. `forward` additionally requires CUDA (`AwqLinear`'s Marlin
        // kernel has no CPU path -- see `load_awq_llama`'s own `bail!`), which is this test's
        // reason to stay CPU-only rather than exercise it.
        let model = AwqQwen3::new(&cfg, vb).expect(
            "a checkpoint whose head_dim overrides the formula must still load -- every \
             attention/rotary call site must read cfg.effective_head_dim(), not recompute it",
        );
        assert!(matches!(model.lm_head, LmHead::Tied(_)));
    }
}
