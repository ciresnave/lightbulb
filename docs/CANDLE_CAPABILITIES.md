# Candle Capabilities Reference

**Purpose**: Comprehensive guide to Candle's built-in capabilities to avoid reinventing wheels.

**Last Updated**: Phase 2D implementation (KV cache integration)

---

## Table of Contents
1. [candle-core](#candle-core) - Tensor operations and devices
2. [candle-nn](#candle-nn) - Neural network layers and utilities
3. [candle-transformers](#candle-transformers) - Pre-built transformer models
4. [Decision Framework](#decision-framework) - When to use vs build

---

## candle-core

### Tensor Operations

**Basic Operations**:
- `matmul()` - Matrix multiplication
- `add()`, `sub()`, `mul()`, `div()` - Element-wise arithmetic
- `reshape()` - Change tensor shape
- `transpose()` - Swap dimensions
- `narrow()` - Extract slice along dimension
- `cat()` - Concatenate tensors
- `stack()` - Stack tensors along new dimension
- `contiguous()` - Ensure memory layout is contiguous

**Indexing**:
- `i(index)` - Index into first dimension
- `IndexOp` trait - For efficient `tensor.i(idx)` operations
- `broadcast_*()` - Broadcasting operations

**Aggregation**:
- `sum()`, `mean()` - Reduce operations
- `argmax()`, `argmin()` - Find indices
- `max()`, `min()` - Maximum/minimum values

### Devices

**Available Devices**:
```rust
use candle_core::Device;

// CPU device (always available)
let device = Device::Cpu;

// CUDA device (requires cuda feature)
let device = Device::new_cuda(0)?;  // GPU 0

// Metal device (macOS, requires metal feature)
let device = Device::new_metal(0)?;
```

**Device Transfer**:
```rust
let tensor_on_cpu = tensor.to_device(&Device::Cpu)?;
let tensor_on_gpu = tensor.to_device(&Device::new_cuda(0)?)?;
```

### Data Types (DTypes)

**Supported Types**:
- `DType::F32` - 32-bit float (most common)
- `DType::F16` - 16-bit float (memory efficient)
- `DType::BF16` - Brain float 16 (better range than F16)
- `DType::F64` - 64-bit float (high precision)
- `DType::U32`, `DType::I64` - Integer types
- Quantized types: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`

**Type Conversion**:
```rust
let tensor_f16 = tensor.to_dtype(DType::F16)?;
let tensor_bf16 = tensor.to_dtype(DType::BF16)?;
```

---

## candle-nn

### Linear Layers

**Linear (Fully Connected)**:
```rust
use candle_nn::{Linear, linear};

// From VarBuilder (recommended)
let layer = linear(in_features, out_features, vb.pp("layer_name"))?;

// Manual construction
let layer = Linear::new(weight, Some(bias));

// Forward pass
let output = layer.forward(&input)?;  // [batch, in] -> [batch, out]
```

### Normalization Layers

**LayerNorm**:
```rust
use candle_nn::{LayerNorm, layer_norm};

let norm = layer_norm(normalized_shape, eps, vb.pp("norm"))?;
let output = norm.forward(&input)?;
```

**RMSNorm** (Root Mean Square Norm - used in Llama):
```rust
use candle_nn::{RmsNorm, rms_norm};

let norm = rms_norm(size, eps, vb.pp("rms_norm"))?;
let output = norm.forward(&input)?;
```

**BatchNorm**:
```rust
use candle_nn::{BatchNorm, batch_norm};

let bn = batch_norm(num_features, momentum, eps, vb.pp("bn"))?;
let output = bn.forward(&input)?;
```

### Activation Functions

**Available Activations**:
```rust
use candle_nn::{Activation, ops};

// ReLU
let output = ops::relu(&input)?;

// Leaky ReLU
let output = ops::leaky_relu(&input, negative_slope)?;

// GELU (Gaussian Error Linear Unit)
let output = ops::gelu(&input)?;

// SiLU (Swish)
let output = ops::silu(&input)?;

// SwiGLU (used in Llama)
let output = ops::swiglu(&input)?;

// Softmax
let output = ops::softmax(&input, dim)?;
let output = ops::softmax_last_dim(&input)?;

// Log Softmax
let output = ops::log_softmax(&input, dim)?;

// Sigmoid
let output = ops::sigmoid(&input)?;

// Hard Sigmoid
let output = ops::hard_sigmoid(&input)?;

// Tanh
let output = input.tanh()?;
```

**PReLU** (Parametric ReLU):
```rust
use candle_nn::{prelu, PReLU};

let prelu = prelu(num_parameters, vb.pp("prelu"))?;
let output = prelu.forward(&input)?;
```

### Convolutional Layers

**Conv1d**:
```rust
use candle_nn::Conv1d;

let conv = Conv1d::new(weight, bias, kernel_size, stride, padding, dilation, groups);
let output = conv.forward(&input)?;
```

**Conv2d**:
```rust
use candle_nn::Conv2d;

let conv = Conv2d::new(weight, bias, kernel_size, stride, padding, dilation, groups);
let output = conv.forward(&input)?;
```

### Pooling Layers

**Pixel Shuffle/Unshuffle**:
```rust
use candle_nn::ops::{pixel_shuffle, pixel_unshuffle};

// Pixel shuffle (depth to space)
let output = pixel_shuffle(&input, upscale_factor)?;

// Pixel unshuffle (space to depth)
let output = pixel_unshuffle(&input, downscale_factor)?;
```

### Recurrent Layers (RNN/LSTM/GRU) ⭐

**LSTM** (Long Short-Term Memory):
```rust
use candle_nn::{lstm, LSTM, LSTMConfig, LSTMState, RNN};

// Create LSTM layer
let config = LSTMConfig {
    layer_idx: 0,
    ..Default::default()
};
let lstm = lstm(input_size, hidden_size, config, vb.pp("lstm"))?;

// Initialize state
let mut state = LSTMState::new(batch_size, hidden_size, &device)?;

// Forward pass
for input_t in inputs {
    state = lstm.step(&input_t, &state)?;
}
```

**GRU** (Gated Recurrent Unit):
```rust
use candle_nn::{gru, GRU, GRUConfig, GRUState, RNN};

let config = GRUConfig {
    layer_idx: 0,
    ..Default::default()
};
let gru = gru(input_size, hidden_size, config, vb.pp("gru"))?;

let mut state = GRUState::new(batch_size, hidden_size, &device)?;
for input_t in inputs {
    state = gru.step(&input_t, &state)?;
}
```

**Features**:
- ✅ Bidirectional support (via `Direction` enum)
- ✅ Multi-layer stacking
- ✅ Configurable dropout
- ✅ State management

### Embedding Layers

**Embedding**:
```rust
use candle_nn::{Embedding, embedding};

let embed = embedding(vocab_size, hidden_dim, vb.pp("embed"))?;
let output = embed.forward(&token_ids)?;  // [batch, seq] -> [batch, seq, hidden]
```

### Loss Functions ⭐

**Cross-Entropy Loss**:
```rust
use candle_nn::loss::cross_entropy;

// For classification tasks
// logits: [batch, num_classes], targets: [batch]
let loss = cross_entropy(&logits, &targets)?;
```

**Mean Squared Error (MSE)**:
```rust
use candle_nn::loss::mse;

// For regression tasks
// predictions: [batch, ...], targets: [batch, ...]
let loss = mse(&predictions, &targets)?;
```

**Negative Log Likelihood (NLL)**:
```rust
use candle_nn::loss::nll;

// log_probs: [batch, num_classes], targets: [batch]
let loss = nll(&log_probs, &targets)?;
```

**Binary Cross-Entropy with Logits**:
```rust
use candle_nn::loss::binary_cross_entropy_with_logit;

// For binary classification
// logits: [batch], targets: [batch]  (values 0 or 1)
let loss = binary_cross_entropy_with_logit(&logits, &targets)?;
```

### Rotary Position Embeddings (RoPE) ⭐

**Primary Implementation** (Use this!):
```rust
use candle_nn::rotary_emb::rope;

// Apply RoPE to query/key tensors
// x: [batch, num_heads, seq_len, head_dim]
// cos/sin: [max_seq_len, head_dim]
let x_rope = rope(&x, &cos_slice, &sin_slice)?;
```

**Variants**:
- `rope()` - Standard RoPE implementation
- `rope_i()` - Interleaved variant (different rotation pattern)
- `rope_thd()` - Thread-optimized variant

**Features**:
- ✅ CPU parallelization with Rayon
- ✅ CUDA kernel support for GPU
- ✅ Works with F16, BF16, F32, F64
- ✅ Batched processing
- ✅ Used by all Candle transformer models

**Formula**: For each pair of dimensions:
```
y0 = x0*cos - x1*sin
y1 = x0*sin + x1*cos
```

**Example Usage** (from our BatchedAttention):
```rust
fn apply_rotary_emb(
    &self,
    x: &Tensor,
    index_pos: usize,
    seq_len: usize,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    // Extract position range
    let cos_slice = cos.narrow(0, index_pos, seq_len)?;
    let sin_slice = sin.narrow(0, index_pos, seq_len)?;
    
    // Apply Candle's optimized RoPE
    Ok(candle_nn::rotary_emb::rope(x, &cos_slice, &sin_slice)?)
}
```

**Cost Savings**: ~200 lines of custom implementation avoided! ⭐

### Optimizers

**SGD**:
```rust
use candle_nn::{SGD, ParamsProxies};

let optimizer = SGD::new(learning_rate, momentum)?;
optimizer.step(&loss, &mut params)?;
```

**Adam/AdamW**:
```rust
use candle_nn::AdamW;

let optimizer = AdamW::new(learning_rate, beta1, beta2, weight_decay)?;
optimizer.step(&loss, &mut params)?;
```

### VarBuilder (Weight Loading)

**From Safetensors**:
```rust
use candle_nn::VarBuilder;

let vb = VarBuilder::from_safetensors(&paths, dtype, device)?;

// Load specific layer
let linear = linear(in_features, out_features, vb.pp("model.layers.0.mlp.fc1"))?;
```

**From Memory**:
```rust
let vb = VarBuilder::from_varmap(&varmap, dtype, device);
```

### Additional Utilities

**Dropout**:
```rust
use candle_nn::{Dropout, ops::dropout};

// During training
let output = dropout(&input, dropout_prob)?;

// With struct
let dropout = Dropout::new(dropout_prob);
let output = dropout.forward(&input, train)?;  // train: bool
```

**Sequential Layer**:
```rust
use candle_nn::{seq, Sequential};

let model = seq()
    .add(linear1)
    .add_fn(|x| x.relu())
    .add(linear2);

let output = model.forward(&input)?;
```

**Encoding Utilities**:
```rust
use candle_nn::encoding;

// One-hot encoding
let one_hot = encoding::one_hot(indices, num_classes, &device)?;
```

---

## candle-transformers

### Position Embeddings

**RoPE (Rotary Position Embeddings)** ✅:
- Location: `candle_nn::rotary_emb`
- Usage: See detailed section above
- Status: **USE THIS** - Don't rebuild!

**ALiBi (Attention with Linear Biases)** ✅:
- Location: `candle_transformers::models::*::alibi`
- Purpose: Alternative to positional embeddings
- Used by: MPT, Bloom models

**Sinusoidal Position Embeddings** ✅:
- Location: Standard in transformer models
- Purpose: Original Transformer positional encoding
- Formula: sin/cos at different frequencies

**Learned Position Embeddings**:
- Just use standard `Embedding` layer
- Common in BERT-style models

### Attention Patterns

**CausalSelfAttention** (Standard):
```rust
// From Llama model
pub struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,  // For GQA
    head_dim: usize,
}
```

**Multi-Query Attention (MQA)**:
- `num_kv_heads = 1`
- All query heads share single K/V head
- Used by: Falcon, StarCoder

**Grouped Query Attention (GQA)**:
- `num_kv_heads < num_heads`
- Query heads grouped to share K/V heads
- Used by: Llama-2, Llama-3, Mistral
- Example: 32 query heads, 8 KV heads (4:1 ratio)

### KV Cache

**Cache Struct** (Standard Llama):
```rust
use candle_transformers::models::llama::Cache;

pub struct Cache {
    cos: Tensor,  // Pre-computed cosine for RoPE
    sin: Tensor,  // Pre-computed sine for RoPE
    // ... (kv cache internals are private)
}

impl Cache {
    pub fn new(use_kv_cache: bool, config: &Config, device: &Device) -> Result<Self>;
}
```

**ScatteredKvCache** (Batched Inference):
```rust
use candle_nn::kv_cache::{ScatteredCacheBuilder, ScatteredKvCache, IndicesAndMask};

// Builder
let cache_builder = ScatteredCacheBuilder::new(
    max_batch_size,
    max_seq_len,
    num_heads,
    head_dim,
    dtype,
    device,
)?;

// Create caches (one per layer)
let cache: ScatteredKvCache = cache_builder.build(layer_idx)?;

// Get indices and mask
let iam: IndicesAndMask = cache_builder.indices_and_mask(seq_len, &batch_mask)?;

// Append K/V and get full history
let (k_full, v_full) = cache.append(&k, &v, &iam)?;
```

### Pre-built Models

**Available Models** (all in `candle-transformers/src/models/`):

**Language Models** (80+ models):
- **Llama family**: `llama`, `llama2_c`, `quantized_llama`
- **Mistral/Mixtral**: `mistral`, `mixtral` (MoE)
- **Phi**: `phi`, `phi3`, `quantized_phi`, `quantized_phi3`
- **Qwen**: `qwen2`, `qwen2_moe`, `quantized_qwen2`
- **Gemma**: `gemma`, `gemma2`, `gemma3`, `recurrent_gemma`, `quantized_gemma3`
- **Mamba**: `mamba` (State Space Models)
- **GPT family**: `bigcode`, `starcoder2`
- **BERT family**: `bert`, `distilbert`, `modernbert`, `jina_bert`, `debertav2`
- **Granite**: Long context transformer
- **Deepseek**: `deepseek2`
- **GLM**: `glm4`, `chatglm`
- **Yi, OLMo, Falcon, Helium, Persimmon**, and many more!

**Vision-Language Models**:
- **CLIP**: `clip`, `openclip`, `mobileclip`, `chinese_clip`, `siglip`
- **BLIP**: `blip`, `blip_text`, `quantized_blip`
- **LLaVA**: Multimodal language + vision
- **Paligemma**: Gemma + SigLIP
- **Moondream**: Vision-to-text
- **Pixtral**: Language-Image Pre-Training
- **Colpali**: Text/image similarity

**Vision Models**:
- **Transformers**: `vit`, `beit`, `dinov2`, `dinov2reg4`, `eva2`
- **CNNs**: `resnet`, `convnext`, `efficientnet`, `mobilenetv4`, `vgg`, `repvgg`
- **Efficient**: `efficientvit`, `fastvit`, `mobileone`
- **Segmentation**: `segformer`, `segment_anything`, `depth_anything_v2`
- **Detection**: Object detection utilities
- **ConvMixer**, `hiera`, and more!

**Audio Models**:
- **Whisper**: Speech recognition ⭐
- **Encodec**: Neural audio codec
- **DAC**: Descript Audio Codec
- **SNAC**: Multi-Scale Neural Audio Codec
- **Mimi**: Audio model
- **MetaVoice**: Text-to-speech
- **Parler TTS**: Text-to-speech synthesis
- **CSM**: Conversational Speech Model

**Diffusion Models**:
- **Stable Diffusion**: Text-to-image ⭐
- **Flux**: Diffusion model
- **Wuerstchen**: Efficient diffusion
- **MMDit**: Multi-scale dilated convolutions

**Encoder-Decoder**:
- **T5**: Text-to-text transfer
- **Marian**: Neural machine translation
- **TrOCR**: Optical character recognition

**Quantized Variants** (GGML/GGUF support):
- Quantized versions of most major models
- `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` formats
- Significant memory savings (4-8x)

**Recurrent Models**:
- **RWKV**: `rwkv_v5`, `rwkv_v6`, `quantized_rwkv_v5`, `quantized_rwkv_v6`
- **GRU/LSTM**: Via candle-nn

**Embedding Models**:
- **NVEmbed**: `nvembed_v2`
- **Stella**: `stella_en_v5`
- Various BERT variants

**Common Pattern**:
```rust
use candle_transformers::models::llama::{Llama, Config};

// Load model
let config = Config::from_json(path)?;
let vb = VarBuilder::from_safetensors(&weight_paths, dtype, device)?;
let model = Llama::load(vb, &config)?;

// Forward pass
let logits = model.forward(&tokens, position, &mut cache)?;
```

### Tokenizers

**Integration**:
```rust
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_file(path)?;

// Encode
let tokens = tokenizer.encode(text, add_special_tokens)?;
let token_ids: Vec<u32> = tokens.get_ids().to_vec();

// Decode
let text = tokenizer.decode(&token_ids, skip_special_tokens)?;
```

### Generation & Sampling ⭐

**LogitsProcessor** (Temperature, Top-K, Top-P):
```rust
use candle_transformers::generation::{LogitsProcessor, Sampling};

// Create processor with sampling strategy
let logits_processor = LogitsProcessor::new(
    seed,
    Some(temperature),      // Temperature scaling (e.g., 0.8)
    Some(top_p),            // Nucleus sampling (e.g., 0.95)
);

// Sample next token
let next_token = logits_processor.sample(&logits)?;
```

**Sampling Strategies**:
```rust
// Greedy (argmax)
let sampling = Sampling::ArgMax;

// Multinomial with temperature
let sampling = Sampling::TopK { k: 50, temperature: 0.8 };

// Nucleus (top-p) sampling
let sampling = Sampling::TopP { p: 0.95, temperature: 0.8 };

// Combined top-k and top-p
let sampling = Sampling::TopKThenTopP { 
    k: 50, 
    p: 0.95, 
    temperature: 0.8 
};
```

**Repeat Penalty** (from `candle-transformers::utils`):
```rust
use candle_transformers::utils::{apply_repeat_penalty};

// Penalize repeated tokens
let logits = apply_repeat_penalty(&logits, penalty, &context_tokens)?;
```

**Repeat K/V for GQA** (Group Query Attention):
```rust
use candle_transformers::utils::repeat_kv;

// Repeat K/V heads for grouped query attention
// From num_kv_heads to num_heads
let k = repeat_kv(k, num_heads / num_kv_heads)?;
let v = repeat_kv(v, num_heads / num_kv_heads)?;
```

---

## Decision Framework

### ✅ USE (Don't Rebuild)

**Core Operations**:
- Tensor operations (matmul, reshape, etc.)
- Device management (CPU/CUDA/Metal)
- Data type conversions

**Layers**:
- ✅ **RoPE** (rotary_emb module) - ~200 lines saved! ⭐
- ✅ Linear layers
- ✅ Normalization (LayerNorm, RMSNorm, BatchNorm, GroupNorm)
- ✅ Activations (ReLU, LeakyReLU, GELU, SiLU, SwiGLU, Softmax, PReLU, etc.)
- ✅ Embeddings
- ✅ Convolutions (Conv1d, Conv2d, ConvTranspose)
- ✅ **RNN/LSTM/GRU** - Full recurrent layer support ⭐
- ✅ Dropout, Sequential, Pixel shuffle/unshuffle

**Loss Functions**:
- ✅ Cross-entropy, MSE, NLL, Binary cross-entropy ⭐
- All standard ML loss functions provided

**Position Embeddings**:
- ✅ RoPE (use candle-nn implementation) ⭐
- ✅ ALiBi (if needed)
- ✅ Sinusoidal (if needed)

**Generation & Sampling**:
- ✅ **LogitsProcessor** - Temperature, Top-K, Top-P, combined strategies ⭐
- ✅ Repeat penalty, repeat_kv for GQA ⭐
- Don't rebuild sampling logic!

**Pre-built Models** (80+ available):
- ✅ Use existing models as reference
- ✅ Quantized variants for memory efficiency
- ✅ 80+ models across all modalities

### ⚠️ ADAPT (Modify for Batching)

**Attention Layers**:
- Standard CausalSelfAttention is sequential
- Need custom BatchedAttention for true batching
- Keep structure, add batched processing

**Transformer Blocks**:
- Standard blocks process one request at a time
- Need BatchedTransformerBlock for parallel processing
- Reuse LayerNorm, activations, etc.

**KV Cache**:
- Standard Cache works for single requests
- ScatteredKvCache for batched inference ✅
- Use our BatchExecutor wrapper

### 🔨 BUILD (Custom Implementation)

**Batched Interfaces**:
- BatchedAttention (✅ complete with KV cache)
- BatchedMLP (⏳ next)
- BatchedTransformerBlock (⏸️ planned)
- BatchedLlama (⏸️ planned)

**Scheduling & Orchestration**:
- BatchMetadata (✅ complete)
- BatchExecutor (✅ complete)
- Request management
- Continuous batching logic

---

## Integration Examples

### Example 1: Using Candle's RoPE (Our Approach)

**Before** (What we almost did):
```rust
// ~200 lines of custom RoPE implementation
fn apply_rope_custom(x, freqs_cos, freqs_sin) -> Tensor {
    // Manual rotation: y0 = x0*cos - x1*sin, y1 = x0*sin + x1*cos
    // Handle reshaping, indexing, etc.
    // ...
}
```

**After** (Using Candle):
```rust
// ~30 lines using built-in
fn apply_rotary_emb(&self, x, index_pos, seq_len, cos, sin) -> Result<Tensor> {
    let cos_slice = cos.narrow(0, index_pos, seq_len)?;
    let sin_slice = sin.narrow(0, index_pos, seq_len)?;
    Ok(candle_nn::rotary_emb::rope(x, &cos_slice, &sin_slice)?)
}
```

**Savings**: ~170 lines, optimized CPU/GPU kernels, battle-tested! ⭐

### Example 2: Building Batched Attention (Adaptation)

**Standard Llama Attention** (Sequential):
```rust
// Process one request at a time
for req in batch {
    let q = self.q_proj.forward(&hidden_states)?;
    let k = self.k_proj.forward(&hidden_states)?;
    let v = self.v_proj.forward(&hidden_states)?;
    // ...
}
```

**Our BatchedAttention** (Parallel):
```rust
// Process entire batch at once
// Input: [batch_size, seq_len, hidden_size]
let q = self.q_proj.forward(hidden_states)?;
let k = self.k_proj.forward(hidden_states)?;
let v = self.v_proj.forward(hidden_states)?;

// Reshape for multi-head: [batch, seq, hidden] -> [batch, heads, seq, dim]
let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

// Apply RoPE (using Candle!)
let q = self.apply_rotary_emb(&q, index_pos, seq_len, cos, sin)?;
let k = self.apply_rotary_emb(&k, index_pos, seq_len, cos, sin)?;

// KV cache integration
let iam = batch_executor.get_indices_and_mask_simple(batch_size, seq_len)?;
let (k_full, v_full) = batch_executor.append_kv(layer_idx, &k, &v, &iam)?;

// Compute attention on full batch
let attn_output = self.compute_attention(&q, &k_full, &v_full)?;
```

**Key Differences**:
- Batched input/output tensors
- Multi-head reshaping with batch dimension
- Batched RoPE application
- ScatteredKvCache for batched K/V storage
- Single attention computation for entire batch

### Example 3: Creating Batched MLP (Next Task)

**Standard MLP** (From Llama):
```rust
pub struct MLP {
    gate_proj: Linear,  // hidden -> intermediate
    up_proj: Linear,    // hidden -> intermediate
    down_proj: Linear,  // intermediate -> hidden
}

impl MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}
```

**Batched MLP** (Our approach):
```rust
pub struct BatchedMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl BatchedMLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Input: [batch, seq, hidden]
        let gate = ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
        // Output: [batch, seq, hidden]
        // Same as standard, but processes entire batch at once!
    }
}
```

**Note**: Candle's Linear layer already handles batched inputs naturally!
- Input: [batch, seq, in_features]
- Output: [batch, seq, out_features]
- No changes needed, just pass batched tensors!

---

## Performance Considerations

### When Candle Optimizations Apply

**CPU Parallelization** (Rayon):
- RoPE operations
- Matrix multiplications (MKL/OpenBLAS)
- Element-wise operations

**GPU Acceleration** (CUDA):
- All tensor operations when on CUDA device
- Custom kernels for RoPE
- cuBLAS for matrix multiplications

**Memory Efficiency**:
- F16/BF16 for 2x memory savings
- Quantization (Q4, Q8) for 4-8x savings
- ScatteredKvCache for efficient batched storage

### Batching Benefits

**Standard Llama** (Sequential):
```
Request 1: 185ms
Request 2: 185ms
Request 3: 185ms
Total: 555ms (0.54 tok/s per request)
```

**BatchedLlama** (Approach 2 target):
```
Batch of 3: ~180ms total
Per-request: 60ms effective
Speedup: 3.08x (or 6x for batch of 6)
```

**Why?**:
- Amortize model loading overhead
- Efficient GPU utilization
- Parallelized matrix operations
- Shared computation (embeddings, norms)

---

## Common Pitfalls

### ❌ Don't Reinvent
```rust
// DON'T: Custom RoPE (200 lines, slower)
fn my_rope_implementation(...) { ... }

// DO: Use Candle's RoPE
let x_rope = candle_nn::rotary_emb::rope(&x, &cos, &sin)?;
```

### ❌ Don't Assume Single Batch
```rust
// DON'T: Hardcode batch_size = 1
let hidden = input.reshape((1, seq_len, hidden_size))?;

// DO: Get actual batch size
let (batch_size, seq_len, _) = input.dims3()?;
let hidden = input.reshape((batch_size, seq_len, hidden_size))?;
```

### ✅ Do Use Contiguous
```rust
// After transpose or complex operations:
let x = x.transpose(1, 2)?.contiguous()?;
// Ensures memory layout is optimal for next operation
```

### ✅ Do Pre-compute When Possible
```rust
// Pre-compute RoPE cos/sin (Cache does this)
let cache = Cache::new(use_kv_cache, &config, &device)?;
// cos/sin are computed once, reused for all tokens

// Use in forward pass:
let q_rope = rope(&q, &cache.cos.narrow(...)?, &cache.sin.narrow(...)?)?;
```

---

## Quick Reference: What We're Using

### Phase 2D Implementation Status

**✅ Using from Candle**:
- `candle_core::Tensor` - All tensor operations
- `candle_core::Device` - CPU device
- `candle_nn::Linear` - Q/K/V/O projections
- `candle_nn::RmsNorm` - Layer normalization
- `candle_nn::rotary_emb::rope` - Position embeddings ⭐
- `candle_nn::kv_cache::ScatteredKvCache` - Batched KV storage
- `candle_nn::ops::silu` - Activation functions

**🔨 Building Custom**:
- ✅ `BatchedAttention` - Batched multi-head attention (332 lines)
- ⏳ `BatchedMLP` - Batched feed-forward (~150 lines)
- ⏸️ `BatchedTransformerBlock` - Full transformer layer (~250 lines)
- ⏸️ `BatchedLlama` - Complete batched model (~600 lines)

**📦 Infrastructure**:
- ✅ `BatchMetadata` - Batch structure description (313 lines)
- ✅ `BatchExecutor` - KV cache management
- ✅ `RequestContext` - Request state tracking

---

## Resources

**Candle Documentation**:
- GitHub: https://github.com/huggingface/candle
- Book: https://huggingface.github.io/candle/
- Examples: `candle/candle-examples/`

**Source Code** (Best Documentation):
- candle-core: `candle/candle-core/src/`
- candle-nn: `candle/candle-nn/src/`
- candle-transformers: `candle/candle-transformers/src/models/`

**Model Implementations**:
- Llama: `candle-transformers/src/models/llama.rs`
- Attention patterns: Look at various models for MQA/GQA examples
- RoPE: `candle-nn/src/rotary_emb.rs`

---

## Summary

**Key Takeaways**:

1. **Don't Rebuild What Exists** ⭐
   - Candle provides RoPE, normalization, activations, loss functions, RNNs, sampling, etc.
   - Always check `candle-nn` and `candle-transformers` first
   - Saved ~200 lines by using built-in RoPE!
   - **NEW**: Loss functions, LSTM/GRU, LogitsProcessor all ready to use!

2. **Adapt for Batching**
   - Standard layers work with batched inputs
   - Attention/Transformer need custom batched versions
   - Keep using Candle's primitives (Linear, RMSNorm, etc.)

3. **Build Infrastructure**
   - Scheduling logic (BatchMetadata, BatchExecutor)
   - Request management
   - Continuous batching orchestration

4. **Leverage Optimizations**
   - Candle handles CPU/GPU dispatch
   - RoPE has optimized kernels
   - ScatteredKvCache for efficient batched storage
   - **NEW**: 80+ pre-built models as references!

5. **Use Generation Utilities** ⭐
   - LogitsProcessor for all sampling strategies
   - Don't rebuild temperature/top-k/top-p logic
   - Repeat penalty and GQA utilities provided

**Result**: Fast development, optimized performance, battle-tested components! 🚀

---

## Additional Candle Crates

### candle-datasets
- MNIST, CIFAR-10, ImageNet loaders
- Common dataset utilities
- [docs.rs/candle-datasets](https://docs.rs/candle-datasets/)

### candle-onnx
- Load and run ONNX models
- Convert between formats
- [docs.rs/candle-onnx](https://docs.rs/candle-onnx/)

### candle-pyo3
- Python bindings for Candle
- Access Candle from Python
- [docs.rs/candle-pyo3](https://docs.rs/candle-pyo3/)

### candle-flash-attn (if available)
- Flash Attention implementations
- Optimized attention kernels
- Check feature flags

---

## Updates Since Initial Version

**Major Additions** ✅:
1. **Loss Functions**: Cross-entropy, MSE, NLL, Binary cross-entropy
2. **RNN/LSTM/GRU**: Full recurrent layer support with bidirectional, multi-layer
3. **Extended Activations**: LeakyReLU, SwiGLU, Hard Sigmoid, PReLU, Log Softmax
4. **Generation & Sampling**: LogitsProcessor with temperature/top-k/top-p/combined
5. **80+ Models**: Comprehensive list across all modalities (LLM, vision, audio, diffusion)
6. **Utilities**: Dropout, Sequential, Pixel shuffle, Encoding, Repeat penalty
7. **Pooling**: Pixel shuffle/unshuffle operations
8. **Other Crates**: candle-datasets, candle-onnx, candle-pyo3 references

**Lines Saved**: ~200 (RoPE) + ~100 (sampling) + ~50 (loss functions) + ~200 (LSTM/GRU if needed) = **~550 lines** by using Candle! ⭐
