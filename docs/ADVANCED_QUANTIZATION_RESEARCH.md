# Advanced Quantization (AWQ/GPTQ) Research & Design

**Date:** November 24, 2025  
**Status:** Research Phase  
**Target:** 2-4x memory reduction with minimal accuracy loss

---

## Executive Summary

**Goal:** Implement state-of-the-art 4-bit quantization (AWQ or GPTQ) to reduce model memory by 2-4x while maintaining 95%+ of FP16 accuracy.

**Expected Benefits:**
- 2-4x memory reduction (enable larger models on same GPU)
- 1.5-2x inference speedup (less memory bandwidth)
- Same or better accuracy than naive INT4

**Implementation Effort:** 4-6 weeks  
**Complexity:** High

---

## Background: Quantization Levels

### Current State: INT8 Support

Lightbulb already has basic quantization via `QuantizableLinear`:

```rust
pub enum QuantizationType {
    None,    // FP32 or FP16
    INT8,    // 8-bit integer quantization
    INT4,    // 4-bit integer quantization (naive)
}
```

**Performance:**
- **FP16:** Baseline (2 bytes/param)
- **INT8:** 2x memory reduction, ~5% accuracy loss
- **INT4:** 4x memory reduction, ~15-25% accuracy loss (not usable)

**Problem:** Naive INT4 loses too much accuracy for production use.

---

## Advanced Quantization Methods

### 1. AWQ (Activation-aware Weight Quantization)

**Paper:** "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"  
**Key Idea:** Protect important weights based on activation magnitudes.

**Algorithm:**
1. Analyze activation distributions (requires calibration data)
2. Identify "salient" channels (high activation magnitude)
3. Apply per-channel scaling to preserve important weights
4. Quantize to 4-bit with minimal error

**Pseudocode:**
```python
# Per-channel scaling
for channel in weights:
    activation_magnitude = analyze_activations(channel, calibration_data)
    scale = compute_scale(activation_magnitude)
    weights[channel] *= scale  # Protect important channels
    
# Quantize
quantized = round(weights / scale_factor).clamp(0, 15)  # 4-bit
```

**Characteristics:**
- ✅ Best accuracy (98-99% of FP16)
- ✅ Fast inference (optimized kernels available)
- ❌ Requires calibration data (128-512 samples)
- ❌ Per-channel scales increase memory slightly

### 2. GPTQ (Generative Pre-trained Transformer Quantization)

**Paper:** "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"  
**Key Idea:** Optimal Brain Quantization (OBQ) applied layer-by-layer.

**Algorithm:**
1. Start with one layer
2. For each weight group:
   - Quantize weights one-by-one
   - Minimize quantization error using Hessian
   - Update remaining weights to compensate
3. Move to next layer

**Characteristics:**
- ✅ Excellent accuracy (97-98% of FP16)
- ✅ No activation statistics needed (only Hessian)
- ❌ Slower quantization process (hours for 70B model)
- ❌ More complex implementation

### 3. Comparison

| Method      | Accuracy | Speed  | Calibration  | Complexity |
| ----------- | -------- | ------ | ------------ | ---------- |
| Naive INT4  | 75-85%   | Fast   | None         | Low        |
| **AWQ**     | 98-99%   | Fast   | 128+ samples | Medium     |
| **GPTQ**    | 97-98%   | Medium | Hessian only | High       |
| SmoothQuant | 95-97%   | Fast   | Activations  | Medium     |

**Recommendation:** Start with AWQ (best accuracy/complexity trade-off)

---

## AWQ Implementation Plan

### Phase 1: Calibration Data Collection

```rust
pub struct CalibrationDataset {
    samples: Vec<String>,  // Text samples for calibration
    tokenized: Vec<Vec<u32>>,
}

impl CalibrationDataset {
    pub fn from_pile(num_samples: usize) -> Result<Self> {
        // Use The Pile or C4 dataset
        // Diverse text: code, math, conversation, etc.
    }
    
    pub fn run_model(&self, model: &BatchedTransformer) -> ActivationStats {
        let mut stats = ActivationStats::new();
        
        for batch in self.tokenized.chunks(batch_size) {
            // Forward pass, collect activations
            let activations = model.forward_with_activations(batch)?;
            stats.accumulate(activations);
        }
        
        stats
    }
}
```

### Phase 2: Activation Analysis

```rust
pub struct ActivationStats {
    // Per-layer, per-channel statistics
    layer_stats: Vec<LayerStats>,
}

pub struct LayerStats {
    layer_idx: usize,
    channel_magnitudes: Vec<f32>,  // Max activation per channel
    channel_variance: Vec<f32>,
}

impl ActivationStats {
    pub fn compute_scales(&self) -> Vec<Vec<f32>> {
        // AWQ scaling formula:
        // s = (activation_magnitude)^α
        // where α ∈ [0, 1] (hyperparameter, typically 0.5)
        
        let alpha = 0.5;
        self.layer_stats.iter().map(|layer| {
            layer.channel_magnitudes.iter()
                .map(|mag| mag.powf(alpha))
                .collect()
        }).collect()
    }
}
```

### Phase 3: Weight Quantization

```rust
pub struct AwqQuantizer {
    scales: Vec<Vec<f32>>,  // Per-layer, per-channel
    zero_points: Vec<Vec<i8>>,
    group_size: usize,  // Typically 128
}

impl AwqQuantizer {
    pub fn quantize_linear(&self, linear: &Linear, layer_idx: usize) -> QuantizedLinear {
        let weights = linear.weight(); // [out_features, in_features]
        let scales = &self.scales[layer_idx];
        
        // Apply per-channel scaling
        let scaled_weights = weights * scales;
        
        // Group-wise quantization (128 elements per group)
        let quantized = self.quantize_grouped(scaled_weights, self.group_size);
        
        QuantizedLinear {
            qweight: quantized.weights,     // INT4 packed
            scales: quantized.scales,       // FP16
            zeros: quantized.zero_points,   // INT4
            group_size: self.group_size,
        }
    }
    
    fn quantize_grouped(&self, weights: &Tensor, group_size: usize) -> QuantizedWeights {
        // Quantize in groups of `group_size` elements
        // Each group has its own scale and zero-point
        // This preserves more accuracy than per-channel quantization
        
        let mut qweights = Vec::new();
        let mut qscales = Vec::new();
        let mut qzeros = Vec::new();
        
        for chunk in weights.chunks(group_size) {
            let min = chunk.min();
            let max = chunk.max();
            let scale = (max - min) / 15.0;  // 4-bit range [0, 15]
            let zero = min;
            
            let quantized = ((chunk - zero) / scale)
                .round()
                .clamp(0.0, 15.0);
            
            qweights.extend(quantized);
            qscales.push(scale);
            qzeros.push(zero);
        }
        
        QuantizedWeights { weights: pack_int4(qweights), scales: qscales, zero_points: qzeros }
    }
}
```

### Phase 4: Dequantization for Inference

```rust
impl QuantizedLinear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Option 1: Dequantize on-the-fly (simpler)
        let weights_fp16 = self.dequantize()?;
        x.matmul(&weights_fp16)
        
        // Option 2: INT4 GEMM (faster, requires custom kernel)
        self.int4_gemm(x)
    }
    
    fn dequantize(&self) -> Result<Tensor> {
        let mut weights = Vec::new();
        
        for (group_idx, group) in self.qweight.chunks(self.group_size / 2).enumerate() {
            // Unpack INT4 (2 weights per byte)
            let unpacked = unpack_int4(group);
            
            // Dequantize: w_fp16 = (w_int4 * scale) + zero
            let scale = self.scales[group_idx];
            let zero = self.zeros[group_idx];
            
            for w in unpacked {
                weights.push((w as f32 * scale) + zero);
            }
        }
        
        Tensor::from_vec(weights, self.shape(), &Device::Cpu)
    }
    
    fn int4_gemm(&self, x: &Tensor) -> Result<Tensor> {
        // Custom CUDA kernel for INT4 x FP16 matrix multiplication
        // Much faster than dequantize + FP16 GEMM
        // Requires: CUDA compute capability 7.5+ (Turing, Ampere, Hopper)
        
        #[cfg(feature = "cuda")]
        unsafe {
            crate::kernels::awq_gemm_forward(
                x.as_ptr(),
                self.qweight.as_ptr(),
                self.scales.as_ptr(),
                self.zeros.as_ptr(),
                self.group_size,
                x.shape(),
                self.shape(),
            )
        }
        
        #[cfg(not(feature = "cuda"))]
        self.dequantize()?.matmul(x)  // Fallback to CPU
    }
}
```

---

## CUDA Kernel Requirements

### INT4 GEMM Kernel

**Existing Implementations:**
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) - PyTorch, CUDA
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU SIMD + CUDA
- [ExLlamaV2](https://github.com/turboderp/exllamav2) - High-performance CUDA

**Key Optimizations:**
1. **Tensor Core acceleration** (INT4 → FP16 on Ampere+)
2. **Warp-level parallelism** (32 threads cooperate)
3. **Shared memory tiling** (reduce DRAM access)
4. **Vectorized loads** (load 8 INT4 weights at once)

**Development Options:**

#### Option A: Port from AutoAWQ
- **Effort:** 2-3 weeks
- **Pros:** Battle-tested, optimized
- **Cons:** PyTorch-specific, needs adaptation

#### Option B: Use llama.cpp kernels
- **Effort:** 1-2 weeks
- **Pros:** Already supports GGUF INT4, active development
- **Cons:** Different memory layout (GGUF vs safetensors)

#### Option C: Custom implementation
- **Effort:** 4-6 weeks
- **Pros:** Full control, optimized for Lightbulb
- **Cons:** High complexity, risk of bugs

**Recommendation:** Start with Option B (llama.cpp), optimize later if needed.

---

## Integration with Existing Code

### 1. Extend QuantizableLinear

```rust
pub enum QuantizationType {
    None,
    INT8,
    INT4Naive,
    AWQ { group_size: usize },   // NEW
    GPTQ { group_size: usize },  // NEW
}

pub struct QuantizableLinear {
    inner: QuantizableLinearInner,
    quant_type: QuantizationType,
}

enum QuantizableLinearInner {
    Full(Linear),           // FP16
    Int8(Int8Linear),
    Awq(AwqLinear),         // NEW
    Gptq(GptqLinear),       // NEW
}
```

### 2. Model Loading

```rust
impl ParallelModelManager {
    pub fn load_awq(
        model_path: &str,
        calibration_data: Option<CalibrationDataset>,
    ) -> Result<Self> {
        // Option 1: Load pre-quantized AWQ model
        if model_path.ends_with("-awq") {
            return Self::load_awq_checkpoint(model_path);
        }
        
        // Option 2: Quantize on-the-fly
        let fp16_model = Self::load(model_path, ...)?;
        
        let calib_data = calibration_data
            .unwrap_or_else(|| CalibrationDataset::default());
        
        let quantizer = AwqQuantizer::calibrate(&fp16_model, &calib_data)?;
        let awq_model = quantizer.quantize(fp16_model)?;
        
        Ok(awq_model)
    }
}
```

### 3. Benchmarking

```rust
#[bench]
fn bench_awq_vs_fp16(b: &mut Bencher) {
    let fp16_model = load_model("llama-7b", QuantizationType::None)?;
    let awq_model = load_model("llama-7b", QuantizationType::AWQ { group_size: 128 })?;
    
    let prompt = "The capital of France is";
    
    // Measure throughput
    b.iter(|| {
        fp16_model.generate(prompt, 50);
    });
    
    b.iter(|| {
        awq_model.generate(prompt, 50);
    });
}
```

---

## Memory Analysis

### Example: Llama-7B

**FP16 Baseline:**
- Weights: 7B params × 2 bytes = 14 GB
- KV cache (batch=16, seq=512): ~0.5 GB
- Activations: ~0.3 GB
- **Total:** ~15 GB

**AWQ INT4:**
- Weights: 7B params × 0.5 bytes = 3.5 GB
- Scales: 7B / 128 × 2 bytes = 109 MB (group_size=128)
- KV cache: ~0.5 GB (unchanged)
- Activations: ~0.3 GB
- **Total:** ~4.4 GB

**Savings:** 15 GB → 4.4 GB = **70% reduction**

### GPU Compatibility

| Model     | FP16   | AWQ INT4 | GPU Required (FP16) | GPU Required (AWQ) |
| --------- | ------ | -------- | ------------------- | ------------------ |
| Llama-7B  | 15 GB  | 4.4 GB   | A100 40GB           | RTX 3090 24GB      |
| Llama-13B | 27 GB  | 7.5 GB   | A100 40GB           | RTX 3090 24GB      |
| Llama-70B | 140 GB | 38 GB    | 2×A100 80GB         | A100 40GB          |

**Key Benefit:** Run 70B models on single A100 instead of 2×A100.

---

## Accuracy Validation

### Benchmarks to Track

1. **Perplexity:** Lower is better (measures prediction quality)
   - Target: <3% increase vs FP16
   
2. **MMLU** (Massive Multitask Language Understanding)
   - Target: <1% drop vs FP16
   
3. **HumanEval** (Code generation)
   - Target: <2% drop vs FP16

4. **Task-specific:** Domain accuracy (math, reasoning, etc.)

### Validation Process

```python
# Pseudocode
fp16_model = load_model("llama-7b-fp16")
awq_model = load_model("llama-7b-awq")

for benchmark in [perplexity, mmlu, humaneval]:
    fp16_score = benchmark.evaluate(fp16_model)
    awq_score = benchmark.evaluate(awq_model)
    
    degradation = (fp16_score - awq_score) / fp16_score
    
    assert degradation < 0.03, f"Accuracy drop too large: {degradation:.1%}"
```

---

## Timeline

### Week 1-2: Research & Setup
- [ ] Study AutoAWQ implementation
- [ ] Set up calibration data pipeline
- [ ] Design Rust API for AWQ

### Week 3: Calibration
- [ ] Implement activation statistics collection
- [ ] Implement scale computation
- [ ] Test on small model (Llama-1B)

### Week 4: Quantization
- [ ] Implement grouped quantization
- [ ] Test accuracy on Llama-7B
- [ ] Compare vs FP16 baseline

### Week 5: Kernel Integration
- [ ] Port INT4 GEMM from llama.cpp or AutoAWQ
- [ ] Benchmark inference speed
- [ ] Optimize memory layout

### Week 6: Production Readiness
- [ ] Add model loading for pre-quantized checkpoints
- [ ] Write documentation
- [ ] Create migration guide

---

## Risks & Mitigation

### Risk 1: Accuracy Degradation
**Impact:** High  
**Mitigation:**
- Use high-quality calibration data (diverse, representative)
- Tune hyperparameters (α, group_size)
- Fall back to FP16 for critical layers (embeddings, LM head)

### Risk 2: CUDA Kernel Complexity
**Impact:** Medium  
**Mitigation:**
- Start with dequantize + FP16 GEMM (slower but simpler)
- Port proven kernels (llama.cpp, AutoAWQ)
- Use Candle's kernel infrastructure when available

### Risk 3: Calibration Data Quality
**Impact:** Medium  
**Mitigation:**
- Use standard datasets (C4, The Pile)
- Sample uniformly across domains
- Validate on hold-out set

---

## Alternative: GGUF Support

**GGUF** (GPT-Generated Unified Format) already includes quantization:
- INT4, INT5, INT8 variants
- Used by llama.cpp ecosystem
- Mature, battle-tested

**Trade-off:**
- ✅ Easier integration (llama.cpp kernels)
- ✅ Large model zoo (Hugging Face)
- ❌ Less flexibility than custom AWQ
- ❌ GGUF-specific format (not safetensors)

**Decision:** Support both
1. AWQ for safetensors models (Hugging Face standard)
2. GGUF for llama.cpp compatibility

---

## Success Criteria

### Phase 1 (Calibration)
- ✅ Activation statistics collection works
- ✅ Scales computed for all layers
- ✅ No accuracy regression on toy model

### Phase 2 (Quantization)
- ✅ 4x memory reduction achieved
- ✅ <3% perplexity increase
- ✅ Inference runs (even if slow)

### Phase 3 (Optimization)
- ✅ 1.5-2x speedup vs FP16
- ✅ Production-ready API
- ✅ Documentation complete

---

## References

### Papers
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- [LLM.int8(): 8-bit Matrix Multiplication](https://arxiv.org/abs/2208.07339)

### Implementations
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [ExLlamaV2](https://github.com/turboderp/exllamav2)
- [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)

### Datasets
- [C4 (Colossal Clean Crawled Corpus)](https://huggingface.co/datasets/c4)
- [The Pile](https://pile.eleuther.ai/)

---

**Next Steps:**
1. Set up calibration data pipeline (C4 or Pile)
2. Implement activation statistics collection
3. Port AWQ quantization logic from AutoAWQ
4. Benchmark accuracy vs FP16

**Status:** 📋 Design complete, awaiting implementation prioritization
