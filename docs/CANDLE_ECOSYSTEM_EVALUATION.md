# Candle Ecosystem Crate Evaluation for Lightbulb

**Date**: October 2025  
**Context**: M3.5 Testing & Hardening - evaluating community crates for potential integration

## Executive Summary

Evaluated 12 Candle ecosystem crates for potential integration into Lightbulb. **Recommendations**:

### ✅ High Priority (Immediate Value)
1. **candle-layer-norm** - Fused LayerNorm/RMSNorm for GPU
2. **candle-ext** - Missing PyTorch operations

### 🟡 Medium Priority (Future Value)
3. **candle-einops** - Tensor reshaping sugar (nice-to-have)
4. **candle-optimisers** - If we add training/fine-tuning (M8+)

### ❌ Low Priority / Not Applicable
5. **candle-onnx** - Model format support (not current focus)
6. **candle-birnn** - Bidirectional RNN (not used in transformer models)
7. **candle_embed** - Unknown/unavailable
8. **mpnet-rs** - Specific embedding model (too narrow)
9. **del-candle** - Unknown domain
10. **dfdx** - Alternative framework (not compatible)
11. **gemm** - Low-level BLAS (Candle already handles)
12. **rlkit** - RL training (M8+ if at all)

---

## Detailed Evaluations

### 1. ✅ candle-layer-norm (HIGH PRIORITY)

**Source**: <https://github.com/huggingface/candle-extensions/candle-layer-norm/>  
**Version**: 0.0.1 (April 2025)  
**Status**: Official Candle extension

**What it provides**:
- Fused dropout + residual + LayerNorm GPU kernel
- Adapted from FlashAttention's LayerNorm implementation
- Supports RMSNorm (used in LLaMA models)
- Pre-norm and post-norm architectures
- Dimensions divisible by 8, up to 8192

**Relevance to Lightbulb**: **VERY HIGH**
- We use RMSNorm extensively in BatchedTransformer
- Current implementation: separate operations (norm → scale → add)
- Fused version: single GPU kernel with better memory efficiency
- Expected speedup: 20-30% on normalization operations

**Integration effort**: **MEDIUM**
- Replace current RMSNorm with fused version
- Feature-gate behind `cuda` feature (like FlashAttention)
- Fallback to manual implementation on CPU
- Minimal API changes (drop-in replacement for candle_nn::RMSNorm)

**Recommendation**: **Integrate in M3.6 or M4**
- Complements FlashAttention (M3.4 just completed)
- Natural fit when working on GPU optimizations
- Official Candle extension (maintained by HuggingFace)

**Action**: Add to ROADMAP under M3.6 or M4 as "Fused LayerNorm/RMSNorm"

---

### 2. ✅ candle-ext (HIGH PRIORITY)

**Source**: <https://github.com/mokeyish/candle-ext>  
**Version**: 0.1.7 (December 2023)  
**Maintainer**: mokeyish (community)

**What it provides** (PyTorch functions not in Candle):
- `scaled_dot_product_attention` - Standard scaled dot-product attention
- `chunk` / `unbind` - Tensor splitting operations
- `cumsum` - Cumulative sum along dimension
- `equal` - Element-wise equality
- `eye` / `full` / `full_like` - Tensor creation utilities
- `triu` / `tril` - Upper/lower triangular matrices
- `masked_fill` - Fill tensor where mask is true
- `logical_not` / `logical_and` / `logical_or` - Boolean operations
- `outer` - Outer product

**Relevance to Lightbulb**: **MEDIUM-HIGH**

**Currently useful**:
- `triu`/`tril`: Could simplify causal mask creation
- `masked_fill`: Useful for attention masking
- `cumsum`: Potential use in prefix sum calculations

**Not currently needed** (we have alternatives):
- `scaled_dot_product_attention`: We have custom batched attention
- `chunk`/`unbind`: Can use Candle's `narrow` + iteration
- Boolean ops: Can implement inline with arithmetic

**Integration effort**: **LOW**
- Add as optional dependency
- Use selectively where it simplifies code
- No major refactoring needed

**Concerns**:
- Community-maintained (not official Candle)
- Last update December 2023 (may be stale)
- Docs.rs build failing (red flag)

**Recommendation**: **CAUTIOUS INTEGRATION**
- Evaluate specific functions we need
- Consider vendoring/copying implementations instead of dependency
- Test thoroughly before production use

**Action**: Consider for utility functions, but don't rely heavily

---

### 3. 🟡 candle-einops (MEDIUM PRIORITY)

**Source**: <https://docs.rs/candle-einops>  
**Version**: 0.1.2  
**Purpose**: Python einops-style tensor reshaping

**What it provides**:
- Macro for tensor transformations: `einops!("b h w c -> b c h w", tensor)`
- Expressive syntax for common reshape/permute/reduce operations
- Rust implementation of popular Python library

**Example**:
```rust
// Instead of:
let x = tensor.transpose(1, 2)?.reshape(&[batch, channels, height * width])?;

// Can write:
let x = einops!("b h w c -> b c (h w)", tensor)?;
```

**Relevance to Lightbulb**: **LOW-MEDIUM**

**Pros**:
- More expressive than nested transpose/reshape
- Reduces cognitive load for complex transformations
- Self-documenting dimension reordering

**Cons**:
- Not critical functionality (syntactic sugar)
- Adds macro dependency and complexity
- We don't have many complex tensor transformations currently

**Recommendation**: **DEFER UNTIL M6+**
- Nice-to-have, not essential
- Consider if tensor manipulation code becomes hard to read
- Re-evaluate when working on advanced architectures (M6+)

---

### 4. 🟡 candle-optimisers (MEDIUM PRIORITY - FUTURE)

**Source**: <https://docs.rs/candle-optimisers>  
**Version**: 0.9.0  
**Purpose**: Training optimizers (Adam, AdamW, SGD, LBFGS, etc.)

**What it provides**:
- Adam/AdamW - Adaptive moment estimation
- Adadelta, Adagrad, Adamax, NAdam, RAdam
- RMSprop, SGD with momentum
- LBFGS - Limited memory quasi-Newton method
- Weight decay strategies
- Learning rate scheduling

**Relevance to Lightbulb**: **LOW (NOW), MEDIUM (M8+)**

**Current**: We're focused on **inference**, not training
- M0-M6 are all inference-focused
- No training workloads in current roadmap

**Future (M8+)**: Modular training infrastructure
- If we implement LoRA fine-tuning
- If we add continuous learning
- If we support model distillation

**Recommendation**: **DEFER TO M8+**
- Not applicable to inference engine (M0-M6)
- Revisit when/if we add training capabilities
- Well-maintained and comprehensive when needed

---

### 5. ❌ candle-onnx (LOW PRIORITY)

**Source**: <https://docs.rs/candle-onnx>  
**Version**: 0.9.1  
**Purpose**: ONNX model format support for Candle

**What it provides**:
- Load ONNX models into Candle
- Convert ONNX operations to Candle operations
- Bridge between ONNX and Candle ecosystems

**Relevance to Lightbulb**: **LOW**

**Why not critical**:
- We focus on native Candle models (LLaMA, Phi, etc.)
- GGUF is our primary quantized format
- HuggingFace safetensors is secondary format
- ONNX support not requested by users

**Possible future use**:
- If users want to deploy ONNX-exported models
- If we want broader model format support
- Cross-framework compatibility

**Recommendation**: **DEFER INDEFINITELY**
- Not on current roadmap
- Focus on GGUF and safetensors first
- Add only if users request ONNX support

---

### 6. ❌ candle-birnn (LOW PRIORITY)

**Source**: <https://docs.rs/candle-birnn>  
**Purpose**: Bidirectional RNN layers

**Relevance to Lightbulb**: **NONE**
- We don't use RNN architectures
- Transformer-based models only (LLaMA, Phi, etc.)
- Bidirectional processing not applicable to autoregressive generation

**Recommendation**: **NOT APPLICABLE**

---

### 7. ❌ candle_embed (UNKNOWN)

**Source**: <https://docs.rs/candle_embed>  
**Status**: Docs.rs page not found / crate may not exist

**Recommendation**: **SKIP** - Cannot evaluate

---

### 8. ❌ mpnet-rs (LOW PRIORITY)

**Source**: <https://docs.rs/mpnet-rs>  
**Purpose**: MPNet sentence embedding model implementation

**Relevance to Lightbulb**: **LOW**
- Specific model implementation, not general infrastructure
- We support arbitrary transformer models, not specific embeddings
- Too narrow for inclusion in core Lightbulb

**Recommendation**: **NOT APPLICABLE**
- Users can implement MPNet on top of Lightbulb if needed
- Not part of core inference engine

---

### 9. ❌ del-candle (UNKNOWN)

**Source**: <https://docs.rs/del-candle/0.1.0/del_candle/>  
**Purpose**: Unclear (docs minimal)

**Recommendation**: **SKIP** - Insufficient information

---

### 10. ❌ dfdx (NOT COMPATIBLE)

**Source**: <https://docs.rs/dfdx>  
**Purpose**: **Alternative** ML framework (not Candle extension)

**What it is**:
- Standalone deep learning framework like Candle
- Different design philosophy and APIs
- Not compatible with Candle ecosystem

**Relevance to Lightbulb**: **NONE**
- We're committed to Candle ecosystem
- dfdx is a competitor, not a complement
- Would require complete rewrite to use

**Recommendation**: **NOT APPLICABLE**

---

### 11. ❌ gemm (LOW PRIORITY)

**Source**: <https://docs.rs/gemm>  
**Version**: 0.18.2  
**Purpose**: Low-level GEMM (matrix multiply) implementations

**What it provides**:
- Hand-optimized matrix multiplication
- Parallelism controls
- WASM SIMD support
- Packing thresholds tuning

**Relevance to Lightbulb**: **LOW**

**Why not needed**:
- Candle already provides optimized matmul via:
  - MKL/OpenBLAS on CPU
  - cuBLAS on CUDA
  - Metal Performance Shaders on Metal
- Adding low-level GEMM would bypass Candle's backend abstraction
- Unlikely to beat vendor-optimized BLAS

**Possible niche use**:
- Custom kernel experiments (academic interest)
- Embedded/constrained environments where vendor BLAS unavailable

**Recommendation**: **NOT APPLICABLE**
- Candle's matmul is already well-optimized
- Focus on higher-level optimizations (FlashAttention, KV caching, etc.)

---

### 12. ❌ rlkit (VERY LOW PRIORITY)

**Source**: <https://github.com/Hifive55555/rlkit>  
**Purpose**: Reinforcement learning training toolkit

**Relevance to Lightbulb**: **VERY LOW**

**Why not applicable**:
- We're an inference engine, not an RL training framework
- RL training is extremely specialized domain
- Would require massive scope expansion

**Possible future** (M8+, speculative):
- If we implement reward modeling
- If we add online RL fine-tuning
- If we build agent learning systems

**Recommendation**: **NOT APPLICABLE** (far future, if ever)

---

## Prioritized Integration Plan

### Phase 1: M3.6 or M4 (GPU Optimizations)
**Priority: HIGH**

1. **candle-layer-norm**
   - Fused LayerNorm/RMSNorm GPU kernels
   - Complements FlashAttention work
   - Expected 20-30% speedup on normalization
   - **Action**: Add to M3.6 or M4 roadmap

### Phase 2: M4-M5 (Code Quality)
**Priority: MEDIUM**

2. **candle-ext (selective)**
   - Evaluate specific functions: `triu`, `tril`, `masked_fill`
   - Consider vendoring implementations instead of full dependency
   - Use only where it significantly simplifies code
   - **Action**: Trial integration, measure value

### Phase 3: M6+ (Advanced Features)
**Priority: LOW**

3. **candle-einops**
   - Re-evaluate if tensor manipulation becomes complex
   - Consider for modular neural architectures (M6)
   - **Action**: Defer until need arises

### Phase 4: M8+ (Training)
**Priority: DEFERRED**

4. **candle-optimisers**
   - Only if we add training/fine-tuning capabilities
   - Well-maintained, ready when needed
   - **Action**: Revisit in M8 planning

### Not Recommended
- candle-onnx, candle-birnn, candle_embed, mpnet-rs, del-candle, dfdx, gemm, rlkit

---

## Implementation Notes

### Adding candle-layer-norm (Recommended First Step)

**In Cargo.toml**:
```toml
[dependencies]
candle-layer-norm = { version = "0.0.1", optional = true }

[features]
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-layer-norm"]
```

**Usage pattern** (similar to FlashAttention):
```rust
#[cfg(feature = "cuda")]
use candle_layer_norm::RMSNorm as FusedRMSNorm;

// Auto-select fused version on CUDA
let rmsnorm = if device.is_cuda() {
    #[cfg(feature = "cuda")]
    return FusedRMSNorm::new(hidden_size, eps)?;
    
    #[cfg(not(feature = "cuda"))]
    candle_nn::RMSNorm::new(hidden_size, eps)?
} else {
    candle_nn::RMSNorm::new(hidden_size, eps)?
};
```

**Testing approach**:
1. Numerical parity tests (like FlashAttention tests)
2. Benchmark prefill/decode performance
3. Validate across different sequence lengths
4. Ensure graceful fallback on CPU

---

## Monitoring & Maintenance

**Stay informed about**:
- Candle core releases (we're tracking upstream)
- New official extensions from HuggingFace
- Community ecosystem development

**Re-evaluate quarterly**:
- Are there new extensions we should adopt?
- Are current dependencies still maintained?
- Have our needs changed (e.g., adding training)?

---

## Conclusion

**Immediate recommendation**: Integrate **candle-layer-norm** in M3.6 or M4
- Highest value-to-effort ratio
- Natural complement to FlashAttention (M3.4)
- Official HuggingFace extension (well-maintained)
- Clear performance benefits for GPU workloads

**Secondary consideration**: Selectively use **candle-ext** utilities
- Trial integration in M4-M5
- Vendor specific functions rather than full dependency
- Measure actual code quality improvement

**All others**: Defer or skip
- Not applicable to current inference focus
- Consider in future milestones if needs change

This evaluation supports Lightbulb's principle: "Build on Candle; don't reimplement unless strictly necessary." We're selective about dependencies, focusing on high-value integrations that enhance our core mission.
