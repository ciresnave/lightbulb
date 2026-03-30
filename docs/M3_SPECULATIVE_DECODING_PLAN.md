# M3: Speculative Decoding Implementation Plan

**Status**: PLANNED (Architecture exists, needs production hardening)  
**Priority**: HIGH (2-4× throughput improvement for long generations)  
**Timeline**: 2-3 weeks  
**Dependencies**: M2 complete, KV cache implementation stable

---

## Overview

Complete and productionize speculative decoding for Lightbulb. A prototype implementation exists (`src/engine/speculative.rs`), but needs:
- Production-grade model loading and management
- Integration with OpenAI API endpoints
- Memory-efficient dual-model support
- Quantization compatibility (AWQ draft + FP16 target)
- Advanced sampling support (temperature, top-p, penalties)

**Performance targets:**
- Throughput: 1.5-2.5× improvement for long sequences (>128 tokens)
- Memory: <15% overhead for draft model (1B draft + 7B target)
- Latency: Minimal impact on first-token latency
- Acceptance rate: >40% for aligned draft/target pairs

---

## Speculative Decoding Primer

### Algorithm

1. **Draft Phase**: Small model generates K tokens quickly (K=4-7 typical)
2. **Verification Phase**: Large model evaluates all K tokens in parallel
3. **Acceptance**: Accept longest prefix where draft and target agree
4. **Correction**: On first mismatch, use target's corrected token

### Key Insight

Target model processes K tokens in parallel (single forward pass) rather than K sequential forward passes. Draft model overhead is negligible if acceptance rate is good.

### Example Timeline

```
Standard decoding (7B model):
Token 1: ████████ (100ms target)
Token 2: ████████ (100ms target)
Token 3: ████████ (100ms target)
Token 4: ████████ (100ms target)
Total: 400ms

Speculative decoding (1B draft + 7B target, 75% acceptance):
Draft K=4: ██ (20ms draft × 4 = 80ms)
Verify:    ████████ (100ms target, verifies all 4)
Accept:    3/4 tokens → Reject 4th → Get corrected token
Total: 180ms for 4 tokens (2.2× speedup)
```

---

## Current Implementation Status

### ✅ Complete

Located in `src/engine/speculative.rs`:

1. **`SpeculativeConfig`** (lines 30-46)
   - Configurable speculation depth (num_speculative_tokens)
   - Auto-fallback on poor acceptance rate
   - Min acceptance threshold

2. **`SpeculativeStats`** (lines 61-111)
   - Acceptance rate tracking
   - Speedup calculation
   - Timing measurements (draft vs target)

3. **`SpeculativeModel` trait** (lines 121-142)
   - Abstract interface for models
   - Methods: `forward_logits()`, `device()`, `vocab_size()`, `reset_cache()`

4. **`SpeculativeDecoder`** (lines 144-315)
   - Main orchestration logic
   - Verification algorithm
   - Fallback to standard decoding
   - Statistics tracking

5. **`BatchedTransformerAdapter`** (lines 1-127 in `src/model/speculative_adapters.rs`)
   - Wraps `BatchedTransformer` to implement `SpeculativeModel`
   - Manages per-model KV cache
   - Converts between interfaces

6. **Demo Example** (`examples/speculative_demo.rs`)
   - Mock models showing verification mechanics
   - 4 test scenarios (perfect/partial/zero acceptance, auto-fallback)

### 🔄 Partially Complete

1. **Model Loading** (`src/init.rs` lines 141-179)
   - Has `configure_speculative_decoding()` helper
   - Estimates memory requirements
   - **Missing**: Actual draft model loading, path specification

2. **Integration Points**
   - Config exists but not wired to API endpoints
   - No CLI flags for enabling/configuring
   - Statistics not exposed via metrics

### ❌ Missing (Critical Gaps)

1. **Production Model Management**
   - No draft model loader
   - No dual-model lifecycle (load, unload, swap)
   - No fallback when draft model unavailable

2. **Memory Management**
   - No memory allocation strategy for two models
   - No sharing of embedding layers (common optimization)
   - No dynamic loading/unloading based on memory pressure

3. **API Integration**
   - OpenAI `/v1/chat/completions` doesn't support draft model param
   - No way to specify draft model path via API
   - No stats reporting in response metadata

4. **Advanced Sampling**
   - Existing code uses simple sampler callback
   - Doesn't support temperature, top-p, frequency penalty
   - No beam search compatibility

5. **Quantization Integration**
   - No support for quantized draft models (AWQ)
   - Mixed precision not tested (quantized draft + FP16 target)
   - No memory estimates for quantized draft

6. **Batching**
   - Single-sequence only
   - No batch-level speculation
   - No per-sequence speculation control

---

## Integration Points with AWQ

### Critical Overlaps

| Aspect             | AWQ Impact                          | Spec Decoding Impact                   | Design Decision                                        |
| ------------------ | ----------------------------------- | -------------------------------------- | ------------------------------------------------------ |
| **Model Loading**  | Quantized weights format            | Need TWO models loaded                 | Loader must support mixed quantization                 |
| **Linear Layers**  | `QuantizedLinear` replaces `Linear` | Both draft/target use same layer types | `SpeculativeModel` trait must be quantization-agnostic |
| **Memory Budget**  | 2× reduction (14GB → 7GB)           | 2 models loaded (+10-15%)              | Quantized draft enables larger targets                 |
| **KV Cache**       | No direct impact                    | Separate caches for draft/target       | Cache allocator must handle multi-model                |
| **Inference Path** | CustomOp for quantized matmul       | Same forward pass API                  | `forward_logits()` works with both                     |

### Example Configuration

**Optimal setup for 24GB GPU:**
```rust
TargetModel {
    path: "/models/llama-7b-fp16",     // 14GB FP16
    dtype: DType::F16,
    quantization: None,
}

DraftModel {
    path: "/models/llama-1b-awq",       // 0.6GB AWQ (vs 2GB FP16)
    dtype: DType::F16,
    quantization: Some(QuantConfig {
        method: "awq",
        bits: 4,
        group_size: 128,
    }),
}

Total: 14.6GB model weights + 4GB KV cache = 18.6GB (fits on 24GB GPU)
```

**Memory savings from quantized draft:**
- FP16 draft (1B): ~2GB
- AWQ draft (1B): ~0.6GB
- **Savings: 1.4GB** → Can increase KV cache size or target model size

### Shared Design Requirements

1. **`SpeculativeModel` trait must work with both `Linear` and `QuantizedLinear`**
   - ✅ Already abstracted via `forward_logits()`
   - ✅ No direct layer access in trait

2. **Model loader must support mixed quantization**
   ```rust
   pub fn load_model_pair(
       target_path: &str,
       target_quant: Option<QuantConfig>,
       draft_path: &str,
       draft_quant: Option<QuantConfig>,
       device: &Device,
   ) -> Result<(Box<dyn SpeculativeModel>, Box<dyn SpeculativeModel>)>;
   ```

3. **Memory allocator must track multi-model usage**
   ```rust
   struct MultiModelMemory {
       target_weights: usize,
       draft_weights: usize,
       target_cache: usize,
       draft_cache: usize,
       activation_overhead: usize,
   }
   ```

4. **KV cache must isolate draft/target caches**
   - Already handled by `BatchedTransformerAdapter` (separate `caches` vec per adapter)
   - Need to ensure cache builder supports multiple allocations

---

## Implementation Phases

### Phase 1: Production Model Management (Week 1)

**Goal**: Load and manage draft + target model pairs reliably

#### Files to Create

1. **`src/loaders/speculative.rs`** (New)
   ```rust
   //! Dual-model loading for speculative decoding
   
   use crate::engine::speculative::SpeculativeModel;
   use crate::model::speculative_adapters::BatchedTransformerAdapter;
   use crate::loaders::{load_local_llama, QuantConfig};
   use anyhow::{Context, Result};
   use candle_core::Device;
   
   /// Configuration for a model in a speculative pair
   #[derive(Debug, Clone)]
   pub struct ModelSpec {
       /// Path to model directory
       pub path: String,
       /// Quantization config (None = FP16/BF16)
       pub quant_config: Option<QuantConfig>,
       /// Data type (F16, BF16, F32)
       pub dtype: candle_core::DType,
   }
   
   /// Speculative model pair (draft + target)
   pub struct SpeculativeModelPair {
       pub target: Box<dyn SpeculativeModel>,
       pub draft: Box<dyn SpeculativeModel>,
       pub target_spec: ModelSpec,
       pub draft_spec: ModelSpec,
   }
   
   impl SpeculativeModelPair {
       /// Load both models from specifications
       pub fn load(
           target_spec: ModelSpec,
           draft_spec: ModelSpec,
           device: Device,
           max_seq_len: usize,
       ) -> Result<Self> {
           // Load target model
           let target_model = load_local_llama(
               &target_spec.path,
               target_spec.dtype,
               device.clone(),
               target_spec.quant_config.as_ref(),
           ).context("Loading target model")?;
           
           let target = Box::new(BatchedTransformerAdapter::new(
               target_model,
               max_seq_len,
           )?) as Box<dyn SpeculativeModel>;
           
           // Load draft model
           let draft_model = load_local_llama(
               &draft_spec.path,
               draft_spec.dtype,
               device.clone(),
               draft_spec.quant_config.as_ref(),
           ).context("Loading draft model")?;
           
           let draft = Box::new(BatchedTransformerAdapter::new(
               draft_model,
               max_seq_len,
           )?) as Box<dyn SpeculativeModel>;
           
           Ok(Self { target, draft, target_spec, draft_spec })
       }
       
       /// Estimate total memory usage
       pub fn estimate_memory(&self) -> MemoryEstimate {
           // Calculate based on model specs and quantization
           todo!()
       }
       
       /// Check if both models fit in available memory
       pub fn fits_in_memory(&self, available_bytes: usize) -> bool {
           self.estimate_memory().total_bytes() < available_bytes
       }
   }
   
   /// Memory usage breakdown for speculative pair
   #[derive(Debug, Clone)]
   pub struct MemoryEstimate {
       pub target_weights_bytes: usize,
       pub draft_weights_bytes: usize,
       pub target_cache_bytes: usize,
       pub draft_cache_bytes: usize,
       pub activation_overhead_bytes: usize,
   }
   
   impl MemoryEstimate {
       pub fn total_bytes(&self) -> usize {
           self.target_weights_bytes
               + self.draft_weights_bytes
               + self.target_cache_bytes
               + self.draft_cache_bytes
               + self.activation_overhead_bytes
       }
   }
   ```

2. **Update `src/loaders/mod.rs`**
   ```rust
   pub mod speculative;
   pub use speculative::{SpeculativeModelPair, ModelSpec, MemoryEstimate};
   ```

3. **`src/config.rs`** - Add speculative config section
   ```rust
   /// Speculative decoding configuration
   #[derive(Debug, Clone, Deserialize)]
   pub struct SpeculativeDecodingConfig {
       /// Enable speculative decoding
       pub enabled: bool,
       
       /// Path to draft model directory
       pub draft_model_path: Option<String>,
       
       /// Quantization for draft model (None = FP16)
       pub draft_quantization: Option<QuantConfig>,
       
       /// Number of speculative tokens to generate
       pub num_speculative_tokens: usize,
       
       /// Minimum acceptance rate before fallback
       pub min_acceptance_rate: f64,
       
       /// Auto-fallback to standard decoding
       pub auto_fallback: bool,
   }
   
   impl Default for SpeculativeDecodingConfig {
       fn default() -> Self {
           Self {
               enabled: false,  // Opt-in feature
               draft_model_path: None,
               draft_quantization: None,
               num_speculative_tokens: 5,
               min_acceptance_rate: 0.3,
               auto_fallback: true,
           }
       }
   }
   ```

**Acceptance Criteria:**
- ✅ Load target + draft model pair successfully
- ✅ Support mixed quantization (FP16 target + AWQ draft)
- ✅ Memory estimation accurate within 10%
- ✅ Fail gracefully if insufficient memory
- ✅ Tests: Load 7B target + 1B draft on mock hardware

---

### Phase 2: Advanced Sampling Integration (Week 1-2)

**Goal**: Support temperature, top-p, frequency penalty in speculative decoding

#### Files to Modify

1. **`src/engine/speculative.rs`** - Update sampler interface
   ```rust
   /// Advanced sampler that can use generation parameters
   pub trait AdvancedSampler: Send + Sync {
       /// Sample next token from logits with parameters
       fn sample(
           &mut self,
           logits: &Tensor,
           generated_tokens: &[u32],
       ) -> Result<u32>;
       
       /// Clone the sampler for parallel use
       fn clone_box(&self) -> Box<dyn AdvancedSampler>;
   }
   
   /// Update SpeculativeDecoder::generate_tokens signature
   pub fn generate_tokens(
       &mut self,
       draft_model: &mut dyn SpeculativeModel,
       target_model: &mut dyn SpeculativeModel,
       context_tokens: &[u32],
       sampler: &mut dyn AdvancedSampler,  // Changed from FnMut
   ) -> Result<Vec<u32>>
   ```

2. **`src/sampling/mod.rs`** - Implement sampler types
   ```rust
   pub struct GreedySampler;
   
   impl AdvancedSampler for GreedySampler {
       fn sample(&mut self, logits: &Tensor, _tokens: &[u32]) -> Result<u32> {
           logits.argmax(D::Minus1)?.to_scalar::<u32>()
       }
       
       fn clone_box(&self) -> Box<dyn AdvancedSampler> {
           Box::new(GreedySampler)
       }
   }
   
   pub struct TopPSampler {
       temperature: f32,
       top_p: f32,
       rng: StdRng,
   }
   
   impl AdvancedSampler for TopPSampler {
       fn sample(&mut self, logits: &Tensor, _tokens: &[u32]) -> Result<u32> {
           // Apply temperature
           let logits = (logits / self.temperature as f64)?;
           
           // Softmax
           let probs = candle_nn::ops::softmax_last_dim(&logits)?;
           
           // Top-p filtering
           let probs_vec: Vec<f32> = probs.to_vec1()?;
           let filtered = top_p_filter(&probs_vec, self.top_p);
           
           // Sample from filtered distribution
           sample_from_distribution(&filtered, &mut self.rng)
       }
       
       fn clone_box(&self) -> Box<dyn AdvancedSampler> {
           Box::new(TopPSampler {
               temperature: self.temperature,
               top_p: self.top_p,
               rng: StdRng::from_entropy(),
           })
       }
   }
   
   pub struct PenaltySampler {
       base_sampler: Box<dyn AdvancedSampler>,
       frequency_penalty: f32,
       presence_penalty: f32,
   }
   
   impl AdvancedSampler for PenaltySampler {
       fn sample(&mut self, logits: &Tensor, tokens: &[u32]) -> Result<u32> {
           // Apply penalties based on generated_tokens
           let penalized = apply_penalties(
               logits,
               tokens,
               self.frequency_penalty,
               self.presence_penalty,
           )?;
           
           // Delegate to base sampler
           self.base_sampler.sample(&penalized, tokens)
       }
       
       fn clone_box(&self) -> Box<dyn AdvancedSampler> {
           Box::new(PenaltySampler {
               base_sampler: self.base_sampler.clone_box(),
               frequency_penalty: self.frequency_penalty,
               presence_penalty: self.presence_penalty,
           })
       }
   }
   ```

3. **`src/engine/speculative.rs`** - Update generate_tokens implementation
   ```rust
   pub fn generate_tokens(
       &mut self,
       draft_model: &mut dyn SpeculativeModel,
       target_model: &mut dyn SpeculativeModel,
       context_tokens: &[u32],
       sampler: &mut dyn AdvancedSampler,
   ) -> Result<Vec<u32>> {
       // ... existing setup ...
       
       // Phase 1: Draft generation (pass accumulated tokens for penalties)
       let mut draft_tokens = Vec::new();
       let mut current_context = context_tokens.to_vec();
       
       for _ in 0..self.config.num_speculative_tokens {
           let logits = draft_model.forward_logits(&current_context, current_context.len())?;
           let token = sampler.sample(&logits, &current_context)?;  // Pass full context
           draft_tokens.push(token);
           current_context.push(token);
       }
       
       // Phase 2: Target verification (same pattern)
       // ... rest unchanged ...
   }
   ```

**Acceptance Criteria:**
- ✅ Greedy, temperature, top-p, top-k sampling all work
- ✅ Frequency and presence penalties applied correctly
- ✅ Sampling behavior identical to standard decoding
- ✅ Tests: Verify sampling distributions match non-speculative path

---

### Phase 3: API Integration (Week 2)

**Goal**: Expose speculative decoding via OpenAI-compatible API

#### Files to Modify

1. **`src/openai/chat.rs`** - Add draft model parameter
   ```rust
   /// Extended chat completion request with speculative decoding support
   #[derive(Debug, Deserialize)]
   pub struct ChatCompletionRequest {
       // ... existing fields ...
       
       /// Enable speculative decoding for this request
       #[serde(default)]
       pub speculative_decoding: bool,
       
       /// Override global draft model (optional)
       pub draft_model: Option<String>,
       
       /// Number of speculative tokens (override config)
       pub num_speculative_tokens: Option<usize>,
   }
   
   /// Extended response metadata
   #[derive(Debug, Serialize)]
   pub struct ChatCompletionUsage {
       // ... existing fields ...
       
       /// Speculative decoding statistics
       #[serde(skip_serializing_if = "Option::is_none")]
       pub speculative_stats: Option<SpeculativeStats>,
   }
   ```

2. **`src/server/handler.rs`** - Wire up speculative path
   ```rust
   async fn handle_chat_completion(
       req: ChatCompletionRequest,
       state: Arc<ServerState>,
   ) -> Result<ChatCompletionResponse> {
       // Check if speculative decoding requested and available
       let use_speculative = req.speculative_decoding
           && state.config.speculative_decoding.enabled
           && state.draft_model.is_some();
       
       if use_speculative {
           generate_with_speculation(req, state).await
       } else {
           generate_standard(req, state).await
       }
   }
   
   async fn generate_with_speculation(
       req: ChatCompletionRequest,
       state: Arc<ServerState>,
   ) -> Result<ChatCompletionResponse> {
       let draft_model = state.draft_model.as_ref().unwrap();
       let target_model = &state.target_model;
       
       // Create sampler from request parameters
       let sampler = create_sampler(
           req.temperature,
           req.top_p,
           req.frequency_penalty,
           req.presence_penalty,
       );
       
       // Use SpeculativeDecoder
       let mut decoder = SpeculativeDecoder::new(/* ... */);
       let tokens = decoder.generate_tokens(
           draft_model.as_mut(),
           target_model.as_mut(),
           &context_tokens,
           sampler.as_mut(),
       )?;
       
       // Return response with stats
       Ok(ChatCompletionResponse {
           // ... standard fields ...
           usage: Some(ChatCompletionUsage {
               // ... token counts ...
               speculative_stats: Some(decoder.stats().clone()),
           }),
       })
   }
   ```

3. **`src/server/state.rs`** - Add draft model to server state
   ```rust
   pub struct ServerState {
       pub target_model: Arc<Mutex<Box<dyn SpeculativeModel>>>,
       pub draft_model: Option<Arc<Mutex<Box<dyn SpeculativeModel>>>>,
       pub config: Config,
       // ... other fields ...
   }
   
   impl ServerState {
       pub fn new(config: Config) -> Result<Self> {
           // Load target model
           let target_model = load_target_model(&config)?;
           
           // Load draft model if speculative decoding enabled
           let draft_model = if config.speculative_decoding.enabled {
               if let Some(ref draft_path) = config.speculative_decoding.draft_model_path {
                   Some(Arc::new(Mutex::new(load_draft_model(draft_path, &config)?)))
               } else {
                   tracing::warn!("Speculative decoding enabled but no draft_model_path provided");
                   None
               }
           } else {
               None
           };
           
           Ok(Self { target_model, draft_model, config })
       }
   }
   ```

**Acceptance Criteria:**
- ✅ API accepts `speculative_decoding: true` in requests
- ✅ Stats returned in `usage.speculative_stats`
- ✅ Graceful fallback if draft model not loaded
- ✅ Tests: E2E API test with speculative decoding enabled

---

### Phase 4: CLI and Configuration (Week 2)

**Goal**: Add CLI flags and config file support for speculative decoding

#### Files to Modify

1. **`src/cli.rs`** - Add CLI arguments
   ```rust
   #[derive(Parser)]
   pub struct Args {
       // ... existing args ...
       
       /// Enable speculative decoding
       #[arg(long, default_value = "false")]
       pub speculative_decoding: bool,
       
       /// Path to draft model for speculative decoding
       #[arg(long)]
       pub draft_model: Option<PathBuf>,
       
       /// Number of speculative tokens per iteration
       #[arg(long, default_value = "5")]
       pub num_speculative_tokens: usize,
       
       /// Quantization for draft model (none, awq, gptq)
       #[arg(long)]
       pub draft_quantization: Option<String>,
   }
   ```

2. **`config.toml`** - Example configuration
   ```toml
   # Speculative Decoding
   [speculative_decoding]
   enabled = true
   draft_model_path = "/models/llama-1b-awq"
   num_speculative_tokens = 5
   min_acceptance_rate = 0.3
   auto_fallback = true
   
   [speculative_decoding.draft_quantization]
   method = "awq"
   bits = 4
   group_size = 128
   ```

3. **`README.md`** - Usage documentation
   ````markdown
   ## Speculative Decoding
   
   Accelerate generation with a smaller draft model:
   
   ```bash
   lightbulb serve \
     --model /models/llama-7b-fp16 \
     --speculative-decoding \
     --draft-model /models/llama-1b-awq \
     --num-speculative-tokens 5
   ```
   
   **Benefits:**
   - 1.5-2.5× throughput improvement
   - Best for long generations (>128 tokens)
   - Automatic fallback if poor acceptance rate
   
   **Recommendations:**
   - Draft model: 5-10× smaller than target
   - Quantize draft model (AWQ) to save memory
   - Use same architecture family (Llama-1B + Llama-7B)
   ````

**Acceptance Criteria:**
- ✅ CLI flags work and override config file
- ✅ Config validation (draft model exists, compatible architecture)
- ✅ Help text explains parameters clearly
- ✅ Example config in `examples/config/speculative.toml`

---

### Phase 5: Memory Management & Optimization (Week 3)

**Goal**: Efficient memory usage and shared resources

#### Optimizations to Implement

1. **Shared Embedding Layer** (Optional, high impact)
   ```rust
   /// Share embedding weights between draft and target
   /// Saves memory if architectures are compatible
   pub struct SharedEmbeddingModels {
       shared_embeddings: Arc<Embedding>,
       draft_layers: Vec<TransformerBlock>,
       target_layers: Vec<TransformerBlock>,
   }
   ```

2. **Dynamic Draft Model Loading**
   ```rust
   /// Load draft model on-demand when memory permits
   pub struct LazyDraftModel {
       spec: ModelSpec,
       model: Option<Box<dyn SpeculativeModel>>,
       memory_available: Arc<AtomicUsize>,
   }
   
   impl LazyDraftModel {
       /// Load draft model if memory available
       pub fn try_load(&mut self) -> Result<bool> {
           let required = self.estimate_memory();
           let available = self.memory_available.load(Ordering::Relaxed);
           
           if available >= required {
               self.model = Some(load_draft_model(&self.spec)?);
               Ok(true)
           } else {
               Ok(false) // Not enough memory
           }
       }
       
       /// Unload draft model to free memory
       pub fn unload(&mut self) {
           self.model = None;
       }
   }
   ```

3. **Memory Pooling for KV Caches**
   ```rust
   /// Reuse cache allocations between requests
   pub struct CachePool {
       draft_caches: Vec<ParallelKvCache>,
       target_caches: Vec<ParallelKvCache>,
       available_draft: Vec<usize>,
       available_target: Vec<usize>,
   }
   
   impl CachePool {
       /// Get cache from pool or allocate new
       pub fn acquire_draft(&mut self) -> Result<ParallelKvCache> {
           if let Some(idx) = self.available_draft.pop() {
               Ok(self.draft_caches[idx].clone())
           } else {
               // Allocate new cache
               self.allocate_draft_cache()
           }
       }
       
       /// Return cache to pool
       pub fn release_draft(&mut self, cache_idx: usize) {
           self.available_draft.push(cache_idx);
       }
   }
   ```

**Acceptance Criteria:**
- ✅ Memory usage stable under load (no leaks)
- ✅ Draft model can be unloaded/reloaded dynamically
- ✅ Shared embeddings work (if architectures compatible)
- ✅ Benchmarks: Memory overhead <15% for 1B draft + 7B target

---

### Phase 6: Testing & Validation (Week 3)

**Goal**: Comprehensive correctness and performance testing

#### Test Files to Create

1. **`tests/speculative_correctness.rs`**
   ```rust
   #[test]
   fn test_speculative_vs_standard_output_identical() {
       // Same seed, same parameters → same output
       let prompt = "The quick brown fox";
       
       // Standard decoding
       let standard_output = generate_standard(&prompt, seed=42)?;
       
       // Speculative decoding
       let speculative_output = generate_speculative(&prompt, seed=42)?;
       
       assert_eq!(standard_output, speculative_output);
   }
   
   #[test]
   fn test_acceptance_rate_tracking() {
       // Verify stats are accurate
       let mut decoder = SpeculativeDecoder::new(config);
       
       // Run 100 rounds, track manually
       let mut manual_accepted = 0;
       let mut manual_total = 0;
       
       for _ in 0..100 {
           let tokens = decoder.generate_tokens(/* ... */)?;
           // Count accepted vs rejected manually
           // ...
       }
       
       let reported_rate = decoder.stats().acceptance_rate();
       let actual_rate = manual_accepted as f64 / manual_total as f64;
       
       assert!((reported_rate - actual_rate).abs() < 0.01);
   }
   
   #[test]
   fn test_fallback_activation() {
       // Ensure auto-fallback works
       let config = SpeculativeConfig {
           auto_fallback: true,
           min_acceptance_rate: 0.3,
           ..Default::default()
       };
       let mut decoder = SpeculativeDecoder::new(config);
       
       // Simulate 20 rounds with 0% acceptance
       for _ in 0..20 {
           // Mock models that always disagree
           let _ = decoder.generate_tokens(/* ... */);
       }
       
       assert!(!decoder.should_speculate()); // Fallback should activate
   }
   ```

2. **`tests/speculative_sampling.rs`**
   ```rust
   #[test]
   fn test_temperature_sampling_identical() {
       // Temperature=0.7 should produce same distribution
       let sampler = TopPSampler::new(0.7, 1.0, seed=42);
       
       // Sample 1000 tokens with same logits
       let logits = create_test_logits()?;
       let standard_samples = sample_n_times(&logits, 1000, &sampler)?;
       
       // Use speculative path
       let speculative_samples = sample_n_times_speculative(&logits, 1000, &sampler)?;
       
       // Distributions should be statistically similar
       assert_distributions_similar(&standard_samples, &speculative_samples, 0.05);
   }
   
   #[test]
   fn test_frequency_penalty_applied() {
       let sampler = PenaltySampler::new(
           Box::new(GreedySampler),
           frequency_penalty=1.0,
       );
       
       let context = vec![1, 2, 3, 1, 1]; // Token 1 appears 3 times
       let logits = uniform_logits(vocab_size=100)?;
       
       let token = sampler.sample(&logits, &context)?;
       
       // Token 1 should be strongly penalized
       assert_ne!(token, 1);
   }
   ```

3. **`tests/speculative_integration.rs`**
   ```rust
   #[tokio::test]
   async fn test_api_with_speculative_decoding() {
       let server = start_test_server_with_speculative()?;
       
       let request = json!({
           "model": "llama-7b",
           "messages": [{"role": "user", "content": "Hello"}],
           "max_tokens": 50,
           "speculative_decoding": true,
       });
       
       let response = server.post("/v1/chat/completions")
           .json(&request)
           .send()
           .await?;
       
       assert_eq!(response.status(), 200);
       
       let body: ChatCompletionResponse = response.json().await?;
       assert!(body.usage.speculative_stats.is_some());
       
       let stats = body.usage.speculative_stats.unwrap();
       assert!(stats.acceptance_rate() > 0.0);
   }
   
   #[tokio::test]
   async fn test_graceful_fallback_no_draft_model() {
       // Server started without draft model
       let server = start_test_server_no_draft()?;
       
       let request = json!({
           "model": "llama-7b",
           "messages": [{"role": "user", "content": "Hello"}],
           "speculative_decoding": true,  // Requested but unavailable
       });
       
       let response = server.post("/v1/chat/completions").json(&request).send().await?;
       
       // Should succeed with standard decoding
       assert_eq!(response.status(), 200);
       
       let body: ChatCompletionResponse = response.json().await?;
       assert!(body.usage.speculative_stats.is_none()); // No speculation happened
   }
   ```

4. **`benchmarks/speculative_bench.rs`**
   ```rust
   fn bench_throughput_comparison(c: &mut Criterion) {
       let mut group = c.benchmark_group("throughput");
       
       let prompt = "Explain quantum computing in simple terms.";
       let max_tokens = 200;
       
       group.bench_function("standard_decoding", |b| {
           b.iter(|| generate_standard(black_box(prompt), max_tokens))
       });
       
       group.bench_function("speculative_decoding", |b| {
           b.iter(|| generate_speculative(black_box(prompt), max_tokens))
       });
       
       group.finish();
   }
   
   fn bench_acceptance_rate_vs_speedup(c: &mut Criterion) {
       // Test different draft model qualities
       let models = vec![
           ("identical", 1.0),      // 100% acceptance
           ("similar", 0.7),        // 70% acceptance
           ("mediocre", 0.4),       // 40% acceptance
           ("poor", 0.1),           // 10% acceptance
       ];
       
       for (name, acceptance_rate) in models {
           c.bench_function(&format!("acceptance_{}", name), |b| {
               let draft = MockDraftModel::with_acceptance(acceptance_rate);
               b.iter(|| run_speculation(black_box(&draft)))
           });
       }
   }
   ```

**Acceptance Criteria:**
- ✅ Correctness: Output identical to standard decoding (same seed)
- ✅ Stats: Acceptance rate tracking accurate within 1%
- ✅ Fallback: Auto-fallback activates after poor performance
- ✅ Sampling: All sampling methods work identically
- ✅ API: Integration tests pass (with/without draft model)
- ✅ Benchmarks: Speedup correlates with acceptance rate

---

## Performance Expectations

### Throughput Improvements

| Draft Acceptance Rate | Expected Speedup | Use Case                                       |
| --------------------- | ---------------- | ---------------------------------------------- |
| 80-100%               | 2.0-2.5×         | Same model family, similar training data       |
| 60-80%                | 1.5-2.0×         | Related architectures, good alignment          |
| 40-60%                | 1.2-1.5×         | Different families, decent alignment           |
| 20-40%                | 1.0-1.2×         | Poor alignment, minimal benefit                |
| 0-20%                 | 0.8-1.0×         | Overhead exceeds benefit, fallback recommended |

### Memory Overhead

| Configuration       | Target Model | Draft Model | Total Memory  |
| ------------------- | ------------ | ----------- | ------------- |
| 7B FP16 + 1B FP16   | 14GB         | 2GB         | 16GB (+14%)   |
| 7B FP16 + 1B AWQ    | 14GB         | 0.6GB       | 14.6GB (+4%)  |
| 13B FP16 + 1.5B AWQ | 26GB         | 0.9GB       | 26.9GB (+3%)  |
| 70B FP16 + 7B AWQ   | 140GB        | 4.2GB       | 144.2GB (+3%) |

**Recommendation**: Always quantize draft model (AWQ/GPTQ) to minimize memory overhead.

### Latency Impact

- **First-token latency**: +10-20ms (draft model forward pass)
- **Per-token latency**: -40-60% (amortized over accepted tokens)
- **Overall generation**: -30-60% latency for sequences >128 tokens

---

## Risks & Mitigations

| Risk                          | Impact | Mitigation                                                      |
| ----------------------------- | ------ | --------------------------------------------------------------- |
| Draft model too large         | HIGH   | Enforce size limits (draft <10% of target), quantize by default |
| Poor acceptance rate          | MEDIUM | Auto-fallback after 10 rounds, warn user in logs                |
| Memory fragmentation          | MEDIUM | Cache pooling, careful allocation order                         |
| Incorrect output vs standard  | HIGH   | Extensive correctness tests, same-seed validation               |
| API complexity                | LOW    | Graceful fallback if draft unavailable, clear error messages    |
| Quantized draft compatibility | MEDIUM | Test AWQ+FP16 before release, document limitations              |

---

## Success Metrics

**Must-have:**
- ✅ Output identical to standard decoding (deterministic)
- ✅ 1.5× throughput improvement at 60% acceptance rate
- ✅ <15% memory overhead (with quantized draft)
- ✅ Zero regressions for standard decoding path

**Nice-to-have:**
- 🎯 2× throughput improvement at 75% acceptance rate
- 🎯 <10% memory overhead (with AWQ draft)
- 🎯 Shared embedding layer support (same architecture families)
- 🎯 Per-request speculation control via API

---

## Timeline

**Week 1:**
- Day 1-2: Dual-model loading infrastructure
- Day 3-4: Advanced sampling integration
- Day 5: Memory estimation and validation

**Week 2:**
- Day 1-2: API integration (request/response)
- Day 3-4: CLI flags and configuration
- Day 5: Integration with existing generation pipeline

**Week 3:**
- Day 1-2: Memory optimizations (pooling, lazy loading)
- Day 3-4: Correctness and performance testing
- Day 5: Documentation, polish, benchmarks

**Total: 15 days of focused work**

---

## Integration with AWQ Plan

### Shared Design Decisions

1. **Model Loader Interface**
   - Must support `Option<QuantConfig>` parameter
   - Same loader for both target and draft models
   - **Action**: Update `load_local_llama()` signature in Phase 1 of both plans

2. **Linear Layer Abstraction**
   - `SpeculativeModel` trait must not depend on layer types
   - Both `Linear` and `QuantizedLinear` work transparently
   - **Action**: Verify trait interface in AWQ Phase 2 and Spec Decoding Phase 1

3. **Memory Accounting**
   - Shared `MemoryEstimate` struct for both features
   - Quantized weights counted correctly
   - **Action**: Create unified memory estimation in AWQ Phase 3

4. **Configuration Schema**
   - Both features in `config.toml` with clear sections
   - CLI flags don't conflict
   - **Action**: Review config schema in both Phase 4 sections

### Implementation Order

**Recommended sequence:**
1. ✅ AWQ Phase 1-2 (CUDA kernels, backend) - **Week 1**
2. ✅ Spec Decoding Phase 1 (dual-model loading) - **Week 1-2**
3. ✅ AWQ Phase 3 (model loader integration) - **Week 2**
4. ✅ Spec Decoding Phase 2-3 (sampling, API) - **Week 2**
5. ✅ Joint testing: AWQ draft + FP16 target - **Week 3**
6. ✅ AWQ Phase 4-5 + Spec Decoding Phase 4-6 (polish) - **Week 3**

**Parallelizable work:**
- AWQ CUDA kernels (independent)
- Speculative sampling refactor (independent)
- Memory estimation (shared - coordinate)

---

## References

### Papers

- **"Fast Inference from Transformers via Speculative Decoding"** (Leviathan et al., 2023)
  - Original speculative decoding algorithm
  - Verification strategy and acceptance criteria
  - https://arxiv.org/abs/2211.17192

- **"SpecInfer: Accelerating Generative LLM Serving with Speculative Inference"** (Miao et al., 2023)
  - System-level optimizations
  - Batch-level speculation
  - https://arxiv.org/abs/2305.09781

- **"Accelerating Large Language Model Decoding with Speculative Sampling"** (Chen et al., 2023)
  - Probabilistic verification
  - Sampling-based acceptance
  - https://arxiv.org/abs/2302.01318

### Implementations

- **HuggingFace Transformers**: `transformers.generation.utils.GenerationMixin.assisted_decoding()`
- **vLLM**: Speculative decoding support with draft models
- **Llama.cpp**: `--draft-model` CLI parameter
- **TensorRT-LLM**: Multi-model speculation

### Existing Code

- `src/engine/speculative.rs` - Core algorithm (lines 1-386)
- `src/model/speculative_adapters.rs` - Model adapters (lines 1-127)
- `examples/speculative_demo.rs` - Mock demonstration
- `src/init.rs` - Config helper (lines 141-179)

---

## Next Steps

1. ✅ **Complete this plan** (done!)
2. 📋 Review AWQ + Spec Decoding plans side-by-side
3. 📋 Create unified `MemoryEstimate` design document
4. 📋 Start Spec Decoding Phase 1 OR AWQ Phase 1 (user choice)
5. 📋 Test quantized draft model as soon as AWQ Phase 3 complete
