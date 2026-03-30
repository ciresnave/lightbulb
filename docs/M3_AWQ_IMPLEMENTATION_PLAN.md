# M3: AWQ/Marlin Quantization Implementation Plan

**Status**: PLANNED (Ready to implement)  
**Priority**: HIGH (enables 4-bit inference with 2× memory reduction)  
**Timeline**: 2-3 weeks  
**Dependencies**: M3.6 Multi-GPU complete, CUDA feature available

---

## Overview

Integrate Marlin kernel for efficient 4-bit AWQ and GPTQ quantized inference. Based on proven implementation from candle-vllm (MIT) and Marlin kernel (Apache-2.0).

**Performance targets:**
- Memory: 2× reduction vs FP16 (7B model: 14GB → 7GB)
- Throughput: 1.5-2× faster than naive 4-bit (Marlin optimization)
- Accuracy: <1% degradation vs FP16

---

## License Compliance

✅ **All licenses compatible with Lightbulb's dual MIT/Apache-2.0 license**

### Source Components

| Component               | License    | Source                                                     | Usage                        |
| ----------------------- | ---------- | ---------------------------------------------------------- | ---------------------------- |
| Marlin CUDA kernels     | Apache-2.0 | [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin)  | Direct copy with attribution |
| candle-vllm integration | MIT        | [candle-vllm](https://github.com/EricLBuehler/candle-vllm) | Adapt backend logic          |
| FFI bindings            | MIT        | candle-vllm                                                | Copy interface definitions   |

### Attribution Requirements

1. Add to `docs/THIRD_PARTY_NOTICES.md`:
   ```markdown
   ## Marlin Kernel
   
   **License**: Apache License 2.0  
   **Source**: https://github.com/IST-DASLab/marlin  
   **Copyright**: IST Austria DASL Lab  
   **Usage**: CUDA kernels for 4-bit GPTQ/AWQ inference
   
   Portions of the quantization kernel implementation are derived from the Marlin
   project, which provides high-performance GPU kernels for quantized matrix
   multiplication.
   
   [Full Apache 2.0 license text...]
   
   ## candle-vllm Integration
   
   **License**: MIT License  
   **Source**: https://github.com/EricLBuehler/candle-vllm  
   **Copyright**: Eric Buehler and contributors  
   **Usage**: Backend integration logic and FFI bindings
   
   The AWQ/GPTQ backend integration is adapted from candle-vllm's quantization
   implementation.
   
   [Full MIT license text...]
   ```

2. Preserve copyright notices in source files:
   ```rust
   // Portions adapted from candle-vllm (MIT License)
   // Copyright (c) Eric Buehler and contributors
   // https://github.com/EricLBuehler/candle-vllm
   ```

3. Update `README.md` acknowledgments section

---

## Implementation Phases

### Phase 1: CUDA Kernel Integration (Week 1)

**Goal**: Get Marlin kernels compiling and callable from Rust

#### Files to Create

1. **`lightbulb/kernels/marlin/`** (Copy from candle-vllm)
   ```
   kernels/marlin/
   ├── marlin.cuh              # Core Marlin kernel (Apache-2.0)
   ├── marlin_dtypes.cuh       # Data type utilities
   └── marlin_cuda_kernel.cuh  # Main kernel templates
   ```

2. **`lightbulb/kernels/src/marlin_*.cu`** (Copy from candle-vllm)
   ```
   kernels/src/
   ├── marlin_matmul_f16.cu       # FP16 inference
   ├── marlin_matmul_bf16.cu      # BF16 inference
   ├── marlin_matmul_awq_f16.cu   # AWQ-specific FP16
   ├── marlin_matmul_awq_bf16.cu  # AWQ-specific BF16
   └── marlin_repack.cu           # Weight repacking
   ```

3. **`lightbulb/kernels/src/marlin_ffi.rs`** (Adapt from candle-vllm `ffi.rs`)
   ```rust
   // FFI bindings for Marlin kernels
   extern "C" {
       pub fn marlin_4bit_f16(
           inputs: *const c_void,
           weight: *const i32,
           scales: *const c_void,
           zeros: *const c_void,
           g_idx: *const c_void,
           out: *mut c_void,
           m: c_int, k: c_int, n: c_int,
           workspace: *const c_void,
           groupsize: c_int,
           stream: i64,
       );
       
       pub fn marlin_awq_4bit_f16(/* similar signature */);
       pub fn gptq_repack(/* weight repacking */);
       pub fn awq_repack(/* AWQ weight repacking */);
   }
   ```

4. **Update `kernels/build.rs`**
   ```rust
   // Add Marlin kernel compilation
   if cfg!(feature = "cuda") {
       let marlin_sources = vec![
           "src/marlin_matmul_f16.cu",
           "src/marlin_matmul_bf16.cu",
           "src/marlin_matmul_awq_f16.cu",
           "src/marlin_matmul_awq_bf16.cu",
           "src/marlin_repack.cu",
       ];
       
       for src in marlin_sources {
           cc::Build::new()
               .cuda(true)
               .flag("-gencode").flag("arch=compute_80,code=sm_80") // Ampere+
               .flag("-gencode").flag("arch=compute_89,code=sm_89") // Ada
               .file(src)
               .compile(&format!("marlin_{}", ...));
       }
   }
   ```

**Acceptance:**
- ✅ Kernels compile without errors on CUDA 11.8+
- ✅ FFI functions callable from Rust
- ✅ Simple test passes (allocate tensors, call kernel, check output shape)

---

### Phase 2: Rust Backend Integration (Week 1-2)

**Goal**: Wrap kernels in Candle-compatible custom ops

#### Files to Create

1. **`src/backend/marlin.rs`** (Adapt from candle-vllm `gptq.rs`)
   ```rust
   use candle::{CustomOp3, Result, Tensor};
   use crate::kernels::marlin_ffi::*;
   
   /// Marlin-optimized quantized matmul
   pub struct MarlinMatMul {
       /// Quantized zero-points (optional for GPTQ)
       qzeros: Option<Tensor>,
       /// Group index mapping (optional)
       g_idx: Option<Tensor>,
       /// Workspace tensor for kernel
       workspace: Tensor,
       /// Number of bits (4 or 8)
       bits: i32,
       /// Group size for quantization (-1 for per-channel)
       group_size: i32,
       /// True if AWQ format, false if GPTQ
       is_awq: bool,
   }
   
   impl CustomOp3 for MarlinMatMul {
       fn name(&self) -> &'static str { "MarlinMatMul" }
       
       fn cpu_fwd(&self, ...) -> Result<...> {
           candle::bail!("Marlin only supports CUDA")
       }
       
       #[cfg(feature = "cuda")]
       fn cuda_fwd(&self, x: &CudaStorage, qweight: &CudaStorage, scales: &CudaStorage) -> Result<...> {
           // Get tensor pointers
           let x_ptr = x.as_cuda_slice::<f16>()?.device_ptr();
           let qw_ptr = qweight.as_cuda_slice::<i32>()?.device_ptr();
           let scale_ptr = scales.as_cuda_slice::<f16>()?.device_ptr();
           
           // Allocate output
           let out = unsafe { device.alloc::<f16>(out_shape.elem_count()) }?;
           
           // Call kernel
           unsafe {
               if self.is_awq {
                   marlin_awq_4bit_f16(x_ptr, qw_ptr, scale_ptr, ...);
               } else {
                   marlin_4bit_f16(x_ptr, qw_ptr, scale_ptr, ...);
               }
           }
           
           Ok((CudaStorage::wrap(out), out_shape))
       }
   }
   
   /// Public API for quantized matmul
   pub fn marlin_matmul(
       x: &Tensor,
       qweight: &Tensor,
       scales: &Tensor,
       qzeros: Option<&Tensor>,
       g_idx: Option<&Tensor>,
       workspace: &Tensor,
       bits: i32,
       group_size: i32,
       is_awq: bool,
   ) -> Result<Tensor> {
       let op = MarlinMatMul {
           qzeros: qzeros.cloned(),
           g_idx: g_idx.cloned(),
           workspace: workspace.clone(),
           bits,
           group_size,
           is_awq,
       };
       x.apply_op3(qweight, scales, op)
   }
   ```

2. **`src/backend/marlin_repack.rs`**
   ```rust
   /// Repack GPTQ/AWQ weights to Marlin format
   pub struct MarlinRepack {
       bits: i32,
       is_awq: bool,
   }
   
   impl CustomOp1 for MarlinRepack {
       fn cuda_fwd(&self, qweight: &CudaStorage) -> Result<...> {
           // Call gptq_repack or awq_repack based on format
           unsafe {
               if self.is_awq {
                   awq_repack(q_ptr, out_ptr, k, n, self.bits, stream);
               } else {
                   gptq_repack(q_ptr, out_ptr, m, n, stream);
               }
           }
       }
   }
   
   pub fn marlin_weight_repack(qweight: &Tensor, bits: i32, is_awq: bool) -> Result<Tensor> {
       qweight.apply_op1(MarlinRepack { bits, is_awq })
   }
   ```

3. **`src/backend/mod.rs`**
   ```rust
   #[cfg(feature = "cuda")]
   pub mod marlin;
   #[cfg(feature = "cuda")]
   pub mod marlin_repack;
   
   #[cfg(feature = "cuda")]
   pub use marlin::{marlin_matmul, MarlinMatMul};
   #[cfg(feature = "cuda")]
   pub use marlin_repack::{marlin_weight_repack, MarlinRepack};
   ```

**Acceptance:**
- ✅ `marlin_matmul()` compiles and runs
- ✅ Simple 4×4 matmul test passes with known weights
- ✅ Weight repacking produces expected output shape
- ✅ Errors gracefully on CPU (no silent failures)

---

### Phase 3: Model Loader Integration (Week 2)

**Goal**: Load quantized models and detect format automatically

#### Files to Modify

1. **`src/loaders.rs`** (Extend `load_local_llama`)
   ```rust
   pub fn load_local_llama(
       model_dir: &str,
       ...
   ) -> Result<(BatchedTransformer, ..., Option<QuantConfig>)> {
       // Load config.json
       let config: ModelConfig = serde_json::from_str(&config_str)?;
       
       // Detect quantization
       let quant_config = if let Some(qc) = config.quantization_config {
           Some(QuantConfig {
               bits: qc.bits,
               group_size: qc.group_size,
               quant_method: qc.quant_method, // "gptq", "awq", or "marlin"
               checkpoint_format: qc.checkpoint_format, // "marlin" if already repacked
               desc_act: qc.desc_act.unwrap_or(false),
               sym: qc.sym.unwrap_or(true),
           })
       } else {
           None
       };
       
       // Pass to transformer builder
       let model = BatchedTransformer::new(config, device, quant_config)?;
       
       Ok((model, ..., quant_config))
   }
   ```

2. **`src/model/custom_transformer.rs`** (Add quantization support)
   ```rust
   pub struct BatchedTransformerConfig {
       // ... existing fields ...
       
       /// Quantization configuration (None = FP16/BF16)
       pub quant_config: Option<QuantConfig>,
   }
   
   impl BatchedTransformer {
       pub fn new(config: BatchedTransformerConfig, device: Device) -> Result<Self> {
           // Build layers with quantization support
           let layers = (0..config.num_layers).map(|_| {
               TransformerBlock::new(&config, device, &config.quant_config)?
           }).collect::<Result<Vec<_>>>()?;
           
           Ok(Self { layers, config, device })
       }
   }
   ```

3. **`src/model/quantized_linear.rs`** (New file)
   ```rust
   use crate::backend::marlin::{marlin_matmul, marlin_weight_repack};
   
   pub struct QuantizedLinear {
       /// Quantized weights (int4 packed)
       qweight: Tensor,
       /// FP16 scales (per-group)
       scales: Tensor,
       /// Optional zero-points (GPTQ only)
       qzeros: Option<Tensor>,
       /// Optional group index (for desc_act=True)
       g_idx: Option<Tensor>,
       /// Workspace for kernel
       workspace: Tensor,
       /// Config
       config: QuantConfig,
   }
   
   impl QuantizedLinear {
       pub fn new(
           in_dim: usize,
           out_dim: usize,
           vb: VarBuilder,
           quant_config: &QuantConfig,
       ) -> Result<Self> {
           // Load quantized weights
           let qweight = vb.get((in_dim / pack_factor, out_dim), "qweight")?;
           let scales = vb.get((in_dim / quant_config.group_size, out_dim), "scales")?;
           
           // Optional components
           let qzeros = if quant_config.quant_method == "gptq" {
               Some(vb.get((in_dim / quant_config.group_size, out_dim / pack_factor), "qzeros")?)
           } else {
               None
           };
           
           // Repack to Marlin format if needed
           let qweight = if quant_config.checkpoint_format != Some("marlin") {
               marlin_weight_repack(&qweight, quant_config.bits, quant_config.quant_method == "awq")?
           } else {
               qweight
           };
           
           // Allocate workspace
           let workspace = Tensor::zeros(out_dim, DType::U32, vb.device())?;
           
           Ok(Self { qweight, scales, qzeros, g_idx: None, workspace, config: quant_config.clone() })
       }
       
       pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
           marlin_matmul(
               x,
               &self.qweight,
               &self.scales,
               self.qzeros.as_ref(),
               self.g_idx.as_ref(),
               &self.workspace,
               self.config.bits,
               self.config.group_size,
               self.config.quant_method == "awq",
           )
       }
   }
   ```

4. **`src/model/custom_transformer_block.rs`** (Conditional layer construction)
   ```rust
   pub struct TransformerBlock {
       attn_qkv: Box<dyn Module>,  // Either Linear or QuantizedLinear
       attn_o: Box<dyn Module>,
       mlp_gate_up: Box<dyn Module>,
       mlp_down: Box<dyn Module>,
       // ... other fields ...
   }
   
   impl TransformerBlock {
       pub fn new(config: &BatchedTransformerConfig, device: Device, quant_config: &Option<QuantConfig>) -> Result<Self> {
           let vb = VarBuilder::zeros(/* ... */);
           
           let (attn_qkv, attn_o, mlp_gate_up, mlp_down) = if let Some(qc) = quant_config {
               // Quantized layers
               (
                   Box::new(QuantizedLinear::new(config.hidden_size, config.hidden_size * 3, vb.clone(), qc)?) as Box<dyn Module>,
                   Box::new(QuantizedLinear::new(config.hidden_size, config.hidden_size, vb.clone(), qc)?) as Box<dyn Module>,
                   Box::new(QuantizedLinear::new(config.hidden_size, config.intermediate_size * 2, vb.clone(), qc)?) as Box<dyn Module>,
                   Box::new(QuantizedLinear::new(config.intermediate_size, config.hidden_size, vb, qc)?) as Box<dyn Module>,
               )
           } else {
               // FP16/BF16 layers
               (
                   Box::new(Linear::new(config.hidden_size, config.hidden_size * 3, vb.clone())?) as Box<dyn Module>,
                   Box::new(Linear::new(config.hidden_size, config.hidden_size, vb.clone())?) as Box<dyn Module>,
                   Box::new(Linear::new(config.hidden_size, config.intermediate_size * 2, vb.clone())?) as Box<dyn Module>,
                   Box::new(Linear::new(config.intermediate_size, config.hidden_size, vb)?) as Box<dyn Module>,
               )
           };
           
           Ok(Self { attn_qkv, attn_o, mlp_gate_up, mlp_down, /* ... */ })
       }
   }
   ```

**Acceptance:**
- ✅ Load quantized Llama model from directory
- ✅ Automatic format detection from config.json
- ✅ Correct weight shapes loaded
- ✅ Model forward pass runs without errors

---

### Phase 4: Conversion Tools (Week 2)

**Goal**: Provide scripts to convert models to Marlin format

#### Files to Create

1. **`examples/convert_awq_marlin.py`** (Copy from candle-vllm)
   ```python
   # Converts AWQ-quantized models to Marlin-compatible format
   # Usage: python examples/convert_awq_marlin.py \
   #   --src /path/to/awq-model \
   #   --dst /path/to/marlin-model \
   #   --bits 4 --method awq --group 128
   ```

2. **`examples/convert_gptq_marlin.py`** (Adapt from candle-vllm)
   ```python
   # Converts GPTQ models to Marlin format
   # Usage: python examples/convert_gptq_marlin.py \
   #   --src /path/to/gptq-model \
   #   --dst /path/to/marlin-model \
   #   --bits 4 --group 128
   ```

3. **`docs/QUANTIZATION_GUIDE.md`**
   ```markdown
   # Quantization Guide
   
   ## Supported Formats
   
   - **GPTQ**: 4-bit, group_size=128, desc_act=False, sym=True
   - **AWQ**: 4-bit, group_size=128
   - **Marlin**: Optimized format (converted from GPTQ/AWQ)
   
   ## Converting Models
   
   ### AWQ → Marlin
   ```bash
   python examples/convert_awq_marlin.py \
     --src /models/llama-7b-awq \
     --dst /models/llama-7b-marlin \
     --bits 4 --method awq --group 128
   ```
   
   ### GPTQ → Marlin
   ```bash
   python examples/convert_gptq_marlin.py \
     --src /models/llama-7b-gptq \
     --dst /models/llama-7b-marlin \
     --bits 4 --group 128
   ```
   
   ## Loading Quantized Models
   
   ```bash
   # Marlin format (fastest)
   lightbulb serve --model /models/llama-7b-marlin --dtype f16
   
   # AWQ format (auto-converts to Marlin on load)
   lightbulb serve --model /models/llama-7b-awq --dtype f16
   
   # GPTQ format (auto-converts to Marlin on load)
   lightbulb serve --model /models/llama-7b-gptq --dtype f16
   ```
   ```

**Acceptance:**
- ✅ Convert AWQ model successfully
- ✅ Convert GPTQ model successfully
- ✅ Converted models load correctly
- ✅ Documentation clear and complete

---

### Phase 5: Testing & Validation (Week 3)

**Goal**: Ensure correctness and performance

#### Test Files

1. **`tests/marlin_correctness.rs`**
   ```rust
   #[test]
   #[cfg(feature = "cuda")]
   fn test_marlin_matmul_vs_fp16() {
       // Load model twice: once FP16, once quantized
       let fp16_model = load_model("llama-7b-fp16", DType::F16)?;
       let quant_model = load_model("llama-7b-awq-marlin", DType::F16)?;
       
       // Same input
       let input = Tensor::randn(&[1, 128, 4096], DType::F16, &device)?;
       
       // Compare outputs
       let fp16_out = fp16_model.forward(&input)?;
       let quant_out = quant_model.forward(&input)?;
       
       // Allow small error (quantization loss)
       let diff = (fp16_out - quant_out)?.abs()?.max(0)?;
       assert!(diff.to_scalar::<f32>()? < 0.1); // <10% error
   }
   
   #[test]
   #[cfg(feature = "cuda")]
   fn test_marlin_generation_quality() {
       let model = load_model("llama-7b-awq-marlin", DType::F16)?;
       let tokenizer = load_tokenizer("llama-7b-awq-marlin")?;
       
       let prompt = "The capital of France is";
       let output = model.generate(prompt, 10)?;
       
       // Check for reasonable completion
       assert!(output.contains("Paris"));
   }
   ```

2. **`tests/marlin_performance.rs`**
   ```rust
   #[test]
   #[cfg(feature = "cuda")]
   fn bench_marlin_vs_fp16() {
       let fp16_model = load_model("llama-7b-fp16", DType::F16)?;
       let quant_model = load_model("llama-7b-awq-marlin", DType::F16)?;
       
       let input = Tensor::randn(&[16, 128, 4096], DType::F16, &device)?;
       
       // Warmup
       for _ in 0..10 {
           fp16_model.forward(&input)?;
           quant_model.forward(&input)?;
       }
       
       // Benchmark
       let fp16_time = benchmark(|| fp16_model.forward(&input), 100)?;
       let quant_time = benchmark(|| quant_model.forward(&input), 100)?;
       
       println!("FP16: {:.2}ms, Quantized: {:.2}ms, Speedup: {:.2}×",
           fp16_time, quant_time, fp16_time / quant_time);
       
       // Should be faster (target: 1.5-2× speedup)
       assert!(quant_time < fp16_time * 0.67);
   }
   ```

3. **`tests/marlin_memory.rs`**
   ```rust
   #[test]
   #[cfg(feature = "cuda")]
   fn test_marlin_memory_usage() {
       let device = Device::cuda_if_available(0)?;
       
       let initial_mem = device.memory_allocated()?;
       
       let model = load_model("llama-7b-awq-marlin", DType::F16)?;
       
       let final_mem = device.memory_allocated()?;
       let model_mem = final_mem - initial_mem;
       
       println!("Model memory: {:.2} GB", model_mem as f64 / 1e9);
       
       // 7B model should be ~7GB in Marlin format (vs ~14GB FP16)
       assert!(model_mem < 8_000_000_000); // <8GB
   }
   ```

4. **`benchmarks/quantization_bench.rs`**
   ```rust
   // Comprehensive benchmark comparing:
   // - FP16 baseline
   // - AWQ (naive 4-bit)
   // - GPTQ (naive 4-bit)
   // - Marlin (optimized 4-bit)
   // 
   // Metrics: throughput (tok/s), latency (ms), memory (GB), accuracy (perplexity)
   ```

**Acceptance:**
- ✅ Correctness: <1% error vs FP16 on generation
- ✅ Performance: 1.5-2× speedup vs FP16
- ✅ Memory: 2× reduction (14GB → 7GB for Llama-7B)
- ✅ Quality: Perplexity degradation <2% on WikiText-2

---

## Hardware Requirements

**Minimum:**
- NVIDIA GPU: Ampere (A100, RTX 30xx) or newer
- CUDA: 11.8+
- VRAM: 8GB+ for 7B models
- Driver: 520.61+

**Optimal:**
- NVIDIA GPU: Ada (RTX 40xx) or Hopper (H100)
- CUDA: 12.0+
- VRAM: 24GB+ for 13B models

**Limitations:**
- Marlin requires Ampere or newer (sm_80+)
- No CPU fallback (CUDA-only)
- Group size must be 128 or -1 (per-channel)
- Only 4-bit supported (8-bit planned)

---

## Feature Flags

```toml
[features]
default = []
cuda = ["candle/cuda"]
marlin = ["cuda"]  # Marlin implies CUDA
```

Usage:
```bash
# Build with Marlin support
cargo build --features marlin

# Run tests
cargo test --features marlin -- --test-threads=1
```

---

## Documentation

1. **User-facing:**
   - `docs/QUANTIZATION_GUIDE.md` - How to use quantized models
   - `README.md` - Quick start with quantization
   - `examples/quantized_inference.rs` - Code example

2. **Developer-facing:**
   - `docs/M3_AWQ_IMPLEMENTATION_PLAN.md` (this file)
   - `docs/MARLIN_KERNEL_DESIGN.md` - Kernel architecture
   - `docs/THIRD_PARTY_NOTICES.md` - License attributions

3. **API reference:**
   - `src/backend/marlin.rs` - Inline docs
   - `src/model/quantized_linear.rs` - Inline docs

---

## Risks & Mitigations

| Risk                                   | Impact | Mitigation                                                          |
| -------------------------------------- | ------ | ------------------------------------------------------------------- |
| Kernel compilation fails on older GPUs | HIGH   | Detect GPU architecture, skip Marlin on sm<80                       |
| Accuracy degradation >2%               | MEDIUM | Validate on multiple benchmarks, document limitations               |
| Memory leaks in CUDA kernels           | HIGH   | Extensive leak testing, valgrind with CUDA                          |
| Integration breaks existing models     | HIGH   | Feature flag, comprehensive regression tests                        |
| Performance worse than expected        | MEDIUM | Profile kernels, optimize hot paths, document hardware requirements |

---

## Success Metrics

**Must-have:**
- ✅ Load and run AWQ/GPTQ models correctly
- ✅ 2× memory reduction vs FP16
- ✅ <1% accuracy loss on WikiText-2
- ✅ Zero regressions for FP16 models

**Nice-to-have:**
- 🎯 1.5-2× throughput improvement
- 🎯 Support 13B models on 24GB GPUs
- 🎯 Automatic format detection and conversion

---

## Timeline

**Week 1:**
- Day 1-2: Copy kernels, set up build system
- Day 3-4: FFI bindings, basic tests
- Day 5: Rust backend wrapper

**Week 2:**
- Day 1-2: Model loader integration
- Day 3-4: QuantizedLinear implementation
- Day 5: Conversion scripts

**Week 3:**
- Day 1-2: Correctness testing
- Day 3-4: Performance benchmarks
- Day 5: Documentation, polish

**Total: 15 days of focused work**

---

## References

- **Marlin kernel**: https://github.com/IST-DASLab/marlin
- **candle-vllm**: https://github.com/EricLBuehler/candle-vllm
- **AWQ paper**: https://arxiv.org/abs/2306.00978
- **GPTQ paper**: https://arxiv.org/abs/2210.17323
- **Quantization survey**: `docs/summaries/asurveyoflowbitlargelanguagemodels.md`

---

## Next Steps

1. ✅ **Complete this plan** (done!)
2. 📋 Create detailed speculative decoding plan (M3 priority)
3. 📋 Start Marlin implementation (Phase 1: Week 1)
