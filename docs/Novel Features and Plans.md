# Novel Features & Research Plan
## Original Vision: Build ON TOP of Candle, Not Replace It

This document covers many of the **novel ideas, research plans, and innovative features** that are intended to be built **on top of Candle** as a solid foundation. These are the things that make our project unique and valuable, not reimplementations of existing ML infrastructure.

---

## 🎯 Core Philosophy

**Our project should:**
1. **Use Candle** for all basic ML operations (tensors, matmul, attention, RoPE, etc.)
2. **Build novel optimizations** on top of Candle's proven implementations
3. **Experiment with research ideas** that aren't in mainstream frameworks yet
4. **Focus on innovative features** that differentiate us from llama.cpp/vLLM/etc.

**We should NOT:**
- Reimplement basic tensor operations
- Recreate standard attention mechanisms
- Rebuild existing quantization formats that Candle already supports

---

## 📊 Three-Phase Roadmap (From AREAS_FOR_FUTURE_RESEARCH.md)

### **Phase 1: Core Engine (Match vLLM/llama.cpp functionality)**
*Foundation - use Candle for all of this:*

1. ✅ **GGUF & Quantization Loading** - Candle already supports this
2. ✅ **PagedAttention** - Build on Candle's memory management
3. ✅ **Continuous Batching** - Scheduling layer on top of Candle inference
4. ✅ **FlashAttention** - Candle has this, ensure integration
5. ✅ **GQA/MQA Kernels** - Candle supports these architectures

**Goal**: Have a working, performant LLM inference engine using Candle

---

### **Phase 2: High-Impact Optimizations (Novel implementations)**

These are areas where we can **add value on top of Candle**:

#### 1. **Speculative Decoding** 🌟
- Use small draft model (1-3B) to predict multiple tokens
- Large model verifies in parallel
- **2-3x speedup** - massive win
- **Our Contribution**: Rust-native implementation with efficient two-model management
- **Papers**: 
  - "SpecInfer: Accelerating Generative LLM Serving with Speculative Inference"
  - Medusa/EAGLE architectures

#### 2. **Advanced Quantization (AWQ/SmoothQuant)** 🌟
- Activation-aware weight quantization
- Better quality than vanilla quantization at same bitrate
- **Our Contribution**: Port these algorithms to work with Candle's tensor ops
- **Papers**: 
  - "AWQ: Activation-aware Weight Quantization for LLM Compression" (2023)
  - "SmoothQuant: Accurate and Efficient Post-Training Quantization"

#### 3. **KV Cache Management** 🌟

**StreamingLLM** (MIT, 2023):
- Keep attention sinks + sliding window
- Enables "infinite" context in fixed memory
- **Our Contribution**: Custom KV cache manager on top of Candle
- **Paper**: "Efficient Streaming Language Models with Attention Sinks"

**H2O (Heavy-Hitter Oracle)** (2023):
- Dynamic KV cache eviction based on attention scores
- 50%+ KV cache reduction with minimal quality loss
- **Our Contribution**: Attention score tracking and intelligent eviction
- **Paper**: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"

**KIVI** (2024):
- Per-channel quantization of KV cache to 2-4 bits
- Huge memory savings for long contexts
- **Our Contribution**: Custom KV cache quantization layer
- **Paper**: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"

#### 4. **Custom CUDA Graphs / Persistent Kernels** 🌟
- Minimize kernel launch overhead for decode loop
- **Our Contribution**: Rust bindings for CUDA Graph API with Candle integration

---

### **Phase 3: Frontier Research (Cutting-edge implementations)**

These are **research-level features** that would make our project truly innovative:

#### 1. **Extreme Quantization** 🔬

**QuIP# (Cornell, 2023)**:
- 2-bit quantization with lattice codebooks
- Much better quality than naive 2-bit
- Can run 70B models where 13B would normally fit
- **Our Contribution**: First Rust implementation of QuIP# on top of Candle
- **Paper**: "QuIP#: Even Better LLM Quantization with Hadamard Incoherence"

**BitNet (Microsoft, 2023-2024)**:
- 1-bit weights (ternary: -1, 0, 1)
- BitNet b1.58 shows competitive performance
- Massive memory savings + potential for specialized hardware
- **Our Contribution**: BitNet inference engine in Rust
- **Papers**: 
  - "BitNet: Scaling 1-bit Transformers for Large Language Models"
  - "The Era of 1-bit LLMs"

#### 2. **PowerInfer (CPU/GPU Hybrid)** 🔬
- Exploits locality in neuron activation patterns
- Keep "hot" neurons on GPU, "cold" ones on CPU
- **11x speedup** for offloading scenarios on consumer GPUs
- **Our Contribution**: Rust's memory safety perfect for complex scheduling
- **Paper**: "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU" (2024)

#### 3. **FlexGen Scheduling** 🔬
- Sophisticated offloading between GPU/CPU/disk
- Throughput optimization given memory constraints
- Can run OPT-175B on a single 16GB GPU
- **Our Contribution**: Rust async + tokio for elegant scheduler
- **Paper**: "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU" (2023)

#### 4. **Flash-Decoding** 🔬
- Specifically optimized for decode phase (not just prefill)
- Different from Flash Attention
- **Our Contribution**: Decode-optimized kernels for Candle
- **Paper**: "Flash-Decoding for long-context inference" (2023)

#### 5. **Multi-Token Prediction** 🔬
- Training models to predict multiple future tokens at once
- Recent Meta research
- **Our Contribution**: Inference support for multi-head output models
- **Research**: Meta's recent work on multi-token prediction

#### 6. **Mixture-of-Experts (MoE) Optimizations** 🔬
- Custom routing logic
- Expert capacity management
- Efficient batching for MoE models like Mixtral
- **Our Contribution**: Rust-native MoE scheduler on Candle

---

## 🔧 System-Level Innovations

### 1. **Rust-Native Advantages**
From AREAS_FOR_FUTURE_RESEARCH.md:

**Memory Safety**:
- Aggressive memory reuse without GC concerns
- Custom allocators for KV cache management
- Arena allocators for generation-specific data

**Fearless Concurrency**:
- `rayon` for CPU-side parallelism (tokenization, sampling)
- `tokio` for high-throughput request scheduler
- Safe concurrent access to shared KV cache

**Cross-Platform with WGPU**:
- Single codebase for NVIDIA, AMD, Apple GPUs
- Candle's WGPU backend + our optimizations
- **Game-changer**: Works on AMD/Apple without CUDA lock-in
### 2. **Advanced Scheduling**
Beyond basic continuous batching:

**Throughput Optimization**:
- Dynamic batch sizing based on GPU utilization
- Priority queuing for different request types
- Memory-aware scheduling (don't OOM)

**Latency Optimization**:
- Chunked prefill (balance prefill/decode)
- Preemption for high-priority requests
- TTFT (Time To First Token) optimization

### 3. **Prompt/Context Optimization**

**LongLLMLingua (Microsoft, 2023)**:
- Compress prompts by 20x while maintaining performance
- Use smaller model to identify important tokens
- **Our Contribution**: Rust implementation using Candle for small model
- **Paper**: "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios"

**Gist Tokens / AutoCompressors**:
- Learn to compress context into small "summary" tokens
- Can represent large contexts in few vectors
- **Our Contribution**: Training harness for compression models

---

## 📈 Benchmarking & Optimization Infrastructure

### **Performance Testing**
*From automated testing work:*

1. **Token-Level Validation**: Compare outputs token-by-token with reference
2. **Latency Benchmarks**: TTFT, inter-token latency, total latency
3. **Throughput Tests**: Requests per second under various loads
4. **Memory Profiling**: Peak memory, KV cache efficiency
5. **Quality Metrics**: Perplexity, BLEU scores vs reference

### **CI/CD Integration**
- Automated regression testing
- Performance regression detection
- Quality benchmarks on every commit

---

## 📚 Key Research Papers to Implement

### **Must-Implement (High Impact)**:
1. ✅ "Flash-Decoding for long-context inference" (2023)
2. ✅ "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM, 2023)
3. ✅ "AWQ: Activation-aware Weight Quantization" (2023)
4. ✅ "SpecInfer: Accelerating Generative LLM Serving with Speculative Inference"
5. ✅ "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU" (2024)

### **High-Value (Performance)**:
6. "FlexGen: High-Throughput Generative Inference" (2023)
7. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference" (2023)
8. "KIVI: Tuning-Free Asymmetric 2bit Quantization for KV Cache" (2024)
9. "Efficient Memory Management for LLM Serving with PagedAttention" (vLLM)
10. "LongLLMLingua: Accelerating LLMs in Long Context Scenarios" (2023)

### **Cutting-Edge (Research)**:
11. "QuIP#: Even Better LLM Quantization with Hadamard Incoherence" (2023)
12. "BitNet: Scaling 1-bit Transformers for Large Language Models" (2023)
13. "The Era of 1-bit LLMs" (2024)
14. "Model Compression and Efficient Inference for LLMs: A Survey" (2024)

---

## 🚀 Implementation Priority

### **Week 1-2: Foundation**
- ✅ Get basic Candle inference working (Llama models)
- ✅ Load GGUF files using Candle's existing support
- ✅ Implement simple continuous batching

### **Week 3-4: First Novel Feature**
- 🎯 StreamingLLM (simplest, highest ROI)
  - Attention sink management
  - Sliding window KV cache
  - Enables infinite context

### **Week 5-8: High-Impact Optimizations**
- 🎯 Speculative Decoding (2-3x speedup)
- 🎯 AWQ Quantization (better quality)
- 🎯 Custom CUDA graphs (lower latency)

### **Week 9-16: Advanced Features**
- 🎯 PowerInfer (CPU/GPU hybrid)
- 🎯 FlexGen scheduling
- 🎯 KIVI KV cache quantization

### **Week 17+: Research & Innovation**
- 🔬 QuIP# extreme quantization
- 🔬 BitNet 1-bit models
- 🔬 Neural-symbolic integration
- 🔬 Lightning Strike reasoning

---

## 📖 Resources

### **Candle Documentation:**
- https://github.com/huggingface/candle
- https://huggingface.co/docs/candle

### **Research Paper Sources:**
- arXiv.org (most LLM papers)
- Papers with Code (implementations)
- HuggingFace Papers (curated)
- NeurIPS/ICML/ICLR proceedings

### **Inspiration:**
- vLLM: https://github.com/vllm-project/vllm
- llama.cpp: https://github.com/ggerganov/llama.cpp  
- SGLang: https://github.com/sgl-project/sglang
- HazyResearch: https://github.com/HazyResearch
