# Changes analysis: candle-vllm and atoma-infer vs Candle

This document summarizes the raw differences between our local Candle fork and two idea sources: candle-vllm and atoma-infer. It also provides hypotheses for why those changes were made and how we might integrate them into a new crate, `lightbulb`.

Sources

- Candle fork (baseline): ./candle
- candle-vllm: ./idea_sources/candle-vllm
- atoma-infer: ./idea_sources/atoma-infer
- Raw diffs: docs/diff-candle_vs_candle-vllm.patch, docs/diff-candle_vs_atoma-infer.patch
- Filtered diffs (exclude .git/ etc.): docs/diff-filtered-candle_vs_candle-vllm.patch, docs/diff-filtered-candle_vs_atoma-infer.patch

Method

- We generated directory-wide diffs using `git diff --no-index` between Candle and each idea source. This reports added/removed files and patches across the trees.
- For large diffs, we focus on top-level modules and key subsystems (model loaders, schedulers, runtime/backends, quantization, attention kernels, sampling, and I/O).

Highlights observed in filtered diffs (see detailed summaries in docs/summaries/):

- candle-vllm (summary: docs/summaries/candle-vllm-summary.txt)
   - Paged KV cache + block-level scheduler
   - CUDA and Metal paged attention kernels, Marlin/GPTQ support
   - OpenAI-compatible server, sampling, and model adapters

- atoma-infer (summary: docs/summaries/atoma-infer-summary.txt)
   - vLLM-style engine with rich scheduler/evictor, multi-GPU/NCCL paths
   - CUDA kernels with Rust FFI for cache manager and flash attention
   - OpenAI-like server and schema validation

High-level themes to integrate into lightbulb:

1) vLLM-style request batching and paged KV cache management
   - Scheduler and engine abstractions for handling multiple concurrent generation requests
   - PagedKV or block-based cache to reduce memory fragmentation and enable preemption
   - Priority / fairness policies, budgeted token allocation per step

2) Specialized attention and sampling optimizations
   - Flash-attention kernels or fused matmul+softmax paths
   - Grouped-query attention, multi-query attention, rotary embeddings fast-paths
   - Advanced sampling (nucleus, Mirostat, typical, repetition penalties) with vectorized ops

3) Quantization and weight loading improvements
   - GGUF/GGML/parquet/safetensors variants, 4-bit/8-bit quant loaders
   - On-the-fly dequant cache and mixed-precision control

4) Runtime and device backends
   - CUDA/cuBLAS/cuDNN or Metal/ROCm bindings via Candle backends
   - Streamed generation I/O (server/CLI), tokenizer pipelines, prompt caching

5) Model registry and config-driven loaders
   - Model families (Llama/Mistral/Gemma/Mixtral/Qwen) with specialized init paths
   - Checkpoint sharding support and lazy weight mapping

Next steps

- Parse the filtered diff files, categorize by crate/module, and extract a table of notable changes.
- Identify reusable building blocks and decide on `lightbulb` module boundaries (engine, scheduler, cache, kernels, loaders, sampling, io, registry).
- Draft a minimal MVP plan:
   - Bring over request scheduler + paged KV from candle-vllm
   - Add sampling + loader enhancements from atoma-infer
   - Keep core math on Candle, behind a clean trait boundary

Appendix

- See the raw diff files for the exact patch. We will refine this doc with concrete file and symbol lists as we ingest those patches programmatically.
