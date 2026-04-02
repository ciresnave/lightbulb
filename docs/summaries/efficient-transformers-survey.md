# Efficient Transformers: Key Takeaways for Lightbulb

Source: Tay et al., "Efficient Transformers: A Survey" (ACM Computing Surveys, 2022)

What matters for Lightbulb now

- Fixed/combined attention patterns, kernel/low-rank, and recurrence are rich research areas, but many require custom kernels or sacrifice autoregressive training parallelism.
- For an inference engine atop Candle, the most pragmatic wins are:
  - Streaming and memory policies (attention sinks, sliding windows, selective retention) to bound KV memory while preserving quality.
  - Batching and scheduling improvements (locality, chunked prefill, dynamic batch sizing) to drive throughput in real workloads.
  - Quantization and cache compression to reduce memory bandwidth and fit larger contexts/models.
- Avoid bespoke block-sparse kernels unless Candle exposes them; portability and maintenance matter.

Taxonomy (condensed)

- Fixed/combined patterns: local windows, strided/dilated, axial, BigBird/ETC (often need custom kernels for max gain).
- Learnable patterns: clustering (Routing), hashing (Reformer), sorting (Sinkhorn) — interesting but higher complexity.
- Low-rank/kernel: Linformer, Performer, Linear Transformer — linear memory, but training constraints; inference may benefit.
- Recurrence: Transformer-XL, Compressive Transformer — orthogonal approach with explicit memory.
- Downsampling: Perceiver, Funnel — resolution reduction; can pair with memory tokens.
- Sparse/MoE: Switch, GShard, GLaM — conditional compute; relevant for expert routing.

Concrete guidance for our roadmap

- Prioritize StreamingLLM-like KV management: combines fixed windows with learned “sinks” to emulate long context with bounded memory.
- Keep a conservative, full-attention path for parity; guard experimental policies behind feature flags.
- Speculative decoding is orthogonal and multiplicative with attention choices — high ROI on decode latency.
- Use Candle’s FlashAttention when available instead of maintaining custom kernels.
- Explore KV cache quantization (KIVI) and eviction heuristics (H2O) for memory reductions with minimal quality loss.

Risks/edge cases

- Some efficient attention methods don’t support causal decoding or require training-time changes; we’re inference-first.
- Kernel availability varies by backend; ensure CPU/WGPU fallbacks.
- Efficient complexity on paper may not translate to throughput without careful batching and memory layout.

How this informs milestones

- M2: implement streaming/sliding KV policy with sinks; wire FA when present.
- M3: add speculative decoding; reduce decode-loop overhead.
- M5: add KV compression/quantization options.
