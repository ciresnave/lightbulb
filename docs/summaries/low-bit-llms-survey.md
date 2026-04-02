# Survey: Low-bit Large Language Models — Summary

Source: A Survey of Low-Bit Large Language Models: Basics, Systems, and Algorithms (2024)

TL;DR

- Low-bit quantization spans formats (INT/FP8/FP4/NF/SF), granularities (tensor/channel/group/element), and scopes (weights, activations, KV cache).
- Practical systems use weight-only 4–8 bit (GPTQ/AWQ) and activation-aware scaling (SmoothQuant/QuaRot), plus dedicated KV cache compression (KVQuant/KIVI/WKVQuant).

Key landscape

- Algorithms: GPTQ (PTQ), AWQ (activation-aware weight quant), SmoothQuant (absorb activation ranges into weights), QuaRot/SpinQuant (rotation + scaling). Newer weight+activation schemes combine per-channel/group scaling.
- KV cache: KVQuant, KIVI, WKVQuant reduce KV precision (2–4 bit) with learned scales or error compensation to preserve attention quality.
- Systems: llama.cpp, TensorRT-LLM, vLLM, QServe integrate quantized execution paths and cache policies; bandwidth and memory are often the bottleneck.

Actionable for Lightbulb

- Expose quantized loaders via Candle where supported (GGUF etc.). Provide a clear matrix of: weight-only (4/8b), weight+activation (SmoothQuant/AWQ-like), and baseline FP16/FP32 paths.
- Plan a KV cache quantization feature flag: start with simple per-head/group scales in 4b for keys/values; add error-aware refinements later.
- Add evaluation hooks: track perplexity deltas on a small corpus and decode latency/memory for each format; surface in docs.

Acceptance criteria (initial)

- Weight-only quantized tiny model runs end-to-end; deterministic tests pass on CPU.
- KV cache 4b prototype shows ≥40% KV memory reduction with negligible degradation on short QA prompts; document longer-context tradeoffs.
- Documentation includes a compatibility table (what works on CPU/WGPU/CUDA in Candle) and expected quality impacts.

Risks/notes

- Activation quantization may require kernel support for scale/dequant; stay aligned with Candle capabilities.
- KV quantization changes attention numerics; keep conservative defaults and easy fallbacks.

Citation

- A Survey of Low-Bit Large Language Models: Basics, Systems, and Algorithms (2024).
