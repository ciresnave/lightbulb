# Implementation priorities and conflicts

Purpose: Sort literature-driven features by expected benefit and note potential conflicts or coupling before implementation.

## High-priority (Phase 0.2–0.4 enabling)

- Continuous batching MVP and hardened KvPager (core infra)
- Prefix KV caching (TTFT reduction) — compatible with most features
- StreamingLLM-style sliding window (bounded KV) — interacts with KV quant/compression, choose precedence at runtime
- Quantized model loaders and low-bit enablement (AWQ/SmoothQuant) — independent toggles
- Early exit (entropy + patience + optional MAB) — independent; add logging and guardrails

## Performance/scale (Phase 0.4–0.6)

- Hybrid linear attention schedule (3:1–6:1 linear:full) — requires backbone support; simulate via layer keep/skip until available
- Pruning utilities (Wanda + tail prune ~25% + partial-layer FT) — orthogonal to KV policies; affects model weights and eval baselines
- MoE-aware scheduler with capacity-aware routing — impacts batching logic, adds per-expert micro-batching and observability
- KV cache compression/quantization (R-KV/KIVI/KVQuant) — interacts with StreamingLLM; pick one KV policy family per run
- Speculative decoding — interacts with early exit (self-speculative) and batch shaping; feature flag recommended

## Potential conflicts and mitigations

- KV policies: Sliding window vs KV compression/quantization — expose a mutually exclusive selector; ensure metrics compare apples-to-apples
- Early exit vs CoLa depth adaptation — both alter depth; run separately for clean attribution, or define a combined policy with clear priority rules
- Pruning vs Quantization — verify order of operations (prune→quantize vs quantize→prune); document supported flows and test each
- MoE scheduling vs speculative/early exit — coordinate per-token control flow; gate interactions behind explicit configuration and robust logging
- Hybrid attention vs pruning — pruning tail layers may reduce the frequency of full-attention anchors; ensure schedule re-derives after pruning

## Measurement and gating

- Define small, representative evals per feature (LM ppl, recall micro-bench, reasoning subset) and require green metrics before enabling by default
- Log per-request traces: KV bytes, depth taken, exits taken, expert queues, token drops, and latencies (p50/p95/p99)
- Keep a conservative baseline path available; all advanced paths behind feature flags with parity tests

## References

- See docs/summaries/* for per-feature details and acceptance criteria.
