# Efficient Reasoning Models: A Survey

**Full PDF:** [View Original](../papers/pdf/EfficientReasoningModels-ASurvey.pdf)  
**Markdown:** [View Markdown](../papers/markdown/efficientreasoningmodels-asurvey.md)

## TL;DR

Comprehensive survey categorizing efficient reasoning methods into three approaches: **shorter** (compressing lengthy Chain-of-Thought reasoning), **smaller** (compact models with strong reasoning via distillation/compression/RL), and **faster** (efficient decoding strategies). Addresses the computational overhead of Large Reasoning Models (LRMs) that generate long CoT sequences.

## Why it matters

- LRMs achieve remarkable performance but with substantial computational cost from long reasoning chains
- Provides systematic taxonomy of efficiency techniques applicable to our inference engine
- Identifies key trade-offs between reasoning quality and computational efficiency
- Directly relevant to our goal of building an efficient ML training/inference library on Candle

## Key technical takeaways

1. **Shorter CoT approaches**: RL with length penalties, SFT on variable-length data, prompt-driven routing, and latent reasoning (performing reasoning in latent space without explicit tokens)
2. **Smaller model approaches**: Knowledge distillation transfers reasoning from large to small models; quantization/pruning can maintain reasoning with careful application; RL can train small models (<2B params) to match larger reasoning models
3. **Faster decoding approaches**: Efficient sampling (early termination of low-quality paths), efficient self-consistency (adaptive sample budgets), and speculative decoding for reasoning chains
4. **Model memory techniques**: Global memory tokens, inducing points (Set Transformers), and learnable sparse patterns reduce quadratic attention costs
5. **Key insight**: Longer CoT doesn't always improve performance—some methods show negative returns beyond optimal lengths

## Implementation steps for lightbulb

- Study latent reasoning approaches for potential token-efficient inference modes in our engine
- Consider implementing efficient attention patterns (e.g., sliding windows, sparse attention) in Candle-based attention kernels
- Evaluate knowledge distillation pipeline for creating compact reasoning models from larger teacher models
- Implement speculative decoding with early path pruning for reasoning workloads
- Add telemetry for CoT length vs. accuracy trade-offs to inform adaptive compute allocation
- Consider routing mechanisms to dynamically allocate compute based on problem difficulty

## Acceptance criteria

- Prototype at least one "shorter" technique (e.g., length-penalized RL loss or prompt-based routing) showing measurable latency improvement
- Implement efficient attention kernel (local or sparse) with benchmarks showing memory/speed gains over dense attention
- Create distillation pipeline capable of transferring reasoning capability with <20% accuracy degradation
- Reasoning telemetry dashboard shows CoT length, latency, and accuracy metrics in real-time
