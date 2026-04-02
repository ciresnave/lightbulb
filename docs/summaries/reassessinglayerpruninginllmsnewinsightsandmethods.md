---
title: Reassessing Layer Pruning in LLMs: New Insights and Methods
source_pdf: [ReassessingLayerPruningInLLMs-NewInsightsAndMethods.pdf](../../../Desktop/books%20and%20courses/Machine%20Learning/ReassessingLayerPruningInLLMs-NewInsightsAndMethods.pdf)
source_markdown: [reassessinglayerpruninginllmsnewinsightsandmethods.md](../papers/markdown/reassessinglayerpruninginllmsnewinsightsandmethods.md)
---

# TL;DR

This paper benchmarks layer pruning in large language models (LLMs), revealing that simple reverse-order pruning and partial-layer fine-tuning outperform complex metrics and LoRA-based methods. One-shot pruning is as effective as iterative pruning, and the authors release pruned models that rival or surpass popular LLMs of similar size.

# Why it matters

- LLMs are resource-intensive, limiting deployment in constrained environments.
- Layer pruning offers a direct, effective way to reduce model size and computational cost.
- The study provides actionable best practices for efficient LLM compression, enabling broader accessibility and sustainability.

# Key technical takeaways

- **Reverse-order Pruning:** Pruning the last several layers yields strong results, outperforming sophisticated metrics.
- **Partial-layer Fine-tuning:** Fine-tuning only the last few layers and the LM head is superior to LoRA for post-pruning recovery.
- **One-shot vs. Iterative:** One-shot pruning matches or exceeds iterative pruning, with lower computational overhead.
- **Sensitivity Analysis:** Calibration sample count, SFT dataset choice, and pruning rate all impact pruned model performance.
- **Released Models:** Llama-3.1-6.3B-It-Alpaca and Llama-3.1-6.3B-It-Dolly outperform several community LLMs with far fewer training tokens.

# Implementation steps (for Candle or similar Rust ML frameworks)

1. **Layer Selection:** Implement reverse-order pruning (remove final N layers).
2. **Fine-tuning:** Freeze all but the last few layers and LM head; fine-tune these on a suitable SFT dataset.
3. **Pruning Strategy:** Prefer one-shot pruning over iterative approaches for efficiency.
4. **Evaluation:** Benchmark pruned models on zero-shot tasks and perplexity using standard datasets (e.g., MMLU, HellaSwag, ARC, WikiText2).
5. **Release:** Document model statistics, training cost, and provide reproducibility resources.

# Acceptance criteria

- [ ] Implements reverse-order layer pruning.
- [ ] Supports partial-layer fine-tuning (LM head + last N layers).
- [ ] Benchmarks against baseline and community models on standard datasets.
- [ ] Results are reproducible and match reported performance improvements.
- [ ] Documentation includes pruning strategy, fine-tuning details, and evaluation outcomes.# Reassessing Layer Pruning in LLMs: New Insights and Methods

**Original markdown:** [reassessinglayerpruninginllmsnewinsightsandmethods.md](../papers/markdown/reassessinglayerpruninginllmsnewinsightsandmethods.md)

**Original PDF:** [ReassessingLayerPruningInLLMsNewInsightsAndMethods.pdf](../../../../Desktop/books%20and%20courses/Machine%20Learning/ReassessingLayerPruningInLLMsNewInsightsAndMethods.pdf)

---

## TL;DR

# Reassessing Layer Pruning in LLMs: New Insights and Methods

**Links:**  
[Original PDF](c:/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/ReassessingLayerPruningInLLMsNewInsightsAndMethods.pdf)  
[Source Markdown](../papers/markdown/reassessinglayerpruninginllmsnewinsightsandmethods.md)

---

## TL;DR

This paper benchmarks layer pruning in large language models (LLMs), revealing that simple reverse-order pruning (removing the last layers) and partial-layer fine-tuning outperform complex metrics and LoRA-based methods. Iterative pruning offers no benefit over one-shot pruning. The authors release pruned Llama-3.1 models that outperform popular LLMs of similar size.

## Why it matters

LLMs are resource-intensive, making deployment on constrained devices challenging. Layer pruning is a direct way to reduce model size and computational cost. This work provides practical, empirically validated guidelines for effective layer pruning, challenging assumptions about the necessity of complex selection metrics and popular fine-tuning methods like LoRA.

## Key technical takeaways

- **Reverse-Order Pruning:**
  - Pruning the final 25% of layers yields strong performance, outperforming sophisticated metrics.
- **Partial-Layer Fine-Tuning:**
  - Fine-tuning only the last few layers and the language model head is more effective than LoRA for post-pruning recovery.
- **Pruning Strategy:**
  - One-shot pruning is as good as or better than iterative pruning, saving training time and resources.
- **Sensitivity Analyses:**
  - Performance depends on calibration sample size, choice of SFT dataset, and pruning rate.
- **Empirical Results:**
  - Pruned Llama-3.1-6.3B models outperform ChatGLM2-6B, Vicuna-7B-v1.5, Qwen1.5-7B, and Baichuan2-7B.
  - Released weights and code for reproducibility.

## Implementation steps (Candle/Rust context)

1. **Layer Pruning:**

- Remove the last N layers (e.g., final 25%) from the LLM architecture.

2. **Fine-Tuning:**

- Freeze all but the last few layers and the LM head; fine-tune only these components on a suitable SFT dataset.
- Avoid LoRA for post-pruning recovery; use partial-layer fine-tuning instead.

3. **Evaluation:**

- Benchmark pruned models against baselines and popular LLMs of similar size.
- Analyze sensitivity to calibration sample size, SFT dataset, and pruning rate.

4. **Release:**

- Provide model weights and code for reproducibility.

## Acceptance criteria

- Implementation follows the recommended pruning and fine-tuning strategies.
- Evaluation demonstrates competitive or superior performance to baselines and popular LLMs.
- Code is modular, reproducible, and links to both the original PDF and markdown source.

## Why it matters

LLMs are resource-intensive, making deployment on constrained devices challenging. Layer pruning offers a direct way to reduce model size and computational cost. This work provides practical, empirically validated best practices for pruning and fine-tuning LLMs, enabling efficient model deployment without significant loss in performance. The findings challenge common assumptions about pruning strategies and fine-tuning methods, guiding future research and real-world applications.

## Key technical takeaways

- Reverse-order pruning (removing last layers) is simple and highly effective, outperforming complex selection metrics.
- Partial-layer fine-tuning (updating only the last few layers and the LM head) restores pruned model performance better than LoRA/QLoRA, with faster training and comparable GPU usage.
- One-shot pruning is as effective as iterative pruning, with less computational overhead and no performance gain from iteration.
- The choice of fine-tuning dataset and number of calibration samples significantly affects pruned model performance.
- Pruned models (Llama-3.1-6.3B-It-Alpaca/Dolly) outperform several popular community LLMs of similar size, using orders of magnitude fewer training tokens.
- Code and model weights are released for reproducibility.

## Implementation steps (for Candle or similar ML library)

1. **Layer Selection:** Use reverse-order metric to select layers for pruning (remove last N layers).
2. **Pruning:** Prune selected layers in one shot (not iteratively).
3. **Fine-Tuning:** Freeze all but the last few layers and LM head; fine-tune only these layers using a suitable SFT dataset (e.g., Alpaca-cleaned, Dolly-15k).
4. **Evaluation:** Benchmark pruned models on zero-shot reasoning tasks and compare against baselines.
5. **Sensitivity Analysis:** Test different calibration sample sizes and fine-tuning datasets for optimal results.
6. **Documentation:** Provide code, model weights, and reproducibility details.

## Acceptance criteria

- Reverse-order pruning and partial-layer fine-tuning are implemented and tested.
- Pruned models are evaluated on at least two reasoning benchmarks and compared to baselines.
- Results show pruned models match or exceed performance of similarly sized community models with fewer training tokens.
- Training time, GPU usage, and parameter counts are documented.
- Code and model weights are released for reproducibility.
- Documentation includes pruning steps, fine-tuning strategy, and evaluation results.
