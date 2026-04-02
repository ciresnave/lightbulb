# ShorterBetter: Guiding Reasoning Models to Find Optimal Inference Length for Efficient Reasoning

**Links:**  
[Original PDF](c:/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/ShorterBetter-GuidingReasoningModelsToFindOptimalInferenceLengthForEfficientReasoning.pdf)  
[Source Markdown](../papers/markdown/shorterbetter-guidingreasoningmodelstofindoptimalinferencelengthforefficientreasoning.md)

---

## TL;DR

ShorterBetter is a reinforcement learning method that teaches reasoning models to find their own optimal Chain-of-Thought (CoT) lengths, reducing output length by 50%-80% while maintaining accuracy. The method uses the Sample Optimal Length (SOL)—the shortest correct response among multiple generations—as a dynamic reward signal for efficient reasoning.

## Why it matters

Extended reasoning traces in LLMs can lead to overthinking, inefficiency, and redundancy. ShorterBetter enables models to autonomously learn concise, targeted reasoning strategies, improving computational efficiency and output quality without sacrificing correctness.

## Key technical takeaways

- **Sample Optimal Length (SOL):**
  - SOL is used as a reward signal, guiding models to produce the shortest correct reasoning trace for each problem.
- **Reinforcement Learning:**
  - Models are trained to dynamically identify and steer toward efficient reasoning lengths during inference.
- **Length Reduction:**
  - Achieves substantial reductions in output length across in-domain and out-of-domain tasks, maintaining accuracy.
- **Reasoning Trace Analysis:**
  - ShorterBetter refines reasoning structure, reducing verbosity, repetition, and unnecessary exploration.
- **Generalization:**
  - Method is effective across different model sizes and reasoning benchmarks.

## Implementation steps (Candle/Rust context)

1. **Reward Signal Design:**
   - Implement logic to compute SOL for each problem and use it as a reward in RL training.
2. **RL Training:**
   - Train reasoning models to minimize output length while preserving correctness.
3. **Trace Analysis:**
   - Analyze and refine reasoning traces to reduce redundancy and improve structure.
4. **Evaluation:**
   - Benchmark length reduction and accuracy on diverse reasoning tasks.
5. **Generalization Testing:**
   - Test method across different model sizes and domains.

## Acceptance criteria

- Implementation uses SOL as a dynamic reward for efficient reasoning.
- RL training achieves substantial length reduction without accuracy loss.
- Evaluation demonstrates improved reasoning efficiency and structure.
- Code is modular, reproducible, and links to both the original PDF and markdown source.
