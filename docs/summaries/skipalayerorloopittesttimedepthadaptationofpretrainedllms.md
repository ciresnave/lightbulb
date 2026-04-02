# Skip a Layer or Loop it? Test-Time Depth Adaptation of Pretrained LLMs

**Links:**  
[Original PDF](c:/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/SkipALayerOrLoopItTestTimeDepthAdaptationOfPretrainedLLMs.pdf)  
[Source Markdown](../papers/markdown/skipalayerorloopittesttimedepthadaptationofpretrainedllms.md)

---

## TL;DR

This paper introduces a method for test-time depth adaptation in pretrained LLMs, allowing layers to be skipped, repeated, or rearranged for each input without any finetuning. Using Monte Carlo Tree Search (MCTS), the optimal chain-of-layers (CoLa) is found per sample, improving inference efficiency and accuracy by customizing model depth dynamically.

## Why it matters

Standard LLMs use a fixed architecture for all inputs, which may be inefficient or suboptimal for varying task complexities. Dynamic depth adaptation enables fast-slow thinking, reduces redundancy, and unlocks generalization power, paving the way for more flexible and efficient inference in large models.

## Key technical takeaways

- **Dynamic Architecture:**
  - Layers can be skipped, repeated, or reordered at test time, creating a custom CoLa for each input.
- **MCTS Protocol:**
  - Monte Carlo Tree Search efficiently explores the space of possible layer compositions to optimize depth and accuracy.
- **Efficiency Gains:**
  - For most samples, shallower or more accurate CoLa can be found, reducing inference cost and correcting errors.
- **Generalization:**
  - The approach works for both pretrained and instruction-finetuned LLMs across diverse reasoning tasks.
- **Layer Utilization Analysis:**
  - In-depth analysis reveals redundancy and alignment of layers to task difficulty, informing future model design.

## Implementation steps (Candle/Rust context)

1. **Layer Composition Logic:**
   - Implement mechanisms to skip, repeat, and reorder layers at inference time.
2. **MCTS Search:**
   - Integrate Monte Carlo Tree Search to find optimal CoLa for each input.
3. **Evaluation:**
   - Benchmark efficiency and accuracy improvements on reasoning tasks.
4. **Layer Analysis:**
   - Analyze layer utilization and impact on performance across tasks.
5. **Generalization Testing:**
   - Test method on different model sizes and domains.

## Acceptance criteria

- Implementation enables dynamic layer composition at test time.
- MCTS search finds optimal CoLa for each input, improving efficiency and accuracy.
- Evaluation demonstrates reduced inference cost and error correction.
- Code is modular, reproducible, and links to both the original PDF and markdown source.
