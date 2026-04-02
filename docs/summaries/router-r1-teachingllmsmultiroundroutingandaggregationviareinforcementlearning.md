# Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning

**Links:**  
[Original PDF](c:/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/Router-R1-TeachingLLMsMultiRoundRoutingAndAggregationViaReinforcementLearning.pdf)  
[Source Markdown](../papers/markdown/router-r1-teachingllmsmultiroundroutingandaggregationviareinforcementlearning.md)

---

## TL;DR

Router-R1 is a reinforcement learning-based framework that enables large language model (LLM) routers to perform multi-round routing and aggregation, leveraging the strengths of multiple LLMs for complex tasks. The router itself is instantiated as an LLM, interleaving internal reasoning and dynamic model selection, and optimizing performance-cost trade-offs.

## Why it matters

Existing LLM routers typically assign queries to a single model in isolation, limiting their ability to solve complex tasks that require coordinated interactions among multiple models. Router-R1 demonstrates that multi-round routing and aggregation, guided by reinforcement learning, can significantly improve answer quality, generalization, and cost management in multi-LLM systems.

## Key technical takeaways

- **Sequential Decision Process:**
  - Routing and aggregation are formulated as a sequential decision-making problem, alternating between internal reasoning ("think") and model invocation ("route").
- **Router as LLM:**
  - The router itself is a capable LLM, enabling flexible interleaving of reasoning and model selection.
- **Rule-Based Reward Function:**
  - Format, outcome, and cost rewards guide RL training, balancing performance and cost.
- **Generalization:**
  - Router-R1 conditions on simple model descriptors (pricing, latency, performance), enabling strong generalization to unseen models.
- **Experimental Results:**
  - Outperforms strong baselines on seven QA benchmarks, achieving superior performance and robust cost management.

## Implementation steps (Candle/Rust context)

1. **Router Model Design:**
   - Instantiate the router as an LLM capable of both reasoning and dynamic model selection.
2. **Routing Logic:**
   - Implement sequential decision-making, alternating between "think" and "route" actions.
3. **Reward Function:**
   - Design rule-based rewards for output format, correctness, and cost.
4. **Training:**
   - Use RL to optimize the router's policy for multi-round routing and aggregation.
5. **Evaluation:**
   - Benchmark on multi-hop QA tasks, measuring performance, generalization, and cost trade-offs.

## Acceptance criteria

- Implementation uses RL-based multi-round routing and aggregation.
- Router is instantiated as an LLM, capable of reasoning and model selection.
- Reward function balances performance and cost.
- Evaluation shows improved answer quality, generalization, and cost management.
- Code is modular, reproducible, and links to both the original PDF and markdown source.
