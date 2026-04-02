# ExploringTheLimitOfOutcomeRewardForLearningMathematicalReasoning

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/ExploringTheLimitOfOutcomeRewardForLearningMathematicalReasoning.pdf)

Markdown: ../papers/markdown/exploringthelimitofoutcomerewardforlearningmathematicalreasoning.md

## TL;DR

OREAL: an outcome-reward RL framework for math reasoning that shows behavior-cloning on BoN-positive trajectories plus token-level reward shaping achieves state-of-the-art RL gains (7B→94% on MATH-500), and provides theoretical conditions for gradient-consistent reward shaping in binary-feedback environments.

## Why it matters

- Demonstrates practical RL techniques (BoN behavior cloning, token-level rewards, negative-sample reshaping) that yield major gains without proprietary-scale models — useful guidance for Lightbulb experiments on fine-tuning and RL-based improvement of reasoning.

## Key technical takeaways

1. Behavior cloning on positive trajectories from best-of-N sampling suffices to learn the KL-regularized optimal policy under binary outcome rewards.
2. Negative samples require reward reshaping to maintain gradient consistency between positive and negative examples.
3. Token-level reward models (important-token sampling) alleviate sparse-reward issues in long chain-of-thought trajectories and stabilize learning.
4. Initial policy quality and training-query selection strongly affect RL outcomes.

## Implementation steps for Lightbulb

- Prototype a BoN sampling harness (n=8..64) for our reasoning prompts and implement a small behavior-cloning fine-tune on the top-k positive traces.
- Add a minimal token-reward head for training experiments (token-level reward regressors) so we can do importance sampling on generated chains.
- Run small-scale OREAL-style experiments on a 7B open model to validate pass@1 improvements on a math subset before scaling.

## Acceptance criteria

- Reproduce a measurable RL improvement (e.g., >10% pass@1) on an internal math-like benchmark with a 7B model in local tests.
- Training harness supports BoN sampling and behavior cloning on selected trajectories; token-reward model integrated as an optional loss term.
- Experiments are tracked and reproducible with configuration files and seed-controlled runs.
