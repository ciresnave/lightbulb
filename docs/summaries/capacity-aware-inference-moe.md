# Capacity-Aware Inference: Mitigating the Straggler in MoE — Summary

Source: Capacity-Aware Inference: Mitigating the Straggler (2024)

TL;DR

- MoE decoding suffers from “stragglers” when some experts receive too many tokens. Introduce capacity-aware token dropping controlled by gating scores and an expanded candidate set to reduce tail latency.
- Reported speedups up to ~1.85× on Mixtral-like models with minimal token drops (capacity factor γ≈1.5); multimodal variant drops image tokens first when needed.

Key ideas

- Token Drop: For each expert, enforce a capacity proportional to average load; drop the lowest-gated tokens when capacity is exceeded.
- Expanded Drop: Add a few “local” experts (m) to the candidate pool to absorb spillover without overloading top experts.
- Maintain accuracy by using gating scores as drop heuristics and keeping small drop ratios.

Actionable for Lightbulb

- Scheduler option for MoE models: capacity-aware routing.
  - Knobs: capacity factor γ (e.g., 1.2–1.8), local expansion m (e.g., 1–2), drop policy (gating-score thresholding), and modality-aware priorities.
  - Batching awareness: group tokens by expert to reduce kernel launches and mitigate tail latency.

Acceptance criteria

- On a small MoE demo, enabling capacity-aware routing reduces 95th percentile step latency by ≥30% with ≤2% token drops over a mixed-load trace.
- Functional parity against baseline routing with the feature disabled; deterministic tests on synthetic gating distributions.

Risks/notes

- Requires model/router introspection to identify experts and gating scores via Candle APIs.
- Need careful logging to validate fairness and drop ratios.

Citation

- Capacity-Aware Inference: Mitigating the Straggler (2024).
