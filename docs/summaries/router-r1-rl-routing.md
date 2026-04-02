# Router-R1: Teaching LLMs Multi-Round Routing via Reinforcement Learning — summary

TL;DR

- Uses RL to train routers that make multi-round routing decisions (e.g., to tools or experts), balancing quality and latency/compute. Demonstrates improved routing quality over static heuristics.

Why it matters for lightbulb

- Extends scheduling with learned routing policies (e.g., tool/MoE/chain-of-modules) and provides a path to prioritize latency vs quality dynamically.

Key points

- Reward shaping for latency/quality; exploration and stability; multi-round sequential decisions.
- Interfaces with MoE-style routing and tool-augmented inference.

Actionable next steps

- Keep routing policy interfaces pluggable; log features needed for RL fine-tuning later.
- Acceptance: doc-only for now; define metrics and traces needed to train/evaluate a router.
