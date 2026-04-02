# Survey on LLM-based Autonomous Agents — summary

TL;DR

- Covers planning, memory, tool use, multi-agent coordination, and evaluation. Emphasizes modular architectures and robust memory/tool abstractions.

Why it matters for lightbulb

- Informs prompt program executor design (planning, tools), memory interfaces, and scheduler orchestration for multi-step/multi-agent workflows.

Actionable next steps

- Stabilize tool/memory APIs with typed IO; add multi-agent orchestration primitives (queues, roles, shared memory keys).
- Acceptance: run a small multi-agent script with clear metrics (latency, tool latency, success rate).
