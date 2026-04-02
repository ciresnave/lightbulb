# A path towards autonomous machines (10356)

TL;DR

This short paper outlines architectural and evaluation considerations for building progressively autonomous systems. It argues for staged capability milestones, verifiable benchmarks, and modular safety checkpoints that enable incremental autonomy.

Why it matters for lightbulb

- Emphasizes verifiable rewards and benchmark design — matches our `Verifier` requirement for math reasoning and RL-driven routing.
- Suggests modular capability gates that map cleanly to `Scheduler` policies (early-exit, staged compute) and `KvPager` safety checkpoints.

Actions

1. Add a verification checklist format to `docs/benchmarks/` describing milestone tests and pass/fail criteria.
2. Add an example `Scheduler` policy that triggers a capability gate when verifier confidence < 0.6.
3. Add an integration test that simulates staged capability progression using a small LLaMA-style model.

Acceptance criteria

- A `docs/benchmarks/autonomy-checklist.md` with 5 concrete tests is added.
- `Scheduler` supports a `capability_gate` policy and has a unit test showing transitions across gates.
