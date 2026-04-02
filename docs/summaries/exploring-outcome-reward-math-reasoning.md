# Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning — summary

TL;DR

- Pure outcome rewards (correct/incorrect final answer) provide a sparse, high-variance learning signal for math reasoning; combining outcome rewards with structured guidance (verification, step scoring, or trajectory pruning) yields more reliable improvements.

Why it matters for lightbulb

- In inference, we can use a lightweight verifier to terminate early or prune unpromising branches during multi-sample decoding. In training, hook points for outcome-reward scoring enable small-scale RL-style fine-tuning without full RLHF stacks.

Key points

- Outcome-only rewards often saturate; step-level or verifier-derived feedback stabilizes learning and selection.
- Self-consistency and small breadth search help, but token cost grows; pruning via confidence/verification pays off.
- For math, reliable checking (unit tests, equation solvers) makes outcome rewards more usable.

Actionable next steps

- Add an optional “verifier hook” to the sampling loop (math mode) that halts or prunes branches failing quick checks.
- Provide a scoring callback API so users can plug in outcome-derived signals to rank n-best candidates.

Acceptance criteria

- On a small math subset, enable verifier-pruned sampling that reduces tokens ≥15% with ≤1pp accuracy loss, or improves accuracy at similar cost.

Implementation sketch

- Verifier hook: a synchronous callback invoked during sampling that receives the partial candidate (or final candidate) and returns one of {accept, reject, continue, score}.
- Sampling loop integration: call the verifier every N tokens or at candidate completion. If "reject", prune the branch; if "score", attach the score to the candidate for n-best ranking.
- Lightweight verifiers: run fast heuristics first (parentheses matching, token-level pattern checks), then run a more expensive deterministic check for final answers (e.g., evaluate arithmetic expression or run a small symbolic checker).

Tiny pseudo-contract (for lightbulb API)

- inputs: Candidate sequence (token ids or string), context metadata (prompt id, sampling temperature, generation step).
- outputs: Enum {Accept, Reject, Continue, Score(f32)} plus optional diagnostic string.
- error modes: verifier timeouts or exceptions must return Continue (conservative) to avoid accidental pruning.

Edge cases and pitfalls

- False negatives: overly strict verifiers can prune correct-but-unusual solutions. Provide a safe "patience" counter to allow low-scoring but diverse candidates to finish.
- Cost tradeoffs: running heavy checks every candidate increases CPU; prefer staged checks (cheap fast filters first). Measure wall-time and token cost.
- Non-determinism: when verifier uses approximate solvers, seed or deterministic mode should be available for reproducible tests.

Tests and benchmarks

- Unit test: verifier returns expected verdicts for a small suite of crafted partial and full math expressions.
- Integration test: sampling loop with verifier enabled yields >=15% token reduction on a curated math prompt set while accuracy drop is <=1 percentage point (or accuracy improves at equal token budget).
- Performance: benchmark verifier overhead per candidate and overall end-to-end latency for 1, 4, and 16 beam/consistency samples.

Related work and references

- Self-consistency and verifier-based selection (Wang et al. style self-consistency papers).
- Techniques for symbolic checking and equation evaluation used in mathematical reasoning benchmarks.

Next steps

- Implement a thin "verifier" trait in the sampling module that matches the contract above and add a fallback verifier that does fast syntactic checks + a numeric evaluator.
- Add integration tests under `tests/` that run in CPU-only CI and exercise the verifier-pruned sampling acceptance criteria.
