# Unsupervised Elicitation of Language Models — summary

TL;DR

- Without labeled data, we can elicit latent capabilities via prompt search, self-improvement loops, and unsupervised objectives, often rivaling small supervised fine-tunes.

Why it matters for lightbulb

- Offers an offline, CPU-friendly path to domain adaptation: prompt/program search plus cached evaluations, no gradient steps required.

Key points

- Techniques include self-consistency sampling, majority voting, prompt evolution, curriculum discovery, and heuristic verifiers.
- A small evaluation set guides prompt/program search; no labels needed if heuristics or constraints exist (format checks, unit tests, regex).

Actionable next steps

- Provide a prompt/program search utility with pluggable evaluators and budget constraints integrated into the CLI.
- Cache model outputs and scores to avoid recomputation in offline mode.

Acceptance criteria

- Demonstrate ≥10% improvement on a domain mini-benchmark using unsupervised prompt evolution vs. baseline prompt at fixed token budget.
