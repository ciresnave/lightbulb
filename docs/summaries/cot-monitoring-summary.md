# Chain-of-Thought Monitoring — summary

TL;DR

- Monitors chain quality/cost in real time using heuristics or light classifiers; can decide when to stop, intervene, or branch.

Why it matters for lightbulb

- Hooks for runtime monitors that interoperate with early-exit, interventions, and budgeted decoding.

Actionable next steps

- Add a monitor interface that receives per-step features and emits stop/intervene/continue signals.
- Acceptance: reduced average cost with neutral accuracy using a simple monitor on a small reasoning eval.
