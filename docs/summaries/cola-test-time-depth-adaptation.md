# Skip a Layer or Loop it? (CoLa) — summary

TL;DR

- Test-time MCTS selects per-input chains that can skip or repeat pretrained layers, improving accuracy and/or reducing depth without training. Joint skip+recurrence outperforms skip-only.

Why it matters for lightbulb

- Suggests a runtime policy layer atop the scheduler to allocate compute adaptively across tokens/inputs, potentially cutting average depth and mitigating overthinking.

Key findings

- Many wrong-to-correct (W→C) fixes use shallower paths; layer usage is task- and size-dependent.
- MCTS explores action sequences (skip/repeat); useful even without fine-tuning.

Actionable next steps

- Prototype an offline harness that replays a decode and simulates skip/repeat decisions with cached activations; log accuracy vs depth curves.
- Acceptance: ≥10–20% mean depth reduction with neutral or improved accuracy on a small reasoning subset; document tasks where recurrence helps.
