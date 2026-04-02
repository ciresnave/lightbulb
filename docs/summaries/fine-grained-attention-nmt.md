# Fine-Grained Attention for Neural Machine Translation — summary

TL;DR

- Splits attention into finer units (e.g., subword/character or structured segments) to improve alignment and reduce over/under-translation; introduces sparsity and locality that can cut compute.

Why it matters for lightbulb

- Suggests head- or token-level sparsity knobs and locality bias we can toggle at inference for speed without large quality loss, especially on long sentences.

Key points

- Fine-grained (subword/char) attention stabilizes alignment and reduces mode collapse.
- Local windows and hierarchical attention keep complexity near-linear for many sequences.
- Head specialization: some heads can be pruned or constrained to local windows.

Actionable next steps

- Add an optional local-window mask for a subset of heads (e.g., last k tokens) during decode.
- Provide a per-head locality configuration (dense vs local) via CLI for experimentation.

Acceptance criteria

- On long prompts, enabling local windows for 25–50% of heads reduces attention time >15% with negligible perplexity delta (<1%).
