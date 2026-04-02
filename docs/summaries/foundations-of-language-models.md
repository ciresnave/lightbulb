# Foundations of Language Models — summary

TL;DR

- LMs are next-token predictors trained with maximum likelihood over tokenized text; scale, data quality/coverage, and inductive biases (positional encoding, normalization, context windows) determine capability and compute-efficiency.

Why it matters for lightbulb

- Clarifies core invariants the engine must preserve at inference time: tokenizer-normalization parity, EOS/BOS handling, causal masking, and stable KV cache semantics across prefill/decoding.

Key points

- Training objective: autoregressive cross-entropy with causal mask; inference mirrors teacher-forced prefill then token-by-token decode.
- Tokenization: normalization, special tokens, and byte-fallback can change effective context length and log-prob calibration.
- Scaling and data curricula: larger models benefit from longer contexts and better dedup; inference must support long-context efficiently.
- Inductive biases: positional encodings and attention variants affect extrapolation and stability.

Actionable next steps

- Add a tokenizer parity check: assert that `tokenizer.json` special tokens and normalization are honored (unit test against a known prompt/EOS).
- Harden BOS/EOS/UNK edge cases in `local-llama-gen` (acceptance: golden tests that match expected token sequences and stop at EOS).
- Ensure prefill+decode parity: a test that compares full teacher-forced pass vs prefill+stepwise decode for identical logits on step 1 (within tolerance).

Acceptance criteria

- Unit tests: tokenizer parity, EOS stopping, and prefill-vs-decode logits parity all PASS on CPU-only CI.
