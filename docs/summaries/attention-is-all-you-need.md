# Attention Is All You Need — summary

TL;DR

- Transformer uses self-attention with positional encoding and feed-forward layers, enabling parallel prefill and strong long-range modeling; decoding is autoregressive.

Why it matters for lightbulb

- Sets baseline attention semantics, masking, and residual/LayerNorm ordering that our runtime must maintain; guides hybrid attention schedules and KV cache layout.

Key points

- Multi-head scaled dot-product attention with causal masking for decoder-only LMs.
- Positional encodings (sinusoidal or learned) resolve order; residual connections + LayerNorm stabilize training/inference.
- Complexity O(L^2) memory/time motivates hybrid/efficient attention and paging.

Actionable next steps

- Verify causal mask integrity in prefill and decode; add a test to ensure no future-token attention.
- Confirm KV cache per-head layout matches model config; add a small head-dim permutation test.
- Instrument attention time/memory to benchmark against hybrid linear attention baselines later.

Acceptance criteria

- Attention mask test PASS; KV layout test PASS; attention timing recorded in a baseline JSON artifact.
