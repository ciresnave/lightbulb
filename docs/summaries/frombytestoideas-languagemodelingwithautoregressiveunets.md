# FromBytesToIdeas-LanguageModelingWithAutoregressiveUNets

Full PDF: [Local PDF](file:///c%3A/Users/cires/OneDrive/Desktop/books%20and%20courses/Machine%20Learning/FromBytesToIdeas-LanguageModelingWithAutoregressiveUNets.pdf)

Markdown: ../papers/markdown/frombytestoideas-languagemodelingwithautoregressiveunets.md

## TL;DR

Autoregressive U-Nets (AU-Nets) learn multi-scale tokenization inside the model by progressively pooling raw bytes into larger units; deeper stages predict further ahead and capture semantic structure, enabling a single model to handle variable granularities and potentially improve long-range prediction.

## Why it matters

- Moving tokenization inside the model removes brittle pre-tokenization choices and enables multi-scale context — an interesting direction for Lightbulb to explore for robust handling of diverse input types and long contexts.

## Key technical takeaways

1. AU-Net builds contracting/expanding stages that compress sequence length (bytes→words→multiword tokens) and expand predictions with skip connections, giving multi-scale representations.
2. Deeper stages focus on broader semantics (predict further ahead) while shallower stages refine local detail; careful compute budgeting is required to match baselines.
3. When tuned, shallow hierarchies match strong BPE baselines; deeper hierarchies show potential gains and cross-lingual robustness.

## Implementation steps for Lightbulb

- Prototype a small AU-Net-style module (2–3 stages) on CPU to test multiscale pooling and decoding behavior on short multilingual samples.
- Compare AU-Net tokenization-internal approach vs our existing tokenizer+loader stack using a small dataset; measure model-size/compute tradeoffs.

## Acceptance criteria

- A minimal AU-Net prototype runs locally and reproduces the stagewise pooling behavior in a notebook.
- Benchmarks show either parity with current tokenization for small compute budgets, or clear metrics documenting tradeoffs.
