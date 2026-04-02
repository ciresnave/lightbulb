# Mamba: Linear-Time Sequence Modeling with Selective SSMs — summary

TL;DR

- Mamba replaces attention with a selective state space model (SSM), achieving linear-time inference and strong performance on long sequences via input-dependent gating and efficient convolutional kernels.

Why it matters for lightbulb

- Provides an alternative or complementary mixer for hybrid schedules (e.g., interleave SSM layers with periodic full attention) to bound KV while retaining recall through anchors.

Key points

- Selective scanning with input-dependent parameters; hardware-friendly kernels.
- Strong scaling on long contexts; competitive with Transformers in several domains.
- Still benefits from occasional global mixing (anchors) for retrieval tasks.

Actionable next steps

- Track Candle support for Mamba-like layers; prototype hybrid schedules that treat SSM layers as “linear” mixers in our policy.
- Acceptance: replicate hybrid-attn targets (≥4× KV reduction, small ppl delta) with SSM mixers as the linear component.
