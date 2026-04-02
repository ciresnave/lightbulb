# Prompting LLMs with the Socratic Method — Summary (Chang, 2023)

Paper: Edward Y. Chang, “Prompting Large Language Models With the Socratic Method,” arXiv:2303.08769v2

Key ideas

- Socratic strategies (definition, elenchus, dialectic, maieutics, counterfactuals) can structure multi-turn prompting to improve accuracy and creativity.
- Prompt ensembles (paraphrases that ask the same thing) and majority voting improve consistency over single prompts.
- “Warm intent priming”: conveying task intent/context before dialogue often boosts results — effectively a reusable system prompt/prefix.
- Sequential prompting (“one-by-one”) tends to yield more detailed, coherent outputs vs a single mega-prompt for complex documents.
- Self-checking: ask for evidence types and rate credibility; surface counterarguments and reevaluate (elenchus/dialectic).

Actionable for Lightbulb

- Prefix KV caching: Treat stable system prompts or instruction prefixes as cacheable KV prefixes shared across requests. Hash prefixes and reuse prefill KV to cut TTFT and total tokens processed.
- Prompt program executor: A small runner that executes multi-turn prompt graphs (sequences/branches with simple conditionals), enabling Socratic templates without app-level glue.
- Ensemble prompting: Allow N paraphrases with majority vote or score-weighted selection; make it optional and measurable (trade tokens for reliability).
- Counterfactual/exploration mode: Optional creative-writing helpers that generate “what-if” variants via a small prompt graph.

Testing/metrics hooks

- Compare sequential vs single-shot outcomes on a tiny QA/RC set; measure token cost and quality proxies (e.g., self-consistency rate).
- Measure TTFT savings from prefix KV reuse with repeated system prompts under mixed workloads.

Relevance to performance

- Prefix KV reuse directly reduces compute and latency for repeated system prompts and common prefix templates.
- Prompt programs benefit from continuous batching and scheduler support for per-request state across steps.

Citation

- E. Y. Chang. Prompting Large Language Models With the Socratic Method. arXiv:2303.08769, 2023.
