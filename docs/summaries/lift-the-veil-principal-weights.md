# LIFT: The Veil for the Truth — principal weights & reasoning-focused fine-tuning

TL;DR
Paper exploring how principal weights emerge after rank-reduction in reasoning-focused supervised fine-tuning. Connects weight-space structure to improved reasoning behaviors.

Why it matters

- May provide insights into which fine-tuning interventions lead to robust reasoning features and how to detect them via weight-space diagnostics.

Actions

- Extract the proposed diagnostics and consider lightweight checks to add to our training validation suite.

Acceptance criteria

- At least one diagnostic implemented as a unit test in `tests/` that demonstrates the metric on a small synthetic model.
