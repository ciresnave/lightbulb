# Dynamic Neural Networks — survey summary

TL;DR

- Dynamically modulating computation (depth/width/routing/resolution) at inference reduces average cost while preserving accuracy, using confidence/uncertainty signals and gating. Applicable to transformers via early exit, adaptive layer skipping, and expert routing.

Why it matters for lightbulb

- Unifies policies (early exit, CoLa, SALM) under a common “dynamic compute” scheduler interface with shared signals and logging.

Key points

- Dimensions: dynamic depth, width, routing (MoE), resolution/chunking; training vs post-hoc calibration.
- Signals: entropy, margin, variance, energy scores; learned gates vs rule-based thresholds.
- Trade-offs: stability, calibration drift across domains, hardware utilization.

Actionable next steps

- Define a Policy trait with pluggable signals and actions (exit/skip/repeat/route); log per-token actions and confidences.
- Acceptance: ≥20–30% average compute reduction with ≤2% accuracy loss on a mixed eval; provide ablation by policy.
