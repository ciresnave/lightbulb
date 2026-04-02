# A Comprehensive Survey of Deep Research Systems — systems survey

TL;DR

An extensive survey covering system-level methodologies, tooling, and benchmarks for large-scale deep learning research. Useful for design decisions around reproducibility, benchmarking, and tooling for `lightbulb`.

Why it matters

- Provides best practices for reproducible benchmarks, dataset/versioning, and system-level metrics which directly inform `lightbulb`'s benchmarking and test harness design.
- Useful as a reference for designing reproducible benchmarks, experiment tracking, and systems-level tooling for model development at scale.

Key takeaways

- Best practices for reproducibility and tooling for large research projects.

Actions

- Extract recommended benchmarking methodology and adapt a minimal reproducible benchmark for CPU-only environments.
- Add CI benchmark targets and data collection hooks to `tests/benchmarks/`.
- Add a `docs/benchmarks/` folder with reproducible harnesses for the roadmap acceptance tests.
- Capture deterministic seed and environment requirements in `docs/Local model setup.md`.
- Extract relevant best practices into `docs/dev/best-practices.md` or the roadmap.
- Add a checklist for reproducible microbenchmarks in `docs/bench`.

Acceptance criteria

- A short benchmark checklist is added to `docs/benchmarks.md` and one CPU benchmark is implemented and documented.
- A minimal benchmark harness for the verifier integration that reproduces results on CI (same seeds, dataset, and metrics).
- Checklist integrated into CI and used for key benchmarks.
