kind: MOD-003 reach-throughs removed
---
### Changed
CodeRipper MOD-003 (reaching through a sibling module's inner modules for items its entry point re-exports) went from 7 findings (17 reach-throughs) to 0 (coderipper 0.43.54, `check MOD-003`). Seven files now import `InferenceJob`, `ModelRunner`, `SpeculativeConfig`, `SlotPoolRequest`, `H2OPolicy`, `SegmentedEvictionPolicy` and the span types from `crate::engine` / `crate::cache` instead of `engine::model_runner::`, `engine::speculative::`, `engine::slot_pool::`, `cache::cache_span::`. Paths only; no behaviour change. Verified: `cargo check --all-targets` compiles, `scripts/clippy_gate.py` exits 0.
