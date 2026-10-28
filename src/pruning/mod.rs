// SPDX-License-Identifier: MIT OR Apache-2.0
//! Model pruning utilities (M5)
//!
//! `PruningPolicy`/`StructuredPattern`/`PruningMask`/`WandaScorer`/
//! `TailPruner`/`PruningLoader`/`PruningValidator`/`gguf_application`
//! (the whole Wanda/tail-pruning algorithm and its GGUF-application driver)
//! removed 2026-10-22 (candlelight removal, CireSnave: "Remove all
//! dependencies on Candle and Candlelight..."): every one of them was
//! candlelight-`Tensor`-typed and had ZERO external callers anywhere in
//! src/tests/examples even before removal (`apply_manifest_to_gguf`'s only
//! two references were its own file and a bare re-export; `PruningManifest`/
//! `PruningMask` were read nowhere outside this now-deleted code). See
//! `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`.
//!
//! `name_mapping` (architecture detection for tensor-name mapping) survives
//! unchanged — it was always candlelight-free and has real callers
//! (`src/tools/mod.rs`, `src/cache/parallel_cache_builder.rs`).

pub mod name_mapping;
