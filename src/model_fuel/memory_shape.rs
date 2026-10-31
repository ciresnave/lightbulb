// SPDX-License-Identifier: MIT OR Apache-2.0
//! Memory shape of a served model, and the budget arithmetic over it (F10 L4).
//!
//! Lightbulb does not guess layer counts. The loader supplies a
//! [`MemoryShape`]: bytes of one KV block and bytes of fixed per-session
//! state (the recurrent layers of a hybrid such as Qwen3.5, which do not grow
//! with context). Everything here is a pure function of those figures, so a
//! hybrid's budget is testable without a model.
//!
//! Not yet wired into the engine: the hybrid figures come from Fuel's hybrid
//! `PagedDecodeModel` (Fuel ROADMAP rows 117/118). Until then non-hybrids
//! keep today's block-count budget and hybrids keep the qwen35 refusal.

use super::preflight::KvShape;
use super::scheduled_llama_engine::KV_BLOCK_SIZE;

/// What one session of a model costs in device memory, besides its weights.
pub trait MemoryShape {
    /// Bytes of one KV block (`KV_BLOCK_SIZE` token slots, K and V, in every
    /// layer that holds KV).
    fn bytes_per_kv_block(&self) -> u64;
    /// Bytes each session holds regardless of context length (0 for a model
    /// whose every layer is attention).
    fn fixed_state_bytes_per_session(&self) -> u64;
    /// Whether some layers hold recurrent state instead of KV.
    fn is_hybrid(&self) -> bool;
}

impl MemoryShape for KvShape {
    fn bytes_per_kv_block(&self) -> u64 {
        self.bytes_per_token().saturating_mul(KV_BLOCK_SIZE as u64)
    }

    fn fixed_state_bytes_per_session(&self) -> u64 {
        0
    }

    fn is_hybrid(&self) -> bool {
        false
    }
}

/// KV blocks one session of up to `context_length` tokens occupies.
fn blocks_per_session(context_length: usize) -> u64 {
    (context_length as u64).div_ceil(KV_BLOCK_SIZE as u64)
}

/// Bytes one session costs at `context_length`: its KV blocks plus its fixed
/// state.
pub fn bytes_per_session(shape: &dyn MemoryShape, context_length: usize) -> u64 {
    blocks_per_session(context_length)
        .saturating_mul(shape.bytes_per_kv_block())
        .saturating_add(shape.fixed_state_bytes_per_session())
}

/// Device bytes needed for `weights_bytes` plus `max_batch_size` sessions of
/// up to `context_length` tokens. A batch of 0 counts as 1, as the engine
/// clamps it.
pub fn bytes_needed(
    shape: &dyn MemoryShape,
    weights_bytes: u64,
    max_batch_size: usize,
    context_length: usize,
) -> u64 {
    let sessions = max_batch_size.max(1) as u64;
    bytes_per_session(shape, context_length)
        .saturating_mul(sessions)
        .saturating_add(weights_bytes)
}

/// Most sessions of up to `context_length` tokens that fit in `capacity_bytes`
/// beside the weights; 0 when the weights alone do not fit.
pub fn max_sessions_that_fit(
    shape: &dyn MemoryShape,
    capacity_bytes: u64,
    weights_bytes: u64,
    context_length: usize,
) -> u64 {
    let per_session = bytes_per_session(shape, context_length).max(1);
    capacity_bytes.saturating_sub(weights_bytes) / per_session
}

/// Whether prefix sharing runs: only when the operator asked for it and the
/// model is not hybrid. Splicing cached KV blocks cannot restore a recurrent
/// layer's state at the prefix boundary, so on a hybrid it would give wrong
/// output with no error (F10 L5).
pub fn effective_prefix_sharing(requested: bool, hybrid: bool) -> bool {
    requested && !hybrid
}

/// Whether the GGUF at `path` declares a hybrid architecture (`qwen35`).
/// `false` for anything else, including a file that cannot be read: the
/// loader reports that itself.
pub fn gguf_is_hybrid(path: &std::path::Path) -> bool {
    use crate::gguf::{Content, GgufArchitecture, detect_architecture};
    path.is_file()
        && Content::read(path)
            .ok()
            .and_then(|c| detect_architecture(c.metadata()).ok())
            .is_some_and(|a| a == GgufArchitecture::Qwen35)
}

/// [`effective_prefix_sharing`] for the model at `model_path`, warning when an
/// operator's request is overridden.
pub fn resolve_prefix_sharing(requested: bool, model_path: &std::path::Path) -> bool {
    let hybrid = requested && gguf_is_hybrid(model_path);
    if hybrid {
        tracing::warn!(
            "prefix sharing requested but disabled: {} is a hybrid model (recurrent layers              hold state a spliced KV prefix cannot restore)",
            model_path.display()
        );
    }
    effective_prefix_sharing(requested, hybrid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_fuel::preflight::kv_pool_bytes;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    /// Qwen3.5 27B from the F10 plan: 16 of 64 layers hold KV (4 KV heads,
    /// head_dim 256, f32), the other 48 hold 3 MiB of f32 state each.
    struct Qwen27b;

    const KV_LAYERS: u64 = 16;
    const STATE_PER_LAYER: u64 = 48 * 128 * 128 * 4;
    const STATE_LAYERS: u64 = 48;

    impl MemoryShape for Qwen27b {
        fn bytes_per_kv_block(&self) -> u64 {
            KV_LAYERS * 2 * 4 * 256 * 4 * KV_BLOCK_SIZE as u64
        }
        fn fixed_state_bytes_per_session(&self) -> u64 {
            STATE_LAYERS * STATE_PER_LAYER
        }
        fn is_hybrid(&self) -> bool {
            true
        }
    }

    /// Same shape with the fixed-state term dropped, the mutation the drill
    /// below asks the 27B case to catch.
    struct Qwen27bNoState;

    impl MemoryShape for Qwen27bNoState {
        fn bytes_per_kv_block(&self) -> u64 {
            Qwen27b.bytes_per_kv_block()
        }
        fn fixed_state_bytes_per_session(&self) -> u64 {
            0
        }
        fn is_hybrid(&self) -> bool {
            true
        }
    }

    const Q4_K_M: u64 = 16_464_440_224; // 15.33 GiB on disk (plan note)
    const P40: u64 = 24 * GIB;

    #[test]
    fn the_27b_shape_costs_128_kib_per_token_and_144_mib_per_session() {
        assert_eq!(Qwen27b.bytes_per_kv_block(), 128 * 1024 * 16);
        assert_eq!(Qwen27b.fixed_state_bytes_per_session(), 144 * MIB);
    }

    #[test]
    fn one_27b_session_of_32k_is_4_gib_of_kv_plus_144_mib_of_state() {
        let per_session = bytes_per_session(&Qwen27b, 32_768);
        assert_eq!(per_session, 4 * GIB + 144 * MIB);
    }

    #[test]
    fn the_27b_on_one_p40_fits_two_32k_sessions_and_not_three() {
        // (24 GiB - 16,464,440,224 B) / (4 GiB + 144 MiB) = 2.04...
        assert_eq!(max_sessions_that_fit(&Qwen27b, P40, Q4_K_M, 32_768), 2);
        assert!(bytes_needed(&Qwen27b, Q4_K_M, 2, 32_768) <= P40);
        assert!(bytes_needed(&Qwen27b, Q4_K_M, 3, 32_768) > P40);
    }

    /// Mutation drill: without the fixed-state term the 27B case must give a
    /// different answer, so the term is what the test above pins. Chosen so
    /// the two shapes disagree on the number of sessions.
    #[test]
    fn dropping_the_fixed_state_term_changes_the_27b_answer() {
        let ctx = 32_768;
        // Weights + 8 GiB + 100 MiB: two KV-only sessions fit, but the
        // second session's 144 MiB of state pushes it over.
        let capacity = Q4_K_M + 2 * (4 * GIB) + 100 * MIB;
        assert_eq!(
            max_sessions_that_fit(&Qwen27bNoState, capacity, Q4_K_M, ctx),
            2
        );
        assert_eq!(max_sessions_that_fit(&Qwen27b, capacity, Q4_K_M, ctx), 1);
    }

    #[test]
    fn weights_that_do_not_fit_leave_no_sessions() {
        assert_eq!(max_sessions_that_fit(&Qwen27b, GIB, Q4_K_M, 4096), 0);
    }

    #[test]
    fn a_non_hybrid_equals_todays_kv_pool_number_exactly() {
        let shape = KvShape {
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
        };
        assert!(!shape.is_hybrid());
        assert_eq!(shape.fixed_state_bytes_per_session(), 0);
        let weights = 5 * GIB;
        for (batch, ctx) in [(1, 17), (2, 4096), (0, 16), (4, 8192)] {
            assert_eq!(
                bytes_needed(&shape, weights, batch, ctx),
                weights + kv_pool_bytes(shape, batch, ctx),
                "batch {batch} ctx {ctx}"
            );
        }
        // Literal pin: 2 sessions x 4096 tokens x 256 KiB per token.
        assert_eq!(bytes_needed(&shape, 0, 2, 4096), 2 * 4096 * 256 * 1024);
    }

    #[test]
    fn prefix_sharing_runs_only_when_asked_for_on_a_non_hybrid() {
        assert!(effective_prefix_sharing(true, false));
        assert!(!effective_prefix_sharing(true, true));
        assert!(!effective_prefix_sharing(false, false));
        assert!(!effective_prefix_sharing(false, true));
    }

    #[test]
    fn a_missing_or_non_gguf_path_is_not_hybrid() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!gguf_is_hybrid(&dir.path().join("absent.gguf")));
        assert!(!gguf_is_hybrid(dir.path()));
        assert!(resolve_prefix_sharing(
            true,
            &dir.path().join("absent.gguf")
        ));
    }
}
