//! Policy shared between Fuel serving paths.
//!
//! Everything above the realize boundary is Lightbulb's: which token to pick,
//! when to stop, whose request runs. Fuel supplies the forward pass, the KV
//! cache and the graph. That split is the one the port is built on.
//!
//! The old serial `FuelEngineModel`/`run_jobs` path this module used to hold
//! was deleted (board item 97 cleanup, 2026-10-03): its last caller was the
//! multi-EOS serial fallback in `scheduled_llama_engine.rs`, removed once
//! fuel#307 widened `eos_ids` to `Vec<u32>` (PR #122). What remains here —
//! `effective_generation_budget` (not an intra-doc link here since this
//! module is `pub` and the function is `pub(crate)`) — is pure policy the
//! batched path (`scheduled_llama_engine.rs`) still calls directly.

/// Cap a request's generation budget to what remains of `context_length`
/// after the prompt, so `should_continue()` stops the decode loop before
/// Fuel's own cache bound would.
///
/// Returns `(effective_max_new_tokens, was_truncated)`. Pure and
/// model-independent on purpose: the caller (`step_one`) needs a loaded
/// checkpoint to reach this decision, but the decision itself does not, so it
/// is split out here to be unit-testable without one.
///
/// The KV cache is pre-allocated at `max_seq_len = (prompt + max_new_tokens +
/// 1).min(context_length)`. The `+1`/`.min()` already bound the cache
/// allocation itself, but nothing previously bounded the LOOP: if `prompt +
/// max_new_tokens + 1 > context_length`, `should_continue()` keeps driving
/// decode steps against the client's original `max_new_tokens` right past the
/// point the cache has room, and Fuel's own bound check (`cached_len + seq >
/// max_seq_len` in fuel-core's lazy realize) returns `Err` mid-decode.
/// `run_jobs`'s Complete mode then discards every token already generated and
/// reports a bare HTTP 500 for a request that was otherwise succeeding.
///
/// Truncating the budget here instead makes the loop stop naturally, so the
/// request returns the tokens that DO fit rather than erroring on the ones
/// that don't. This is deliberately not wrap-around (candlelight's `position
/// % context` reuse of cache slots): silently reusing cache positions changes
/// what the model attends to. Returning fewer tokens is honest; returning
/// tokens computed against the wrong context would not be.
pub(crate) fn effective_generation_budget(
    prompt_tokens: usize,
    requested_max_new_tokens: usize,
    context_length: usize,
) -> (usize, bool) {
    let want = prompt_tokens
        .saturating_add(requested_max_new_tokens)
        .saturating_add(1);
    if want <= context_length {
        (requested_max_new_tokens, false)
    } else {
        let truncated = context_length
            .saturating_sub(prompt_tokens)
            .saturating_sub(1);
        (truncated, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_generation_budget_untouched_when_it_fits() {
        // 10 prompt + 20 requested + 1 = 31 <= 64 context: no truncation.
        let (budget, truncated) = effective_generation_budget(10, 20, 64);
        assert_eq!(budget, 20);
        assert!(!truncated);
    }

    #[test]
    fn effective_generation_budget_truncates_on_the_reviews_concrete_case() {
        // The failure named in the review: context_length 512, a 500-token
        // prompt, default max_tokens 100. 500 + 100 + 1 = 601 > 512, so the
        // budget must shrink to what's actually left: 512 - 500 - 1 = 11.
        let (budget, truncated) = effective_generation_budget(500, 100, 512);
        assert_eq!(budget, 11);
        assert!(truncated);
    }

    #[test]
    fn effective_generation_budget_floors_at_zero_without_underflow() {
        // Prompt fills the context to one token short: no room for a
        // generation budget at all. Must floor at 0, not underflow/panic.
        let (budget, truncated) = effective_generation_budget(511, 100, 512);
        assert_eq!(budget, 0);
        assert!(truncated);
    }

    #[test]
    fn effective_generation_budget_boundary_is_not_truncated() {
        // Exactly fits: prompt + requested + 1 == context_length is fine as
        // requested, not a truncation case.
        let (budget, truncated) = effective_generation_budget(10, 21, 32);
        assert_eq!(budget, 21);
        assert!(!truncated);
    }
}
