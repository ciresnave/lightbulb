//! The Fuel path, wearing the engine's interface.
//!
//! Everything above the realize boundary is Lightbulb's: which token to pick,
//! when to stop, whose request runs. Fuel supplies the forward pass, the KV
//! cache and the graph. That split is the one the port is built on.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::Path;

use anyhow::Result;

use crate::engine::model_runner::EngineModel;
use crate::engine::{RequestContext, RequestState};

use super::decoder::FuelDecoder;
use super::loader::LoadedLlama;
use super::session::SessionState;

/// Pick a token from a realized logit row.
///
/// `temperature <= 0.0` is greedy, routed through the EXISTING public
/// `generate::argmax` rather than a copy — a second implementation of a
/// tie-break is a second thing to keep in sync, and the golden pins that one.
fn select_token(logits: &[f32], temperature: f64, seed: u64) -> u32 {
    if temperature <= 0.0 {
        return super::generate::argmax(logits);
    }
    let mut l = logits.to_vec();
    crate::sampling::apply_temperature(&mut l, temperature as f32);
    crate::sampling::sample_from_logits(&l, seed) as u32
}

/// Derive one token's sampling seed from the request it belongs to and its
/// position within that request.
///
/// NOT a shared counter on the model. A single counter incremented across
/// every request (the original shape) makes one request's sampled tokens
/// depend on how much traffic the process served before it — the same
/// prompt at `temperature: 0.7` would return different text depending on
/// what ran earlier. Harmless only by accident: `run_jobs` currently drives
/// one request at a time, so nothing interleaves. It becomes real
/// cross-tenant coupling the moment `step_batch` sees a genuine batch, since
/// two requests stepping in the same call would then share the counter's
/// sequence. Hashing `(request_id, token_index)` instead makes each
/// request's seed sequence depend only on itself.
fn seed_for(request_id: &str, token_index: usize) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    request_id.hash(&mut hasher);
    token_index.hash(&mut hasher);
    hasher.finish()
}

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
fn effective_generation_budget(
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

/// A loaded Fuel model plus one decode session per in-flight request.
///
/// `pub(crate)`, not `pub`: it implements `EngineModel`
/// (`crate::engine::model_runner::EngineModel`), which is itself `pub(crate)`
/// so that `engine/`'s seam to the tensor layer does not become public API by
/// accident. Widening one without the other would only re-expose the same
/// seam through the back door, so both stay crate-private together. Nothing
/// outside this crate constructs a `FuelEngineModel` today; if that changes,
/// widen deliberately rather than as a side effect of a compiler complaint.
pub(crate) struct FuelEngineModel {
    loaded: LoadedLlama,
    /// Keyed by `RequestContext.request.id`. Entries are created on the
    /// request's first step and dropped when it completes — a session holds a
    /// pre-allocated KV cache (~92 MiB for TinyLlama at 2048 f32), so leaking
    /// them leaks real memory.
    ///
    /// # Do not "optimise" this into a slot pool that reuses contexts
    ///
    /// One `SessionState` per REQUEST — each with its own `InferenceContext`
    /// and its own `KvCache` — is correct and confirmed correct by Fuel
    /// (2026-08-05). The obvious next optimisation is a fixed pool of slots
    /// that reuses a context across successive requests to avoid rebuilding
    /// the decode plan per request. **That is unsafe as Fuel stands**, and it
    /// fails silently:
    ///
    /// - The held plan's KV Arcs are bound ONCE at build time and mutate in
    ///   place via `Op::WriteSlice`. `rebind_and_realize_prebuilt` rebinds
    ///   `token_ids`/`rope_cos`/`rope_sin`/`mask`/offset and **not** the KV
    ///   nodes — so a plan is welded to the exact `KvCache` it was built on.
    /// - `DecodeSession::is_valid_for`'s key is `(seq, max_seq_len, n_layers,
    ///   cache_dtype)` — pure geometry. It cannot distinguish two `KvCache`
    ///   instances of the same shape, which in a slot pool is the normal case.
    ///
    /// Retire request A from a slot, admit B with a fresh same-shaped cache,
    /// reuse the slot's context: the key matches, the plan is reused, and B
    /// decodes over A's KV buffers. No error, plausible tokens, one tenant
    /// reading another's context.
    ///
    /// If plan reuse across requests is wanted later, the safe shape is a slot
    /// owning a PERSISTENT `KvCache` and reusing **cache and context together**,
    /// resetting the cache between requests rather than allocating a new one.
    /// Tell Fuel first — they offered to add cache identity to the validity key
    /// so the unsafe version fails loudly instead of silently.
    sessions: HashMap<String, SessionState>,
    context_length: usize,
}

impl FuelEngineModel {
    /// Load a SafeTensors checkpoint directory.
    ///
    /// Fails here rather than at first request if the tokenizer or config is
    /// missing — see `LoadedLlama`.
    pub(crate) fn load(model_dir: &Path, context_length: usize) -> Result<Self> {
        let loaded = super::loader_f32::load_llama_f32_from_dir(model_dir)?;
        Ok(Self {
            loaded,
            sessions: HashMap::new(),
            context_length,
        })
    }

    /// Advance one request by at most one token.
    fn step_one(&mut self, ctx: &mut RequestContext) -> Result<Option<u32>> {
        match ctx.state {
            RequestState::Completed | RequestState::AwaitingToolResult { .. } => Ok(None),
            RequestState::Pending => {
                // `ctx.add_special_tokens`, never a literal `true`: a prompt
                // rendered through a chat template already carries the
                // checkpoint's BOS, and TinyLlama/Llama's `TemplateProcessing`
                // post_processor would prepend a second one. See
                // `RequestContext::add_special_tokens`.
                let ids: Vec<u32> = self
                    .loaded
                    .tokenizer
                    .encode(ctx.request.prompt.as_str(), ctx.add_special_tokens)
                    .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?
                    .get_ids()
                    .to_vec();

                ctx.prompt_tokens = ids.len();

                // Generation can overflow the context even when the prompt
                // alone fits: shrink the request's own budget up front so the
                // decode loop's `should_continue()` stops naturally instead
                // of running into Fuel's cache bound mid-generation. See
                // `effective_generation_budget` for why this is truncation,
                // not wrap-around.
                let (effective_budget, truncated) = effective_generation_budget(
                    ids.len(),
                    ctx.request.max_new_tokens,
                    self.context_length,
                );
                if truncated {
                    tracing::warn!(
                        request_id = %ctx.request.id,
                        requested_max_new_tokens = ctx.request.max_new_tokens,
                        truncated_max_new_tokens = effective_budget,
                        context_length = self.context_length,
                        prompt_tokens = ids.len(),
                        "requested max_new_tokens does not fit in context_length; \
                         truncating the generation budget so this request returns \
                         the tokens it can generate instead of erroring mid-decode"
                    );
                    ctx.request.max_new_tokens = effective_budget;
                }

                // Cache capacity is fixed at allocation, so it must cover
                // prompt + generation. Capped at context_length because the KV
                // cache is pre-allocated and an oversized request would
                // otherwise allocate unboundedly.
                //
                // `saturating_add`: `max_new_tokens` comes straight from
                // client JSON, unvalidated. A plain `+` panics on overflow in
                // debug (killing the runner thread and stranding every later
                // request) and silently wraps in release.
                let want = ids
                    .len()
                    .saturating_add(ctx.request.max_new_tokens)
                    .saturating_add(1);
                let max_seq_len = want.min(self.context_length);
                if ids.len() >= max_seq_len {
                    anyhow::bail!(
                        "prompt of {} tokens does not fit in a context of {}",
                        ids.len(),
                        max_seq_len
                    );
                }

                let mut st =
                    SessionState::new(&self.loaded.config, max_seq_len, &self.loaded.device)?;
                let logits = self.loaded.model.prefill(&ids, &mut st)?;
                self.sessions.insert(ctx.request.id.clone(), st);

                let seed = seed_for(&ctx.request.id, ctx.tokens_generated);
                let tok = select_token(&logits, ctx.request_temperature(), seed);
                ctx.generated_tokens.push(tok);
                ctx.start_decoding();
                // Prefill advances the cache by the WHOLE prompt, not by one
                // token. `record_token()`'s `position += 1` is the decode-step
                // meaning; candlelight's own prefill arm
                // (`parallel_model_manager.rs`: `ctx.position += seq_len`)
                // uses the cache-position meaning instead, and both backends
                // feed the same `RequestContext` — `await_tool_result` reads
                // `position` as a cache offset — so this matches that rather
                // than calling `record_token()` here.
                ctx.position += ids.len();
                ctx.tokens_generated += 1;
                // Mirrors candlelight's own completion check (`is_eos_token ||
                // tokens_generated >= max_new_tokens`) rather than EOS alone.
                // EOS-only completion leaves the ordinary "ran out the
                // budget" request stuck in `Decoding` forever: `run_jobs`
                // still exits its loop via `should_continue()`, but nothing
                // ever removes this request's session, leaking its ~92 MiB KV
                // cache on every non-EOS completion — the common case, since
                // `chat.rs` defaults `max_tokens` to 100.
                let hit_eos = self.loaded.is_eos(tok);
                if hit_eos || ctx.tokens_generated >= ctx.request.max_new_tokens {
                    ctx.stopped_on_eos = hit_eos;
                    ctx.complete();
                    self.sessions.remove(&ctx.request.id);
                }
                Ok(Some(tok))
            }
            RequestState::Decoding => {
                let last = *ctx
                    .generated_tokens
                    .last()
                    .ok_or_else(|| anyhow::anyhow!("decoding with no previous token"))?;
                let st = self
                    .sessions
                    .get_mut(&ctx.request.id)
                    .ok_or_else(|| anyhow::anyhow!("no session for request {}", ctx.request.id))?;
                let logits = self.loaded.model.step(last, st)?;

                let seed = seed_for(&ctx.request.id, ctx.tokens_generated);
                let tok = select_token(&logits, ctx.request_temperature(), seed);
                ctx.generated_tokens.push(tok);
                ctx.record_token();
                let hit_eos = self.loaded.is_eos(tok);
                if hit_eos || ctx.tokens_generated >= ctx.request.max_new_tokens {
                    ctx.stopped_on_eos = hit_eos;
                    ctx.complete();
                    self.sessions.remove(&ctx.request.id);
                }
                Ok(Some(tok))
            }
        }
    }
}

impl EngineModel for FuelEngineModel {
    fn step_batch(&mut self, batch: &mut [RequestContext]) -> Result<Vec<Option<u32>>> {
        // A request can stop without ever reaching EOS or max_new_tokens
        // inside `step_one`: a client disconnect (the streaming send in
        // `run_jobs` failing), a decode error propagated up through `?`, or
        // a detokenize failure all break `run_jobs`'s loop from the OUTSIDE,
        // so `step_one` never runs again for that request and the
        // completion-based removal above never fires. Sweeping here catches
        // exactly those paths: anything left in `sessions` whose id is not
        // in the batch `run_jobs` is currently driving belongs to a request
        // that will never be stepped again.
        let live: HashSet<&str> = batch.iter().map(|ctx| ctx.request.id.as_str()).collect();
        self.sessions.retain(|id, _| live.contains(id.as_str()));

        let mut out = Vec::with_capacity(batch.len());
        for ctx in batch.iter_mut() {
            out.push(self.step_one(ctx)?);
        }
        Ok(out)
    }

    fn decode_text(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        self.loaded
            .tokenizer
            .decode(tokens, skip_special)
            .map_err(|e| anyhow::anyhow!("detokenizing: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Greedy selection is deterministic; sampled selection at the same seed is
    /// reproducible; sampled selection at different seeds actually varies.
    ///
    /// The first three assertions alone are a tautology, not a test: `f(x) ==
    /// f(x)` holds for ANY pure function of `(logits, temperature, seed)`,
    /// including one that ignores `temperature` and `seed` entirely (`fn
    /// select_token(l, _, _) { argmax(l) }` passes all three). The final
    /// assertion is the one that can fail informatively — it needs the seed
    /// to actually move the output, which needs a distribution flat enough
    /// for sampling to have somewhere to go. `peaked` below is too sharp for
    /// that: softmax at temperature 1.0 puts the overwhelming majority of the
    /// mass on index 1, so most seeds land there anyway and would prove
    /// nothing, which is why a second, flat row is used for that assertion.
    #[test]
    fn select_token_is_greedy_at_zero_and_seeded_above() {
        let peaked = vec![0.1, 3.0, 0.2, 2.9, 0.3];
        assert_eq!(
            select_token(&peaked, 0.0, 1),
            1,
            "temperature 0 must be argmax"
        );
        assert_eq!(
            select_token(&peaked, 0.0, 999),
            1,
            "greedy must ignore the seed"
        );
        assert_eq!(
            select_token(&peaked, 1.0, 7),
            select_token(&peaked, 1.0, 7),
            "same seed must reproduce"
        );

        let flat = vec![1.0; 8];
        let tokens: HashSet<u32> = (1u64..=5)
            .map(|seed| select_token(&flat, 1.0, seed))
            .collect();
        assert!(
            tokens.len() > 1,
            "seeds 1..=5 over a flat distribution all produced the same token \
             ({tokens:?}) — select_token is not threading the seed through"
        );
    }

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
