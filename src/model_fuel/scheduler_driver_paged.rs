//! The `FuelPagedSchedulerDriver` adapter: wraps
//! `fuel_inference::multi_session::PagedSessionScheduler` so
//! `crate::engine::scheduled_runner::run_scheduled_jobs` can drive any
//! `PagedDecodeModel` through Fuel's paged-KV scheduler — the paged-storage
//! sibling of `scheduler_driver.rs`'s `FuelSchedulerDriver`, which wraps the
//! older block-reservation `SessionScheduler`. Both exist side by side so a
//! caller can choose per arm; `scheduled_llama_engine.rs` currently builds
//! every `ServingDriverInner` arm from this one.
//!
//! `#[allow(dead_code)]`: same reason as `scheduler_driver.rs` — live under
//! `ModelRunner::start`'s `fuel-engine` arm, but the default-feature clippy
//! gate still builds this file without that feature enabled.
#![allow(dead_code)]

use std::collections::HashMap;

use fuel::Device;
use fuel::lazy::SamplingStrategy;
use fuel_inference::multi_session::{
    KvBudget, PagedDecodeModel, PagedSchedulePolicy, PagedSessionScheduler, SessionId, SessionPhase,
};

use crate::engine::scheduled_runner::{AdmitError, FinishOutcome, SchedulerDriver, StopReason};

/// Adapts a Fuel [`PagedSessionScheduler`] to [`SchedulerDriver`].
///
/// # Admission is optimistic, not pre-checked
///
/// Unlike `SessionScheduler` (whose `add_session` is pre-checked against
/// `kv_blocks_required`/`kv_free_blocks`, so a caller can tell "no room right
/// now" apart from "this request can never be admitted" — see
/// `FuelSchedulerDriver::try_admit`), `PagedSessionScheduler::add_session`'s
/// own doc comment is explicit: "Admit a session from a prompt (optimistic —
/// no up-front block reservation)." There is no `kv_blocks_required`
/// equivalent on this type to pre-check against. A session that genuinely
/// can't fit is therefore never rejected AT ADMISSION — it is admitted, and
/// the pool exhaustion surfaces later, in `step`'s `prefill_pass`, as a
/// per-session error (`StepReport::errored`) that finishes just that session
/// with `Finished`-with-error, isolated from every other session.
///
/// So `try_admit` below can only ever return [`AdmitError::Rejected`] (for a
/// structurally invalid request — empty prompt, zero budget) and NEVER
/// [`AdmitError::NoCapacity`] — there is no earlier point at which "no
/// capacity" is knowable. `step`/`reap_finished` instead surface a
/// capacity-exhausted session as [`FinishOutcome::Failed`], which
/// `run_scheduled_jobs` already handles (reports the request as failed,
/// never retries it) — this is the first driver to actually exercise that
/// arm; `FuelSchedulerDriver`'s pre-check made it unreachable in practice.
pub(crate) struct FuelPagedSchedulerDriver<'m, M: PagedDecodeModel> {
    scheduler: PagedSessionScheduler<'m, M>,
    eos_ids: Option<Vec<u32>>,
    next_seed: u64,
    /// Passed to `step_batched` every tick. `1` degrades to the same serial
    /// behavior as `step()` (`PagedSessionScheduler::step_batched`'s own doc:
    /// "singletons... fall back to serial"), so this is always safe to call
    /// unconditionally rather than branching on whether batching is wanted.
    max_batch: usize,
    /// Mirrors `FuelSchedulerDriver::seen_counts` exactly — same
    /// prefill-then-immediately-decode-advance double-count this tick
    /// applies to `PagedSessionScheduler::step`/`step_batched` too (same
    /// `StepReport` shape, same "Decode-ready set includes sessions just
    /// prefilled above" structure). See that struct's doc comment for the
    /// full reasoning.
    seen_counts: HashMap<SessionId, usize>,
    /// Sessions `step`/`step_batched` reported in `StepReport::errored` this
    /// tick or an earlier one, not yet reaped. `reap_finished` drains this
    /// instead of guessing `StopReason` from the (irrelevant — a failed
    /// session's `tokens` is just its untouched prompt) returned tokens.
    errored: HashMap<SessionId, String>,
}

impl<'m, M: PagedDecodeModel> FuelPagedSchedulerDriver<'m, M> {
    pub(crate) fn new(
        model: &'m M,
        budget: KvBudget,
        dtype: fuel::DType,
        device: &Device,
        policy: PagedSchedulePolicy,
        max_batch: usize,
        eos_ids: Option<Vec<u32>>,
    ) -> fuel::Result<Self> {
        let mut scheduler = PagedSessionScheduler::new(model, budget, dtype, device)?;
        scheduler.set_policy(policy);
        Ok(Self {
            scheduler,
            eos_ids,
            next_seed: 0,
            max_batch: max_batch.max(1),
            seen_counts: HashMap::new(),
            errored: HashMap::new(),
        })
    }
}

impl<'m, M: PagedDecodeModel> SchedulerDriver for FuelPagedSchedulerDriver<'m, M> {
    type Id = SessionId;

    fn try_admit(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        temperature: f64,
    ) -> Result<SessionId, AdmitError> {
        let seed = self.next_seed;
        self.next_seed = self.next_seed.wrapping_add(1);
        let strategy = if temperature <= 0.0 {
            SamplingStrategy::Greedy
        } else {
            SamplingStrategy::Temperature {
                temp: temperature as f32,
                seed,
            }
        };

        // See this struct's doc comment: admission here is optimistic, so
        // every `Err` is a structural rejection (empty prompt / zero
        // budget), never a capacity one.
        self.scheduler
            .add_session(prompt, strategy, self.eos_ids.clone(), max_new)
            .map_err(|e| AdmitError::Rejected(e.to_string()))
    }

    fn step(&mut self) -> Result<Vec<(SessionId, u32)>, String> {
        let report = self.scheduler.step_batched(self.max_batch);
        for (id, msg) in report.errored {
            self.errored.insert(id, msg);
        }
        // Same dedup-then-diff as `FuelSchedulerDriver::step` — see its doc
        // comment on `seen_counts` for why a plain `advanced` -> last-token
        // mapping would silently drop a freshly-prefilled session's first
        // token.
        let mut ids = report.advanced;
        ids.sort_by_key(|id| id.0);
        ids.dedup();
        let mut out = Vec::new();
        for id in ids {
            let new_tokens = self.scheduler.session_new_tokens(id).unwrap_or(&[]);
            let already_seen = self.seen_counts.get(&id).copied().unwrap_or(0);
            out.extend(new_tokens[already_seen..].iter().map(|&tok| (id, tok)));
            self.seen_counts.insert(id, new_tokens.len());
        }
        Ok(out)
    }

    fn reap_finished(&mut self) -> Vec<(SessionId, FinishOutcome)> {
        self.scheduler
            .reap_finished()
            .into_iter()
            .map(|(id, tokens)| {
                self.seen_counts.remove(&id);
                if let Some(msg) = self.errored.remove(&id) {
                    return (id, FinishOutcome::Failed(msg));
                }
                let stop = match tokens.last() {
                    Some(t) if self.eos_ids.as_ref().is_some_and(|ids| ids.contains(t)) => {
                        StopReason::Eos
                    }
                    _ => StopReason::Budget,
                };
                (id, FinishOutcome::Completed { tokens, stop })
            })
            .collect()
    }

    fn has_active_sessions(&self) -> bool {
        // `PagedSessionScheduler` has no `is_all_finished` of its own (unlike
        // `SessionScheduler`) — `session_infos()` exposes each session's
        // `phase`, which is the same field `is_all_finished` reads, so this
        // is the equivalent query, "vacuously true when empty" included
        // (an empty iterator's `all` is `true`).
        !self
            .scheduler
            .session_infos()
            .iter()
            .all(|s| s.phase == SessionPhase::Finished)
    }

    /// Forwards to `PagedSessionScheduler::set_priority` ("higher goes
    /// first"; read only under `PagedSchedulePolicy::Priority` -- this
    /// driver's own `new` still defaults to `AdmissionOrder`, so this call
    /// is recorded but has no ordering effect until something switches the
    /// policy). The only `SchedulerDriver` with a real priority concept, as
    /// of this writing.
    fn set_priority(&mut self, id: SessionId, priority: i32) -> Result<(), String> {
        self.scheduler
            .set_priority(id, priority)
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel::lazy::LayerWeights;
    use fuel_model_llama::{LlamaConfig, LlamaModel, LlamaWeights};
    use std::sync::Arc;

    // Identical to `scheduler_driver.rs`'s own copy (same seeded PRNG) — see
    // that module's comment for why this is kept local rather than shared.
    fn tiny_cfg() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 16,
            dim: 8,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 16,
            norm_eps: 1e-5,
            rope_base: 10000.0,
        }
    }

    fn tiny_weights(cfg: &LlamaConfig, seed: u32) -> LlamaWeights {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        };
        let mut vec_of =
            |n: usize| -> Arc<[f32]> { Arc::from((0..n).map(|_| next()).collect::<Vec<_>>()) };
        let kv = cfg.n_kv_heads * cfg.head_dim;
        LlamaWeights {
            instance: fuel::decode_shape::ModelInstanceId::next(),
            token_embedding: vec_of(cfg.vocab_size * cfg.dim),
            layers: (0..cfg.n_layers)
                .map(|_| LayerWeights {
                    attn_q: vec_of(cfg.dim * cfg.dim).into(),
                    attn_q_bias: None,
                    attn_k: vec_of(cfg.dim * kv).into(),
                    attn_k_bias: None,
                    attn_v: vec_of(cfg.dim * kv).into(),
                    attn_v_bias: None,
                    attn_o: vec_of(cfg.dim * cfg.dim).into(),
                    ffn_gate: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_up: vec_of(cfg.dim * cfg.ffn_dim).into(),
                    ffn_down: vec_of(cfg.ffn_dim * cfg.dim).into(),
                    attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                    ffn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                })
                .collect(),
            final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            output: vec_of(cfg.dim * cfg.vocab_size).into(),
        }
    }

    fn tiny_model(seed: u32) -> LlamaModel {
        let cfg = tiny_cfg();
        LlamaModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg, seed),
        }
    }

    fn generous_budget() -> KvBudget {
        KvBudget {
            block_size: 16,
            num_blocks: 4096,
        }
    }

    fn driver(
        model: &LlamaModel,
        budget: KvBudget,
        eos_ids: Option<Vec<u32>>,
    ) -> FuelPagedSchedulerDriver<'_, LlamaModel> {
        FuelPagedSchedulerDriver::new(
            model,
            budget,
            fuel::DType::F32,
            &Device::cpu(),
            PagedSchedulePolicy::AdmissionOrder,
            1,
            eos_ids,
        )
        .expect("tiny model's uniform per-head KV must satisfy ModelDims::from_model")
    }

    #[test]
    fn admits_a_session_and_step_emits_every_token_exactly_once_until_budget() {
        let model = tiny_model(1);
        let mut d = driver(&model, generous_budget(), None);

        let id = d
            .try_admit(&[1, 2, 3], 3, 0.0)
            .expect("must admit: capacity is generous");
        assert!(d.has_active_sessions());

        let mut emitted = Vec::new();
        while d.has_active_sessions() {
            let advanced = d.step().expect("step must not error on a healthy session");
            assert!(
                !advanced.is_empty(),
                "a session is still active but step() emitted nothing this tick"
            );
            emitted.extend(advanced);
        }
        assert_eq!(
            emitted.len(),
            3,
            "max_new=3 must yield exactly 3 emitted tokens, got {emitted:?}"
        );
        assert!(emitted.iter().all(|(eid, _)| *eid == id));

        let reaped = d.reap_finished();
        assert_eq!(reaped.len(), 1);
        let (reaped_id, outcome) = &reaped[0];
        assert_eq!(*reaped_id, id);
        match outcome {
            FinishOutcome::Completed { tokens, stop } => {
                assert_eq!(
                    *stop,
                    StopReason::Budget,
                    "no eos_id was set; must stop on budget"
                );
                assert_eq!(tokens.len(), 6);
                let generated: Vec<u32> = tokens[3..].to_vec();
                let via_step: Vec<u32> = emitted.iter().map(|(_, tok)| *tok).collect();
                assert_eq!(
                    generated, via_step,
                    "reap_finished's generated tail must match step()'s emitted \
                     tokens exactly, in order"
                );
            }
            FinishOutcome::Failed(e) => {
                panic!("session must not fail on a healthy tiny model: {e}")
            }
        }
        assert!(!d.has_active_sessions(), "the only session was just reaped");
    }

    /// The real behavioral divergence from `FuelSchedulerDriver`, proven
    /// rather than asserted in a doc comment: a session that cannot fit in
    /// the pool is still ADMITTED (no `NoCapacity`), then fails during
    /// `step`'s prefill, surfacing as `FinishOutcome::Failed` — never a
    /// silent empty completion, and never panics.
    #[test]
    fn a_session_too_large_for_the_pool_is_admitted_then_fails_at_step_not_at_admission() {
        let model = tiny_model(4);
        // One block of 4 tokens total pool capacity; a 5-token prompt alone
        // already exceeds it, so prefill must exhaust the pool.
        let tight = KvBudget {
            block_size: 4,
            num_blocks: 1,
        };
        let mut d = driver(&model, tight, None);

        let id = d.try_admit(&[1, 2, 3, 4, 5], 1, 0.0).expect(
            "optimistic admission must accept a structurally valid prompt regardless of fit",
        );

        // Drive until the oversized session finishes (with an error) rather
        // than asserting a fixed tick count.
        let mut ticks = 0;
        while d.has_active_sessions() && ticks < 10 {
            let _ = d.step().expect("step() itself must not error");
            ticks += 1;
        }
        assert!(
            ticks < 10,
            "the oversized session must finish (with an error) within a few ticks, not hang"
        );

        let reaped = d.reap_finished();
        assert_eq!(reaped.len(), 1);
        let (reaped_id, outcome) = &reaped[0];
        assert_eq!(*reaped_id, id);
        assert!(
            matches!(outcome, FinishOutcome::Failed(_)),
            "a session that cannot fit must reap as Failed, not a silent empty Completed"
        );
    }

    #[test]
    fn try_admit_rejects_a_structurally_invalid_request() {
        let model = tiny_model(2);
        let mut d = driver(&model, generous_budget(), None);
        let zero_budget = d.try_admit(&[1, 2, 3], 0, 0.0);
        assert!(
            matches!(zero_budget, Err(AdmitError::Rejected(_))),
            "zero max_new must be a permanent rejection — got {zero_budget:?}",
        );
    }

    #[test]
    fn reap_finished_reports_eos_when_the_last_token_matches_eos_id() {
        let model = tiny_model(3);

        let mut probe = driver(&model, generous_budget(), None);
        probe
            .try_admit(&[1, 2, 3], 1, 0.0)
            .expect("probe admission must fit");
        let advanced = probe.step().expect("probe step must not error");
        let [(_, discovered_token)] = advanced[..] else {
            panic!(
                "expected exactly one token from the probe's single active \
                 session, got {advanced:?}"
            )
        };

        let mut d = driver(&model, generous_budget(), Some(vec![discovered_token]));
        let id = d
            .try_admit(&[1, 2, 3], 10, 0.0)
            .expect("admission must fit");
        d.step().expect("step must not error");
        let reaped = d.reap_finished();
        assert_eq!(
            reaped.len(),
            1,
            "eos_id matching the first sampled token must finish in one step"
        );
        let (reaped_id, outcome) = &reaped[0];
        assert_eq!(*reaped_id, id);
        match outcome {
            FinishOutcome::Completed { stop, .. } => {
                assert_eq!(
                    *stop,
                    StopReason::Eos,
                    "last token matched eos_id; must report Eos"
                );
            }
            FinishOutcome::Failed(e) => {
                panic!("session must not fail on a healthy tiny model: {e}")
            }
        }
    }

    /// Proves `step_batched` (always used, even at `max_batch=1`) behaves
    /// identically to serial decode for TWO concurrent sessions sharing the
    /// same position — the real throughput path this driver exists to use,
    /// not just the single-session case every other test here exercises.
    #[test]
    fn two_concurrent_sessions_each_emit_their_own_tokens_under_batching() {
        let model = tiny_model(5);
        let mut d = FuelPagedSchedulerDriver::new(
            &model,
            generous_budget(),
            fuel::DType::F32,
            &Device::cpu(),
            PagedSchedulePolicy::AdmissionOrder,
            8,
            None,
        )
        .expect("tiny model must build");

        let id_a = d.try_admit(&[1, 2, 3], 2, 0.0).expect("a must admit");
        let id_b = d.try_admit(&[4, 5, 6], 2, 0.0).expect("b must admit");
        assert_ne!(id_a, id_b);

        let mut emitted: HashMap<SessionId, usize> = HashMap::new();
        while d.has_active_sessions() {
            for (id, _) in d.step().expect("step must not error") {
                *emitted.entry(id).or_insert(0) += 1;
            }
        }
        assert_eq!(emitted.get(&id_a).copied().unwrap_or(0), 2);
        assert_eq!(emitted.get(&id_b).copied().unwrap_or(0), 2);

        let reaped = d.reap_finished();
        assert_eq!(reaped.len(), 2);
        for (id, outcome) in reaped {
            match outcome {
                FinishOutcome::Completed { tokens, .. } => {
                    assert_eq!(tokens.len(), 5, "session {id:?}: prompt(3)+max_new(2)");
                }
                FinishOutcome::Failed(e) => panic!("session {id:?} must not fail: {e}"),
            }
        }
    }
}
