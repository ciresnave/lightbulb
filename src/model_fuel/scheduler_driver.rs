//! The real `SchedulerDriver` (`crate::engine::scheduled_runner::SchedulerDriver`
//! — `pub(crate)`, so not an intra-doc link here: this module is `pub`, and
//! rustdoc denies a public item linking to a private one) adapter: wraps
//! `fuel_inference::multi_session::SessionScheduler` so
//! `crate::engine::scheduled_runner::run_scheduled_jobs` (board item 97) can
//! drive any `DecodeModel` through Fuel's own proven batched multi-session
//! scheduler, instead of `model_runner.rs`'s existing `run_jobs` processing
//! one request fully to completion before looking at the next.
//!
//! Generic over `M: DecodeModel`, not hardwired to one architecture — Llama
//! (`fuel_model_llama::LlamaModel`, `fuel::lazy_llama_full::Llama3Model`,
//! `fuel::lazy_quantized_llama::QuantizedLlama3Model`) and, as of fuel#284,
//! Qwen3 (`fuel::lazy_qwen3::Qwen3Model`, `fuel::lazy_quantized_qwen3::QuantizedQwen3Model`)
//! all implement it, so this one adapter serves either family.
//!
//! `#[allow(dead_code)]`: this is live under `ModelRunner::start`'s
//! `fuel-engine` arm, but the default-feature `clippy` gate still builds this
//! file without that feature enabled, so it has no caller there. Same shape
//! as `scheduled_runner.rs`'s own allowance.
#![allow(dead_code)]

use fuel::Device;
use fuel::lazy::SamplingStrategy;
use fuel_inference::multi_session::{
    DecodeModel, KvBudget, SchedulePolicy, SessionId, SessionScheduler,
};

use crate::engine::scheduled_runner::{AdmitError, FinishOutcome, SchedulerDriver, StopReason};

/// Adapts a Fuel [`SessionScheduler`] to [`SchedulerDriver`].
///
/// # Per-session sampling seed
///
/// `SchedulerDriver::try_admit` carries no request identity (by design — see
/// its doc comment), unlike `crate::sampling::seed_for(request_id, ...)` used
/// by the old serial `FuelEngineModel` path (deleted, board item 97 cleanup).
/// `SessionScheduler` samples with ONE `SamplingStrategy` fixed at admission
/// (not reseeded per token), so this driver mints a fresh seed per admitted
/// session from a monotonic counter instead. That was a real, deliberate
/// difference from the serial path's determinism model (reproducible by
/// request id) — documented here rather than silently assumed equivalent.
pub(crate) struct FuelSchedulerDriver<'m, M: DecodeModel> {
    scheduler: SessionScheduler<'m, M>,
    eos_ids: Option<Vec<u32>>,
    next_seed: u64,
    /// How many of each session's `new_tokens` this driver has already
    /// reported out of [`step`](SchedulerDriver::step). **Load-bearing, not
    /// an optimization**: `SessionScheduler::step`'s prefill pass samples a
    /// freshly-admitted session's FIRST token and then immediately also
    /// decode-advances it in the SAME tick (its own doc comment: "(2) collect
    /// the Decode-ready set" explicitly "includes sessions just prefilled
    /// above") — so a newly admitted session can produce TWO tokens on its
    /// first tick, and `StepReport::advanced` lists that session's id twice.
    /// Returning only `session_new_tokens(id).last()` per `advanced` entry
    /// would report the second token twice and silently drop the first —
    /// invisible in a `Complete` response (the final text is still right,
    /// `reap_finished` returns the full sequence) but a real missing token
    /// in a `Streaming` response. Diffing against this map's length instead
    /// of trusting one `last()` per report entry is what fixes that.
    seen_counts: std::collections::HashMap<SessionId, usize>,
}

impl<'m, M: DecodeModel> FuelSchedulerDriver<'m, M> {
    pub(crate) fn new(
        model: &'m M,
        device: Device,
        dtype: fuel::DType,
        policy: SchedulePolicy,
        budget: KvBudget,
        eos_ids: Option<Vec<u32>>,
    ) -> fuel::Result<Self> {
        let scheduler = SessionScheduler::new(model, device, dtype, policy, budget)?;
        Ok(Self {
            scheduler,
            eos_ids,
            next_seed: 0,
            seen_counts: std::collections::HashMap::new(),
        })
    }
}

impl<'m, M: DecodeModel> SchedulerDriver for FuelSchedulerDriver<'m, M> {
    type Id = SessionId;

    fn try_admit(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        temperature: f64,
    ) -> Result<SessionId, AdmitError> {
        // Pre-check capacity ourselves so we can tell "no room" apart from
        // "this request can never be admitted" — `add_session` conflates
        // both into one `Err` (see `AdmitError`'s doc comment for why that
        // distinction matters: an empty prompt retried forever hangs the
        // client instead of ever erroring back).
        let needed = self.scheduler.kv_blocks_required(prompt.len(), max_new);
        if needed > self.scheduler.kv_free_blocks() {
            return Err(AdmitError::NoCapacity);
        }

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

        // Capacity already confirmed above, so any `Err` reaching here is a
        // permanent rejection (empty prompt, zero budget) — never retried.
        self.scheduler
            .add_session(prompt, strategy, self.eos_ids.clone(), max_new)
            .map_err(|e| AdmitError::Rejected(e.to_string()))
    }

    fn step(&mut self) -> Result<Vec<(SessionId, u32)>, String> {
        let report = self.scheduler.step().map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        // `report.advanced` can repeat an id this tick (see `seen_counts`'
        // doc comment) — dedupe first, THEN diff each id's `new_tokens`
        // against what was already reported, so a session that produced two
        // tokens this tick yields two (id, token) pairs, in order, exactly
        // once each, not the last token twice.
        let mut ids = report.advanced;
        ids.sort_by_key(|id| id.0);
        ids.dedup();
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
                // Mirrors `SessionState`'s own transition rule exactly
                // (`fuel-inference/src/multi_session.rs`: `self.eos_ids
                // .contains(next) || self.remaining == 0` flips a session to
                // `Finished`, post fuel#307's multi-id widening) — the only
                // two ways a session stops, and the same check the scheduler
                // itself used internally.
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
        // "vacuously true when empty" (the method's own doc comment) means
        // this is also correct with zero sessions: no sessions -> not all
        // finished is false -> no active sessions. A session that finished
        // this tick is always reaped in the same tick `run_scheduled_jobs`
        // calls `step` (see its loop body), so finished-but-unreaped never
        // persists across the check this method answers.
        !self.scheduler.is_all_finished()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel::lazy::LayerWeights;
    use fuel_model_llama::{LlamaConfig, LlamaModel, LlamaWeights};
    use std::sync::Arc;

    // Mirrors fuel-inference's own `multi_session::tests::tiny_cfg`/
    // `tiny_weights`/`tiny_model` byte-for-byte (same seeded PRNG), so this
    // drives the IDENTICAL real `DecodeModel` fuel's own SessionScheduler
    // tests use — a real model, not a hand-rolled mock, per this project's
    // "real code over mocks" TDD discipline. Kept local rather than imported:
    // fuel's copy is `#[cfg(test)]`-private to its own crate.
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
    ) -> FuelSchedulerDriver<'_, LlamaModel> {
        FuelSchedulerDriver::new(
            model,
            Device::cpu(),
            fuel::DType::F32,
            SchedulePolicy::RoundRobin,
            budget,
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

        // Deliberately NOT asserting a fixed tick count: fuel's own `step()`
        // samples a freshly-admitted session's prefill token AND decode-
        // advances it again in that SAME tick (its doc comment: the decode
        // ready set "includes sessions just prefilled above"), so the first
        // tick alone can emit 2 of the 3 tokens. What must hold regardless of
        // that internal batching is the `SchedulerDriver` contract: every
        // token emitted exactly once, in order, across as many `step()`
        // calls as it takes.
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
                // prompt (3) + max_new (3) generated tokens.
                assert_eq!(tokens.len(), 6);
                let generated: Vec<u32> = tokens[3..].to_vec();
                let via_step: Vec<u32> = emitted.iter().map(|(_, tok)| *tok).collect();
                assert_eq!(
                    generated, via_step,
                    "reap_finished's generated tail must match step()'s emitted \
                     tokens exactly, in order — this is the invariant the \
                     seen_counts dedup/diff logic exists to preserve"
                );
            }
            FinishOutcome::Failed(e) => {
                panic!("session must not fail on a healthy tiny model: {e}")
            }
        }
        assert!(!d.has_active_sessions(), "the only session was just reaped");
    }

    #[test]
    fn try_admit_reports_no_capacity_distinctly_from_a_permanent_rejection() {
        let model = tiny_model(2);
        // One block of 4 tokens: the first session (prompt 3 + max_new 1 = 4)
        // exactly exhausts it, so a second admission must find zero room.
        let tight = KvBudget {
            block_size: 4,
            num_blocks: 1,
        };
        let mut d = driver(&model, tight, None);

        d.try_admit(&[1, 2, 3], 1, 0.0)
            .expect("first session must fit exactly");
        let second = d.try_admit(&[4, 5, 6], 1, 0.0);
        assert!(
            matches!(second, Err(AdmitError::NoCapacity)),
            "a full pool must report NoCapacity (retryable), not Rejected \
             (permanent) — got {second:?}",
        );

        // A genuinely-unadmittable request (zero budget) must be Rejected —
        // never retried, regardless of how much capacity is free.
        let roomy = KvBudget {
            block_size: 16,
            num_blocks: 4096,
        };
        let mut d2 = driver(&model, roomy, None);
        let zero_budget = d2.try_admit(&[1, 2, 3], 0, 0.0);
        assert!(
            matches!(zero_budget, Err(AdmitError::Rejected(_))),
            "zero max_new must be a permanent rejection with capacity free — got {zero_budget:?}",
        );
    }

    #[test]
    fn reap_finished_reports_eos_when_the_last_token_matches_eos_id() {
        let model = tiny_model(3);

        // Discover this tiny model's fixed-point greedy token without
        // hardcoding its bit pattern (see fuel's own CAUTION on this
        // fixture: greedy collapses to one token regardless of context).
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

    /// One case of
    /// [`reap_finished_reports_eos_when_either_of_two_configured_ids_matches`]:
    /// admit a session with the given `eos_ids`, step once, and assert it
    /// finished with `StopReason::Eos`. Split out purely to keep that test
    /// under Codacy's per-method line limit.
    fn assert_stops_on_eos(model: &LlamaModel, case: &str, eos_ids: Vec<u32>) {
        let mut d = driver(model, generous_budget(), Some(eos_ids));
        let id = d
            .try_admit(&[1, 2, 3], 10, 0.0)
            .unwrap_or_else(|e| panic!("[{case}] admission must fit: {e:?}"));
        d.step()
            .unwrap_or_else(|e| panic!("[{case}] step must not error: {e}"));
        let reaped = d.reap_finished();
        assert_eq!(
            reaped.len(),
            1,
            "[{case}] a configured eos id matching the first sampled token \
             must finish in one step"
        );
        let (reaped_id, outcome) = &reaped[0];
        assert_eq!(*reaped_id, id);
        match outcome {
            FinishOutcome::Completed { stop, .. } => {
                assert_eq!(
                    *stop,
                    StopReason::Eos,
                    "[{case}] the last token matched one of the configured \
                     eos ids; must report Eos regardless of its position \
                     in the set"
                );
            }
            FinishOutcome::Failed(e) => {
                panic!("[{case}] session must not fail on a healthy tiny model: {e}")
            }
        }
    }

    /// fuel#307 widened `eos_id: Option<u32>` to `eos_ids: Option<Vec<u32>>` —
    /// the PM's explicit ask after that: prove a checkpoint declaring TWO
    /// stop tokens actually stops on EITHER one, not just the first slot
    /// (which `Vec::contains` already guarantees over a positional/first-only
    /// match, but is exactly the kind of thing worth proving rather than
    /// assuming once the type allows more than one).
    #[test]
    fn reap_finished_reports_eos_when_either_of_two_configured_ids_matches() {
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
        // A dummy id that cannot be the real token: `discovered_token` is a
        // `u32` sampled from a `vocab_size: 16` tiny model, so anything at or
        // above that vocab size is never producible.
        let unreachable_id = 1_000_000;

        assert_stops_on_eos(
            &model,
            "discovered id first",
            vec![discovered_token, unreachable_id],
        );
        assert_stops_on_eos(
            &model,
            "discovered id second",
            vec![unreachable_id, discovered_token],
        );
    }
}
