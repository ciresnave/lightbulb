// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tick-based, multi-session job loop for a batched scheduler (board item 97:
//! "get Candle out of Lightbulb" — the integration half of wiring
//! `fuel_inference::multi_session::SessionScheduler`'s already-proven batched
//! decode into the serving path).
//!
//! # Why this is a separate loop from `run_jobs`
//!
//! `run_jobs` (this module's existing function) processes exactly one request
//! fully to completion before it even looks at the next job — true for BOTH
//! the candlelight and `fuel-engine` arms of `ModelRunner::start` today. That
//! is the real baseline this design improves on, not a hypothetical: a new
//! request can currently wait behind an entire prior request's full
//! generation (up to `max_new_tokens` steps), not just one step.
//!
//! This loop instead admits multiple concurrent sessions into a scheduler
//! that can batch them, and drives it one tick at a time. It is deliberately
//! generic over [`SchedulerDriver`] rather than hardwired to Fuel's
//! `SessionScheduler`, so the admission/dispatch logic — the part with real
//! bugs to find — is testable without a model, a checkpoint, or a GPU.
//!
//! # What is NOT yet here
//!
//! The real adapter wrapping `fuel_inference::multi_session::SessionScheduler`
//! is a follow-up, not this file. `SessionScheduler` has no accessor for a
//! session's newest token mid-generation — only `reap_finished` returns
//! tokens, and only once a session is `Finished` — so streaming mode cannot
//! be honestly implemented against it yet (asked the fuel lane for the
//! missing accessor; tracked, not guessed around). The trait and tick loop
//! below support streaming at the abstraction level so this file does not
//! need to change again once that lands.
//!
//! `#![allow(dead_code)]`: this module predates its own wiring into
//! `ModelRunner::start`'s `fuel-engine` arm (board item 97, PR #118); kept
//! since the default-feature `clippy` gate still builds this file without
//! that feature enabled, so some of what it declares is unreachable there.
//! Allowed here at this one site, not by raising the gate's ceiling.
#![allow(dead_code)]

use std::sync::mpsc::Receiver;

use super::model_runner::{
    CompletionResult, ControlMsg, FinishReason, InferenceJob, ResponseMode, StreamItem,
};

/// Why a session stopped generating — the information
/// [`SchedulerDriver::reap_finished`] must recover and report, since Fuel's
/// own `StepReport` (per board item 97's research) only says a session
/// finished, not EOS-vs-budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StopReason {
    Eos,
    Budget,
}

/// One finished session's outcome.
pub(crate) enum FinishOutcome {
    Completed { tokens: Vec<u32>, stop: StopReason },
    Failed(String),
}

/// Why [`SchedulerDriver::try_admit`] failed to admit a session — the two
/// cases need opposite handling, so they cannot share one untyped `Err`.
///
/// Found while wiring the real `fuel_inference::multi_session::SessionScheduler`
/// adapter: `SessionScheduler::add_session` can fail for a transient reason
/// (no KV capacity right now — retry once something is reaped) OR a permanent
/// one (empty prompt, zero budget — the request itself is invalid and will
/// NEVER fit, no matter how much capacity frees up). The original single-`Err`
/// signature could not tell these apart, so [`run_scheduled_jobs`] retried
/// EVERY admit failure forever — a permanently-invalid prompt would re-queue
/// at the front of `pending` every tick and silently hang that client forever
/// instead of ever reporting an error back.
#[derive(Debug)]
pub(crate) enum AdmitError {
    /// No room right now; leave the job unconsumed and retry on a later tick
    /// once [`SchedulerDriver::reap_finished`] frees capacity.
    NoCapacity,
    /// The request itself can never be admitted (e.g. empty prompt, zero
    /// budget) — report it as a request failure immediately, never retry.
    Rejected(String),
}

/// Abstracts a batched multi-session scheduler so [`run_scheduled_jobs`]'s
/// admission/dispatch logic is testable without a real model or GPU. The real
/// adapter wraps `fuel_inference::multi_session::SessionScheduler`.
pub(crate) trait SchedulerDriver {
    type Id: Copy + Eq + std::hash::Hash;

    /// Try to admit a new session. See [`AdmitError`] for how the two
    /// failure cases differ.
    fn try_admit(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        temperature: f64,
    ) -> Result<Self::Id, AdmitError>;

    /// Advance every active session by at most one step. Returns `(id, token)`
    /// for every session that produced a token THIS tick — including a
    /// session's own last token on the same tick it also appears in
    /// [`reap_finished`](Self::reap_finished).
    fn step(&mut self) -> Result<Vec<(Self::Id, u32)>, String>;

    /// Drain every session that finished since the last call. Per-session
    /// errors are isolated here, never surfacing as a `step` error for other
    /// sessions (Fuel's own structural guarantee — this trait just has to
    /// preserve it).
    fn reap_finished(&mut self) -> Vec<(Self::Id, FinishOutcome)>;

    /// Whether there is any active (admitted, not yet reaped) session. Used
    /// to decide whether the loop can block on new jobs or must keep
    /// stepping.
    fn has_active_sessions(&self) -> bool;

    /// Handle an administrative [`ControlMsg`] (board PM task, 2026-10-22:
    /// `cache_clear`'s real implementation). Default no-op so every existing
    /// implementor (including this file's own `FakeDriver` test fixture)
    /// stays correct without change -- only `ServingDriver`
    /// (`src/model_fuel/scheduled_llama_engine.rs`) has a real cache to
    /// clear.
    fn handle_control(&mut self, _msg: ControlMsg) {}

    /// Record `id`'s scheduling priority, called right after `try_admit`
    /// returns it successfully (see [`run_scheduled_jobs`]'s admission
    /// loop). `try_admit` itself carries no priority -- this is a separate
    /// step so a driver with no priority concept (every `SchedulerDriver`
    /// except the paged Fuel one, as of this writing) can simply not
    /// override it. Default no-op for that reason, same pattern as
    /// `handle_control`. A failure here is reported (`run_scheduled_jobs`
    /// logs it) but never fails the admission that already succeeded --
    /// priority is a best-effort ordering hint, not a correctness
    /// requirement.
    fn set_priority(&mut self, _id: Self::Id, _priority: i32) -> Result<(), String> {
        Ok(())
    }
}

/// Drive `driver` against jobs arriving on `rx`, dispatching results to each
/// job's own response channel. Exits when `rx` disconnects AND no job is
/// pending admission or active in the driver — never while there is
/// outstanding work, so a dropped sender during a burst cannot silently
/// orphan an in-flight generation.
pub(crate) fn run_scheduled_jobs<D: SchedulerDriver>(
    mut driver: D,
    rx: Receiver<InferenceJob>,
    ctrl_rx: Receiver<ControlMsg>,
    encode: impl Fn(&str, bool) -> anyhow::Result<Vec<u32>>,
    decode: impl Fn(&[u32], bool) -> anyhow::Result<String>,
) {
    use std::collections::{HashMap, VecDeque};

    struct JobMeta {
        response_mode: ResponseMode,
        prompt_tokens: usize,
    }

    fn send_error(mode: ResponseMode, err: anyhow::Error) {
        match mode {
            ResponseMode::Complete(tx) => {
                let _ = tx.send(Err(err));
            }
            ResponseMode::Streaming(tx) => {
                let _ = tx.send(Err(err));
            }
        }
    }

    let mut pending: VecDeque<InferenceJob> = VecDeque::new();
    let mut active: HashMap<D::Id, JobMeta> = HashMap::new();
    let mut channel_open = true;

    loop {
        while let Ok(msg) = ctrl_rx.try_recv() {
            driver.handle_control(msg);
        }

        while let Ok(job) = rx.try_recv() {
            pending.push_back(job);
        }

        if pending.is_empty() && !driver.has_active_sessions() {
            if !channel_open {
                break;
            }
            // `recv_timeout`, not `recv`: a bare blocking `recv()` here
            // would starve `ctrl_rx` -- a control message sent while the
            // loop is idle (no pending jobs, no active sessions) would
            // never be observed until a NEW inference job arrived to wake
            // this `recv()` up, which could be arbitrarily long after a
            // caller asked to clear the cache. The poll interval only
            // bounds how stale an idle clear can be, not correctness: a
            // disconnected sender is detected immediately, same as before
            // (`Err` from `recv()` and `RecvTimeoutError::Disconnected`
            // both fire without waiting out the duration).
            match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(job) => pending.push_back(job),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    channel_open = false;
                    continue;
                }
            }
        }

        // Admit as many pending jobs as the driver has capacity for. A
        // capacity `Err` puts the job back at the FRONT of `pending` and
        // stops the admission pass for this tick — it is retried next tick
        // after `reap_finished` frees room.
        while let Some(job) = pending.pop_front() {
            let prompt_tokens = match encode(&job.prompt, job.add_special_tokens) {
                Ok(t) => t,
                Err(e) => {
                    send_error(job.response_mode, e);
                    continue;
                }
            };
            let priority = job.priority;
            match driver.try_admit(&prompt_tokens, job.max_new_tokens, job.temperature) {
                Ok(id) => {
                    // M4 slice 5.4c (Option B) wiring: best-effort only --
                    // a driver with no priority concept returns `Ok(())`
                    // from the trait's default no-op, and a real failure
                    // here never un-admits a session that already
                    // succeeded (see `SchedulerDriver::set_priority`'s doc
                    // comment).
                    if let Err(e) = driver.set_priority(id, priority as i32) {
                        tracing::warn!(
                            error = %e,
                            "set_priority failed after admission; session stays admitted \
                             without priority ordering"
                        );
                    }
                    active.insert(
                        id,
                        JobMeta {
                            response_mode: job.response_mode,
                            prompt_tokens: prompt_tokens.len(),
                        },
                    );
                }
                Err(AdmitError::NoCapacity) => {
                    pending.push_front(job);
                    break;
                }
                Err(AdmitError::Rejected(msg)) => {
                    send_error(job.response_mode, anyhow::anyhow!(msg));
                }
            }
        }

        if active.is_empty() {
            continue;
        }

        match driver.step() {
            Ok(advanced) => {
                for (id, token) in advanced {
                    if let Some(JobMeta {
                        response_mode: ResponseMode::Streaming(tx),
                        ..
                    }) = active.get(&id)
                    {
                        match decode(&[token], false) {
                            Ok(text) => {
                                let _ = tx.send(Ok(StreamItem::Token(text)));
                            }
                            Err(e) => {
                                let _ = tx.send(Err(e));
                            }
                        }
                    }
                }
            }
            Err(e) => {
                // A driver-level step error (not a per-session one) fails
                // every currently active session — there is no narrower
                // owner to blame it on.
                for (_, meta) in active.drain() {
                    send_error(meta.response_mode, anyhow::anyhow!("{e}"));
                }
                continue;
            }
        }

        for (id, outcome) in driver.reap_finished() {
            let Some(meta) = active.remove(&id) else {
                continue;
            };
            match outcome {
                FinishOutcome::Completed { tokens, stop } => {
                    let finish_reason = match stop {
                        StopReason::Eos => FinishReason::Stop,
                        StopReason::Budget => FinishReason::Length,
                    };
                    let completion_tokens = tokens.len().saturating_sub(meta.prompt_tokens);
                    match meta.response_mode {
                        ResponseMode::Complete(resp_tx) => {
                            let result = decode(&tokens[meta.prompt_tokens..], true).map(|text| {
                                CompletionResult {
                                    text,
                                    prompt_tokens: meta.prompt_tokens,
                                    completion_tokens,
                                    finish_reason,
                                }
                            });
                            let _ = resp_tx.send(result);
                        }
                        ResponseMode::Streaming(tx) => {
                            let _ = tx.send(Ok(StreamItem::Done {
                                finish_reason,
                                prompt_tokens: meta.prompt_tokens,
                                completion_tokens,
                            }));
                        }
                    }
                }
                FinishOutcome::Failed(e) => {
                    send_error(meta.response_mode, anyhow::anyhow!(e));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::mpsc;
    use tokio::sync::oneshot;

    /// Treats each byte of the prompt as a token (lossy for non-ASCII, fine
    /// for these fixtures) so tests don't need a real tokenizer.
    fn fake_encode(s: &str, _add_special: bool) -> anyhow::Result<Vec<u32>> {
        Ok(s.bytes().map(|b| b as u32).collect())
    }
    fn fake_decode(toks: &[u32], _skip_special: bool) -> anyhow::Result<String> {
        Ok(toks.iter().map(|&t| t as u8 as char).collect())
    }

    /// A control channel with nothing on it and its sender already dropped
    /// -- fine for every test that isn't exercising control messages
    /// themselves: `ctrl_rx.try_recv()` just reports empty, same as an
    /// absent control plane would.
    fn no_control() -> Receiver<ControlMsg> {
        mpsc::channel().1
    }

    /// A fake prompt's first byte selects the fake session's behavior, so
    /// each test can drive a specific path without a real model.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum FakePlan {
        RunToBudget,
        HitEos,
        Fail,
        RejectAdmit,
    }

    struct FakeSession {
        prompt_len: usize,
        max_new: usize,
        produced: usize,
        plan: FakePlan,
    }

    struct FakeDriver {
        capacity: usize,
        next_id: u64,
        sessions: HashMap<u64, FakeSession>,
        /// Every `set_priority` call this driver has received, in order --
        /// proves `run_scheduled_jobs` actually threads `InferenceJob::priority`
        /// through, not just that it compiles against the trait's default
        /// no-op. `Rc<RefCell<_>>`, not a plain field, because
        /// `run_scheduled_jobs` takes ownership of the driver and never hands
        /// it back; a clone taken before the call survives to assert on.
        priority_log: std::rc::Rc<std::cell::RefCell<Vec<(u64, i32)>>>,
    }

    impl FakeDriver {
        fn new(capacity: usize) -> Self {
            Self {
                capacity,
                next_id: 0,
                sessions: HashMap::new(),
                priority_log: Default::default(),
            }
        }
    }

    impl SchedulerDriver for FakeDriver {
        type Id = u64;

        fn try_admit(
            &mut self,
            prompt: &[u32],
            max_new: usize,
            _temperature: f64,
        ) -> Result<u64, AdmitError> {
            if self.sessions.len() >= self.capacity {
                return Err(AdmitError::NoCapacity);
            }
            let plan = match prompt.first() {
                Some(69) => FakePlan::HitEos,      // 'E'
                Some(70) => FakePlan::Fail,        // 'F'
                Some(82) => FakePlan::RejectAdmit, // 'R'
                _ => FakePlan::RunToBudget,
            };
            if plan == FakePlan::RejectAdmit {
                return Err(AdmitError::Rejected(
                    "fake: permanently invalid prompt".to_string(),
                ));
            }
            let id = self.next_id;
            self.next_id += 1;
            self.sessions.insert(
                id,
                FakeSession {
                    prompt_len: prompt.len(),
                    max_new,
                    produced: 0,
                    plan,
                },
            );
            Ok(id)
        }

        fn step(&mut self) -> Result<Vec<(u64, u32)>, String> {
            let mut advanced = Vec::new();
            for (&id, s) in self.sessions.iter_mut() {
                if s.plan == FakePlan::Fail {
                    continue; // never produces a token; reaped as Failed below
                }
                let target = match s.plan {
                    FakePlan::HitEos => 1,
                    _ => s.max_new,
                };
                if s.produced >= target {
                    continue;
                }
                s.produced += 1;
                advanced.push((id, 100 + s.produced as u32));
            }
            Ok(advanced)
        }

        fn reap_finished(&mut self) -> Vec<(u64, FinishOutcome)> {
            let done_ids: Vec<u64> = self
                .sessions
                .iter()
                .filter(|(_, s)| match s.plan {
                    FakePlan::Fail => true,
                    FakePlan::HitEos => s.produced >= 1,
                    FakePlan::RunToBudget => s.produced >= s.max_new,
                    // Never inserted into `sessions` — try_admit rejects it
                    // before construction. Listed so this match stays total.
                    FakePlan::RejectAdmit => {
                        unreachable!("RejectAdmit sessions are never admitted into `sessions`")
                    }
                })
                .map(|(&id, _)| id)
                .collect();
            done_ids
                .into_iter()
                .map(|id| {
                    let s = self.sessions.remove(&id).unwrap();
                    let outcome = match s.plan {
                        FakePlan::Fail => FinishOutcome::Failed("fake failure".to_string()),
                        FakePlan::RejectAdmit => {
                            unreachable!("RejectAdmit sessions are never admitted into `sessions`")
                        }
                        FakePlan::HitEos | FakePlan::RunToBudget => {
                            let stop = if s.plan == FakePlan::HitEos {
                                StopReason::Eos
                            } else {
                                StopReason::Budget
                            };
                            let prompt_echo = std::iter::repeat_n(0u32, s.prompt_len);
                            let generated = (1..=s.produced as u32).map(|i| 100 + i);
                            FinishOutcome::Completed {
                                tokens: prompt_echo.chain(generated).collect(),
                                stop,
                            }
                        }
                    };
                    (id, outcome)
                })
                .collect()
        }

        fn has_active_sessions(&self) -> bool {
            !self.sessions.is_empty()
        }

        fn set_priority(&mut self, id: u64, priority: i32) -> Result<(), String> {
            self.priority_log.borrow_mut().push((id, priority));
            Ok(())
        }
    }

    fn complete_job(
        prompt: &str,
        max_new: usize,
    ) -> (
        InferenceJob,
        oneshot::Receiver<anyhow::Result<CompletionResult>>,
    ) {
        let (tx, rx) = oneshot::channel();
        let job = InferenceJob {
            id: "test".to_string(),
            prompt: prompt.to_string(),
            max_new_tokens: max_new,
            temperature: 0.0,
            add_special_tokens: true,
            priority: crate::engine::Priority::Normal,
            response_mode: ResponseMode::Complete(tx),
        };
        (job, rx)
    }

    fn complete_job_with_priority(
        prompt: &str,
        max_new: usize,
        priority: crate::engine::Priority,
    ) -> (
        InferenceJob,
        oneshot::Receiver<anyhow::Result<CompletionResult>>,
    ) {
        let (job, rx) = complete_job(prompt, max_new);
        (InferenceJob { priority, ..job }, rx)
    }

    fn streaming_job(
        prompt: &str,
        max_new: usize,
    ) -> (
        InferenceJob,
        tokio::sync::mpsc::UnboundedReceiver<anyhow::Result<StreamItem>>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let job = InferenceJob {
            id: "test".to_string(),
            prompt: prompt.to_string(),
            max_new_tokens: max_new,
            temperature: 0.0,
            add_special_tokens: true,
            priority: crate::engine::Priority::Normal,
            response_mode: ResponseMode::Streaming(tx),
        };
        (job, rx)
    }

    #[test]
    fn a_jobs_priority_is_set_on_the_driver_right_after_admission() {
        let (tx, rx) = mpsc::channel();
        let (job, mut result_rx) =
            complete_job_with_priority("hello", 3, crate::engine::Priority::High);
        tx.send(job).unwrap();
        drop(tx);

        let driver = FakeDriver::new(4);
        let priority_log = driver.priority_log.clone();

        run_scheduled_jobs(driver, rx, no_control(), fake_encode, fake_decode);

        result_rx.try_recv().unwrap().expect("job should succeed");
        assert_eq!(
            *priority_log.borrow(),
            vec![(0, crate::engine::Priority::High as i32)],
            "the job's own priority, not Normal, must reach set_priority"
        );
    }

    #[test]
    fn a_single_complete_job_runs_to_budget_and_reports_length() {
        let (tx, rx) = mpsc::channel();
        let (job, mut result_rx) = complete_job("hello", 3);
        tx.send(job).unwrap();
        drop(tx); // closes the channel once drained, so the loop can exit

        run_scheduled_jobs(
            FakeDriver::new(4),
            rx,
            no_control(),
            fake_encode,
            fake_decode,
        );

        let result = result_rx.try_recv().unwrap().expect("job should succeed");
        assert_eq!(result.finish_reason, FinishReason::Length);
        assert_eq!(result.completion_tokens, 3);
        assert_eq!(result.prompt_tokens, 5); // "hello".len()
    }

    #[test]
    fn a_single_complete_job_hitting_eos_reports_stop() {
        let (tx, rx) = mpsc::channel();
        // Prompt starting with 'E' (byte 69) selects the fake EOS plan.
        let (job, mut result_rx) = complete_job("Exyz", 10);
        tx.send(job).unwrap();
        drop(tx);

        run_scheduled_jobs(
            FakeDriver::new(4),
            rx,
            no_control(),
            fake_encode,
            fake_decode,
        );

        let result = result_rx.try_recv().unwrap().expect("job should succeed");
        assert_eq!(result.finish_reason, FinishReason::Stop);
        assert_eq!(result.completion_tokens, 1);
    }

    #[test]
    fn a_failed_session_sends_an_error_to_only_its_own_job() {
        let (tx, rx) = mpsc::channel();
        // 'F' (byte 70) selects the fake failure plan.
        let (good_job, mut good_rx) = complete_job("hello", 2);
        let (bad_job, mut bad_rx) = complete_job("Fxyz", 2);
        tx.send(good_job).unwrap();
        tx.send(bad_job).unwrap();
        drop(tx);

        run_scheduled_jobs(
            FakeDriver::new(4),
            rx,
            no_control(),
            fake_encode,
            fake_decode,
        );

        assert!(
            good_rx.try_recv().unwrap().is_ok(),
            "the healthy job must not be affected by the failing one"
        );
        assert!(bad_rx.try_recv().unwrap().is_err());
    }

    #[test]
    fn a_streaming_job_receives_each_token_then_done() {
        let (tx, rx) = mpsc::channel();
        let (job, mut stream_rx) = streaming_job("hi", 2);
        tx.send(job).unwrap();
        drop(tx);

        run_scheduled_jobs(
            FakeDriver::new(4),
            rx,
            no_control(),
            fake_encode,
            fake_decode,
        );

        let mut items = Vec::new();
        while let Ok(item) = stream_rx.try_recv() {
            items.push(item.unwrap());
        }
        assert_eq!(items.len(), 3, "2 tokens + 1 Done, got {items:?}");
        assert!(matches!(items[0], StreamItem::Token(_)));
        assert!(matches!(items[1], StreamItem::Token(_)));
        assert!(matches!(
            items[2],
            StreamItem::Done {
                finish_reason: FinishReason::Length,
                ..
            }
        ));
    }

    #[test]
    fn a_permanently_rejected_job_errors_immediately_instead_of_retrying_forever() {
        let (tx, rx) = mpsc::channel();
        // 'R' (byte 82) selects the fake permanent-rejection plan.
        let (bad_job, mut bad_rx) = complete_job("Rxyz", 2);
        tx.send(bad_job).unwrap();
        drop(tx);

        // If this regresses to re-queuing Rejected the same as NoCapacity,
        // the job never leaves `pending` (FakeDriver never admits it and
        // `has_active_sessions` never becomes true), so `rx.recv()` blocks
        // forever and this test hangs instead of failing — the exact bug
        // being guarded against.
        run_scheduled_jobs(
            FakeDriver::new(4),
            rx,
            no_control(),
            fake_encode,
            fake_decode,
        );

        assert!(
            bad_rx.try_recv().unwrap().is_err(),
            "a permanently-invalid prompt must be reported as a request \
             failure, not retried forever"
        );
    }

    #[test]
    fn a_job_beyond_capacity_waits_and_is_admitted_once_room_frees() {
        let (tx, rx) = mpsc::channel();
        // Capacity 1: the second job cannot be admitted until the first
        // finishes and is reaped.
        let (job1, mut rx1) = complete_job("hello", 1);
        let (job2, mut rx2) = complete_job("world", 1);
        tx.send(job1).unwrap();
        tx.send(job2).unwrap();
        drop(tx);

        run_scheduled_jobs(
            FakeDriver::new(1),
            rx,
            no_control(),
            fake_encode,
            fake_decode,
        );

        assert!(
            rx1.try_recv().unwrap().is_ok(),
            "first job must still complete"
        );
        assert!(
            rx2.try_recv().unwrap().is_ok(),
            "second job must be admitted once capacity frees, not dropped"
        );
    }

    #[test]
    fn two_concurrent_jobs_are_both_admitted_and_both_complete() {
        let (tx, rx) = mpsc::channel();
        let (job1, mut rx1) = complete_job("hello", 2);
        let (job2, mut rx2) = complete_job("world", 2);
        tx.send(job1).unwrap();
        tx.send(job2).unwrap();
        drop(tx);

        run_scheduled_jobs(
            FakeDriver::new(4),
            rx,
            no_control(),
            fake_encode,
            fake_decode,
        );

        assert!(rx1.try_recv().unwrap().is_ok());
        assert!(rx2.try_recv().unwrap().is_ok());
    }

    /// A control message must reach `SchedulerDriver::handle_control` even
    /// when the loop is otherwise idle -- the whole reason `run_scheduled_jobs`
    /// polls `ctrl_rx` instead of only ever checking it between job
    /// admissions. If the mutation under test here (`recv_timeout` reverted
    /// to a bare blocking `recv()` with the ctrl-check moved inside the
    /// idle branch instead of the top of the loop) ever regresses, this
    /// test hangs or fails rather than silently passing.
    struct ControlRecordingDriver {
        received: std::sync::Arc<std::sync::Mutex<Vec<ControlMsg>>>,
    }

    impl SchedulerDriver for ControlRecordingDriver {
        type Id = u64;

        fn try_admit(&mut self, _: &[u32], _: usize, _: f64) -> Result<u64, AdmitError> {
            Err(AdmitError::Rejected("not under test here".to_string()))
        }
        fn step(&mut self) -> Result<Vec<(u64, u32)>, String> {
            Ok(Vec::new())
        }
        fn reap_finished(&mut self) -> Vec<(u64, FinishOutcome)> {
            Vec::new()
        }
        fn has_active_sessions(&self) -> bool {
            false
        }
        fn handle_control(&mut self, msg: ControlMsg) {
            self.received.lock().unwrap().push(msg);
        }
    }

    #[test]
    fn a_control_message_is_handled_even_with_no_jobs_at_all() {
        // Sourcery (PR #206) correctly flagged the first version of this test:
        // dropping `job_tx` up front disconnects the job channel, and a
        // disconnected mpsc::Receiver::recv() returns immediately whether the
        // call is blocking or has a timeout -- so that version passed even
        // against a regression back to a blocking `rx.recv()`, since the
        // loop would exit the idle wait via the Disconnected arm right away
        // regardless. To actually exercise the 50ms-timeout idle wait, the
        // job channel must stay OPEN (sender alive, no job ever sent) while
        // the control message is observed; only after that does the test
        // drop the sender to let the loop end.
        let (job_tx, job_rx) = mpsc::channel();
        let (ctrl_tx, ctrl_rx) = mpsc::channel();

        let received = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let driver = ControlRecordingDriver {
            received: received.clone(),
        };

        let handle = std::thread::spawn(move || {
            run_scheduled_jobs(driver, job_rx, ctrl_rx, fake_encode, fake_decode);
        });

        ctrl_tx.send(ControlMsg::ClearPrefixCache).unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while received.lock().unwrap().is_empty() {
            assert!(
                std::time::Instant::now() < deadline,
                "control message was not handled within 5s while the job \
                 channel was still open and no job was ever sent -- a \
                 blocking rx.recv() would starve it indefinitely"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert_eq!(
            received.lock().unwrap().as_slice(),
            [ControlMsg::ClearPrefixCache],
            "a queued control message must be handled even when no inference \
             job is ever sent"
        );

        drop(ctrl_tx);
        drop(job_tx); // disconnect now so the loop can end
        handle.join().unwrap();
    }
}
