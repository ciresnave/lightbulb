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
//! `#![allow(dead_code)]`: this whole module is unreachable until the real
//! `SessionScheduler` adapter + `ModelRunner::start` wiring (a follow-up) call
//! into it — same shape as `src/model_fuel/engine_model.rs`'s own
//! `#[allow(dead_code)]` on `LoadedModel` pending its own wiring. Allowed here
//! at this one site, not by raising the gate's ceiling.
#![allow(dead_code)]

use std::sync::mpsc::Receiver;

use super::model_runner::{CompletionResult, FinishReason, InferenceJob, ResponseMode, StreamItem};

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

/// Abstracts a batched multi-session scheduler so [`run_scheduled_jobs`]'s
/// admission/dispatch logic is testable without a real model or GPU. The real
/// adapter wraps `fuel_inference::multi_session::SessionScheduler`.
pub(crate) trait SchedulerDriver {
    type Id: Copy + Eq + std::hash::Hash;

    /// Try to admit a new session. `Err` means "no capacity right now" — the
    /// caller must leave the job unconsumed and retry on a later tick, not
    /// drop it or report it as a request failure.
    fn try_admit(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        temperature: f64,
    ) -> Result<Self::Id, String>;

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
}

/// Drive `driver` against jobs arriving on `rx`, dispatching results to each
/// job's own response channel. Exits when `rx` disconnects AND no job is
/// pending admission or active in the driver — never while there is
/// outstanding work, so a dropped sender during a burst cannot silently
/// orphan an in-flight generation.
pub(crate) fn run_scheduled_jobs<D: SchedulerDriver>(
    mut driver: D,
    rx: Receiver<InferenceJob>,
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
        while let Ok(job) = rx.try_recv() {
            pending.push_back(job);
        }

        if pending.is_empty() && !driver.has_active_sessions() {
            if !channel_open {
                break;
            }
            match rx.recv() {
                Ok(job) => pending.push_back(job),
                Err(_) => {
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
            match driver.try_admit(&prompt_tokens, job.max_new_tokens, job.temperature) {
                Ok(id) => {
                    active.insert(
                        id,
                        JobMeta {
                            response_mode: job.response_mode,
                            prompt_tokens: prompt_tokens.len(),
                        },
                    );
                }
                Err(_capacity) => {
                    pending.push_front(job);
                    break;
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
                            let _ = tx.send(Ok(StreamItem::Done { finish_reason }));
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

    /// A fake prompt's first byte selects the fake session's behavior, so
    /// each test can drive a specific path without a real model.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum FakePlan {
        RunToBudget,
        HitEos,
        Fail,
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
    }

    impl FakeDriver {
        fn new(capacity: usize) -> Self {
            Self {
                capacity,
                next_id: 0,
                sessions: HashMap::new(),
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
        ) -> Result<u64, String> {
            if self.sessions.len() >= self.capacity {
                return Err("fake: at capacity".to_string());
            }
            let plan = match prompt.first() {
                Some(69) => FakePlan::HitEos, // 'E'
                Some(70) => FakePlan::Fail,   // 'F'
                _ => FakePlan::RunToBudget,
            };
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
                })
                .map(|(&id, _)| id)
                .collect();
            done_ids
                .into_iter()
                .map(|id| {
                    let s = self.sessions.remove(&id).unwrap();
                    let outcome = match s.plan {
                        FakePlan::Fail => FinishOutcome::Failed("fake failure".to_string()),
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
            response_mode: ResponseMode::Complete(tx),
        };
        (job, rx)
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
            response_mode: ResponseMode::Streaming(tx),
        };
        (job, rx)
    }

    #[test]
    fn a_single_complete_job_runs_to_budget_and_reports_length() {
        let (tx, rx) = mpsc::channel();
        let (job, mut result_rx) = complete_job("hello", 3);
        tx.send(job).unwrap();
        drop(tx); // closes the channel once drained, so the loop can exit

        run_scheduled_jobs(FakeDriver::new(4), rx, fake_encode, fake_decode);

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

        run_scheduled_jobs(FakeDriver::new(4), rx, fake_encode, fake_decode);

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

        run_scheduled_jobs(FakeDriver::new(4), rx, fake_encode, fake_decode);

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

        run_scheduled_jobs(FakeDriver::new(4), rx, fake_encode, fake_decode);

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
                finish_reason: FinishReason::Length
            }
        ));
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

        run_scheduled_jobs(FakeDriver::new(1), rx, fake_encode, fake_decode);

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

        run_scheduled_jobs(FakeDriver::new(4), rx, fake_encode, fake_decode);

        assert!(rx1.try_recv().unwrap().is_ok());
        assert!(rx2.try_recv().unwrap().is_ok());
    }
}
