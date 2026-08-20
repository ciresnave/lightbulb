//! Rolling EOS-fire rate, as a signal that a heuristically-chosen chat
//! template is wrong.
//!
//! A correctly-templated chat model answering short questions terminates
//! nearly always. With the legacy `role: content` join, TinyLlama-1.1B-Chat
//! emitted EOS in 1 of 6 trials — it behaved like a base model and continued
//! text until the token budget ran out. `finish_reason` is therefore the
//! cheapest observable that discriminates "this model is being prompted with
//! its own template" from "this model is being prompted with something else",
//! and it is already computed for every completion the server returns.
//!
//! # This observes and never acts
//!
//! Nothing here changes what gets prompted, rendered or retried, and nothing
//! may be added that does. The rejected alternative — re-resolving the template
//! when the rate drops, or retrying a request under a different candidate — is
//! rejected on debuggability, not on cost (spec §3): a server that changes its
//! prompting mid-flight cannot be reasoned about from a request/response pair,
//! whereas one that is consistently wrong and says so can be. The only output
//! of this module is a number and a log line.
//!
//! # The constants are one model's measurement, not a calibration
//!
//! [`DEFAULT_WINDOW`] and [`DEFAULT_MIN_STOP_RATE`] come from the TinyLlama
//! figure above and nothing else. If the warning proves noisy, raise the window
//! before lowering the rate: the failure mode of a too-eager warning is a log
//! line pointing at the wrong subsystem, which costs more than a late one.

use std::collections::VecDeque;

use parking_lot::Mutex;

use crate::engine::model_runner::FinishReason;

/// How many completions a reading is computed from.
pub const DEFAULT_WINDOW: usize = 20;

/// Below this fraction of EOS-terminated completions, the template is suspect.
pub const DEFAULT_MIN_STOP_RATE: f64 = 0.25;

/// The samples, plus whether the last full-window reading was already below the
/// threshold.
///
/// The edge flag lives *inside* the mutex with the samples rather than beside
/// it in an `AtomicBool`, so that "push a sample, read the rate, decide whether
/// this is a new crossing" is one atomic step. Split across two locks, two
/// concurrent requests can both observe the same crossing and both log it,
/// which is the noise this edge-triggering exists to remove.
#[derive(Debug)]
struct Samples {
    recent: VecDeque<bool>,
    below: bool,
}

/// Rolling EOS-rate counter over the last `window` completions.
#[derive(Debug)]
pub struct EosMonitor {
    window: usize,
    min_stop_rate: f64,
    samples: Mutex<Samples>,
}

impl EosMonitor {
    /// `window` is clamped to at least 1.
    ///
    /// **Not** to avoid a divide by zero. The private `rate` helper divides by
    /// `recent.len()`, never by `window`, so a zero window yields an ordinary
    /// finite rate: probed with the clamp removed, `EosMonitor::new(0, 0.25)`
    /// reports `stop_rate() == Some(0.333…)` after three samples and
    /// `is_nan() == false`. This doc claimed the `NaN` reading for one commit
    /// and it was never reachable.
    ///
    /// The real hazard is the eviction guard in [`Self::record`]:
    /// `recent.len() == self.window` is true only *before* the first push when
    /// `window` is 0, so nothing is ever popped. The deque grows without bound
    /// for the life of the process and the reading silently stops being a
    /// rolling one — it becomes the server's lifetime EOS rate, which no
    /// recovery can pull back up and no fresh degradation can pull down.
    /// `a_zero_window_is_a_one_wide_window_not_an_unbounded_one` is the guard.
    pub fn new(window: usize, min_stop_rate: f64) -> Self {
        let window = window.max(1);
        Self {
            window,
            min_stop_rate,
            samples: Mutex::new(Samples {
                recent: VecDeque::with_capacity(window),
                below: false,
            }),
        }
    }

    /// How many completions a reading covers. Exposed so the warning can name
    /// the window it was actually built with — quoting [`DEFAULT_WINDOW`] in
    /// the message would misreport any monitor constructed with another.
    ///
    /// That is this accessor's only production caller, and for one commit the
    /// difference between it and the constant was invisible: every test asserted
    /// `stop_rate()` and none read the log. `api::openai::chat::tests::
    /// the_warning_reports_the_monitors_own_window_not_the_default` now reads it,
    /// through two monitors of different widths.
    pub fn window(&self) -> usize {
        self.window
    }

    /// Record how one completion ended.
    ///
    /// Returns `Some(rate)` **only on the transition** into the below-threshold
    /// state, and `None` on every other call — including every subsequent
    /// completion while the rate stays low. A level-triggered signal here would
    /// put one warning in the log per request for as long as the condition
    /// lasts, burying the render failures in `api::openai::chat` that are
    /// genuinely per-request. The state resets when a reading comes back to or
    /// above the threshold, so a server that recovers and degrades again warns
    /// again.
    ///
    /// Takes `&self`: `AppState` is cloned per request, so every handler holds
    /// its own clone of the same `Arc<EosMonitor>`.
    pub fn record(&self, reason: FinishReason) -> Option<f64> {
        let mut s = self.samples.lock();
        if s.recent.len() == self.window {
            s.recent.pop_front();
        }
        s.recent.push_back(matches!(reason, FinishReason::Stop));

        // `?` before touching `below`: while the window is filling there is no
        // reading, so there is nothing to have crossed.
        let rate = Self::rate(&s.recent, self.window)?;
        let below = rate < self.min_stop_rate;
        let crossed = below && !s.below;
        s.below = below;
        crossed.then_some(rate)
    }

    /// The fraction of the last `window` completions that stopped on EOS, or
    /// `None` while fewer than `window` have been seen.
    ///
    /// `None` rather than a rate over however many samples exist: a cold server
    /// that has answered three requests has no evidence about its template, and
    /// a warning drawn from three samples points the reader at the wrong
    /// subsystem at exactly the moment they know least about the system.
    pub fn stop_rate(&self) -> Option<f64> {
        Self::rate(&self.samples.lock().recent, self.window)
    }

    /// Whether the current reading is below the threshold.
    ///
    /// Level-triggered and side-effect free, unlike [`Self::record`]'s return
    /// value. `false` while the window is filling, for the reason
    /// [`Self::stop_rate`] returns `None` there.
    ///
    /// **Test-only, and compiled out of the library on purpose.** It shipped
    /// `pub` with zero production callers, which reads as an oversight; it is
    /// not one, and `#[cfg(test)]` is how that is said in a way that stays true.
    /// The server's one decision about the rate is [`Self::record`]'s edge. A
    /// level-triggered read of the same state is what a caller reaches for when
    /// it wants to *act* on the rate — re-resolve, retry, downgrade — which is
    /// the design this module rejects (see the module docs), so a production
    /// caller appearing here is a change of design and now has to say so by
    /// deleting this attribute rather than by adding a line.
    ///
    /// The tests that need it: the threshold-boundary case below, and
    /// `api::openai::chat::tests`' two never-acts tests, which must establish
    /// that the server is in the warning state before asserting that being in it
    /// changed nothing.
    #[cfg(test)]
    pub(crate) fn should_warn(&self) -> bool {
        self.stop_rate().is_some_and(|r| r < self.min_stop_rate)
    }

    fn rate(recent: &VecDeque<bool>, window: usize) -> Option<f64> {
        if recent.len() < window {
            return None;
        }
        Some(recent.iter().filter(|stopped| **stopped).count() as f64 / recent.len() as f64)
    }
}

impl Default for EosMonitor {
    fn default() -> Self {
        Self::new(DEFAULT_WINDOW, DEFAULT_MIN_STOP_RATE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped constants, pinned as values.
    ///
    /// Every other test in this module hands `new` a window of its own, and the
    /// one that uses twenty writes it as a literal — so for one commit
    /// `DEFAULT_WINDOW = 1_000_000` (a server that can never produce a reading,
    /// let alone a warning) and `DEFAULT_MIN_STOP_RATE = 0.9` (a warning on
    /// every healthy server whose EOS rate is not near-perfect) both left the
    /// whole suite green. Only `0.0` was caught, and only incidentally.
    ///
    /// These are one model's measurement — TinyLlama-1.1B-Chat, 1 EOS in 6 under
    /// the legacy join — not a calibration. Changing them is allowed; changing
    /// them *silently*, as a typo or an "obvious" tightening, is what this
    /// catches. Raise the window before lowering the rate, per the module docs.
    #[test]
    fn the_shipped_defaults_are_the_measured_ones() {
        assert_eq!(
            DEFAULT_WINDOW, 20,
            "the default window changed; a wide one delays every warning and a \
             very wide one silences it outright on a server that never sees that \
             many completions"
        );
        assert_eq!(
            DEFAULT_MIN_STOP_RATE, 0.25,
            "the default threshold changed; above it, ordinary servers whose \
             clients ask for long answers warn about their chat template"
        );
    }

    /// The configuration the server actually ships, exercised through
    /// [`EosMonitor::default`] rather than a hand-built window.
    ///
    /// `ApiServer::new` is the sole production construction and builds exactly
    /// this monitor; its only test coverage is `tests/api_integration_tests.rs`,
    /// which is `#[ignore]`d wholesale for wanting PostgreSQL. So the shipped
    /// configuration was, until this test, exercised by nothing at all.
    ///
    /// The assertions are behavioural rather than a second reading of the
    /// constants: a reading appears at the twentieth completion and not the
    /// nineteenth (pinning the width), 5 in 20 passes and 4 in 20 warns
    /// (pinning the threshold from both sides — a threshold above 0.25 fails the
    /// first, one at or below 0.2 fails the second).
    #[test]
    fn the_default_monitor_reads_twenty_completions_at_a_quarter() {
        let m = EosMonitor::default();

        for _ in 0..5 {
            m.record(FinishReason::Stop);
        }
        for i in 0..14 {
            assert_eq!(
                m.record(FinishReason::Length),
                None,
                "sample {} of the default window produced a crossing",
                i + 6
            );
        }
        assert_eq!(
            m.stop_rate(),
            None,
            "nineteen completions produced a reading from the default monitor, \
             so its window is narrower than DEFAULT_WINDOW claims"
        );

        assert_eq!(
            m.record(FinishReason::Length),
            None,
            "the twentieth completion sits exactly ON the default threshold, \
             which is not below it"
        );
        assert_eq!(
            m.stop_rate(),
            Some(0.25),
            "twenty completions did not produce a reading, so the default \
             monitor cannot warn on any server that reaches twenty requests"
        );
        assert!(
            !m.should_warn(),
            "5 EOS in 20 is exactly the default threshold and must pass; a \
             default above it warns about the chat template on healthy servers"
        );

        // A twenty-first Length ages the oldest Stop out: 4 in 20, below.
        assert_eq!(
            m.record(FinishReason::Length),
            Some(0.2),
            "the reading that first falls below the default threshold was not \
             reported"
        );
        assert!(m.should_warn());
    }

    /// A zero window is a one-wide window, not an unbounded one.
    ///
    /// The clamp in [`EosMonitor::new`] was untested, and the rationale written
    /// above it was false: removing it produces no `NaN` and no divide by zero,
    /// because [`EosMonitor::rate`] divides by `recent.len()`. What it produces
    /// is an eviction guard (`recent.len() == self.window`) that is true only
    /// before the first push, so the deque grows for the life of the process and
    /// the "rolling" rate becomes the lifetime one. That is the failure this
    /// asserts: with the clamp the second completion displaces the first and the
    /// rate is 0.0; without it, both samples are still there and the rate is 0.5
    /// — which is above the threshold, so the monitor also stays quiet.
    #[test]
    fn a_zero_window_is_a_one_wide_window_not_an_unbounded_one() {
        let m = EosMonitor::new(0, DEFAULT_MIN_STOP_RATE);
        assert_eq!(m.window(), 1, "the window was not clamped");

        assert_eq!(
            m.record(FinishReason::Stop),
            None,
            "1 of 1 is not below the threshold"
        );
        assert_eq!(
            m.stop_rate(),
            Some(1.0),
            "one completion is a full reading of a one-wide window"
        );

        assert_eq!(
            m.record(FinishReason::Length),
            Some(0.0),
            "the first sample never aged out: the window is growing instead of \
             sliding, so this reads the server's lifetime rate (0.5 here) and \
             the monitor is quiet"
        );
        assert_eq!(m.stop_rate(), Some(0.0));
    }

    /// One crossing produces one report, however many requests observe it at
    /// once.
    ///
    /// This is the property the `Samples` doc asserts and nothing defended:
    /// push, read and the `below` flag are one atomic step *specifically* so
    /// that concurrent requests cannot both see the same crossing. Splitting the
    /// check-and-set across two lock acquisitions — the design that doc
    /// explicitly rejects — left every other test in the repo green.
    ///
    /// # Shape
    ///
    /// Each trial has its own monitor, **warmed to one sample short of a
    /// reading** with nothing but `Length`. So the first concurrent completion
    /// to land completes the window below the threshold, and so does every one
    /// after it (each pops a `Length` and pushes another). That warm-up is what
    /// makes the race reachable at all, and the numbers are not close. Against
    /// the split-lock version, on this box:
    ///
    ///   8 threads, 200 trials, window 8, started empty     0 duplicates (green)
    ///   8 threads, 200 trials, window 4, warmed            2 duplicates
    ///  24 threads, 300 trials, window 4, warmed            5 duplicates
    ///  24 threads, 2 000 trials, window 4, warmed         45 duplicates
    ///
    /// The first row is the obvious probe — fill the window with the threads
    /// themselves — and it does not work: only the completion that *lands* on the
    /// boundary produces a reading, every other thread returns on the `?` before
    /// touching `below`, and the broken implementation passes. Warmed, all
    /// THREADS completions reach the decision.
    ///
    /// The second row is the reason for the third and fourth: two hits in a run
    /// is a coin flip dressed as a test.
    ///
    /// The threads are spawned once and walk the trials together through a
    /// reused barrier, rather than being spawned per trial: spawning dominated
    /// the runtime and capped the trial count, and the barrier releases the
    /// threads far closer together than 24 `spawn` calls do.
    ///
    /// # Why it cannot flake green-when-broken, or red-when-correct
    ///
    /// The discrimination is one-sided. A correct monitor reports exactly one
    /// crossing per trial *deterministically* — the flag is read and set under
    /// one lock, so whichever completion wins the lock first is the only one
    /// that can see `below == false` — so a red here is never chance. Broken,
    /// each trial is an independent chance to duplicate and the test fails if
    /// **any** trial does, so the false-green probability falls off as
    /// `e^-(duplicates)`: at the measured 45 in 2 000, roughly `e^-45`. A box
    /// with fewer cores hits it less often, which is what the trial count buys.
    ///
    /// A milder split (push separated from the read, but the flag's
    /// read-modify-write kept under a single lock) does *not* double-report and
    /// this test passes it. That is the boundary that matters, and it is why the
    /// `Samples` doc names the read-and-set rather than the whole body.
    #[test]
    fn one_crossing_is_reported_once_however_many_requests_observe_it() {
        use std::sync::Arc;
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};

        const THREADS: usize = 24;
        const TRIALS: usize = 2_000;
        const WINDOW: usize = 4;

        let monitors: Arc<Vec<EosMonitor>> = Arc::new(
            (0..TRIALS)
                .map(|_| {
                    let m = EosMonitor::new(WINDOW, DEFAULT_MIN_STOP_RATE);
                    for _ in 0..WINDOW - 1 {
                        assert_eq!(
                            m.record(FinishReason::Length),
                            None,
                            "the warm-up produced a reading, so the window is \
                             not the width this trial assumes"
                        );
                    }
                    m
                })
                .collect(),
        );
        let reports: Arc<Vec<AtomicUsize>> =
            Arc::new((0..TRIALS).map(|_| AtomicUsize::new(0)).collect());
        let gate = Arc::new(Barrier::new(THREADS));

        let threads: Vec<_> = (0..THREADS)
            .map(|_| {
                let monitors = Arc::clone(&monitors);
                let reports = Arc::clone(&reports);
                let gate = Arc::clone(&gate);
                std::thread::spawn(move || {
                    for (monitor, reported) in monitors.iter().zip(reports.iter()) {
                        // Every thread waits here, so one trial's completions
                        // land as close to simultaneously as the OS allows.
                        gate.wait();
                        if monitor.record(FinishReason::Length).is_some() {
                            reported.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("a recording thread panicked");
        }

        let duplicated: Vec<(usize, usize)> = reports
            .iter()
            .map(|n| n.load(Ordering::Relaxed))
            .enumerate()
            .filter(|(_, reports)| *reports != 1)
            .collect();
        assert!(
            duplicated.is_empty(),
            "{} of {TRIALS} trials did not report exactly one crossing \
             (trial, reports): {duplicated:?}. Concurrent completions saw the \
             same crossing because the push, the read and the `below` flag are \
             no longer one atomic step",
            duplicated.len()
        );
    }

    /// Below the window there is no reading at all — not a reading drawn from
    /// too few samples.
    ///
    /// The assertion above the boundary is an equality against a rate derived
    /// by hand (one `Stop` in twenty), not `is_some()`: an implementation that
    /// returned a constant, or one that reported `1/19` at 19 samples and kept
    /// reporting it, satisfies `is_none()`-then-`is_some()`.
    #[test]
    fn reports_nothing_until_the_window_fills() {
        let m = EosMonitor::new(20, DEFAULT_MIN_STOP_RATE);

        m.record(FinishReason::Stop);
        for _ in 0..18 {
            m.record(FinishReason::Length);
        }
        assert_eq!(
            m.stop_rate(),
            None,
            "19 of 20 samples produced a reading; a rate computed from a partial \
             window is what makes a cold server warn about the wrong subsystem"
        );

        m.record(FinishReason::Length);
        assert_eq!(
            m.stop_rate(),
            Some(1.0 / 20.0),
            "the twentieth completion must yield the rate over exactly those \
             twenty: one Stop, nineteen Length"
        );
    }

    /// The window slides one sample at a time: old completions age out, or a
    /// server that recovers never stops being reported as broken.
    #[test]
    fn the_window_slides_one_sample_at_a_time() {
        let m = EosMonitor::new(4, DEFAULT_MIN_STOP_RATE);
        for _ in 0..4 {
            m.record(FinishReason::Length);
        }
        assert_eq!(m.stop_rate(), Some(0.0));

        m.record(FinishReason::Stop);
        assert_eq!(
            m.stop_rate(),
            Some(1.0 / 4.0),
            "the fifth sample must displace the first; 1/5 here means the \
             window grew instead of sliding"
        );

        for _ in 0..3 {
            m.record(FinishReason::Stop);
        }
        assert_eq!(
            m.stop_rate(),
            Some(1.0),
            "the four Length samples never aged out — eight samples in a \
             four-wide window read 4/8"
        );
    }

    /// A rate exactly at the threshold passes. Stated as a test because `<` and
    /// `<=` are one character apart and the difference is a warning on every
    /// server whose real rate sits on the constant.
    #[test]
    fn the_threshold_boundary_passes_at_exactly_the_minimum() {
        let at = EosMonitor::new(4, 0.25);
        at.record(FinishReason::Stop);
        for _ in 0..3 {
            at.record(FinishReason::Length);
        }
        assert_eq!(at.stop_rate(), Some(0.25));
        assert!(
            !at.should_warn(),
            "0.25 is not BELOW a 0.25 threshold, so this reading must pass"
        );

        let below = EosMonitor::new(4, 0.25);
        for _ in 0..4 {
            below.record(FinishReason::Length);
        }
        assert_eq!(below.stop_rate(), Some(0.0));
        assert!(
            below.should_warn(),
            "0 of 4 completions stopped on EOS and the monitor stayed quiet"
        );
    }

    /// A monitor that has not filled its window says nothing, however bad the
    /// samples it does have look.
    #[test]
    fn a_cold_monitor_never_warns() {
        let m = EosMonitor::new(4, DEFAULT_MIN_STOP_RATE);
        for i in 0..3 {
            assert_eq!(
                m.record(FinishReason::Length),
                None,
                "sample {i} of a four-wide window produced a crossing"
            );
        }
        assert_eq!(m.stop_rate(), None);
        assert!(
            !m.should_warn(),
            "warned from three samples — `is_some_and` degraded to `map_or(true, …)`"
        );

        // The fourth completes the window and is the first legitimate reading.
        assert_eq!(m.record(FinishReason::Length), Some(0.0));
        assert!(m.should_warn());
    }

    /// The warning fires once per crossing, not once per completion — and
    /// fires again after a recovery.
    ///
    /// Each `assert_eq!` here is on the exact `Option<f64>`, so a
    /// level-triggered implementation (third call returns `Some`), a latching
    /// one (the last call returns `None`), and one that never resets are all
    /// distinguished.
    #[test]
    fn the_warning_is_edge_triggered() {
        let m = EosMonitor::new(2, 0.5);

        assert_eq!(
            m.record(FinishReason::Length),
            None,
            "the window is not full"
        );
        assert_eq!(
            m.record(FinishReason::Length),
            Some(0.0),
            "the reading that first crosses the threshold must be reported"
        );
        assert_eq!(
            m.record(FinishReason::Length),
            None,
            "the same crossing was reported twice — one warning per completion \
             for as long as the condition lasts buries the per-request render \
             failures in the same log"
        );

        // Recover: 1/2 is not below 0.5, so the monitor is healthy again.
        assert_eq!(m.record(FinishReason::Stop), None);
        assert_eq!(m.record(FinishReason::Stop), None);
        assert_eq!(m.stop_rate(), Some(1.0));

        assert_eq!(m.record(FinishReason::Length), None, "1/2 is not below 0.5");
        assert_eq!(
            m.record(FinishReason::Length),
            Some(0.0),
            "a server that recovered and degraded again never said so"
        );
    }
}
