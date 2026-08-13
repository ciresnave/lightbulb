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
    /// `window` is clamped to at least 1: a zero-length window would divide by
    /// zero and report `NaN`, and `NaN < min` is `false`, so the monitor would
    /// go permanently and silently quiet rather than fail visibly.
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
    pub fn should_warn(&self) -> bool {
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
