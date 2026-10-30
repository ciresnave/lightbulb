// SPDX-License-Identifier: MIT OR Apache-2.0
//! Pre-load memory check (ROADMAP B9).
//!
//! Asks the serving backend whether a model's weights can fit, through Fuel's
//! `BackendRuntime` (`available_bytes` / `would_fit`), before the loader
//! allocates anything. The backend's honesty contract applies: "cannot
//! measure" is [`Preflight::Unknown`] and never refuses a load.

use fuel_backend_contract::backend::BackendRuntime;
use fuel_ir::backend::FitStatus;

/// What the backend says about an allocation of the weights' size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preflight {
    /// Fits, below the backend's pressure threshold.
    Fits,
    /// Fits, but leaves the device tight.
    Tight,
    /// Larger than the memory the backend reports available.
    WontFit {
        /// Bytes asked for.
        needed: u64,
        /// Bytes the backend reported available (0 if it stopped answering).
        available: u64,
    },
    /// The backend has no signal (or there is no backend runtime).
    Unknown,
}

/// Classify `weights_bytes` against `runtime`. `None` (a device with no
/// runtime impl) is [`Preflight::Unknown`].
pub fn check_with(weights_bytes: u64, runtime: Option<&dyn BackendRuntime>) -> Preflight {
    let Some(rt) = runtime else {
        return Preflight::Unknown;
    };
    match rt.would_fit(weights_bytes) {
        FitStatus::Comfortable => Preflight::Fits,
        FitStatus::Tight => Preflight::Tight,
        FitStatus::WontFit => Preflight::WontFit {
            needed: weights_bytes,
            available: rt.available_bytes().unwrap_or(0),
        },
        FitStatus::Unknown => Preflight::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Mock {
        available: Option<u64>,
        total: Option<u64>,
    }
    impl BackendRuntime for Mock {
        fn available_bytes(&self) -> Option<u64> {
            self.available
        }
        fn total_bytes(&self) -> Option<u64> {
            self.total
        }
    }

    #[test]
    fn weights_larger_than_available_will_not_fit_and_report_both_numbers() {
        let rt = Mock {
            available: Some(1_000),
            total: Some(10_000),
        };
        assert_eq!(
            check_with(1_001, Some(&rt)),
            Preflight::WontFit {
                needed: 1_001,
                available: 1_000
            }
        );
    }

    #[test]
    fn weights_well_under_available_fit() {
        let rt = Mock {
            available: Some(80),
            total: Some(100),
        };
        assert_eq!(check_with(10, Some(&rt)), Preflight::Fits);
    }

    #[test]
    fn weights_that_push_usage_past_the_threshold_are_tight_not_refused() {
        // total 100, available 80: 20 used. +50 => 70/100 = 0.7 -> Fits;
        // +70 => 90/100 = 0.9 > 0.85 -> Tight.
        let rt = Mock {
            available: Some(80),
            total: Some(100),
        };
        assert_eq!(check_with(70, Some(&rt)), Preflight::Tight);
    }

    #[test]
    fn a_backend_that_cannot_measure_is_unknown_never_wont_fit() {
        let rt = Mock {
            available: None,
            total: None,
        };
        assert_eq!(check_with(u64::MAX, Some(&rt)), Preflight::Unknown);
    }

    #[test]
    fn no_runtime_is_unknown() {
        assert_eq!(check_with(1, None), Preflight::Unknown);
    }
}
