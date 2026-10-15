//! Shared tracing-subscriber initialization for every binary in this crate.
//!
//! Extracted from `main.rs` after Sourcery's review of the JSON-logging PR
//! (#154) found `lightbulb-probe` initializing its own, separate subscriber
//! that ignored `LIGHTBULB_LOG_FORMAT` entirely -- a second copy of this
//! logic is a second thing to drift, which is exactly what happened the
//! first time there were two.

use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize the global tracing subscriber. Plain-text by default
/// (`LIGHTBULB_LOG_FORMAT` unset or anything other than `"json"`) -- no
/// existing deployment's log format changes under it. `fmt::layer()` and
/// `fmt::layer().json()` are different concrete types (different event
/// formatters), so they can't both feed `.with()` directly; boxed as `dyn
/// Layer` to pick one at runtime instead of compile time. The trait
/// object's subscriber type parameter must be the type `.with()`
/// accumulates so far (env-filter over the base registry), not the bare
/// `Registry`, or it fails to compile with an unsatisfied-trait-bounds
/// error.
///
/// `default_env_filter` is the `EnvFilter` directive used when `RUST_LOG`
/// (or whichever variable `EnvFilter::try_from_default_env` reads) isn't
/// set -- callers differ here today (the server defaults wider than the
/// probe did), so it stays a parameter rather than a shared constant.
pub fn init_tracing(default_env_filter: &str) {
    let json_logs = std::env::var("LIGHTBULB_LOG_FORMAT").is_ok_and(|v| v == "json");
    let registry = tracing_subscriber::registry().with(
        tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| default_env_filter.into()),
    );
    let fmt_layer: Box<
        dyn Layer<
                tracing_subscriber::layer::Layered<
                    tracing_subscriber::EnvFilter,
                    tracing_subscriber::Registry,
                >,
            > + Send
            + Sync,
    > = if json_logs {
        Box::new(tracing_subscriber::fmt::layer().json())
    } else {
        Box::new(tracing_subscriber::fmt::layer())
    };

    registry.with(fmt_layer).init();
}
