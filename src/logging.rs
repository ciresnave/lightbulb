// SPDX-License-Identifier: MIT OR Apache-2.0
//! Shared tracing-subscriber initialization for every binary in this crate.
//!
//! Extracted from `main.rs` after Sourcery's review of the JSON-logging PR
//! (#154) found `lightbulb-probe` initializing its own, separate subscriber
//! that ignored `LIGHTBULB_LOG_FORMAT` entirely -- a second copy of this
//! logic is a second thing to drift, which is exactly what happened the
//! first time there were two.

use std::io::IsTerminal;
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
        // ANSI colour only when the log stream is a terminal and `NO_COLOR` is
        // unset: piped or captured output must not carry escape codes
        // (CodeRipper OPS-011). The fmt layer writes to stdout.
        let ansi = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        Box::new(tracing_subscriber::fmt::layer().with_ansi(ansi))
    };

    registry.with(fmt_layer).init();
}

/// Placeholder logged when a database URL cannot be shown safely.
pub const REDACTED: &str = "<redacted>";

/// A database URL that is safe to log: `scheme://host[:port]/path` only.
///
/// Userinfo (user and password, percent-encoded or not) and the query string
/// and fragment are dropped -- `?password=` and `?sslpassword=` are valid
/// connection-string parameters, so a URL without userinfo can still carry a
/// secret. Anything that does not parse as a URL with a host, including a
/// string that merely looks like `user:pass@host/db`, becomes `<redacted>`
/// and is never echoed. To be handed to RedactAnt when it exists (ROADMAP B16).
pub fn redact_database_url(raw: &str) -> String {
    let Ok(url) = url::Url::parse(raw.trim()) else {
        return REDACTED.to_string();
    };
    let Some(host) = url.host_str() else {
        return REDACTED.to_string();
    };
    let port = url.port().map(|p| format!(":{p}")).unwrap_or_default();
    format!("{}://{}{}{}", url.scheme(), host, port, url.path())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_scheme_host_port_and_database_only() {
        assert_eq!(
            redact_database_url("postgres://user:hunter2@db.example.com:5432/lightbulb"),
            "postgres://db.example.com:5432/lightbulb"
        );
    }

    #[test]
    fn url_without_userinfo_or_port_is_unchanged() {
        assert_eq!(
            redact_database_url("postgres://localhost/lightbulb"),
            "postgres://localhost/lightbulb"
        );
    }

    #[test]
    fn userinfo_without_password_is_dropped_too() {
        assert_eq!(
            redact_database_url("postgres://admin@h:5432/d"),
            "postgres://h:5432/d"
        );
    }

    #[test]
    fn percent_encoded_password_is_dropped() {
        let out = redact_database_url("postgres://u:p%40ss%3Aw0rd@h:5432/d");
        assert_eq!(out, "postgres://h:5432/d");
        assert!(!out.contains("ss") && !out.contains('%'));
    }

    #[test]
    fn query_string_secrets_are_dropped() {
        for q in [
            "?password=hunter2",
            "?sslpassword=hunter2&sslmode=require",
            "?user=u&password=hunter2#frag",
        ] {
            let out = redact_database_url(&format!("postgres://h:5432/d{q}"));
            assert_eq!(out, "postgres://h:5432/d", "{q}");
        }
    }

    #[test]
    fn ipv6_host_keeps_its_brackets() {
        assert_eq!(
            redact_database_url("postgres://u:pw@[::1]:5432/d"),
            "postgres://[::1]:5432/d"
        );
    }

    #[test]
    fn unparseable_input_is_fully_redacted_and_never_echoed() {
        for bad in [
            "",
            "   ",
            "not a url",
            "user:hunter2@host:5432/db",
            "postgres://",
            "postgres:///db?password=hunter2",
            "://hunter2",
        ] {
            let out = redact_database_url(bad);
            assert_eq!(out, REDACTED, "{bad:?}");
        }
    }

    #[test]
    fn multi_host_urls_never_leak_the_password() {
        let out = redact_database_url("postgres://u:hunter2@h1:5432,h2:5433/d");
        assert!(!out.contains("hunter2") && !out.contains('@'), "{out}");
    }

    #[test]
    fn surrounding_whitespace_is_ignored() {
        assert_eq!(
            redact_database_url("  postgres://u:pw@h/d \n"),
            "postgres://h/d"
        );
    }
}
