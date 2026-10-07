//! Security alerting hook (security audit item 4).
//!
//! Fired from `auth_middleware::pre_auth_attempt_limiter_middleware` the
//! moment a client IP CROSSES `max_auth_attempts_per_minute_per_ip` — not
//! on every request after it, which would spam a sink once per second for
//! the rest of a sustained attack and bury the one event that actually
//! matters. [`just_crossed_threshold`] is the pure rule for "this is the
//! crossing, not a repeat."
//!
//! # What this is NOT
//!
//! [`EmailSink`] does not send email. It logs what it would send. Real
//! delivery (SMTP, a webhook, anything that reaches a person) is explicit
//! follow-up work for when lightbulb is actually deployed — PM ruling,
//! 2026-10-06: "the PR and docs must say plainly that alert DELIVERY is
//! not implemented ... so nobody counts it as alerting a person." Treat
//! this module as a place to attach a real transport later, not as one.

use std::net::IpAddr;
use std::sync::Arc;

/// True exactly when `count` is the FIRST request over `limit` for this
/// window — the moment a client crosses the threshold. `count` keeps
/// climbing on every subsequent request from a sustained attacker; this
/// stays false for all of them, so a sink is notified once per crossing,
/// not once per request.
pub fn just_crossed_threshold(count: i64, limit: i64) -> bool {
    count == limit + 1
}

/// A security-relevant event worth telling someone about.
#[derive(Debug, Clone)]
pub enum SecurityEvent {
    /// A client IP exceeded `max_auth_attempts_per_minute_per_ip`.
    RepeatedAuthFailures {
        client_ip: IpAddr,
        count: i64,
        limit: i64,
    },
}

/// Something that can be told about a [`SecurityEvent`].
///
/// `Send + Sync + 'static` because sinks live in `AppState` (cloned into
/// every request's handler/middleware stack) and are called from async
/// code.
pub trait SecuritySink: Send + Sync {
    fn notify(&self, event: &SecurityEvent);
}

/// Always-on sink: writes the event to the application's own log at
/// `error` level. This is the ONLY sink that actually tells anyone
/// anything today.
pub struct LogSink;

impl SecuritySink for LogSink {
    fn notify(&self, event: &SecurityEvent) {
        match event {
            SecurityEvent::RepeatedAuthFailures {
                client_ip,
                count,
                limit,
            } => {
                tracing::error!(
                    %client_ip,
                    count,
                    limit,
                    "SECURITY ALERT: client crossed the pre-auth attempt threshold"
                );
            }
        }
    }
}

/// STUB. See this module's doc comment — `notify` logs what it would send
/// and sends nothing. `to` is read from the `SECURITY_ALERT_EMAIL`
/// environment variable only (a deployment secret; never from a contact
/// file), threaded through `ApiConfig::security_alert_email`.
pub struct EmailSink {
    pub to: String,
}

impl EmailSink {
    /// The line `notify` logs, factored out so it's assertable in a test
    /// without a tracing subscriber.
    ///
    /// CIRESNAVE-CONTACT: the alert destination address lives only in the
    /// deployment secret (`SECURITY_ALERT_EMAIL`) — never in logs, PRs or
    /// docs. This MUST NOT reference `self.to`'s value anywhere; it states
    /// only that an address is configured, which is always true once an
    /// `EmailSink` exists at all (`build_sinks` only constructs one when
    /// `security_alert_email` is `Some`).
    fn log_line(&self, event: &SecurityEvent) -> String {
        format!(
            "EmailSink is a STUB: an alert address is configured and a security alert email \
             would be sent to it here, but delivery is not implemented. See \
             src/api/security_event.rs. event={event:?}"
        )
    }
}

impl SecuritySink for EmailSink {
    fn notify(&self, event: &SecurityEvent) {
        tracing::warn!("{}", self.log_line(event));
    }
}

/// Build the sink list for an `AppState`: `LogSink` always, plus an
/// `EmailSink` when `security_alert_email` is configured.
pub fn build_sinks(security_alert_email: &Option<String>) -> Vec<Arc<dyn SecuritySink>> {
    let mut sinks: Vec<Arc<dyn SecuritySink>> = vec![Arc::new(LogSink)];
    if let Some(to) = security_alert_email {
        sinks.push(Arc::new(EmailSink { to: to.clone() }));
    }
    sinks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn just_crossed_threshold_is_true_at_exactly_limit_plus_one() {
        assert!(just_crossed_threshold(21, 20));
    }

    #[test]
    fn just_crossed_threshold_is_false_at_the_limit_itself() {
        assert!(!just_crossed_threshold(20, 20));
    }

    #[test]
    fn just_crossed_threshold_is_false_well_under_the_limit() {
        assert!(!just_crossed_threshold(5, 20));
    }

    #[test]
    fn just_crossed_threshold_is_false_for_every_request_after_the_crossing() {
        // A sustained attacker keeps incrementing past the crossing point;
        // none of those later counts should re-trigger it.
        for count in 22..30 {
            assert!(
                !just_crossed_threshold(count, 20),
                "count {count} must not re-trigger the crossing"
            );
        }
    }

    struct RecordingSink {
        events: std::sync::Mutex<Vec<SecurityEvent>>,
    }

    impl RecordingSink {
        fn new() -> Self {
            Self {
                events: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl SecuritySink for RecordingSink {
        fn notify(&self, event: &SecurityEvent) {
            self.events.lock().unwrap().push(event.clone());
        }
    }

    #[test]
    fn build_sinks_includes_log_sink_only_when_no_email_is_configured() {
        let sinks = build_sinks(&None);
        assert_eq!(sinks.len(), 1);
    }

    #[test]
    fn build_sinks_includes_both_sinks_when_email_is_configured() {
        let sinks = build_sinks(&Some("ops@example.com".to_string()));
        assert_eq!(sinks.len(), 2);
    }

    #[test]
    fn a_recording_sink_receives_the_dispatched_event() {
        let sink = RecordingSink::new();
        let event = SecurityEvent::RepeatedAuthFailures {
            client_ip: "1.2.3.4".parse().unwrap(),
            count: 21,
            limit: 20,
        };

        sink.notify(&event);

        let events = sink.events.lock().unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            SecurityEvent::RepeatedAuthFailures { count, limit, .. } => {
                assert_eq!(*count, 21);
                assert_eq!(*limit, 20);
            }
        }
    }

    /// CIRESNAVE-CONTACT: the alert destination address lives only in the
    /// deployment secret — never in logs, PRs or docs. PM review of #128
    /// (2026-10-07) caught `notify` logging `to = %self.to` directly, which
    /// put the configured address into the application's own log output.
    ///
    /// Tests `EmailSink::log_line` (the pure string-building half of
    /// `notify`, factored out so this is assertable without a tracing
    /// subscriber) rather than `notify` itself.
    #[test]
    fn email_sink_log_line_never_contains_the_configured_address() {
        let sink = EmailSink {
            to: "ops@example.com".to_string(),
        };
        let event = SecurityEvent::RepeatedAuthFailures {
            client_ip: "1.2.3.4".parse().unwrap(),
            count: 21,
            limit: 20,
        };

        let line = sink.log_line(&event);

        assert!(
            !line.contains("ops@example.com"),
            "log line must never contain the configured alert address: {line:?}"
        );

        // Positive control: prove the assertion above would actually catch
        // a leak — this is the shape the line had BEFORE this fix
        // (`to = %self.to` interpolated straight into the message), and the
        // same `.contains()` check must flag it.
        let old_style_line = format!("to={} would send an alert here", sink.to);
        assert!(
            old_style_line.contains("ops@example.com"),
            "control itself is broken: it should contain the address"
        );
    }
}
