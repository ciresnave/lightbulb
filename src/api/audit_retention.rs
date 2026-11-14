// SPDX-License-Identifier: MIT OR Apache-2.0
//! Audit-log retention (ROADMAP A7): `LIGHTBULB_AUDIT_RETENTION_DAYS=N` deletes `audit_logs` rows older than N days,
//! once an hour. Unset keeps every row, as before.

use persistant_relational::{RelationalBackend, Value};

/// Parse `LIGHTBULB_AUDIT_RETENTION_DAYS`: blank means keep everything, otherwise a whole number of days of at least 1.
/// Zero is refused rather than read as "delete everything", and so is anything that is not a number.
pub fn parse_retention_days(raw: &str) -> Result<Option<u32>, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(None);
    }
    match raw.parse::<u32>() {
        Ok(0) => Err("must be at least 1 (unset the variable to keep every row)".to_string()),
        Ok(days) => Ok(Some(days)),
        Err(_) => Err(format!("{raw:?} is not a whole number of days")),
    }
}

const DELETE_OLD: &str =
    "DELETE FROM audit_logs WHERE created_at < NOW() - ($1::int * INTERVAL '1 day')";

/// Start the hourly purge. One background task, so the request path pays nothing for it.
pub(crate) fn spawn_cleanup(backend: Option<persistant_sqlx::PgBackend>, days: Option<u32>) {
    let (Some(backend), Some(days)) = (backend, days) else {
        return;
    };
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600));
        loop {
            interval.tick().await;
            if let Err(e) = backend
                .execute(DELETE_OLD, &[Value::Int(i64::from(days))])
                .await
            {
                tracing::warn!(error = %e, "audit_logs retention purge failed");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_keeps_everything_and_a_number_is_days() {
        assert_eq!(parse_retention_days(""), Ok(None));
        assert_eq!(parse_retention_days("  "), Ok(None));
        assert_eq!(parse_retention_days(" 90 "), Ok(Some(90)));
    }

    #[test]
    fn zero_negative_and_words_are_refused() {
        for bad in ["0", "-5", "ninety", "1.5", "30d"] {
            assert!(
                parse_retention_days(bad).is_err(),
                "{bad:?} must be refused"
            );
        }
    }

    #[test]
    fn the_purge_only_deletes_audit_rows_older_than_the_bound() {
        assert!(DELETE_OLD.contains("FROM audit_logs"));
        assert!(DELETE_OLD.contains("created_at < NOW()"));
        assert!(!DELETE_OLD.contains("auth_"));
    }
}
