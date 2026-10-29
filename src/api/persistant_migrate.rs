// SPDX-License-Identifier: MIT OR Apache-2.0
//! PersistAnt adoption: the 6 `migrations/V{n}__*.sql` files ported to
//! `persistant_relational::Migration`, run via `persistant_sqlx::PgBackend`.
//! Step 1 proved this path non-destructive alongside the (then still
//! present) `refinery` runner; Step 5 made it the sole migration path in
//! `src/api/mod.rs::ApiServer::new` -- `refinery`/`deadpool_postgres`/
//! `tokio_postgres` are gone from this repo's dependency tree.
//!
//! `migrations/*.sql` themselves are untouched and still exist in the repo
//! (refinery just no longer runs them) -- kept for provenance/history, not
//! because anything reads them anymore.
//!
//! ## Why the SQL text below is not byte-identical to the `.sql` files
//!
//! PersistAnt's own ledger (`_sqlx_migrations`, via `persistant-sqlx`) is a
//! different table from refinery's tracking table. A database that refinery
//! already migrated has an EMPTY `_sqlx_migrations` ledger, so the first time
//! `PgBackend::migrate` runs there it has no record that these 6 migrations'
//! *effects* already happened -- it tries to run every statement fresh.
//!
//! None of the original files' `CREATE INDEX` statements have
//! `IF NOT EXISTS`, and V6's `ALTER TABLE ... ADD COLUMN` has no existence
//! guard either, so a literal verbatim copy run against a DB refinery already
//! migrated fails with "relation already exists" / "column already exists"
//! on first contact -- the opposite of the non-destructiveness this step is
//! supposed to prove. Every `CREATE INDEX` below adds `IF NOT EXISTS`
//! (Postgres supports this on `CREATE INDEX`), and V6's `ADD COLUMN` is
//! wrapped in a guarded `DO` block. These are the only changes from the
//! original files' text. **The resulting schema is identical either way** --
//! only the migration SQL's idempotency changed, not what it produces. The
//! proof is `persistant_migrate_applies_cleanly_on_top_of_refinery` in
//! `tests/persistant_migrate_tests.rs`.

use persistant_relational::{Migration, Value};
use uuid::Uuid;

const V1_CREATE_API_KEYS: &str = "CREATE TABLE IF NOT EXISTS api_keys (\
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),\
    key_hash VARCHAR(64) NOT NULL UNIQUE,\
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'admin', 'llm')),\
    description TEXT,\
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),\
    expires_at TIMESTAMP,\
    last_used_at TIMESTAMP\
 );\
 CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);\
 CREATE INDEX IF NOT EXISTS idx_api_keys_created_at ON api_keys(created_at);\
 CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at);";

const V2_CREATE_AUDIT_LOGS: &str = "CREATE TABLE IF NOT EXISTS audit_logs (\
    id BIGSERIAL PRIMARY KEY,\
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,\
    endpoint VARCHAR(255) NOT NULL,\
    method VARCHAR(10) NOT NULL,\
    status_code INTEGER NOT NULL,\
    latency_ms INTEGER NOT NULL,\
    request_size_bytes BIGINT,\
    response_size_bytes BIGINT,\
    error_message TEXT,\
    created_at TIMESTAMP NOT NULL DEFAULT NOW()\
 );\
 CREATE INDEX IF NOT EXISTS idx_audit_logs_api_key_id ON audit_logs(api_key_id);\
 CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);\
 CREATE INDEX IF NOT EXISTS idx_audit_logs_endpoint ON audit_logs(endpoint);\
 CREATE INDEX IF NOT EXISTS idx_audit_logs_status_code ON audit_logs(status_code);";

const V3_CREATE_SESSIONS: &str = "CREATE TABLE IF NOT EXISTS user_sessions (\
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),\
    api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,\
    session_token VARCHAR(128) NOT NULL UNIQUE,\
    ip_address INET,\
    user_agent TEXT,\
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),\
    last_accessed_at TIMESTAMP NOT NULL DEFAULT NOW(),\
    expires_at TIMESTAMP NOT NULL\
 );\
 CREATE INDEX IF NOT EXISTS idx_user_sessions_session_token ON user_sessions(session_token);\
 CREATE INDEX IF NOT EXISTS idx_user_sessions_api_key_id ON user_sessions(api_key_id);\
 CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);\
 CREATE INDEX IF NOT EXISTS idx_user_sessions_last_accessed_at ON user_sessions(last_accessed_at);";

const V4_CREATE_RATE_LIMITS: &str = "CREATE TABLE IF NOT EXISTS api_key_usage (\
    api_key_id UUID NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,\
    window_start TIMESTAMP NOT NULL,\
    request_count INTEGER NOT NULL DEFAULT 0,\
    PRIMARY KEY (api_key_id, window_start)\
 );\
 CREATE INDEX IF NOT EXISTS idx_api_key_usage_window_start ON api_key_usage(window_start);";

const V5_CREATE_AUTH_ATTEMPT_COUNTS: &str = "CREATE TABLE IF NOT EXISTS auth_attempt_counts (\
    client_ip INET NOT NULL,\
    window_start TIMESTAMP NOT NULL,\
    request_count INTEGER NOT NULL DEFAULT 0,\
    PRIMARY KEY (client_ip, window_start)\
 );\
 CREATE INDEX IF NOT EXISTS idx_auth_attempt_counts_window_start ON auth_attempt_counts(window_start);";

const V6_ADD_API_KEY_REVOCATION: &str = "DO $$ BEGIN \
    IF NOT EXISTS ( \
        SELECT 1 FROM information_schema.columns \
        WHERE table_name = 'api_keys' AND column_name = 'revoked_at' \
    ) THEN \
        ALTER TABLE api_keys ADD COLUMN revoked_at TIMESTAMP; \
    END IF; \
 END $$;\
 CREATE INDEX IF NOT EXISTS idx_api_keys_revoked_at ON api_keys(revoked_at);";

/// Issue #130: a SEPARATE table from `auth_attempt_counts` (V5) -- see
/// `migrations/V7__create_auth_failure_counts.sql`'s own comment for why.
const V7_CREATE_AUTH_FAILURE_COUNTS: &str = "CREATE TABLE IF NOT EXISTS auth_failure_counts (\
    client_ip INET NOT NULL,\
    window_start TIMESTAMP NOT NULL,\
    failure_count INTEGER NOT NULL DEFAULT 0,\
    PRIMARY KEY (client_ip, window_start)\
 );\
 CREATE INDEX IF NOT EXISTS idx_auth_failure_counts_window_start ON auth_failure_counts(window_start);";

/// ROADMAP A9: `user_sessions` (V3) was dead schema -- no reader or writer in
/// `src/`. Dropping it also drops its indexes. Irreversible; see
/// `migrations/V8__drop_user_sessions.sql`.
const V8_DROP_USER_SESSIONS: &str = "DROP TABLE IF EXISTS user_sessions;";

/// The 8 migrations, ported (V1-V7) plus V8, which is new. Versions and descriptions match the original
/// `V{n}__{description}.sql` filenames; each migration's SQL text lives in
/// its own module-level const above (kept out of this function so its own
/// line count stays short -- the SQL itself, not the wiring, is what's long).
pub fn migrations() -> Vec<Migration> {
    vec![
        Migration::new(1, "create_api_keys", V1_CREATE_API_KEYS),
        Migration::new(2, "create_audit_logs", V2_CREATE_AUDIT_LOGS),
        Migration::new(3, "create_sessions", V3_CREATE_SESSIONS),
        Migration::new(4, "create_rate_limits", V4_CREATE_RATE_LIMITS),
        Migration::new(
            5,
            "create_auth_attempt_counts",
            V5_CREATE_AUTH_ATTEMPT_COUNTS,
        ),
        Migration::new(6, "add_api_key_revocation", V6_ADD_API_KEY_REVOCATION),
        Migration::new(
            7,
            "create_auth_failure_counts",
            V7_CREATE_AUTH_FAILURE_COUNTS,
        ),
        Migration::new(8, "drop_user_sessions", V8_DROP_USER_SESSIONS),
    ]
}

/// `persistant_relational::Value` has no `Uuid` variant (deliberately narrow --
/// see docs/AUTH_PERSISTANT_ADOPTION_FIT.md §2). Per the adoption design's Q3
/// answer: stringify through one helper rather than changing the schema, so
/// the port's resulting schema stays identical to today's `UUID` columns.
pub fn uuid_to_value(id: Uuid) -> Value {
    Value::Text(id.to_string())
}

/// The other direction: a `Value::Text` holding a UUID's string form, as read
/// back from a query result row.
pub fn value_to_uuid(value: &Value) -> Result<Uuid, String> {
    match value {
        Value::Text(s) => {
            Uuid::parse_str(s).map_err(|e| format!("value is not a valid UUID string: {e}"))
        }
        other => Err(format!(
            "expected Value::Text holding a UUID, got {other:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrations_are_versioned_1_through_7_in_order() {
        let versions: Vec<i64> = migrations().iter().map(|m| m.version).collect();
        assert_eq!(versions, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn every_create_index_is_guarded() {
        for m in migrations() {
            for line in m.sql.split(';') {
                let trimmed = line.trim_start();
                if trimmed.to_uppercase().starts_with("CREATE INDEX") {
                    assert!(
                        trimmed.to_uppercase().contains("IF NOT EXISTS"),
                        "migration {} has an unguarded CREATE INDEX: {trimmed}",
                        m.version
                    );
                }
            }
        }
    }

    #[test]
    fn uuid_roundtrips_through_value() {
        let id = Uuid::new_v4();
        let value = uuid_to_value(id);
        let back = value_to_uuid(&value).expect("roundtrip should parse");
        assert_eq!(id, back);
    }

    #[test]
    fn value_to_uuid_rejects_non_text_values() {
        assert!(value_to_uuid(&Value::Int(1)).is_err());
        assert!(value_to_uuid(&Value::Null).is_err());
    }

    #[test]
    fn value_to_uuid_rejects_malformed_text() {
        assert!(value_to_uuid(&Value::Text("not-a-uuid".to_string())).is_err());
    }
}
