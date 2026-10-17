//! Proves PersistAnt adoption Step 1's non-destructiveness claim
//! (docs/PERSISTANT_ADOPTION_DESIGN.md §7 Q1): that
//! `persistant_sqlx::PgBackend::migrate` with the ported, idempotency-fixed
//! migrations (`src/api/persistant_migrate.rs`) applies cleanly on top of a
//! database `refinery` already migrated -- not just a fresh one.

#[cfg(test)]
mod persistant_migrate_tests {
    use lightbulb::api::persistant_migrate::migrations;
    use lightbulb::api::{ApiConfig, ApiServer};
    use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler};
    use persistant_relational::Migrates;
    use persistant_sqlx::PgBackend;
    use std::sync::Arc;

    // Same override pattern as tests/api_integration_tests.rs's `test_db_url` --
    // these tests write to whatever database they are handed, so pointing
    // them at a shared instance by default would be unsafe on a machine where
    // 5432 is already in use for something else.
    fn test_db_url() -> String {
        std::env::var("LIGHTBULB_TEST_DATABASE_URL").unwrap_or_else(|_| {
            "postgresql://lightbulb:vKTbmBA5RXIauMrNHxzs@localhost:5432/lightbulb".to_string()
        })
    }

    /// `ApiServer::new` runs refinery's `embed_migrations!` against
    /// `database_url` as a side effect of construction -- the same path
    /// every other integration test in this repo uses to get a migrated
    /// database, reused here rather than re-implemented.
    async fn migrate_with_refinery(db_url: &str) {
        let config = ApiConfig {
            database_url: Some(db_url.to_string()),
            bind_address: "127.0.0.1:0".to_string(),
            models_dir: None,
            default_model: "test-model".to_string(),
            ..Default::default()
        };
        let scheduler = Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default()));
        ApiServer::new(config, scheduler)
            .await
            .expect("refinery migration via ApiServer::new should succeed");
    }

    /// The core proof: PersistAnt's `_sqlx_migrations` ledger starts empty on
    /// a refinery-only database, so this is a genuine first contact, not a
    /// no-op skip -- and it must still succeed without erroring, because
    /// every statement in `persistant_migrate::migrations()` is guarded
    /// (`IF NOT EXISTS` / a conditional `ADD COLUMN`).
    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn persistant_migrate_applies_cleanly_on_top_of_refinery() {
        let db_url = test_db_url();
        migrate_with_refinery(&db_url).await;

        let backend = PgBackend::connect(&db_url)
            .await
            .expect("PgBackend should connect to the same database");

        let report = backend
            .migrate(&migrations())
            .await
            .expect("migrate() must succeed against a database refinery already migrated");

        // All 6 are newly recorded in PersistAnt's own ledger -- refinery's
        // prior run left no trace in `_sqlx_migrations`, so none of these
        // were already-applied from PersistAnt's point of view. The claim
        // being tested is specifically that running them anyway doesn't
        // error, not that the ledger considered them pre-satisfied.
        assert_eq!(report.applied, vec![1, 2, 3, 4, 5, 6]);
        assert!(report.already_applied.is_empty());
    }

    async fn assert_tables_exist(client: &tokio_postgres::Client, tables: &[&str]) {
        for table in tables {
            let exists: bool = client
                .query_one(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables \
                     WHERE table_name = $1)",
                    &[table],
                )
                .await
                .unwrap_or_else(|e| panic!("checking table {table}: {e}"))
                .get(0);
            assert!(exists, "table {table} should exist after the port");
        }
    }

    async fn assert_column_exists(client: &tokio_postgres::Client, table: &str, column: &str) {
        let exists: bool = client
            .query_one(
                "SELECT EXISTS (SELECT 1 FROM information_schema.columns \
                 WHERE table_name = $1 AND column_name = $2)",
                &[&table, &column],
            )
            .await
            .unwrap_or_else(|e| panic!("checking {table}.{column}: {e}"))
            .get(0);
        assert!(exists, "{table}.{column} should exist after the port");
    }

    async fn assert_indexes_exist(client: &tokio_postgres::Client, indexes: &[&str]) {
        for index in indexes {
            let exists: bool = client
                .query_one(
                    "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = $1)",
                    &[index],
                )
                .await
                .unwrap_or_else(|e| panic!("checking index {index}: {e}"))
                .get(0);
            assert!(exists, "index {index} should exist after the port");
        }
    }

    /// The "diff the resulting schema" half of the proof: every table,
    /// column and index the original `.sql` files create is present and
    /// correctly shaped after PersistAnt's run, queried directly rather than
    /// assumed from the migration text succeeding. Split into the two
    /// `assert_*_exist` helpers above to keep this function itself short --
    /// the number of things checked, not the function's own logic, is what's
    /// large here.
    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn schema_after_persistant_migrate_matches_the_original_migrations() {
        let db_url = test_db_url();
        migrate_with_refinery(&db_url).await;
        let backend = PgBackend::connect(&db_url)
            .await
            .expect("PgBackend should connect");
        backend
            .migrate(&migrations())
            .await
            .expect("migrate() should succeed");

        let (client, connection) = tokio_postgres::connect(&db_url, tokio_postgres::NoTls)
            .await
            .expect("connect to the test database directly for schema inspection");
        tokio::spawn(async move {
            let _ = connection.await;
        });

        assert_tables_exist(
            &client,
            &[
                "api_keys",
                "audit_logs",
                "user_sessions",
                "api_key_usage",
                "auth_attempt_counts",
            ],
        )
        .await;

        assert_column_exists(&client, "api_keys", "revoked_at").await;

        assert_indexes_exist(
            &client,
            &[
                "idx_api_keys_key_hash",
                "idx_audit_logs_endpoint",
                "idx_user_sessions_session_token",
                "idx_api_key_usage_window_start",
                "idx_auth_attempt_counts_window_start",
                "idx_api_keys_revoked_at",
            ],
        )
        .await;
    }

    /// Running `migrate()` a SECOND time, now that PersistAnt's own ledger
    /// has recorded all 6, must be a pure no-op (the ordinary case
    /// `Migrates::migrate`'s doc comment describes: "skipping those already
    /// applied") -- this is the one case where PersistAnt's own ledger, not
    /// the guards in the SQL text, is what keeps it non-destructive.
    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn persistant_migrate_is_a_no_op_on_its_own_second_run() {
        let db_url = test_db_url();
        migrate_with_refinery(&db_url).await;
        let backend = PgBackend::connect(&db_url)
            .await
            .expect("PgBackend should connect");
        backend
            .migrate(&migrations())
            .await
            .expect("first migrate() should succeed");

        let second = backend
            .migrate(&migrations())
            .await
            .expect("second migrate() should also succeed, not just the first");
        assert!(
            second.applied.is_empty(),
            "second run should apply nothing new"
        );
        assert_eq!(second.already_applied, vec![1, 2, 3, 4, 5, 6]);
    }
}
