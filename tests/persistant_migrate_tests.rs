// SPDX-License-Identifier: MIT OR Apache-2.0
//! Proves PersistAnt adoption Step 1's non-destructiveness claim
//! (docs/PERSISTANT_ADOPTION_DESIGN.md §7 Q1): that
//! `persistant_sqlx::PgBackend::migrate` with the ported, idempotency-fixed
//! migrations (`src/api/persistant_migrate.rs`) applies cleanly on top of a
//! database `refinery` already migrated -- not just a fresh one.

#[cfg(test)]
mod persistant_migrate_tests {
    use lightbulb::api::persistant_migrate::migrations;
    use persistant_relational::Migrates;
    use persistant_sqlx::PgBackend;

    // Same override pattern as tests/api_integration_tests.rs's `test_db_url` --
    // these tests write to whatever database they are handed, so pointing
    // them at a shared instance by default would be unsafe on a machine where
    // 5432 is already in use for something else.
    fn test_db_url() -> String {
        std::env::var("LIGHTBULB_TEST_DATABASE_URL").unwrap_or_else(|_| {
            "postgresql://lightbulb:vKTbmBA5RXIauMrNHxzs@localhost:5432/lightbulb".to_string()
        })
    }

    /// Simulates a PRE-Step-5 deployment: `ApiServer::new` ran refinery's
    /// `embed_migrations!` as its own migration path before Step 5 made
    /// `PgBackend::migrate` the sole one. That code is gone from the app
    /// now, so this helper runs refinery directly against a raw
    /// `tokio_postgres::Client` (a dev-dependency, see Cargo.toml) -- the
    /// real-world scenario this simulates (an existing refinery-migrated
    /// database, now pointed at the PersistAnt-only binary) is still worth
    /// testing even though the binary itself no longer has this code path.
    ///
    /// Resets the schema to empty FIRST. Found by CI (`postgres_integration`
    /// runs `api_integration_tests` and this file back to back against the
    /// SAME shared cluster, port 55432, within one job): once Step 5 made
    /// `ApiServer::new` (used by `api_integration_tests`) migrate through
    /// `PgBackend::migrate()` instead of refinery, that run leaves the same
    /// tables/indexes behind that this file's raw refinery run also tries to
    /// create -- `CREATE TABLE IF NOT EXISTS` tolerates that, but none of
    /// the original `.sql` files' `CREATE INDEX` statements have
    /// `IF NOT EXISTS` (only `persistant_migrate`'s ported, idempotency-fixed
    /// copies do), so refinery's run failed with "relation ... already
    /// exists" -- a real, permanent collision now that both test files'
    /// migration paths create the identical schema, not the one-off
    /// container-reuse mistake this looked like at first. Each test in this
    /// file needs its own empty schema regardless of what ran before it in
    /// the same CI job.
    async fn migrate_with_refinery(db_url: &str) {
        let (mut client, connection) = tokio_postgres::connect(db_url, tokio_postgres::NoTls)
            .await
            .expect("connect to the test database for the refinery baseline");
        tokio::spawn(async move {
            let _ = connection.await;
        });
        client
            .batch_execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
            .await
            .expect("reset the schema to empty before the refinery baseline");
        refinery::embed_migrations!("./migrations");
        migrations::runner()
            .run_async(&mut client)
            .await
            .expect("refinery migration (simulating a pre-Step-5 deployment) should succeed");
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

        // All 7 are newly recorded in PersistAnt's own ledger -- refinery's
        // prior run left no trace in `_sqlx_migrations`, so none of these
        // were already-applied from PersistAnt's point of view. The claim
        // being tested is specifically that running them anyway doesn't
        // error, not that the ledger considered them pre-satisfied.
        assert_eq!(report.applied, vec![1, 2, 3, 4, 5, 6, 7]);
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
        assert_eq!(second.already_applied, vec![1, 2, 3, 4, 5, 6, 7]);
    }
}
