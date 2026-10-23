//! Integration tests for the API server
//!
//! Tests OpenAI compatibility and Lightbulb-specific features.

#[cfg(test)]
mod api_tests {
    use axum::{
        body::Body,
        extract::ConnectInfo,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use serde_json::json;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use tower::ServiceExt; // For oneshot

    use lightbulb::api::{ApiConfig, ApiServer};
    use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler};

    /// Every real connection carries `ConnectInfo<SocketAddr>`, inserted by
    /// `into_make_service_with_connect_info` (see `ApiServer::serve`) — but
    /// `Router::oneshot` in these tests never goes through a real listener,
    /// so nothing inserts it. `pre_auth_attempt_limiter_middleware`
    /// (security audit item 3) extracts `ConnectInfo<SocketAddr>` and would
    /// otherwise 500 on "missing request extension" before ever reaching
    /// `auth_middleware`, let alone the handler these tests mean to pin.
    /// This fakes a peer address the same way a real connection would carry
    /// one.
    ///
    /// Takes the peer IP explicitly rather than hardcoding one: the
    /// pre-auth limiter keys its attempt counter by IP alone
    /// (`auth_middleware::pre_auth_attempt_limiter_middleware` binds
    /// `peer.ip()`, dropping the port), and that counter lives in
    /// `auth_attempt_counts`, a table every test in this file shares across
    /// one CI run. Every call site below passes its own distinct IP so
    /// that one test deliberately saturating its bucket (the two
    /// `pre_auth_attempt_limiter_*` tests below) cannot leak into another
    /// test's expectations, regardless of what order the test binary picks
    /// -- found when extending `postgres_integration` to run this whole
    /// file: with a single shared fake IP, a saturating test running before
    /// an ordinary one would make the ordinary one observe 429 instead of
    /// its expected 401/200, non-deterministically.
    fn with_fake_peer_at(mut request: Request<Body>, peer_ip: [u8; 4]) -> Request<Body> {
        request
            .extensions_mut()
            .insert(ConnectInfo(SocketAddr::from((peer_ip, 54321))));
        request
    }

    /// Insert a known API key so the authenticated cases have something real to
    /// present.
    ///
    /// `auth_middleware` hashes the bearer token with SHA-256, hex-encodes it
    /// lowercase, and looks that up in `api_keys` — so this reproduces the whole
    /// contract and nothing more. Idempotent, because `key_hash` is UNIQUE and
    /// the suite is run repeatedly against a persistent container.
    async fn provision_test_api_key(db_url: &str, token: &str, role: &str) {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(token.as_bytes());
        let key_hash: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect();

        let (client, connection) = tokio_postgres::connect(db_url, tokio_postgres::NoTls)
            .await
            .expect("connect to the test database — is the container up on this URL?");
        tokio::spawn(async move {
            let _ = connection.await;
        });
        client
            .execute(
                "INSERT INTO api_keys (key_hash, role, description) \
                 VALUES ($1, $2, 'lightbulb integration test key') \
                 ON CONFLICT (key_hash) DO NOTHING",
                &[&key_hash, &role],
            )
            .await
            .expect("provision the test API key");
    }

    /// Same contract as `provision_test_api_key`, but returns the row's id
    /// -- needed to call the revoke/rotate admin routes, which address a
    /// key by id, not by its (secret, never-looked-up-by-value-again) hash.
    async fn provision_test_api_key_returning_id(
        db_url: &str,
        token: &str,
        role: &str,
    ) -> uuid::Uuid {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(token.as_bytes());
        let key_hash: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect();

        let (client, connection) = tokio_postgres::connect(db_url, tokio_postgres::NoTls)
            .await
            .expect("connect to the test database — is the container up on this URL?");
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let row = client
            .query_one(
                "INSERT INTO api_keys (key_hash, role, description) \
                 VALUES ($1, $2, 'lightbulb integration test key') \
                 ON CONFLICT (key_hash) DO UPDATE SET role = EXCLUDED.role \
                 RETURNING id",
                &[&key_hash, &role],
            )
            .await
            .expect("provision the test API key");
        row.get("id")
    }

    // Overridable, because these tests WRITE to whatever database they are
    // handed and 5432 is routinely occupied on a shared machine — pointing
    // them at someone else's instance by default is how a test suite
    // corrupts a neighbour's data. Set `LIGHTBULB_TEST_DATABASE_URL` to a
    // throwaway container, e.g.:
    //
    //   docker run -d --name lightbulb-test-db \
    //     -e POSTGRES_USER=lightbulb -e POSTGRES_PASSWORD=... \
    //     -e POSTGRES_DB=lightbulb -p 5433:5432 postgres:18
    //   $env:LIGHTBULB_TEST_DATABASE_URL =
    //     "postgresql://lightbulb:...@localhost:5433/lightbulb"
    fn test_db_url() -> String {
        std::env::var("LIGHTBULB_TEST_DATABASE_URL").unwrap_or_else(|_| {
            "postgresql://lightbulb:vKTbmBA5RXIauMrNHxzs@localhost:5432/lightbulb".to_string()
        })
    }

    /// Create test API server
    async fn create_test_server() -> ApiServer {
        // Otherwise every `tracing::error!` in the request path (e.g. the
        // pre-auth attempt limiter failing closed on its own DB error) goes
        // nowhere: no subscriber means the event is silently dropped, and a
        // resulting 503 shows up with no indication of why. `try_init`
        // because multiple tests in this binary each call this function.
        let _ = tracing_subscriber::fmt()
            .with_test_writer()
            .with_env_filter("warn")
            .try_init();

        let db_url = test_db_url();

        let config = ApiConfig {
            database_url: Some(db_url.clone()),
            bind_address: "127.0.0.1:0".to_string(),
            enable_openai_api: true,
            enable_admin_api: true,
            enable_lightbulb_extensions: true,
            jwt_secret: Some("test-secret".to_string()),
            rate_limit_per_minute: 1000,
            enable_audit_log: false, // Disable for tests
            models_dir: None,        // No model loading for tests
            default_model: "test-model".to_string(),
            model_max_batch_size: 8,
            model_context_length: 2048,
            // Fields this test does not exercise (currently `tls`) take their
            // defaults. Listing every field exhaustively is what broke this
            // suite when `tls` was added: a test that must be edited each time
            // an unrelated field appears is a maintenance tax that buys no
            // safety, because the test asserts nothing about those fields.
            ..Default::default()
        };

        let scheduler_config = MemoryAwareConfig::default();
        let scheduler = Arc::new(std::sync::Mutex::new(MemoryAwareScheduler::new(
            scheduler_config,
        )));

        // `ApiServer::new` runs the embedded migrations -- this MUST happen
        // before any `provision_test_api_key*` call, not after. It used to
        // run after (provisioning sat above this call), which silently
        // depended on a persistent, already-migrated database from some
        // earlier run; a genuinely fresh database (this repo's
        // `postgres_integration` CI job, `initdb`-created every run) hit
        // "relation \"api_keys\" does not exist" on the very first
        // provisioning call, before migrations ever ran at all.
        let server = ApiServer::new(config, scheduler)
            .await
            .expect("Failed to create test server");

        // Real auth needs a real key. Before this existed the authenticated
        // cases sent `Bearer test-token` and expected 200 on the strength of a
        // comment reading "mock auth allows any token" — a premise that has not
        // been true since auth started validating against the database. They
        // were asserting that an arbitrary bearer token grants access, which is
        // the opposite of what should be pinned.
        provision_test_api_key(&db_url, "test-token", "admin").await;

        // Belt-and-suspenders alongside per-test fake IPs (`with_fake_peer_at`):
        // CI's `postgres_integration` job starts a brand-new `initdb`-created
        // cluster every run, so this table is empty there regardless. Locally,
        // against the persistent container `test_db_url`'s own doc comment
        // describes, repeated runs within the same per-minute window would
        // otherwise accumulate stale counts under reused IPs. Every test calls
        // `create_test_server` first, so clearing here gives each one a clean
        // slate no matter what ran before it, in either environment.
        let (client, connection) = tokio_postgres::connect(&db_url, tokio_postgres::NoTls)
            .await
            .expect("connect to the test database — is the container up on this URL?");
        tokio::spawn(async move {
            let _ = connection.await;
        });
        client
            .execute("TRUNCATE TABLE auth_attempt_counts", &[])
            .await
            .expect("clear auth_attempt_counts before the test");

        server
    }

    /// `create_test_server` configures a real database and no model
    /// (`models_dir: None`) -- the readiness check's real-world "database
    /// configured and reachable, model never requested" shape. Database
    /// `ok` here is itself the interesting assertion: `pool.get()` plus a
    /// real `SELECT 1` against `postgres_integration`'s live cluster, not a
    /// stand-in.
    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_readiness_check_reports_ready_with_db_configured_and_no_model() {
        let server = create_test_server().await;
        let app = server.build_router().with_state(server.state().clone());

        let response = app
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .uri("/health/ready")
                    .body(Body::empty())
                    .unwrap(),
                [10, 0, 0, 14],
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["status"], "ready");
        assert_eq!(body["database"], "ok");
        assert_eq!(body["model"], "not_configured");
    }

    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_health_check() {
        let server = create_test_server().await;
        let app = server.build_router().with_state(server.state().clone());

        let response = app
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
                [10, 0, 0, 1],
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_chat_completions_without_auth() {
        let server = create_test_server().await;
        let app = server.build_router().with_state(server.state().clone());

        let request_body = json!({
            "model": "lightbulb-7b",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        });

        let response = app
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                    .unwrap(),
                [10, 0, 0, 2],
            ))
            .await
            .unwrap();

        // Should return 401 Unauthorized without Bearer token
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_chat_completions_with_auth() {
        let server = create_test_server().await;
        let app = server.build_router().with_state(server.state().clone());

        let request_body = json!({
            "model": "lightbulb-7b",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        });

        let response = app
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                    .unwrap(),
                [10, 0, 0, 3],
            ))
            .await
            .unwrap();

        // Should succeed with Bearer token (mock auth allows any token)
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_admin_cache_stats() {
        let server = create_test_server().await;
        let app = server.build_router().with_state(server.state().clone());

        let response = app
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .uri("/v1/lightbulb/admin/cache/stats")
                    .header("authorization", "Bearer test-token")
                    .body(Body::empty())
                    .unwrap(),
                [10, 0, 0, 4],
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_knowledge_base_query() {
        let server = create_test_server().await;
        let app = server.build_router().with_state(server.state().clone());

        let request_body = json!({
            "query": "What is machine learning?",
            "max_results": 5
        });

        let response = app
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .method("POST")
                    .uri("/v1/lightbulb/knowledge/query")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                    .unwrap(),
                [10, 0, 0, 5],
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_lightbulb_extensions_in_chat() {
        let server = create_test_server().await;
        let app = server.build_router().with_state(server.state().clone());

        let request_body = json!({
            "model": "lightbulb-7b",
            "messages": [
                {"role": "user", "content": "Explain quantum computing"}
            ],
            "lightbulb": {
                "reasoning_budget": {
                    "max_chains": 5,
                    "max_steps": 10
                },
                "use_knowledge_base": true,
                "metadata": {
                    "priority": "high",
                    "tags": ["research", "physics"]
                }
            }
        });

        let response = app
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                    .unwrap(),
                [10, 0, 0, 6],
            ))
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    /// Security audit item 3: the pre-auth attempt limiter runs before the
    /// bearer-token lookup and refuses once `max_auth_attempts_per_minute_per_ip`
    /// is exceeded for a given `ConnectInfo` peer — independent of whether
    /// any of those attempts carried a valid key.
    ///
    /// Now run by `postgres_integration`'s full-file pass — this is the
    /// positive control for the `_is_keyed_per_ip` test below: it proves the
    /// limiter actually engages and refuses at `limit + 1` for the IP that
    /// hits it this many times, so that test's "a different IP is let
    /// through" result means something rather than the limiter being a
    /// no-op.
    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_pre_auth_attempt_limiter_refuses_after_threshold() {
        let server = create_test_server().await;
        let limit = server.state().config.max_auth_attempts_per_minute_per_ip;
        let app = server.build_router().with_state(server.state().clone());

        // `/health` is deliberately registered AFTER every `.layer()` call in
        // `build_router` (so load balancers never need a bearer token), which
        // means it is also outside the pre-auth limiter — it cannot be used
        // to exercise this. `/v1/chat/completions` with no Authorization
        // header reaches the limiter first; it would 401 afterward, but the
        // limiter must refuse with 429 before auth is ever reached.
        let mut last_status = StatusCode::OK;
        for _ in 0..=limit {
            let request = with_fake_peer_at(unauthenticated_chat_request(), [10, 0, 0, 7]);
            last_status = app.clone().oneshot(request).await.unwrap().status();
        }

        assert_eq!(
            last_status,
            StatusCode::TOO_MANY_REQUESTS,
            "the (limit+1)th request from the same peer within the window must be refused \
             before auth ever runs, regardless of whether it carries a valid key"
        );
    }

    /// Security audit item 3: a DIFFERENT peer IP must not be affected by
    /// another IP's attempt count — the limiter is keyed per-IP, not
    /// global.
    ///
    /// The "other peer" below must differ in IP, not just port: the
    /// limiter binds `peer.ip()` only (`auth_middleware.rs`), discarding
    /// the port, so two requests that share an IP share a bucket
    /// regardless of port. A prior version of this test varied only the
    /// port (127.0.0.1:54321 vs 127.0.0.1:9999) and so never actually
    /// exercised per-IP keying — found when extending `postgres_integration`
    /// to run this file for real. Mutation-checked: temporarily keying the
    /// limiter by `SocketAddr` (ip+port) instead of `IpAddr` made this
    /// version of the test fail as expected; the saturating test above is
    /// the positive control proving the limiter engages at all.
    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_pre_auth_attempt_limiter_is_keyed_per_ip() {
        let server = create_test_server().await;
        let limit = server.state().config.max_auth_attempts_per_minute_per_ip;
        let app = server.build_router().with_state(server.state().clone());

        for _ in 0..=limit {
            let request = with_fake_peer_at(unauthenticated_chat_request(), [10, 0, 0, 8]);
            let _ = app.clone().oneshot(request).await.unwrap();
        }

        let other_peer_request = with_fake_peer_at(unauthenticated_chat_request(), [10, 0, 0, 9]);

        let response = app.oneshot(other_peer_request).await.unwrap();

        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "a different peer IP must not inherit another IP's attempt count — it should \
             reach auth (and 401 there, with no Authorization header) rather than being \
             refused by the limiter"
        );
    }

    /// Regression test for a critical pre-existing bug (found via the new
    /// `postgres_integration` CI job -- this is the first time any of this
    /// file's Postgres-gated tests have ever run against a real database in
    /// CI): `pre_auth_attempt_limiter_middleware` bound a `String` (the
    /// client IP's `.to_string()`) to a `$1::inet`-cast parameter.
    /// `String`'s `ToSql` only covers TEXT-family OIDs, so encoding failed
    /// on EVERY call, unconditionally -- not just past the rate limit. On
    /// any real deployment with a database configured, this meant every
    /// single authenticated request failed closed with a 503, since this
    /// middleware runs in front of every route. A single ordinary request,
    /// nowhere near the attempt limit, is enough to reproduce it.
    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_pre_auth_attempt_limiter_does_not_503_an_ordinary_request() {
        let server = create_test_server().await;
        let app = server.build_router().with_state(server.state().clone());

        let response = app
            .oneshot(with_fake_peer_at(
                unauthenticated_chat_request(),
                [10, 0, 0, 10],
            ))
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "a single, ordinary, nowhere-near-the-limit request must reach auth (and 401 \
             there, with no Authorization header) -- a 503 here means the pre-auth limiter's \
             own database call is failing, not that the client did anything wrong"
        );
    }

    /// Phase 4 hardening item 3 (API key revoke/rotate), PM condition 3:
    /// this is the one behaviour from this item that must have a test that
    /// RUNS, not an `#[ignore]`-only assertion -- see the CI job
    /// `postgres_integration` in `.github/workflows/ci.yml`, which starts a
    /// fresh `initdb`-created Postgres and runs exactly the tests in this
    /// block by name filter, `--include-ignored`.
    ///
    /// Drives the real `DELETE /v1/lightbulb/admin/api-keys/{id}` handler
    /// through the real router, against the real `auth_middleware` lookup
    /// -- not a direct database mutation standing in for the endpoint.
    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_revoking_an_api_key_rejects_it_on_the_next_request() {
        let db_url = test_db_url();
        let server = create_test_server().await; // provisions admin "test-token"
        let target_id =
            provision_test_api_key_returning_id(&db_url, "revoke-target-token", "user").await;
        let app = server.build_router().with_state(server.state().clone());

        // The target key authenticates before revocation.
        let before = app
            .clone()
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .uri("/v1/lightbulb/admin/cache/stats")
                    .header("authorization", "Bearer revoke-target-token")
                    .body(Body::empty())
                    .unwrap(),
                [10, 0, 0, 11],
            ))
            .await
            .unwrap();
        // "user" role authenticates fine but isn't admin, so
        // admin_check_middleware returns exactly 403, not 401 -- asserted
        // precisely (not assert_ne! against UNAUTHORIZED alone) because
        // that weaker check would also pass on a 503, which is exactly the
        // failure this test needs to catch, not let through silently.
        assert_eq!(
            before.status(),
            StatusCode::FORBIDDEN,
            "the key must be live before revocation"
        );

        let revoke_response = app
            .clone()
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/v1/lightbulb/admin/api-keys/{target_id}"))
                    .header("authorization", "Bearer test-token")
                    .body(Body::empty())
                    .unwrap(),
                [10, 0, 0, 11],
            ))
            .await
            .unwrap();
        assert_eq!(revoke_response.status(), StatusCode::OK);

        let after = app
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .uri("/v1/lightbulb/admin/cache/stats")
                    .header("authorization", "Bearer revoke-target-token")
                    .body(Body::empty())
                    .unwrap(),
                [10, 0, 0, 11],
            ))
            .await
            .unwrap();
        assert_eq!(
            after.status(),
            StatusCode::UNAUTHORIZED,
            "a revoked key must be rejected on its very next request"
        );
    }

    /// PM condition 2: revoking the caller's OWN key must be refused unless
    /// another unexpired, unrevoked admin key exists -- otherwise this
    /// endpoint is the request that locks every admin out of every admin
    /// route, including the one that could undo it.
    ///
    /// Self-contained rather than order-dependent: revokes every OTHER
    /// admin key first (there may be several, left over from
    /// `create_test_server()` calls in other tests sharing this database),
    /// so this test's own result does not depend on what ran before it. Only
    /// the new CI job's dedicated, name-filtered Postgres sees this
    /// disruption -- see the module doc comment on the test above.
    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_revoke_refuses_to_lock_out_the_last_admin_key() {
        let db_url = test_db_url();
        // Migrations must run (inside create_test_server, via ApiServer::new)
        // before any provisioning call against a fresh database -- same
        // ordering bug as create_test_server's own fix above, and the exact
        // reason this test itself first failed with "relation \"api_keys\"
        // does not exist" against the new postgres_integration CI job.
        let server = create_test_server().await;
        let only_admin_id =
            provision_test_api_key_returning_id(&db_url, "lockout-test-admin-token", "admin").await;

        let (client, connection) = tokio_postgres::connect(&db_url, tokio_postgres::NoTls)
            .await
            .expect("connect to the test database");
        tokio::spawn(async move {
            let _ = connection.await;
        });
        // Capture exactly which rows this blanket UPDATE touches (not just
        // its row count) so they can be restored afterward -- this
        // database is shared with every other test in the same CI run,
        // including the "test-token" admin key `create_test_server`
        // provisions for all of them. Revoking it here and never
        // restoring it broke two sibling tests the first time this ran in
        // CI (their own "Bearer test-token" admin calls got 401, not the
        // 200/201 they expected) -- a real, order-dependent bug this
        // comment originally only claimed not to have, rather than one
        // this code actually prevented.
        let revoked_ids: Vec<uuid::Uuid> = client
            .query(
                "UPDATE api_keys SET revoked_at = NOW() \
                 WHERE role = 'admin' AND id != $1 AND revoked_at IS NULL \
                 RETURNING id",
                &[&only_admin_id],
            )
            .await
            .expect("revoke every other admin key")
            .iter()
            .map(|row| row.get("id"))
            .collect();

        let app = server.build_router().with_state(server.state().clone());

        let response = app
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/v1/lightbulb/admin/api-keys/{only_admin_id}"))
                    .header("authorization", "Bearer lockout-test-admin-token")
                    .body(Body::empty())
                    .unwrap(),
                [10, 0, 0, 12],
            ))
            .await
            .unwrap();

        client
            .execute(
                "UPDATE api_keys SET revoked_at = NULL WHERE id = ANY($1)",
                &[&revoked_ids],
            )
            .await
            .expect("restore the admin keys this test revoked");

        assert_eq!(
            response.status(),
            StatusCode::CONFLICT,
            "revoking the only live admin key via itself must be refused, not honoured"
        );
    }

    /// PM condition 1: rotate must be one transaction -- the old key dead
    /// and the new key live, never both or neither. Observed from outside
    /// the transaction (the only vantage a caller has): the old token is
    /// rejected and the new token (returned once, in the response body) is
    /// accepted, on the very next requests after the rotate call returns.
    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_rotating_an_api_key_revokes_the_old_and_issues_a_live_new_one() {
        let db_url = test_db_url();
        let server = create_test_server().await; // provisions admin "test-token"
        let old_id =
            provision_test_api_key_returning_id(&db_url, "rotate-target-token", "user").await;
        let app = server.build_router().with_state(server.state().clone());

        let rotate_response = app
            .clone()
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .method("POST")
                    .uri(format!("/v1/lightbulb/admin/api-keys/{old_id}/rotate"))
                    .header("authorization", "Bearer test-token")
                    .body(Body::empty())
                    .unwrap(),
                [10, 0, 0, 13],
            ))
            .await
            .unwrap();
        assert_eq!(rotate_response.status(), StatusCode::CREATED);

        let body = rotate_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let new_api_key = body["api_key"]
            .as_str()
            .expect("api_key in response")
            .to_string();
        assert_eq!(body["rotated_from"].as_str().unwrap(), old_id.to_string());

        let old_token_response = app
            .clone()
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .uri("/v1/lightbulb/admin/cache/stats")
                    .header("authorization", "Bearer rotate-target-token")
                    .body(Body::empty())
                    .unwrap(),
                [10, 0, 0, 13],
            ))
            .await
            .unwrap();
        assert_eq!(
            old_token_response.status(),
            StatusCode::UNAUTHORIZED,
            "the rotated-out key must be dead"
        );

        let new_token_response = app
            .oneshot(with_fake_peer_at(
                Request::builder()
                    .uri("/v1/lightbulb/admin/cache/stats")
                    .header("authorization", format!("Bearer {new_api_key}"))
                    .body(Body::empty())
                    .unwrap(),
                [10, 0, 0, 13],
            ))
            .await
            .unwrap();
        assert_eq!(
            new_token_response.status(),
            StatusCode::FORBIDDEN,
            "the rotated-in key must be live, same role as the old key ('user'): exactly 403 \
             on this admin-only route, not 401 (dead) and not a weaker assert_ne! that would \
             also pass on a 503"
        );
    }

    fn unauthenticated_chat_request() -> Request<Body> {
        let request_body = json!({
            "model": "lightbulb-7b",
            "messages": [{"role": "user", "content": "Hello!"}],
        });
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&request_body).unwrap()))
            .unwrap()
    }

    /// Same shape as [`unauthenticated_chat_request`], with an
    /// `Authorization` header — `token` need not be provisioned; an unknown
    /// token still reaches `auth_middleware`'s hash lookup and 401s there,
    /// which is exactly what issue #130's failure limiter counts.
    fn chat_request_with_token(token: &str) -> Request<Body> {
        let request_body = json!({
            "model": "lightbulb-7b",
            "messages": [{"role": "user", "content": "Hello!"}],
        });
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::from(serde_json::to_string(&request_body).unwrap()))
            .unwrap()
    }

    /// Issue #130: the strict, failures-only limiter escalates
    /// `auth_middleware`'s ordinary 401 to 429 once
    /// `max_auth_failures_per_minute_per_ip` real authentication failures
    /// (never-provisioned token, so every request 401s at the hash lookup)
    /// land from the same peer within the window.
    ///
    /// This is the positive control for the two tests below: it proves the
    /// limiter actually engages and escalates at `limit + 1`, so a result
    /// of "still 401" elsewhere in this file means the limiter correctly
    /// did NOT engage, not that it is a no-op.
    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_auth_failure_limiter_escalates_to_429_after_threshold() {
        let server = create_test_server().await;
        let limit = server.state().config.max_auth_failures_per_minute_per_ip;
        let app = server.build_router().with_state(server.state().clone());

        let mut last_status = StatusCode::OK;
        for _ in 0..=limit {
            let request = with_fake_peer_at(
                chat_request_with_token("never-provisioned-token"),
                [10, 0, 1, 1],
            );
            last_status = app.clone().oneshot(request).await.unwrap().status();
        }

        assert_eq!(
            last_status,
            StatusCode::TOO_MANY_REQUESTS,
            "the (limit+1)th real authentication failure from the same peer within the \
             window must escalate to 429, not the ordinary 401"
        );
    }

    /// Issue #130: a DIFFERENT peer IP must not be affected by another IP's
    /// failure count — same per-IP keying requirement as the flood
    /// limiter's own `_is_keyed_per_ip` test, for the separate failures-only
    /// counter.
    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_auth_failure_limiter_is_keyed_per_ip() {
        let server = create_test_server().await;
        let limit = server.state().config.max_auth_failures_per_minute_per_ip;
        let app = server.build_router().with_state(server.state().clone());

        for _ in 0..=limit {
            let request = with_fake_peer_at(
                chat_request_with_token("never-provisioned-token"),
                [10, 0, 1, 2],
            );
            let _ = app.clone().oneshot(request).await.unwrap();
        }

        let other_peer_request = with_fake_peer_at(
            chat_request_with_token("never-provisioned-token"),
            [10, 0, 1, 3],
        );
        let response = app.oneshot(other_peer_request).await.unwrap();

        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "a different peer IP must not inherit another IP's failure count — it should \
             get the ordinary 401 for its own first-ever failure, not 429"
        );
    }

    /// Issue #130's core behavioral claim: this counter is FAILURES-only.
    /// `limit` SUCCESSFUL requests (a real provisioned key) from one peer
    /// must never trip it — only a subsequent genuine authentication
    /// failure from that same peer should. Distinguishes this limiter from
    /// `pre_auth_attempt_limiter_middleware`'s flood limiter, which counts
    /// every request including successes; this test would fail if the new
    /// limiter accidentally counted successes the same way.
    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_auth_failure_limiter_does_not_count_successful_requests() {
        let server = create_test_server().await; // provisions admin "test-token"
        let limit = server.state().config.max_auth_failures_per_minute_per_ip;
        let app = server.build_router().with_state(server.state().clone());

        for _ in 0..(limit + 5) {
            let request = with_fake_peer_at(chat_request_with_token("test-token"), [10, 0, 1, 4]);
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "a valid key must keep succeeding regardless of how many times it's used — \
                 confirms these requests never reach the failure path at all"
            );
        }

        let failing_request = with_fake_peer_at(
            chat_request_with_token("never-provisioned-token"),
            [10, 0, 1, 4],
        );
        let response = app.oneshot(failing_request).await.unwrap();

        assert_eq!(
            response.status(),
            StatusCode::UNAUTHORIZED,
            "more successful requests than `limit` from this peer must not have armed the \
             failures-only counter — this first real failure should still get the ordinary \
             401, not 429"
        );
    }
}
