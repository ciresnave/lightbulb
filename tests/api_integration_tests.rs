//! Integration tests for the API server
//!
//! Tests OpenAI compatibility and Lightbulb-specific features.

#[cfg(test)]
mod api_tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use serde_json::json;
    use std::sync::Arc;
    use tower::ServiceExt; // For oneshot

    use lightbulb::api::{ApiConfig, ApiServer};
    use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler};

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

    /// Create test API server
    async fn create_test_server() -> ApiServer {
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
        let db_url = std::env::var("LIGHTBULB_TEST_DATABASE_URL").unwrap_or_else(|_| {
            "postgresql://lightbulb:vKTbmBA5RXIauMrNHxzs@localhost:5432/lightbulb".to_string()
        });

        // Real auth needs a real key. Before this existed the authenticated
        // cases sent `Bearer test-token` and expected 200 on the strength of a
        // comment reading "mock auth allows any token" — a premise that has not
        // been true since auth started validating against the database. They
        // were asserting that an arbitrary bearer token grants access, which is
        // the opposite of what should be pinned.
        provision_test_api_key(&db_url, "test-token", "admin").await;

        let config = ApiConfig {
            database_url: Some(db_url.clone()),
            bind_address: "127.0.0.1:0".to_string(),
            enable_openai_api: true,
            enable_admin_api: true,
            enable_lightbulb_extensions: true,
            jwt_secret: "test-secret".to_string(),
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
        let scheduler = Arc::new(MemoryAwareScheduler::new(scheduler_config));

        ApiServer::new(config, scheduler)
            .await
            .expect("Failed to create test server")
    }

    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_health_check() {
        let server = create_test_server().await;
        let app = server.build_router().with_state(server.state().clone());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
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
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                    .unwrap(),
            )
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
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                    .unwrap(),
            )
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
            .oneshot(
                Request::builder()
                    .uri("/v1/lightbulb/admin/cache/stats")
                    .header("authorization", "Bearer test-token")
                    .body(Body::empty())
                    .unwrap(),
            )
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
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/lightbulb/knowledge/query")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                    .unwrap(),
            )
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
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(serde_json::to_string(&request_body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
