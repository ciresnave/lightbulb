//! Admin API Module
//!
//! Provides administrative endpoints for system management, cache control,
//! scheduler inspection, and metrics.

use axum::{
    Json, Router,
    extract::{Extension, Path, State},
    http::StatusCode,
    middleware,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use chrono::NaiveDateTime;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::api::AppState;
use crate::api::auth_middleware::{ApiKeyInfo, admin_check_middleware};

/// Create admin API routes.
///
/// `admin_check_middleware` is attached HERE, not just defined in
/// `auth_middleware.rs` — it was defined there and never wired to any route
/// (security audit, board PM task, 2026-10-04): every admin endpoint,
/// including key creation, was reachable by any authenticated caller
/// regardless of role. Attaching it as this sub-router's own `.layer()`
/// scopes the check to exactly these routes, not the OpenAI-compatible
/// or Lightbulb-extension routers this one gets `.merge()`d with.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/lightbulb/admin/cache/stats", get(cache_stats))
        .route("/v1/lightbulb/admin/cache/clear", post(cache_clear))
        .route("/v1/lightbulb/admin/scheduler/queue", get(scheduler_queue))
        .route("/v1/lightbulb/admin/scheduler/stats", get(scheduler_stats))
        .route("/v1/lightbulb/admin/metrics", get(system_metrics))
        .route("/v1/lightbulb/admin/api-keys", post(create_api_key))
        .route("/v1/lightbulb/admin/api-keys/{id}", delete(revoke_api_key))
        .route(
            "/v1/lightbulb/admin/api-keys/{id}/rotate",
            post(rotate_api_key),
        )
        .layer(middleware::from_fn(admin_check_middleware))
}

/// Privilege rank of an API key role. `"admin"` is the only privileged tier
/// this codebase recognizes today — `"user"` and `"llm"` are peers below it,
/// not a further-ordered pair — so a caller may never request a role ranked
/// above their own.
fn role_rank(role: &str) -> u8 {
    if role == "admin" { 1 } else { 0 }
}

/// Cache statistics response
#[derive(Debug, Clone, Serialize)]
pub struct CacheStatsResponse {
    pub total_entries: usize,
    pub total_size_bytes: usize,
    pub hit_rate: f32,
    pub eviction_count: usize,
    pub layers: Vec<LayerCacheStats>,
}

/// Per-layer cache statistics
#[derive(Debug, Clone, Serialize)]
pub struct LayerCacheStats {
    pub layer_idx: usize,
    pub entries: usize,
    pub size_bytes: usize,
    pub compression_ratio: f32,
}

/// Get cache statistics
///
/// Reports the prefix cache's real stats (board item M4 slice 1), written by
/// the model-runner thread after every job -- see `AppState::cache_stats`.
/// `layers` stays empty: no per-layer breakdown exists in the prefix cache
/// today, so this is an honest absence, not a mock.
async fn cache_stats(State(state): State<AppState>) -> impl IntoResponse {
    let stats = state
        .cache_stats
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_default();

    let response = CacheStatsResponse {
        total_entries: stats.entries,
        total_size_bytes: stats.size_bytes,
        hit_rate: stats.hit_rate() as f32,
        eviction_count: stats.evictions,
        layers: vec![],
    };

    (StatusCode::OK, Json(response))
}

/// Clear cache request
#[derive(Debug, Clone, Deserialize)]
pub struct ClearCacheRequest {
    /// Clear all layers or specific layers
    #[serde(default)]
    pub layer_indices: Option<Vec<usize>>,

    /// Clear only entries older than this many seconds
    #[serde(default)]
    pub older_than_seconds: Option<u64>,
}

/// Clear cache response
#[derive(Debug, Clone, Serialize)]
pub struct ClearCacheResponse {
    pub cleared_entries: usize,
    pub freed_bytes: usize,
}

/// Clear cache
async fn cache_clear(
    State(state): State<AppState>,
    Json(request): Json<ClearCacheRequest>,
) -> impl IntoResponse {
    // TODO: Implement actual cache clearing
    let _scheduler = &state.scheduler;
    let _layer_indices = request.layer_indices;
    let _older_than = request.older_than_seconds;

    let response = ClearCacheResponse {
        cleared_entries: 0,
        freed_bytes: 0,
    };

    (StatusCode::OK, Json(response))
}

/// Scheduler queue response
#[derive(Debug, Clone, Serialize)]
pub struct SchedulerQueueResponse {
    pub pending_requests: usize,
    pub running_requests: usize,
    pub queue: Vec<QueuedRequest>,
}

/// Queued request info
#[derive(Debug, Clone, Serialize)]
pub struct QueuedRequest {
    pub request_id: String,
    pub priority: String,
    pub queued_at: String,
    pub estimated_tokens: usize,
}

/// Get scheduler queue
async fn scheduler_queue(State(state): State<AppState>) -> impl IntoResponse {
    let _scheduler = &state.scheduler;

    // TODO: Get actual queue from scheduler
    let response = SchedulerQueueResponse {
        pending_requests: 0,
        running_requests: 0,
        queue: vec![],
    };

    (StatusCode::OK, Json(response))
}

/// Scheduler statistics response
#[derive(Debug, Clone, Serialize)]
pub struct SchedulerStatsResponse {
    pub total_requests: usize,
    pub completed_requests: usize,
    pub failed_requests: usize,
    pub average_latency_ms: f32,
    pub throughput_tokens_per_sec: f32,
    pub memory_usage_bytes: usize,
}

/// Get scheduler statistics
async fn scheduler_stats(State(state): State<AppState>) -> impl IntoResponse {
    let _scheduler = &state.scheduler;

    // TODO: Get actual stats from scheduler
    let response = SchedulerStatsResponse {
        total_requests: 0,
        completed_requests: 0,
        failed_requests: 0,
        average_latency_ms: 0.0,
        throughput_tokens_per_sec: 0.0,
        memory_usage_bytes: 0,
    };

    (StatusCode::OK, Json(response))
}

/// System metrics response
#[derive(Debug, Clone, Serialize)]
pub struct SystemMetricsResponse {
    pub cpu_usage_percent: f32,
    pub memory_total_bytes: usize,
    pub memory_used_bytes: usize,
    pub gpu_metrics: Vec<GpuMetrics>,
}

/// GPU metrics
#[derive(Debug, Clone, Serialize)]
pub struct GpuMetrics {
    pub gpu_id: usize,
    pub utilization_percent: f32,
    pub memory_total_bytes: usize,
    pub memory_used_bytes: usize,
    pub temperature_celsius: f32,
}

/// Get system metrics
///
/// CPU and memory come from a persistent `sysinfo::System` held in
/// `AppState` (`live_system`) -- CPU usage is a diff between refreshes, so a
/// freshly-constructed `System` would read 0 on every call; the long-lived
/// instance lets repeated calls see real deltas.
///
/// `gpu_metrics` stays an honest empty `Vec`: no GPU telemetry source is
/// wired (`sysinfo`'s `system` feature and `system-analysis` both report
/// only static GPU capability -- name, VRAM size, compute capability -- not
/// live utilization or temperature; that needs vendor-specific NVML
/// bindings, tracked as its own ROADMAP item, not built here). An empty Vec
/// is a true "nothing to report," unlike the previous hardcoded 0.0 scalar.
async fn system_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let (cpu_usage_percent, memory_total_bytes, memory_used_bytes) = {
        let mut sys = state.live_system.lock().unwrap_or_else(|e| e.into_inner());
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        (
            sys.global_cpu_usage(),
            sys.total_memory() as usize,
            sys.used_memory() as usize,
        )
    };

    let response = SystemMetricsResponse {
        cpu_usage_percent,
        memory_total_bytes,
        memory_used_bytes,
        gpu_metrics: vec![],
    };

    (StatusCode::OK, Json(response))
}

// ===== API Key Management =====

/// Create API key request
#[derive(Debug, Clone, Deserialize)]
pub struct CreateApiKeyRequest {
    /// Role for the API key (user, admin, llm)
    pub role: String,

    /// Optional expiration time in seconds from now
    #[serde(default)]
    pub expires_in_seconds: Option<i64>,

    /// Optional description/name for the key
    #[serde(default)]
    pub description: Option<String>,
}

/// Create API key response
#[derive(Debug, Clone, Serialize)]
pub struct CreateApiKeyResponse {
    /// The raw API key (only shown once!)
    pub api_key: String,

    /// The UUID of the created key record
    pub key_id: String,

    /// The role assigned to this key
    pub role: String,

    /// When the key expires (if set)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
}

/// Create a new API key
#[axum_macros::debug_handler]
async fn create_api_key(
    State(state): State<AppState>,
    Extension(caller): Extension<ApiKeyInfo>,
    Json(request): Json<CreateApiKeyRequest>,
) -> impl IntoResponse {
    // Validate role
    if !["user", "admin", "llm"].contains(&request.role.as_str()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": "Invalid role. Must be one of: user, admin, llm",
                    "type": "invalid_request_error",
                }
            })),
        )
            .into_response();
    }

    // Defense in depth: `routes()` already requires an admin caller to reach
    // this handler at all (`admin_check_middleware`), but that only ever
    // enforced "caller is admin", never "requested role is no higher than
    // caller's role" — the actual escalation this blocks. Checked again
    // here, independent of the router wiring, so a future looser role on
    // this route (or a direct unit-test call bypassing the router) cannot
    // reopen it silently.
    if role_rank(&request.role) > role_rank(&caller.role) {
        return (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": {
                    "message": "cannot create an API key with greater privilege than your own",
                    "type": "permission_error",
                    "code": "forbidden",
                }
            })),
        )
            .into_response();
    }

    // Generate a random API key (32 bytes = 64 hex chars)
    let api_key = {
        let mut rng = rand::thread_rng();
        let mut random_bytes = [0u8; 32];
        rng.fill_bytes(&mut random_bytes);
        format!("lb-{}", hex::encode(random_bytes))
    };

    // Compute SHA-256 hash of the key
    let mut hasher = Sha256::new();
    hasher.update(api_key.as_bytes());
    let key_hash = format!("{:x}", hasher.finalize());

    // Calculate expiration if requested
    let expires_at = request.expires_in_seconds.map(|seconds| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        now + seconds
    });

    // Insert into database
    let db_pool = match &state.db_pool {
        Some(pool) => pool,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": {
                        "message": "Database not configured",
                        "type": "service_unavailable",
                    }
                })),
            )
                .into_response();
        }
    };
    let client = match db_pool.get().await {
        Ok(client) => client,
        Err(e) => {
            eprintln!("DB connection error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": "Failed to connect to database",
                        "type": "internal_error",
                    }
                })),
            )
                .into_response();
        }
    };

    let result = client
        .query_one(
            r#"
            INSERT INTO api_keys (key_hash, role, description, expires_at)
            VALUES ($1, $2, $3,
                CASE 
                    WHEN $4::bigint IS NOT NULL 
                    THEN to_timestamp($4::bigint)
                    ELSE NULL
                END
            )
            RETURNING id, expires_at AT TIME ZONE 'UTC' AS expires_at
            "#,
            &[&key_hash, &request.role, &request.description, &expires_at],
        )
        .await;

    match result {
        Ok(record) => {
            let id: uuid::Uuid = record.get("id");
            let expires_at_value: Option<chrono::DateTime<chrono::Utc>> = record.get("expires_at");

            let response = CreateApiKeyResponse {
                api_key: api_key.clone(),
                key_id: id.to_string(),
                role: request.role.clone(),
                expires_at: expires_at_value.map(|ts| ts.to_rfc3339()),
            };

            println!(
                "✓ Created API key {} (role={}, expires={:?})",
                id, request.role, expires_at_value
            );

            (StatusCode::CREATED, Json(response)).into_response()
        }
        Err(e) => {
            eprintln!("Failed to create API key: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": "Failed to create API key",
                        "type": "internal_error",
                    }
                })),
            )
                .into_response()
        }
    }
}

/// Revoke an API key response
#[derive(Debug, Clone, Serialize)]
pub struct RevokeApiKeyResponse {
    pub key_id: String,
    pub revoked: bool,
}

/// Build the `{"error": {"message", "type", "code"?}}` body this module's
/// handlers share. Extracted once the duplicated-inline form pushed
/// `revoke_api_key`/`rotate_api_key` well past Codacy's per-method line
/// limit (each error path was ~9 lines of identical shape) -- not a
/// pre-existing-debt case like `run_jobs`' complexity finding (board item
/// M4 slice 1), so minimizing it for real rather than disputing the count.
fn api_error(status: StatusCode, message: &str, error_type: &str, code: Option<&str>) -> Response {
    let mut error = serde_json::json!({
        "message": message,
        "type": error_type,
    });
    if let Some(code) = code {
        error["code"] = serde_json::json!(code);
    }
    (status, Json(serde_json::json!({ "error": error }))).into_response()
}

/// Parse a path-extracted key id, or the 400 `revoke_api_key`/`rotate_api_key` share for one that isn't a UUID.
/// `(status, message, error_type)` -- small and `Copy`, unlike a full
/// `Response` (clippy::result_large_err: axum's `Response` is well past the
/// 128-byte threshold for an `Err` variant). The caller turns this into an
/// actual `Response` via `api_error` only once it has decided to return.
type ApiErrorParts = (StatusCode, &'static str, &'static str);

fn parse_key_id(id: &str) -> Result<uuid::Uuid, ApiErrorParts> {
    id.parse().map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            "key id must be a UUID",
            "invalid_request_error",
        )
    })
}

/// Get a pooled DB connection, or the 503/500 `revoke_api_key`/`rotate_api_key` share when no
/// database is configured or the pool can't hand one out.
async fn get_db_client(state: &AppState) -> Result<deadpool_postgres::Client, ApiErrorParts> {
    let db_pool = state.db_pool.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Database not configured",
        "service_unavailable",
    ))?;
    db_pool.get().await.map_err(|e| {
        eprintln!("DB connection error: {}", e);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to connect to database",
            "internal_error",
        )
    })
}

/// The 500 every internal step of `rotate_api_key`'s transaction shares,
/// logging `context` (which step failed) alongside the real error.
fn rotate_internal_error(e: impl std::fmt::Display, context: &str) -> Response {
    eprintln!("{context}: {e}");
    api_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "Failed to rotate API key",
        "internal_error",
        None,
    )
}

/// Revoke an API key.
///
/// Idempotent on an already-revoked key's id (0 rows updated reads as "not
/// found", matching the already-gone case rather than inventing a third
/// state). Refuses to revoke the caller's OWN key unless another unexpired,
/// unrevoked admin key exists -- otherwise this would be the request that
/// locks every admin out of every admin route, including the one that could
/// undo it. This does **not** generalize to "the last admin key anywhere":
/// only self-revocation is guarded, per the PM's scoping of this item.
async fn revoke_api_key(
    State(state): State<AppState>,
    Extension(caller): Extension<ApiKeyInfo>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let key_id = match parse_key_id(&id) {
        Ok(id) => id,
        Err((status, msg, ty)) => return api_error(status, msg, ty, None),
    };
    let client = match get_db_client(&state).await {
        Ok(client) => client,
        Err((status, msg, ty)) => return api_error(status, msg, ty, None),
    };

    if key_id == caller.api_key_id {
        let other_admin_key_exists = client
            .query_one(
                r#"SELECT EXISTS (
                    SELECT 1 FROM api_keys
                    WHERE role = 'admin'
                      AND id != $1
                      AND revoked_at IS NULL
                      AND (expires_at IS NULL OR expires_at > NOW())
                ) AS exists_flag"#,
                &[&key_id],
            )
            .await;

        match other_admin_key_exists {
            Ok(row) => {
                let exists: bool = row.get("exists_flag");
                if !exists {
                    return api_error(
                        StatusCode::CONFLICT,
                        "refusing to revoke your own key: it is the only unexpired admin key, \
                         and revoking it would lock out all admin access",
                        "conflict_error",
                        Some("would_lock_out_admin_access"),
                    );
                }
            }
            Err(e) => {
                eprintln!("Failed to check for another admin key: {}", e);
                return api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to revoke API key",
                    "internal_error",
                    None,
                );
            }
        }
    }

    let result = client
        .execute(
            "UPDATE api_keys SET revoked_at = NOW() WHERE id = $1 AND revoked_at IS NULL",
            &[&key_id],
        )
        .await;

    match result {
        Ok(0) => api_error(
            StatusCode::NOT_FOUND,
            "API key not found or already revoked",
            "not_found_error",
            None,
        ),
        Ok(_) => (
            StatusCode::OK,
            Json(RevokeApiKeyResponse {
                key_id: key_id.to_string(),
                revoked: true,
            }),
        )
            .into_response(),
        Err(e) => {
            eprintln!("Failed to revoke API key: {}", e);
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to revoke API key",
                "internal_error",
                None,
            )
        }
    }
}

/// Rotate an API key response
#[derive(Debug, Clone, Serialize)]
pub struct RotateApiKeyResponse {
    /// The new raw API key (only shown once!)
    pub api_key: String,
    pub key_id: String,
    pub role: String,
    pub rotated_from: String,
}

/// Rotate an API key: issue a replacement with the same role and
/// description, and revoke the original -- in one database transaction, so
/// a crash or connection drop between the two halves can never leave both
/// keys live (a privilege-duplication window) or neither live (a self-
/// inflicted lockout). No self-lockout check is needed here the way
/// `revoke_api_key` needs one: the replacement key exists, committed
/// atomically with the old one's revocation, before any caller could
/// observe the old key as gone.
async fn rotate_api_key(
    State(state): State<AppState>,
    Extension(_caller): Extension<ApiKeyInfo>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let old_id = match parse_key_id(&id) {
        Ok(id) => id,
        Err((status, msg, ty)) => return api_error(status, msg, ty, None),
    };
    let mut client = match get_db_client(&state).await {
        Ok(client) => client,
        Err((status, msg, ty)) => return api_error(status, msg, ty, None),
    };

    let txn = match client.transaction().await {
        Ok(txn) => txn,
        Err(e) => return rotate_internal_error(e, "Failed to start transaction"),
    };

    // Lock the row for the duration of the transaction: a concurrent rotate
    // or revoke on the same id waits rather than racing.
    let old_row = txn
        .query_opt(
            "SELECT role, description FROM api_keys \
             WHERE id = $1 AND revoked_at IS NULL FOR UPDATE",
            &[&old_id],
        )
        .await;

    let (role, description): (String, Option<String>) = match old_row {
        Ok(Some(row)) => (row.get("role"), row.get("description")),
        Ok(None) => {
            return api_error(
                StatusCode::NOT_FOUND,
                "API key not found or already revoked",
                "not_found_error",
                None,
            );
        }
        Err(e) => return rotate_internal_error(e, "Failed to read API key"),
    };

    let api_key = {
        let mut rng = rand::rng();
        let mut random_bytes = [0u8; 32];
        rng.fill_bytes(&mut random_bytes);
        format!("lb-{}", hex::encode(random_bytes))
    };
    let mut hasher = Sha256::new();
    hasher.update(api_key.as_bytes());
    let key_hash = format!("{:x}", hasher.finalize());

    let new_id: uuid::Uuid = match txn
        .query_one(
            "INSERT INTO api_keys (key_hash, role, description) VALUES ($1, $2, $3) \
             RETURNING id",
            &[&key_hash, &role, &description],
        )
        .await
    {
        Ok(row) => row.get("id"),
        Err(e) => return rotate_internal_error(e, "Failed to insert rotated API key"),
    };

    if let Err(e) = txn
        .execute(
            "UPDATE api_keys SET revoked_at = NOW() WHERE id = $1",
            &[&old_id],
        )
        .await
    {
        return rotate_internal_error(e, "Failed to revoke the rotated-out API key");
    }

    if let Err(e) = txn.commit().await {
        return rotate_internal_error(e, "Failed to commit API key rotation");
    }

    (
        StatusCode::CREATED,
        Json(RotateApiKeyResponse {
            api_key,
            key_id: new_id.to_string(),
            role,
            rotated_from: old_id.to_string(),
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::auth_middleware::ApiKeyInfo;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn state() -> AppState {
        use crate::engine::{MemoryAwareConfig, MemoryAwareScheduler};
        AppState {
            scheduler: std::sync::Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default())),
            config: crate::api::ApiConfig::default(),
            db_pool: None,
            inference_tx: None,
            chat_template: None,
            eos_monitor: std::sync::Arc::new(crate::engine::eos_monitor::EosMonitor::default()),
            security_sinks: Vec::new(),
            cache_stats: std::sync::Arc::new(std::sync::Mutex::new(
                crate::cache::PrefixCacheStats::default(),
            )),
            live_system: crate::api::new_shared_system_metrics(),
            queue_depth: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    fn key_info(role: &str) -> ApiKeyInfo {
        ApiKeyInfo {
            api_key_id: uuid::Uuid::new_v4(),
            role: role.to_string(),
        }
    }

    /// Reproduces the privilege-escalation gap: with no caller-role check at
    /// all, a "user" key's request for `role: "admin"` sailed past this
    /// handler and only failed later for an unrelated reason (no database
    /// configured in the test). A 503 here — not a 403 — would mean the
    /// escalation path is still open; the test is written to fail for that
    /// exact reason before the fix.
    #[tokio::test]
    async fn create_api_key_refuses_a_user_key_requesting_admin_role() {
        let req = Request::builder()
            .method("POST")
            .uri("/v1/lightbulb/admin/api-keys")
            .header("content-type", "application/json")
            .extension(key_info("user"))
            .body(Body::from(r#"{"role":"admin"}"#))
            .unwrap();

        let response = routes().with_state(state()).oneshot(req).await.unwrap();

        assert_eq!(
            response.status(),
            StatusCode::FORBIDDEN,
            "a user-role key must not be able to mint an admin-role key"
        );
    }

    #[tokio::test]
    async fn create_api_key_refuses_an_llm_key_requesting_admin_role() {
        let req = Request::builder()
            .method("POST")
            .uri("/v1/lightbulb/admin/api-keys")
            .header("content-type", "application/json")
            .extension(key_info("llm"))
            .body(Body::from(r#"{"role":"admin"}"#))
            .unwrap();

        let response = routes().with_state(state()).oneshot(req).await.unwrap();

        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    /// A caller requesting a role AT or BELOW their own rank must clear the
    /// privilege check and reach the (here, unconfigured) database — 503,
    /// not 403. This is what distinguishes "the check ran and passed" from
    /// "the check never ran": a handler with no check at all would also
    /// return 503 for the admin case below, but the two preceding REFUSES
    /// tests prove the check is live, so this one's 503 means passthrough,
    /// not absence.
    #[tokio::test]
    async fn create_api_key_allows_an_admin_key_requesting_admin_role() {
        let req = Request::builder()
            .method("POST")
            .uri("/v1/lightbulb/admin/api-keys")
            .header("content-type", "application/json")
            .extension(key_info("admin"))
            .body(Body::from(r#"{"role":"admin"}"#))
            .unwrap();

        let response = routes().with_state(state()).oneshot(req).await.unwrap();

        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "an admin key requesting admin must pass the privilege check and \
             reach the no-database branch, not be forbidden"
        );
    }

    /// `admin_check_middleware` gates every admin route on the caller being
    /// admin, so a non-admin caller never reaches `create_api_key`'s own
    /// privilege check at all (covered by
    /// `admin_routes_refuse_a_non_admin_caller_on_every_route`) — only an
    /// admin caller can reach it, and only a REQUEST for a role ranked at or
    /// below the caller's clears it. This exercises that admin-requesting-a
    /// -lower-role case: it must pass the privilege check (not 403) and
    /// reach the no-database branch (503).
    #[tokio::test]
    async fn create_api_key_allows_an_admin_key_requesting_llm_role() {
        let req = Request::builder()
            .method("POST")
            .uri("/v1/lightbulb/admin/api-keys")
            .header("content-type", "application/json")
            .extension(key_info("admin"))
            .body(Body::from(r#"{"role":"llm"}"#))
            .unwrap();

        let response = routes().with_state(state()).oneshot(req).await.unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// Router-level, not just handler-level: proves `admin_check_middleware`
    /// is actually attached to `routes()`, the same `Router` `ApiServer`
    /// serves — a unit test of the middleware function alone would pass even
    /// if nobody ever wired it to a route (which was the actual bug: the
    /// function existed, `grep -rn "admin_check_middleware"` found zero call
    /// sites besides its own definition).
    #[tokio::test]
    async fn admin_routes_refuse_a_non_admin_caller_on_every_route() {
        let cases: &[(&str, &str)] = &[
            ("GET", "/v1/lightbulb/admin/cache/stats"),
            ("POST", "/v1/lightbulb/admin/cache/clear"),
            ("GET", "/v1/lightbulb/admin/scheduler/queue"),
            ("GET", "/v1/lightbulb/admin/scheduler/stats"),
            ("GET", "/v1/lightbulb/admin/metrics"),
            ("POST", "/v1/lightbulb/admin/api-keys"),
        ];

        for (method, uri) in cases {
            let req = Request::builder()
                .method(*method)
                .uri(*uri)
                .header("content-type", "application/json")
                .extension(key_info("user"))
                .body(Body::from("{}"))
                .unwrap();

            let response = routes().with_state(state()).oneshot(req).await.unwrap();

            assert_eq!(
                response.status(),
                StatusCode::FORBIDDEN,
                "{method} {uri} must refuse a non-admin caller"
            );
        }
    }

    #[tokio::test]
    async fn admin_routes_allow_an_admin_caller_through_to_the_handler() {
        // Routes that need no database to answer 200.
        let cases: &[(&str, &str)] = &[
            ("GET", "/v1/lightbulb/admin/cache/stats"),
            ("POST", "/v1/lightbulb/admin/cache/clear"),
            ("GET", "/v1/lightbulb/admin/scheduler/queue"),
            ("GET", "/v1/lightbulb/admin/scheduler/stats"),
            ("GET", "/v1/lightbulb/admin/metrics"),
        ];

        for (method, uri) in cases {
            let req = Request::builder()
                .method(*method)
                .uri(*uri)
                .header("content-type", "application/json")
                .extension(key_info("admin"))
                .body(Body::from("{}"))
                .unwrap();

            let response = routes().with_state(state()).oneshot(req).await.unwrap();

            assert_eq!(
                response.status(),
                StatusCode::OK,
                "{method} {uri} must let an admin caller through"
            );
        }
    }

    /// Board item M4 slice 1: `cache_stats` must report whatever is in
    /// `AppState.cache_stats`, not a hardcoded mock. Populate the shared
    /// handle with values a real model-runner thread would never produce by
    /// coincidence (asymmetric, non-zero, non-default), then assert the HTTP
    /// response carries exactly those values through the real router.
    #[tokio::test]
    async fn cache_stats_reports_the_shared_handles_real_values_not_a_mock() {
        let app_state = state();
        {
            let mut guard = app_state.cache_stats.lock().unwrap();
            guard.hits = 7;
            guard.misses = 3;
            guard.evictions = 2;
            guard.entries = 5;
            guard.size_bytes = 4096;
        }

        let req = Request::builder()
            .method("GET")
            .uri("/v1/lightbulb/admin/cache/stats")
            .extension(key_info("admin"))
            .body(Body::empty())
            .unwrap();
        let response = routes().with_state(app_state).oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(body["total_entries"], 5, "full body: {body:?}");
        assert_eq!(body["total_size_bytes"], 4096, "full body: {body:?}");
        assert_eq!(body["eviction_count"], 2, "full body: {body:?}");
        // hit_rate = hits / (hits + misses) = 7 / 10
        assert!(
            (body["hit_rate"].as_f64().unwrap() - 0.7).abs() < 1e-6,
            "full body: {body:?}"
        );
    }

    /// `system_metrics` must report the machine's real total memory (a
    /// hardcoded-zero stub would read 0 here), and `cpu_usage_percent` must
    /// be a real, finite reading rather than a fixed sentinel. `gpu_metrics`
    /// must stay an honest empty `Vec`, not a mocked entry, since no GPU
    /// telemetry source is wired.
    #[tokio::test]
    async fn system_metrics_reports_real_memory_not_a_hardcoded_zero() {
        let expected_total_bytes = sysinfo::System::new_all().total_memory();

        let req = Request::builder()
            .method("GET")
            .uri("/v1/lightbulb/admin/metrics")
            .extension(key_info("admin"))
            .body(Body::empty())
            .unwrap();
        let response = routes().with_state(state()).oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(
            body["memory_total_bytes"].as_u64().unwrap(),
            expected_total_bytes,
            "full body: {body:?}"
        );
        assert!(
            body["memory_used_bytes"].as_u64().unwrap() > 0,
            "full body: {body:?}"
        );
        let cpu = body["cpu_usage_percent"].as_f64().unwrap();
        assert!(cpu.is_finite() && cpu >= 0.0, "full body: {body:?}");
        assert_eq!(
            body["gpu_metrics"].as_array().unwrap().len(),
            0,
            "full body: {body:?}"
        );
    }
}
