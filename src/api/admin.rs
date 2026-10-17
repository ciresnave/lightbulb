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
use crate::api::persistant_migrate;
use persistant_relational::{Isolation, RelationalBackend, RelationalTx};

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

/// Scheduler queue response.
///
/// All three fields are `None`/empty, always, with `unavailable_reason`
/// explaining why -- NOT the `0`/`[]` a previous version of this returned,
/// which looked like a real (empty) queue reading rather than no reading
/// at all. `MemoryAwareScheduler`'s `priority_queue`/`slot_pool` exist
/// (`src/engine/memory_aware_scheduler.rs`) but nothing in the live
/// request path calls `submit_priority_request`/`allocate_pending_requests`
/// -- only `check_admission`, a pure memory-budget check, is used. There
/// is no real priority-queued-request flow to report a depth for.
#[derive(Debug, Clone, Serialize)]
pub struct SchedulerQueueResponse {
    pub pending_requests: Option<usize>,
    pub running_requests: Option<usize>,
    pub queue: Option<Vec<QueuedRequest>>,
    pub unavailable_reason: Option<String>,
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

    let response = SchedulerQueueResponse {
        pending_requests: None,
        running_requests: None,
        queue: None,
        unavailable_reason: Some(
            "the scheduler's priority queue is not populated by the live request path -- \
             there is no real per-priority queue depth to report"
                .to_string(),
        ),
    };

    (StatusCode::OK, Json(response))
}

/// Scheduler statistics response.
///
/// Only `memory_usage_bytes` is a real reading (same live `sysinfo`
/// sampler `GET /v1/lightbulb/admin/metrics` already reads). The other
/// five are `None`, always, with `unavailable_reason` explaining why --
/// NOT the `0`/`0.0` a previous version of this returned, which looked
/// like real zero-valued readings rather than no reading at all. Request
/// count, latency and throughput ARE tracked, just not here: `GET
/// /metrics` (Phase 4 observability, Prometheus format) records
/// `lightbulb_http_requests_total` (a counter, labeled route/method/
/// status) and `lightbulb_http_request_duration_seconds` (a histogram) --
/// reading a live number back out of the installed `PrometheusHandle`
/// to answer with here isn't something that crate's API supports (only
/// scrape-format text rendering), and duplicating those counters in a
/// second, parallel shape was rejected (PM, 2026-10-16) as the wrong kind
/// of effort.
#[derive(Debug, Clone, Serialize)]
pub struct SchedulerStatsResponse {
    pub total_requests: Option<usize>,
    pub completed_requests: Option<usize>,
    pub failed_requests: Option<usize>,
    pub average_latency_ms: Option<f32>,
    pub throughput_tokens_per_sec: Option<f32>,
    pub memory_usage_bytes: usize,
    pub unavailable_reason: Option<String>,
}

/// Get scheduler statistics
async fn scheduler_stats(State(state): State<AppState>) -> impl IntoResponse {
    let _scheduler = &state.scheduler;

    let memory_usage_bytes = {
        let mut sys = state.live_system.lock().unwrap_or_else(|e| e.into_inner());
        sys.refresh_memory();
        sys.used_memory() as usize
    };

    let response = SchedulerStatsResponse {
        total_requests: None,
        completed_requests: None,
        failed_requests: None,
        average_latency_ms: None,
        throughput_tokens_per_sec: None,
        memory_usage_bytes,
        unavailable_reason: Some(
            "request count, latency and throughput are not tracked at this endpoint -- see \
             GET /metrics (lightbulb_http_requests_total, lightbulb_http_request_duration_seconds)"
                .to_string(),
        ),
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

/// `create_api_key`'s two request-shape checks: the requested role is one of
/// the three known values, and it is no higher than the caller's own role
/// (`routes()`'s `admin_check_middleware` only enforces "caller is admin",
/// never "requested role is no higher than caller's role" -- checked again
/// here, independent of the router wiring, so a future looser role on this
/// route, or a direct unit-test call bypassing the router, cannot reopen
/// this escalation silently). `Some(response)` short-circuits the caller.
fn validate_create_api_key_role(requested_role: &str, caller_role: &str) -> Option<Response> {
    if !["user", "admin", "llm"].contains(&requested_role) {
        return Some(
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": "Invalid role. Must be one of: user, admin, llm",
                        "type": "invalid_request_error",
                    }
                })),
            )
                .into_response(),
        );
    }
    if role_rank(requested_role) > role_rank(caller_role) {
        return Some(
            (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({
                    "error": {
                        "message": "cannot create an API key with greater privilege than your own",
                        "type": "permission_error",
                        "code": "forbidden",
                    }
                })),
            )
                .into_response(),
        );
    }
    None
}

/// Create a new API key
#[axum_macros::debug_handler]
async fn create_api_key(
    State(state): State<AppState>,
    Extension(caller): Extension<ApiKeyInfo>,
    Json(request): Json<CreateApiKeyRequest>,
) -> impl IntoResponse {
    if let Some(response) = validate_create_api_key_role(&request.role, &caller.role) {
        return response;
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

    // Insert into database. PersistAnt adoption Step 3
    // (docs/PERSISTANT_ADOPTION_DESIGN.md): this call site is one of the 5
    // single-statement writes ported to `AppState::pg_backend` for this
    // step (the design doc's own list named only 3; see its correction
    // note). `insert_api_key_row` carries the `::text` casts and decoding
    // this needs.
    let (id, expires_at_value) = match insert_api_key_row(
        &state,
        &key_hash,
        &request.role,
        request.description.as_deref(),
        expires_at,
    )
    .await
    {
        Ok(row) => row,
        Err((status, msg, ty)) => return api_error(status, msg, ty, None),
    };

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

/// Inserts the new key row via `AppState::pg_backend`, decoding the
/// `RETURNING` row. `id` and `expires_at` are cast to `::text` in the SQL
/// itself -- `persistant_relational::Value` has no `Uuid` or timestamp
/// variant (docs/AUTH_PERSISTANT_ADOPTION_FIT.md §2) -- then parsed back:
/// `id` via `persistant_migrate::value_to_uuid`, `expires_at` via
/// `to_char(..., 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')` server-side so the text
/// is RFC3339-parseable (Postgres's own default `::text` cast of a
/// `timestamptz` is NOT RFC3339 -- e.g. a space instead of `T` -- and would
/// fail `DateTime::parse_from_rfc3339`).
const INSERT_API_KEY_SQL: &str = r#"
    INSERT INTO api_keys (key_hash, role, description, expires_at)
    VALUES ($1, $2, $3,
        CASE
            WHEN $4::bigint IS NOT NULL
            THEN to_timestamp($4::bigint)
            ELSE NULL
        END
    )
    RETURNING
        id::text AS id,
        to_char(
            expires_at AT TIME ZONE 'UTC',
            'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
        ) AS expires_at
"#;

async fn insert_api_key_row(
    state: &AppState,
    key_hash: &str,
    role: &str,
    description: Option<&str>,
    expires_at: Option<i64>,
) -> Result<(uuid::Uuid, Option<chrono::DateTime<chrono::Utc>>), ApiErrorParts> {
    let backend = state.pg_backend.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "Database not configured",
        "service_unavailable",
    ))?;

    let description_value = description.map_or(persistant_relational::Value::Null, |d| {
        persistant_relational::Value::Text(d.to_string())
    });
    let expires_at_value = expires_at.map_or(persistant_relational::Value::Null, |v| {
        persistant_relational::Value::Int(v)
    });

    let rows = backend
        .query(
            INSERT_API_KEY_SQL,
            &[
                persistant_relational::Value::Text(key_hash.to_string()),
                persistant_relational::Value::Text(role.to_string()),
                description_value,
                expires_at_value,
            ],
        )
        .await
        .map_err(|e| {
            eprintln!("Failed to create API key: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to create API key",
                "internal_error",
            )
        })?;

    decode_inserted_key_row(rows.first())
}

const INSERT_KEY_DECODE_ERROR: ApiErrorParts = (
    StatusCode::INTERNAL_SERVER_ERROR,
    "Failed to create API key",
    "internal_error",
);

/// Decodes `insert_api_key_row`'s `RETURNING` row: `id` (always present) and
/// `expires_at` (`Value::Null` when the key has no expiry).
fn decode_inserted_key_row(
    row: Option<&persistant_relational::Row>,
) -> Result<(uuid::Uuid, Option<chrono::DateTime<chrono::Utc>>), ApiErrorParts> {
    let row = row.ok_or_else(|| {
        eprintln!("Failed to create API key: INSERT ... RETURNING produced no row");
        INSERT_KEY_DECODE_ERROR
    })?;

    let id_value = row.values.first().ok_or_else(|| {
        eprintln!("Failed to create API key: RETURNING row missing id column");
        INSERT_KEY_DECODE_ERROR
    })?;
    let id = persistant_migrate::value_to_uuid(id_value).map_err(|e| {
        eprintln!("Failed to create API key: {e}");
        INSERT_KEY_DECODE_ERROR
    })?;

    let expires_at = match row.values.get(1) {
        Some(persistant_relational::Value::Text(s)) => Some(
            chrono::DateTime::parse_from_rfc3339(s)
                .map_err(|e| {
                    eprintln!("Failed to create API key: expires_at not RFC3339: {e}");
                    INSERT_KEY_DECODE_ERROR
                })?
                .with_timezone(&chrono::Utc),
        ),
        Some(persistant_relational::Value::Null) | None => None,
        Some(_) => {
            eprintln!("Failed to create API key: expires_at column has an unexpected type");
            return Err(INSERT_KEY_DECODE_ERROR);
        }
    };

    Ok((id, expires_at))
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
/// Interprets the `SELECT EXISTS (...)` result from
/// `check_other_admin_key_exists`'s query: `Some(response)` is the response
/// to short-circuit the caller with (the 409 lockout refusal or a 500 on a
/// DB error), `None` means it's safe to proceed with the revoke.
fn interpret_other_admin_key_result(
    rows: Result<Vec<persistant_relational::Row>, persistant_relational::RelError>,
) -> Option<Response> {
    match rows {
        Ok(rows) => {
            let exists = matches!(
                rows.first().and_then(|row| row.values.first()),
                Some(persistant_relational::Value::Bool(true))
            );
            if exists {
                None
            } else {
                Some(api_error(
                    StatusCode::CONFLICT,
                    "refusing to revoke your own key: it is the only unexpired admin key, \
                     and revoking it would lock out all admin access",
                    "conflict_error",
                    Some("would_lock_out_admin_access"),
                ))
            }
        }
        Err(e) => {
            eprintln!("Failed to check for another admin key: {e}");
            Some(api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to revoke API key",
                "internal_error",
                None,
            ))
        }
    }
}

/// `revoke_api_key`'s self-lockout check: is there another live, unexpired
/// admin key besides the one being revoked? `Some(response)` short-circuits
/// the caller with the 409 lockout refusal or a 500 on a DB error; `None`
/// means it's safe to proceed with the revoke.
///
/// PersistAnt adoption Step 2 (docs/PERSISTANT_ADOPTION_DESIGN.md): this is
/// one of the 2 read-only call sites ported to `AppState::pg_backend` for
/// this step (the design doc's own line-number list for this step was
/// wrong -- see the doc's correction note). Everything else in this module
/// still goes through `state.db_pool` until a later step.
///
/// `id != $1` doesn't type-check against a `Value::Text` parameter --
/// Postgres has no `uuid <> text` operator (confirmed on a real run:
/// "operator does not exist: uuid <> text"). Casting the COLUMN to `::text`
/// (not the parameter) matches `auth_middleware.rs::lookup_api_key_by_hash`'s
/// existing pattern for the same `Value`-has-no-Uuid-variant gap.
async fn check_other_admin_key_exists(state: &AppState, key_id: uuid::Uuid) -> Option<Response> {
    let backend = match state.pg_backend.as_ref() {
        Some(backend) => backend,
        None => {
            return Some(api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Database not configured",
                "service_unavailable",
                None,
            ));
        }
    };

    let rows = backend
        .query(
            r#"SELECT EXISTS (
                SELECT 1 FROM api_keys
                WHERE role = 'admin'
                  AND id::text != $1
                  AND revoked_at IS NULL
                  AND (expires_at IS NULL OR expires_at > NOW())
            ) AS exists_flag"#,
            &[persistant_migrate::uuid_to_value(key_id)],
        )
        .await;

    interpret_other_admin_key_result(rows)
}

async fn revoke_api_key(
    State(state): State<AppState>,
    Extension(caller): Extension<ApiKeyInfo>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let key_id = match parse_key_id(&id) {
        Ok(id) => id,
        Err((status, msg, ty)) => return api_error(status, msg, ty, None),
    };

    if key_id == caller.api_key_id
        && let Some(response) = check_other_admin_key_exists(&state, key_id).await
    {
        return response;
    }

    // PersistAnt adoption Step 3 (docs/PERSISTANT_ADOPTION_DESIGN.md): one
    // of the 5 single-statement writes ported to `AppState::pg_backend` for
    // this step.
    let backend = match state.pg_backend.as_ref() {
        Some(backend) => backend,
        None => {
            return api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Database not configured",
                "service_unavailable",
                None,
            );
        }
    };

    let result = backend
        .execute(
            "UPDATE api_keys SET revoked_at = NOW() WHERE id::text = $1 AND revoked_at IS NULL",
            &[persistant_migrate::uuid_to_value(key_id)],
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
            eprintln!("Failed to revoke API key: {e}");
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
/// `rotate_api_key`'s row-locked read: the role/description of the key
/// being rotated, or `Ok(None)` if it's already gone/revoked (not an error
/// -- the caller turns that into a 404). `id::text = $1` is the same
/// column-cast pattern `check_other_admin_key_exists` uses for the same
/// `Value`-has-no-Uuid-variant gap.
async fn lock_api_key_for_rotation(
    tx: &mut persistant_sqlx::PgTx<'_>,
    old_id: uuid::Uuid,
) -> Result<Option<(String, Option<String>)>, persistant_relational::RelError> {
    let rows = tx
        .query(
            "SELECT role, description FROM api_keys \
             WHERE id::text = $1 AND revoked_at IS NULL FOR UPDATE",
            &[persistant_migrate::uuid_to_value(old_id)],
        )
        .await?;
    let Some(row) = rows.first() else {
        return Ok(None);
    };
    let role = row
        .text(0)
        .ok_or_else(|| persistant_relational::RelError::Unsupported("role column not text".into()))?
        .to_string();
    let description = row.text(1).map(str::to_string);
    Ok(Some((role, description)))
}

/// `rotate_api_key`'s replacement-key insert, inside the same transaction
/// as the lock above. `RETURNING id::text` / `value_to_uuid` is the same
/// cast pattern Step 3's `create_api_key` uses.
async fn insert_rotated_key(
    tx: &mut persistant_sqlx::PgTx<'_>,
    key_hash: &str,
    role: &str,
    description: Option<&str>,
) -> Result<uuid::Uuid, persistant_relational::RelError> {
    let description_value = description.map_or(persistant_relational::Value::Null, |d| {
        persistant_relational::Value::Text(d.to_string())
    });
    let rows = tx
        .query(
            "INSERT INTO api_keys (key_hash, role, description) VALUES ($1, $2, $3) \
             RETURNING id::text AS id",
            &[
                persistant_relational::Value::Text(key_hash.to_string()),
                persistant_relational::Value::Text(role.to_string()),
                description_value,
            ],
        )
        .await?;
    let id_value = rows.first().and_then(|row| row.values.first()).ok_or(
        persistant_relational::RelError::Unsupported("INSERT ... RETURNING produced no id".into()),
    )?;
    persistant_migrate::value_to_uuid(id_value)
        .map_err(persistant_relational::RelError::Unsupported)
}

async fn rotate_api_key(
    State(state): State<AppState>,
    Extension(_caller): Extension<ApiKeyInfo>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let old_id = match parse_key_id(&id) {
        Ok(id) => id,
        Err((status, msg, ty)) => return api_error(status, msg, ty, None),
    };
    let backend = match state.pg_backend.as_ref() {
        Some(backend) => backend,
        None => {
            return api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Database not configured",
                "service_unavailable",
                None,
            );
        }
    };

    let mut txn = match backend.begin(Isolation::ReadCommitted).await {
        Ok(txn) => txn,
        Err(e) => return rotate_internal_error(e, "Failed to start transaction"),
    };

    // Lock the row for the duration of the transaction: a concurrent rotate
    // or revoke on the same id waits rather than racing.
    let (role, description) = match lock_api_key_for_rotation(&mut txn, old_id).await {
        Ok(Some(row)) => row,
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

    let new_id = match insert_rotated_key(&mut txn, &key_hash, &role, description.as_deref()).await
    {
        Ok(id) => id,
        Err(e) => return rotate_internal_error(e, "Failed to insert rotated API key"),
    };

    if let Err(e) = txn
        .execute(
            "UPDATE api_keys SET revoked_at = NOW() WHERE id::text = $1",
            &[persistant_migrate::uuid_to_value(old_id)],
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
            pg_backend: None,
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

    /// Breaking change (version bump per project rule): the hardcoded-zero
    /// stub this replaces returned `0`/`0.0`/`[]`, which LOOKS like a real
    /// (idle) reading. Every field with no real backing must now be
    /// `null`, with `unavailable_reason` explaining why -- a caller
    /// checking `== 0` to mean "idle" would have been silently wrong
    /// either way, but `null` at least cannot be mistaken for a number.
    #[tokio::test]
    async fn scheduler_queue_reports_unavailable_not_zero() {
        let req = Request::builder()
            .method("GET")
            .uri("/v1/lightbulb/admin/scheduler/queue")
            .extension(key_info("admin"))
            .body(Body::empty())
            .unwrap();
        let response = routes().with_state(state()).oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert!(body["pending_requests"].is_null(), "full body: {body:?}");
        assert!(body["running_requests"].is_null(), "full body: {body:?}");
        assert!(body["queue"].is_null(), "full body: {body:?}");
        assert!(
            body["unavailable_reason"].is_string(),
            "full body: {body:?}"
        );
    }

    /// Same discipline as the queue endpoint above, but `memory_usage_bytes`
    /// is the one field that IS real now (same `live_system` source
    /// `system_metrics` already reads) -- it must stay a real number, not
    /// also go `null`, while every other field does.
    #[tokio::test]
    async fn scheduler_stats_reports_real_memory_and_unavailable_for_the_rest() {
        let expected_used_bytes = sysinfo::System::new_all().used_memory();

        let req = Request::builder()
            .method("GET")
            .uri("/v1/lightbulb/admin/scheduler/stats")
            .extension(key_info("admin"))
            .body(Body::empty())
            .unwrap();
        let response = routes().with_state(state()).oneshot(req).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert!(body["total_requests"].is_null(), "full body: {body:?}");
        assert!(body["completed_requests"].is_null(), "full body: {body:?}");
        assert!(body["failed_requests"].is_null(), "full body: {body:?}");
        assert!(body["average_latency_ms"].is_null(), "full body: {body:?}");
        assert!(
            body["throughput_tokens_per_sec"].is_null(),
            "full body: {body:?}"
        );
        assert!(
            body["unavailable_reason"].is_string(),
            "full body: {body:?}"
        );

        // Real memory reading, not null and not a hardcoded zero. Exact
        // equality would be flaky (another process could change used
        // memory between the two samples); a generous tolerance pins
        // "real and in the right ballpark" without asserting a snapshot.
        let actual = body["memory_usage_bytes"].as_u64().unwrap();
        let tolerance = expected_used_bytes / 10; // 10%
        assert!(
            actual.abs_diff(expected_used_bytes) <= tolerance,
            "expected memory_usage_bytes near {expected_used_bytes}, got {actual}: full body: {body:?}"
        );
    }
}
