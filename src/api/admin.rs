//! Admin API Module
//!
//! Provides administrative endpoints for system management, cache control,
//! scheduler inspection, and metrics.

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::NaiveDateTime;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::api::AppState;

/// Create admin API routes
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/lightbulb/admin/cache/stats", get(cache_stats))
        .route("/v1/lightbulb/admin/cache/clear", post(cache_clear))
        .route("/v1/lightbulb/admin/scheduler/queue", get(scheduler_queue))
        .route("/v1/lightbulb/admin/scheduler/stats", get(scheduler_stats))
        .route("/v1/lightbulb/admin/metrics", get(system_metrics))
        .route("/v1/lightbulb/admin/api-keys", post(create_api_key))
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
async fn cache_stats(State(state): State<AppState>) -> impl IntoResponse {
    // Get stats from scheduler's cache
    let scheduler = &state.scheduler;

    // TODO: Implement actual cache stats retrieval
    // For now, return mock data
    let response = CacheStatsResponse {
        total_entries: 0,
        total_size_bytes: 0,
        hit_rate: 0.0,
        eviction_count: 0,
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
async fn system_metrics(State(_state): State<AppState>) -> impl IntoResponse {
    // TODO: Get actual system metrics via system-analysis crate
    let response = SystemMetricsResponse {
        cpu_usage_percent: 0.0,
        memory_total_bytes: 0,
        memory_used_bytes: 0,
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
