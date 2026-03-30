//! Middleware Module
//!
//! Provides authentication, rate limiting, and audit logging middleware.

use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::time::Instant;

use crate::api::AppState;

/// Error response
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error detail
#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

/// Authentication middleware
pub async fn auth_middleware(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Response {
    // Extract Bearer token from Authorization header
    let auth_header = request
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok());

    let token = if let Some(auth) = auth_header {
        if auth.starts_with("Bearer ") {
            Some(&auth[7..])
        } else {
            None
        }
    } else {
        None
    };

    // Validate token
    if let Some(token) = token {
        // Hash the provided token and look up the API key in Postgres
        let mut hasher = Sha256::new();
        hasher.update(token.as_bytes());
        let digest = hasher.finalize();
        let key_hash = digest
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();

        // Query the database for a matching, non-expired key
        let client = match state.db_pool.get().await {
            Ok(client) => client,
            Err(e) => {
                eprintln!("DB connection error: {}", e);
                let error = ErrorResponse {
                    error: ErrorDetail {
                        message: "Internal server error".to_string(),
                        r#type: "server_error".to_string(),
                        code: Some("internal_error".to_string()),
                    },
                };
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response();
            }
        };

        match client
            .query_opt(
                r#"SELECT id, role
                   FROM api_keys
                   WHERE key_hash = $1
                     AND (expires_at IS NULL OR expires_at > NOW())
                   LIMIT 1"#,
                &[&key_hash],
            )
            .await
        {
            Ok(Some(row)) => {
                let id: uuid::Uuid = row.get("id");
                let role: String = row.get("role");

                // Insert API key info for handlers and audit
                request.extensions_mut().insert(ApiKeyInfo {
                    api_key_id: id,
                    role: role,
                });

                return next.run(request).await;
            }
            Ok(None) => {
                // Not found or expired
                let error = ErrorResponse {
                    error: ErrorDetail {
                        message: "Invalid or expired API key".to_string(),
                        r#type: "authentication_error".to_string(),
                        code: Some("unauthorized".to_string()),
                    },
                };

                return (StatusCode::UNAUTHORIZED, Json(error)).into_response();
            }
            Err(e) => {
                // Database error
                eprintln!("DB error checking API key: {}", e);
                let error = ErrorResponse {
                    error: ErrorDetail {
                        message: "Internal server error".to_string(),
                        r#type: "server_error".to_string(),
                        code: Some("internal_error".to_string()),
                    },
                };

                return (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response();
            }
        }
    } else {
        // Unauthorized
        let error = ErrorResponse {
            error: ErrorDetail {
                message: "Missing or invalid authorization header".to_string(),
                r#type: "authentication_error".to_string(),
                code: Some("unauthorized".to_string()),
            },
        };

        (StatusCode::UNAUTHORIZED, Json(error)).into_response()
    }
}

/// API key information stored in request extensions
#[derive(Debug, Clone)]
pub struct ApiKeyInfo {
    pub api_key_id: uuid::Uuid,
    pub role: String,
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    // Get API key info from extensions
    let api_key_info = request.extensions().get::<ApiKeyInfo>().cloned();

    if let Some(key_info) = api_key_info {
        // Check and increment fixed-window (per-minute) counters in Postgres.
        let limit = state.config.rate_limit_per_minute as i64;
        let api_key_id = key_info.api_key_id;

        // Perform an atomic upsert and return the new request_count for the current minute window
        let client = match state.db_pool.get().await {
            Ok(client) => client,
            Err(e) => {
                eprintln!("DB connection error: {}", e);
                // On DB error, be conservative and allow the request (don't block traffic for DB hiccups)
                return next.run(request).await;
            }
        };

        match client
            .query_one(
                r#"
                INSERT INTO api_key_usage (api_key_id, window_start, request_count)
                VALUES ($1, date_trunc('minute', now())::timestamp, 1)
                ON CONFLICT (api_key_id, window_start) DO UPDATE
                  SET request_count = api_key_usage.request_count + 1
                RETURNING request_count
                "#,
                &[&api_key_id],
            )
            .await
        {
            Ok(row) => {
                let count: i32 = row.get("request_count");
                let count = count as i64;
                if count > limit {
                    let error = ErrorResponse {
                        error: ErrorDetail {
                            message: "Rate limit exceeded".to_string(),
                            r#type: "rate_limit_error".to_string(),
                            code: Some("rate_limit_exceeded".to_string()),
                        },
                    };

                    return (StatusCode::TOO_MANY_REQUESTS, Json(error)).into_response();
                }
            }
            Err(e) => {
                eprintln!("Rate limit DB error: {}", e);
                // On DB error, be conservative and allow the request (don't block traffic for DB hiccups)
            }
        }

        next.run(request).await
    } else {
        next.run(request).await
    }
}

/// Audit logging middleware
pub async fn audit_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    if !state.config.enable_audit_log {
        return next.run(request).await;
    }

    // Extract request info
    let method = request.method().to_string();
    let uri = request.uri().path().to_string();
    let api_key_info = request.extensions().get::<ApiKeyInfo>().cloned();

    let start = Instant::now();

    // Process request
    let response = next.run(request).await;

    let latency_ms = start.elapsed().as_millis() as i32;
    let status_code = response.status().as_u16() as i32;

    // Log to database asynchronously
    if state.config.enable_audit_log {
        let db_pool = state.db_pool.clone();
        let api_key_id = api_key_info.map(|info| info.api_key_id);

        tokio::spawn(async move {
            if let Ok(client) = db_pool.get().await {
                let _ = client
                    .execute(
                        r#"
                        INSERT INTO audit_logs (api_key_id, endpoint, method, status_code, latency_ms)
                        VALUES ($1, $2, $3, $4, $5)
                        "#,
                        &[&api_key_id, &uri, &method, &status_code, &latency_ms],
                    )
                    .await;
            }
        });
    }

    response
}

/// Admin role check middleware
pub async fn admin_check_middleware(request: Request, next: Next) -> Response {
    // Get API key info from extensions
    let api_key_info = request.extensions().get::<ApiKeyInfo>();

    if let Some(key_info) = api_key_info {
        if key_info.role == "admin" {
            next.run(request).await
        } else {
            let error = ErrorResponse {
                error: ErrorDetail {
                    message: "Admin role required".to_string(),
                    r#type: "permission_error".to_string(),
                    code: Some("forbidden".to_string()),
                },
            };

            (StatusCode::FORBIDDEN, Json(error)).into_response()
        }
    } else {
        let error = ErrorResponse {
            error: ErrorDetail {
                message: "Authentication required".to_string(),
                r#type: "authentication_error".to_string(),
                code: Some("unauthorized".to_string()),
            },
        };

        (StatusCode::UNAUTHORIZED, Json(error)).into_response()
    }
}
