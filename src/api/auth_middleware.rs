//! Middleware Module
//!
//! Provides authentication, rate limiting, and audit logging middleware.

use axum::{
    Json,
    extract::{ConnectInfo, Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::net::SocketAddr;
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

/// Pre-auth, connection-keyed attempt limiter (security audit item 3).
///
/// Runs BEFORE `auth_middleware`'s bearer-token hash and database lookup —
/// that ordering is the point: a brute-force guessing campaign never
/// presents a valid key, so throttling only after a failed lookup would
/// still pay the hash+query cost on every guess. This counts every request
/// that reaches it, successful or not, per client IP (see
/// `effective_client_ip`), and refuses once a per-minute threshold is
/// crossed — before the key lookup ever runs.
///
/// Skipped entirely when no database is configured: `validate_auth_policy`
/// already restricts that configuration to a loopback bind (security audit
/// item 2), so there is no untrusted network attacker to throttle.
///
/// On a database error, fails CLOSED — refuses the request rather than
/// letting it through unlimited — because this sits in front of
/// authentication: an attacker who can make this middleware's own database
/// call fail would otherwise get unmetered guessing for free. Logged at
/// `error` level so an operator can tell "the limiter is down" apart from
/// "a client is rate-limited."
pub async fn pre_auth_attempt_limiter_middleware(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    let Some(pool) = state.db_pool.clone() else {
        return next.run(request).await;
    };

    let client_ip =
        effective_client_ip(peer.ip(), request.headers(), &state.config.trusted_proxies);

    let client = match pool.get().await {
        Ok(client) => client,
        Err(e) => {
            tracing::error!(
                error = %e,
                %client_ip,
                "pre-auth attempt limiter: database connection failed; failing closed"
            );
            return service_unavailable();
        }
    };

    let limit = state.config.max_auth_attempts_per_minute_per_ip as i64;
    let ip_text = client_ip.to_string();

    let count: i64 = match client
        .query_one(
            r#"
            INSERT INTO auth_attempt_counts (client_ip, window_start, request_count)
            VALUES ($1::inet, date_trunc('minute', now())::timestamp, 1)
            ON CONFLICT (client_ip, window_start) DO UPDATE
              SET request_count = auth_attempt_counts.request_count + 1
            RETURNING request_count
            "#,
            &[&ip_text],
        )
        .await
    {
        Ok(row) => {
            let count: i32 = row.get("request_count");
            count as i64
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                %client_ip,
                "pre-auth attempt limiter: database query failed; failing closed"
            );
            return service_unavailable();
        }
    };

    if count > limit {
        tracing::warn!(
            %client_ip,
            count,
            limit,
            "pre-auth attempt limiter: refusing before the key lookup"
        );
        let error = ErrorResponse {
            error: ErrorDetail {
                message: "Too many authentication attempts from this client".to_string(),
                r#type: "rate_limit_error".to_string(),
                code: Some("auth_attempts_exceeded".to_string()),
            },
        };
        return (StatusCode::TOO_MANY_REQUESTS, Json(error)).into_response();
    }

    next.run(request).await
}

fn service_unavailable() -> Response {
    let error = ErrorResponse {
        error: ErrorDetail {
            message: "Internal server error".to_string(),
            r#type: "server_error".to_string(),
            code: Some("internal_error".to_string()),
        },
    };
    (StatusCode::SERVICE_UNAVAILABLE, Json(error)).into_response()
}

/// Authentication middleware
pub async fn auth_middleware(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Response {
    // Skip authentication when no database is configured
    if state.db_pool.is_none() {
        return next.run(request).await;
    }

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
        let client = match state.db_pool.as_ref().unwrap().get().await {
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
    // Skip rate limiting when no database is configured
    if state.db_pool.is_none() {
        return next.run(request).await;
    }

    // Get API key info from extensions
    let api_key_info = request.extensions().get::<ApiKeyInfo>().cloned();

    if let Some(key_info) = api_key_info {
        // Check and increment fixed-window (per-minute) counters in Postgres.
        let limit = state.config.rate_limit_per_minute as i64;
        let api_key_id = key_info.api_key_id;

        // Perform an atomic upsert and return the new request_count for the current minute window
        let client = match state.db_pool.as_ref().unwrap().get().await {
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
    if !state.config.enable_audit_log || state.db_pool.is_none() {
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
        let db_pool = state.db_pool.clone().unwrap();
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

/// Which client IP a request should be attributed to for the pre-auth
/// attempt limiter (`pre_auth_attempt_limiter_middleware`).
///
/// Trusts `X-Forwarded-For` ONLY when `peer_ip` (the real TCP peer) is in
/// the operator-configured `trusted_proxies` allowlist — never inferred.
/// Behind an untrusted or unconfigured reverse proxy every client appears
/// to arrive from the proxy's own IP, so trusting the header unconditionally
/// would let any client CLAIM any IP, which both defeats the limiter and
/// lets one client lock out an innocent IP it names.
pub(crate) fn effective_client_ip(
    peer_ip: std::net::IpAddr,
    headers: &axum::http::HeaderMap,
    trusted_proxies: &[std::net::IpAddr],
) -> std::net::IpAddr {
    if !trusted_proxies.contains(&peer_ip) {
        return peer_ip;
    }

    headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .map(str::trim)
        .and_then(|s| s.parse::<std::net::IpAddr>().ok())
        .unwrap_or(peer_ip)
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;
    use std::net::IpAddr;

    fn headers_with_xff(value: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("x-forwarded-for", value.parse().unwrap());
        h
    }

    #[test]
    fn effective_client_ip_uses_peer_ip_when_peer_is_not_a_trusted_proxy() {
        let peer: IpAddr = "1.2.3.4".parse().unwrap();
        let headers = headers_with_xff("9.9.9.9");
        let trusted: &[IpAddr] = &[];

        assert_eq!(effective_client_ip(peer, &headers, trusted), peer);
    }

    #[test]
    fn effective_client_ip_uses_xff_first_entry_when_peer_is_a_trusted_proxy() {
        let proxy: IpAddr = "10.0.0.1".parse().unwrap();
        let real_client: IpAddr = "9.9.9.9".parse().unwrap();
        let headers = headers_with_xff("9.9.9.9, 10.0.0.1");

        assert_eq!(effective_client_ip(proxy, &headers, &[proxy]), real_client);
    }

    #[test]
    fn effective_client_ip_falls_back_to_peer_when_trusted_but_no_xff_header() {
        let proxy: IpAddr = "10.0.0.1".parse().unwrap();
        let headers = HeaderMap::new();

        assert_eq!(effective_client_ip(proxy, &headers, &[proxy]), proxy);
    }

    #[test]
    fn effective_client_ip_falls_back_to_peer_when_xff_is_malformed() {
        let proxy: IpAddr = "10.0.0.1".parse().unwrap();
        let headers = headers_with_xff("not-an-ip-address");

        assert_eq!(effective_client_ip(proxy, &headers, &[proxy]), proxy);
    }
}
