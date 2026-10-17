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
use crate::api::persistant_migrate;
use persistant_relational::RelationalBackend;

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
    if state.pg_backend.is_none() {
        return next.run(request).await;
    }

    let client_ip =
        effective_client_ip(peer.ip(), request.headers(), &state.config.trusted_proxies);

    let limit = state.config.max_auth_attempts_per_minute_per_ip as i64;

    // PersistAnt adoption Step 3 (docs/PERSISTANT_ADOPTION_DESIGN.md): one
    // of the 5 single-statement writes ported to `AppState::pg_backend` for
    // this step.
    let count = match record_auth_attempt(&state, client_ip).await {
        Ok(count) => count,
        Err(()) => {
            tracing::error!(
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
        if crate::api::security_event::just_crossed_threshold(count, limit) {
            let event = crate::api::security_event::SecurityEvent::RepeatedAuthFailures {
                client_ip,
                count,
                limit,
            };
            for sink in &state.security_sinks {
                sink.notify(&event);
            }
        }
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

/// Atomic upsert into `auth_attempt_counts`, returning the request count
/// for the current minute window. `Err(())` means the query itself failed
/// or no `AppState::pg_backend` is configured -- the caller fails closed
/// either way, same as before this port.
///
/// `persistant_relational::Value` has no INET variant either (the same gap
/// `persistant_migrate::uuid_to_value`'s doc comment describes for UUID),
/// so `client_ip` binds through `Value::Text` with an explicit `::inet`
/// cast on the PARAMETER (not a column, since this is an insert, not a
/// lookup against a stored value) -- Postgres has no bare
/// text-into-inet-column coercion.
async fn record_auth_attempt(state: &AppState, client_ip: std::net::IpAddr) -> Result<i64, ()> {
    let backend = state.pg_backend.as_ref().ok_or(())?;
    let rows = backend
        .query(
            r#"
            INSERT INTO auth_attempt_counts (client_ip, window_start, request_count)
            VALUES ($1::inet, date_trunc('minute', now())::timestamp, 1)
            ON CONFLICT (client_ip, window_start) DO UPDATE
              SET request_count = auth_attempt_counts.request_count + 1
            RETURNING request_count
            "#,
            &[persistant_relational::Value::Text(client_ip.to_string())],
        )
        .await
        .map_err(|_| ())?;
    match rows.first().and_then(|row| row.values.first()) {
        Some(persistant_relational::Value::Int(n)) => Ok(*n),
        _ => Err(()),
    }
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

/// Look up an API key by its hash, via `AppState::pg_backend`.
///
/// `Ok(None)` means not found/expired/revoked (a normal outcome, not an
/// error); `Err(())` means the lookup itself failed (DB unreachable or
/// unconfigured) -- the caller logs and turns that into a 500, matching the
/// previous `deadpool_postgres` path's behavior exactly.
///
/// The `id` column is cast to `::text` in the query itself:
/// `persistant_relational`'s row decoder (`persistant-sqlx`'s `decode()`)
/// has no case for Postgres's `UUID` type at all -- only
/// `BOOL`/`INT2`/`INT4`/`INT8`/`FLOAT4`/`FLOAT8`/`TEXT`/`VARCHAR`/`NAME`/
/// `BPCHAR`/`BYTEA` -- so selecting a raw `uuid` column errors with
/// `RelError::Unsupported`. Casting server-side sidesteps that; the column
/// arrives as `Value::Text`, decoded the same way `uuid_to_value`/
/// `value_to_uuid` already handle UUIDs everywhere else in this adoption.
async fn lookup_api_key_by_hash(
    state: &AppState,
    key_hash: &str,
) -> Result<Option<ApiKeyInfo>, ()> {
    let backend = match state.pg_backend.as_ref() {
        Some(backend) => backend,
        None => {
            eprintln!("PgBackend not configured");
            return Err(());
        }
    };

    let rows = backend
        .query(
            r#"SELECT id::text AS id, role
               FROM api_keys
               WHERE key_hash = $1
                 AND (expires_at IS NULL OR expires_at > NOW())
                 AND revoked_at IS NULL
               LIMIT 1"#,
            &[persistant_relational::Value::Text(key_hash.to_string())],
        )
        .await
        .map_err(|e| eprintln!("DB error checking API key: {e}"))?;

    let Some(row) = rows.first() else {
        return Ok(None);
    };
    let id_text = row.text(0).ok_or_else(|| {
        eprintln!("DB error checking API key: id column was not text after ::text cast");
    })?;
    let api_key_id = uuid::Uuid::parse_str(id_text)
        .map_err(|e| eprintln!("DB error checking API key: id column is not a valid UUID: {e}"))?;
    let role = row
        .text(1)
        .ok_or_else(|| eprintln!("DB error checking API key: role column was not text"))?
        .to_string();

    Ok(Some(ApiKeyInfo { api_key_id, role }))
}

/// Authentication middleware
pub async fn auth_middleware(
    State(state): State<AppState>,
    mut request: Request,
    next: Next,
) -> Response {
    // Skip authentication when no database is configured
    if state.pg_backend.is_none() {
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

        // PersistAnt adoption Step 2 (docs/PERSISTANT_ADOPTION_DESIGN.md):
        // this hash lookup is one of the 2 read-only call sites ported to
        // `AppState::pg_backend` for this step.
        match lookup_api_key_by_hash(&state, &key_hash).await {
            Ok(Some(info)) => {
                request.extensions_mut().insert(info);
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
            Err(()) => {
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
    if state.pg_backend.is_none() {
        return next.run(request).await;
    }

    // Get API key info from extensions
    let api_key_info = request.extensions().get::<ApiKeyInfo>().cloned();

    if let Some(key_info) = api_key_info {
        // Check and increment fixed-window (per-minute) counters in Postgres.
        //
        // PersistAnt adoption Step 3 (docs/PERSISTANT_ADOPTION_DESIGN.md):
        // one of the 5 single-statement writes ported to
        // `AppState::pg_backend` for this step.
        let limit = state.config.rate_limit_per_minute as i64;
        let api_key_id = key_info.api_key_id;

        match record_api_key_usage(&state, api_key_id).await {
            Ok(count) => {
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
                eprintln!("Rate limit DB error: {e}");
                // On DB error, be conservative and allow the request (don't block traffic for DB hiccups)
            }
        }

        next.run(request).await
    } else {
        next.run(request).await
    }
}

/// Atomic upsert into `api_key_usage`, returning the request count for the
/// current minute window. `api_key_id` binds through `Value::Text` with an
/// explicit `::uuid` cast on the parameter -- same INET-on-insert pattern
/// as `record_auth_attempt`, for the same reason (`Value` has no Uuid or
/// Inet variant).
async fn record_api_key_usage(state: &AppState, api_key_id: uuid::Uuid) -> Result<i64, String> {
    let backend = state
        .pg_backend
        .as_ref()
        .ok_or_else(|| "pg_backend not configured".to_string())?;
    let rows = backend
        .query(
            r#"
            INSERT INTO api_key_usage (api_key_id, window_start, request_count)
            VALUES ($1::uuid, date_trunc('minute', now())::timestamp, 1)
            ON CONFLICT (api_key_id, window_start) DO UPDATE
              SET request_count = api_key_usage.request_count + 1
            RETURNING request_count
            "#,
            &[persistant_migrate::uuid_to_value(api_key_id)],
        )
        .await
        .map_err(|e| e.to_string())?;
    match rows.first().and_then(|row| row.values.first()) {
        Some(persistant_relational::Value::Int(n)) => Ok(*n),
        other => Err(format!("unexpected request_count row shape: {other:?}")),
    }
}

/// Audit logging middleware
pub async fn audit_middleware(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    if !state.config.enable_audit_log || state.pg_backend.is_none() {
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

    // Log to database asynchronously. PersistAnt adoption Step 3
    // (docs/PERSISTANT_ADOPTION_DESIGN.md): one of the 5 single-statement
    // writes ported to `AppState::pg_backend` for this step -- best-effort,
    // same as before this port (the spawned task's result is dropped).
    if state.config.enable_audit_log {
        let backend = state.pg_backend.clone().unwrap();
        let api_key_id = api_key_info.map(|info| info.api_key_id);

        tokio::spawn(async move {
            let api_key_id_value = api_key_id.map_or(persistant_relational::Value::Null, |id| {
                persistant_migrate::uuid_to_value(id)
            });
            let _ = backend
                    .execute(
                        r#"
                        INSERT INTO audit_logs (api_key_id, endpoint, method, status_code, latency_ms)
                        VALUES ($1::uuid, $2, $3, $4, $5)
                        "#,
                        &[
                            api_key_id_value,
                            persistant_relational::Value::Text(uri),
                            persistant_relational::Value::Text(method),
                            persistant_relational::Value::Int(status_code as i64),
                            persistant_relational::Value::Int(latency_ms as i64),
                        ],
                    )
                    .await;
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
