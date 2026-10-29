// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lightbulb API Server
//!
//! Main entry point for the Lightbulb inference server.

use anyhow::Result;
use clap::Parser;
use lightbulb::api::{ApiConfig, ApiServer};
use lightbulb::engine::{MemoryAwareScheduler, memory_aware_scheduler::MemoryAwareConfig};
use std::env;
use std::sync::Arc;

/// Lightbulb inference server. Configuration is by environment variable
/// (`DATABASE_URL`, `LIGHTBULB_BIND_ADDRESS`, `LIGHTBULB_MODELS_DIR`, ...; see
/// QUICKSTART.md); the only flag is `--no-auth`.
#[derive(Parser, Debug)]
#[command(name = "lightbulb", version)]
struct Cli {
    /// Run without authentication, for this one invocation. Deliberately a
    /// flag and not an env var: it must be a choice made per start, not
    /// something that can sit in a deployment's environment and silently
    /// apply to every future start. See `lightbulb::api::validate_auth_policy`.
    #[arg(long)]
    no_auth: bool,
}

fn main() -> Result<()> {
    // Flags first, in a synchronous entrypoint: `--help`/`--version` (and a
    // rejected flag) must exit before the async runtime, logging,
    // configuration or any database access exist (CodeRipper OPS-011).
    run(Cli::parse())
}

#[tokio::main]
async fn run(cli: Cli) -> Result<()> {
    lightbulb::logging::init_tracing("info,lightbulb=debug");

    // Load configuration from environment
    let database_url = env::var("DATABASE_URL").ok();

    let bind_address =
        env::var("LIGHTBULB_BIND_ADDRESS").unwrap_or_else(|_| "0.0.0.0:8080".to_string());

    let models_dir = env::var("LIGHTBULB_MODELS_DIR").ok();

    let default_model =
        env::var("LIGHTBULB_DEFAULT_MODEL").unwrap_or_else(|_| "default".to_string());

    // No fallback (security audit item 5): a hardcoded
    // "change-me-in-production" default meant every deployment that never
    // set this started with the same known secret. `None` when unset — see
    // `ApiConfig::jwt_secret`'s own doc for why that's required rather than
    // a weaker default now that nothing reads this value yet anyway.
    let jwt_secret = env::var("LIGHTBULB_JWT_SECRET").ok();

    // `--no-auth`: see `Cli::no_auth` for why this is a flag and not an env var.
    let no_auth = cli.no_auth;

    // Only a peer IP listed here may supply `X-Forwarded-For` to the
    // pre-auth attempt limiter (security audit item 3) — see
    // `ApiConfig::trusted_proxies`'s own doc for why an unconfigured
    // default must be empty, not "trust everyone" or "trust no one
    // forever." Comma-separated; an entry that fails to parse as an IP is
    // dropped rather than refusing startup over it.
    let trusted_proxies: Vec<std::net::IpAddr> = env::var("LIGHTBULB_TRUSTED_PROXIES")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_default();

    let max_auth_attempts_per_minute_per_ip =
        env::var("LIGHTBULB_MAX_AUTH_ATTEMPTS_PER_MINUTE_PER_IP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);

    // Issue #130: the strict, failures-only brute-force limiter, separate
    // from the flood limiter above — see `ApiConfig::
    // max_auth_failures_per_minute_per_ip`'s own doc.
    let max_auth_failures_per_minute_per_ip =
        env::var("LIGHTBULB_MAX_AUTH_FAILURES_PER_MINUTE_PER_IP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);

    let request_timeout_ms = env::var("LIGHTBULB_REQUEST_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300_000);

    // Security audit item 4. A deployment secret, env var only — never a
    // contact file. `None` means only `LogSink` is active; see
    // `ApiConfig::security_alert_email`'s own doc for why a configured
    // address here does NOT mean alert delivery is implemented.
    let security_alert_email = env::var("SECURITY_ALERT_EMAIL").ok();

    // Phase 4 config-management item: these four had no env override at
    // all before this -- a deployment that needed a different batch
    // size, context length, queue depth or rate limit than the hardcoded
    // default had no way to set one short of recompiling.
    let rate_limit_per_minute = env::var("LIGHTBULB_RATE_LIMIT_PER_MINUTE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60);
    let model_max_batch_size = env::var("LIGHTBULB_MODEL_MAX_BATCH_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let model_context_length = env::var("LIGHTBULB_MODEL_CONTEXT_LENGTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let model_max_queue_depth = env::var("LIGHTBULB_MODEL_MAX_QUEUE_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    // Off unless explicitly enabled: not default-on until a measured
    // benchmark shows the win (ROADMAP research-track item 1, slice 3).
    let model_prefix_sharing = env::var("LIGHTBULB_PREFIX_SHARING")
        .map(|v| parse_flag(&v))
        .unwrap_or(false);

    let cors_allowed_origins: Vec<String> = env::var("LIGHTBULB_CORS_ORIGINS")
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    // Create API configuration
    let config = ApiConfig {
        database_url: database_url.clone(),
        bind_address,
        enable_openai_api: true,
        enable_admin_api: database_url.is_some(), // Disable admin API without database
        enable_lightbulb_extensions: true,
        enable_metrics_endpoint: true,
        jwt_secret,
        rate_limit_per_minute,
        models_dir,
        default_model,
        model_max_batch_size,
        model_context_length,
        model_max_queue_depth,
        model_prefix_sharing,
        request_timeout_ms,
        enable_audit_log: database_url.is_some(), // Disable audit logging without database
        tls: Default::default(),                  // TLS disabled by default
        no_auth,
        trusted_proxies,
        max_auth_attempts_per_minute_per_ip,
        max_auth_failures_per_minute_per_ip,
        security_alert_email,
        cors_allowed_origins,
    };

    tracing::info!("Starting Lightbulb API server");
    match &database_url {
        Some(url) => tracing::info!("Database: {}", url),
        None => tracing::info!("Database: none (auth/audit disabled)"),
    }
    tracing::info!("Listening on: {}", config.bind_address);

    // Create memory-aware scheduler. Phase 4 config-management item:
    // these five had no env override and no startup validation at all
    // before this -- see MemoryAwareConfig::validate's own doc comment.
    let scheduler_config = MemoryAwareConfig {
        max_memory_bytes: env::var("LIGHTBULB_MAX_MEMORY_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8 * 1024 * 1024 * 1024), // 8GB
        memory_per_slot_base: env::var("LIGHTBULB_MEMORY_PER_SLOT_BASE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100 * 1024 * 1024), // 100MB
        memory_per_token: env::var("LIGHTBULB_MEMORY_PER_TOKEN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50 * 1024), // 50KB per token
        eviction_pressure_threshold: env::var("LIGHTBULB_EVICTION_PRESSURE_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.85),
        memory_safety_margin: env::var("LIGHTBULB_MEMORY_SAFETY_MARGIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.1),
    };
    scheduler_config.validate()?;

    let scheduler = Arc::new(std::sync::Mutex::new(MemoryAwareScheduler::new(
        scheduler_config,
    )));

    // Create and start server
    let server = ApiServer::new(config, scheduler).await?;
    server.serve().await?;

    Ok(())
}

/// `1`/`true`/`yes`/`on` (any case, surrounding whitespace ignored) enable a
/// flag; anything else -- including an empty or misspelt value -- leaves it off.
fn parse_flag(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(test)]
mod tests {
    use super::parse_flag;

    #[test]
    fn parse_flag_enables_only_on_explicit_truthy_values() {
        for on in ["1", "true", "TRUE", " yes ", "On"] {
            assert!(parse_flag(on), "{on:?} must enable");
        }
        for off in ["", "0", "false", "off", "ture", "2"] {
            assert!(!parse_flag(off), "{off:?} must not enable");
        }
    }
}
