//! Lightbulb API Server
//!
//! Main entry point for the Lightbulb inference server.

use anyhow::Result;
use lightbulb::api::{ApiConfig, ApiServer};
use lightbulb::engine::{MemoryAwareScheduler, memory_aware_scheduler::MemoryAwareConfig};
use std::env;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,lightbulb=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

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

    // `--no-auth` is a CLI flag, deliberately NOT an env var: running
    // without authentication must be a choice made for this one invocation,
    // not something that can end up sitting in a deployment's environment
    // and silently apply to every future start. See
    // `lightbulb::api::validate_auth_policy`.
    let no_auth = env::args().any(|arg| arg == "--no-auth");

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

    // Security audit item 4. A deployment secret, env var only — never a
    // contact file. `None` means only `LogSink` is active; see
    // `ApiConfig::security_alert_email`'s own doc for why a configured
    // address here does NOT mean alert delivery is implemented.
    let security_alert_email = env::var("SECURITY_ALERT_EMAIL").ok();

    // Create API configuration
    let config = ApiConfig {
        database_url: database_url.clone(),
        bind_address,
        enable_openai_api: true,
        enable_admin_api: database_url.is_some(), // Disable admin API without database
        enable_lightbulb_extensions: true,
        jwt_secret,
        rate_limit_per_minute: 60,
        models_dir,
        default_model,
        model_max_batch_size: 16,
        model_context_length: 4096,
        enable_audit_log: database_url.is_some(), // Disable audit logging without database
        tls: Default::default(),                  // TLS disabled by default
        no_auth,
        trusted_proxies,
        max_auth_attempts_per_minute_per_ip,
        security_alert_email,
    };

    tracing::info!("Starting Lightbulb API server");
    match &database_url {
        Some(url) => tracing::info!("Database: {}", url),
        None => tracing::info!("Database: none (auth/audit disabled)"),
    }
    tracing::info!("Listening on: {}", config.bind_address);

    // Create memory-aware scheduler
    let scheduler_config = MemoryAwareConfig {
        max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
        memory_per_slot_base: 100 * 1024 * 1024,  // 100MB
        memory_per_token: 50 * 1024,              // 50KB per token
        eviction_pressure_threshold: 0.85,
        memory_safety_margin: 0.1,
    };

    let scheduler = Arc::new(MemoryAwareScheduler::new(scheduler_config));

    // Create and start server
    let server = ApiServer::new(config, scheduler).await?;
    server.serve().await?;

    Ok(())
}
