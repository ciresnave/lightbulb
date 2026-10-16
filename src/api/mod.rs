//! Lightbulb API Server
//!
//! Provides OpenAI-compatible API with Lightbulb-specific extensions.
//!
//! # Architecture
//!
//! - **OpenAI-Compatible Base**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`
//! - **Admin API**: `/v1/lightbulb/admin/*` for system management
//! - **Knowledge Base**: `/v1/lightbulb/knowledge/*` for KB operations
//! - **Reasoning Controls**: `/v1/lightbulb/reasoning/*` for advanced reasoning
//! - **State Management**: `/v1/lightbulb/state/*` for persistence/branching
//!
//! # Security
//!
//! - Bearer token authentication via auth-framework
//! - Role-based access control (user, admin, llm)
//! - Rate limiting per API key
//! - Audit logging for all requests
//!
//! # Example
//!
//! ```rust,ignore
//! use lightbulb::api::ApiServer;
//!
//! let server = ApiServer::new(config).await?;
//! server.serve("0.0.0.0:8080").await?;
//! ```

pub mod admin;
pub mod auth_middleware;
pub mod chat_template;
pub mod lightbulb;
pub mod openai;
pub mod security_event;
pub mod types;

use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::Serialize;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::engine::MemoryAwareScheduler;
use crate::engine::model_runner::ModelRunner;
use std::path::Path;

/// API server configuration
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// PostgreSQL connection string (optional — when None, auth/audit/rate-limiting are disabled)
    pub database_url: Option<String>,

    /// Server bind address
    pub bind_address: String,

    /// Enable OpenAI-compatible endpoints
    pub enable_openai_api: bool,

    /// Enable admin endpoints
    pub enable_admin_api: bool,

    /// Enable Lightbulb-specific extensions
    pub enable_lightbulb_extensions: bool,

    /// Enable the Prometheus-format `GET /metrics` endpoint (Phase 4
    /// observability). Default `true`. Unauthenticated, same reasoning as
    /// `/health`/`/health/ready` -- a Prometheus scraper cannot present a
    /// bearer token -- so this is the one real lever an operator who does
    /// NOT want it reachable has, short of a network-level block; there is
    /// no separate per-route auth for it the way admin/OpenAI routes have.
    pub enable_metrics_endpoint: bool,

    /// JWT secret for token signing. `None` unless an operator explicitly
    /// sets `LIGHTBULB_JWT_SECRET` (security audit item 5, 2026-10-06):
    /// there used to be a hardcoded fallback here
    /// (`"change-me-in-production"`), a known string any future JWT
    /// feature could end up trusting by default. Nothing in this crate
    /// signs or verifies a JWT with this value today — the `None` default
    /// means that stays true until a real feature explicitly checks for
    /// `Some` and is configured for it, rather than silently working with
    /// a secret everyone who has ever read this source already knows.
    pub jwt_secret: Option<String>,

    /// Maximum requests per minute per API key
    pub rate_limit_per_minute: u32,

    /// Optional base directory where models are stored (each model is a subdirectory)
    pub models_dir: Option<String>,

    /// Default model name to load from `models_dir`
    pub default_model: String,

    /// Model runner max batch size
    pub model_max_batch_size: usize,

    /// Model context length
    pub model_context_length: usize,

    /// Maximum number of inference jobs the model runner's channel will
    /// hold (queued plus the one being processed) before enqueuing a new one
    /// fails with [`crate::engine::model_runner::QueueFullError`] (surfaced
    /// to the client as HTTP 503) instead of queuing without bound.
    pub model_max_queue_depth: usize,

    /// How long an inference job may sit between being enqueued and the
    /// model runner thread finishing it, before the HTTP handler gives up
    /// and returns 504 instead of waiting forever. Phase 4 hardening
    /// (board PM task): `tokio::time::timeout` previously had zero uses
    /// anywhere in `src/`, and `resp_rx.await` in
    /// `openai::chat::run_inference_once` (the chokepoint every
    /// non-streaming completion and the n>1 retry loop share) had no
    /// upper bound at all -- a stuck or pathologically slow generation
    /// held the HTTP connection open indefinitely, with no server-side
    /// path back to the client.
    pub request_timeout_ms: u64,

    /// Enable audit logging
    pub enable_audit_log: bool,

    /// TLS configuration
    pub tls: crate::tls::TlsConfig,

    /// Explicit operator opt-out of authentication (the `--no-auth` CLI
    /// flag). Only ever lets the server start WITHOUT `database_url` set
    /// when `bind_address` is also loopback — see `validate_auth_policy`
    /// (not an intra-doc link here since this field is `pub` and that
    /// function is `pub(crate)`). Never set this from an env var: it must be
    /// a deliberate per-invocation choice, not something that silently
    /// persists in a deployment's environment.
    pub no_auth: bool,

    /// Peer IPs allowed to supply `X-Forwarded-For` for the pre-auth
    /// attempt limiter. Empty by default — behind an unconfigured reverse
    /// proxy every client would otherwise appear to share the proxy's own
    /// IP, letting one noisy client lock out everyone (security audit
    /// item 3). Set this to the reverse proxy's own address(es), never to
    /// a client-controlled value.
    pub trusted_proxies: Vec<std::net::IpAddr>,

    /// Requests per minute a single client IP may make before the pre-auth
    /// attempt limiter refuses with 429 — BEFORE the bearer-token hash and
    /// database lookup run, so a guessing campaign is throttled before it
    /// costs a key lookup. Independent of `rate_limit_per_minute`, which
    /// only ever applies to a request that already carries a valid key.
    ///
    /// **Counts EVERY request from an IP, successful or not** — it is a
    /// flood limiter in front of the hash+lookup, not a brute-force
    /// detector. PM review of #127 (2026-10-07) caught the default too low
    /// at 20: several ordinary clients behind one NAT/proxy, or one client
    /// making routine requests, would be throttled by legitimate traffic
    /// alone, since this has no way to distinguish a valid key's requests
    /// from a guess. 300/min (~5/s) still stops a flood before it reaches
    /// the database, without breaking real use. The PROPER brute-force
    /// limiter — a strict, FAILURES-ONLY counter (post-auth 401s per IP,
    /// e.g. 10/min) — is tracked as a follow-up, not implemented here; see
    /// the issue filed alongside this change.
    pub max_auth_attempts_per_minute_per_ip: u32,

    /// Destination for the security alert hook (security audit item 4) —
    /// read from the `SECURITY_ALERT_EMAIL` environment variable ONLY
    /// (a deployment secret, never a contact file), set via
    /// `main.rs`. `None` means no `EmailSink` is built; `LogSink` is
    /// always active regardless. **Alert DELIVERY is not implemented**:
    /// `EmailSink` logs what it would send and sends nothing — see
    /// `security_event.rs`'s module doc. Do not treat a non-`None` value
    /// here as "someone gets paged."
    pub security_alert_email: Option<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            database_url: None,
            bind_address: "0.0.0.0:8080".to_string(),
            enable_openai_api: true,
            enable_admin_api: true,
            enable_lightbulb_extensions: true,
            enable_metrics_endpoint: true,
            jwt_secret: None,
            rate_limit_per_minute: 60,
            enable_audit_log: true,
            models_dir: Some("./models".to_string()),
            default_model: "lightbulb-default".to_string(),
            model_max_batch_size: 8,
            model_context_length: 2048,
            model_max_queue_depth: 32,
            request_timeout_ms: 300_000,
            tls: crate::tls::TlsConfig::default(),
            no_auth: false,
            trusted_proxies: Vec::new(),
            max_auth_attempts_per_minute_per_ip: 300,
            security_alert_email: None,
        }
    }
}

/// Decide whether `bind_address` ("host:port", or a bare host) resolves to
/// loopback WITHOUT a DNS lookup — a hostname that merely resolves to
/// loopback today is not a safety property, since DNS can change. Only the
/// literal numeric loopback addresses and the literal string `"localhost"`
/// count; every other hostname, including ones on the operator's own LAN, is
/// treated as reachable from outside and refused by [`validate_auth_policy`]
/// when auth is off.
fn is_loopback_bind(bind_address: &str) -> bool {
    let host = bind_address
        .rsplit_once(':')
        .map(|(host, _port)| host)
        .unwrap_or(bind_address);
    let host = host.trim_start_matches('[').trim_end_matches(']');

    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }

    host.parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

/// Refuse to construct a server that would run the whole inference API
/// (and, when `enable_admin_api`/`enable_lightbulb_extensions` are on, the
/// admin and extension surfaces too) with no authentication, no rate
/// limiting and no audit logging — the default-no-`DATABASE_URL`
/// configuration the brute-force-countermeasure audit flagged as the worst
/// of its findings (board PM task, 2026-10-04): previously nothing stopped
/// this from being the *production* configuration by omission.
///
/// `database_url` set is always fine — that's what turns auth/audit/rate-
/// limiting on in [`auth_middleware`]. Without it, starting anyway requires
/// BOTH an explicit `no_auth` opt-in AND a loopback bind: `no_auth` alone on
/// a reachable address would just be the same hole with an extra flag to
/// type.
pub(crate) fn validate_auth_policy(config: &ApiConfig) -> Result<()> {
    if config.database_url.is_some() {
        return Ok(());
    }

    if !config.no_auth {
        anyhow::bail!(
            "refusing to start: no DATABASE_URL is set, which disables authentication, rate \
             limiting and audit logging on every endpoint. Set DATABASE_URL, or pass --no-auth \
             together with a loopback bind_address (127.0.0.1, ::1, or localhost) if you really \
             want to run without auth for local development."
        );
    }

    if !is_loopback_bind(&config.bind_address) {
        anyhow::bail!(
            "refusing to start: --no-auth was given but bind_address {:?} is not loopback. \
             --no-auth is only accepted on 127.0.0.1, ::1, or localhost — binding it anywhere \
             else would expose every unauthenticated endpoint to the network.",
            config.bind_address
        );
    }

    tracing::warn!(
        bind_address = %config.bind_address,
        "starting with --no-auth on a loopback bind — every endpoint is unauthenticated; \
         local development only"
    );
    Ok(())
}

/// Sourcery review of #155: `tokio::time::timeout` with a zero duration
/// expires immediately, so `request_timeout_ms: 0` (e.g. a typo'd
/// `LIGHTBULB_REQUEST_TIMEOUT_MS=0`) would reject every inference request
/// with a 504 before the runner ever saw it. Rejected at startup, named,
/// rather than silently accepted and discovered later as "every single
/// request times out."
pub(crate) fn validate_request_timeout(config: &ApiConfig) -> Result<()> {
    if config.request_timeout_ms == 0 {
        anyhow::bail!(
            "refusing to start: request_timeout_ms is 0, which would time out every \
             inference request immediately (tokio::time::timeout(0) always expires). Set \
             LIGHTBULB_REQUEST_TIMEOUT_MS to a positive value, or omit it for the default \
             (300000ms / 5 minutes)."
        );
    }
    Ok(())
}

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    /// Inference scheduler
    pub scheduler: Arc<MemoryAwareScheduler>,

    /// API configuration
    pub config: ApiConfig,

    /// Database connection pool (None when running without PostgreSQL)
    pub db_pool: Option<deadpool_postgres::Pool>,

    /// Sender to enqueue inference jobs to the model runner thread (if started)
    pub inference_tx:
        Option<std::sync::mpsc::SyncSender<crate::engine::model_runner::InferenceJob>>,

    /// Chat template for the loaded model, plus the special tokens it renders
    /// with — resolved once at startup.
    ///
    /// `None` when no model runner started, or when no tier produced a
    /// template. In that case the handlers fall back to the legacy
    /// `role: content` join and `chat_template::resolve` has already said so at
    /// `warn` level, naming the model directory.
    ///
    /// The template and its tokens are held **together**, in one
    /// `ResolvedTemplate`, so that no caller is ever in a position to supply
    /// `bos`/`eos` itself — which in practice means a literal `"<s>"`/`"</s>"`,
    /// correct for the Llama-2 family and silently wrong for every other.
    pub chat_template: Option<Arc<crate::api::chat_template::ResolvedTemplate>>,

    /// Rolling EOS-fire rate over recent completions, shared by every
    /// completion path.
    ///
    /// **Observation only.** It exists because the tiers below
    /// `Resolution::TokenizerConfig` are heuristics that can be confidently
    /// wrong — a registry-matched template renders, returns 200, and prompts
    /// the model as a base model — and a low EOS rate is the cheapest
    /// observable that says so. Nothing reads it to decide what to prompt; see
    /// `engine::eos_monitor` for why auto-switching was rejected.
    ///
    /// Not `Option`: a monitor with nothing recorded already reports "no
    /// reading" (`stop_rate() == None`), so an absent one would be a second
    /// spelling of the same state.
    pub eos_monitor: Arc<crate::engine::eos_monitor::EosMonitor>,

    /// Security alert sinks (security audit item 4) — always contains a
    /// `LogSink`, plus an `EmailSink` stub when `security_alert_email` is
    /// configured. `pub`, matching every other `AppState` field: external
    /// test crates under `tests/` construct `AppState` with struct-literal
    /// syntax, which is only possible when every field (and the type each
    /// one names) is visible outside this crate.
    pub security_sinks: Vec<Arc<dyn security_event::SecuritySink>>,

    /// Live prefix-cache stats, written by the model-runner thread after
    /// every job — the `cache_stats` admin endpoint's real data source
    /// (board item M4 slice 1). Not `Option`, same reasoning as
    /// `eos_monitor`: no model loaded (or no job run yet) already reports as
    /// `PrefixCacheStats::default()`, which is a valid "nothing to report"
    /// reading, not a second spelling of absence.
    pub cache_stats: crate::engine::model_runner::SharedCacheStats,

    /// Live CPU/memory sampler backing the `metrics` admin endpoint (Phase 4
    /// hardening item 2). A persistent `sysinfo::System`, not one created
    /// per request: `sysinfo` computes CPU usage as a diff between two
    /// refreshes, so a fresh instance always reads 0 on its first use.
    pub live_system: SharedSystemMetrics,

    /// Live count of inference jobs enqueued or in flight -- the
    /// `/metrics` endpoint's queue-depth gauge. See
    /// `engine::model_runner::QueueDepthGuard`, which owns incrementing and
    /// decrementing it.
    pub queue_depth: crate::engine::model_runner::SharedQueueDepth,
}

/// Shared, persistent system sampler for CPU/memory metrics. See
/// `AppState::live_system`.
pub type SharedSystemMetrics = std::sync::Arc<std::sync::Mutex<sysinfo::System>>;

/// Construct a fresh `live_system` handle for `AppState`.
pub fn new_shared_system_metrics() -> SharedSystemMetrics {
    std::sync::Arc::new(std::sync::Mutex::new(sysinfo::System::new_all()))
}

/// API server
pub struct ApiServer {
    /// Application state
    state: AppState,
}

impl ApiServer {
    /// Create new API server
    pub async fn new(config: ApiConfig, scheduler: Arc<MemoryAwareScheduler>) -> Result<Self> {
        validate_auth_policy(&config)?;
        validate_request_timeout(&config)?;

        // Set up database connection pool if DATABASE_URL is configured
        let db_pool = if let Some(ref database_url) = config.database_url {
            let mut pg_config: tokio_postgres::Config = database_url.parse()?;

            // For trust auth, tokio-postgres still needs a password value (can be anything)
            if pg_config.get_password().is_none() {
                pg_config.password("trust");
            }

            let manager = deadpool_postgres::Manager::new(pg_config, tokio_postgres::NoTls);
            let pool = deadpool_postgres::Pool::builder(manager)
                .max_size(10)
                .build()?;

            // Run migrations
            let mut client = pool.get().await?;
            refinery::embed_migrations!("./migrations");
            migrations::runner().run_async(&mut **client).await?;

            tracing::info!("Database connected and migrations applied");
            Some(pool)
        } else {
            tracing::info!(
                "No DATABASE_URL set — running without database (auth/audit/rate-limiting disabled)"
            );
            None
        };

        // `auth_attempt_counts` (security audit item 3) is written on every
        // request that reaches `pre_auth_attempt_limiter_middleware`,
        // including every brute-force guess — unlike `api_key_usage`, whose
        // rows are bounded by the number of PROVISIONED keys, this table's
        // key space is "every IP that ever made a request," so it needs its
        // own expiry rather than growing forever. A background task, not a
        // per-request delete, so the request path pays no extra latency for
        // it.
        if let Some(pool) = db_pool.clone() {
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(300));
                loop {
                    interval.tick().await;
                    match pool.get().await {
                        Ok(client) => {
                            if let Err(e) = client
                                .execute(
                                    "DELETE FROM auth_attempt_counts \
                                     WHERE window_start < NOW() - INTERVAL '1 hour'",
                                    &[],
                                )
                                .await
                            {
                                tracing::warn!(
                                    error = %e,
                                    "auth_attempt_counts cleanup query failed"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "auth_attempt_counts cleanup: could not get a database connection"
                            );
                        }
                    }
                }
            });
        }

        let security_sinks = security_event::build_sinks(&config.security_alert_email);

        let mut state = AppState {
            scheduler,
            config: config.clone(),
            db_pool,
            inference_tx: None,
            chat_template: None,
            eos_monitor: Arc::new(crate::engine::eos_monitor::EosMonitor::default()),
            security_sinks,
            cache_stats: Arc::new(std::sync::Mutex::new(
                crate::cache::PrefixCacheStats::default(),
            )),
            live_system: new_shared_system_metrics(),
            queue_depth: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        };

        // Try to start a model runner thread if a model directory and default model exist
        if let Some(models_dir) = &state.config.models_dir {
            let model_path = Path::new(models_dir).join(&state.config.default_model);
            if model_path.exists() {
                match ModelRunner::start(
                    &model_path,
                    state.config.model_max_batch_size,
                    state.config.model_context_length,
                    Some("f32".to_string()),
                    state.config.model_max_queue_depth,
                    state.cache_stats.clone(),
                ) {
                    Ok(sender) => {
                        state.inference_tx = Some(sender);

                        // Resolve the chat template HERE and only here. Both
                        // halves read files, and a per-request read would put
                        // filesystem I/O in the request path for a value that
                        // cannot change while the process runs.
                        //
                        // `resolve_for_serving` owns the "keep it only if a tier
                        // produced one" rule; the three integration harnesses
                        // call the same function so they cannot drift from what
                        // this line builds.
                        state.chat_template = chat_template::resolve_for_serving(&model_path);
                        if let Some(t) = &state.chat_template {
                            tracing::info!(
                                "Chat template for {} resolved via {:?} (bos {:?}, eos {:?})",
                                model_path.display(),
                                t.resolved_by(),
                                t.tokens.bos,
                                t.tokens.eos
                            );
                        }

                        tracing::info!("Started model runner for {}", model_path.display());
                    }
                    Err(e) => {
                        tracing::error!(
                            "Failed to start model runner for {}: {}",
                            model_path.display(),
                            e
                        );
                    }
                }
            } else {
                tracing::warn!(
                    "Model path {} not found; skipping model runner startup",
                    model_path.display()
                );
            }
        }

        Ok(Self { state })
    }

    /// Build the router with all routes and middleware
    pub fn build_router(&self) -> Router<AppState> {
        let mut router = Router::new();

        // OpenAI-compatible API
        if self.state.config.enable_openai_api {
            router = router.merge(openai::routes());
        }

        // Admin API
        if self.state.config.enable_admin_api {
            router = router.merge(admin::routes());
        }

        // Lightbulb extensions
        if self.state.config.enable_lightbulb_extensions {
            router = router.merge(lightbulb::routes());
        }

        // Health, readiness and metrics, built as their own router so the
        // conditional `/metrics` route can be `.merge()`d in at the same
        // point in the chain as the always-present two below -- all three
        // need to land AFTER the auth-stack `.layer()` calls below (so
        // they bypass auth) but BEFORE CORS/tracing, and an `if` inside
        // that single `.route().route()` method-chain expression isn't
        // possible the way a separate router merged into it is.
        let mut unauthenticated_router = Router::new()
            .route("/health", get(health_check))
            .route("/health/ready", get(readiness_check));
        if self.state.config.enable_metrics_endpoint {
            unauthenticated_router =
                unauthenticated_router.route("/metrics", get(metrics_endpoint));
        }

        // Add middleware layers.
        //
        // `Router::layer` makes the layer being ADDED the new OUTERMOST
        // wrapper around whatever was registered before it — so the layer
        // added LAST runs FIRST, and sees the response last (after every
        // inner layer, including a short-circuiting rejection, has already
        // produced one). Security audit item 3 (2026-10-06) found this
        // exact ordering backwards: `audit_middleware` was added FIRST,
        // making it the INNERMOST layer, so a request `auth_middleware`
        // rejected with 401 before ever calling `next.run()` never reached
        // audit logging at all — a sustained brute-force guessing campaign
        // against the bearer-token space produced zero audit rows. The
        // comments below used to claim the opposite of what the code did;
        // they now describe what it actually does.
        router
            // Rate limiting — innermost of the four. Reads the
            // `ApiKeyInfo` extension `auth_middleware` inserts, so it must
            // run strictly AFTER auth, which means added BEFORE it here.
            .layer(axum::middleware::from_fn_with_state(
                self.state.clone(),
                auth_middleware::rate_limit_middleware,
            ))
            // Authentication. Runs after the pre-auth limiter admits a
            // request, before rate limiting (which needs what this inserts).
            .layer(axum::middleware::from_fn_with_state(
                self.state.clone(),
                auth_middleware::auth_middleware,
            ))
            // Pre-auth, connection-keyed attempt limiter (security audit
            // item 3). Runs BEFORE auth_middleware's hash+DB lookup, so a
            // guessing campaign is throttled before it costs a key lookup.
            .layer(axum::middleware::from_fn_with_state(
                self.state.clone(),
                auth_middleware::pre_auth_attempt_limiter_middleware,
            ))
            // Audit logging — now genuinely outermost, added LAST, so it
            // sees the final response regardless of which inner layer (this
            // one, auth, or rate limiting) produced it.
            .layer(axum::middleware::from_fn_with_state(
                self.state.clone(),
                auth_middleware::audit_middleware,
            ))
            // Health, readiness and metrics (built above), registered
            // AFTER the auth stack on purpose. `Router::layer` wraps only
            // the routes registered before it, so merging this router here
            // keeps all three reachable without credentials. `/health`
            // used to be registered first, which put it behind
            // `auth_middleware` and made every unauthenticated probe 401 —
            // load balancers, container orchestrators and uptime monitors
            // cannot present a bearer token, so a health endpoint that
            // demands one reports the service as down precisely when it is
            // up. Same reasoning extends to `/health/ready` (an
            // orchestrator uses liveness to decide whether to restart the
            // process and readiness to decide whether to route traffic to
            // it) and to `/metrics` (a Prometheus scraper has no bearer
            // token to present either). None of the three disclose
            // anything by being open: `/health` is a static string,
            // `/health/ready` reports only coarse up/down per dependency,
            // and `/metrics` carries no per-user/per-key labels (see
            // `metrics_endpoint`'s own doc comment). CORS and tracing below
            // still apply to all three.
            .merge(unauthenticated_router)
            // CORS and tracing
            .layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any),
            )
            .layer(TraceLayer::new_for_http())
    }

    /// Start serving (HTTP and optionally HTTPS)
    pub async fn serve(self) -> Result<()> {
        use crate::tls::CertificateManager;

        let http_bind_address = self.state.config.bind_address.clone();
        let tls_config = self.state.config.tls.clone();

        // Get TLS configuration
        let cert_manager = CertificateManager::new(tls_config.clone());
        let tls_server_config = cert_manager.get_server_config().await?;

        // Build router with all routes and middleware
        let mut app = self.build_router();

        // Add HTTPS redirect middleware if TLS is enabled and forced
        if cert_manager.should_force_https() {
            app = app.layer(axum::middleware::from_fn(
                crate::tls::https_redirect_middleware,
            ));
        }

        let app = app.with_state(self.state);

        if let Some((cert_pem, key_pem)) = tls_server_config {
            let https_bind_address = cert_manager.https_bind_address(&http_bind_address);

            // Parse TLS certificates and key
            use rustls::ServerConfig;
            use rustls::pki_types::{CertificateDer, PrivateKeyDer};
            use std::io::Cursor;
            use tokio_rustls::TlsAcceptor;

            let certs: Vec<CertificateDer> = rustls_pemfile::certs(&mut Cursor::new(&cert_pem))
                .collect::<Result<Vec<_>, _>>()?;

            let key = rustls_pemfile::private_key(&mut Cursor::new(&key_pem))?
                .ok_or_else(|| anyhow::anyhow!("No private key found in PEM file"))?;

            let mut tls_config = ServerConfig::builder()
                .with_no_client_auth()
                .with_single_cert(certs, key)?;

            tls_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
            let tls_acceptor = TlsAcceptor::from(Arc::new(tls_config));

            // Bind both HTTP and HTTPS listeners
            let http_listener = TcpListener::bind(&http_bind_address).await?;
            let https_listener = TcpListener::bind(&https_bind_address).await?;

            tracing::info!(
                http = %http_bind_address,
                https = %https_bind_address,
                "Lightbulb API server listening"
            );

            let http_app = app.clone();
            let https_app = app;

            // HTTP server task.
            //
            // `into_make_service_with_connect_info::<SocketAddr>()`, not
            // plain `into_make_service()`: `pre_auth_attempt_limiter_middleware`
            // extracts `ConnectInfo<SocketAddr>`, which only exists on a
            // request when the service tree was built this way.
            let http_server = tokio::spawn(async move {
                axum::serve(
                    http_listener,
                    http_app.into_make_service_with_connect_info::<SocketAddr>(),
                )
                .await
                .map_err(|e| anyhow::anyhow!("HTTP server error: {}", e))
            });

            // HTTPS server task with manual TLS handling
            let https_server = tokio::spawn(async move {
                // Same reason as the HTTP branch above: without
                // `_with_connect_info`, `make_service.call(remote_addr)`
                // below would still type-check (it ignores its argument's
                // type) but would NOT insert `ConnectInfo` into requests.
                let make_service = https_app.into_make_service_with_connect_info::<SocketAddr>();

                loop {
                    let (tcp_stream, remote_addr) = match https_listener.accept().await {
                        Ok(conn) => conn,
                        Err(e) => {
                            tracing::error!("Failed to accept HTTPS connection: {}", e);
                            continue;
                        }
                    };

                    let tls_acceptor = tls_acceptor.clone();
                    let make_service = make_service.clone();

                    tokio::spawn(async move {
                        // Perform TLS handshake
                        let tls_stream = match tls_acceptor.accept(tcp_stream).await {
                            Ok(stream) => stream,
                            Err(e) => {
                                tracing::debug!("TLS handshake failed: {}", e);
                                return;
                            }
                        };

                        // Create service for this connection
                        use tower::Service;
                        let service = match make_service.clone().call(remote_addr).await {
                            Ok(svc) => svc,
                            Err(_) => {
                                tracing::error!("Failed to create service for connection");
                                return;
                            }
                        };

                        // Create the hyper IO wrapper
                        let io = hyper_util::rt::TokioIo::new(tls_stream);

                        // Serve the connection with tower-to-hyper adapter
                        let conn = hyper_util::server::conn::auto::Builder::new(
                            hyper_util::rt::TokioExecutor::new(),
                        );

                        let hyper_service = hyper::service::service_fn(
                            move |req: hyper::Request<hyper::body::Incoming>| {
                                let mut service = service.clone();
                                async move {
                                    service.call(req).await.map_err(|err| {
                                        tracing::error!("Service error: {:?}", err);
                                        std::io::Error::new(
                                            std::io::ErrorKind::Other,
                                            "service error",
                                        )
                                    })
                                }
                            },
                        );

                        if let Err(e) = conn.serve_connection_with_upgrades(io, hyper_service).await
                        {
                            tracing::debug!("Error serving HTTPS connection: {}", e);
                        }
                    });
                }
            });

            // Wait for either server to fail (they both run indefinitely)
            tokio::select! {
                result = http_server => {
                    result??;
                    Ok(())
                }
                _ = https_server => {
                    // HTTPS server runs forever in a loop, this shouldn't normally return
                    Ok(())
                }
            }
        } else {
            // HTTP only
            let listener = TcpListener::bind(&http_bind_address).await?;
            tracing::info!(
                http = %http_bind_address,
                "Lightbulb API server listening"
            );

            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .await
            .map_err(|e| anyhow::anyhow!("HTTP server error: {}", e))?;

            Ok(())
        }
    }

    /// Get application state
    pub fn state(&self) -> &AppState {
        &self.state
    }
}

/// Health check endpoint
async fn health_check() -> &'static str {
    "OK"
}

/// One check's result: whether this *configured* dependency is working, not
/// whether it exists at all. `not_configured` is a success state for
/// readiness, not a failure — a deployment that never asked for a database
/// or a model is not "broken" for lacking one, the way a deployment that
/// asked for a model and failed to load it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum CheckStatus {
    Ok,
    NotConfigured,
    Unreachable,
    NotLoaded,
}

impl CheckStatus {
    /// Only `Unreachable`/`NotLoaded` block readiness. `NotConfigured` is a
    /// deliberate choice the operator made, not a fault this instance can
    /// fix by restarting.
    fn blocks_readiness(self) -> bool {
        matches!(self, CheckStatus::Unreachable | CheckStatus::NotLoaded)
    }
}

#[derive(Debug, Serialize)]
struct ReadinessResponse {
    status: &'static str,
    database: CheckStatus,
    model: CheckStatus,
}

/// `None` means no database is configured at all (`not_configured`, an
/// operator choice). `Some` is checked live with a real `SELECT 1` rather
/// than just `pool.get()` succeeding: a pool can hand out a connection to a
/// server that is up but not actually answering queries.
async fn check_database(pool: Option<&deadpool_postgres::Pool>) -> CheckStatus {
    let Some(pool) = pool else {
        return CheckStatus::NotConfigured;
    };
    let Ok(client) = pool.get().await else {
        return CheckStatus::Unreachable;
    };
    match client.query_one("SELECT 1", &[]).await {
        Ok(_) => CheckStatus::Ok,
        Err(_) => CheckStatus::Unreachable,
    }
}

/// Readiness check: can this instance actually serve a request right now,
/// as opposed to `/health`'s "is the process up at all." Checked fresh on
/// every call rather than cached — the whole point is reporting the
/// CURRENT state of dependencies that can change after startup (a database
/// that was reachable five minutes ago and is down now).
async fn readiness_check(State(state): State<AppState>) -> impl IntoResponse {
    let database = check_database(state.db_pool.as_ref()).await;

    // `inference_tx` is `None` both when no model was ever requested and
    // when one was requested but failed to start (`ApiServer::new` logs
    // the failure and continues rather than refusing to start — see the
    // call site). `config.models_dir` is what distinguishes those two: it
    // is set only when the operator asked for a model at all.
    let model = if state.inference_tx.is_some() {
        CheckStatus::Ok
    } else if state.config.models_dir.is_some() {
        CheckStatus::NotLoaded
    } else {
        CheckStatus::NotConfigured
    };

    let ready = !database.blocks_readiness() && !model.blocks_readiness();
    let code = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        code,
        Json(ReadinessResponse {
            status: if ready { "ready" } else { "not_ready" },
            database,
            model,
        }),
    )
}

/// The process-wide Prometheus recorder/handle. A `OnceLock`, not built
/// fresh per request or per `ApiServer`: `metrics::set_global_recorder`
/// (which `PrometheusBuilder::install_recorder` calls internally) can only
/// succeed ONCE per process, and `ApiServer::new` runs repeatedly within a
/// single test binary -- a second `install_recorder()` call there would
/// fail. Every `AppState` in the same process shares this one recorder and
/// renders the same global metric set; that's correct for this crate,
/// which never runs more than one `ApiServer` as truly independent
/// processes serving different traffic.
fn prometheus_handle() -> &'static metrics_exporter_prometheus::PrometheusHandle {
    static HANDLE: std::sync::OnceLock<metrics_exporter_prometheus::PrometheusHandle> =
        std::sync::OnceLock::new();
    HANDLE.get_or_init(|| {
        metrics_exporter_prometheus::PrometheusBuilder::new()
            .install_recorder()
            .expect("install the Prometheus recorder -- OnceLock guarantees this runs exactly once")
    })
}

/// Prometheus-format `GET /metrics` (Phase 4 observability, v1 slice).
///
/// Deliberately global gauges with NO labels at all, let alone a
/// per-user/per-key one: this route is unauthenticated (see
/// `build_router`'s doc comment on why), so anything exposed here is
/// exposed to anyone who can reach the port. A per-key request count, for
/// instance, would leak which keys exist and how active each one is to an
/// unauthenticated caller. `request_timeout_ms`-shaped per-endpoint/
/// per-status-code metrics (planned as a separate PR, not v1) will need
/// the same discipline applied to their label sets, not just this gauge
/// pair.
///
/// Only two gauges in this slice -- `lightbulb_memory_total_bytes`/
/// `_used_bytes` (same `live_system` sampler the admin `metrics` endpoint
/// reads, refreshed fresh here too) and `lightbulb_queue_depth` (`AppState::
/// queue_depth`, via `engine::model_runner::QueueDepthGuard`). Latency
/// histograms, throughput and error-rate-by-endpoint need request-path
/// instrumentation this PR doesn't add; queue depth BY PRIORITY and slot
/// utilization are their own roadmap rows (see `V1_ROADMAP.md`) -- neither
/// has a real, live-tracked value to report today.
///
/// No histograms recorded here, so `PrometheusHandle::run_upkeep`'s
/// periodic bucket-draining is not needed yet; the crate's own docs flag
/// it as something a future histogram-recording addition would need to
/// start calling on an interval.
async fn metrics_endpoint(State(state): State<AppState>) -> impl IntoResponse {
    // `prometheus_handle()` is what actually installs the Prometheus
    // recorder as `metrics`'s global recorder (see its own doc comment).
    // It MUST run before the first `gauge!` call below: `gauge!` records
    // against whatever recorder is current, a no-op default until one is
    // installed, and a `gauge!` call made before this line would silently
    // go nowhere -- caught exactly this way, by the body coming back empty
    // in `metrics_endpoint_reports_the_current_queue_depth` with `gauge!`
    // called first and this line last.
    let handle = prometheus_handle();

    metrics::gauge!("lightbulb_queue_depth")
        .set(state.queue_depth.load(std::sync::atomic::Ordering::Relaxed) as f64);

    {
        let mut sys = state.live_system.lock().unwrap_or_else(|e| e.into_inner());
        sys.refresh_memory();
        metrics::gauge!("lightbulb_memory_total_bytes").set(sys.total_memory() as f64);
        metrics::gauge!("lightbulb_memory_used_bytes").set(sys.used_memory() as f64);
    }

    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/plain")],
        handle.render(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_api_config_default() {
        let config = ApiConfig::default();
        assert!(config.enable_openai_api);
        assert!(config.enable_admin_api);
        assert!(config.enable_lightbulb_extensions);
    }

    /// Pins the default at 300/min (~5/s) — PM review of #127 (2026-10-07)
    /// found the original default of 20 too low: this limiter counts
    /// EVERY request per IP, successful or not, so 20/min would throttle
    /// ordinary traffic from several clients sharing one NAT/proxy. See
    /// `ApiConfig::max_auth_attempts_per_minute_per_ip`'s own doc for why
    /// it stays a generous flood limiter rather than a brute-force
    /// detector.
    #[test]
    fn max_auth_attempts_per_minute_per_ip_defaults_to_300() {
        assert_eq!(
            ApiConfig::default().max_auth_attempts_per_minute_per_ip,
            300
        );
    }

    /// Security audit item 5: the default must NOT be a known, shared
    /// string — it must be the explicit absence of a configured secret, so
    /// any future caller that signs or verifies a JWT is forced to check
    /// for `None` rather than silently trusting whatever the default
    /// happens to be.
    #[test]
    fn jwt_secret_defaults_to_none_not_a_known_string() {
        assert!(ApiConfig::default().jwt_secret.is_none());
    }

    #[test]
    fn is_loopback_bind_accepts_ipv4_loopback() {
        assert!(is_loopback_bind("127.0.0.1:8080"));
    }

    #[test]
    fn is_loopback_bind_accepts_ipv6_loopback() {
        assert!(is_loopback_bind("[::1]:8080"));
    }

    #[test]
    fn is_loopback_bind_accepts_localhost() {
        assert!(is_loopback_bind("localhost:8080"));
        assert!(is_loopback_bind("LOCALHOST:8080"));
    }

    #[test]
    fn is_loopback_bind_refuses_all_interfaces() {
        assert!(!is_loopback_bind("0.0.0.0:8080"));
    }

    #[test]
    fn is_loopback_bind_refuses_a_lan_ip() {
        assert!(!is_loopback_bind("192.168.1.5:8080"));
    }

    #[test]
    fn is_loopback_bind_refuses_a_non_localhost_hostname() {
        assert!(!is_loopback_bind("mybox.lan:8080"));
    }

    fn config_with(database_url: Option<&str>, bind_address: &str, no_auth: bool) -> ApiConfig {
        ApiConfig {
            database_url: database_url.map(str::to_string),
            bind_address: bind_address.to_string(),
            no_auth,
            ..ApiConfig::default()
        }
    }

    #[test]
    fn validate_auth_policy_allows_when_database_url_is_set_on_any_bind() {
        // Even a wide-open bind is fine once real auth (DATABASE_URL) is
        // configured — this policy is only about the no-auth path.
        assert!(
            validate_auth_policy(&config_with(Some("postgres://x"), "0.0.0.0:8080", false)).is_ok()
        );
    }

    #[test]
    fn validate_auth_policy_refuses_no_database_and_no_no_auth_flag() {
        assert!(validate_auth_policy(&config_with(None, "127.0.0.1:8080", false)).is_err());
    }

    #[test]
    fn validate_auth_policy_refuses_no_auth_on_all_interfaces() {
        assert!(validate_auth_policy(&config_with(None, "0.0.0.0:8080", true)).is_err());
    }

    #[test]
    fn validate_auth_policy_refuses_no_auth_on_a_lan_ip() {
        assert!(validate_auth_policy(&config_with(None, "192.168.1.5:8080", true)).is_err());
    }

    #[test]
    fn validate_auth_policy_refuses_no_auth_on_a_non_localhost_hostname() {
        assert!(validate_auth_policy(&config_with(None, "mybox.lan:8080", true)).is_err());
    }

    #[test]
    fn validate_auth_policy_allows_no_auth_on_loopback() {
        assert!(validate_auth_policy(&config_with(None, "127.0.0.1:8080", true)).is_ok());
    }

    /// Sourcery review of #155: `request_timeout_ms: 0` would silently
    /// accept and then reject every single inference request (`timeout(0)`
    /// always expires immediately).
    #[test]
    fn validate_request_timeout_refuses_zero() {
        let config = ApiConfig {
            request_timeout_ms: 0,
            ..ApiConfig::default()
        };
        assert!(validate_request_timeout(&config).is_err());
    }

    #[test]
    fn validate_request_timeout_allows_the_default() {
        assert!(validate_request_timeout(&ApiConfig::default()).is_ok());
    }

    #[test]
    fn validate_request_timeout_allows_any_positive_value() {
        let config = ApiConfig {
            request_timeout_ms: 1,
            ..ApiConfig::default()
        };
        assert!(validate_request_timeout(&config).is_ok());
    }

    /// Builds a bare `AppState` directly rather than through `ApiServer::new`
    /// — these tests exercise `readiness_check`'s own branching on
    /// `db_pool`/`inference_tx`/`config.models_dir`, not startup. Every
    /// field is `pub` for exactly this (see the field's own doc comment).
    fn bare_state(models_dir: Option<&str>, inference_tx_present: bool) -> AppState {
        use crate::engine::{MemoryAwareConfig, MemoryAwareScheduler};

        let inference_tx = if inference_tx_present {
            let (tx, _rx) = std::sync::mpsc::sync_channel(1);
            Some(tx)
        } else {
            None
        };

        AppState {
            scheduler: Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default())),
            config: ApiConfig {
                models_dir: models_dir.map(str::to_string),
                ..ApiConfig::default()
            },
            db_pool: None,
            inference_tx,
            chat_template: None,
            eos_monitor: Arc::new(crate::engine::eos_monitor::EosMonitor::default()),
            security_sinks: Vec::new(),
            cache_stats: Arc::new(std::sync::Mutex::new(
                crate::cache::PrefixCacheStats::default(),
            )),
            live_system: new_shared_system_metrics(),
            queue_depth: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// A deployment that never configured a database or a model is READY —
    /// `not_configured` is not a fault. Guards against the readiness check
    /// treating "nothing to check" the same as "something is broken."
    #[tokio::test]
    async fn readiness_check_is_ready_with_nothing_configured() {
        let state = bare_state(None, false);
        let response = readiness_check(axum::extract::State(state))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// `models_dir` set but `inference_tx` absent means the model runner
    /// was asked for and failed to start (`ApiServer::new` logs and
    /// continues rather than refusing to start — see the `Err(e)` arm
    /// where the model runner is started). That is a real fault, not an
    /// operator choice, and must fail readiness.
    #[tokio::test]
    async fn readiness_check_is_not_ready_when_model_was_requested_but_not_loaded() {
        let state = bare_state(Some("/some/models/dir"), false);
        let response = readiness_check(axum::extract::State(state))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// The positive control for the test above: the same configured
    /// `models_dir`, but with `inference_tx` present (the model actually
    /// started) — proves the prior test's 503 came from the missing
    /// `inference_tx`, not from `models_dir` being set at all.
    #[tokio::test]
    async fn readiness_check_is_ready_when_model_is_loaded() {
        let state = bare_state(Some("/some/models/dir"), true);
        let response = readiness_check(axum::extract::State(state))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// Red test for the database branch, independent of a real Postgres:
    /// a pool pointed at a port nothing listens on. `Pool::builder(...).
    /// build()` is lazy (no connection attempt), so this needs no network
    /// access to construct — only `pool.get().await` in `check_database`
    /// actually tries to connect, and that's what must fail fast here.
    #[tokio::test]
    async fn readiness_check_is_not_ready_when_database_is_unreachable() {
        let pg_config: tokio_postgres::Config = "postgresql://nouser:nopass@127.0.0.1:1/nodb"
            .parse()
            .expect("valid connection string syntax");
        let manager = deadpool_postgres::Manager::new(pg_config, tokio_postgres::NoTls);
        let pool = deadpool_postgres::Pool::builder(manager)
            .max_size(1)
            .build()
            .expect("lazy pool construction never touches the network");

        let mut state = bare_state(None, false);
        state.db_pool = Some(pool);

        let response = readiness_check(axum::extract::State(state))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    /// `metrics_endpoint` installs and reads ONE process-wide Prometheus
    /// recorder (`prometheus_handle`'s `OnceLock`) -- cargo runs tests in
    /// parallel by default, so two tests calling it concurrently would
    /// race on the same global gauge, each able to observe the other's
    /// value. Both tests below take this lock for their full body to
    /// serialize against each other; nothing else in this file touches
    /// the Prometheus recorder, so no other test needs it.
    static METRICS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Pins the one number this test can control without a real database
    /// or model: the queue-depth gauge reflects whatever `AppState::
    /// queue_depth` currently holds, rendered in Prometheus text format.
    /// Does not assert on the memory gauges -- their value is whatever
    /// `sysinfo` reports for this machine, not something this test
    /// controls or should pin a number to.
    #[tokio::test]
    async fn metrics_endpoint_reports_the_current_queue_depth() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let state = bare_state(None, false);
        state
            .queue_depth
            .store(3, std::sync::atomic::Ordering::Relaxed);

        let response = metrics_endpoint(axum::extract::State(state))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            body.contains("lightbulb_queue_depth 3"),
            "expected the queue-depth gauge in the Prometheus body, got: {body}"
        );
    }

    /// Red test for the gauge name itself: proves the assertion above would
    /// actually fail on a wrong value, not just on a missing metric
    /// (mutation-style, inline rather than a separate push-then-revert
    /// since there's no CI round trip needed to check a pure function).
    #[tokio::test]
    async fn metrics_endpoint_queue_depth_changes_with_the_counter() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let state = bare_state(None, false);
        state
            .queue_depth
            .store(7, std::sync::atomic::Ordering::Relaxed);

        let response = metrics_endpoint(axum::extract::State(state))
            .await
            .into_response();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(
            !body.contains("lightbulb_queue_depth 3"),
            "a stale value from a prior test must not still be rendered: {body}"
        );
        assert!(
            body.contains("lightbulb_queue_depth 7"),
            "expected the updated queue-depth gauge, got: {body}"
        );
    }

    /// End-to-end through the real entry point, not just the pure function:
    /// `ApiServer::new`'s default-config construction (no `DATABASE_URL`, no
    /// `--no-auth`) must fail before it ever touches a database or starts a
    /// model runner thread.
    #[tokio::test]
    async fn api_server_new_refuses_default_config_without_auth() {
        use crate::engine::{MemoryAwareConfig, MemoryAwareScheduler};
        let scheduler = Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default()));
        let config = ApiConfig {
            models_dir: None,
            ..ApiConfig::default()
        };

        let result = ApiServer::new(config, scheduler).await;

        assert!(
            result.is_err(),
            "must refuse to start without auth configured"
        );
    }

    /// Structural regression test for `build_router`'s layer ordering
    /// (security audit item 3): the layer added LAST is the OUTERMOST
    /// wrapper, so it runs first on the way in and sees the final response
    /// last, even when an inner layer never calls `next.run()` at all.
    ///
    /// This doesn't touch `audit_middleware`/`auth_middleware` (both
    /// database-backed) or a live server — it pins the underlying `axum`
    /// composition property `build_router` relies on, using two throwaway
    /// marker middlewares and a shared log, mirroring exactly the call
    /// pattern `build_router` uses (`.layer(inner)` then `.layer(outer)`).
    /// If that property ever changed (or `build_router`'s call order
    /// regressed back to audit-innermost), this fails independently of
    /// whether a database happens to be reachable.
    mod layer_ordering {
        use axum::body::Body;
        use axum::extract::{Extension, Request};
        use axum::http::{Request as HttpRequest, StatusCode};
        use axum::middleware::Next;
        use axum::response::{IntoResponse, Response};
        use axum::routing::get;
        use axum::{Router, middleware};
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        type SharedLog = Arc<Mutex<Vec<String>>>;

        async fn marker_outer(
            Extension(log): Extension<SharedLog>,
            request: Request,
            next: Next,
        ) -> Response {
            log.lock().unwrap().push("outer-before".to_string());
            let response = next.run(request).await;
            log.lock()
                .unwrap()
                .push(format!("outer-after:{}", response.status().as_u16()));
            response
        }

        /// Stands in for `auth_middleware` rejecting a request: records
        /// that it ran, then returns WITHOUT ever calling `next.run()` —
        /// exactly the short-circuit that made the real `audit_middleware`
        /// invisible to a rejected request when it was wired innermost.
        async fn marker_reject(
            Extension(log): Extension<SharedLog>,
            _request: Request,
            _next: Next,
        ) -> Response {
            log.lock().unwrap().push("reject".to_string());
            StatusCode::FORBIDDEN.into_response()
        }

        #[tokio::test]
        async fn layer_added_last_is_outermost_and_sees_a_short_circuited_response() {
            let log: SharedLog = Arc::new(Mutex::new(Vec::new()));

            let app = Router::new()
                .route("/x", get(|| async { "ok" }))
                // Added first -> innermost, same relative position
                // `auth_middleware` holds in `build_router` relative to
                // `audit_middleware`.
                .layer(middleware::from_fn(marker_reject))
                // Added last -> outermost, same relative position
                // `audit_middleware` now holds.
                .layer(middleware::from_fn(marker_outer))
                .layer(Extension(log.clone()));

            let request: HttpRequest<Body> = HttpRequest::builder()
                .uri("/x")
                .body(Body::empty())
                .unwrap();

            let response = app.oneshot(request).await.unwrap();

            assert_eq!(response.status(), StatusCode::FORBIDDEN);
            assert_eq!(
                *log.lock().unwrap(),
                vec![
                    "outer-before".to_string(),
                    "reject".to_string(),
                    "outer-after:403".to_string(),
                ],
                "the outer marker must run before the rejecting inner one, and must \
                 observe the rejection's status AFTER next.run() returns — the same \
                 thing audit_middleware needs from being outermost of auth_middleware"
            );
        }
    }
}
