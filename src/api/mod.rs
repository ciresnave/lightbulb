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
pub mod types;

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::engine::model_runner::ModelRunner;
use crate::engine::MemoryAwareScheduler;
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

    /// JWT secret for token signing
    pub jwt_secret: String,

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

    /// Enable audit logging
    pub enable_audit_log: bool,

    /// TLS configuration
    pub tls: crate::tls::TlsConfig,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            database_url: None,
            bind_address: "0.0.0.0:8080".to_string(),
            enable_openai_api: true,
            enable_admin_api: true,
            enable_lightbulb_extensions: true,
            jwt_secret: "change-me-in-production".to_string(),
            rate_limit_per_minute: 60,
            enable_audit_log: true,
            models_dir: Some("./models".to_string()),
            default_model: "lightbulb-default".to_string(),
            model_max_batch_size: 8,
            model_context_length: 2048,
            tls: crate::tls::TlsConfig::default(),
        }
    }
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
    pub inference_tx: Option<std::sync::mpsc::Sender<crate::engine::model_runner::InferenceJob>>,

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
}

/// API server
pub struct ApiServer {
    /// Application state
    state: AppState,
}

impl ApiServer {
    /// Create new API server
    pub async fn new(config: ApiConfig, scheduler: Arc<MemoryAwareScheduler>) -> Result<Self> {
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

            println!("Database connected and migrations applied");
            Some(pool)
        } else {
            println!("No DATABASE_URL set — running without database (auth/audit/rate-limiting disabled)");
            None
        };

        let mut state = AppState {
            scheduler,
            config: config.clone(),
            db_pool,
            inference_tx: None,
            chat_template: None,
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
                ) {
                    Ok(sender) => {
                        state.inference_tx = Some(sender);

                        // Resolve the chat template HERE and only here. Both
                        // halves read files, and a per-request read would put
                        // filesystem I/O in the request path for a value that
                        // cannot change while the process runs.
                        //
                        // `Resolution::None` is stored as `None` rather than as
                        // an empty template: an empty source renders to an empty
                        // prompt, which reaches the model as a request to
                        // continue nothing. The handlers need to know the
                        // difference so they can fall back to the legacy join.
                        let t = chat_template::resolve_for_model(&model_path);
                        if t.resolved_by() != chat_template::Resolution::None {
                            println!(
                                "Chat template for {} resolved via {:?} (bos {:?}, eos {:?})",
                                model_path.display(),
                                t.resolved_by(),
                                t.tokens.bos,
                                t.tokens.eos
                            );
                            state.chat_template = Some(Arc::new(t));
                        }

                        println!("Started model runner for {}", model_path.display());
                    }
                    Err(e) => {
                        eprintln!(
                            "Failed to start model runner for {}: {}",
                            model_path.display(),
                            e
                        );
                    }
                }
            } else {
                println!(
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

        // Add middleware layers
        router
            // Audit logging (outermost - logs everything)
            .layer(axum::middleware::from_fn_with_state(
                self.state.clone(),
                auth_middleware::audit_middleware,
            ))
            // Rate limiting
            .layer(axum::middleware::from_fn_with_state(
                self.state.clone(),
                auth_middleware::rate_limit_middleware,
            ))
            // Authentication (innermost - validates before processing)
            .layer(axum::middleware::from_fn_with_state(
                self.state.clone(),
                auth_middleware::auth_middleware,
            ))
            // Health check, registered AFTER the auth stack on purpose.
            // `Router::layer` wraps only the routes registered before it, so
            // placing `/health` here keeps it reachable without credentials.
            // It used to be registered first, which put it behind
            // `auth_middleware` and made every unauthenticated probe 401 — load
            // balancers, container orchestrators and uptime monitors cannot
            // present a bearer token, so a health endpoint that demands one
            // reports the service as down precisely when it is up. The handler
            // returns a static string, so nothing is disclosed by leaving it
            // open. CORS and tracing below still apply to it.
            .route("/health", get(health_check))
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
            use rustls::pki_types::{CertificateDer, PrivateKeyDer};
            use rustls::ServerConfig;
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

            println!("Lightbulb API server listening on:");
            println!("  HTTP:  {}", http_bind_address);
            println!("  HTTPS: {}", https_bind_address);

            let http_app = app.clone();
            let https_app = app;

            // HTTP server task
            let http_server = tokio::spawn(async move {
                axum::serve(http_listener, http_app)
                    .await
                    .map_err(|e| anyhow::anyhow!("HTTP server error: {}", e))
            });

            // HTTPS server task with manual TLS handling
            let https_server = tokio::spawn(async move {
                let make_service = https_app.into_make_service();

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
            println!(
                "Lightbulb API server listening on HTTP: {}",
                http_bind_address
            );

            axum::serve(listener, app)
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
}
