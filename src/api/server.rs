// SPDX-License-Identifier: MIT OR Apache-2.0
//! `ApiServer`: construction, router assembly and serving.

use super::*;

impl ApiServer {
    /// Create new API server
    pub async fn new(config: ApiConfig, scheduler: SharedScheduler) -> Result<Self> {
        validate_auth_policy(&config)?;
        validate_request_timeout(&config)?;
        validate_server_config(&config)?;

        // PersistAnt adoption Step 5 (docs/PERSISTANT_ADOPTION_DESIGN.md): the
        // sole database path. `PgBackend::connect` is EAGER (opens a real
        // connection and runs `SHOW server_version` before returning), so a
        // momentary DB blip at startup (e.g. container orchestration
        // starting the app and the database at the same time) would
        // otherwise fail `ApiServer::new` itself. `connect_with_retry`
        // absorbs that with a short bounded retry; see its own doc comment
        // for what happens if the database is genuinely unreachable.
        let pg_backend = match &config.database_url {
            Some(database_url) => connect_with_retry(database_url).await,
            None => {
                tracing::info!(
                    "No DATABASE_URL set — running without database (auth/audit/rate-limiting disabled)"
                );
                None
            }
        };

        spawn_auth_attempt_counts_cleanup(pg_backend.clone());
        audit_retention::spawn_cleanup(pg_backend.clone(), config.audit_retention_days);

        // Install the Prometheus recorder NOW, not lazily on the first
        // `/metrics` request. `request_metrics_middleware` (Phase 4
        // observability, latency/throughput/error-rate PR) records
        // against whatever recorder is globally current the moment EVERY
        // request is handled, starting with the very first one this
        // server ever serves -- if that happened before anything called
        // `prometheus_handle()`, the recording would silently go to the
        // no-op default recorder and be lost, the same bug class
        // `metrics_endpoint` itself hit during development (see its own
        // doc comment). Calling it here, unconditionally, removes the
        // "first request wins or loses the race" question entirely.
        let metrics_handle = prometheus_handle();

        // Recording histogram/summary data (unlike the plain gauges
        // `metrics_endpoint` sets) needs periodic upkeep -- the crate's
        // own docs: `install_recorder`/`build_recorder` (unlike `install`)
        // don't spawn this automatically, and skipping it lets internal
        // bucket/decay state grow unbounded. Same pattern as the
        // `auth_attempt_counts` cleanup task above, but unconditional: the
        // Prometheus recorder exists regardless of whether a database is
        // configured.
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
            loop {
                interval.tick().await;
                metrics_handle.run_upkeep();
            }
        });

        let security_sinks = security_event::build_sinks(&config.security_alert_email);

        let mut state = AppState {
            scheduler,
            config: config.clone(),
            pg_backend,
            manager: None,
            models: ModelRegistry::default(),
            eos_monitor: Arc::new(crate::engine::eos_monitor::EosMonitor::default()),
            security_sinks,
            cache_stats: Arc::new(std::sync::Mutex::new(
                crate::cache::PrefixCacheStats::default(),
            )),
            live_system: new_shared_system_metrics(),
            queue_depth: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        };

        try_start_model_runner(&mut state);
        state.manager = build_manager(&state);

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
        let router = router
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
            // Request-body cap (see `MAX_REQUEST_BODY_BYTES`; configurable through `max_request_body_bytes`).
            .layer(axum::extract::DefaultBodyLimit::max(
                self.state.config.max_request_body_bytes,
            ))
            // CORS, tracing, request id: see with_cors_trace_request_id
            ;
        let router = with_cors_trace_request_id(router, &self.state.config.cors_allowed_origins);
        router
            // Request-duration histogram + request-count counter, added
            // LAST so it is the truly outermost layer and times EVERY
            // request this server serves -- `/health`/`/health/ready`/
            // `/metrics` included, not just the authenticated routes.
            // Placed after CORS/tracing rather than before: a preflight
            // CORS response short-circuited by the `CorsLayer` itself
            // should still be counted, and being outermost guarantees
            // that regardless of which inner layer produced the response
            // (same "last .layer() = outermost" rule this function
            // documents throughout).
            .layer(axum::middleware::from_fn(request_metrics_middleware))
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
