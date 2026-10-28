// SPDX-License-Identifier: MIT OR Apache-2.0
//! Lightbulb Extensions API Module
//!
//! The `/v1/lightbulb/{knowledge,reasoning,state,tools}/*` routes are
//! reserved but NOT IMPLEMENTED: every one answers `501 Not Implemented`.
//!
//! Until 2026-10-25 each handler ended in a `// TODO` and returned a success
//! body that did nothing (`knowledge/add` returned `success: true` and a
//! random UUID and stored nothing). A client that believed that response lost
//! data without any error, so the routes now say what is true. Keeping the
//! routes (rather than deleting them) means an authenticated client gets a
//! machine-readable 501, not a 404 that could mean a typo, and the paths stay
//! claimed for when the backends exist (ROADMAP "Spec-to-roadmap audit" A2;
//! PM ruling 2026-10-25: 501 until real).

use axum::{
    Json, Router,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};

use crate::api::AppState;

/// Every reserved route, with the method it will use when implemented.
const RESERVED: &[(&str, bool)] = &[
    // (path, is_post)
    ("/v1/lightbulb/knowledge/query", true),
    ("/v1/lightbulb/knowledge/add", true),
    ("/v1/lightbulb/knowledge/validate", true),
    ("/v1/lightbulb/reasoning/budget", true),
    ("/v1/lightbulb/reasoning/convergence", false),
    ("/v1/lightbulb/reasoning/stats", false),
    ("/v1/lightbulb/state/save", true),
    ("/v1/lightbulb/state/restore", true),
    ("/v1/lightbulb/state/branch", true),
    ("/v1/lightbulb/state/list", false),
    ("/v1/lightbulb/tools/list", false),
    ("/v1/lightbulb/tools/register", true),
];

/// Create Lightbulb extension routes (all 501 until implemented).
pub fn routes() -> Router<AppState> {
    RESERVED
        .iter()
        .fold(Router::new(), |router, &(path, is_post)| {
            if is_post {
                router.route(path, post(not_implemented))
            } else {
                router.route(path, get(not_implemented))
            }
        })
}

/// The one handler behind every reserved route.
async fn not_implemented() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": "This /v1/lightbulb extension endpoint is reserved but not \
                            implemented; nothing was stored or changed.",
                "type": "not_implemented",
                "code": "not_implemented",
            }
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request};
    use tower::ServiceExt;

    fn state() -> AppState {
        use crate::engine::{MemoryAwareConfig, MemoryAwareScheduler};
        AppState {
            scheduler: std::sync::Arc::new(std::sync::Mutex::new(MemoryAwareScheduler::new(
                MemoryAwareConfig::default(),
            ))),
            config: crate::api::ApiConfig::default(),
            pg_backend: None,
            inference_tx: None,
            cache_control: None,
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

    /// Every reserved route answers 501 with the typed error body, on the
    /// method it will use, and never a success. `RESERVED` is the list under
    /// test, so a route added to it is covered without editing this test.
    #[tokio::test]
    async fn every_reserved_route_is_501_and_never_a_fake_success() {
        assert_eq!(RESERVED.len(), 12, "the 12 routes the stubs used to serve");
        for &(path, is_post) in RESERVED {
            let method = if is_post { Method::POST } else { Method::GET };
            let resp = routes()
                .with_state(state())
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .header("content-type", "application/json")
                        .body(Body::from("{}"))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED, "{path}");
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["error"]["code"], "not_implemented", "{path}");
            assert!(body.get("success").is_none(), "{path}: {body}");
        }
    }

    /// The wrong method on a reserved route is not a 501: the route table
    /// really is per-method, so the test above is exercising the right arm.
    #[tokio::test]
    async fn wrong_method_is_405_not_501() {
        let resp = routes()
            .with_state(state())
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/lightbulb/knowledge/add")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    }
}
