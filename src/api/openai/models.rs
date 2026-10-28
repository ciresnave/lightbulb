// SPDX-License-Identifier: MIT OR Apache-2.0
//! Models Endpoint
//!
//! OpenAI-compatible `/v1/models` (list) and `/v1/models/{model}` (retrieve).
//!
//! The roster is what is actually loaded: the one model the runner thread
//! was started with (`state.config.default_model`), and only while
//! `state.inference_tx` is `Some`. With no model loaded the list is empty,
//! which is the truth; it used to be a hardcoded roster of three names
//! (`lightbulb-default`, `lightbulb-7b`, `lightbulb-13b`) that no server ever
//! had, which sends a client that picks from the list straight into the
//! `model ... is not loaded` 400 that `chat_completions` returns.

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::Serialize;
use std::time::UNIX_EPOCH;

use crate::api::AppState;

/// Model list response (OpenAI-compatible)
#[derive(Debug, Clone, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Model information
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// The models this server can actually serve right now.
fn loaded_models(state: &AppState) -> Vec<ModelInfo> {
    if state.inference_tx.is_none() {
        return Vec::new();
    }
    let id = state.config.default_model.clone();
    // `created` is the model file's mtime when it can be read, else 0
    // (OpenAI clients only require an integer).
    let created = state
        .config
        .models_dir
        .as_ref()
        .and_then(|dir| std::fs::metadata(std::path::Path::new(dir).join(&id)).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    vec![ModelInfo {
        id,
        object: "model".to_string(),
        created,
        owned_by: "lightbulb".to_string(),
    }]
}

/// `GET /v1/models` (and `POST`, kept because it was the registered method
/// before 2026-10-25 and nothing is gained by breaking a caller that used it).
pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(ModelListResponse {
            object: "list".to_string(),
            data: loaded_models(&state),
        }),
    )
}

/// `GET /v1/models/{model}`: the model object, or an OpenAI-shaped 404.
pub async fn retrieve_model(
    State(state): State<AppState>,
    Path(model): Path<String>,
) -> impl IntoResponse {
    match loaded_models(&state).into_iter().find(|m| m.id == model) {
        Some(info) => (StatusCode::OK, Json(serde_json::json!(info))).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("The model '{model}' is not loaded on this server"),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, Request};
    use tower::ServiceExt;

    fn state(loaded: bool) -> AppState {
        use crate::engine::{MemoryAwareConfig, MemoryAwareScheduler};
        let inference_tx = loaded.then(|| std::sync::mpsc::sync_channel(1).0);
        AppState {
            scheduler: std::sync::Arc::new(std::sync::Mutex::new(MemoryAwareScheduler::new(
                MemoryAwareConfig::default(),
            ))),
            config: crate::api::ApiConfig {
                default_model: "tiny.gguf".to_string(),
                ..crate::api::ApiConfig::default()
            },
            pg_backend: None,
            inference_tx,
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

    async fn call(state: AppState, method: Method, uri: &str) -> (StatusCode, serde_json::Value) {
        let resp = super::super::routes()
            .with_state(state)
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn get_lists_exactly_the_loaded_model() {
        let (status, body) = call(state(true), Method::GET, "/v1/models").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["object"], "list");
        let data = body["data"].as_array().unwrap();
        assert_eq!(data.len(), 1, "{body}");
        assert_eq!(data[0]["id"], "tiny.gguf");
        assert_eq!(data[0]["object"], "model");
    }

    #[tokio::test]
    async fn no_loaded_model_means_an_empty_list_not_a_mock_roster() {
        let (status, body) = call(state(false), Method::GET, "/v1/models").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["data"].as_array().unwrap().len(), 0, "{body}");
        assert!(!body.to_string().contains("lightbulb-7b"), "{body}");
    }

    #[tokio::test]
    async fn post_still_answers_like_get() {
        let (status, body) = call(state(true), Method::POST, "/v1/models").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["data"][0]["id"], "tiny.gguf");
    }

    #[tokio::test]
    async fn retrieve_finds_the_loaded_model_and_404s_others() {
        let (status, body) = call(state(true), Method::GET, "/v1/models/tiny.gguf").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["id"], "tiny.gguf");

        let (status, body) = call(state(true), Method::GET, "/v1/models/nope").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(body["error"]["code"], "model_not_found");

        let (status, _) = call(state(false), Method::GET, "/v1/models/tiny.gguf").await;
        assert_eq!(
            status,
            StatusCode::NOT_FOUND,
            "nothing loaded -> nothing found"
        );
    }
}
