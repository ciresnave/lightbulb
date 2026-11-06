// SPDX-License-Identifier: MIT OR Apache-2.0
//! Models Endpoint
//!
//! OpenAI-compatible `/v1/models` (list) and `/v1/models/{model}` (retrieve).
//!
//! Without a model manager the roster is what is actually loaded: the one model the
//! runner thread was started with (`state.config.default_model`), and only while the
//! model entry's `inference_tx` is `Some`. With no model loaded the list is empty, which
//! is the truth; it used to be a hardcoded roster of three names that no server ever
//! had, which sends a client that picks from the list straight into a 400.
//!
//! With a manager (ROADMAP F2, P3) a request can name any model in the models directory
//! and it is loaded on demand, so the roster is every `.gguf` there, each marked
//! `lightbulb_loaded` true or false.

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
    /// Not part of the OpenAI schema. Absent when the model's files cannot
    /// be read.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lightbulb_memory: Option<ModelMemory>,
    /// Not part of the OpenAI schema. The device the model load selected
    /// (`"cpu"` or `"cuda:<n>"`), so a run can prove where it ran. Absent when
    /// no selection has been recorded in this process.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lightbulb_device: Option<String>,
    /// Not part of the OpenAI schema. Whether the model is in memory now. A request for an
    /// unloaded model loads it (evicting idle ones if the device is full).
    pub lightbulb_loaded: bool,
}

/// What lightbulb knows about the loaded model's memory footprint.
#[derive(Debug, Clone, Serialize)]
pub struct ModelMemory {
    /// Weight bytes on disk (the `.gguf` file, or the SafeTensors shards).
    /// A lower bound on the weights' memory: the f32 loader upcasts, and the
    /// KV cache and activations come on top.
    pub weights_bytes_on_disk: u64,
}

/// One entry of the list. `models_dir` is where `id` lives, if known.
fn model_info(id: String, models_dir: Option<&str>, loaded: bool) -> ModelInfo {
    // `created` is the model file's mtime when it can be read, else 0
    // (OpenAI clients only require an integer).
    let model_path = models_dir.map(|dir| std::path::Path::new(dir).join(&id));
    let created = model_path
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    ModelInfo {
        id,
        object: "model".to_string(),
        created,
        owned_by: "lightbulb".to_string(),
        lightbulb_memory: model_path
            .as_deref()
            .and_then(crate::model_fuel::preflight::weights_lower_bound)
            .map(|weights_bytes_on_disk| ModelMemory {
                weights_bytes_on_disk,
            }),
        lightbulb_device: loaded.then(crate::model_fuel::device::chosen).flatten(),
        lightbulb_loaded: loaded,
    }
}

/// The `.gguf` files directly under `dir`, sorted (hidden files are not models: the
/// loader refuses a name that starts with '.').
fn gguf_names(dir: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| !n.starts_with('.') && n.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    names.sort();
    names
}

/// The models a request can name: with a [`ModelManager`](crate::api::manager::ModelManager)
/// every model in the models directory (loaded or not, `lightbulb_loaded` says which);
/// without one, only the loaded model.
fn available_models(state: &AppState) -> Vec<ModelInfo> {
    let dir = state.config.models_dir.as_deref();
    if state.manager.is_none() {
        if state.model().inference_tx.is_none() {
            return Vec::new();
        }
        return vec![model_info(state.config.default_model.clone(), dir, true)];
    }
    let Some(dir) = dir else {
        return Vec::new();
    };
    gguf_names(dir)
        .into_iter()
        .map(|name| {
            let loaded = state
                .models
                .get(&name)
                .is_some_and(|e| e.inference_tx.is_some());
            model_info(name, Some(dir), loaded)
        })
        .collect()
}

/// `GET /v1/models` (and `POST`, kept because it was the registered method
/// before 2026-10-25 and nothing is gained by breaking a caller that used it).
pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(ModelListResponse {
            object: "list".to_string(),
            data: available_models(&state),
        }),
    )
}

/// `GET /v1/models/{model}`: the model object, or an OpenAI-shaped 404.
pub async fn retrieve_model(
    State(state): State<AppState>,
    Path(model): Path<String>,
) -> impl IntoResponse {
    match available_models(&state).into_iter().find(|m| m.id == model) {
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
            manager: None,
            models: crate::api::ModelRegistry::single(
                "test-model",
                crate::api::ModelEntry {
                    inference_tx,
                    cache_control: None,
                    chat_template: None,
                    ..Default::default()
                },
            ),
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

    struct NoLoad;
    impl crate::api::manager::ModelLoader for NoLoad {
        fn needed_bytes(&self, n: &str) -> Result<u64, crate::api::manager::LoadError> {
            Err(crate::api::manager::LoadError::NotFound(n.into()))
        }
        fn load(
            &self,
            _: &crate::api::ModelRegistry,
            n: &str,
        ) -> Result<(), crate::api::manager::LoadError> {
            Err(crate::api::manager::LoadError::NotFound(n.into()))
        }
    }

    #[tokio::test]
    async fn with_a_manager_every_gguf_in_the_directory_is_listed_with_its_loaded_flag() {
        let dir = tempfile::tempdir().unwrap();
        for f in ["b.gguf", "a.GGUF", ".hidden.gguf", "notes.txt"] {
            std::fs::write(dir.path().join(f), b"x").unwrap();
        }
        std::fs::create_dir(dir.path().join("sub.gguf")).unwrap();
        let mut st = state(false);
        st.config.models_dir = Some(dir.path().to_string_lossy().into_owned());
        st.models = crate::api::ModelRegistry::default();
        st.models.set(
            "b.gguf",
            crate::api::ModelEntry {
                inference_tx: Some(std::sync::mpsc::sync_channel(1).0),
                ..Default::default()
            },
        );
        st.manager = Some(std::sync::Arc::new(crate::api::manager::ModelManager::new(
            st.models.clone(),
            std::sync::Arc::new(NoLoad),
            std::sync::Arc::new(|| u64::MAX),
            std::time::Duration::from_secs(1),
        )));
        let (_, body) = call(st.clone(), Method::GET, "/v1/models").await;
        let rows: Vec<(String, bool)> = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                (
                    m["id"].as_str().unwrap().to_string(),
                    m["lightbulb_loaded"].as_bool().unwrap(),
                )
            })
            .collect();
        // Not the hidden file, the text file or the directory; sorted; loaded flag per model.
        assert_eq!(
            rows,
            vec![("a.GGUF".to_string(), false), ("b.gguf".to_string(), true)],
            "{body}"
        );
        // An unloaded model can be retrieved (a request would load it).
        let (status, body) = call(st, Method::GET, "/v1/models/a.GGUF").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["lightbulb_loaded"], false);
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
    async fn reports_the_weights_size_on_disk_and_omits_it_when_unreadable() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("tiny.gguf"), vec![0u8; 123]).unwrap();
        let mut with_file = state(true);
        with_file.config.models_dir = Some(dir.path().to_string_lossy().into_owned());
        let (_, body) = call(with_file, Method::GET, "/v1/models").await;
        assert_eq!(
            body["data"][0]["lightbulb_memory"]["weights_bytes_on_disk"], 123,
            "{body}"
        );

        // Control: same server, no file at the path -> the field is absent,
        // not 0 and not an error.
        let empty = tempfile::tempdir().unwrap();
        let mut without = state(true);
        without.config.models_dir = Some(empty.path().to_string_lossy().into_owned());
        let (status, body) = call(without, Method::GET, "/v1/models").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body["data"][0].get("lightbulb_memory").is_none(), "{body}");
    }

    /// A run must be able to PROVE its device (a GPU run that silently served on the
    /// CPU was reported as a GPU result on 2026-11-05). `/v1/models` carries the device
    /// the model load selected, and omits it when nothing has been selected.
    #[tokio::test]
    async fn reports_the_device_the_load_selected() {
        let _ = crate::model_fuel::device::select();
        let want = crate::model_fuel::device::chosen().expect("select() records its choice");
        let (_, body) = call(state(true), Method::GET, "/v1/models").await;
        assert_eq!(body["data"][0]["lightbulb_device"], want, "{body}");
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
