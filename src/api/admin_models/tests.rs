// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::api::admin;
use crate::api::auth_middleware::ApiKeyInfo;
use crate::api::manager::{FreeBytes, LoadError, ModelLoader, ModelManager};
use crate::api::registry::{ModelEntry, ModelRegistry};
use crate::api::tests::bare_state;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

struct Fake;

impl ModelLoader for Fake {
    fn needed_bytes(&self, _: &str) -> Result<u64, LoadError> {
        Ok(1)
    }
    fn load(&self, registry: &ModelRegistry, name: &str) -> Result<(), LoadError> {
        let (tx, rx) =
            std::sync::mpsc::sync_channel::<crate::engine::model_runner::InferenceJob>(1);
        let join = std::thread::spawn(move || while rx.recv().is_ok() {});
        registry.set(
            name,
            ModelEntry {
                inference_tx: Some(tx),
                join: Some(Arc::new(std::sync::Mutex::new(Some(join)))),
                ..ModelEntry::default()
            },
        );
        Ok(())
    }
}

fn managed() -> crate::api::AppState {
    let mut state = bare_state(Some("."), false);
    state.models = ModelRegistry::default();
    let free: FreeBytes = Arc::new(|| u64::MAX);
    state.manager = Some(Arc::new(ModelManager::new(
        state.models.clone(),
        Arc::new(Fake),
        free,
        Duration::from_secs(5),
    )));
    state
}

async fn post(
    state: crate::api::AppState,
    role: &str,
    name: &str,
) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method("POST")
        .uri(format!("/v1/lightbulb/admin/models/{name}/unload"))
        .extension(ApiKeyInfo {
            api_key_id: uuid::Uuid::new_v4(),
            role: role.to_string(),
        })
        .body(Body::empty())
        .unwrap();
    let resp = admin::routes()
        .with_state(state)
        .oneshot(req)
        .await
        .unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or_default())
}

#[tokio::test]
async fn an_admin_unloads_an_idle_model_and_it_is_gone_from_the_registry() {
    let state = managed();
    let lease = state.manager.as_ref().unwrap().acquire("m.gguf").unwrap();
    drop(lease);
    let (status, body) = post(state.clone(), "admin", "m.gguf").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["unloaded"], "m.gguf");
    assert_eq!(body["runner_joined"], true);
    assert!(state.models.names().is_empty());
}

#[tokio::test]
async fn unknown_busy_and_unmanaged_are_404_409_501() {
    let state = managed();
    assert_eq!(
        post(state.clone(), "admin", "nope").await.0,
        StatusCode::NOT_FOUND
    );

    let lease = state.manager.as_ref().unwrap().acquire("m.gguf").unwrap();
    assert_eq!(
        post(state.clone(), "admin", "m.gguf").await.0,
        StatusCode::CONFLICT
    );
    assert_eq!(state.models.names(), vec!["m.gguf".to_string()]);
    drop(lease);

    let unmanaged = bare_state(None, true);
    assert_eq!(
        post(unmanaged, "admin", "x").await.0,
        StatusCode::NOT_IMPLEMENTED
    );
}

#[tokio::test]
async fn a_non_admin_key_cannot_unload() {
    let state = managed();
    drop(state.manager.as_ref().unwrap().acquire("m.gguf").unwrap());
    let (status, _) = post(state.clone(), "user", "m.gguf").await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(
        state.models.names(),
        vec!["m.gguf".to_string()],
        "still loaded"
    );
}
