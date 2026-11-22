// SPDX-License-Identifier: MIT OR Apache-2.0
//! The models-directory path end to end on SYNTHETIC GGUFs (ROADMAP C1): `/v1/models` lists what the directory
//! holds, a request naming a model loads it on demand through the real `FuelLoader`, a second request naming
//! another model loads that one too, and an unknown name is refused. The two files differ only in the token their
//! model always emits, so which model answered is readable from the reply.
//!
//! Real loader, real runner, real handlers; no fake `ModelLoader`. Not a trained model, and the device is the CPU,
//! so eviction under memory pressure is not exercised here (the manager's own tests cover it with a fake).

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use lightbulb::api::fuel_loader::{FuelLoader, StartParams};
use lightbulb::api::manager::{FreeBytes, ModelManager};
use lightbulb::api::{ApiConfig, AppState};
use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler};

#[path = "support/gguf.rs"]
#[allow(dead_code)]
mod gguf_writer;

/// A models directory holding `a.gguf` (always emits `&`, token 7) and `b.gguf` (always emits `'`, token 8).
fn models_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    for (name, token) in [("a.gguf", 7), ("b.gguf", 8)] {
        std::fs::write(
            dir.path().join(name),
            gguf_writer::constant_output_gguf_bytes(token),
        )
        .expect("writing a gguf");
    }
    dir
}

fn state_over(dir: &Path) -> AppState {
    let scheduler = Arc::new(std::sync::Mutex::new(MemoryAwareScheduler::new(
        MemoryAwareConfig::default(),
    )));
    let mut state = AppState {
        scheduler,
        config: ApiConfig {
            models_dir: Some(dir.to_string_lossy().into_owned()),
            default_model: "a.gguf".to_string(),
            ..ApiConfig::default()
        },
        pg_backend: None,
        manager: None,
        models: lightbulb::api::ModelRegistry::default(),
        eos_monitor: Default::default(),
        security_sinks: Vec::new(),
        cache_stats: Arc::new(std::sync::Mutex::new(
            lightbulb::cache::PrefixCacheStats::default(),
        )),
        live_system: lightbulb::api::new_shared_system_metrics(),
        queue_depth: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    };
    let loader = FuelLoader::new(dir, StartParams::from_state(&state));
    let plenty: FreeBytes = Arc::new(|| u64::MAX);
    state.manager = Some(Arc::new(ModelManager::new(
        state.models.clone(),
        Arc::new(loader),
        plenty,
        Duration::from_secs(30),
    )));
    state
}

async fn call(
    state: &AppState,
    method: &str,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let app = lightbulb::api::openai::routes().with_state(state.clone());
    let resp = app
        .oneshot(
            Request::builder()
                .method(method)
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .expect("router returned no response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or_default())
}

fn chat(model: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "ab cd"}],
        "max_tokens": 2,
        "temperature": 0.0,
    })
}

/// The directory's models are listed (both servable, neither loaded yet), a request loads the one it names and the
/// reply is that model's (`&&` for a, `''` for b), a later request for the other loads it too, and the listing
/// then shows each as loaded. An unknown name is refused (the control: the directory is not served blindly).
#[tokio::test]
async fn models_are_listed_loaded_on_demand_and_switched_between() {
    let dir = models_dir();
    let state = state_over(dir.path());

    let (status, list) = call(&state, "GET", "/v1/models", serde_json::json!({})).await;
    assert_eq!(status, StatusCode::OK, "{list}");
    let ids: Vec<&str> = list["data"]
        .as_array()
        .expect("data")
        .iter()
        .map(|m| m["id"].as_str().expect("id"))
        .collect();
    assert_eq!(ids, ["a.gguf", "b.gguf"], "{list}");
    assert!(list["data"][0]["lightbulb_loaded"] == false, "{list}");

    let (status, a) = call(&state, "POST", "/v1/chat/completions", chat("a.gguf")).await;
    assert_eq!(status, StatusCode::OK, "{a}");
    assert_eq!(a["choices"][0]["message"]["content"], "&&", "{a}");

    let (status, b) = call(&state, "POST", "/v1/chat/completions", chat("b.gguf")).await;
    assert_eq!(status, StatusCode::OK, "{b}");
    assert_eq!(b["choices"][0]["message"]["content"], "''", "{b}");

    let (_, list) = call(&state, "GET", "/v1/models", serde_json::json!({})).await;
    assert!(
        list["data"]
            .as_array()
            .unwrap()
            .iter()
            .all(|m| m["lightbulb_loaded"] == true),
        "both models are loaded after use: {list}"
    );

    let (status, nope) = call(&state, "POST", "/v1/chat/completions", chat("nope.gguf")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{nope}");
}
