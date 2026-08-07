//! The API must report what the runner observed, not plausible constants.
//!
//! Goes through the real router and handler via `ServiceExt::oneshot`, so
//! routing, JSON, the channel and the runner are all exercised. Only the TCP
//! socket is skipped, and the socket is not what is at risk.
//!
//! Run: `cargo test --release --features fuel-engine --test api_result_metadata -- --ignored --nocapture`
#![cfg(feature = "fuel-engine")]

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use lightbulb::api::{ApiConfig, AppState};
use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler, ModelRunner};

fn tinyllama_dir() -> Option<PathBuf> {
    let p = match std::env::var_os("TINYLLAMA_DIR") {
        Some(v) => PathBuf::from(v),
        None => PathBuf::from(
            "C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        ),
    };
    p.join("model.safetensors").is_file().then_some(p)
}

async fn post_json(path: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let dir = tinyllama_dir()
        .expect("no TinyLlama snapshot — this asserts API behaviour, so it fails rather than skipping");
    let tx = ModelRunner::start(&dir, 1, 512, Some("f32".to_string()))
        .expect("starting the model runner");
    let state = AppState {
        scheduler: Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default())),
        config: ApiConfig::default(),
        db_pool: None,
        inference_tx: Some(tx),
    };
    let app = lightbulb::api::openai::routes().with_state(state);
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .expect("router returned no response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&bytes).expect("response was not JSON"))
}

/// A generation cut off by `max_tokens` must say so. Reporting "stop" here
/// tells a client the model finished when it was truncated, so the client does
/// not continue. This FAILS before the fix.
#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint"]
async fn truncated_generation_reports_length() {
    let (status, v) = post_json(
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tinyllama",
            "messages": [{"role": "user", "content": "Write a long essay about the sea."}],
            "max_tokens": 6,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        v["choices"][0]["finish_reason"], "length",
        "a generation stopped by max_tokens reported {:?}",
        v["choices"][0]["finish_reason"]
    );
}

/// `usage` must describe the actual work. `completion_tokens` was hardcoded 0
/// and `prompt_tokens` was a whitespace word count.
#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint"]
async fn usage_counts_real_tokens() {
    let (status, v) = post_json(
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tinyllama",
            "messages": [{"role": "user", "content": "Name the capital of France."}],
            "max_tokens": 6,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let completion = v["usage"]["completion_tokens"].as_u64().expect("no completion_tokens");
    let prompt = v["usage"]["prompt_tokens"].as_u64().expect("no prompt_tokens");
    let total = v["usage"]["total_tokens"].as_u64().expect("no total_tokens");

    // Exact, not `> 0`: the request caps generation at 6, and a truncated
    // generation produces exactly that. `> 0` would pass on 1.
    assert_eq!(completion, 6, "completion_tokens should equal the tokens generated");
    assert!(prompt > 0, "prompt_tokens is zero — the prompt was never counted");
    assert_eq!(total, prompt + completion, "total_tokens is not the sum of its parts");
}
