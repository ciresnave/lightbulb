//! A5 `response_format` against a real checkpoint (CPU), through the real router.
//!
//! The unit tests use a stub runner; this asks what a real model does with the
//! prompt-and-check loop. A small model may exhaust the 3 attempts and get a
//! 502 `FormatUnmet`, which is a correct outcome and a finding, so pick a
//! checkpoint that can follow instructions: the test ASSERTS a 200 with content
//! that satisfies the format, and prints each outcome. Set `LIGHTBULB_GGUF`; run with
//! `--ignored --nocapture` to read the per-case outcomes.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use lightbulb::api::chat_template;
use lightbulb::api::{ApiConfig, AppState};
use lightbulb::engine::model_runner::ModelRunner;
use lightbulb::engine::{MemoryAwareScheduler, memory_aware_scheduler::MemoryAwareConfig};
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceExt;

async fn post(app: &axum::Router, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
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

/// The router over a real runner on the checkpoint in `LIGHTBULB_GGUF`.
fn app_on_the_checkpoint() -> axum::Router {
    let path = PathBuf::from(std::env::var_os("LIGHTBULB_GGUF").expect("set LIGHTBULB_GGUF"));
    assert!(path.is_file(), "LIGHTBULB_GGUF is not a file");
    let scheduler = Arc::new(std::sync::Mutex::new(MemoryAwareScheduler::new(
        MemoryAwareConfig::default(),
    )));
    let (tx, ctrl_tx) = ModelRunner::start(
        &path,
        1,
        1024,
        Some("f32".to_string()),
        8,
        Arc::new(std::sync::Mutex::new(
            lightbulb::cache::PrefixCacheStats::default(),
        )),
        scheduler.clone(),
    )
    .expect("starting the model runner");
    let state = AppState {
        scheduler,
        // CPU generation here is about 5 s per token (Qwen3-0.6B Q8_0, release), so the
        // 300 s default expires before one JSON reply; this test measures the
        // feature, not the timeout, so give it room.
        config: ApiConfig {
            request_timeout_ms: 3_600_000,
            ..ApiConfig::default()
        },
        pg_backend: None,
        models: lightbulb::api::ModelRegistry::single(
            "test-model",
            lightbulb::api::ModelEntry {
                inference_tx: Some(tx),
                cache_control: Some(ctrl_tx),
                chat_template: chat_template::resolve_for_serving(&path),
                ..Default::default()
            },
        ),
        eos_monitor: Default::default(),
        security_sinks: Vec::new(),
        cache_stats: Arc::new(std::sync::Mutex::new(
            lightbulb::cache::PrefixCacheStats::default(),
        )),
        live_system: lightbulb::api::new_shared_system_metrics(),
        queue_depth: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    };
    lightbulb::api::openai::routes().with_state(state)
}

fn capital_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
        "additionalProperties": false
    })
}

/// A plain, tiny request, timed, so a slow result below can be told apart from a slow CPU.
async fn print_baseline(app: &axum::Router) {
    let t = std::time::Instant::now();
    let (status, v) = post(
        app,
        serde_json::json!({"model": "lightbulb-default",
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 8, "temperature": 0.0}),
    )
    .await;
    println!(
        "BASELINE plain 8 tokens: status={status} elapsed={:?} content={:?}",
        t.elapsed(),
        v["choices"][0]["message"]["content"]
    );
}

/// One `response_format` request: print the outcome, then assert it.
///
/// The checkpoint under test is chosen to be able to follow the format, so anything
/// but a 200 with conforming content is a failure, not a tolerated outcome.
async fn check_case(app: &axum::Router, label: &str, rf: serde_json::Value, prompt: &str) {
    let t = std::time::Instant::now();
    let (status, v) = post(
        app,
        serde_json::json!({
            "model": "lightbulb-default",
            "messages": [{"role": "user", "content": format!("{prompt} /no_think")}],
            "max_tokens": 256,
            "temperature": 0.0,
            "response_format": rf,
        }),
    )
    .await;
    let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
    println!(
        "CASE {label}: elapsed={:?} status={status} content={content:?} body={v}",
        t.elapsed()
    );
    assert_eq!(status, StatusCode::OK, "{label}: body={v}");
    let parsed: serde_json::Value =
        serde_json::from_str(content).expect("200 content must be bare JSON");
    assert!(parsed.is_object(), "{label}: 200 content is not an object");
    if label == "json_schema" {
        assert!(
            jsonschema::is_valid(&capital_schema(), &parsed),
            "{label}: 200 content breaks the schema"
        );
    }
}

#[tokio::test]
#[ignore = "needs a GGUF checkpoint; set LIGHTBULB_GGUF"]
async fn response_format_against_a_real_model() {
    let app = app_on_the_checkpoint();
    print_baseline(&app).await;
    check_case(
        &app,
        "json_object",
        serde_json::json!({"type": "json_object"}),
        "Give the capital of France as a JSON object with keys city and country.",
    )
    .await;
    check_case(
        &app,
        "json_schema",
        serde_json::json!({"type": "json_schema",
            "json_schema": {"name": "capital", "schema": capital_schema()}}),
        "Give the capital of France.",
    )
    .await;
}
