//! Behavioural: a chat model prompted with its own template answers and stops.
//!
//! Run: `cargo test --release --features fuel-engine --test chat_template_e2e -- --ignored --nocapture --test-threads=1`
//!
//! `--test-threads=1` is load-bearing, not tidiness: every test here starts its
//! own runner and so loads its own ~2.2 GB copy of the checkpoint. Run in
//! parallel they exhaust host memory and the process dies before any assertion
//! is reached.
#![cfg(feature = "fuel-engine")]

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use lightbulb::api::{ApiConfig, AppState};
use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler, ModelRunner};

// The harness below is the same shape as `tests/api_result_metadata.rs`'s.
// It is copied rather than shared: sharing across integration-test binaries
// needs a `tests/common/` module, which is a larger change than this task
// warrants.

/// Locate the checkpoint, preferring `TINYLLAMA_DIR`.
fn tinyllama_dir() -> Option<PathBuf> {
    let p = match std::env::var_os("TINYLLAMA_DIR") {
        Some(v) => PathBuf::from(v),
        None => PathBuf::from(
            "C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        ),
    };
    p.join("model.safetensors").is_file().then_some(p)
}

const MISSING_CHECKPOINT: &str = "no TinyLlama checkpoint. Set TINYLLAMA_DIR to a \
     directory containing model.safetensors and tokenizer.json. These tests assert \
     real API behaviour against a real model, so they fail rather than skipping.";

/// Resolve the checkpoint's chat template exactly as `ApiServer::new` does.
///
/// Mirroring startup is load-bearing rather than tidy. A harness that leaves
/// this `None` builds a state the server never builds, so the model sees a
/// different prompt here than in production — and every assertion below is
/// about what the model does with the prompt.
///
/// "Exactly as" is now literal: it calls the same
/// `chat_template::resolve_for_serving` that `ApiServer::new` assigns from,
/// rather than re-implementing its `Resolution::None` check. Four copies of
/// that check existed, each documenting the drift hazard it was an instance of.
fn chat_template_for(
    dir: &std::path::Path,
) -> Option<Arc<lightbulb::api::chat_template::ResolvedTemplate>> {
    let t = lightbulb::api::chat_template::resolve_for_serving(dir);
    match &t {
        Some(t) => eprintln!(
            "chat template resolved via {:?}, bos {:?}, eos {:?}",
            t.resolved_by(),
            t.tokens.bos,
            t.tokens.eos
        ),
        None => eprintln!("no chat template resolved; the handler will use the legacy join"),
    }
    t
}

/// Drive the real router with a real runner and collect the whole response
/// body. `to_bytes` awaits end-of-stream, so this works for an SSE response
/// too: the stream ends when the runner drops `stream_tx`.
async fn post_raw(path: &str, body: serde_json::Value) -> (StatusCode, Vec<u8>) {
    let dir = tinyllama_dir().expect(MISSING_CHECKPOINT);
    let tx = ModelRunner::start(&dir, 1, 512, Some("f32".to_string()))
        .expect("starting the model runner");
    let state = AppState {
        scheduler: Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default())),
        config: ApiConfig::default(),
        db_pool: None,
        inference_tx: Some(tx),
        chat_template: chat_template_for(&dir),
        // Default window (20): these harnesses make a handful of requests, so
        // the monitor never fills it and never logs. It is here because
        // `AppState` requires it, not as anything under test.
        eos_monitor: Default::default(),
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
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, bytes.to_vec())
}

async fn post_json(path: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let (status, bytes) = post_raw(path, body).await;
    (
        status,
        serde_json::from_slice(&bytes).expect("response was not JSON"),
    )
}

/// Parse an SSE body into its `data:` payloads, decoded as JSON.
fn sse_chunks(body: &[u8]) -> Vec<serde_json::Value> {
    std::str::from_utf8(body)
        .expect("SSE body was not UTF-8")
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data payload was not JSON"))
        .collect()
}

/// The rendered prompt must not be the legacy join, on the NON-STREAMING path.
///
/// Asserted through behaviour rather than by inspecting the prompt: with the
/// template applied, TinyLlama answers and emits EOS. `finish_reason` is the
/// observable, and it is the same signal that revealed the original defect.
#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint"]
async fn templated_chat_stops_on_eos() {
    let (status, v) = post_json(
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tinyllama",
            "messages": [{"role": "user", "content": "Name the capital of France."}],
            "max_tokens": 64,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    eprintln!("templated_chat_stops_on_eos: {v}");
    assert_eq!(
        v["choices"][0]["finish_reason"], "stop",
        "a templated chat model ran its whole budget instead of stopping — \
         the template is not reaching the prompt"
    );
}

/// The streaming path uses the same template. This test exists because
/// `chat.rs` has TWO join sites and fixing only the non-streaming one leaves
/// this green-looking and wrong.
#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint"]
async fn templated_streaming_stops_on_eos() {
    let (status, body) = post_raw(
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tinyllama",
            "messages": [{"role": "user", "content": "Name the capital of France."}],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let chunks = sse_chunks(&body);
    eprintln!("templated_streaming_stops_on_eos chunks: {}", chunks.len());
    if let Some(err) = chunks.iter().find(|c| !c["error"].is_null()) {
        panic!("the stream carried an error frame: {}", err["error"]);
    }
    let last = chunks.last().expect("no data frames");
    eprintln!("templated_streaming_stops_on_eos terminal chunk: {last}");
    assert_eq!(
        last["choices"][0]["finish_reason"], "stop",
        "streaming ran its whole budget — chat.rs's streaming path still uses \
         the legacy join"
    );
}
