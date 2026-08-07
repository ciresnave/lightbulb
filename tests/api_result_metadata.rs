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

/// Drive the real router with a real runner and collect the whole response
/// body. `to_bytes` awaits end-of-stream, so this works for an SSE response
/// too: the stream ends when the runner drops `stream_tx` at the end of its
/// `Streaming` arm.
async fn post_raw(path: &str, body: serde_json::Value) -> (StatusCode, Vec<u8>) {
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
    (status, bytes.to_vec())
}

async fn post_json(path: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let (status, bytes) = post_raw(path, body).await;
    (status, serde_json::from_slice(&bytes).expect("response was not JSON"))
}

/// Parse an SSE body into its `data:` payloads, decoded as JSON.
///
/// `Sse::new` at `chat.rs:156` is built without `keep_alive`, so no comment
/// frames interleave and every non-blank line is a `data:` line.
fn sse_chunks(body: &[u8]) -> Vec<serde_json::Value> {
    std::str::from_utf8(body)
        .expect("SSE body was not UTF-8")
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data payload was not JSON"))
        .collect()
}

/// A generation cut off by `max_tokens` must say so. Reporting "stop" here
/// tells a client the model finished when it was truncated, so the client does
/// not continue. This FAILS before the fix.
///
/// ## Prompt choice is load-bearing here
///
/// The original prompt, "Write a long essay about the sea.", does NOT
/// exercise this at all: at `temperature: 0.0` it makes TinyLlama emit EOS
/// as its very first generated token (`completion_tokens: 1`, content
/// `"</s>"`), so the 6-token cap is never reached and `finish_reason:
/// "stop"` is the CORRECT answer for that generation — the test failed for
/// the wrong reason (an unrepresentative prompt), not because the fix was
/// broken. Confirmed independently on `usage_counts_real_tokens`'s prompt
/// below, which reliably runs to the cap (`completion_tokens: 6`,
/// non-EOS-looking text `"\n\n1. Answer:"`) across two separate runs and,
/// post-fix, correctly reports `"length"` there. This test reuses that
/// same proven-to-truncate prompt rather than gambling on an untested one.
#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint"]
async fn truncated_generation_reports_length() {
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
    eprintln!("truncated_generation_reports_length JSON: {v}");
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
    eprintln!("usage_counts_real_tokens JSON: {v}");

    let completion = v["usage"]["completion_tokens"].as_u64().expect("no completion_tokens");
    let prompt = v["usage"]["prompt_tokens"].as_u64().expect("no prompt_tokens");
    let total = v["usage"]["total_tokens"].as_u64().expect("no total_tokens");

    // Exact, not `> 0`: the request caps generation at 6, and a truncated
    // generation produces exactly that. `> 0` would pass on 1.
    assert_eq!(completion, 6, "completion_tokens should equal the tokens generated");
    assert!(prompt > 0, "prompt_tokens is zero — the prompt was never counted");
    assert_eq!(total, prompt + completion, "total_tokens is not the sum of its parts");
}

/// The streaming terminal chunk must carry the reason the runner observed.
///
/// ## Why this test exists at all
///
/// The plan for this task mandated shipping NO test here, on the stated
/// grounds that "asserting on SSE needs a streaming HTTP client this repo does
/// not set up". That premise was false, and a reviewer disproved it by
/// building a probe from the harness already in this file: `oneshot` +
/// `to_bytes` collects a complete SSE body, because `to_bytes` awaits
/// end-of-stream and the runner ends the stream by dropping `stream_tx`. This
/// test is the same cost and class as `truncated_generation_reports_length`
/// above — which the non-streaming half of this same fix already shipped.
///
/// ## Prompt choice is load-bearing, same as above
///
/// Reuses the prompt the two tests above proved runs to a 6-token cap rather
/// than emitting EOS early. `create_chat_stream` applies the identical
/// `"{role}: {content}"` join as the non-streaming path, so the model sees the
/// same text and truncates the same way.
#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint"]
async fn streaming_terminal_chunk_reports_length() {
    let (status, body) = post_raw(
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tinyllama",
            "messages": [{"role": "user", "content": "Name the capital of France."}],
            "max_tokens": 6,
            "temperature": 0.0,
            "stream": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let chunks = sse_chunks(&body);
    eprintln!("streaming_terminal_chunk_reports_length chunks: {chunks:#?}");
    assert!(!chunks.is_empty(), "the stream produced no data frames");

    // Exactly one chunk carries a finish_reason, and it is the last one.
    // Asserting on the count as well as the value catches a regression that
    // emits a terminal chunk per token, which a "last chunk" check alone
    // would pass.
    let terminal: Vec<_> = chunks
        .iter()
        .filter(|c| !c["choices"][0]["finish_reason"].is_null())
        .collect();
    assert_eq!(
        terminal.len(),
        1,
        "expected exactly one chunk with a finish_reason, got {}",
        terminal.len()
    );
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "length",
        "the terminal chunk did not report the observed reason"
    );
}
