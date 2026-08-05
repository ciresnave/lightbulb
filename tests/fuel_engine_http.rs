//! The acceptance gate for the Fuel-backed runner.
//!
//! Asserts the completion is COHERENT ENGLISH CONTINUING THE PROMPT, not merely
//! that a 200 came back. A model wired up wrongly — transposed projection,
//! mis-set RoPE base, wrong norm placement — still returns tokens; it just
//! returns nonsense. Status code, token count, and absence of panic all pass on
//! garbage. Only reading the text catches it.
//!
//! Goes through the REAL router and the REAL handler via `ServiceExt::oneshot`,
//! so routing, JSON deserialization, the channel and the runner are all
//! exercised. Only the TCP socket is skipped, and the socket is not what is at
//! risk.
//!
//! ## Deviation from the brief: `/v1/chat/completions`, not `/v1/completions`
//!
//! The brief that generated this file targeted `/v1/completions`. As written,
//! `src/api/openai/completions.rs::create_completion` takes `_state: AppState`
//! (leading underscore — never reads it) and is an explicit
//! `// TODO: Integrate with actual inference engine / For now, return a mock
//! response` stub: it always returns the literal string `"This is a
//! placeholder completion. Inference engine integration pending."`, no matter
//! what `inference_tx` is. That predates this whole engine-wiring plan.
//!
//! Pointing an acceptance gate at that handler would make the content
//! assertion fail on every run, including a perfectly-wired Fuel engine — a
//! gate that is always red proves nothing, and is arguably worse than the
//! "gate reports success having verified nothing" failure mode this project
//! exists to stop, because there the correct-implementation case at least
//! COULD pass.
//!
//! `src/api/openai/chat.rs::create_chat_completion` does read
//! `state.inference_tx` and drives exactly the `InferenceJob` /
//! `ResponseMode::Complete` / `ModelRunner` round trip this gate is meant to
//! exercise — it is the only OpenAI-compatible HTTP handler in this codebase
//! actually wired to the model runner (Fuel or candlelight). So it is the
//! correct target. This task's own instructions license exactly this kind of
//! substitution ("if the test does not match the real API, fix THE TEST" /
//! "do not modify `src/api/` to suit the test") — applied here to a routing
//! assumption instead of a struct signature.
//!
//! One consequence: `chat.rs` builds its prompt as `"{role}: {content}"` per
//! message, so the model actually receives `"user: The capital of France
//! is"`, not the bare prompt string. That prefixing is `chat.rs`'s existing
//! behaviour, not something introduced here, which is why the diagnosis
//! order below cross-checks against `generate_greedy` on the UNPREFIXED
//! prompt: if that passes and this fails, the Fuel wiring is exonerated and
//! the defect is `chat.rs`'s prompt formatting, not the engine.
//!
//! Diagnose in this order, because it separates plumbing from maths:
//!   1. Non-200 -> the handler or the channel. Plumbing.
//!   2. 200 with empty text -> state transitions in `FuelEngineModel::step_one`.
//!   3. 200, fluent but wrong content -> model wiring. Cross-check with
//!      `cargo test --release --lib model_fuel::generate -- --ignored --nocapture`
//!      (`generate_greedy` on the bare prompt). If THAT also produces
//!      nonsense, the defect predates this plan.
//!   4. 200, right content, never stops -> EOS.
//!
//! Run: `cargo test --release --features fuel-engine --test fuel_engine_http -- --ignored --nocapture`
#![cfg(feature = "fuel-engine")]

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use lightbulb::api::{ApiConfig, AppState};
use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler, ModelRunner};

fn tinyllama_dir() -> Option<PathBuf> {
    let p = PathBuf::from(
        "C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    );
    p.join("model.safetensors").is_file().then_some(p)
}

#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint; minutes on CPU"]
async fn fuel_runner_serves_a_coherent_completion_over_http() {
    let dir = tinyllama_dir()
        .expect("no TinyLlama snapshot — this is an acceptance gate, so it fails rather than skipping");

    let tx = ModelRunner::start(&dir, 1, 512, Some("f32".to_string()))
        .expect("starting the Fuel model runner");

    let state = AppState {
        scheduler: Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default())),
        config: ApiConfig::default(),
        db_pool: None,
        inference_tx: Some(tx),
    };

    let app = lightbulb::api::openai::routes().with_state(state);

    // temperature 0.0 PINNED, not defaulted: this gate asserts content, so a
    // future default that makes sampling stochastic would make it flaky.
    let body = serde_json::json!({
        "model": "tinyllama",
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": 8,
        "temperature": 0.0,
    });

    let resp = app
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

    assert_eq!(resp.status(), StatusCode::OK, "completion request failed");

    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("response was not JSON");
    let text = v["choices"][0]["message"]["content"]
        .as_str()
        .expect("no choices[0].message.content in the response")
        .to_string();

    eprintln!("completion: {text:?}");

    assert!(!text.trim().is_empty(), "the model returned no text");
    assert!(
        text.to_lowercase().contains("paris"),
        "expected the continuation to name Paris, got {text:?} — the server \
         responds but the model is producing nonsense, which points at the \
         wiring (projection layout, RoPE base, norm placement) rather than the \
         plumbing"
    );
}
