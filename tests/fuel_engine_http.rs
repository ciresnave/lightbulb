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

/// The acceptance gate for the path a DEFAULT client actually takes.
///
/// The gate above pins `"temperature": 0.0` so it can assert exact content
/// ("paris"). But `ChatCompletionRequest::default_temperature`
/// (`src/api/openai/chat.rs:538`) fills in **1.0** — not 0.0 — whenever a
/// request omits `temperature` entirely, and the Fuel path honours it:
/// `run_inference_once` -> `FuelEngineModel::step_one` -> `select_token`
/// (`src/model_fuel/engine_model.rs`) runs real multinomial sampling at
/// temperature 1.0, not argmax. A plain `curl` with no `temperature` field —
/// the common case for a default OpenAI-compatible client — goes through
/// that stochastic path, and the greedy gate above never exercises it.
///
/// This test sends exactly that: no `temperature` key in the JSON body at
/// all, so the server's own default applies.
///
/// ## Why the assertions differ from the greedy gate's
///
/// Sampling means the SAME prompt can legally produce different token
/// sequences on different runs. Asserting specific content (e.g. "paris")
/// here would be flaky by construction, and a flaky gate is worse than no
/// gate — it trains people to re-run failures instead of reading them. So
/// this test can only assert PROPERTIES that hold regardless of which
/// tokens were drawn: shape (valid JSON, the expected fields present),
/// provenance (the completion came from the model, not a fallback path),
/// and termination (the model chose to stop, rather than `max_tokens`
/// cutting it off mid-thought). It cannot and does not assert the
/// completion is the RIGHT text.
///
/// Both tests are needed together, not as alternatives:
/// - `fuel_runner_serves_a_coherent_completion_over_http` (above) proves the
///   model produces the RIGHT text, on the one path (`temperature: 0.0`)
///   where "right" is a well-defined, non-flaky assertion.
/// - This test proves the DEFAULT path — the one every client that doesn't
///   set `temperature` actually takes — reaches the model at all, using the
///   real HTTP round trip and the real sampling code, not the greedy
///   shortcut.
///
/// Neither subsumes the other: a broken `select_token` for `temperature !=
/// 0.0` (e.g. a softmax/RNG bug that only manifests off the greedy branch)
/// would pass the first test and fail this one; a broken router or a dead
/// `inference_tx` would fail both.
#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint; minutes on CPU"]
async fn fuel_runner_serves_a_default_temperature_completion() {
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

    // `temperature` is DELIBERATELY ABSENT — that omission is the entire
    // point of this test. `serde(default = "default_temperature")` fills in
    // 1.0 server-side, exercising the exact request shape a default client
    // sends.
    let max_tokens = 24;
    let body = serde_json::json!({
        "model": "tinyllama",
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": max_tokens,
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

    eprintln!("default-temperature completion: {text:?}");

    assert!(!text.trim().is_empty(), "the model returned no text");

    // LOAD-BEARING. Without this, the test passes even when `inference_tx`
    // routing is dead and `create_chat_completion` fell through to the
    // no-model placeholder (`src/api/openai/chat.rs:232-241`) instead of
    // reaching the runner at all — exactly the "gate reports success having
    // verified nothing" failure mode this whole file exists to exclude.
    assert!(
        !text.contains("No model available on the server"),
        "got the no-model fallback text instead of a real completion: \
         {text:?} — inference_tx routing is broken, the model never ran"
    );

    // Terminated because the model chose to stop, not because `max_tokens`
    // cut it off. `finish_reason` and `usage.completion_tokens` can't answer
    // this: `create_chat_completion` hardcodes `finish_reason: "stop"`
    // (chat.rs:221) on every response including the ones that DO exhaust the
    // budget, and hardcodes `completion_tokens: 0` (chat.rs:225) always — so
    // neither field carries real information here.
    //
    // What's left is the decoded text itself. `run_jobs`
    // (`src/engine/model_runner.rs:394`) calls `decode_text(&generated_tokens,
    // false)` — `skip_special: false` — and `FuelEngineModel::step_one`
    // (`src/model_fuel/engine_model.rs:245,266`) pushes the sampled token
    // into `generated_tokens` and completes the request the instant
    // `self.loaded.is_eos(tok)` is true. TinyLlama's checkpoint declares
    // `eos_token_id: 2` in `config.json`, and its `tokenizer.json` maps id 2
    // to the added special token whose literal `content` is `"</s>"`.
    // HF's `tokenizers::Tokenizer::decode` emits that content verbatim when
    // `skip_special_tokens` is false (verified directly against this
    // checkpoint's tokenizer.json: `decode([..., 2], false)` produces
    // `"...</s>"` with no separator, vs. `"..."` when `skip_special_tokens`
    // is true). So `"</s>"` appearing in `text` is direct, checkpoint-
    // specific evidence that `is_eos` fired and ended generation — not that
    // `max_tokens` ran out the clock.
    assert!(
        text.contains("</s>"),
        "expected the EOS marker \"</s>\" in the decoded text (the runner \
         decodes with skip_special=false), got {text:?} — generation used \
         all {max_tokens} max_tokens without the model ever choosing to \
         stop, i.e. max_tokens acted as a hard cap rather than genuine \
         termination. Either the sampling distribution never favoured EOS \
         within budget (try raising max_tokens) or generation is not \
         terminating correctly."
    );
}
