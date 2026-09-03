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
//! The brief that generated this file targeted `/v1/completions`. At the time,
//! `src/api/openai/completions.rs::create_completion` took `_state: AppState`
//! (leading underscore — never read it) and was an explicit
//! `// TODO: Integrate with actual inference engine / For now, return a mock
//! response` stub: it always returned the literal string `"This is a
//! placeholder completion. Inference engine integration pending."`, no matter
//! what `inference_tx` was.
//!
//! Pointing an acceptance gate at that handler would have made the content
//! assertion fail on every run, including a perfectly-wired Fuel engine — a
//! gate that is always red proves nothing, and is arguably worse than the
//! "gate reports success having verified nothing" failure mode this project
//! exists to stop, because there the correct-implementation case at least
//! COULD pass.
//!
//! **That is no longer true, as of `d65bafd` (the runner-result-metadata
//! branch): `/v1/completions` now dispatches through the runner and is covered
//! by `tests/api_result_metadata.rs`.** The paragraphs above are kept as the
//! record of why this gate was aimed where it was, not as a live description
//! of `completions.rs`. The target stays `/v1/chat/completions` because that
//! is what this gate has always measured and moving it would silently change
//! what a green run means — not because the other endpoint is still a stub.
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
//! One consequence: `chat.rs` does not send the bare prompt string. It used to
//! build `"{role}: {content}"` per message, so the model received `"user: The
//! capital of France is"`; as of the chat-template work it renders the
//! checkpoint's own template instead, so TinyLlama receives
//! `"<|user|>\nThe capital of France is</s>\n<|assistant|>\n"` — and that text
//! is tokenized with `add_special_tokens = false`, because a templated prompt
//! carries whatever special tokens its template emits (see `chat.rs`'s
//! `build_prompt`). TinyLlama resolves to `registry::ZEPHYR`, which emits no
//! `bos_token`, so its prompt now carries no BOS at all — which is exactly what
//! `transformers` produces for this checkpoint, since `apply_chat_template`
//! tokenizes with `add_special_tokens=False`. Either way the text is not the
//! bare prompt, which is why the diagnosis order below cross-checks against
//! `generate_greedy` on the UNPREFIXED prompt: if that passes and this fails,
//! the Fuel wiring is exonerated and the defect is `chat.rs`'s prompt
//! construction, not the engine.
//!
//! The state built below resolves the template exactly as `ApiServer::new`
//! does, so this gate measures the prompt path the server actually ships.
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
//!
//! # ⚠️ THIS FILE IS NEVER RUN BY CI, BY DECISION
//!
//! CireSnave ruled on 2026-08-27 that Lightbulb gets no GPU CI runner — a cost
//! constraint, not a judgement about these tests. `.github/workflows/ci.yml`
//! COMPILES this file (via `--features fuel-engine --no-run`, which needs no
//! GPU) so it cannot rot into non-compiling state, but it never executes a
//! single test here.
//!
//! **So the claim this file makes is verified by a person running it on demand,
//! and by nothing else.** "Nobody ran it this month" and "it passes" are
//! indistinguishable from the repository, which is why `scripts/check.sh`
//! prints this file in its NOT RUN block on every invocation including
//! successful ones.
//!
//! If you are about to trust a green board on a change that touches prompting,
//! templating or decoding: this is the test that would have caught it, and it
//! did not run.
#![cfg(feature = "fuel-engine")]

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use lightbulb::api::{ApiConfig, AppState};
use lightbulb::engine::{MemoryAwareConfig, MemoryAwareScheduler, ModelRunner};

/// Locate the TinyLlama checkpoint.
///
/// `TINYLLAMA_DIR` overrides the built-in default, which is one developer's
/// HuggingFace cache including a specific snapshot hash. Without the override
/// this gate can only run on that machine — it fails loudly rather than
/// false-passing, but "fails everywhere else" is not a useful acceptance gate.
fn tinyllama_dir() -> Option<PathBuf> {
    // one locator, not twelve copies of an absolute path into one home directory
    lightbulb::test_models::tinyllama_dir()
}

/// Resolve the checkpoint's chat template exactly as `ApiServer::new` does.
///
/// An acceptance gate that builds a state the server never builds is not an
/// acceptance gate for the server. Leaving this `None` would keep these two
/// tests on the pre-template `"{role}: {content}"` join for ever, so they would
/// go on passing while the shipped prompt path was never exercised.
///
/// The `Resolution::None` check lives in `chat_template::resolve_for_serving`,
/// which is what `ApiServer::new` assigns from, so this cannot drift from
/// startup by being edited in one place and not the other.
fn chat_template_for(
    dir: &std::path::Path,
) -> Option<Arc<lightbulb::api::chat_template::ResolvedTemplate>> {
    lightbulb::api::chat_template::resolve_for_serving(dir)
}

#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint; minutes on CPU"]
async fn fuel_runner_serves_a_coherent_completion_over_http() {
    let dir = tinyllama_dir().expect(
        "no TinyLlama snapshot — this is an acceptance gate, so it fails rather than skipping",
    );

    let tx = ModelRunner::start(&dir, 1, 512, Some("f32".to_string()))
        .expect("starting the Fuel model runner");

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

    // temperature 0.0 PINNED, not defaulted: this gate asserts content, so a
    // future default that makes sampling stochastic would make it flaky.
    //
    // `max_tokens` is 24, not 8, and the headroom is the point. At 8 the
    // observed completion was exactly 8 tokens with `"Paris"` at token 6 — the
    // gate consumed 100% of its budget, so ONE extra leading token (a chat
    // template gaining a system turn, a BOS moving, a checkpoint swap) pushed
    // `"Paris"` past the cap and turned the gate red with the message below:
    // *"the model is producing nonsense, which points at the wiring (projection
    // layout, RoPE base, norm placement)"*. That message would have sent a
    // reader to the Fuel decode path for a prompt-length change. A gate whose
    // failure text names the wrong subsystem is worse than a slow gate; 24 is
    // three times the observed answer length and still bounded. It costs three
    // times as many decode steps, which is the price and is accepted — the
    // default-temperature gate below already runs 24.
    //
    // In practice it costs nothing: observed 2026-08-08 at 24, the model
    // answers `"The capital of France is Paris.</s>"` and stops on EOS after 8
    // tokens, so the extra 16 are never decoded. The headroom is only spent on
    // the run where something HAS shifted — which is the run where it matters.
    let body = serde_json::json!({
        "model": "tinyllama",
        "messages": [{"role": "user", "content": "The capital of France is"}],
        "max_tokens": 24,
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
/// tokens were drawn: shape (valid JSON, the expected fields present) and
/// provenance (the completion came from the model, not a fallback path).
/// It cannot and does not assert the completion is the RIGHT text.
///
/// It also does not assert TERMINATION (that generation stopped via EOS
/// rather than exhausting `max_tokens`), because termination at temperature
/// 1.0 is itself a property of the draw: re-measured under the chat template
/// on 2026-08-08, EOS fired in 2 of 6 trials. See the comment at the bottom of
/// the test body, where that assertion used to live, for the trial data and
/// for why the deterministic version of this claim lives in
/// `tests/chat_template_e2e.rs` instead.
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
/// Neither subsumes the other: this test proves the DEFAULT (no-`temperature`
/// -field) request path reaches the model and returns real decoded output —
/// not the no-model fallback — using the real HTTP round trip and the real
/// sampling code. It does NOT validate sampling quality or correctness: with
/// the termination assertion removed (see below), a broken `select_token` for
/// `temperature != 0.0` (e.g. a softmax/RNG bug off the greedy branch) would
/// still decode to non-empty, non-fallback text and PASS this test. Only a
/// broken router or a dead `inference_tx` — something that stops the request
/// from reaching the model at all — would fail it.
#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint; minutes on CPU"]
async fn fuel_runner_serves_a_default_temperature_completion() {
    let dir = tinyllama_dir().expect(
        "no TinyLlama snapshot — this is an acceptance gate, so it fails rather than skipping",
    );

    let tx = ModelRunner::start(&dir, 1, 512, Some("f32".to_string()))
        .expect("starting the Fuel model runner");

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

    // ── Termination: still not asserted, for a NEW reason. Re-decided
    //    2026-08-08 on this branch, with fresh measurement. ──
    //
    // This test deliberately does NOT assert termination (that generation
    // stopped via EOS rather than running out `max_tokens`), even though an
    // earlier version of it did, via checking for the literal `"</s>"` EOS
    // marker that `run_jobs` leaves in the decoded text (it calls
    // `decode_text(&generated_tokens, false)` — `skip_special: false`).
    //
    // ## The rationale that used to be here is false and has been deleted
    //
    // It read: EOS fires in only ~1 of 6 trials because the model is prompted
    // with `chat.rs`'s ad-hoc `"user: <content>"` prefix instead of its native
    // template; that is a defect in `chat.rs`'s prompt construction; `src/api/`
    // is out of scope for this branch (verified zero-diff) and queued for a
    // later one; a termination gate belongs on the branch that fixes the chat
    // template, once `chat.rs` is back in scope.
    //
    // **This IS that branch.** `chat.rs` renders the checkpoint's own template,
    // `chat_template_for` above resolves it exactly as `ApiServer::new` does,
    // and the model receives `"<|user|>\nThe capital of France is</s>\n
    // <|assistant|>\n"`. Every clause about scope and sequencing in that
    // paragraph is now wrong, so leaving it would have told the next reader to
    // wait for a branch that has already landed.
    //
    // ## Measured again, after the template, and the answer is still no
    //
    // 2026-08-08, `--release`, `--features fuel-engine`, this exact request
    // (no `temperature` field, so the server's 1.0 default applies;
    // `max_tokens: 24`), six trials against one loaded runner:
    //
    //   stop,   8 tokens   "The capital of France is Paris.</s>"        x2
    //   length, 24 tokens  "…Paris, located in the Île-de-France region…"
    //   length, 24 tokens  "Yes, absolute facts do exist! The capital…"
    //   length, 24 tokens  "Yes, the capital city of France is Paris, …"
    //   length, 24 tokens  "Yes, the capital of France is Paris. Paris…"
    //
    // EOS fired in **2 of 6**. The template raised it from 1 of 6 and made
    // every completion on-topic — a real improvement, visible above — but it
    // did not make termination deterministic, and it was never going to: at
    // temperature 1.0 the next token is DRAWN, so whether the draw lands on
    // EOS is a property of the sample, not of the wiring. Restoring the
    // assertion would give this gate a ~2-in-3 failure rate.
    //
    // ## And the claim already has a home where it IS deterministic
    //
    // `tests/chat_template_e2e.rs::templated_chat_stops_on_eos` asserts exactly
    // "a templated chat model stops on EOS", at `temperature: 0.0`, where the
    // answer is argmax and reproducible — observed `finish_reason: "stop"`,
    // `completion_tokens: 8`. So the coverage the old comment was deferring
    // exists; it just belongs on the greedy path rather than the sampled one.
    //
    // What stays true from the original reasoning: this test's contract is to
    // assert only PROPERTIES THAT HOLD REGARDLESS OF WHICH TOKENS WERE DRAWN.
    // Termination at temperature 1.0 is not one of those, so asserting it here
    // would be a category error even if the observed rate were 6 of 6 — six
    // successes do not make a sampled outcome invariant.
}
