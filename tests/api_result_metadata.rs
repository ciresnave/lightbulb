//! The API must report what the runner observed, not plausible constants.
//!
//! Goes through the real router and handler via `ServiceExt::oneshot`, so
//! routing, JSON, the channel and the runner are all exercised. Only the TCP
//! socket is skipped, and the socket is not what is at risk.
//!
//! Run: `cargo test --release --features fuel-engine --test api_result_metadata -- --ignored --nocapture --test-threads=1`
//!
//! `--test-threads=1` is load-bearing, not tidiness: every test here starts its
//! own runner and so loads its own ~2.2 GB copy of the checkpoint. Run in
//! parallel they exhaust host memory and the process dies with
//! `STATUS_STACK_BUFFER_OVERRUN` (or `memory allocation of N bytes failed`)
//! before any assertion is reached.
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

/// Count `text` with the **model's own** tokenizer, on the same checkpoint the
/// runner loaded.
///
/// This is the whole point of the `prompt_tokens` assertions: a word count is
/// the defect (`chat.rs` used `prompt.split_whitespace()`), and a word count
/// satisfies `> 0` just as happily as a real token count does, so only an
/// equality check against the tokenizer has any teeth.
///
/// `add_special_tokens = true` is not a guess, and must never be traded for a
/// fudge factor. It mirrors the single call the backend makes during
/// prefill — `model_fuel/engine_model.rs`'s `encode(ctx.request.prompt, true)`,
/// whose `ids.len()` becomes `ctx.prompt_tokens` verbatim. For TinyLlama that
/// prepends BOS, which is why `"The capital of France is"` counts 6 (`<s> ▁The
/// ▁capital ▁of ▁France ▁is`) and not 5. If a count here is ever off by one,
/// the fix is this flag, never a `+ 1` at the call site.
fn tokenizer_count(text: &str) -> u64 {
    let dir = tinyllama_dir()
        .expect("no TinyLlama snapshot — this asserts API behaviour, so it fails rather than skipping");
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .expect("loading the checkpoint's tokenizer.json");
    tok.encode(text, true)
        .expect("tokenizing the prompt")
        .get_ids()
        .len() as u64
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
    //
    // Note this one alone does NOT close the "always report the cap" mutation
    // (`completion_tokens: batch[0].request.max_new_tokens`), because here the
    // right answer and the cap are the same number. `eos_terminated_generation_
    // reports_stop` below is what discriminates them: 1 against a cap of 32.
    assert_eq!(completion, 6, "completion_tokens should equal the tokens generated");

    // The string the model saw is NOT the message content. `create_chat_
    // completion` joins messages as `format!("{}: {}", role, content)` and
    // `join("\n")`s them, so a single user message reaches the tokenizer as the
    // literal below. Counting the bare content would compare against a prompt
    // nothing ever tokenized.
    let expected_prompt = tokenizer_count("user: Name the capital of France.");
    eprintln!("tokenizer says prompt_tokens should be {expected_prompt}");
    assert_eq!(
        prompt, expected_prompt,
        "prompt_tokens disagrees with the model's own tokenizer — \
         a word count of this prompt is 6, which is why `> 0` never caught it"
    );
    assert_eq!(total, prompt + completion, "total_tokens is not the sum of its parts");
}

/// A generation that ends on EOS must say `"stop"` — the inverse of
/// `truncated_generation_reports_length`, and the only test in the suite that
/// exercises `stopped_on_eos == true` from a real backend.
///
/// ## Why the whole suite needed this
///
/// Every other integration test here asserts `"length"`, and the unit tests in
/// `model_runner.rs` set `stopped_on_eos` directly, so they exercise the
/// direction of `finish_reason_for`'s if/else and nothing that produces the
/// flag. Making the EOS check unconditionally false at all four completion
/// sites — `model/parallel_model_manager.rs` and `model_fuel/engine_model.rs`
/// — left the entire suite green: every EOS-terminated response would have
/// reported `"length"`. This test is what turns that mutation red.
///
/// ## Prompt choice, and why `completion_tokens < 32` is half the assertion
///
/// `"Write a long essay about the sea."` at `temperature: 0.0` makes TinyLlama
/// emit EOS as its FIRST generated token (`completion_tokens: 1`, content
/// `"</s>"`) — observed while `truncated_generation_reports_length` was being
/// debugged, where that behaviour was the problem rather than the point. A cap
/// of 32 against a run that ends at 1 means `"stop"` cannot be arrived at by
/// running out of budget, and the token assertion is what proves it: `"stop"`
/// alone would also pass if the run went the full 32 and the reason were
/// hardcoded.
///
/// ## The `"</s>"` in the content is a real, pre-existing wart
///
/// Expect `choices[0].message.content` to be the literal `"</s>"`. Both
/// backends push the sampled token and count it BEFORE testing it for EOS, and
/// `run_jobs` calls `decode_text(.., false)` — `skip_special = false` — so the
/// EOS token is decoded straight into the returned text. That is a real
/// pre-existing defect and deliberately NOT fixed here; it is out of this
/// pass's scope. It is recorded rather than asserted so that fixing it later
/// does not break this test, which is about `finish_reason`.
#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint"]
async fn eos_terminated_generation_reports_stop() {
    let (status, v) = post_json(
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tinyllama",
            "messages": [{"role": "user", "content": "Write a long essay about the sea."}],
            "max_tokens": 32,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    eprintln!("eos_terminated_generation_reports_stop JSON: {v}");

    assert_eq!(
        v["choices"][0]["finish_reason"], "stop",
        "a generation that ended on EOS reported {:?}",
        v["choices"][0]["finish_reason"]
    );

    // The other half: `"stop"` is only meaningful if the run ended early. A
    // generation that reached the cap and still said "stop" is the bug.
    let completion = v["usage"]["completion_tokens"]
        .as_u64()
        .expect("no completion_tokens");
    assert!(
        completion < 32,
        "reported \"stop\" but generated {completion} of 32 tokens — \
         this ran to the cap, so \"stop\" is not an observed EOS"
    );
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

/// `/v1/completions` must reach the model. It returned a hardcoded placeholder
/// with HTTP 200, so status and shape assertions all passed against a stub —
/// which is why the provenance assertion below is the load-bearing one.
///
/// ## The `finish_reason` here rests on observation, not on the chat tests
///
/// The prompt choice cannot be borrowed from the two tests above.
/// `/v1/completions` applies NO chat template, so the model sees the bare
/// `"The capital of France is"` rather than chat's `"user: ..."` join — a
/// different input, which could perfectly well emit EOS as its first token the
/// way `truncated_generation_reports_length`'s original prompt did. It does
/// not. Observed post-fix at `temperature: 0.0`, twice: `finish_reason:
/// "length"`, `prompt_tokens: 6`, `completion_tokens: 6`, text
/// `"Paris.\n\n2."` — no EOS, the 6-token cap is what stops it, so `"length"`
/// is the correct answer for this prompt and not merely the convenient one.
#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint"]
async fn completions_endpoint_returns_model_output() {
    let (status, v) = post_json(
        "/v1/completions",
        serde_json::json!({
            "model": "tinyllama",
            "prompt": "The capital of France is",
            "max_tokens": 6,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    eprintln!("completions_endpoint_returns_model_output JSON: {v}");

    let text = v["choices"][0]["text"].as_str().expect("no choices[0].text");
    assert!(!text.trim().is_empty(), "the endpoint returned no text");
    assert!(
        !text.contains("placeholder completion"),
        "the endpoint is still returning its hardcoded placeholder: {text:?}"
    );
    assert_eq!(
        v["choices"][0]["finish_reason"], "length",
        "a 6-token cap should truncate this prompt"
    );

    // `usage` needs its own assertions here. `completions.rs` declares its OWN
    // `Usage` struct and writes its own sum, a code path `usage_counts_real_
    // tokens` never reaches, so chat's coverage buys this endpoint nothing.
    // The stub this replaced returned `prompt_tokens: 10, completion_tokens:
    // 15, total_tokens: 25` — internally consistent, so even a `total ==
    // prompt + completion` check passes against it. Only the two exact values
    // below rule it out.
    let completion = v["usage"]["completion_tokens"]
        .as_u64()
        .expect("no completion_tokens");
    let prompt = v["usage"]["prompt_tokens"].as_u64().expect("no prompt_tokens");
    let total = v["usage"]["total_tokens"].as_u64().expect("no total_tokens");

    assert_eq!(completion, 6, "completion_tokens should equal the tokens generated");
    // NO chat template on this endpoint, so the model sees the prompt exactly
    // as sent — no `"user: "` prefix, unlike the chat tests above.
    let expected_prompt = tokenizer_count("The capital of France is");
    eprintln!("tokenizer says prompt_tokens should be {expected_prompt}");
    assert_eq!(
        prompt, expected_prompt,
        "prompt_tokens disagrees with the model's own tokenizer"
    );
    assert_eq!(total, prompt + completion, "total_tokens is not the sum of its parts");
}
