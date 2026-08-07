# Runner Result Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Lightbulb's OpenAI-compatible API report `finish_reason` and
`usage` from what the runner observed, instead of fabricating them, and wire
`/v1/completions` to the model instead of returning a placeholder.

**Architecture:** The model runner's response channel carries a bare `String`,
so handlers have nowhere to read "how did this end" or "how many tokens" and
invent both. This widens that channel to a `CompletionResult` struct filled from
`RequestContext` — which already tracks token counts and which exit the loop
took — then updates both endpoints and both response modes to use it.

**Tech Stack:** Rust 2024, `axum` 0.8, `tokio` oneshot/mpsc, `anyhow`.

## Global Constraints

Copied from `docs/superpowers/specs/2026-08-06-runner-result-metadata-design.md`
and this project's standing rules. Every task's requirements include these.

- **One `cargo` invocation at a time.** The build-directory lock serializes them.
- **Never trust a piped exit code.** `cmd | tail` reports `tail`'s status. Always
  `; echo "CARGO EXIT: ${PIPESTATUS[0]}"`.
- **Do NOT build with `--features fuel-cuda`** (~24 minutes). `fuel-engine` is a
  different, cheap feature and IS needed.
- **Both feature configurations must compile**: default (candlelight) and
  `--features fuel-engine`. Check them sequentially, never concurrently.
- **`C:\Projects\fuel` and `C:\Projects\fuel-lightbulb-port` are READ-ONLY.**
  Read them to check APIs; never write, never run a mutating git command.
- **Checkpoint tests fail rather than skip.** An early `return` from a `#[test]`
  is a PASS. `TINYLLAMA_DIR` overrides the checkpoint path.
- **`FinishReason` has exactly two variants**, `Stop` and `Length`. OpenAI also
  defines `content_filter` and `tool_calls`; nothing produces them, and an enum
  with unconstructible variants is worse than one that grows.
- **`StreamItem::Done` carries only `finish_reason`.** Adding `completion_tokens`
  for symmetry is explicitly rejected: OpenAI omits `usage` from stream chunks,
  so nothing would consume it.
- **`/v1/completions` applies no chat template.** It is OpenAI's raw-text
  endpoint. Templating belongs to the sibling spec.
- Baseline before starting: `cargo test --lib` = **608 passed / 0 failed /
  14 ignored**.

---

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `src/engine/model_runner.rs` | modify | `FinishReason`, `CompletionResult`, `StreamItem`, `finish_reason_for`, and the two `run_jobs` arms that fill them |
| `src/engine/mod.rs` | modify | `RequestContext.prompt_tokens` |
| `src/model/parallel_model_manager.rs` | modify | Set `prompt_tokens` during prefill (candlelight backend) |
| `src/model_fuel/engine_model.rs` | modify | Set `prompt_tokens` during prefill (Fuel backend) |
| `src/api/openai/chat.rs` | modify | Consume real values in both response modes |
| `src/api/openai/completions.rs` | modify | Dispatch through `inference_tx` |
| `src/contracts/executor.rs` | modify | Take `.text` from the widened result |
| `tests/api_result_metadata.rs` | **create** | The HTTP gate for honest `finish_reason` and `usage` |

**Note on `contracts/executor.rs`:** it duplicates `run_inference_once`'s body
inline (it needs an owned `tx` inside a `Fn` closure) rather than calling
`run_inference_once_owned`, which exists for that purpose. Do not refactor that
duplication here — it is unrelated to this change and would enlarge the diff.
Just take `.text`.

---

## Task 1: The result type and the observed values

The core of the change. Everything else consumes what this produces.

**Files:**
- Modify: `src/engine/model_runner.rs`
- Modify: `src/engine/mod.rs`
- Modify: `src/model/parallel_model_manager.rs`
- Modify: `src/model_fuel/engine_model.rs`
- Modify: `src/api/openai/chat.rs` (minimal — extract `.text` so it compiles)
- Modify: `src/contracts/executor.rs` (minimal — extract `.text`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub enum FinishReason { Stop, Length }` with `pub fn as_str(&self) -> &'static str` returning `"stop"` / `"length"`
  - `pub struct CompletionResult { pub text: String, pub prompt_tokens: usize, pub completion_tokens: usize, pub finish_reason: FinishReason }`
  - `pub fn finish_reason_for(ctx: &RequestContext) -> FinishReason`
  - `ResponseMode::Complete(oneshot::Sender<anyhow::Result<CompletionResult>>)`
  - `RequestContext.prompt_tokens: usize`

- [ ] **Step 1: Write the failing test**

`finish_reason_for` is deliberately a free function over `RequestContext` so it
is unit-testable **without a model or a checkpoint**. Add to the bottom of
`src/engine/model_runner.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Request, RequestContext};

    fn ctx_with(max_new: usize, generated: usize, completed: bool) -> RequestContext {
        let mut c = RequestContext::new(Request {
            id: "t".to_string(),
            prompt: "p".to_string(),
            max_new_tokens: max_new,
        });
        c.start_decoding();
        for _ in 0..generated {
            c.record_token();
        }
        if completed {
            c.complete();
        }
        c
    }

    /// Hitting the cap must report Length. This is the defect: the API says
    /// "stop" here, which tells a client the model finished when it was cut
    /// off, so the client does not continue a truncated generation.
    #[test]
    fn reaching_the_token_cap_is_length() {
        let c = ctx_with(8, 8, false);
        assert_eq!(finish_reason_for(&c), FinishReason::Length);
    }

    /// EOS must report Stop. Guards the inverse error — an implementation that
    /// hardcoded Length would pass the test above and fail this one.
    #[test]
    fn completing_before_the_cap_is_stop() {
        let c = ctx_with(8, 3, true);
        assert_eq!(finish_reason_for(&c), FinishReason::Stop);
    }

    /// EOS on the very last allowed token is Stop, not Length. Both conditions
    /// are true at once here, and the order of the checks decides the answer —
    /// an implementation testing the cap first gets this wrong.
    #[test]
    fn eos_on_the_final_allowed_token_is_stop() {
        let c = ctx_with(8, 8, true);
        assert_eq!(finish_reason_for(&c), FinishReason::Stop);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test --lib engine::model_runner 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: FAIL to compile — `cannot find function 'finish_reason_for'` and
`cannot find type 'FinishReason'`.

- [ ] **Step 3: Add the types and the classifier**

In `src/engine/model_runner.rs`, above `pub enum ResponseMode`:

```rust
/// Why generation stopped, in OpenAI's vocabulary.
///
/// Two variants only. OpenAI also defines `content_filter` and `tool_calls`;
/// nothing in Lightbulb produces either, and an enum carrying variants no
/// code can construct is worse than one that grows when a producer appears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model emitted an end-of-sequence token.
    Stop,
    /// `max_new_tokens` was reached with no EOS. Continuing is meaningful,
    /// which is why a client needs to be able to tell this from `Stop`.
    Length,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

/// Everything a completion response needs, reported by the runner rather than
/// guessed by the handler.
#[derive(Debug)]
pub struct CompletionResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
}

/// Classify how a finished request ended.
///
/// **`complete()` is checked FIRST, and the order is load-bearing.** A request
/// whose EOS lands on its final allowed token satisfies both conditions, and it
/// stopped because the model said so — reporting `Length` there would tell a
/// client to continue a generation the model had already finished.
///
/// A free function over `RequestContext` rather than a method, so it is
/// unit-testable without a model, a checkpoint, or a runner thread.
pub fn finish_reason_for(ctx: &RequestContext) -> FinishReason {
    if matches!(ctx.state, RequestState::Completed) {
        FinishReason::Stop
    } else {
        FinishReason::Length
    }
}
```

Change `ResponseMode::Complete` to carry the new type:

```rust
    Complete(oneshot::Sender<anyhow::Result<CompletionResult>>),
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo test --lib engine::model_runner 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: PASS, 3 tests. Compilation of the rest of the crate will still be
broken — that is Step 5.

- [ ] **Step 5: Add `prompt_tokens` and fill the result**

In `src/engine/mod.rs`, add to `RequestContext`:

```rust
    /// How many tokens the model's own tokenizer produced for the prompt.
    ///
    /// Set by whichever backend tokenized during prefill. Defaults to 0 so
    /// existing constructors are unchanged; a handler must not substitute its
    /// own tokenizer, which is how this number drifts from what the model saw.
    pub prompt_tokens: usize,
```

initialize it to `0` in `RequestContext::new`, and fix any struct-literal
construction sites the compiler reports (there is one in
`tests/parallel_model_manager_integration.rs`).

In `src/model/parallel_model_manager.rs`, in `forward_batch`'s `Pending` arm,
immediately after `let tokens = self.tokenize(&ctx.request.prompt, true)?;`:

```rust
                    // Record the real count before any prefix-cache shortcut:
                    // `usage.prompt_tokens` must describe the whole prompt, not
                    // the uncached remainder.
                    ctx.prompt_tokens = tokens.len();
```

In `src/model_fuel/engine_model.rs`, in `step_one`'s `Pending` arm, immediately
after the `ids` vector is built:

```rust
                ctx.prompt_tokens = ids.len();
```

In `run_jobs`'s `ResponseMode::Complete` arm, replace the `String` result with
the struct. The existing code decodes `batch[0].generated_tokens`; keep that and
add the metadata:

```rust
                if final_result.is_none() {
                    let generated_tokens = batch[0].generated_tokens.clone();
                    match model.decode_text(&generated_tokens, false) {
                        Ok(text) => {
                            final_result = Some(Ok(CompletionResult {
                                text,
                                prompt_tokens: batch[0].prompt_tokens,
                                completion_tokens: batch[0].tokens_generated,
                                finish_reason: finish_reason_for(&batch[0]),
                            }))
                        }
                        Err(e) => final_result = Some(Err(e)),
                    }
                }
```

- [ ] **Step 6: Update the consuming call sites minimally**

Only enough to compile. Using the new fields is Tasks 2–4.

`src/api/openai/chat.rs` — `run_inference_once` returns the full result; the
owned variant keeps returning `String` because its only caller wants text:

```rust
) -> anyhow::Result<crate::engine::model_runner::CompletionResult> {
```

and in `run_inference_once_owned`, change the final expression to:

```rust
    run_inference_once(&tx, prompt, max_new_tokens, temperature)
        .await
        .map(|r| r.text)
```

At `chat.rs`'s non-streaming call site, bind the result and take `.text` for now:

```rust
        let result = run_inference_once(tx, prompt.clone(), max_new_tokens, temperature).await?;
        let text = result.text.clone();
```

`src/contracts/executor.rs` — the closure awaits the oneshot directly. Change
its final expression to take the text:

```rust
                resp_rx
                    .await
                    .map_err(|e| anyhow::anyhow!("inference runner dropped: {}", e))?
                    .map(|r| r.text)
```

- [ ] **Step 7: Verify both configurations compile and nothing regressed**

Sequentially, one cargo invocation at a time:

```bash
cargo check --tests 2>&1 | grep -E "^error" | head -10; echo "DEFAULT: ${PIPESTATUS[0]}"
```

```bash
cargo check --tests --features fuel-engine 2>&1 | grep -E "^error" | head -10; echo "FUEL: ${PIPESTATUS[0]}"
```

```bash
cargo test --lib 2>&1 | tail -5; echo "LIB: ${PIPESTATUS[0]}"
```

Expected: both checks exit 0; `cargo test --lib` reports **611 passed** (608
baseline + the 3 new tests) / 0 failed / 14 ignored.

- [ ] **Step 8: Commit**

```bash
git add src/engine/model_runner.rs src/engine/mod.rs src/model/parallel_model_manager.rs src/model_fuel/engine_model.rs src/api/openai/chat.rs src/contracts/executor.rs tests/parallel_model_manager_integration.rs
git commit -m "feat(engine): Report finish_reason and token counts from the runner

The response channel carried a bare String, so handlers had nowhere to
read how generation ended or how many tokens it used, and invented both.

finish_reason_for is a free function over RequestContext so the
classification is unit-testable without a model. complete() is checked
before the token cap, and the order matters: an EOS landing on the final
allowed token satisfies both conditions, and reporting Length there tells
a client to continue a generation the model had already finished.

prompt_tokens comes from the tokenizer the model actually used, recorded
before any prefix-cache shortcut so it describes the whole prompt."
```

---

## Task 2: The chat endpoint reports what happened

**Files:**
- Modify: `src/api/openai/chat.rs`
- Test: `tests/api_result_metadata.rs` (create)

**Interfaces:**
- Consumes: `CompletionResult { text, prompt_tokens, completion_tokens, finish_reason }`, `FinishReason::as_str()` from Task 1.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the failing test**

Create `tests/api_result_metadata.rs`:

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test --release --features fuel-engine --test api_result_metadata -- --ignored --nocapture 2>&1 | tail -25; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: FAIL. `truncated_generation_reports_length` gets `"stop"`;
`usage_counts_real_tokens` gets `completion_tokens == 0`.

If it fails to compile instead, fix the test against the real `AppState` /
`MemoryAwareScheduler` signatures — do **not** change `src/api/` to suit it.

- [ ] **Step 3: Use the real values**

In `chat.rs`'s non-streaming runner branch, replace the hardcoded response
fields:

```rust
        let result = run_inference_once(tx, prompt.clone(), max_new_tokens, temperature).await?;

        let created = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)?
            .as_secs();

        return Ok(ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created,
            model: request.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: result.text,
                    name: None,
                },
                finish_reason: result.finish_reason.as_str().to_string(),
            }],
            usage: Some(Usage {
                prompt_tokens: result.prompt_tokens,
                completion_tokens: result.completion_tokens,
                total_tokens: result.prompt_tokens + result.completion_tokens,
            }),
            lightbulb_result: None,
        });
```

Leave the **no-runner fallback** below it alone: it generated nothing, so
`"stop"` with zero counts is accurate there rather than fabricated.

Delete the now-unused `prompt_tokens` whitespace-splitting block only if the
compiler reports it unused — the fallback branch still uses it for its
informational message.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo test --release --features fuel-engine --test api_result_metadata -- --ignored --nocapture 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/api/openai/chat.rs tests/api_result_metadata.rs
git commit -m "fix(api): Report real finish_reason and usage on chat completions

finish_reason was hardcoded \"stop\" even at the token cap — the one case
where a client should continue a truncated generation. completion_tokens
was hardcoded 0, and prompt_tokens was a whitespace word count under a
'TODO: Use proper tokenizer', so usage was wrong on all three fields.

The no-runner fallback keeps \"stop\" and zero counts: it generated
nothing, so those are accurate there rather than invented."
```

---

## Task 3: Streaming reports the observed reason

**Files:**
- Modify: `src/engine/model_runner.rs`
- Modify: `src/api/openai/chat.rs`

**Interfaces:**
- Consumes: `FinishReason`, `finish_reason_for` from Task 1.
- Produces: `pub enum StreamItem { Token(String), Done { finish_reason: FinishReason } }`, and `ResponseMode::Streaming(async_mpsc::UnboundedSender<Result<StreamItem>>)`.

A streaming client sees the same lie: `chat.rs`'s terminal chunk emits a
hardcoded `"stop"`. Leaving one mode honest and the other not makes the API's
truthfulness depend on which mode you called.

- [ ] **Step 1: Add the stream item type**

In `src/engine/model_runner.rs`, beside `CompletionResult`:

```rust
/// One item on the streaming channel.
///
/// `Done` carries ONLY the finish reason. Adding `completion_tokens` for
/// symmetry with `CompletionResult` is deliberately rejected: OpenAI omits
/// `usage` from stream chunks, so nothing would consume it, and a field that
/// exists because it looked symmetrical is one the next reader has to work out
/// is unused.
#[derive(Debug)]
pub enum StreamItem {
    Token(String),
    Done { finish_reason: FinishReason },
}
```

Change `ResponseMode::Streaming` to carry it:

```rust
    Streaming(async_mpsc::UnboundedSender<Result<StreamItem>>),
```

- [ ] **Step 2: Send tokens and a terminal Done**

In `run_jobs`'s `ResponseMode::Streaming` arm, wrap each emitted token and send
exactly one `Done` when the loop exits. The existing send becomes:

```rust
                                        if stream_tx.send(Ok(StreamItem::Token(token_text))).is_err() {
                                            // Receiver dropped, stop generation
                                            break;
                                        }
```

and immediately after the decode loop ends (all `break` paths converge there),
before the arm returns:

```rust
                // Exactly one Done, carrying what the runner observed rather
                // than what the handler would otherwise assume. Sent on the
                // normal exit only — an error path already sent Err, and a
                // dropped receiver cannot receive this either.
                let _ = stream_tx.send(Ok(StreamItem::Done {
                    finish_reason: finish_reason_for(&batch[0]),
                }));
```

- [ ] **Step 3: Build the SSE terminal chunk from it**

In `chat.rs`'s stream assembly, the `.scan(...)` closure now matches on
`StreamItem`. `Token` produces today's delta chunk unchanged. `Done` produces
the terminal chunk, replacing the hardcoded `"stop"`:

```rust
                        Ok(StreamItem::Done { finish_reason }) => Some(Ok(Event::default().data(
                            serde_json::json!({
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason.as_str()
                                }]
                            })
                            .to_string(),
                        ))),
```

Remove the separately-appended hardcoded `"stop"` chunk, since `Done` now
produces it. If a `[DONE]` sentinel is appended after it, leave that alone —
it is the SSE protocol terminator, not a finish reason.

- [ ] **Step 4: Verify both configurations compile and the suite is green**

```bash
cargo check --tests 2>&1 | grep -E "^error" | head -10; echo "DEFAULT: ${PIPESTATUS[0]}"
```

```bash
cargo check --tests --features fuel-engine 2>&1 | grep -E "^error" | head -10; echo "FUEL: ${PIPESTATUS[0]}"
```

```bash
cargo test --lib 2>&1 | tail -5; echo "LIB: ${PIPESTATUS[0]}"
```

Expected: both checks exit 0, 611 passed / 0 failed / 14 ignored.

**No new automated test in this task.** The SSE path needs a streaming HTTP
client to assert against, which no existing test in this repo sets up. Say so in
the commit rather than adding a test that asserts nothing — and note it as a
gap, since this is now the least-covered path in the file.

- [ ] **Step 5: Commit**

```bash
git add src/engine/model_runner.rs src/api/openai/chat.rs
git commit -m "fix(api): Streaming reports the observed finish reason

The terminal SSE chunk hardcoded \"stop\", so a streaming client could not
tell a truncated generation from a finished one either. The channel now
carries StreamItem::{Token, Done} and the runner sends exactly one Done
with what it observed.

Done carries only finish_reason. Adding completion_tokens for symmetry
was rejected: OpenAI omits usage from stream chunks, so nothing would
consume it.

NOT COVERED BY A TEST: asserting on SSE needs a streaming client this repo
does not currently set up. Recorded as a gap rather than papered over."
```

---

## Task 4: `/v1/completions` reaches the model

**Files:**
- Modify: `src/api/openai/completions.rs`
- Modify: `tests/api_result_metadata.rs`

**Interfaces:**
- Consumes: `run_inference_once` (Task 1), `CompletionResult`, `FinishReason::as_str()`.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

Append to `tests/api_result_metadata.rs`:

```rust
/// `/v1/completions` must reach the model. It returned a hardcoded placeholder
/// with HTTP 200, so status and shape assertions all passed against a stub —
/// which is why the provenance assertion below is the load-bearing one.
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
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cargo test --release --features fuel-engine --test api_result_metadata completions_endpoint -- --ignored --nocapture 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: FAIL — the returned text contains "placeholder completion".

- [ ] **Step 3: Dispatch through the runner**

In `src/api/openai/completions.rs`, `create_completion` takes `state` and uses
it. Replace the placeholder body:

```rust
async fn create_completion(
    state: AppState,
    request: CompletionRequest,
) -> anyhow::Result<CompletionResponse> {
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();

    let prompt_text = match &request.prompt {
        PromptInput::Single(s) => s.clone(),
        PromptInput::Multiple(arr) => arr.join("\n"),
    };

    // NO CHAT TEMPLATE. `/v1/completions` is OpenAI's raw-text endpoint; the
    // prompt is used exactly as given. Templating belongs to
    // `/v1/chat/completions` — see the chat-template-resolution spec.
    let Some(tx) = &state.inference_tx else {
        // Degrade the same way the chat endpoint does: the server is up, the
        // model is not, and that is information rather than an error.
        return Ok(CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion".to_string(),
            created,
            model: request.model.clone(),
            choices: vec![CompletionChoice {
                text: "No model available on the server.".to_string(),
                index: 0,
                logprobs: None,
                finish_reason: "stop".to_string(),
            }],
            usage: None,
        });
    };

    let max_new_tokens = request.max_tokens.unwrap_or(100);
    let temperature = request.temperature as f64;
    let result = crate::api::openai::chat::run_inference_once(
        tx,
        prompt_text.clone(),
        max_new_tokens,
        temperature,
    )
    .await?;

    let completion_text = if request.echo {
        format!("{}{}", prompt_text, result.text)
    } else {
        result.text
    };

    Ok(CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created,
        model: request.model.clone(),
        choices: vec![CompletionChoice {
            text: completion_text,
            index: 0,
            logprobs: None,
            finish_reason: result.finish_reason.as_str().to_string(),
        }],
        usage: Some(Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.prompt_tokens + result.completion_tokens,
        }),
    })
}
```

Adjust the struct literals to the real field sets in that file —
`CompletionChoice` and `Usage` there may have different or additional fields
than shown. Read them; do not assume this sketch matches.

`run_inference_once` is `pub(crate)` in `chat.rs`; if the module path differs,
use the real one rather than adding a re-export.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo test --release --features fuel-engine --test api_result_metadata -- --ignored --nocapture 2>&1 | tail -20; echo "CARGO EXIT: ${PIPESTATUS[0]}"
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Verify the default build is unaffected**

```bash
cargo check --tests 2>&1 | grep -E "^error" | head -10; echo "DEFAULT: ${PIPESTATUS[0]}"
```

```bash
cargo test --lib 2>&1 | tail -5; echo "LIB: ${PIPESTATUS[0]}"
```

Expected: exit 0, 611 passed / 0 failed / 14 ignored.

- [ ] **Step 6: Commit**

```bash
git add src/api/openai/completions.rs tests/api_result_metadata.rs
git commit -m "fix(api): Wire /v1/completions to the model

create_completion took _state and returned a hardcoded placeholder with
HTTP 200 — an advertised endpoint that lied in a way status-code and
shape assertions could not catch. That is why the new test asserts the
response does NOT contain the placeholder string: provenance, not shape.

No chat template applied. /v1/completions is OpenAI's raw-text endpoint;
templating belongs to /v1/chat/completions."
```

---

## Self-Review

**Spec coverage.** §1 result type → Task 1. §2 observed values → Task 1 Steps 3
and 5. §3 streaming → Task 3. §4 construction sites → Task 1 Step 6 (all three:
`chat.rs:280`, `chat.rs:408` via Task 3, `contracts/executor.rs:61`). §5
`/v1/completions` → Task 4. §6 error handling → Task 4 Step 3 (no-runner arm)
and the existing `Err` propagation. §7 testing → Tasks 1, 2, 4.

**One gap, stated rather than hidden:** §7's "streaming's terminal chunk carries
the observed reason" has **no automated test**. Asserting on SSE requires a
streaming HTTP client this repo does not set up, and Task 3 Step 4 says so
explicitly rather than adding a test that asserts nothing. This is the weakest
part of the plan and a reviewer should treat it as such.

**Placeholder scan.** No TBD/TODO/"handle appropriately". Two steps say to read
real struct definitions rather than trust a sketch (Task 4 Step 3, Task 2 Step 2)
— that is a deliberate instruction to verify, not a placeholder.

**Type consistency.** `CompletionResult`'s four fields are named identically in
Tasks 1, 2 and 4. `FinishReason::as_str()` is defined in Task 1 and called in
Tasks 2, 3 and 4. `finish_reason_for` is defined in Task 1 and called in Tasks 1
and 3. `StreamItem::{Token, Done}` is defined in Task 3 and used only there.
`RequestContext.prompt_tokens` is added in Task 1 Step 5 and read in Task 1
Step 5's result construction.

**Verification status of the sketched code.** Task 2's `AppState` /
`MemoryAwareScheduler` construction is copied from the working
`tests/fuel_engine_http.rs`, so it is verified by precedent. Task 4's struct
literals were checked against the real definitions while writing this plan and
**match**: `CompletionChoice { text, index, logprobs, finish_reason }`
(`completions.rs:72-77`) and `Usage { prompt_tokens, completion_tokens,
total_tokens }` (`:81-85`, defined locally in that file rather than imported).

Task 4 Step 3 still says to read them before writing. That instruction stays
deliberately: a plan written today can be stale by the time it executes, and the
cost of re-reading two struct definitions is seconds.
