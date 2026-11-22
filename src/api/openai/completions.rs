// SPDX-License-Identifier: MIT OR Apache-2.0
//! Completions Endpoint
//!
//! OpenAI-compatible `/v1/completions` endpoint for raw text completion
//! (non-chat format).

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

use crate::api::AppState;
use crate::api::model_select::{hold, scope_to_model};

/// Completion request (OpenAI-compatible)
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier
    pub model: String,

    /// Prompt text or array of prompts
    pub prompt: PromptInput,

    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Temperature for sampling (0.0-2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Sampling seed (OpenAI `seed`). With `temperature > 0`, the same seed, prompt and settings reproduce the
    /// same output on the same build and device (best effort, as OpenAI documents it). Ignored when greedy.
    /// Negative values are accepted and reinterpreted as `u64`.
    #[serde(default)]
    pub seed: Option<i64>,

    /// Nucleus sampling mass in `(0, 1]`. `1.0` (the default) keeps the whole distribution. Anything outside the
    /// range is a 400.
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// `top_k`, `min_p` and `repetition_penalty` (see [`crate::api::openai::chat::ExtraSampling`]).
    #[serde(flatten)]
    pub extra_sampling: crate::api::openai::chat::ExtraSampling,

    /// OpenAI `presence_penalty`, `-2.0..=2.0`: subtracted once from the logit of any token already generated.
    #[serde(default)]
    pub presence_penalty: f32,

    /// OpenAI `frequency_penalty`, `-2.0..=2.0`: subtracted from a token's logit per occurrence so far.
    #[serde(default)]
    pub frequency_penalty: f32,

    /// Number of completions to generate. `1` (the default) is a no-op;
    /// anything else is rejected — see `chat::validate_n`.
    #[serde(default = "default_n")]
    pub n: usize,

    /// Stop sequences. Accepts either shape the spec allows (a bare string
    /// or an array) — see [`crate::api::openai::chat::StopSequences`].
    #[serde(default)]
    pub stop: Option<crate::api::openai::chat::StopSequences>,

    /// Echo the prompt in the completion
    #[serde(default)]
    pub echo: bool,

    /// OpenAI `logprobs`: how many alternatives to return with each generated token's log-probability (`0`: the
    /// chosen token only), `0..=20`. Not combinable with `echo` (400).
    #[serde(default)]
    pub logprobs: Option<u32>,
}

/// Prompt input can be a string or array of strings
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    Single(String),
    Multiple(Vec<String>),
}

/// Completion response (OpenAI-compatible)
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Option<Usage>,
}

/// Completion choice
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Completions endpoint handler
pub async fn completions(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(request): Json<CompletionRequest>,
) -> Response {
    // Same resolution and the same reasoning as `openai::chat::chat_completions`.
    let scoped = match scope_to_model(&state, &request.model).await {
        Ok(s) => s,
        Err(response) => return *response,
    };
    let response = completions_scoped(scoped.state, headers, request)
        .await
        .into_response();
    hold(response, scoped.lease)
}

/// [`completions`] once `state` is scoped to the request's model.
async fn completions_scoped(
    state: AppState,
    headers: axum::http::HeaderMap,
    request: CompletionRequest,
) -> impl IntoResponse {
    let priority = crate::api::openai::chat::priority_from_headers(&headers);

    // `n` and `top_p` accepted-and-silently-discarded is the defect class
    // board item 71's item 2 exists to close — see the reasoning on
    // `chat::validate_n` and on this struct's `top_p` field doc.
    if let Some(rejected) = reject_unsupported(&request) {
        return rejected;
    }

    // Same check and reasoning as `openai::chat::chat_completions` (board
    // item M4 slice 2): a `chars/4 + max_tokens` estimate, not a real token
    // count -- see that function's doc comment for why.
    let estimated_prompt_chars: usize = match &request.prompt {
        PromptInput::Single(s) => s.len(),
        PromptInput::Multiple(v) => v.iter().map(|s| s.len()).sum(),
    };
    let estimated_tokens = estimated_prompt_chars / 4 + request.max_tokens.unwrap_or(100);
    if let Err(reason) = state
        .scheduler
        .lock()
        .unwrap()
        .check_admission(estimated_tokens)
    {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": {
                    "message": reason,
                    "type": "server_busy"
                }
            })),
        )
            .into_response();
    }

    match create_completion(state, request, priority).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => generation_error(&e),
    }
}

/// 400 for `n` / `top_p` / penalty values this server cannot honour.
fn reject_unsupported(request: &CompletionRequest) -> Option<Response> {
    if let Err(msg) = validate_completion_logprobs(request) {
        return Some(
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": msg})),
            )
                .into_response(),
        );
    }
    if let Err(msg) = crate::api::openai::chat::validate_n(request.n) {
        return Some(
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": msg})),
            )
                .into_response(),
        );
    }
    crate::api::openai::chat::validate_sampling_extras(
        request.top_p,
        request.presence_penalty,
        request.frequency_penalty,
        &request.extra_sampling,
    )
    .err()
    .map(|msg| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": msg})),
        )
            .into_response()
    })
}

/// 400 text for a `logprobs` this server cannot honour: out of range, or with `echo` (the prompt's own
/// logprobs are not computed).
fn validate_completion_logprobs(request: &CompletionRequest) -> Result<(), String> {
    let wanted = crate::api::openai::logprobs::validate_completions(request.logprobs)?;
    if wanted.is_some() && request.echo {
        return Err(
            "`logprobs` cannot be combined with `echo`: prompt-token logprobs are not computed"
                .to_string(),
        );
    }
    Ok(())
}

/// `QueueFullError` (503, safe to retry) and `RequestTimeoutError` (504, not obviously
/// safe to retry) get their own statuses; every other enqueue or generation failure keeps
/// the generic 500 — see `chat::queue_error_response`, which this mirrors.
fn generation_error(e: &anyhow::Error) -> Response {
    let status = if e
        .downcast_ref::<crate::engine::model_runner::QueueFullError>()
        .is_some()
    {
        StatusCode::SERVICE_UNAVAILABLE
    } else if e
        .downcast_ref::<crate::engine::model_runner::RequestTimeoutError>()
        .is_some()
    {
        StatusCode::GATEWAY_TIMEOUT
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    (status, Json(serde_json::json!({"error": e.to_string()}))).into_response()
}

/// Create completion
async fn create_completion(
    state: AppState,
    request: CompletionRequest,
    priority: crate::engine::Priority,
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
    let Some(tx) = state.model().inference_tx else {
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
    // `BuiltPrompt::raw`: no template was applied, so the prompt carries no
    // special tokens and the tokenizer must add them. The chat endpoint's
    // templated prompts take the opposite answer — see
    // `chat::build_prompt`'s docs.
    let result = crate::api::openai::chat::run_inference_once(
        &tx,
        &state.queue_depth,
        crate::api::openai::chat::BuiltPrompt::raw(prompt_text.clone()),
        max_new_tokens,
        crate::api::openai::chat::Sampling {
            temperature,
            seed: request.seed.map(|s| s as u64),
            extras: crate::engine::model_runner::SamplingExtras {
                logprobs: request.logprobs.map(|n| n as usize),
                ..crate::api::openai::chat::sampling_extras(
                    request.top_p,
                    request.presence_penalty,
                    request.frequency_penalty,
                    &request.extra_sampling,
                )
            },
        },
        state.config.request_timeout_ms,
        priority,
    )
    .await?;

    // Stop-sequence truncation applies to the GENERATED continuation only,
    // before `echo` concatenation — a stop string that happens to occur in
    // the prompt itself must not truncate the prompt when `echo: true`.
    let stop_list = request
        .stop
        .as_ref()
        .map(crate::api::openai::chat::StopSequences::as_list)
        .unwrap_or_default();
    let (generated, stopped) = if stop_list.is_empty() {
        (result.text, false)
    } else {
        crate::api::openai::chat::apply_stop_sequences(&result.text, &stop_list)
    };
    let finish_reason = if stopped {
        crate::engine::model_runner::FinishReason::Stop
    } else {
        result.finish_reason
    };

    // `echo` concatenates with no separator: OpenAI returns the prompt
    // followed immediately by its continuation, because the two are one
    // continuous text. The placeholder's blank line was an artifact of the
    // prompt and the placeholder being unrelated strings.
    let logprobs = request.logprobs.map(|_| {
        crate::api::openai::logprobs::completions_json(crate::api::openai::logprobs::within_text(
            &result.logprobs,
            &generated,
        ))
    });
    let completion_text = if request.echo {
        format!("{}{}", prompt_text, generated)
    } else {
        generated
    };

    Ok(CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion".to_string(),
        created,
        model: request.model.clone(),
        choices: vec![CompletionChoice {
            text: completion_text,
            index: 0,
            logprobs,
            finish_reason: finish_reason.as_str().to_string(),
        }],
        usage: Some(Usage {
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            total_tokens: result.prompt_tokens + result.completion_tokens,
        }),
    })
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_n() -> usize {
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;

    fn request() -> CompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "prompt": "hi",
        }))
        .unwrap()
    }

    fn state_with_runner(
        inference_tx: Option<
            std::sync::mpsc::SyncSender<crate::engine::model_runner::InferenceJob>,
        >,
    ) -> AppState {
        use crate::engine::{MemoryAwareConfig, MemoryAwareScheduler};
        AppState {
            scheduler: std::sync::Arc::new(std::sync::Mutex::new(MemoryAwareScheduler::new(
                MemoryAwareConfig::default(),
            ))),
            config: crate::api::ApiConfig::default(),
            pg_backend: None,
            manager: None,
            models: crate::api::ModelRegistry::single(
                "test-model",
                crate::api::ModelEntry {
                    inference_tx,
                    cache_control: None,
                    chat_template: None,
                    ..Default::default()
                },
            ),
            eos_monitor: std::sync::Arc::new(crate::engine::eos_monitor::EosMonitor::default()),
            security_sinks: Vec::new(),
            cache_stats: std::sync::Arc::new(std::sync::Mutex::new(
                crate::cache::PrefixCacheStats::default(),
            )),
            live_system: crate::api::new_shared_system_metrics(),
            queue_depth: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// A runner that answers every job with fixed text, so a truncation test
    /// can force a known generation rather than depending on a real model.
    fn stub_runner_with_text(
        text: &str,
        finish_reason: crate::engine::model_runner::FinishReason,
    ) -> (
        std::sync::mpsc::SyncSender<crate::engine::model_runner::InferenceJob>,
        std::thread::JoinHandle<()>,
    ) {
        use crate::engine::model_runner::{CompletionResult, InferenceJob, ResponseMode};
        let (tx, rx) = std::sync::mpsc::sync_channel::<InferenceJob>(32);
        let text = text.to_string();
        let handle = std::thread::spawn(move || {
            while let Ok(job) = rx.recv() {
                if let ResponseMode::Complete(resp) = job.response_mode {
                    let _ = resp.send(Ok(CompletionResult {
                        text: text.clone(),
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        logprobs: Vec::new(),
                        finish_reason,
                    }));
                }
            }
        });
        (tx, handle)
    }

    /// `logprobs: N` reaches the job as the alternatives count and comes back in the completions shape
    /// (parallel arrays, `top_logprobs` as token -> logprob maps, `text_offset` into the generated text); a stop
    /// sequence cuts the entries with the text; `echo` with `logprobs` and an out-of-range count are a 400; a
    /// request that did not ask gets `null`.
    #[tokio::test]
    async fn completions_logprobs_round_trip_and_refusals() {
        use crate::engine::model_runner::{
            CompletionResult, FinishReason, InferenceJob, ResponseMode, TokenLogprob,
        };
        let (tx, rx) = std::sync::mpsc::sync_channel::<InferenceJob>(32);
        let runner = std::thread::spawn(move || {
            let mut asked = Vec::new();
            while let Ok(job) = rx.recv() {
                asked.push(job.extras.logprobs);
                let entry = |t: &str, lp: f32| TokenLogprob {
                    token: t.to_string(),
                    logprob: lp,
                    top: vec![(t.to_string(), lp)],
                };
                if let ResponseMode::Complete(resp) = job.response_mode {
                    let _ = resp.send(Ok(CompletionResult {
                        text: "Hello".to_string(),
                        prompt_tokens: 1,
                        completion_tokens: 2,
                        finish_reason: FinishReason::Stop,
                        logprobs: if job.extras.logprobs.is_some() {
                            vec![entry("Hel", -0.5), entry("lo", -0.25)]
                        } else {
                            Vec::new()
                        },
                    }));
                }
            }
            asked
        });
        let state = state_with_runner(Some(tx));
        let ask = |extra: serde_json::Value| {
            let mut b = serde_json::json!({"model": "test-model", "prompt": "hi"});
            b.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            serde_json::from_value::<CompletionRequest>(b).unwrap()
        };
        let priority = crate::engine::Priority::Normal;

        let r = create_completion(
            state.clone(),
            ask(serde_json::json!({"logprobs": 1})),
            priority,
        )
        .await
        .unwrap();
        let lp = serde_json::to_value(&r.choices[0].logprobs).unwrap();
        assert_eq!(lp["tokens"], serde_json::json!(["Hel", "lo"]));
        assert_eq!(lp["token_logprobs"], serde_json::json!([-0.5, -0.25]));
        assert_eq!(lp["text_offset"], serde_json::json!([0, 3]));
        assert_eq!(lp["top_logprobs"][1]["lo"], serde_json::json!(-0.25));

        let r = create_completion(
            state.clone(),
            ask(serde_json::json!({"logprobs": 0, "stop": "lo"})),
            priority,
        )
        .await
        .unwrap();
        assert_eq!(r.choices[0].text, "Hel");
        let lp = serde_json::to_value(&r.choices[0].logprobs).unwrap();
        assert_eq!(
            lp["tokens"],
            serde_json::json!(["Hel"]),
            "cut with the stop sequence"
        );

        let r = create_completion(state.clone(), ask(serde_json::json!({})), priority)
            .await
            .unwrap();
        assert!(r.choices[0].logprobs.is_none());

        for (why, extra) in [
            ("echo", serde_json::json!({"logprobs": 1, "echo": true})),
            ("out of range", serde_json::json!({"logprobs": 21})),
        ] {
            assert!(
                reject_unsupported(&ask(extra)).is_some(),
                "{why} must be a 400"
            );
        }
        assert!(reject_unsupported(&ask(serde_json::json!({"logprobs": 20}))).is_none());

        drop(state);
        assert_eq!(runner.join().unwrap(), [Some(1), Some(0), None]);
    }

    /// Out-of-range `top_p` / penalties are a 400; in-range ones are not (the control).
    #[tokio::test]
    async fn completions_rejects_out_of_range_sampling_extras() {
        for (field, value, ok) in [
            ("top_p", 0.0, false),
            ("top_p", 1.5, false),
            ("top_p", 0.9, true),
            ("presence_penalty", 2.5, false),
            ("presence_penalty", -2.0, true),
            ("frequency_penalty", -2.5, false),
            ("frequency_penalty", 2.0, true),
        ] {
            let mut req = request();
            match field {
                "top_p" => req.top_p = value,
                "presence_penalty" => req.presence_penalty = value,
                _ => req.frequency_penalty = value,
            }
            let response = self::completions(
                State(state_with_runner(None)),
                axum::http::HeaderMap::new(),
                Json(req),
            )
            .await
            .into_response();
            let want = if ok {
                StatusCode::OK
            } else {
                StatusCode::BAD_REQUEST
            };
            assert_eq!(response.status(), want, "{field}={value}");
        }
    }

    /// `top_p` and the penalties reach the enqueued completion job.
    #[tokio::test]
    async fn sampling_extras_reach_the_enqueued_completion_job() {
        use crate::engine::model_runner::{
            CompletionResult, FinishReason, InferenceJob, ResponseMode, SamplingExtras,
        };
        let (tx, rx) = std::sync::mpsc::sync_channel::<InferenceJob>(32);
        let runner = std::thread::spawn(move || {
            let mut seen = Vec::new();
            while let Ok(job) = rx.recv() {
                seen.push(job.extras);
                if let ResponseMode::Complete(resp) = job.response_mode {
                    let _ = resp.send(Ok(CompletionResult {
                        text: "ok".to_string(),
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        logprobs: Vec::new(),
                        finish_reason: FinishReason::Stop,
                    }));
                }
            }
            seen
        });
        let state = state_with_runner(Some(tx));
        create_completion(state.clone(), request(), crate::engine::Priority::Normal)
            .await
            .expect("stub generation must not error");
        let mut req = request();
        req.top_p = 0.9;
        req.presence_penalty = 0.5;
        req.frequency_penalty = -0.25;
        create_completion(state, req, crate::engine::Priority::Normal)
            .await
            .expect("stub generation must not error");
        assert_eq!(
            runner.join().expect("stub runner panicked"),
            [
                SamplingExtras::default(),
                SamplingExtras {
                    top_p: Some(0.9),
                    presence_penalty: 0.5,
                    frequency_penalty: -0.25,
                    ..SamplingExtras::default()
                }
            ]
        );
    }

    #[tokio::test]
    async fn completions_accepts_the_default_top_p() {
        let state = state_with_runner(None);
        let req = request();
        assert_eq!(req.top_p, 1.0);

        let response = self::completions(State(state), axum::http::HeaderMap::new(), Json(req))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// Board item M4 slice 2, completions' twin of the chat.rs test of the
    /// same shape: a tiny budget rejects any request regardless of token
    /// count (the base per-slot cost alone exceeds it).
    #[tokio::test]
    async fn completions_rejects_a_request_over_the_memory_budget() {
        use crate::engine::{MemoryAwareConfig, MemoryAwareScheduler};

        let mut state = state_with_runner(None);
        state.scheduler = std::sync::Arc::new(std::sync::Mutex::new(MemoryAwareScheduler::new(
            MemoryAwareConfig {
                max_memory_bytes: 1_000,
                ..Default::default()
            },
        )));

        let response =
            self::completions(State(state), axum::http::HeaderMap::new(), Json(request()))
                .await
                .into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn completions_rejects_n_greater_than_one() {
        let state = state_with_runner(None);
        let mut req = request();
        req.n = 3;

        let response = self::completions(State(state), axum::http::HeaderMap::new(), Json(req))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// The request's `seed` reaches the enqueued job; absent, none is sent (the driver's counter decides).
    #[tokio::test]
    async fn the_request_seed_reaches_the_enqueued_completion_job() {
        use crate::engine::model_runner::{
            CompletionResult, FinishReason, InferenceJob, ResponseMode,
        };
        let run = |seed: Option<i64>| async move {
            let (tx, rx) = std::sync::mpsc::sync_channel::<InferenceJob>(32);
            let runner = std::thread::spawn(move || {
                let mut seen = Vec::new();
                while let Ok(job) = rx.recv() {
                    seen.push(job.seed);
                    if let ResponseMode::Complete(resp) = job.response_mode {
                        let _ = resp.send(Ok(CompletionResult {
                            text: "ok".to_string(),
                            prompt_tokens: 1,
                            completion_tokens: 1,
                            logprobs: Vec::new(),
                            finish_reason: FinishReason::Stop,
                        }));
                    }
                }
                seen
            });
            let mut req = request();
            req.seed = seed;
            let state = state_with_runner(Some(tx));
            create_completion(state, req, crate::engine::Priority::Normal)
                .await
                .expect("stub generation must not error");
            runner.join().expect("stub runner panicked")
        };
        assert_eq!(run(None).await, [None]);
        assert_eq!(run(Some(5)).await, [Some(5)]);
    }

    #[tokio::test]
    async fn create_completion_truncates_at_a_stop_sequence() {
        let (tx, runner) = stub_runner_with_text(
            "Paris is the capital.</s> unwanted continuation",
            crate::engine::model_runner::FinishReason::Length,
        );
        let state = state_with_runner(Some(tx));

        let mut req = request();
        req.stop = Some(crate::api::openai::chat::StopSequences::Single(
            "</s>".to_string(),
        ));

        let response = create_completion(state, req, crate::engine::Priority::Normal)
            .await
            .expect("forced generation must not error");
        drop(runner);

        assert_eq!(response.choices[0].text, "Paris is the capital.");
        assert_eq!(response.choices[0].finish_reason, "stop");
    }

    #[tokio::test]
    async fn create_completion_stop_truncation_does_not_touch_the_echoed_prompt() {
        // The prompt itself contains the stop string; only the GENERATED
        // continuation may be truncated by it.
        let (tx, runner) = stub_runner_with_text(
            " continuation</s> more",
            crate::engine::model_runner::FinishReason::Length,
        );
        let state = state_with_runner(Some(tx));

        let mut req = request();
        req.prompt = PromptInput::Single("prompt</s>text".to_string());
        req.echo = true;
        req.stop = Some(crate::api::openai::chat::StopSequences::Single(
            "</s>".to_string(),
        ));

        let response = create_completion(state, req, crate::engine::Priority::Normal)
            .await
            .expect("forced generation must not error");
        drop(runner);

        assert_eq!(response.choices[0].text, "prompt</s>text continuation");
    }

    /// A full queue must answer 503, not block the request task or report a
    /// generic 500. `sync_channel(0)` is a rendezvous channel: with no
    /// thread calling `recv`, `try_send` reports `Full` on every attempt,
    /// deterministically — no timing dependency on a consumer actually
    /// draining anything.
    #[tokio::test]
    async fn completions_returns_503_when_the_inference_queue_is_full() {
        let (tx, _rx) = std::sync::mpsc::sync_channel(0);
        let state = state_with_runner(Some(tx));
        // Must match `state.config.default_model` (the `ApiConfig` default)
        // or the handler's model-name check rejects the request with 400
        // before ever reaching the queue.
        let mut req = request();
        req.model = state.config.default_model.clone();

        let response = self::completions(State(state), axum::http::HeaderMap::new(), Json(req))
            .await
            .into_response();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
