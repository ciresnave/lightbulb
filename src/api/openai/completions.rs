//! Completions Endpoint
//!
//! OpenAI-compatible `/v1/completions` endpoint for raw text completion
//! (non-chat format).

use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

use crate::api::AppState;

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

    /// Top-p sampling. `1.0` (the default) is a no-op; any other value is
    /// rejected with a 400 — nucleus sampling is not wired into either
    /// decode path (`src/sampling.rs::top_p_filter` exists but is dead code,
    /// called from neither), so accepting it would silently do nothing.
    #[serde(default = "default_top_p")]
    pub top_p: f32,

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
    Json(request): Json<CompletionRequest>,
) -> impl IntoResponse {
    // Same check and same reasoning as `openai::chat::chat_completions` —
    // reject a `model` that does not name what is actually loaded, only
    // while a model IS loaded, so the existing "No model available on the
    // server" fallback in `create_completion` still handles the other case.
    if state.inference_tx.is_some() && request.model != state.config.default_model {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!(
                    "model '{}' is not loaded on this server; the loaded model is '{}'",
                    request.model, state.config.default_model
                )
            })),
        )
            .into_response();
    }

    // `n` and `top_p` accepted-and-silently-discarded is the defect class
    // board item 71's item 2 exists to close — see the reasoning on
    // `chat::validate_n` and on this struct's `top_p` field doc.
    if let Err(msg) = crate::api::openai::chat::validate_n(request.n) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": msg})),
        )
            .into_response();
    }
    if request.top_p != 1.0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!(
                    "top_p={} is not supported: nucleus sampling is not wired into either \
                     decode path. Send top_p=1.0 (the default) or omit the field.",
                    request.top_p
                )
            })),
        )
            .into_response();
    }

    match create_completion(state, request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// Create completion
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
    // `BuiltPrompt::raw`: no template was applied, so the prompt carries no
    // special tokens and the tokenizer must add them. The chat endpoint's
    // templated prompts take the opposite answer — see
    // `chat::build_prompt`'s docs.
    let result = crate::api::openai::chat::run_inference_once(
        tx,
        crate::api::openai::chat::BuiltPrompt::raw(prompt_text.clone()),
        max_new_tokens,
        temperature,
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
            logprobs: None,
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
        inference_tx: Option<std::sync::mpsc::Sender<crate::engine::model_runner::InferenceJob>>,
    ) -> AppState {
        use crate::engine::{MemoryAwareConfig, MemoryAwareScheduler};
        AppState {
            scheduler: std::sync::Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default())),
            config: crate::api::ApiConfig::default(),
            db_pool: None,
            inference_tx,
            chat_template: None,
            eos_monitor: std::sync::Arc::new(crate::engine::eos_monitor::EosMonitor::default()),
        }
    }

    /// A runner that answers every job with fixed text, so a truncation test
    /// can force a known generation rather than depending on a real model.
    fn stub_runner_with_text(
        text: &str,
        finish_reason: crate::engine::model_runner::FinishReason,
    ) -> (
        std::sync::mpsc::Sender<crate::engine::model_runner::InferenceJob>,
        std::thread::JoinHandle<()>,
    ) {
        use crate::engine::model_runner::{CompletionResult, InferenceJob, ResponseMode};
        let (tx, rx) = std::sync::mpsc::channel::<InferenceJob>();
        let text = text.to_string();
        let handle = std::thread::spawn(move || {
            while let Ok(job) = rx.recv() {
                if let ResponseMode::Complete(resp) = job.response_mode {
                    let _ = resp.send(Ok(CompletionResult {
                        text: text.clone(),
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        finish_reason,
                    }));
                }
            }
        });
        (tx, handle)
    }

    #[tokio::test]
    async fn completions_rejects_top_p_not_equal_one() {
        let state = state_with_runner(None);
        let mut req = request();
        req.top_p = 0.9;

        let response = self::completions(State(state), Json(req))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn completions_accepts_the_default_top_p() {
        let state = state_with_runner(None);
        let req = request();
        assert_eq!(req.top_p, 1.0);

        let response = self::completions(State(state), Json(req))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn completions_rejects_n_greater_than_one() {
        let state = state_with_runner(None);
        let mut req = request();
        req.n = 3;

        let response = self::completions(State(state), Json(req))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
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

        let response = create_completion(state, req)
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

        let response = create_completion(state, req)
            .await
            .expect("forced generation must not error");
        drop(runner);

        assert_eq!(response.choices[0].text, "prompt</s>text continuation");
    }
}
