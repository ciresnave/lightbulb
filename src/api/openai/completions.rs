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

    /// Top-p sampling
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Number of completions to generate
    #[serde(default = "default_n")]
    pub n: usize,

    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,

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

    // `echo` concatenates with no separator: OpenAI returns the prompt
    // followed immediately by its continuation, because the two are one
    // continuous text. The placeholder's blank line was an artifact of the
    // prompt and the placeholder being unrelated strings.
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

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_n() -> usize {
    1
}
