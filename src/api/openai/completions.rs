//! Completions Endpoint
//!
//! OpenAI-compatible `/v1/completions` endpoint for raw text completion
//! (non-chat format).

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
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
    _state: AppState,
    request: CompletionRequest,
) -> anyhow::Result<CompletionResponse> {
    // TODO: Integrate with actual inference engine
    // For now, return a mock response

    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();

    let prompt_text = match &request.prompt {
        PromptInput::Single(s) => s.clone(),
        PromptInput::Multiple(arr) => arr.join("\n"),
    };

    let completion_text = if request.echo {
        format!(
            "{}\n\nThis is a placeholder completion. Inference engine integration pending.",
            prompt_text
        )
    } else {
        "This is a placeholder completion. Inference engine integration pending.".to_string()
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
            finish_reason: "stop".to_string(),
        }],
        usage: Some(Usage {
            prompt_tokens: 10,
            completion_tokens: 15,
            total_tokens: 25,
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
