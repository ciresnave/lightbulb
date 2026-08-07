//! Chat Completions Endpoint
//!
//! OpenAI-compatible `/v1/chat/completions` endpoint with streaming support
//! and Lightbulb-specific extensions.

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Sse, sse::Event},
};
use futures::stream::{self, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::SystemTime;

use crate::api::AppState;

/// Chat completion request (OpenAI-compatible)
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier
    pub model: String,

    /// List of messages in the conversation
    pub messages: Vec<ChatMessage>,

    /// Temperature for sampling (0.0-2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// Enable streaming responses
    #[serde(default)]
    pub stream: bool,

    /// Number of completions to generate
    #[serde(default = "default_n")]
    pub n: usize,

    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,

    /// Lightbulb-specific extensions
    #[serde(default)]
    pub lightbulb: Option<LightbulbExtensions>,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role: system, user, assistant, function
    pub role: String,

    /// Message content
    pub content: String,

    /// Optional name for function calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Lightbulb-specific request extensions
#[derive(Debug, Clone, Deserialize)]
pub struct LightbulbExtensions {
    /// Reasoning budget
    #[serde(default)]
    pub reasoning_budget: Option<ReasoningBudget>,

    /// Enable knowledge base query
    #[serde(default)]
    pub use_knowledge_base: bool,

    /// Request metadata
    #[serde(default)]
    pub metadata: Option<RequestMetadata>,

    /// State branch to use
    #[serde(default)]
    pub state_branch: Option<String>,

    /// Output contract for structured responses from small/weak LLMs.
    /// Instructs Lightbulb to wrap the prompt with formatting instructions,
    /// parse the response, and retry with tighter prompts on parse failure.
    #[serde(default)]
    pub output_contract: Option<crate::contracts::OutputContractSpec>,

    /// Maximum inference attempts for the contract retry loop (default: 3).
    /// Each failed parse triggers a re-prompt with stricter instructions.
    #[serde(default)]
    pub max_attempts: Option<u32>,

    /// Fallback contract chain tried (in order) after the primary contract
    /// exhausts all attempts without a successful parse.
    #[serde(default)]
    pub fallback_contracts: Option<Vec<crate::contracts::OutputContractSpec>>,
}

/// Reasoning budget configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ReasoningBudget {
    pub max_chains: Option<usize>,
    pub max_steps: Option<usize>,
    pub max_tokens: Option<usize>,
}

/// Request metadata for scheduling
#[derive(Debug, Clone, Deserialize)]
pub struct RequestMetadata {
    pub priority: Option<String>,
    pub tags: Option<Vec<String>>,
}

/// Chat completion response (OpenAI-compatible)
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<Usage>,
    /// Contract execution metadata (only present when `output_contract` was used).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lightbulb_result: Option<crate::contracts::ContractResult>,
}

/// Chat completion choice
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Chat completions endpoint handler
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    if request.stream {
        // Streaming response
        let stream = create_chat_stream(state, request);
        Sse::new(stream).into_response()
    } else {
        // Non-streaming response
        match create_chat_completion(state, request).await {
            Ok(response) => (StatusCode::OK, Json(response)).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": e.to_string()})),
            )
                .into_response(),
        }
    }
}

/// Create non-streaming chat completion
async fn create_chat_completion(
    state: AppState,
    request: ChatCompletionRequest,
) -> anyhow::Result<ChatCompletionResponse> {
    // Delegate to the contract executor when an output contract is requested.
    if let Some(ref lb) = request.lightbulb {
        if lb.output_contract.is_some() {
            return create_chat_completion_with_contract(state, request).await;
        }
    }

    // Convert messages to prompt text
    let prompt = request
        .messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n");

    // For now, use a simple tokenizer approximation (split by whitespace)
    // TODO: Use proper tokenizer
    let prompt_tokens: Vec<u32> = prompt
        .split_whitespace()
        .enumerate()
        .map(|(i, _)| i as u32)
        .collect();

    let max_new_tokens = request.max_tokens.unwrap_or(100);
    let temperature = request.temperature as f64;

    // If a model runner is available, enqueue the request and wait for generated text.
    if let Some(tx) = &state.inference_tx {
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
    }

    // Fallback: no model runner available — return informational placeholder
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();

    let response_content = format!(
        "Received request for model '{}' with {} prompt tokens. No model available on the server.",
        request.model,
        prompt_tokens.len()
    );

    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created,
        model: request.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response_content,
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Some(Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: 0,
            total_tokens: prompt_tokens.len(),
        }),
        lightbulb_result: None,
    })
}

/// Run a single **complete-mode** inference call by dispatching an
/// [`InferenceJob`] to the model runner thread and awaiting the result.
pub(crate) async fn run_inference_once(
    tx: &std::sync::mpsc::Sender<crate::engine::model_runner::InferenceJob>,
    prompt: String,
    max_new_tokens: usize,
    temperature: f64,
) -> anyhow::Result<crate::engine::model_runner::CompletionResult> {
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let job = crate::engine::model_runner::InferenceJob {
        id: uuid::Uuid::new_v4().to_string(),
        prompt,
        max_new_tokens,
        temperature,
        response_mode: crate::engine::model_runner::ResponseMode::Complete(resp_tx),
    };
    tx.send(job)
        .map_err(|e| anyhow::anyhow!("failed to enqueue inference job: {}", e))?;
    resp_rx
        .await
        .map_err(|e| anyhow::anyhow!("inference runner dropped: {}", e))?
}

/// Owned-sender variant used by the `execute_contract` closure (which must
/// be `Fn`, so it clones `tx` on each call rather than borrowing).
pub(crate) async fn run_inference_once_owned(
    tx: std::sync::mpsc::Sender<crate::engine::model_runner::InferenceJob>,
    prompt: String,
    max_new_tokens: usize,
    temperature: f64,
) -> anyhow::Result<String> {
    run_inference_once(&tx, prompt, max_new_tokens, temperature)
        .await
        .map(|r| r.text)
}

/// Drive the contract retry loop for a request that carries an
/// [`OutputContractSpec`].
///
/// # Algorithm
///
/// 1. Inject the contract's system instruction into the message list.
/// 2. Call the model; try to parse the response.
/// 3. On parse failure (and while attempts remain): append the bad response as
///    `assistant` context and a tightening `user` correction, then retry.
/// 4. If the primary contract exhausts all attempts, repeat with each fallback.
/// 5. Return a [`ContractResult`] with `parse_success: false` and a `Raw`
///    output when everything fails.
async fn create_chat_completion_with_contract(
    state: AppState,
    request: ChatCompletionRequest,
) -> anyhow::Result<ChatCompletionResponse> {
    use crate::contracts::executor::execute_contract;
    use crate::contracts::validation::RawMessage;

    let lb = request.lightbulb.as_ref().expect("checked before calling");
    let primary_contract = lb.output_contract.as_ref().expect("checked before calling");
    let max_attempts = lb.max_attempts.unwrap_or(3).max(1);
    let fallbacks = lb.fallback_contracts.as_deref().unwrap_or(&[]);

    let max_new_tokens = request.max_tokens.unwrap_or(200);
    let temperature = request.temperature as f64;

    // No model runner → error immediately.
    let Some(tx) = state.inference_tx.clone() else {
        return Err(anyhow::anyhow!("no model runner available"));
    };

    let msgs: Vec<RawMessage> = request
        .messages
        .iter()
        .map(|m| RawMessage { role: m.role.clone(), content: m.content.clone() })
        .collect();

    let exec = execute_contract(
        &msgs,
        &request.model,
        primary_contract,
        max_attempts,
        fallbacks,
        move |prompt| {
            let tx = tx.clone();
            async move { run_inference_once_owned(tx, prompt, max_new_tokens, temperature).await }
        },
    )
    .await?;

    build_contract_response(&request, &exec.final_text, exec.result)
}

/// Build a [`ChatCompletionResponse`] that carries a contract result in the
/// `lightbulb_result` field alongside the raw model output in `choices`.
fn build_contract_response(
    request: &ChatCompletionRequest,
    content: &str,
    result: crate::contracts::ContractResult,
) -> anyhow::Result<ChatCompletionResponse> {
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();
    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created,
        model: request.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: content.to_string(),
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: None,
        lightbulb_result: Some(result),
    })
}

/// Create streaming chat completion
fn create_chat_stream(
    state: AppState,
    request: ChatCompletionRequest,
) -> impl Stream<Item = Result<Event, Infallible>> {
    // Convert messages to prompt text
    let prompt = request
        .messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n");

    let max_new_tokens = request.max_tokens.unwrap_or(100);
    let model = request.model.clone();

    // If a model runner is available, create a streaming job
    if let Some(tx) = state.inference_tx {
        let (stream_tx, stream_rx) = tokio::sync::mpsc::unbounded_channel();

        let job = crate::engine::model_runner::InferenceJob {
            id: uuid::Uuid::new_v4().to_string(),
            prompt: prompt.clone(),
            max_new_tokens,
            temperature: request.temperature as f64,
            response_mode: crate::engine::model_runner::ResponseMode::Streaming(stream_tx),
        };

        // Send the job to the model runner thread
        if tx.send(job).is_err() {
            // Failed to enqueue - return error stream
            return stream::once(async {
                let data = serde_json::json!({
                    "error": {
                        "message": "Failed to enqueue inference job",
                        "type": "internal_error"
                    }
                });
                Ok(Event::default().data(data.to_string()))
            })
            .boxed();
        }

        // Create stream from the receiver using futures combinators
        let created = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let chat_id_for_stream = chat_id.clone();
        let model_for_stream = model.clone();
        let chat_id_for_done = chat_id.clone();
        let model_for_done = model.clone();

        use futures::stream::StreamExt as FuturesStreamExt;
        use tokio_stream::wrappers::UnboundedReceiverStream;

        let token_stream = UnboundedReceiverStream::new(stream_rx)
            .scan(true, move |is_first, result| {
                let chat_id = chat_id_for_stream.clone();
                let model = model_for_stream.clone();
                let first = *is_first;
                *is_first = false;

                async move {
                    match result {
                        Ok(token_text) => Some(Ok(Event::default().data(
                            serde_json::json!({
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "role": if first { Some("assistant") } else { None },
                                        "content": token_text
                                    },
                                    "finish_reason": None::<String>
                                }]
                            })
                            .to_string(),
                        ))),
                        Err(e) => Some(Ok(Event::default().data(
                            serde_json::json!({
                                "error": {
                                    "message": e.to_string(),
                                    "type": "inference_error"
                                }
                            })
                            .to_string(),
                        ))),
                    }
                }
            })
            .chain(stream::once({
                let chat_id = chat_id_for_done.clone();
                let model = model_for_done.clone();
                async move {
                    Ok(Event::default().data(
                        serde_json::json!({
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        })
                        .to_string(),
                    ))
                }
            }));

        return token_stream.boxed();
    }

    // Fallback: no model runner available - return mock stream
    let model_name = request.model.clone();
    let chunks = vec![
        "No ",
        "model ",
        "runner ",
        "available. ",
        "Please ",
        "configure ",
        "model ",
        "path.",
    ];

    stream::iter(chunks.into_iter().enumerate().map(move |(i, chunk)| {
        let data = serde_json::json!({
            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            "object": "chat.completion.chunk",
            "created": SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            "model": model_name.clone(),
            "choices": [{
                "index": 0,
                "delta": {
                    "content": chunk
                },
                "finish_reason": if i == 7 { Some("stop") } else { None }
            }]
        });

        Ok(Event::default().data(data.to_string()))
    }))
    .boxed()
}

fn default_temperature() -> f32 {
    1.0
}

fn default_n() -> usize {
    1
}
