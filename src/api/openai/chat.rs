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
        "Received request for model '{}'. No model available on the server.",
        request.model
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
        // `None`, not zeros and not a word count. There is no model here, so
        // there is no tokenizer, so the prompt's token count is genuinely
        // unknown — and this arm previously reported
        // `prompt.split_whitespace().count()` under a `TODO: Use proper
        // tokenizer`, which is defect 4 in the spec's table. Reporting a word
        // count as a token count is the exact lie this branch exists to remove;
        // it survived here because the plan's justification for keeping this
        // fallback ("it generated nothing, so `stop` and zero counts are
        // accurate") covers `finish_reason` and `completion_tokens` and says
        // nothing about `prompt_tokens`. `/v1/completions` already degrades
        // this way, so the two endpoints now match.
        usage: None,
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
) -> anyhow::Result<crate::engine::model_runner::CompletionResult> {
    run_inference_once(&tx, prompt, max_new_tokens, temperature).await
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

    build_contract_response(&request, exec)
}

/// Build a [`ChatCompletionResponse`] that carries a contract result in the
/// `lightbulb_result` field alongside the raw model output in `choices`.
///
/// `finish_reason` and `usage` come from the runner via
/// [`ContractExecutionResult`], not from this function. `usage` therefore
/// covers **every** attempt the retry loop made, while `finish_reason`
/// describes only the final attempt — the one whose text is in `choices[0]`.
/// Both rules are documented on the type.
///
/// [`ContractExecutionResult`]: crate::contracts::executor::ContractExecutionResult
fn build_contract_response(
    request: &ChatCompletionRequest,
    exec: crate::contracts::executor::ContractExecutionResult,
) -> anyhow::Result<ChatCompletionResponse> {
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();
    let finish_reason = exec
        .finish_reason
        .map(|r| r.as_str().to_string())
        // `None` means the loop ran zero inferences, which requires
        // `max_attempts == 0`. The caller above clamps it to `>= 1`, so this
        // arm is unreachable from the HTTP path; it exists so a direct
        // `execute_contract` caller does not get a panic.
        .unwrap_or_else(|| "stop".to_string());
    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created,
        model: request.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: exec.final_text,
                name: None,
            },
            finish_reason,
        }],
        usage: Some(Usage {
            prompt_tokens: exec.prompt_tokens,
            completion_tokens: exec.completion_tokens,
            total_tokens: exec.prompt_tokens + exec.completion_tokens,
        }),
        lightbulb_result: Some(exec.result),
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

        use crate::engine::model_runner::StreamItem;
        use futures::stream::StreamExt as FuturesStreamExt;
        use tokio_stream::wrappers::UnboundedReceiverStream;

        // `chat_id` and `model` are moved into the closure and re-cloned per
        // chunk; nothing after this point uses either (the no-runner fallback
        // below re-derives the model name from `request`).
        let token_stream =
            UnboundedReceiverStream::new(stream_rx).scan(true, move |is_first, result| {
                let chat_id = chat_id.clone();
                let model = model.clone();
                let first = *is_first;
                *is_first = false;

                async move {
                    match result {
                        Ok(StreamItem::Token(token_text)) => Some(Ok(Event::default().data(
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
            });

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::executor::ContractExecutionResult;
    use crate::contracts::{ContractOutput, ContractResult};
    use crate::engine::model_runner::FinishReason;

    fn request() -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap()
    }

    fn exec_result(finish_reason: Option<FinishReason>) -> ContractExecutionResult {
        ContractExecutionResult {
            final_text: "yes".to_string(),
            prompt_tokens: 34,
            completion_tokens: 10,
            finish_reason,
            result: ContractResult {
                contract_type: "enum_choice".to_string(),
                attempts: 2,
                parse_success: true,
                model_id: "test-model".to_string(),
                latency_ms: 7,
                output: ContractOutput::EnumChoice { choice: "yes".to_string() },
            },
        }
    }

    /// The contract path used to hardcode `finish_reason: "stop"` and
    /// `usage: None` regardless of what the runner reported. Fails if either
    /// regresses to a constant.
    #[test]
    fn contract_response_reports_runner_metadata_not_constants() {
        let resp = build_contract_response(&request(), exec_result(Some(FinishReason::Length)))
            .expect("response should build");

        assert_eq!(
            resp.choices[0].finish_reason, "length",
            "finish_reason must come from the runner, not a hardcoded \"stop\""
        );
        let usage = resp.usage.expect("usage must be reported, not None");
        assert_eq!(usage.prompt_tokens, 34);
        assert_eq!(usage.completion_tokens, 10);
        assert_eq!(usage.total_tokens, 44);
        // The one thing this change must NOT touch.
        assert_eq!(resp.choices[0].message.content, "yes");
        assert!(resp.lightbulb_result.is_some());
    }

    /// `None` (zero inferences ran) degrades to `"stop"` rather than panicking.
    /// Unreachable from the handler, which clamps `max_attempts` to `>= 1`.
    #[test]
    fn contract_response_tolerates_absent_finish_reason() {
        let resp = build_contract_response(&request(), exec_result(None)).unwrap();
        assert_eq!(resp.choices[0].finish_reason, "stop");
    }
}
