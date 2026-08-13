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

/// The pre-template prompt construction: `"{role}: {content}"` per message,
/// newline-joined.
///
/// This is not any model's chat template. It is kept only as the fallback for a
/// checkpoint no tier could resolve a template for, because a server that
/// refuses to answer is worse than one that answers badly and says why — and
/// `chat_template::resolve` warns at startup, naming the directory, whenever
/// this is what will run.
///
/// It must have exactly one definition. It previously had three (both sites in
/// this file plus `contracts::validation::messages_to_prompt`), which is how
/// the streaming path kept the defect after the non-streaming path was looked
/// at. `messages_to_prompt` was deleted rather than re-pointed at the template
/// for the same reason: a second renderer is a second thing to drift.
fn legacy_join(messages: &[crate::contracts::validation::RawMessage]) -> String {
    messages
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A prompt, together with how the backend must tokenize it.
///
/// The two travel as one value because they are one decision. A `String`
/// return type forces every caller to answer "does this already contain BOS?"
/// from context it does not have, and the answer that looks safe — `true`,
/// matching every other `encode` call in the codebase — is the wrong one for
/// the only path that renders a template.
pub(crate) struct BuiltPrompt {
    pub(crate) text: String,
    /// Passed straight to the tokenizer at prefill; see
    /// `engine::RequestContext::add_special_tokens`.
    pub(crate) add_special_tokens: bool,
}

impl BuiltPrompt {
    /// A prompt that is raw text: no chat template was applied, so it contains
    /// no special tokens of its own and the tokenizer must supply them.
    ///
    /// `/v1/completions` is the caller. Its prompt is used exactly as the
    /// client sent it — OpenAI's raw-text endpoint, spec §6 — so it is in the
    /// same position as the legacy join: `add_special_tokens = true` is
    /// correct, and is what the pre-template server did for every request.
    pub(crate) fn raw(text: String) -> Self {
        Self {
            text,
            add_special_tokens: true,
        }
    }
}

/// Build the prompt for `messages`, preferring the model's own chat template.
///
/// Used by BOTH `create_chat_completion` and `create_chat_stream`. Sharing one
/// function rather than repeating the logic is the point: the two paths having
/// separate prompt construction is the defect this replaces.
///
/// The template carries its own `bos`/`eos` (see
/// `chat_template::ResolvedTemplate`), so there is no token argument here to
/// get wrong.
///
/// # Why the tokenizer flag is decided HERE
///
/// Real chat templates interpolate `bos_token` into the prompt **text**:
/// Meta-Llama-3-8B-Instruct's, Mistral-7B-Instruct's and Gemma's all begin
/// `{{ bos_token }}`, and Llama-2's emits it inside its message loop.
/// TinyLlama's tokenizer (and every Llama-family one)
/// has a `TemplateProcessing` post_processor of the form
/// `single: [SpecialToken "<s>", Sequence A]`, so `encode(text, true)` prepends
/// BOS a *second* time and the model receives `128000, 128000, 128006, …` — a
/// pair it never saw in training. The request still returns 200, the render
/// still succeeds, nothing falls through and nothing is logged. HuggingFace
/// avoids this by tokenizing `apply_chat_template` output with
/// `add_special_tokens=False`; this function is where that decision is made,
/// because it is the only place that knows whether a template was applied.
///
/// The legacy-join branch keeps `true`. A joined prompt is ordinary text with
/// no special tokens in it, so the tokenizer must still supply BOS — the same
/// as `/v1/completions`, which by spec §6 applies no template at all.
///
/// ## Rejected: binding `bos_token` to `""` in the template environment
///
/// It is one line in `ChatTemplate::render` and would make every template emit
/// no BOS, so `add_special_tokens = true` would then be correct everywhere and
/// none of this plumbing would exist. Rejected on two grounds:
///
/// 1. **It is only correct for `bos_token` at position 0.** A template is free
///    to reference `bos_token` anywhere, and Llama-2's official one does: it
///    emits `bos_token + '[INST] ' + content + ' [/INST]'` **inside** the
///    message loop, so a three-turn conversation contains three of them. A
///    tokenizer's post_processor puts back only the leading one. Blanking
///    `bos_token` therefore deletes the turn-internal markers outright — the
///    prompt loses structure instead of gaining a duplicate. Same class of
///    defect, caught by nothing, just quieter.
/// 2. **It changes the rendered output rather than fixing the tokenization.**
///    The bug is in how a correct string is encoded. Editing the string so the
///    encoder's mistake cancels out leaves `ResolvedTemplate::render` producing
///    something that is not what `transformers` produces for the same
///    checkpoint, which breaks the one property that makes this module
///    checkable against upstream at all — including
///    `rendering_uses_the_checkpoints_own_tokens`, whose whole assertion is that
///    the rendered text matches the checkpoint's own.
fn build_prompt(state: &AppState, messages: &[ChatMessage]) -> BuiltPrompt {
    let raw: Vec<crate::contracts::validation::RawMessage> = messages
        .iter()
        .map(|m| crate::contracts::validation::RawMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();

    build_prompt_from_raw(state.chat_template.as_deref(), &raw)
}

/// [`build_prompt`] over the contracts module's own message type, taking the
/// template directly rather than through [`AppState`].
///
/// This is the shared body; `build_prompt` is the `ChatMessage`/`AppState`
/// adapter over it. It exists as a separate function because the contract
/// executor cannot reuse `build_prompt`: it re-renders per attempt against a
/// message list it has mutated (`contracts::executor` pushes the failed
/// assistant reply and a tightening user message between attempts), and it runs
/// from `execute_contract_with_runner`, which has no `AppState` at all.
///
/// Both of those callers go through **this** function rather than rendering
/// themselves, so `BuiltPrompt::add_special_tokens` is decided in exactly one
/// place for every prompt the server sends. `contracts::validation::
/// messages_to_prompt` was the fourth renderer and is deleted; the flag was the
/// reason a fifth could not be tolerated, since a call site that renders also
/// has to restate the flag and `true` is both the wrong answer and the one that
/// matches every other `encode` call in the codebase.
pub(crate) fn build_prompt_from_raw(
    template: Option<&crate::api::chat_template::ResolvedTemplate>,
    raw: &[crate::contracts::validation::RawMessage],
) -> BuiltPrompt {
    // The one place the polarity is written down. Callers move the pair around
    // together, so there is no `!` at a call site to get backwards.
    let joined = |text: String| BuiltPrompt {
        text,
        add_special_tokens: true,
    };

    match template {
        Some(t) => match t.render(raw) {
            Ok(text) => BuiltPrompt {
                text,
                add_special_tokens: false,
            },
            // Per-request and rare, so `warn`: a template that resolved at
            // startup and then failed on a particular message list is a real
            // anomaly worth seeing every time it happens.
            //
            // The flag follows the prompt that is actually used, not the one
            // that was attempted: this arm falls back to the join, so the
            // tokenizer must add BOS after all.
            Err(e) => {
                tracing::warn!(
                    "chat template ({:?}) failed to render: {e}; using the legacy join",
                    t.resolved_by()
                );
                joined(legacy_join(raw))
            }
        },
        // `debug`, not `warn`: `resolve` already warned once at startup with the
        // model directory and the remedy. Repeating that per request would bury
        // the render failure above, which is the one that is genuinely new
        // information.
        None => {
            tracing::debug!("no chat template for this model; using the legacy join");
            joined(legacy_join(raw))
        }
    }
}

/// Record how one completion ended, and log when this reading is the first to
/// fall below the threshold since the last healthy one.
///
/// **Observes and never acts.** Nothing downstream of this call reads the
/// monitor to decide what to prompt, render or retry, and nothing may be added
/// that does: a server that switches template mid-flight on a heuristic cannot
/// be reasoned about from a request/response pair, while one that is
/// consistently wrong and says so can be (spec §3).
///
/// Takes the monitor and the resolution rather than `&AppState`, because the
/// streaming path calls it from a `'static` closure that outlives the state —
/// the field it needs has to be cloned out before `state.inference_tx` is moved.
///
/// Called from all three sites that observe a `FinishReason`: the plain
/// completion, the streaming `Done` frame, and the contract loop's final
/// attempt. Wiring only the first is the same shape of miss as fixing only one
/// of this file's two former prompt-construction sites — on a server whose
/// clients stream, a monitor recording nothing never fills its window and never
/// reports anything, which is indistinguishable in the log from a healthy model.
fn record_completion(
    monitor: &crate::engine::eos_monitor::EosMonitor,
    resolved_by: Option<crate::api::chat_template::Resolution>,
    finish_reason: crate::engine::model_runner::FinishReason,
) {
    // `record` is edge-triggered: `Some` only on the reading that crosses,
    // `None` on every completion after it while the rate stays low.
    let Some(rate) = monitor.record(finish_reason) else {
        return;
    };
    tracing::warn!(
        "only {:.0}% of the last {} completions stopped on EOS. The chat template \
         for this model ({}) may be wrong for it: a model prompted with someone \
         else's template continues text instead of answering, and runs to the \
         token budget. This is a warning only — no template will be changed \
         automatically.",
        rate * 100.0,
        monitor.window(),
        match resolved_by {
            Some(r) => format!("resolved via {r:?}"),
            None => "none resolved; the legacy role: content join is in use".to_string(),
        },
    );
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

    let prompt = build_prompt(&state, &request.messages);

    let max_new_tokens = request.max_tokens.unwrap_or(100);
    let temperature = request.temperature as f64;

    // If a model runner is available, enqueue the request and wait for generated text.
    if let Some(tx) = &state.inference_tx {
        let result = run_inference_once(tx, prompt, max_new_tokens, temperature).await?;

        record_completion(
            &state.eos_monitor,
            state.chat_template.as_ref().map(|t| t.resolved_by()),
            result.finish_reason,
        );

        // Only here, inside the branch where a model actually generated. The
        // no-runner fallback below reports `finish_reason: "stop"` for a
        // canned string; recording that would report a perfect EOS rate for a
        // server that has never run the model.

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
    prompt: BuiltPrompt,
    max_new_tokens: usize,
    temperature: f64,
) -> anyhow::Result<crate::engine::model_runner::CompletionResult> {
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let job = crate::engine::model_runner::InferenceJob {
        id: uuid::Uuid::new_v4().to_string(),
        prompt: prompt.text,
        max_new_tokens,
        temperature,
        add_special_tokens: prompt.add_special_tokens,
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
    prompt: BuiltPrompt,
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
        .map(|m| RawMessage {
            role: m.role.clone(),
            content: m.content.clone(),
        })
        .collect();

    // Cloned out of `state` before the closure, because the closure must be
    // `Fn` and outlives this borrow.
    let template = state.chat_template.clone();

    let exec = execute_contract(
        &msgs,
        &request.model,
        primary_contract,
        max_attempts,
        fallbacks,
        // The closure receives the message list, not a prompt, and renders it
        // **per attempt** — `execute_contract` mutates that list between
        // attempts, so a prompt rendered once here would be stale from attempt
        // 2 onward.
        //
        // `build_prompt_from_raw` is the same function the plain and streaming
        // paths use. What that guarantees is narrower than it looks: the
        // *logic* — including the `add_special_tokens` answer — has one
        // definition, so it cannot be changed here without changing it there.
        // It does not guarantee this path calls it. The call below can be
        // replaced wholesale (`BuiltPrompt::raw(legacy_join(&msgs))` is a
        // one-line revert of this change for every HTTP contract request) with
        // the shared builder left untouched and every one of its own tests
        // still green. That is why this site has a test of its own —
        // `the_http_contract_path_renders_each_attempt_through_the_template`,
        // asserting on the `InferenceJob`s this handler enqueues — separate
        // from the one covering `execute_contract_with_runner`'s call.
        move |msgs: Vec<RawMessage>| {
            let tx = tx.clone();
            let prompt = build_prompt_from_raw(template.as_deref(), &msgs);
            async move { run_inference_once_owned(tx, prompt, max_new_tokens, temperature).await }
        },
    )
    .await?;

    // One record per request, not per attempt: `ContractExecutionResult`
    // reports the finish reason of the final attempt only — the one whose text
    // the client receives — and the retries in between are prompts this server
    // constructed, not requests a client made. `None` means the loop ran zero
    // inferences, so there is nothing to record.

    if let Some(reason) = exec.finish_reason {
        record_completion(
            &state.eos_monitor,
            state.chat_template.as_ref().map(|t| t.resolved_by()),
            reason,
        );
    }

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
    // The SAME prompt construction as the non-streaming path. These two sites
    // drifting apart is not hypothetical: this one was the second of the two
    // `role: content` joins, and it is the one a fix aimed at
    // `create_chat_completion` leaves behind — the endpoint looks correct,
    // returns 200, and prompts a chat model like a base model on every
    // `"stream": true` request.
    let prompt = build_prompt(&state, &request.messages);

    let max_new_tokens = request.max_tokens.unwrap_or(100);
    let model = request.model.clone();

    // Cloned out of `state` BEFORE `state.inference_tx` is moved below, and
    // before the token stream — which is `'static` and outlives this function —
    // captures them. The streaming path observing its own completions is the
    // point: it is the path a fix aimed at `create_chat_completion` leaves
    // behind, and a monitor fed by only one of the two never fills its window
    // on a server whose clients stream.
    let eos_monitor = state.eos_monitor.clone();
    let resolved_by = state.chat_template.as_ref().map(|t| t.resolved_by());

    // If a model runner is available, create a streaming job
    if let Some(tx) = state.inference_tx {
        let (stream_tx, stream_rx) = tokio::sync::mpsc::unbounded_channel();

        let job = crate::engine::model_runner::InferenceJob {
            id: uuid::Uuid::new_v4().to_string(),
            prompt: prompt.text,
            max_new_tokens,
            temperature: request.temperature as f64,
            // Same value the non-streaming path sends, because it comes from
            // the same `build_prompt` call. The two paths carrying separate
            // answers to this is the shape of the defect this file already has
            // history with.
            add_special_tokens: prompt.add_special_tokens,
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
                let eos_monitor = eos_monitor.clone();
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
                        Ok(StreamItem::Done { finish_reason }) => {
                            // The stream's one observation of how generation
                            // ended. It is the same value the frame below
                            // reports to the client, read here rather than
                            // re-derived.
                            record_completion(&eos_monitor, resolved_by, finish_reason);
                            Some(Ok(Event::default().data(
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
                            )))
                        }
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
    use crate::engine::eos_monitor::{DEFAULT_MIN_STOP_RATE, EosMonitor};
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
                output: ContractOutput::EnumChoice {
                    choice: "yes".to_string(),
                },
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

    // ─── The BOS-doubling guard ─────────────────────────────────────────────
    //
    // These assert on ENCODED IDS, not on rendered text. Rendered text cannot
    // discriminate the defect: the render is correct either way, and the whole
    // failure lives one layer down, in what `encode` does with a string that
    // already starts with BOS.
    //
    // The fixture is a synthetic Llama-3-shaped checkpoint rather than the
    // TinyLlama snapshot, and that is load-bearing. TinyLlama resolves at tier
    // 3 to `registry::ZEPHYR`, which never mentions `bos_token`, so its prompt
    // is *accidentally* correct under either value of the flag — a test built
    // on it would pass with the flag hardcoded to `true`. The template below
    // opens with `{{ bos_token }}` the way Meta-Llama-3-8B-Instruct's real one
    // does, and the tokenizer's `post_processor` is the `TemplateProcessing`
    // shape that Llama-2, Llama-3 and TinyLlama all actually ship.

    const FIXTURE_BOS: &str = "<|begin_of_text|>";
    const FIXTURE_BOS_ID: u32 = 1;

    /// The Llama-3-shaped template: opens with `{{ bos_token }}`, exactly as
    /// Meta-Llama-3-8B-Instruct's real one does.
    const LLAMA3_SHAPED_TEMPLATE: &str = r#"{{ bos_token }}{% for m in messages %}{{ '<|start_header_id|>' + m.role + '<|end_header_id|>\n\n' + m.content + eos_token }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"#;

    /// A template that `raise_exception`s on any role other than user or
    /// assistant — i.e. on the system message
    /// `validation::inject_contract_instruction` adds to **every** contract
    /// request.
    ///
    /// This is not a contrived fixture. It is the shape of
    /// Mistral-7B-Instruct-v0.1/v0.2's and Gemma's real templates, which is why
    /// `build_prompt_from_raw`'s `Err` arm is a live path on the contract route
    /// rather than a rare one: on such a checkpoint every attempt of every
    /// contract request takes it.
    const RAISES_ON_SYSTEM_TEMPLATE: &str = r#"{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"#;

    /// A real `tokenizers` tokenizer. `WordLevel` keeps the vocab to a dozen
    /// entries; the part that matters is `post_processor`, copied in shape
    /// from TinyLlama's own tokenizer.json: `single` is
    /// `[SpecialToken, Sequence A]`, so `encode(text, true)` prepends BOS.
    const FIXTURE_TOKENIZER_JSON: &str = r#"{
              "version": "1.0",
              "truncation": null,
              "padding": null,
              "added_tokens": [
                {"id":1,"content":"<|begin_of_text|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                {"id":2,"content":"<|eot_id|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                {"id":3,"content":"<|start_header_id|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
                {"id":4,"content":"<|end_header_id|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
              ],
              "normalizer": null,
              "pre_tokenizer": {"type": "Whitespace"},
              "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                  {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
                  {"Sequence": {"id": "A", "type_id": 0}}
                ],
                "pair": [
                  {"Sequence": {"id": "A", "type_id": 0}},
                  {"Sequence": {"id": "B", "type_id": 1}}
                ],
                "special_tokens": {
                  "<|begin_of_text|>": {
                    "id": "<|begin_of_text|>",
                    "ids": [1],
                    "tokens": ["<|begin_of_text|>"]
                  }
                }
              },
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": {
                  "[UNK]": 0,
                  "<|begin_of_text|>": 1,
                  "<|eot_id|>": 2,
                  "<|start_header_id|>": 3,
                  "<|end_header_id|>": 4,
                  "user": 5,
                  "assistant": 6,
                  "Name": 7,
                  "the": 8,
                  "capital": 9,
                  "of": 10,
                  "France": 11,
                  ".": 12
                },
                "unk_token": "[UNK]"
              }
            }"#;

    /// A checkpoint directory carrying `chat_template` verbatim, whose
    /// tokenizer prepends BOS — i.e. one that doubles unless the flag is right.
    ///
    /// `chat_template` is spliced into `tokenizer_config.json` as a JSON string
    /// body, so it must already be JSON-escaped (neither template above needs
    /// escaping: they use single-quoted Jinja literals and no backslashes but
    /// the `\n` the JSON parser is meant to unescape).
    fn checkpoint_with_template(name: &str, chat_template: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!(
            "lb-bos-{name}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();

        // Tier 1 resolves the template AND the special tokens from this file.
        std::fs::write(
            d.join("tokenizer_config.json"),
            format!(
                r#"{{"bos_token":"<|begin_of_text|>",
                    "eos_token":"<|eot_id|>",
                    "chat_template":"{chat_template}"}}"#
            ),
        )
        .unwrap();

        std::fs::write(d.join("tokenizer.json"), FIXTURE_TOKENIZER_JSON).unwrap();
        d
    }

    /// A checkpoint directory whose template emits BOS **and** whose tokenizer
    /// prepends BOS — i.e. one that doubles unless the flag is right.
    fn bos_doubling_checkpoint(name: &str) -> std::path::PathBuf {
        checkpoint_with_template(name, LLAMA3_SHAPED_TEMPLATE)
    }

    /// What [`LLAMA3_SHAPED_TEMPLATE`] must produce for `turns`, written out
    /// by hand.
    ///
    /// Deliberately **not** `template.render(...)`: an expectation computed by
    /// the renderer under test agrees with it however wrong it is, which is how
    /// a prompt fully decoupled from the request (rendering only `msgs[..1]`,
    /// or a hard-coded message list) passes a test that checks only "starts
    /// with BOS".
    fn expected_llama3_render(turns: &[(&str, &str)]) -> String {
        let mut out = String::from(FIXTURE_BOS);
        for (role, content) in turns {
            out.push_str(&format!(
                "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            ));
        }
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        out
    }

    /// What the legacy join must produce for `turns`, written out by hand for
    /// the same reason as [`expected_llama3_render`].
    fn expected_legacy_join(turns: &[(&str, &str)]) -> String {
        turns
            .iter()
            .map(|(role, content)| format!("{role}: {content}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn state_with(
        chat_template: Option<std::sync::Arc<crate::api::chat_template::ResolvedTemplate>>,
    ) -> AppState {
        state_with_runner(chat_template, None)
    }

    fn state_with_runner(
        chat_template: Option<std::sync::Arc<crate::api::chat_template::ResolvedTemplate>>,
        inference_tx: Option<std::sync::mpsc::Sender<crate::engine::model_runner::InferenceJob>>,
    ) -> AppState {
        state_with_monitor(
            chat_template,
            inference_tx,
            // The default window is 20, so no test that does not ask for a
            // narrower one can fill it — which is what keeps the monitor out of
            // every assertion in this module that is not about it.
            std::sync::Arc::new(EosMonitor::default()),
        )
    }

    fn state_with_monitor(
        chat_template: Option<std::sync::Arc<crate::api::chat_template::ResolvedTemplate>>,
        inference_tx: Option<std::sync::mpsc::Sender<crate::engine::model_runner::InferenceJob>>,
        eos_monitor: std::sync::Arc<EosMonitor>,
    ) -> AppState {
        use crate::engine::{MemoryAwareConfig, MemoryAwareScheduler};
        AppState {
            scheduler: std::sync::Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default())),
            config: crate::api::ApiConfig::default(),
            db_pool: None,
            inference_tx,
            chat_template,
            eos_monitor,
        }
    }

    fn one_user_message() -> Vec<ChatMessage> {
        vec![ChatMessage {
            role: "user".to_string(),
            content: "Name the capital of France.".to_string(),
            name: None,
        }]
    }

    /// A templated prompt must reach the model with ONE BOS, not two.
    ///
    /// Fails if `build_prompt` returns `add_special_tokens: true` for a
    /// templated prompt — the state this branch shipped before, under which
    /// Meta-Llama-3-8B-Instruct received `128000, 128000, 128006, …`.
    ///
    /// Verified by mutation: reverting that one field to `true` makes this test
    /// fail and leaves every other test in the repo green. The ids on this
    /// fixture, for the rendered prompt
    /// `"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nName the
    /// capital of France.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"`:
    ///
    ///   flag `true`  (the defect)  `[1, 1, 3, 5, 4, 7, 8, 9, 10, 11, 12, 2, 3, 6, 4]`
    ///   flag `false` (correct)     `[1, 3, 5, 4, 7, 8, 9, 10, 11, 12, 2, 3, 6, 4]`
    ///
    /// One id of difference, at position 0, in a prompt that renders
    /// byte-identically either way — which is why the rendered-text assertions
    /// in `tests/chat_template_render.rs` cannot see this and this one can.
    #[test]
    fn a_templated_prompt_is_tokenized_without_a_second_bos() {
        let dir = bos_doubling_checkpoint("templated");
        let state = state_with(crate::api::chat_template::resolve_for_serving(&dir));
        assert!(
            state.chat_template.is_some(),
            "the fixture's tokenizer_config.json did not resolve a template, so \
             this test would be measuring the legacy join"
        );

        let built = build_prompt(&state, &one_user_message());
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .expect("loading the fixture tokenizer");

        // Control, independent of what production decided: the fixture really
        // does double. Without this, a fixture that quietly stopped prepending
        // BOS would make every assertion below vacuous.
        let doubled: Vec<u32> = tok
            .encode(built.text.as_str(), true)
            .unwrap()
            .get_ids()
            .to_vec();
        assert_eq!(
            &doubled[..2],
            &[FIXTURE_BOS_ID, FIXTURE_BOS_ID],
            "the fixture no longer reproduces the defect: encoding a templated \
             prompt with add_special_tokens=true must yield two BOS ids, got {doubled:?}"
        );
        assert!(
            built.text.starts_with(FIXTURE_BOS),
            "the template stopped emitting bos_token, so the doubling this test \
             guards can no longer occur through it: {:?}",
            built.text
        );

        // The claim.
        assert!(
            !built.add_special_tokens,
            "a prompt rendered through a chat template already carries the \
             checkpoint's BOS; asking the tokenizer for special tokens as well \
             sends the model a token pair it never saw in training"
        );
        let ids: Vec<u32> = tok
            .encode(built.text.as_str(), built.add_special_tokens)
            .unwrap()
            .get_ids()
            .to_vec();
        assert_eq!(
            ids.iter().filter(|&&i| i == FIXTURE_BOS_ID).count(),
            1,
            "the model was sent {} BOS ids: {ids:?}",
            ids.iter().filter(|&&i| i == FIXTURE_BOS_ID).count()
        );
        assert_eq!(ids[0], FIXTURE_BOS_ID, "BOS is not first: {ids:?}");
    }

    /// The inverse, and the reason the flag cannot simply be hardcoded `false`:
    /// an unresolvable checkpoint falls back to the legacy join, whose text
    /// contains no special tokens at all, so the tokenizer must still add them.
    #[test]
    fn an_untemplated_prompt_still_asks_the_tokenizer_for_its_special_tokens() {
        let dir = bos_doubling_checkpoint("untemplated");
        // `None` is the state `ApiServer::new` builds when no tier resolved a
        // template — not a contrived value.
        let state = state_with(None);

        let built = build_prompt(&state, &one_user_message());
        assert_eq!(built.text, "user: Name the capital of France.");
        assert!(
            built.add_special_tokens,
            "the legacy join carries no BOS of its own, so the tokenizer has to \
             supply it — hardcoding the flag `false` would send a Llama model a \
             prompt with no BOS at all"
        );

        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let ids: Vec<u32> = tok
            .encode(built.text.as_str(), built.add_special_tokens)
            .unwrap()
            .get_ids()
            .to_vec();
        assert_eq!(
            ids[0], FIXTURE_BOS_ID,
            "the joined prompt lost its BOS: {ids:?}"
        );
        assert_eq!(
            ids.iter().filter(|&&i| i == FIXTURE_BOS_ID).count(),
            1,
            "{ids:?}"
        );
    }

    /// `/v1/completions` builds raw prompts through this constructor. It is the
    /// only thing standing between that endpoint and a prompt with no BOS.
    #[test]
    fn a_raw_prompt_asks_for_special_tokens() {
        assert!(BuiltPrompt::raw("The capital of France is".to_string()).add_special_tokens);
    }

    // ─── The contract path, at both of its entry points ─────────────────────
    //
    // `build_prompt_from_raw` has two contract call sites — `contracts::
    // executor::execute_contract_with_runner` for direct callers, and
    // `create_chat_completion_with_contract` for a real
    // `POST /v1/chat/completions` carrying `lightbulb.output_contract`. Sharing
    // the builder makes the two agree about *logic*; it does not stop either
    // `build_prompt_from_raw(...)` call from being replaced wholesale with
    // `BuiltPrompt::raw(legacy_join(&msgs))`, which is a one-line revert of this
    // change at whichever site is unguarded. So each site has its own test
    // below, and both go through the shared assertion.

    const CONTRACT_QUESTION: &str = "Name the capital of France.";

    /// The stub runner's reply. It matches no choice and contains no digit, so
    /// `EnumChoice` never accepts it and the loop spends its whole budget.
    const UNPARSEABLE_REPLY: &str = "unparseable";

    fn yes_no_contract() -> crate::contracts::OutputContractSpec {
        crate::contracts::OutputContractSpec::EnumChoice {
            choices: vec!["yes".to_string(), "no".to_string()],
            case_sensitive: false,
            allow_index: true,
        }
    }

    /// A stub model runner on a real thread: records `(prompt,
    /// add_special_tokens)` for every job and answers each with
    /// [`UNPARSEABLE_REPLY`], reporting [`FinishReason::Stop`].
    ///
    /// It returns its recording once every `Sender` clone has been dropped,
    /// which is why each caller below lets the value holding the sender go out
    /// of scope before `join`ing.
    fn stub_runner() -> (
        std::sync::mpsc::Sender<crate::engine::model_runner::InferenceJob>,
        std::thread::JoinHandle<Vec<(String, bool)>>,
    ) {
        stub_runner_with(FinishReason::Stop)
    }

    /// [`stub_runner`] answering every job — streaming or not — with a chosen
    /// finish reason.
    ///
    /// The reason is a parameter because a monitor test whose stub always
    /// reports `Stop` cannot tell a monitor that records the completion from
    /// one that records a constant. Each such test runs both values and asserts
    /// a different rate for each.
    ///
    /// The `Streaming` arm is served rather than dropped: dropping the sender
    /// ends the client's stream with no `Done` frame at all, so the streaming
    /// path would have nothing to observe and a test of it would pass against a
    /// handler that never records.
    fn stub_runner_with(
        finish_reason: FinishReason,
    ) -> (
        std::sync::mpsc::Sender<crate::engine::model_runner::InferenceJob>,
        std::thread::JoinHandle<Vec<(String, bool)>>,
    ) {
        use crate::engine::model_runner::{
            CompletionResult, InferenceJob, ResponseMode, StreamItem,
        };
        let (tx, rx) = std::sync::mpsc::channel::<InferenceJob>();
        let handle = std::thread::spawn(move || {
            let mut jobs: Vec<(String, bool)> = Vec::new();
            while let Ok(job) = rx.recv() {
                jobs.push((job.prompt.clone(), job.add_special_tokens));
                match job.response_mode {
                    ResponseMode::Complete(resp) => {
                        let _ = resp.send(Ok(CompletionResult {
                            text: UNPARSEABLE_REPLY.to_string(),
                            prompt_tokens: 1,
                            completion_tokens: 1,
                            finish_reason,
                        }));
                    }
                    ResponseMode::Streaming(chunks) => {
                        let _ = chunks.send(Ok(StreamItem::Token(UNPARSEABLE_REPLY.to_string())));
                        let _ = chunks.send(Ok(StreamItem::Done { finish_reason }));
                    }
                }
            }
            jobs
        });
        (tx, handle)
    }

    /// The two message lists a two-attempt `yes`/`no` contract run over
    /// [`CONTRACT_QUESTION`] must produce.
    ///
    /// Derived from the documented behaviour of the executor rather than by
    /// observing it: `inject_contract_instruction` prepends one system message
    /// (trimming its leading newline) because the request carries none, and the
    /// retry appends the rejected reply as `assistant` followed by the
    /// tightening correction as `user`. The roles are part of the expectation
    /// on purpose — since this change they select a template branch rather than
    /// decorating a flat join.
    fn expected_contract_message_lists() -> (Vec<(String, String)>, Vec<(String, String)>) {
        let spec = yes_no_contract();
        let system = crate::contracts::validation::build_system_instruction(&spec)
            .trim_start_matches('\n')
            .to_string();
        let tightening = crate::contracts::validation::tightening_message(1, &spec);
        let first = vec![
            ("system".to_string(), system),
            ("user".to_string(), CONTRACT_QUESTION.to_string()),
        ];
        let mut second = first.clone();
        second.push(("assistant".to_string(), UNPARSEABLE_REPLY.to_string()));
        second.push(("user".to_string(), tightening));
        (first, second)
    }

    fn as_turns(msgs: &[(String, String)]) -> Vec<(&str, &str)> {
        msgs.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect()
    }

    /// Assert that a two-attempt contract run reached the runner as two
    /// *different*, fully templated prompts, each tokenized without a second
    /// BOS.
    ///
    /// The content assertions are equalities against a rendering written out by
    /// hand, not `starts_with(FIXTURE_BOS)`: a prompt built from `msgs[..1]`,
    /// or from a hard-coded message list ignoring `msgs` entirely, also starts
    /// with BOS on every attempt while carrying none of the request.
    fn assert_two_templated_attempts(dir: &std::path::Path, jobs: &[(String, bool)]) {
        assert_eq!(jobs.len(), 2, "expected two attempts, saw {}", jobs.len());

        let (first, second) = expected_contract_message_lists();
        let expected = [
            expected_llama3_render(&as_turns(&first)),
            expected_llama3_render(&as_turns(&second)),
        ];

        assert_ne!(
            jobs[1].0, jobs[0].0,
            "both attempts sent the model byte-identical prompts, so the retry's \
             added context — the rejected reply and the tightening correction — \
             never reached it"
        );

        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .expect("loading the fixture tokenizer");

        // Control, independent of what production decided: the fixture really
        // does double. Without this, a fixture whose post_processor stopped
        // prepending BOS would make the encode assertions below vacuous.
        let doubled: Vec<u32> = tok
            .encode(jobs[0].0.as_str(), true)
            .unwrap()
            .get_ids()
            .to_vec();
        assert_eq!(
            &doubled[..2],
            &[FIXTURE_BOS_ID, FIXTURE_BOS_ID],
            "the fixture no longer reproduces the defect: encoding a templated \
             prompt with add_special_tokens=true must yield two BOS ids, got {doubled:?}"
        );

        for (i, (prompt, add_special_tokens)) in jobs.iter().enumerate() {
            assert_eq!(
                prompt, &expected[i],
                "attempt {i} did not reach the model as this request rendered \
                 through the chat template"
            );
            assert!(
                !add_special_tokens,
                "attempt {i} asked the tokenizer for special tokens on top of a \
                 templated prompt, which doubles BOS"
            );
            let ids: Vec<u32> = tok
                .encode(prompt.as_str(), *add_special_tokens)
                .unwrap()
                .get_ids()
                .to_vec();
            assert_eq!(
                ids.iter().filter(|&&t| t == FIXTURE_BOS_ID).count(),
                1,
                "attempt {i} reached the model with {ids:?}"
            );
        }
    }

    /// The contract path renders every attempt through the template, and
    /// tokenizes the result without a second BOS — at the
    /// `execute_contract_with_runner` entry point.
    ///
    /// This test lives here rather than in `contracts::executor` because the
    /// fixture above is what makes it able to fail: TinyLlama resolves to
    /// `registry::ZEPHYR`, which never emits `bos_token`, so a contract test
    /// built on the real checkpoint would pass with the flag hardcoded either
    /// way.
    ///
    /// It asserts on the `InferenceJob`s the entry point actually enqueues —
    /// not on `build_prompt_from_raw` in isolation — because the defect being
    /// guarded is a *call site* restating the flag. A unit test of the builder
    /// alone would stay green if `execute_contract_with_runner` went on writing
    /// `add_special_tokens: true` next to it, which is exactly what that
    /// function did before this change.
    ///
    /// Both attempts are checked, not just the first: the loop re-renders per
    /// attempt against a mutated list, so a second render path is where a
    /// divergence would hide.
    #[tokio::test]
    async fn the_contract_path_renders_each_attempt_without_a_second_bos() {
        use crate::contracts::validation::RawMessage;

        let dir = bos_doubling_checkpoint("contract");
        let template = crate::api::chat_template::resolve_for_serving(&dir);
        assert!(
            template.is_some(),
            "the fixture's tokenizer_config.json did not resolve a template, so \
             this test would be measuring the legacy join"
        );

        let (tx, runner) = stub_runner();
        let msgs = vec![RawMessage {
            role: "user".to_string(),
            content: CONTRACT_QUESTION.to_string(),
        }];
        crate::contracts::executor::execute_contract_with_runner(
            tx,
            template,
            &msgs,
            "fixture",
            &yes_no_contract(),
            2,
            &[],
            8,
            0.0,
        )
        .await
        .expect("the contract loop should return its raw fallback, not an error");

        // Every clone of `tx` is dropped now that the call has returned, so the
        // stub's `recv` ends and it hands back what it saw.
        let jobs = runner.join().expect("stub runner panicked");
        assert_two_templated_attempts(&dir, &jobs);
    }

    /// The same claim at the entry point a real client reaches: the axum
    /// handler, given a deserialized request body carrying
    /// `lightbulb.output_contract`.
    ///
    /// The sibling test above covers `execute_contract_with_runner`, a
    /// convenience wrapper for direct callers. It cannot cover this: the two
    /// share `build_prompt_from_raw`, but sharing a builder only makes the
    /// two agree about the *logic* — either call to it can be replaced
    /// wholesale, and replacing this one is a revert of the entire change at
    /// the only site an HTTP request touches.
    #[tokio::test]
    async fn the_http_contract_path_renders_each_attempt_through_the_template() {
        let dir = bos_doubling_checkpoint("http-contract");
        let template = crate::api::chat_template::resolve_for_serving(&dir);
        assert!(
            template.is_some(),
            "the fixture's tokenizer_config.json did not resolve a template, so \
             this test would be measuring the legacy join"
        );

        let (tx, runner) = stub_runner();
        let state = state_with_runner(template, Some(tx));

        // Deserialized from a body, not constructed field-by-field: the shape
        // below is what a client actually posts.
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "fixture",
            "messages": [{"role": "user", "content": CONTRACT_QUESTION}],
            "max_tokens": 8,
            "temperature": 0.0,
            "lightbulb": {
                "output_contract": {
                    "type": "enum_choice",
                    "choices": ["yes", "no"],
                    "case_sensitive": false,
                    "allow_index": true
                },
                "max_attempts": 2
            }
        }))
        .expect("the request body must deserialize");

        let response = chat_completions(State(state), Json(request))
            .await
            .into_response();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "the contract request did not complete"
        );

        // `state` was moved into the handler and dropped with it, so the stub's
        // `recv` has ended.
        let jobs = runner.join().expect("stub runner panicked");
        assert_two_templated_attempts(&dir, &jobs);
    }

    /// A template that rejects the system role falls back to the legacy join —
    /// with `add_special_tokens: true`, because the join carries no BOS.
    ///
    /// `build_prompt_from_raw`'s `Err` arm is not a rare path on the contract
    /// route. `inject_contract_instruction` puts a system message in *every*
    /// contract request, and Mistral-7B-Instruct-v0.1/v0.2 and Gemma all ship
    /// templates that `raise_exception` on one — so on those checkpoints this
    /// arm runs on every attempt of every contract request, and a wrong flag
    /// there sends a Llama-family model a prompt with no BOS at all.
    #[tokio::test]
    async fn a_template_that_rejects_the_system_role_falls_back_to_the_join() {
        use crate::contracts::validation::RawMessage;

        let dir = checkpoint_with_template("raises-on-system", RAISES_ON_SYSTEM_TEMPLATE);
        let template = crate::api::chat_template::resolve_for_serving(&dir)
            .expect("the fixture's tokenizer_config.json must resolve a template");

        // Controls: the fixture is a working template that fails on exactly the
        // message the contract path adds. Without the first, a template broken
        // in some other way would take the same arm and prove nothing about
        // `raise_exception`; without the second, this test would be measuring
        // the `Ok` arm.
        let question = RawMessage {
            role: "user".to_string(),
            content: CONTRACT_QUESTION.to_string(),
        };
        assert!(
            template.render(std::slice::from_ref(&question)).is_ok(),
            "the fixture template must render a plain user turn"
        );
        assert!(
            template
                .render(&[
                    RawMessage {
                        role: "system".to_string(),
                        content: "Answer with one of: yes, no".to_string(),
                    },
                    question.clone(),
                ])
                .is_err(),
            "the fixture template must raise on a system message, or this test \
             never reaches the fallback arm"
        );

        let (tx, runner) = stub_runner();
        let state = state_with_runner(Some(template), Some(tx));
        let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "fixture",
            "messages": [{"role": "user", "content": CONTRACT_QUESTION}],
            "max_tokens": 8,
            "temperature": 0.0,
            "lightbulb": {
                "output_contract": {
                    "type": "enum_choice",
                    "choices": ["yes", "no"],
                    "case_sensitive": false,
                    "allow_index": true
                },
                "max_attempts": 2
            }
        }))
        .expect("the request body must deserialize");

        let response = chat_completions(State(state), Json(request))
            .await
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);

        let jobs = runner.join().expect("stub runner panicked");
        assert_eq!(jobs.len(), 2, "expected two attempts, saw {}", jobs.len());

        let (first, second) = expected_contract_message_lists();
        let expected = [
            expected_legacy_join(&as_turns(&first)),
            expected_legacy_join(&as_turns(&second)),
        ];
        for (i, (prompt, add_special_tokens)) in jobs.iter().enumerate() {
            assert_eq!(
                prompt, &expected[i],
                "attempt {i} did not fall back to the legacy join over this \
                 request's messages"
            );
            assert!(
                add_special_tokens,
                "attempt {i} fell back to the join, whose text contains no \
                 special tokens, and then told the tokenizer not to add any — \
                 the model receives no BOS at all"
            );
        }

        // And the fallback really is the join, not a template render that
        // happened to succeed.
        assert!(
            !jobs[0].0.contains("[INST]"),
            "the template rendered after all: {:?}",
            jobs[0].0
        );
    }

    // ─── The EOS monitor, at each of the three completion sites ─────────────
    //
    // Every test below runs the stub twice, once reporting `Stop` and once
    // `Length`, and asserts a DIFFERENT rate for each. That pairing is what
    // makes them able to fail: a handler that records nothing reports `None`
    // for both, and one that records a constant reports the same number for
    // both, so neither can be told from a correct handler by a single run.
    //
    // The monitors are built with a window of one or two rather than the
    // default twenty, so a reading exists after one or two completions and can
    // be asserted as an exact value instead of a direction.

    /// A request body carrying no contract and no `stream`, so it takes
    /// `create_chat_completion`'s plain path.
    fn plain_request(stream: bool) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "fixture",
            "messages": [{"role": "user", "content": CONTRACT_QUESTION}],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": stream
        }))
        .expect("the request body must deserialize")
    }

    /// The plain path records the reason the runner reported, not a constant.
    #[tokio::test]
    async fn a_plain_completion_is_recorded_with_the_reason_the_runner_reported() {
        for (reason, expected_rate) in [(FinishReason::Stop, 1.0), (FinishReason::Length, 0.0)] {
            let (tx, runner) = stub_runner_with(reason);
            // Window of one: a single completion is a full reading, so the rate
            // is exactly that completion's finish reason.
            let monitor = std::sync::Arc::new(EosMonitor::new(1, DEFAULT_MIN_STOP_RATE));
            let state = state_with_monitor(None, Some(tx), monitor.clone());

            assert_eq!(
                monitor.stop_rate(),
                None,
                "control: the monitor already held a reading before the request ran"
            );

            let response = chat_completions(State(state), Json(plain_request(false)))
                .await
                .into_response();
            assert_eq!(response.status(), StatusCode::OK);

            // `state` was moved into the handler and dropped with it, so the
            // stub's `recv` has ended.
            let jobs = runner.join().expect("stub runner panicked");
            assert_eq!(jobs.len(), 1, "expected one inference, saw {}", jobs.len());
            assert_eq!(
                monitor.stop_rate(),
                Some(expected_rate),
                "a completion the runner ended with {reason:?} was not what the \
                 monitor recorded"
            );
        }
    }

    /// The streaming path records too.
    ///
    /// This is the site a fix aimed at `create_chat_completion` leaves behind —
    /// the same miss this file already has history with, when it carried two
    /// prompt-construction sites and one of them was fixed. A monitor fed only
    /// by the plain path never fills its window on a server whose clients
    /// stream, and a monitor that never reports is indistinguishable in the log
    /// from a model that is answering correctly.
    #[tokio::test]
    async fn a_streaming_completion_is_recorded_with_the_reason_the_runner_reported() {
        for (reason, expected_rate) in [(FinishReason::Stop, 1.0), (FinishReason::Length, 0.0)] {
            let (tx, runner) = stub_runner_with(reason);
            let monitor = std::sync::Arc::new(EosMonitor::new(1, DEFAULT_MIN_STOP_RATE));
            let state = state_with_monitor(None, Some(tx), monitor.clone());

            // Driven to completion: the observation happens on the `Done` frame,
            // which is the last thing the stream yields.
            let frames = create_chat_stream(state, plain_request(true))
                .collect::<Vec<_>>()
                .await;
            assert_eq!(
                frames.len(),
                2,
                "expected one token frame and one done frame, saw {}",
                frames.len()
            );

            let jobs = runner.join().expect("stub runner panicked");
            assert_eq!(jobs.len(), 1, "expected one inference, saw {}", jobs.len());
            assert_eq!(
                monitor.stop_rate(),
                Some(expected_rate),
                "a streamed completion the runner ended with {reason:?} was not \
                 what the monitor recorded"
            );
        }
    }

    /// The contract path records **once per request**, carrying the finish
    /// reason of the attempt whose text the client receives.
    ///
    /// The window is two and the run makes two requests of two attempts each,
    /// so the count is pinned rather than only the value: a handler recording
    /// per attempt fills the window during the first request and the
    /// mid-run `None` assertion fails. That distinction matters because the
    /// retries are prompts this server constructed, not completions a client
    /// asked for — counting them would let a contract-heavy workload dominate
    /// the rate.
    #[tokio::test]
    async fn a_contract_completion_is_recorded_once_with_its_final_finish_reason() {
        for (reason, expected_rate) in [(FinishReason::Stop, 1.0), (FinishReason::Length, 0.0)] {
            let (tx, runner) = stub_runner_with(reason);
            let monitor = std::sync::Arc::new(EosMonitor::new(2, DEFAULT_MIN_STOP_RATE));
            let state = state_with_monitor(None, Some(tx), monitor.clone());

            let request: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
                "model": "fixture",
                "messages": [{"role": "user", "content": CONTRACT_QUESTION}],
                "max_tokens": 8,
                "temperature": 0.0,
                "lightbulb": {
                    "output_contract": {
                        "type": "enum_choice",
                        "choices": ["yes", "no"],
                        "case_sensitive": false,
                        "allow_index": true
                    },
                    "max_attempts": 2
                }
            }))
            .expect("the request body must deserialize");

            let first = chat_completions(State(state.clone()), Json(request.clone()))
                .await
                .into_response();
            assert_eq!(first.status(), StatusCode::OK);
            assert_eq!(
                monitor.stop_rate(),
                None,
                "one request of two attempts filled a two-wide window, so every \
                 retry this server constructed was counted as a completion"
            );

            let second = chat_completions(State(state), Json(request))
                .await
                .into_response();
            assert_eq!(second.status(), StatusCode::OK);

            let jobs = runner.join().expect("stub runner panicked");
            assert_eq!(
                jobs.len(),
                4,
                "expected two attempts per request over two requests, saw {}",
                jobs.len()
            );
            assert_eq!(
                monitor.stop_rate(),
                Some(expected_rate),
                "two contract requests the runner ended with {reason:?} were not \
                 what the monitor recorded"
            );
        }
    }

    /// **The monitor observes and never acts.**
    ///
    /// A request made while the monitor is warning must be prompted exactly as
    /// one made while it is quiet. This is the claim the whole design rests on:
    /// re-resolving the template, or retrying under another candidate, when the
    /// rate drops was rejected on debuggability (spec §3) — a server that
    /// changes its prompting mid-flight cannot be diagnosed from a
    /// request/response pair.
    ///
    /// Verified by mutation: adding
    /// `if state.eos_monitor.should_warn() { return joined(legacy_join(&raw)); }`
    /// to `build_prompt` — the smallest plausible way for the monitor to start
    /// acting — makes the second request arrive as `"user: Name the capital of
    /// France."` and this test fail, with every other test in the repo green.
    #[tokio::test]
    async fn a_warning_monitor_does_not_change_what_is_prompted() {
        let dir = bos_doubling_checkpoint("monitor-never-acts");
        let template = crate::api::chat_template::resolve_for_serving(&dir);
        assert!(
            template.is_some(),
            "the fixture's tokenizer_config.json did not resolve a template, so \
             this test would be measuring the legacy join"
        );

        // Window of one, and a runner that never emits EOS: the first request
        // alone puts the monitor below the threshold, so the second is made by a
        // server that is already warning.
        let (tx, runner) = stub_runner_with(FinishReason::Length);
        let monitor = std::sync::Arc::new(EosMonitor::new(1, DEFAULT_MIN_STOP_RATE));
        let state = state_with_monitor(template, Some(tx), monitor.clone());

        let first = chat_completions(State(state.clone()), Json(plain_request(false)))
            .await
            .into_response();
        assert_eq!(first.status(), StatusCode::OK);
        assert!(
            monitor.should_warn(),
            "the monitor is not in the warning state after a completion that ran \
             to its token budget, so this test never exercises its claim"
        );

        let second = chat_completions(State(state), Json(plain_request(false)))
            .await
            .into_response();
        assert_eq!(second.status(), StatusCode::OK);

        let jobs = runner.join().expect("stub runner panicked");
        assert_eq!(jobs.len(), 2, "expected two inferences, saw {}", jobs.len());

        // Equality against a rendering written out by hand, not against
        // `jobs[0]`: two requests prompted identically *wrongly* would satisfy
        // `jobs[1] == jobs[0]`.
        let expected = expected_llama3_render(&[("user", CONTRACT_QUESTION)]);
        for (i, (prompt, add_special_tokens)) in jobs.iter().enumerate() {
            assert_eq!(
                prompt, &expected,
                "request {i} was not prompted with this model's template — the \
                 monitor changed what is sent instead of only reporting on it"
            );
            assert!(
                !add_special_tokens,
                "request {i} changed how the prompt is tokenized"
            );
        }
    }
}
