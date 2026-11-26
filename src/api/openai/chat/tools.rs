// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tool-call handling for chat completions: demand detection, prompt strategy,
//! tool-choice enforcement and text-mode call extraction.

use super::*;

/// Whether this request should have tools prompted for and parsed at all.
///
/// `false` for an empty `tools` list (a client that sends `"tools": []` is
/// not asking for anything) and for `tool_choice: "none"`.
pub(super) fn request_wants_tools(request: &ChatCompletionRequest) -> bool {
    request.tools.as_deref().is_some_and(|t| !t.is_empty())
        && !matches!(&request.tool_choice, Some(v) if v.as_str() == Some("none"))
}

/// What `tool_choice` demands of a reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ToolDemand {
    /// `auto` / `none` / absent: the model may answer in prose or call.
    Free,
    /// `"required"`: at least one tool call.
    Any,
    /// `{"type":"function","function":{"name":N}}`: calls to `N` only.
    Named(String),
}

/// The function name in a named `tool_choice` object, if it is one.
pub(super) fn named_tool_choice(v: &serde_json::Value) -> Option<&str> {
    if v.get("type")?.as_str()? != "function" {
        return None;
    }
    v.get("function")?.get("name")?.as_str()
}

pub(super) fn tool_demand(request: &ChatCompletionRequest) -> ToolDemand {
    match &request.tool_choice {
        Some(v) if v.as_str() == Some("required") => ToolDemand::Any,
        Some(v) => named_tool_choice(v).map_or(ToolDemand::Free, |n| ToolDemand::Named(n.into())),
        None => ToolDemand::Free,
    }
}

/// Validate `tool_choice`: `auto`, `none`, `required` (needs `tools`) and a
/// named function that is in `tools` are accepted; anything else is a 400.
///
/// `required` and a named function are **enforced by prompting and checking**,
/// not by constraining decoding (nothing in the serving path can mask tokens):
/// see [`create_chat_completion_honouring_tool_choice`].
pub(super) fn validate_tool_choice(request: &ChatCompletionRequest) -> Result<(), String> {
    let Some(v) = &request.tool_choice else {
        return Ok(());
    };
    if matches!(v.as_str(), Some("auto" | "none")) {
        return Ok(());
    }
    let tools: &[Tool] = request.tools.as_deref().unwrap_or(&[]);
    if v.as_str() == Some("required") {
        return if tools.is_empty() {
            Err("tool_choice \"required\" needs a non-empty `tools` list".to_string())
        } else {
            Ok(())
        };
    }
    match named_tool_choice(v) {
        Some(name)
            if tools
                .iter()
                .any(|Tool::Function { function }| function.name == name) =>
        {
            Ok(())
        }
        Some(name) => Err(format!(
            "tool_choice names \"{name}\", which is not in `tools`"
        )),
        None => Err(format!(
            "tool_choice {v} is not supported; use \"auto\", \"none\", \"required\" or \
             {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}"
        )),
    }
}

/// How many times a reply that ignores `tool_choice` is regenerated.
pub(super) const TOOL_CHOICE_ATTEMPTS: usize = 3;

/// A `tool_choice` of `required` or a named function that the model did not
/// honour in [`TOOL_CHOICE_ATTEMPTS`] tries. Answered with a 502 rather than
/// a prose reply the client asked not to get.
#[derive(Debug)]
pub(crate) struct ToolChoiceUnmet(String);

impl std::fmt::Display for ToolChoiceUnmet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ToolChoiceUnmet {}

pub(super) fn honours_demand(resp: &ChatCompletionResponse, demand: &ToolDemand) -> bool {
    let calls: &[ToolCall] = resp
        .choices
        .first()
        .and_then(|c| c.message.tool_calls.as_deref())
        .unwrap_or(&[]);
    match demand {
        ToolDemand::Free => true,
        ToolDemand::Any => !calls.is_empty(),
        ToolDemand::Named(n) => !calls.is_empty() && calls.iter().all(|c| c.function.name == *n),
    }
}

/// `create_chat_completion` plus `tool_choice` enforcement. The model is told
/// what is required (see `build_tools_instruction`); a reply that still does
/// not comply is regenerated up to [`TOOL_CHOICE_ATTEMPTS`] times, then
/// refused with [`ToolChoiceUnmet`]. A named function is also the only tool
/// the prompt lists. Note that at `temperature` 0 a regeneration repeats the
/// same reply, so the retries only help sampled requests.
pub(super) async fn create_chat_completion_honouring_tool_choice(
    state: AppState,
    mut request: ChatCompletionRequest,
    priority: crate::engine::Priority,
) -> anyhow::Result<ChatCompletionResponse> {
    if let Some(format) = response_format::active(&request) {
        return response_format::run(state, request, priority, format).await;
    }
    let demand = tool_demand(&request);
    if demand == ToolDemand::Free {
        return create_chat_completion(state, request, priority).await;
    }
    if let ToolDemand::Named(n) = &demand {
        if let Some(tools) = request.tools.as_mut() {
            tools.retain(|Tool::Function { function }| function.name == *n);
        }
    }
    for _ in 0..TOOL_CHOICE_ATTEMPTS {
        let resp = create_chat_completion(state.clone(), request.clone(), priority).await?;
        if honours_demand(&resp, &demand) {
            return Ok(resp);
        }
    }
    Err(ToolChoiceUnmet(format!(
        "tool_choice {} was not honoured by the model in {TOOL_CHOICE_ATTEMPTS} attempts",
        request
            .tool_choice
            .as_ref()
            .map_or_else(String::new, |v| v.to_string())
    ))
    .into())
}

/// The most choices one chat request may ask for. Each is a full generation.
pub(super) const MAX_CHAT_N: usize = 8;

/// `n` for `/v1/chat/completions`: `1..=MAX_CHAT_N`, and only `1` with
/// `stream: true` (interleaved per-choice deltas are not implemented).
/// `/v1/completions` still uses [`validate_n`], which accepts only `1`.
pub(super) fn validate_chat_n(n: usize, stream: bool) -> Result<(), String> {
    if n == 0 || n > MAX_CHAT_N {
        return Err(format!("n={n} is out of range: send 1 to {MAX_CHAT_N}"));
    }
    if n > 1 && stream {
        return Err(format!(
            "n={n} with `stream: true` is not supported: streaming one choice at a time is \
             all that is implemented. Send `stream: false` to get {n} choices, or n=1 to stream."
        ));
    }
    Ok(())
}

/// Run the request `n` times concurrently and return one response whose
/// `choices` are the runs' choices (indices `0..n`) and whose `usage` is the
/// sum. Greedy requests (`temperature` 0) give `n` identical choices.
pub(super) async fn create_n_completions(
    state: AppState,
    request: ChatCompletionRequest,
    priority: crate::engine::Priority,
) -> anyhow::Result<ChatCompletionResponse> {
    // With a `seed`, run `i` takes `seed + i`: the choices differ from each other (the same seed would give `n`
    // identical ones) and the whole set reproduces. Without one, each run draws from the driver's own counter.
    let runs = (0..request.n.max(1)).map(|i| {
        let mut run = request.clone();
        run.seed = request.seed.map(|s| s.wrapping_add(i as i64));
        create_chat_completion_honouring_tool_choice(state.clone(), run, priority)
    });
    Ok(merge_choices(futures::future::try_join_all(runs).await?))
}

/// Fold several single-choice responses into the first one. `resps` is non-empty.
pub(super) fn merge_choices(resps: Vec<ChatCompletionResponse>) -> ChatCompletionResponse {
    let mut it = resps.into_iter();
    let mut merged = it.next().expect("at least one response");
    for r in it {
        for mut choice in r.choices {
            choice.index = merged.choices.len();
            merged.choices.push(choice);
        }
        merged.usage = match (merged.usage.take(), r.usage) {
            (Some(a), Some(b)) => Some(Usage {
                prompt_tokens: a.prompt_tokens + b.prompt_tokens,
                completion_tokens: a.completion_tokens + b.completion_tokens,
                total_tokens: a.total_tokens + b.total_tokens,
            }),
            (a, b) => a.or(b),
        };
    }
    merged
}

/// Validate `n` against what Lightbulb actually implements: only `1` (the
/// spec default; a no-op most clients send unconditionally). `n > 1` is
/// rejected with a 400 rather than silently returning one choice.
///
/// Deliberately NOT implemented — real sampling landed (lightbulb#102), so
/// `n` independent runs of the same prompt now genuinely differ, unlike when
/// this rejection was first written. The remaining blocker is response
/// shaping, not determinism: today's non-streaming path returns exactly one
/// `ChatChoice`/`CompletionChoice`; `n` needs `n` of them (one per
/// `run_inference_once` call, indices `0..n`, `usage` summed across all
/// `n`), which is real, straightforward, undone work — sized separately,
/// not folded into whichever PR happens to touch this function next.
pub(crate) fn validate_n(n: usize) -> Result<(), String> {
    if n == 1 {
        return Ok(());
    }
    Err(format!(
        "n={n} is not supported: Lightbulb's non-streaming response shape returns exactly \
         one choice per request today. Running inference {n} times and returning {n} choices \
         is planned but not yet implemented. Send n=1 (the default) until it lands."
    ))
}

/// Truncate `text` at the earliest occurrence of any `stop_sequences` entry,
/// excluding the matched sequence itself from the returned text — matching
/// OpenAI's own behavior (the stop string never appears in the output).
///
/// Returns `(possibly-truncated text, whether a sequence matched)`. The
/// second value is what lets a caller override `finish_reason` to `"stop"`
/// even when the runner's own reason was `Length` — hitting a stop sequence
/// IS a stop, regardless of why the runner itself stopped generating.
pub(crate) fn apply_stop_sequences(text: &str, stop_sequences: &[&str]) -> (String, bool) {
    let earliest = stop_sequences.iter().filter_map(|seq| text.find(seq)).min();
    match earliest {
        Some(pos) => (text[..pos].to_string(), true),
        None => (text.to_string(), false),
    }
}

/// How `tools` reach the model's context for this request.
///
/// A real decision point, not a single hardcoded path — per the tool-calls
/// pivot's condition 2, a checkpoint whose own chat template already
/// branches on a `tools` Jinja variable should eventually be prompted that
/// way instead of via Lightbulb's injected system message, since talking
/// over a model's own trained convention with a second, different one is
/// more likely to confuse it than help. `NativeTemplate` renders `tools`
/// through the checkpoint's own template (`build_native_tools_prompt`) and
/// parses the reply with the per-family parsers; it falls back to `Injected`
/// when that render fails or does not name the tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolPromptStrategy {
    NativeTemplate,
    Injected,
}

/// The prompt for a request, and whether `tools` reached the model through the checkpoint's own template
/// (`true`) or through Lightbulb's injected instruction (`false`, also the answer when there are no tools).
/// The reply is parsed to match: native replies also go through the per-family parsers.
pub(super) fn build_request_prompt(
    state: &AppState,
    request: &ChatCompletionRequest,
    tools: &[Tool],
    wants_tools: bool,
) -> (BuiltPrompt, bool) {
    if !wants_tools {
        return (build_prompt(state, &request.messages), false);
    }
    let demand = tool_demand(request);
    let template = state.model().chat_template.clone();
    if tool_prompt_strategy(template.as_deref()) == ToolPromptStrategy::NativeTemplate {
        if let Some(prompt) = template
            .as_deref()
            .and_then(|t| build_native_tools_prompt(t, &request.messages, tools, &demand))
        {
            return (prompt, true);
        }
        tracing::warn!(
            "the chat template mentions `tools` but did not render them; using the injected tool instruction"
        );
    }
    let spliced = inject_tools_instruction(&request.messages, tools, &demand);
    (build_prompt(state, &spliced), false)
}

/// Render `tools` through the checkpoint's own template. `None` when it fails to render, or renders a prompt that
/// does not name every offered tool: `supports_native_tools` is a substring check, so a template that merely
/// mentions the word would otherwise leave the model never told about its tools.
pub(super) fn build_native_tools_prompt(
    template: &crate::api::chat_template::ResolvedTemplate,
    messages: &[ChatMessage],
    tools: &[Tool],
    demand: &ToolDemand,
) -> Option<BuiltPrompt> {
    let owned;
    let messages = match demand {
        ToolDemand::Free => messages,
        ToolDemand::Any => {
            owned = inject_system_text(
                messages,
                "You MUST call a tool; do not answer in prose.".into(),
            );
            &owned
        }
        ToolDemand::Named(n) => {
            owned = inject_system_text(
                messages,
                format!("You MUST call the tool \"{n}\"; do not answer in prose."),
            );
            &owned
        }
    };
    let text = template
        .render_json_with_tools(
            &template_messages(messages),
            &super::super::tool_parsers::tools_for_template(tools),
        )
        .ok()?;
    let names_every_tool = tools
        .iter()
        .all(|Tool::Function { function }| text.contains(&function.name));
    names_every_tool.then(|| built_from_rendered(template, text))
}

/// The conversation in the HF shape the checkpoint's template reads: `role` and `content`, and, where the message
/// has them, an assistant's `tool_calls` (arguments as an object, as `transformers` passes them, falling back to
/// the string when it is not JSON) and a tool result's `tool_call_id` and `name`.
pub(super) fn template_messages(messages: &[ChatMessage]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|m| {
            let mut v = serde_json::json!({ "role": m.role, "content": m.content });
            if let Some(calls) = &m.tool_calls {
                v["tool_calls"] = calls
                    .iter()
                    .map(|c| {
                        let arguments =
                            serde_json::from_str(&c.function.arguments).unwrap_or_else(|_| {
                                serde_json::Value::String(c.function.arguments.clone())
                            });
                        serde_json::json!({
                            "id": c.id,
                            "type": c.kind,
                            "function": { "name": c.function.name, "arguments": arguments },
                        })
                    })
                    .collect();
            }
            if let Some(id) = &m.tool_call_id {
                v["tool_call_id"] = serde_json::json!(id);
            }
            if let Some(name) = &m.name {
                v["name"] = serde_json::json!(name);
            }
            v
        })
        .collect()
}

/// Tool calls in a reply: Lightbulb's own `<tool_call>{json}</tool_call>` tag always, and, when the tools were
/// rendered by the checkpoint's own template (`native`), the families' formats too (see `tool_parsers`).
pub(super) fn extract_calls(
    text: &str,
    tools: &[Tool],
    native: bool,
) -> Option<(String, Vec<ToolCall>)> {
    extract_tool_calls(text).or_else(|| {
        if !native {
            return None;
        }
        let (leading, calls) = super::super::tool_parsers::parse_family_calls(text, tools)?;
        let calls = calls
            .into_iter()
            .map(|c| ToolCall {
                id: format!("call_{}", uuid::Uuid::new_v4()),
                kind: "function".to_string(),
                function: ToolCallFunction {
                    name: c.name,
                    arguments: c.arguments.to_string(),
                },
            })
            .collect();
        Some((leading, calls))
    })
}

pub(super) fn tool_prompt_strategy(
    template: Option<&crate::api::chat_template::ResolvedTemplate>,
) -> ToolPromptStrategy {
    match template {
        Some(t) if t.template.supports_native_tools() => ToolPromptStrategy::NativeTemplate,
        _ => ToolPromptStrategy::Injected,
    }
}

/// Build the instruction text describing `tools` and the exact tag Lightbulb
/// asks the model to answer with.
pub(super) fn build_tools_instruction(tools: &[Tool], demand: &ToolDemand) -> String {
    let mut s = String::from(
        "You have access to the following tools. When you need to call one, \
         respond with exactly one line per call, in this form and no other:\n",
    );
    s.push_str(TOOL_CALL_TAG_START);
    s.push_str(r#"{"name": "<tool name>", "arguments": {<argument object>}}"#);
    s.push_str(TOOL_CALL_TAG_END);
    s.push_str("\nDo not use any other format.");
    match demand {
        ToolDemand::Free => {
            s.push_str(" If no tool call is needed, answer normally without that tag.")
        }
        ToolDemand::Any => s.push_str(" You MUST call a tool; do not answer in prose."),
        ToolDemand::Named(n) => s.push_str(&format!(
            " You MUST call the tool \"{n}\"; do not answer in prose."
        )),
    }
    s.push_str("\n\nAvailable tools:\n");
    for tool in tools {
        let Tool::Function { function } = tool;
        s.push_str("- ");
        s.push_str(&function.name);
        if let Some(desc) = &function.description {
            s.push_str(": ");
            s.push_str(desc);
        }
        if let Some(params) = &function.parameters {
            s.push_str("\n  arguments schema: ");
            s.push_str(&params.to_string());
        }
        s.push('\n');
    }
    s
}

/// Splice the tools instruction into `messages`.
///
/// **Merge, not duplicate** (tool-calls pivot condition 3): a caller's own
/// existing system message gets the instruction appended to it rather than
/// Lightbulb inserting a second system turn. Several templates on hand,
/// TinyLlama's included, render exactly one system slot; two would either
/// collide in the render or silently lose one. If the caller sent no system
/// message, one is inserted at the front. This changes the conversation the
/// caller sent — documented here and in the PR description precisely so it
/// is observable rather than a surprise.
pub(super) fn inject_tools_instruction(
    messages: &[ChatMessage],
    tools: &[Tool],
    demand: &ToolDemand,
) -> Vec<ChatMessage> {
    inject_system_text(messages, build_tools_instruction(tools, demand))
}

/// Append `instruction` to the first system message, or insert a system
/// message holding it at the front (see [`inject_tools_instruction`]).
pub(super) fn inject_system_text(
    messages: &[ChatMessage],
    instruction: String,
) -> Vec<ChatMessage> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    let mut merged = false;
    for m in messages {
        if !merged && m.role == "system" {
            out.push(ChatMessage {
                role: m.role.clone(),
                content: format!("{}\n\n{}", m.content, instruction),
                name: m.name.clone(),
                tool_calls: m.tool_calls.clone(),
                tool_call_id: m.tool_call_id.clone(),
            });
            merged = true;
        } else {
            out.push(m.clone());
        }
    }
    if !merged {
        out.insert(
            0,
            ChatMessage {
                role: "system".to_string(),
                content: instruction,
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        );
    }
    out
}

/// Extract Lightbulb's own `<tool_call>{json}</tool_call>` tool calls from
/// complete generated text, returning the leading (pre-tag) text as the
/// message's real `content` alongside the parsed calls.
///
/// Deliberately independent of the deleted `engine::tool_call` streaming
/// detector (see the tool-calls pivot, condition 4): this runs once, after
/// generation has finished, over the whole text. It parses exactly one
/// format because the tag is Lightbulb's own choice, not a family convention
/// being reverse-engineered — there is nothing to key per checkpoint here,
/// unlike the per-family parser the original design doc costed for Q2.
///
/// Returns `None` when no well-formed tag is found — the ordinary "answered
/// normally" case. A tag that opens but never closes, or whose body is not
/// the expected JSON object, is also `None` rather than an error: a small
/// model asked to comply with a new format may simply decline to, and that
/// is a model-capability finding, not a parser bug (tool-calls pivot,
/// condition 1).
pub(super) fn extract_tool_calls(text: &str) -> Option<(String, Vec<ToolCall>)> {
    let mut calls = Vec::new();
    let mut leading: Option<String> = None;
    let mut rest = text;
    while let Some(start) = rest.find(TOOL_CALL_TAG_START) {
        if leading.is_none() {
            leading = Some(rest[..start].to_string());
        }
        let after_start = &rest[start + TOOL_CALL_TAG_START.len()..];
        let Some(end) = after_start.find(TOOL_CALL_TAG_END) else {
            break;
        };
        let body = after_start[..end].trim();
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(body)
            && let Some(name) = v.get("name").and_then(|n| n.as_str())
        {
            let arguments = v
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
            calls.push(ToolCall {
                id: format!("call_{}", uuid::Uuid::new_v4()),
                kind: "function".to_string(),
                function: ToolCallFunction {
                    name: name.to_string(),
                    arguments: arguments.to_string(),
                },
            });
        }
        rest = &after_start[end + TOOL_CALL_TAG_END.len()..];
    }
    if calls.is_empty() {
        None
    } else {
        Some((leading.unwrap_or_default().trim().to_string(), calls))
    }
}
