//! Generic contract executor — the retry/fallback loop.
//!
//! This module contains `execute_contract`, which drives the full
//! structured-output loop:
//!
//! 1. Inject contract instructions into the message list.
//! 2. Hand the message list to the inference closure, which renders it.
//! 3. Try to parse the response; if it fails, append model output as context,
//!    add a tightening message, and retry up to `max_attempts` times.
//! 4. On primary contract exhaustion, repeat with each fallback contract.
//! 5. Return a [`ContractExecutionResult`] either way.
//!
//! # This module does not know how a prompt is formatted
//!
//! Step 2 hands over `Vec<RawMessage>`, not a `String`. It used to call
//! `validation::messages_to_prompt`, a third copy of the legacy
//! `role: content` join, which meant every contract request prompted a chat
//! model as though it were a base model no matter what template the checkpoint
//! shipped. Rendering upstream instead is not available to this loop: it
//! *mutates* the message list between attempts (the tightening message and the
//! rejected assistant reply are both pushed below), so a prompt rendered once
//! by the caller would be stale from attempt 2 onward. Passing the list and
//! letting the caller render each attempt is what let `messages_to_prompt` be
//! deleted rather than re-pointed — and re-pointing it would have left a
//! second renderer, which is a second place to get
//! `InferenceJob::add_special_tokens` wrong.
//!
//! # Testability
//!
//! The inference callable is a generic `Fn(Vec<RawMessage>) -> impl
//! Future<Output = anyhow::Result<CompletionResult>>`.  Production code passes
//! a closure that renders through the model's chat template and wraps the
//! `mpsc::Sender` to the model runner thread.  Tests pass a closure that
//! returns scripted results, so every branch of this loop can be exercised
//! without a real model.

use std::future::Future;

use super::validation::{self, RawMessage};
use super::{ContractOutput, ContractResult, OutputContractSpec};
use crate::engine::model_runner::{CompletionResult, FinishReason};

// ─── Convenience wrapper for ModelRunner-backed inference ──────────────────

/// Run `execute_contract` using a live [`ModelRunner`] sender.
///
/// This is the public entry point for code that already holds an
/// `mpsc::Sender<InferenceJob>` (e.g. integration tests, CLI tools) and
/// doesn't want to wire up the closure manually.
///
/// `template` is the checkpoint's resolved chat template, exactly as
/// `AppState.chat_template` holds it: `None` when no tier produced one, in
/// which case each attempt falls back to the legacy join. It is a parameter
/// rather than something this function resolves for itself because resolution
/// reads files and its answer cannot change while the process runs — the HTTP
/// server resolves once at startup, and a caller here should pass the same
/// value rather than re-deriving a possibly different one.
///
/// [`ModelRunner`]: crate::engine::model_runner::ModelRunner
pub async fn execute_contract_with_runner(
    tx: std::sync::mpsc::Sender<crate::engine::model_runner::InferenceJob>,
    template: Option<std::sync::Arc<crate::api::chat_template::ResolvedTemplate>>,
    messages: &[RawMessage],
    model_id: &str,
    primary: &OutputContractSpec,
    max_attempts: u32,
    fallbacks: &[OutputContractSpec],
    max_new_tokens: usize,
    temperature: f64,
) -> anyhow::Result<ContractExecutionResult> {
    execute_contract(
        messages,
        model_id,
        primary,
        max_attempts,
        fallbacks,
        move |msgs: Vec<RawMessage>| {
            let tx = tx.clone();
            // The same builder `/v1/chat/completions` uses, which is the point:
            // `add_special_tokens` is a property of the prompt, and letting
            // this site decide it separately is how the two answers drift. A
            // templated prompt already carries the checkpoint's BOS, so `true`
            // here would send the model two — see `chat::build_prompt_from_raw`.
            let prompt =
                crate::api::openai::chat::build_prompt_from_raw(template.as_deref(), &msgs);
            async move {
                use crate::engine::model_runner::{InferenceJob, ResponseMode};
                let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
                let job = InferenceJob {
                    id: uuid::Uuid::new_v4().to_string(),
                    prompt: prompt.text,
                    max_new_tokens,
                    temperature,
                    add_special_tokens: prompt.add_special_tokens,
                    response_mode: ResponseMode::Complete(resp_tx),
                };
                tx.send(job)
                    .map_err(|e| anyhow::anyhow!("failed to send inference job: {}", e))?;
                resp_rx
                    .await
                    .map_err(|e| anyhow::anyhow!("inference runner dropped: {}", e))?
            }
        },
    )
    .await
}

// ─── Public result type ────────────────────────────────────────────────────

/// Everything the caller needs after the loop is done.
///
/// # How the aggregate metadata is aggregated
///
/// One call to [`execute_contract`] can run many inferences: up to
/// `max_attempts` for the primary contract, then up to `max_attempts` again
/// for each fallback. The token counts and the finish reason are therefore
/// combined in **deliberately different** ways, and neither rule is derivable
/// from the types:
///
/// - **`prompt_tokens` and `completion_tokens` are SUMMED over every attempt**,
///   including attempts whose text was thrown away. Each retry sent a real
///   prompt through a real model and got real tokens back; reporting only the
///   final attempt would under-report work the server actually performed —
///   the same class of untruth the rest of this change set exists to remove.
///   Divide by [`ContractResult::attempts`] if a per-attempt average is wanted.
/// - **`finish_reason` is the FINAL attempt's**, not a merge and not the first
///   one's. It describes `final_text` — the text that becomes
///   `choices[0].message.content`. A reason inherited from a discarded attempt
///   would describe output the client never sees, and there is no meaningful
///   way to combine `Stop` and `Length` into one answer anyway.
///
/// [`ContractResult::attempts`]: super::ContractResult::attempts
#[derive(Debug)]
pub struct ContractExecutionResult {
    /// The structured (or raw-fallback) output.
    pub result: ContractResult,
    /// The final raw text from the model (used as the `choices[0].content`
    /// in the HTTP response).
    pub final_text: String,
    /// Prompt tokens **summed across every attempt**. See the type docs.
    pub prompt_tokens: usize,
    /// Generated tokens **summed across every attempt**. See the type docs.
    pub completion_tokens: usize,
    /// Why the **final** attempt stopped. See the type docs.
    ///
    /// `None` only when no inference ran at all (`max_attempts == 0`), where
    /// there is no generation for a reason to describe.
    pub finish_reason: Option<FinishReason>,
}

// ─── Core loop ─────────────────────────────────────────────────────────────

/// Drive the contract retry/fallback loop.
///
/// # Parameters
///
/// - `messages`     — the caller's original message list (will be cloned /
///                    mutated per-contract internally; caller's copy is untouched).
/// - `model_id`     — used only for logging and the returned `ContractResult`.
/// - `primary`      — the first contract to attempt.
/// - `max_attempts` — how many inference calls to make per contract before
///                    moving to the next one.
/// - `fallbacks`    — additional contracts tried (in order) after `primary`
///                    exhausts all attempts.
/// - `infer`        — async callable `(msgs: Vec<RawMessage>) ->
///                    Result<CompletionResult>`. Called once per attempt, with
///                    **that attempt's** message list; the caller renders it.
///                    See the module docs for why the list, not a prompt.
///
/// Each `infer` result's `text` drives contract validation; its token counts
/// and finish reason are aggregated into the returned
/// [`ContractExecutionResult`] under the rules documented on that type.
pub async fn execute_contract<F, Fut>(
    messages: &[RawMessage],
    model_id: &str,
    primary: &OutputContractSpec,
    max_attempts: u32,
    fallbacks: &[OutputContractSpec],
    infer: F,
) -> anyhow::Result<ContractExecutionResult>
where
    F: Fn(Vec<RawMessage>) -> Fut,
    Fut: Future<Output = anyhow::Result<CompletionResult>>,
{
    let start = std::time::Instant::now();
    let mut total_attempts: u32 = 0;
    let mut last_raw = String::new();

    // Aggregates. Tokens accumulate over every attempt; the finish reason is
    // overwritten each attempt so it ends up holding the last one's.
    let mut prompt_tokens: usize = 0;
    let mut completion_tokens: usize = 0;
    let mut finish_reason: Option<FinishReason> = None;

    let contracts: Vec<&OutputContractSpec> =
        std::iter::once(primary).chain(fallbacks.iter()).collect();

    for contract in &contracts {
        // Fresh message list for each contract so fallbacks start clean.
        let mut msgs: Vec<RawMessage> = messages.to_vec();

        validation::inject_contract_instruction(&mut msgs, contract);

        for attempt in 0..max_attempts {
            total_attempts += 1;

            if attempt > 0 {
                msgs.push(RawMessage {
                    role: "user".to_string(),
                    content: validation::tightening_message(attempt, contract),
                });
            }

            let attempt_start = std::time::Instant::now();
            // The caller renders — it owns the template. Cloned rather than
            // borrowed: a borrowing callback returning a `Future` needs an HRTB
            // that buys nothing here, and this list is already a per-contract
            // copy of the caller's.
            let completion = infer(msgs.clone()).await?;
            let attempt_ms = attempt_start.elapsed().as_millis() as u64;

            // This attempt's tokens were spent whether or not its text is kept.
            prompt_tokens += completion.prompt_tokens;
            completion_tokens += completion.completion_tokens;
            // Last write wins, so this ends as the final attempt's reason.
            finish_reason = Some(completion.finish_reason);

            let raw_text = completion.text;
            last_raw = raw_text.clone();

            let parsed = validation::try_parse(&raw_text, contract);
            let parse_ok = parsed
                .as_ref()
                .map(|o| validation::is_parse_successful(o, contract))
                .unwrap_or(false);

            validation::log_attempt(
                contract.type_name(),
                total_attempts,
                parse_ok,
                model_id,
                attempt_ms,
            );

            if parse_ok {
                let output = parsed.unwrap();
                let latency_ms = start.elapsed().as_millis() as u64;
                return Ok(ContractExecutionResult {
                    final_text: raw_text,
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                    result: ContractResult {
                        contract_type: contract.type_name().to_string(),
                        attempts: total_attempts,
                        parse_success: true,
                        model_id: model_id.to_string(),
                        latency_ms,
                        output,
                    },
                });
            }

            // Append bad assistant response as context before the next retry.
            if attempt + 1 < max_attempts {
                msgs.push(RawMessage {
                    role: "assistant".to_string(),
                    content: raw_text,
                });
            }
        }
    }

    // Everything failed.
    let latency_ms = start.elapsed().as_millis() as u64;
    tracing::error!(
        model = model_id,
        total_attempts,
        latency_ms,
        "all contract attempts failed; returning raw fallback"
    );

    Ok(ContractExecutionResult {
        final_text: last_raw.clone(),
        prompt_tokens,
        completion_tokens,
        finish_reason,
        result: ContractResult {
            contract_type: primary.type_name().to_string(),
            attempts: total_attempts,
            parse_success: false,
            model_id: model_id.to_string(),
            latency_ms,
            output: ContractOutput::Raw { text: last_raw },
        },
    })
}

// ─── Aggregation tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    /// An inference stub that replays canned [`CompletionResult`]s in order.
    fn scripted(
        results: Vec<CompletionResult>,
    ) -> impl Fn(Vec<RawMessage>) -> std::future::Ready<anyhow::Result<CompletionResult>> {
        let queue = Arc::new(Mutex::new(VecDeque::from(results)));
        move |_msgs: Vec<RawMessage>| {
            let next = queue
                .lock()
                .unwrap()
                .pop_front()
                .expect("execute_contract called infer more times than scripted");
            std::future::ready(Ok(next))
        }
    }

    fn completion(
        text: &str,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: FinishReason,
    ) -> CompletionResult {
        CompletionResult {
            text: text.to_string(),
            prompt_tokens,
            completion_tokens,
            finish_reason,
        }
    }

    fn user_msgs(content: &str) -> Vec<RawMessage> {
        vec![RawMessage {
            role: "user".to_string(),
            content: content.to_string(),
        }]
    }

    fn yes_no() -> OutputContractSpec {
        OutputContractSpec::EnumChoice {
            choices: vec!["yes".to_string(), "no".to_string()],
            case_sensitive: false,
            allow_index: true,
        }
    }

    /// Success-on-retry path. Guards both aggregation rules at once:
    /// summing the discarded first attempt's tokens, and reporting the
    /// *second* attempt's finish reason. Deliberately gives the two attempts
    /// different reasons so a "first attempt wins" bug cannot pass.
    #[tokio::test]
    async fn sums_usage_over_attempts_and_reports_final_finish_reason() {
        let exec = execute_contract(
            &user_msgs("Is it raining?"),
            "test-model",
            &yes_no(),
            3,
            &[],
            scripted(vec![
                completion("I cannot decide.", 11, 7, FinishReason::Length),
                completion("yes", 23, 3, FinishReason::Stop),
            ]),
        )
        .await
        .unwrap();

        assert_eq!(
            exec.result.attempts, 2,
            "test is only meaningful if two attempts actually ran"
        );
        assert_eq!(
            exec.prompt_tokens,
            11 + 23,
            "prompt tokens must be summed across attempts, not taken from the last"
        );
        assert_eq!(
            exec.completion_tokens,
            7 + 3,
            "completion tokens must be summed across attempts, not taken from the last"
        );
        assert_eq!(
            exec.finish_reason,
            Some(FinishReason::Stop),
            "finish reason must come from the final attempt, not the discarded first"
        );
        assert_eq!(exec.final_text, "yes");
    }

    /// Total-failure path — the other `ContractExecutionResult` construction
    /// site. Spans the primary contract and a fallback so the sum crosses the
    /// contract boundary too.
    #[tokio::test]
    async fn aggregates_across_contracts_on_total_failure() {
        let fallback = OutputContractSpec::EnumChoice {
            choices: vec!["positive".to_string(), "negative".to_string()],
            case_sensitive: false,
            allow_index: true,
        };

        let exec = execute_contract(
            &user_msgs("Pick one."),
            "test-model",
            &yes_no(),
            2,
            &[fallback],
            scripted(vec![
                // Every term distinct on BOTH axes. Four equal
                // `completion_tokens` would let `attempts * last` produce the
                // right total, so the assertion could not tell summing from
                // multiplying — the same class of hole this branch has been
                // closing all along.
                completion("I cannot say.", 10, 5, FinishReason::Stop),
                completion("Still unclear.", 20, 6, FinishReason::Stop),
                completion("Hmm.", 30, 7, FinishReason::Stop),
                completion("Truncated mid-thoug", 40, 8, FinishReason::Length),
            ]),
        )
        .await
        .unwrap();

        assert!(!exec.result.parse_success);
        assert_eq!(exec.result.attempts, 4, "2 attempts x 2 contracts");
        assert_eq!(exec.prompt_tokens, 10 + 20 + 30 + 40);
        assert_eq!(exec.completion_tokens, 5 + 6 + 7 + 8);
        assert_eq!(
            exec.finish_reason,
            Some(FinishReason::Length),
            "the raw fallback returns the last attempt's text, so it must \
             carry the last attempt's reason"
        );
        assert_eq!(exec.final_text, "Truncated mid-thoug");
    }
}
