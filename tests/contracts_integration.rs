//! Integration tests for the Output Contracts system.
//!
//! These tests exercise `execute_contract` end-to-end — prompt injection,
//! parsing, retry counting, fallback chaining, and total failure handling —
//! using a **mock inference closure** instead of a real model.  Every branch
//! of the retry loop is covered.
//!
//! The mock works by pre-loading a queue of scripted responses.  Each call to
//! the closure pops the next response off the queue, simulating a model that
//! replies differently on successive attempts (e.g., garbage first, then the
//! correct label).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use lightbulb::contracts::{
    ContractOutput, OutputContractSpec, executor::execute_contract, validation::RawMessage,
};
use lightbulb::engine::model_runner::{CompletionResult, FinishReason};

// ─── Mock helpers ──────────────────────────────────────────────────────────

/// Wrap scripted text as a [`CompletionResult`], which is what the inference
/// closure yields.
///
/// The token counts and finish reason are placeholders: these tests are about
/// parsing, retry counting and fallback chaining, and assert nothing about
/// metadata. How `execute_contract` aggregates that metadata across attempts
/// is covered by the unit tests in `src/contracts/executor.rs`.
fn canned(text: impl Into<String>) -> CompletionResult {
    CompletionResult {
        text: text.into(),
        prompt_tokens: 0,
        completion_tokens: 0,
        finish_reason: FinishReason::Stop,
    }
}

/// Create an async inference closure that returns scripted responses in order.
/// Panics if called more times than there are scripted responses.
fn mock_infer(
    responses: Vec<&'static str>,
) -> (
    Arc<Mutex<VecDeque<String>>>,
    impl Fn(Vec<RawMessage>) -> std::future::Ready<anyhow::Result<CompletionResult>> + Clone,
) {
    let queue: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(
        responses.iter().map(|s| s.to_string()).collect(),
    ));
    let q2 = queue.clone();
    let f = move |_msgs: Vec<RawMessage>| {
        let text = q2
            .lock()
            .unwrap()
            .pop_front()
            .expect("mock_infer called more times than scripted responses");
        std::future::ready(Ok::<CompletionResult, anyhow::Error>(canned(text)))
    };
    (queue, f)
}

/// A simple user-only message list.
fn user_messages(content: &str) -> Vec<RawMessage> {
    vec![RawMessage {
        role: "user".to_string(),
        content: content.to_string(),
    }]
}

fn choices() -> Vec<String> {
    vec!["yes".to_string(), "no".to_string(), "maybe".to_string()]
}

fn allowed_tags() -> Vec<String> {
    vec!["MODE".to_string(), "CONF".to_string()]
}

// ─── EnumChoice tests ──────────────────────────────────────────────────────

/// Happy path: model returns the correct label on the very first call.
#[tokio::test]
async fn enum_choice_succeeds_first_try() {
    let spec = OutputContractSpec::EnumChoice {
        choices: choices(),
        case_sensitive: false,
        allow_index: true,
    };
    let (_, infer) = mock_infer(vec!["After careful thought, the answer is YES"]);

    let result = execute_contract(
        &user_messages("Is the service healthy?"),
        "test-model",
        &spec,
        3,
        &[],
        infer,
    )
    .await
    .unwrap();

    assert!(result.result.parse_success, "should parse successfully");
    assert_eq!(result.result.attempts, 1, "should need only one attempt");
    assert!(
        matches!(&result.result.output, ContractOutput::EnumChoice { choice } if choice == "yes")
    );
}

/// Model replies with garbage on attempt 1, then gives the label on attempt 2.
#[tokio::test]
async fn enum_choice_retries_on_garbage() {
    let spec = OutputContractSpec::EnumChoice {
        choices: choices(),
        case_sensitive: false,
        allow_index: true,
    };
    let (_, infer) = mock_infer(vec![
        "I am unable to determine the answer at this time.",
        "no",
    ]);

    let result = execute_contract(
        &user_messages("Is it raining?"),
        "test-model",
        &spec,
        3,
        &[],
        infer,
    )
    .await
    .unwrap();

    assert!(result.result.parse_success);
    assert_eq!(result.result.attempts, 2, "should need two attempts");
    assert!(
        matches!(&result.result.output, ContractOutput::EnumChoice { choice } if choice == "no")
    );
}

/// Primary contract (3 labels) exhausts all attempts; fallback (2 labels) succeeds.
#[tokio::test]
async fn enum_choice_falls_back_to_simpler_contract() {
    let primary = OutputContractSpec::EnumChoice {
        choices: choices(),
        case_sensitive: false,
        allow_index: true,
    };
    let fallback = OutputContractSpec::EnumChoice {
        choices: vec!["positive".to_string(), "negative".to_string()],
        case_sensitive: false,
        allow_index: true,
    };

    // Attempts 1-3: garbage (primary fails); attempt 4: correct fallback label.
    let (_, infer) = mock_infer(vec![
        "I cannot say.",
        "It depends on many factors.",
        "Unknown.",
        "The sentiment is clearly POSITIVE.",
    ]);

    let result = execute_contract(
        &user_messages("What is the sentiment?"),
        "test-model",
        &primary,
        3,
        &[fallback],
        infer,
    )
    .await
    .unwrap();

    assert!(result.result.parse_success);
    assert_eq!(result.result.attempts, 4);
    assert_eq!(result.result.contract_type, "enum_choice");
    assert!(
        matches!(&result.result.output, ContractOutput::EnumChoice { choice } if choice == "positive")
    );
}

/// All attempts for all contracts fail → `parse_success = false`, `Raw` output returned.
#[tokio::test]
async fn enum_choice_total_failure_returns_raw() {
    let spec = OutputContractSpec::EnumChoice {
        choices: choices(),
        case_sensitive: false,
        allow_index: true,
    };
    let (_, infer) = mock_infer(vec!["I don't know.", "Unclear.", "Cannot say."]);

    let result = execute_contract(
        &user_messages("Pick one."),
        "test-model",
        &spec,
        3,
        &[],
        infer,
    )
    .await
    .unwrap();

    assert!(!result.result.parse_success, "should report failure");
    assert_eq!(result.result.attempts, 3, "should exhaust all attempts");
    assert!(matches!(result.result.output, ContractOutput::Raw { .. }));
    // The final_text should be the last raw model response.
    assert_eq!(result.final_text, "Cannot say.");
}

/// Retry count respects `max_attempts = 1` — no retries, just one call.
#[tokio::test]
async fn enum_choice_respects_max_attempts_one() {
    let spec = OutputContractSpec::EnumChoice {
        choices: choices(),
        case_sensitive: false,
        allow_index: true,
    };
    let (_, infer) = mock_infer(vec!["garbage (only one allowed)"]);

    let result = execute_contract(
        &user_messages("Pick one."),
        "test-model",
        &spec,
        1,
        &[],
        infer,
    )
    .await
    .unwrap();

    assert!(!result.result.parse_success);
    assert_eq!(result.result.attempts, 1);
}

// ─── TaggedFields tests ────────────────────────────────────────────────────

/// Model returns all required tags correctly on the first try.
#[tokio::test]
async fn tagged_fields_succeeds_first_try() {
    let spec = OutputContractSpec::TaggedFields {
        allowed_tags: allowed_tags(),
        required_tags: vec![],
        repeatable_tags: vec![],
    };
    let (_, infer) = mock_infer(vec!["<MODE>typecheck_error</MODE>\n<CONF>0.85</CONF>"]);

    let result = execute_contract(
        &user_messages("Diagnose this error."),
        "test-model",
        &spec,
        3,
        &[],
        infer,
    )
    .await
    .unwrap();

    assert!(result.result.parse_success);
    assert_eq!(result.result.attempts, 1);
    if let ContractOutput::TaggedFields { tags } = &result.result.output {
        assert_eq!(
            tags.get("MODE").and_then(|v| v.first()).map(String::as_str),
            Some("typecheck_error")
        );
        assert_eq!(
            tags.get("CONF").and_then(|v| v.first()).map(String::as_str),
            Some("0.85")
        );
    } else {
        panic!("expected TaggedFields output");
    }
}

/// Model omits tags on first attempt, provides them on retry.
#[tokio::test]
async fn tagged_fields_retries_when_tags_missing() {
    let spec = OutputContractSpec::TaggedFields {
        allowed_tags: allowed_tags(),
        required_tags: vec![],
        repeatable_tags: vec![],
    };
    let (_, infer) = mock_infer(vec![
        "The error seems to be a runtime issue with moderate confidence.",
        "<MODE>runtime_error</MODE><CONF>0.60</CONF>",
    ]);

    let result = execute_contract(
        &user_messages("Diagnose this error."),
        "test-model",
        &spec,
        3,
        &[],
        infer,
    )
    .await
    .unwrap();

    assert!(result.result.parse_success);
    assert_eq!(result.result.attempts, 2);
}

/// `require_all = true` fails when only one of two fields is present,
/// then succeeds when both are present on retry.
#[tokio::test]
async fn tagged_fields_require_all_retries_until_complete() {
    let spec = OutputContractSpec::TaggedFields {
        allowed_tags: allowed_tags(),
        required_tags: allowed_tags(),
        repeatable_tags: vec![],
    };
    let (_, infer) = mock_infer(vec![
        "<MODE>lint_error</MODE>",                  // only MODE — fails require_all
        "<MODE>lint_error</MODE><CONF>0.92</CONF>", // both fields — passes
    ]);

    let result = execute_contract(
        &user_messages("Diagnose this."),
        "test-model",
        &spec,
        3,
        &[],
        infer,
    )
    .await
    .unwrap();

    assert!(result.result.parse_success);
    assert_eq!(result.result.attempts, 2);
}

// ─── Prompt injection tests ────────────────────────────────────────────────

/// Flatten a captured message list for the substring assertions below.
///
/// The callback receives `Vec<RawMessage>` rather than a rendered prompt (the
/// caller owns rendering), so these tests assert on the message list's content.
/// That is the right level for them: what they are about is *which messages*
/// the executor assembles, and the executor no longer decides how those become
/// text.
fn flatten(msgs: &[RawMessage]) -> String {
    msgs.iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Verify that the contract instruction is actually present in the messages
/// that reach the inference closure.
#[tokio::test]
async fn contract_instruction_injected_into_prompt() {
    let choices = vec!["alpha".to_string(), "beta".to_string()];
    let spec = OutputContractSpec::EnumChoice {
        choices: choices.clone(),
        case_sensitive: false,
        allow_index: true,
    };

    let captured_prompt: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let captured = captured_prompt.clone();

    let infer = move |msgs: Vec<RawMessage>| {
        *captured.lock().unwrap() = Some(flatten(&msgs));
        std::future::ready(Ok::<CompletionResult, anyhow::Error>(canned("alpha")))
    };

    execute_contract(
        &user_messages("Pick a letter."),
        "test-model",
        &spec,
        1,
        &[],
        infer,
    )
    .await
    .unwrap();

    let prompt = captured_prompt.lock().unwrap().clone().unwrap();
    // The contract instruction should embed the choice names.
    assert!(
        prompt.to_lowercase().contains("alpha"),
        "prompt should contain choice 'alpha'; got:\n{prompt}"
    );
    assert!(
        prompt.to_lowercase().contains("beta"),
        "prompt should contain choice 'beta'; got:\n{prompt}"
    );
}

/// On retry, the previous bad response must be included in the prompt as
/// assistant context, and a tightening user message must follow it.
#[tokio::test]
async fn retry_appends_context_and_tightening_message() {
    let spec = OutputContractSpec::EnumChoice {
        choices: vec!["yes".to_string(), "no".to_string()],
        case_sensitive: false,
        allow_index: true,
    };

    let prompts: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured = prompts.clone();

    // First call: return garbage so a retry is triggered.
    // Second call: return the correct label.
    let call_count = Arc::new(Mutex::new(0usize));
    let cc = call_count.clone();
    let infer = move |msgs: Vec<RawMessage>| {
        captured.lock().unwrap().push(flatten(&msgs));
        let idx = {
            let mut c = cc.lock().unwrap();
            let i = *c;
            *c += 1;
            i
        };
        let response = if idx == 0 { "I cannot decide." } else { "yes" };
        std::future::ready(Ok::<CompletionResult, anyhow::Error>(canned(response)))
    };

    execute_contract(
        &user_messages("Quick question."),
        "test-model",
        &spec,
        3,
        &[],
        infer,
    )
    .await
    .unwrap();

    let all_prompts = prompts.lock().unwrap();
    assert_eq!(
        all_prompts.len(),
        2,
        "should make exactly 2 inference calls"
    );

    // The second prompt must contain the first bad response as assistant
    // context. The `assistant: ` prefix is part of the claim: since the
    // executor's message lists are rendered through the model's chat template,
    // the role selects a template branch (`<|start_header_id|>{role}`) and, on
    // a template that checks user/assistant alternation, decides whether the
    // render raises at all. Before that change a wrong role here was a
    // cosmetic prefix in a flat join.
    assert!(
        all_prompts[1].contains("assistant: I cannot decide."),
        "retry prompt should include the prior bad response as an ASSISTANT \
         turn; got:\n{}",
        all_prompts[1]
    );

    // And a tightening correction from the user.
    //
    // Written out in full rather than checked for "yes" or "no": the contract's
    // own system instruction enumerates both choices and is present in every
    // attempt, so a label search here holds no matter what the tightening
    // message says — including when it says nothing. Nor is it compared against
    // `validation::tightening_message`, because `contains` of a value derived
    // from the function under test passes trivially when that function returns
    // `String::new()`.
    let expected_tightening = "user: Your previous response did not end with exactly one of the \
                               required choices. Please respond again — the VERY LAST LINE must \
                               contain ONLY one of: YES (1), NO (2)";
    assert!(
        all_prompts[1].ends_with(expected_tightening),
        "the retry's last message must be the tightening correction, as a USER \
         turn.\nexpected to end with:\n{expected_tightening}\ngot:\n{}",
        all_prompts[1]
    );
}

/// The callback must see the message list as it stands on EACH attempt.
///
/// This is the property that makes rendering upstream impossible, and therefore
/// the reason `execute_contract` takes `Fn(Vec<RawMessage>)` rather than
/// `Fn(String)`. The list grows between attempts —
/// `inject_contract_instruction` prepends a system message once, then every
/// retry pushes the rejected assistant reply and a tightening user message — so
/// a prompt rendered once by the caller would be stale from attempt 2 onward.
///
/// Asserted as exact lengths rather than "grew", derived independently of the
/// implementation: 1 user + 1 injected system = 2 on attempt 1, then +2 per
/// retry. A `>` comparison would also pass if the executor handed over the
/// *same* list object it later mutated, which is the failure this guards.
#[tokio::test]
async fn contract_callback_sees_each_mutated_message_list() {
    let spec = OutputContractSpec::EnumChoice {
        choices: choices(),
        case_sensitive: false,
        allow_index: true,
    };

    let seen: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let seen2 = seen.clone();

    // "garbage" is none of yes/no/maybe and holds no digit for `allow_index`,
    // so no attempt parses and all three run.
    let infer = move |msgs: Vec<RawMessage>| {
        seen2.lock().unwrap().push(msgs.len());
        std::future::ready(Ok::<CompletionResult, anyhow::Error>(canned("garbage")))
    };

    let result = execute_contract(
        &user_messages("Is the service healthy?"),
        "test-model",
        &spec,
        3,
        &[],
        infer,
    )
    .await
    .unwrap();

    assert!(
        !result.result.parse_success,
        "the stub must never parse, or fewer than three attempts run"
    );
    let seen = seen.lock().unwrap();
    assert_eq!(
        *seen,
        vec![2, 4, 6],
        "each attempt must receive that attempt's own message list"
    );
}
