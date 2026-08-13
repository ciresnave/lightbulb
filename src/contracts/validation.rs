//! Shared contract helpers: prompt injection, parsing dispatch, retry policy.
//!
//! This module is **pure** — no async, no AppState.  All network/model I/O lives
//! in `chat.rs`; this module only knows about text in and text out.
//!
//! ## How the integration works
//!
//! 1. `inject_contract_instruction` patches the caller's message list so the
//!    model "sees" the contract instructions via the system message.
//! 2. The caller renders that list through the model's own chat template and
//!    calls the model runner.  **This module deliberately does not know how a
//!    prompt is formatted.**  It used to, via a `messages_to_prompt` that
//!    hard-coded the legacy `role: content` join; that made the contract path
//!    a third copy of a defect the chat endpoint had already fixed, so it was
//!    deleted rather than re-pointed at the template.
//! 3. `try_parse` dispatches to the right contract parser and returns
//!    `Some(ContractOutput)` when it succeeds, `None` otherwise.
//! 4. `is_parse_successful` gates whether the result is "good enough" (e.g.
//!    `TaggedFields` with `require_all` checks all fields are present).
//! 5. If the parse failed, `tightening_message` returns the next correction
//!    user message and the caller appends it before the next attempt.

use super::{ContractOutput, OutputContractSpec};

// ─── Prompt injection ──────────────────────────────────────────────────────

/// A minimal representation of a chat message used only within this module
/// for prompt manipulation.  The real `ChatMessage` type lives in `chat.rs`;
/// this module takes `(role, content)` pairs to avoid a circular dependency.
///
/// `Clone` because `executor` hands each attempt's list to the inference
/// callback by value — see that module's docs for why it cannot pass a borrow.
#[derive(Clone, Debug)]
pub struct RawMessage {
    pub role: String,
    pub content: String,
}

/// Inject contract instructions into a mutable list of raw messages.
///
/// If a system message already exists it is amended; otherwise a new system
/// message is prepended.
pub fn inject_contract_instruction(messages: &mut Vec<RawMessage>, spec: &OutputContractSpec) {
    let instruction = build_system_instruction(spec);

    // Find an existing system message and amend it.
    if let Some(sys) = messages.iter_mut().find(|m| m.role == "system") {
        sys.content.push_str(&instruction);
        return;
    }

    // No system message — prepend one.
    messages.insert(
        0,
        RawMessage {
            role: "system".to_string(),
            content: instruction.trim_start_matches('\n').to_string(),
        },
    );
}

/// Build the system-prompt fragment for the given contract.
pub fn build_system_instruction(spec: &OutputContractSpec) -> String {
    match spec {
        OutputContractSpec::EnumChoice {
            choices,
            case_sensitive,
            allow_index,
        } => super::enum_choice::build_system_instruction(choices, *case_sensitive, *allow_index),
        OutputContractSpec::TaggedFields {
            allowed_tags,
            required_tags,
            repeatable_tags,
        } => super::tagged_fields::build_system_instruction(
            allowed_tags,
            required_tags,
            repeatable_tags,
        ),
        OutputContractSpec::CommitBlock {
            block_type,
            must_be_last,
        } => super::commit_block::build_system_instruction(block_type, *must_be_last),
        OutputContractSpec::Json => {
            "\n\nIMPORTANT — OUTPUT FORMAT: Respond with valid JSON only.".to_string()
        }
    }
}

// ─── Tightening prompt ─────────────────────────────────────────────────────

/// Return the correction user message to append before retry `attempt`
/// (1-indexed; 1 = second inference call, 2 = third, …).
pub fn tightening_message(attempt: u32, spec: &OutputContractSpec) -> String {
    match spec {
        OutputContractSpec::EnumChoice {
            choices,
            allow_index,
            ..
        } => super::enum_choice::build_tightening_message(attempt, choices, *allow_index),
        OutputContractSpec::TaggedFields {
            allowed_tags,
            required_tags,
            ..
        } => super::tagged_fields::build_tightening_message(attempt, allowed_tags, required_tags),
        OutputContractSpec::CommitBlock { block_type, .. } => {
            super::commit_block::build_tightening_message(attempt, block_type)
        }
        OutputContractSpec::Json => {
            "Your previous response was not valid JSON. Please respond with JSON only.".to_string()
        }
    }
}

// ─── Parsing ───────────────────────────────────────────────────────────────

/// Attempt to parse the model's raw `text` according to `spec`.
///
/// Returns `Some(ContractOutput)` on any non-empty parse, `None` on complete
/// failure.  Callers should further check `is_parse_successful` if they need
/// to enforce `required_tags` for `TaggedFields`.
pub fn try_parse(text: &str, spec: &OutputContractSpec) -> Option<ContractOutput> {
    match spec {
        OutputContractSpec::EnumChoice {
            choices,
            case_sensitive,
            allow_index,
        } => super::enum_choice::parse(text, choices, *case_sensitive, *allow_index)
            .map(|choice| ContractOutput::EnumChoice { choice }),
        OutputContractSpec::TaggedFields {
            allowed_tags,
            repeatable_tags,
            ..
        } => {
            let map = super::tagged_fields::parse(text, allowed_tags, repeatable_tags);
            if map.is_empty() {
                None
            } else {
                Some(ContractOutput::TaggedFields { tags: map })
            }
        }
        OutputContractSpec::CommitBlock {
            block_type,
            must_be_last,
        } => super::commit_block::parse(text, block_type, *must_be_last)
            .map(|block_text| ContractOutput::CommitBlock { block_text }),
        // Json is not yet implemented; fall through to None.
        OutputContractSpec::Json => None,
    }
}

/// Returns `true` when the parsed output is considered *sufficient* given the
/// contract's requirements.
///
/// For `EnumChoice` / `Json` / `CommitBlock`: any successful parse is sufficient.
/// For `TaggedFields`: every tag in `required_tags` must be present in the map.
pub fn is_parse_successful(output: &ContractOutput, spec: &OutputContractSpec) -> bool {
    match (output, spec) {
        (
            ContractOutput::TaggedFields { tags: parsed_map },
            OutputContractSpec::TaggedFields { required_tags, .. },
        ) => super::tagged_fields::check_cardinality(parsed_map, required_tags),
        (ContractOutput::Raw { .. }, _) => false,
        _ => true,
    }
}

// ─── Logging helpers ───────────────────────────────────────────────────────

/// Log a contract attempt result.  Uses `tracing` so output is captured by the
/// existing subscriber.
pub fn log_attempt(
    contract_type: &str,
    attempt: u32,
    parse_success: bool,
    model_id: &str,
    latency_ms: u64,
) {
    if parse_success {
        tracing::info!(
            contract = contract_type,
            attempt,
            parse_success,
            model = model_id,
            latency_ms,
            "contract parse succeeded"
        );
    } else {
        tracing::warn!(
            contract = contract_type,
            attempt,
            parse_success,
            model = model_id,
            latency_ms,
            "contract parse failed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inject_amends_existing_system_message() {
        let mut msgs = vec![
            RawMessage {
                role: "system".to_string(),
                content: "Be helpful.".to_string(),
            },
            RawMessage {
                role: "user".to_string(),
                content: "Is the sky blue?".to_string(),
            },
        ];
        let spec = OutputContractSpec::EnumChoice {
            choices: vec!["yes".to_string(), "no".to_string()],
            case_sensitive: false,
            allow_index: true,
        };
        inject_contract_instruction(&mut msgs, &spec);

        // System message should be amended, not duplicated.
        let sys_count = msgs.iter().filter(|m| m.role == "system").count();
        assert_eq!(sys_count, 1);
        assert!(msgs[0].content.contains("yes") || msgs[0].content.contains("YES"));
    }

    #[test]
    fn inject_prepends_when_no_system_message() {
        let mut msgs = vec![RawMessage {
            role: "user".to_string(),
            content: "Pick a colour.".to_string(),
        }];
        let spec = OutputContractSpec::EnumChoice {
            choices: vec!["red".to_string(), "blue".to_string()],
            case_sensitive: false,
            allow_index: true,
        };
        inject_contract_instruction(&mut msgs, &spec);

        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn try_parse_enum_choice() {
        let spec = OutputContractSpec::EnumChoice {
            choices: vec!["yes".to_string(), "no".to_string()],
            case_sensitive: false,
            allow_index: true,
        };
        let out = try_parse("My answer is yes.", &spec);
        assert!(matches!(out, Some(ContractOutput::EnumChoice { choice }) if choice == "yes"));
    }

    #[test]
    fn try_parse_tagged_fields() {
        let spec = OutputContractSpec::TaggedFields {
            allowed_tags: vec!["MODE".to_string()],
            required_tags: vec![],
            repeatable_tags: vec![],
        };
        let out = try_parse("<MODE>lint_error</MODE>", &spec);
        assert!(matches!(out, Some(ContractOutput::TaggedFields { .. })));
    }

    #[test]
    fn required_tags_gate() {
        let tags = vec!["MODE".to_string(), "CONF".to_string()];
        let parsed = {
            let mut m = std::collections::HashMap::new();
            m.insert("MODE".to_string(), vec!["x".to_string()]);
            m
        };
        let out = ContractOutput::TaggedFields { tags: parsed };
        let spec_strict = OutputContractSpec::TaggedFields {
            allowed_tags: tags.clone(),
            required_tags: tags.clone(),
            repeatable_tags: vec![],
        };
        let spec_loose = OutputContractSpec::TaggedFields {
            allowed_tags: tags.clone(),
            required_tags: vec![],
            repeatable_tags: vec![],
        };
        assert!(!is_parse_successful(&out, &spec_strict));
        assert!(is_parse_successful(&out, &spec_loose));
    }
}
