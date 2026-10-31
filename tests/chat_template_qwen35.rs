// SPDX-License-Identifier: MIT OR Apache-2.0
//! The real `qwen35` (Qwen3.5 / Qwen3.8) chat template renders through
//! Lightbulb's `ChatTemplate::render` exactly as the reference engine renders it
//! (ROADMAP F10, L6).
//!
//! The template is `Qwen/Qwen3.5-0.8B`'s own `chat_template.jinja`, byte for
//! byte (see `tests/fixtures/qwen35_chat_template/README.md` for origin, licence
//! and how the answer key was made). It exercises features no other template in
//! this suite does: `namespace`, a macro with `is iterable`/`is mapping`/
//! `is string` tests, `loop.previtem`/`loop.nextitem`, `[::-1]` slicing, `|trim`,
//! `.split`/`.rstrip`/`.lstrip`, and `raise_exception`.
//!
//! The expected strings were produced by Python `jinja2` 3.1.6 in
//! `transformers`' dialect (`ImmutableSandboxedEnvironment(trim_blocks=True,
//! lstrip_blocks=True)`), an engine independent of minijinja, so agreement here
//! is not one implementation agreeing with itself.
//!
//! **Not covered, because `render` does not pass them:** `tools`,
//! `enable_thinking`, `add_vision_id`, structured (list) content and
//! `tool_calls` on assistant messages. With `enable_thinking` undefined the
//! template ends the prompt with an empty closed think block
//! (`<think>\n\n</think>\n\n`), i.e. non-thinking mode; that is what the cases
//! below pin, and it is what Lightbulb would serve today.

use lightbulb::api::chat_template::{ChatTemplate, Resolution};
use lightbulb::contracts::validation::RawMessage;
use sha2::{Digest, Sha256};

const TEMPLATE: &str = include_str!("fixtures/qwen35_chat_template/chat_template.jinja");

type Case = (
    &'static str,
    &'static [(&'static str, &'static str)],
    &'static str,
);

const RENDERS: &[Case] = &[
    (
        "user_only",
        &[("user", "Name the capital of France.")],
        "<|im_start|>user\nName the capital of France.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
    (
        "system_and_user",
        &[
            ("system", "  Be brief.  "),
            ("user", "Name the capital of France."),
        ],
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nName the capital of France.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
    (
        "think_history_dropped_before_last_query",
        &[
            ("system", "Be brief."),
            ("user", "Hi"),
            ("assistant", "<think>\nsmall talk\n</think>\n\nHello!"),
            ("user", "Name the capital of France."),
        ],
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nName the capital of France.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
    (
        "assistant_after_last_query_keeps_think",
        &[
            ("user", "Hi"),
            ("assistant", "<think>\nsmall talk\n</think>\n\nHello!"),
        ],
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nsmall talk\n</think>\n\nHello!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
    (
        "assistant_without_think_after_last_query",
        &[("user", "Hi"), ("assistant", "Hello!")],
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
    (
        "tool_message_after_user",
        &[("user", "Hi"), ("tool", "{\"a\": 1}")],
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>user\n<tool_response>\n{\"a\": 1}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
    (
        "two_tool_messages",
        &[
            ("user", "Hi"),
            ("tool", "one"),
            ("tool", "two"),
            ("user", "Thanks"),
        ],
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>user\n<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\n</tool_response><|im_end|>\n<|im_start|>user\nThanks<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
];

const REFUSALS: &[Case] = &[
    (
        "system_not_first",
        &[("user", "Hi"), ("system", "late")],
        "System message must be at the beginning.",
    ),
    (
        "no_user_query",
        &[("system", "only a system message")],
        "No user query found in messages.",
    ),
    (
        "unexpected_role",
        &[("user", "Hi"), ("narrator", "x")],
        "Unexpected message role.",
    ),
    ("empty_messages", &[], "No messages provided."),
];

fn render(msgs: &[(&str, &str)]) -> anyhow::Result<String> {
    let t = ChatTemplate {
        source: TEMPLATE.to_string(),
        resolved_by: Resolution::GgufMetadata,
    };
    let raw: Vec<RawMessage> = msgs
        .iter()
        .map(|(role, content)| RawMessage {
            role: (*role).into(),
            content: (*content).into(),
        })
        .collect();
    t.render(&raw, "<s>", "<|im_end|>")
}

/// The fixture is the file it claims to be: a CRLF rewrite or an edit would
/// change the whitespace the template renders, and the answer key with it.
#[test]
fn the_fixture_is_the_published_file_byte_for_byte() {
    assert_eq!(TEMPLATE.len(), 7755);
    assert_eq!(
        hex::encode(Sha256::digest(TEMPLATE.as_bytes())),
        "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
    );
}

#[test]
fn every_render_case_matches_the_reference_engine_exactly() {
    for (name, msgs, expected) in RENDERS {
        let got = render(msgs).unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert_eq!(&got, expected, "{name}");
    }
}

/// The template's `raise_exception` guards must refuse the same inputs the
/// reference refuses, with its message in ours.
#[test]
fn every_refusal_case_is_refused_with_the_templates_own_message() {
    for (name, msgs, message) in REFUSALS {
        let err = render(msgs).expect_err(name);
        assert!(format!("{err:#}").contains(message), "{name}: {err:#}");
    }
}

/// Control for the two tests above: a different message list does not render to
/// the same string, so a `render` that ignored its input could not pass.
#[test]
fn the_cases_are_distinguishable() {
    let outs: std::collections::HashSet<_> = RENDERS.iter().map(|(_, _, e)| *e).collect();
    assert_eq!(outs.len(), RENDERS.len());
}
