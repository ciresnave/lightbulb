//! Rendering is asserted against literal expected output, never "contains".
//! A template that emitted the right markers in the wrong order would pass a
//! `contains` check while prompting the model incorrectly — which is the exact
//! defect this work exists to remove.

use lightbulb::api::chat_template::{ChatTemplate, Resolution};
use lightbulb::contracts::validation::RawMessage;

fn msgs() -> Vec<RawMessage> {
    vec![RawMessage {
        role: "user".into(),
        content: "Name the capital of France.".into(),
    }]
}

/// TinyLlama-1.1B-Chat uses the Zephyr form. Asserted as an exact string.
///
/// The expected value is the verified rendering of TinyLlama-1.1B-Chat-v1.0's
/// own `chat_template`, traced under `trim_blocks=True`: its generation prompt
/// is `{{ '<|assistant|>' }}` followed by a newline that survives because it
/// trails a VARIABLE tag, not a block tag.
///
/// That is why the trailing newline below sits inside `{{ '...' }}` rather than
/// at the end of the source. A newline in trailing source position is stripped
/// by Jinja2 and minijinja alike (`keep_trailing_newline` defaults to false in
/// both, and transformers does not override it), so writing this template as
/// `...{% endfor %}<|assistant|>\n` renders `<|assistant|>` with the newline
/// silently eaten — a real prompt defect that reads as a typo in the test.
#[test]
fn zephyr_template_renders_exactly() {
    let src = "{% for m in messages %}<|{{ m.role }}|>\n{{ m.content }}{{ eos_token }}\n{% endfor %}{{ '<|assistant|>\n' }}";
    let t = ChatTemplate {
        source: src.to_string(),
        resolved_by: Resolution::Registry,
    };
    let out = t.render(&msgs(), "<s>", "</s>").expect("render failed");
    assert_eq!(
        out,
        "<|user|>\nName the capital of France.</s>\n<|assistant|>\n"
    );
}

/// Templates call `raise_exception` to reject message orders they do not
/// support. Unregistered, minijinja reports "unknown function", which is
/// indistinguishable from our own bug. This asserts we support the construct,
/// not merely that it happened not to be used.
#[test]
fn raise_exception_is_registered() {
    let src = "{{ raise_exception('nope') }}";
    let t = ChatTemplate {
        source: src.to_string(),
        resolved_by: Resolution::Registry,
    };
    let err = t.render(&msgs(), "<s>", "</s>").unwrap_err().to_string();
    assert!(
        err.contains("nope"),
        "raise_exception did not surface its message; got: {err}"
    );
    assert!(
        !err.contains("unknown function"),
        "raise_exception is not registered; got: {err}"
    );
}

/// bos/eos are referenced as bare variables by real templates.
#[test]
fn bos_and_eos_are_bound() {
    let src = "{{ bos_token }}|{{ eos_token }}";
    let t = ChatTemplate {
        source: src.to_string(),
        resolved_by: Resolution::Registry,
    };
    assert_eq!(t.render(&msgs(), "<s>", "</s>").unwrap(), "<s>|</s>");
}
