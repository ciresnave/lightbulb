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

fn tmpl(src: &str) -> ChatTemplate {
    ChatTemplate {
        source: src.to_string(),
        resolved_by: Resolution::Registry,
    }
}

/// TinyLlama-1.1B-Chat's REAL template, verbatim from its Hub
/// `tokenizer_config.json` — not a flattened paraphrase.
///
/// Using the real multi-line form is load-bearing, not fidelity theatre. A
/// one-liner has no newline after any `%}` and no leading whitespace before any
/// block tag, so `trim_blocks`/`lstrip_blocks` are **no-ops for it** and the
/// test passes identically whether or not `render` sets them. The real template
/// renders `"\n\n<|user|>\n…\n\n\n<|assistant|>\n\n"` when they are unset. That
/// difference is the only thing standing between this module and silent
/// mis-rendering, which is the one failure mode that does NOT fall through to
/// the next tier and does NOT get logged (spec §8 assigns this test that job).
#[test]
fn real_tinyllama_template_renders_exactly() {
    let src = "{% for message in messages %}\n\
               {% if message['role'] == 'user' %}\n\
               {{ '<|user|>\\n' + message['content'] + eos_token }}\n\
               {% elif message['role'] == 'system' %}\n\
               {{ '<|system|>\\n' + message['content'] + eos_token }}\n\
               {% elif message['role'] == 'assistant' %}\n\
               {{ '<|assistant|>\\n' + message['content'] + eos_token }}\n\
               {% endif %}\n\
               {% if loop.last and add_generation_prompt %}\n\
               {{ '<|assistant|>' }}\n\
               {% endif %}\n\
               {% endfor %}";
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

/// `lstrip_blocks` on its own, because TinyLlama's real template happens to
/// indent nothing — so the test above discriminates `trim_blocks` only, and
/// deleting `set_lstrip_blocks` would otherwise still be invisible. Real
/// checkpoint templates (Llama-3.x, Qwen) indent their block tags heavily.
#[test]
fn leading_whitespace_before_a_block_tag_is_stripped() {
    // No newline follows any `%}`, so trim_blocks cannot account for this.
    assert_eq!(
        tmpl("  {% if true %}A{% endif %}")
            .render(&msgs(), "<s>", "</s>")
            .unwrap(),
        "A"
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

// ── Dialect coverage ────────────────────────────────────────────────────────
//
// One test per construct that a narrower minijinja feature list rejects at
// PARSE time. These are positive controls: they fail loudly if the feature list
// is trimmed, which is what makes it demonstrably sufficient rather than merely
// longer. Each was measured to fail under `default-features = false,
// features = ["builtins"]` — `{% break %}`/`{% macro %}` as parse errors,
// `tojson` as "unknown filter".
//
// The failure they guard against is not a broken render. It is tier 1 refusing
// a checkpoint's own `tokenizer_config.json` and resolution falling through to
// a registry guess while `resolved_by` reports success at a lower tier.

/// `{% macro %}` — Hermes and Qwen tool-calling templates define one per tool.
/// Needs the `macros` feature; without it this is a parse error.
#[test]
fn macros_are_supported() {
    let src = "{% macro render(m) %}<{{ m['role'] }}>{{ m['content'] }}{% endmacro %}\
               {% for m in messages %}{{ render(m) }}{% endfor %}";
    assert_eq!(
        tmpl(src).render(&msgs(), "<s>", "</s>").unwrap(),
        "<user>Name the capital of France."
    );
}

/// `{% break %}` / `{% continue %}` — needs `loop_controls`. `transformers`
/// loads `jinja2.ext.loopcontrols` for exactly this, so templates authored
/// against it use them freely.
#[test]
fn loop_controls_are_supported() {
    let src = "{% for i in [1, 2, 3, 4] %}{% if i == 3 %}{% break %}{% endif %}\
               {% if i == 1 %}{% continue %}{% endif %}{{ i }}{% endfor %}";
    assert_eq!(tmpl(src).render(&msgs(), "<s>", "</s>").unwrap(), "2");
}

/// `| tojson` — Llama-3.1 serialises each tool definition with it. Needs the
/// `json` feature; without it minijinja reports "unknown filter".
#[test]
fn tojson_filter_is_supported() {
    let src = "{{ messages[0]['role'] | tojson }}|{{ [1, 2] | tojson }}";
    assert_eq!(
        tmpl(src).render(&msgs(), "<s>", "</s>").unwrap(),
        "\"user\"|[1,2]"
    );
}

/// `strftime_now(fmt)` — Llama-3.x opens with
/// `{%- set date_string = strftime_now("%d %b %Y") %}`. It is a *global*, not a
/// feature: no feature list makes it appear, so it has to be registered.
///
/// Asserted on shape rather than a literal date, because the value is the wall
/// clock. The shape still discriminates every way this can be wrong: an
/// unregistered function errors, and a passthrough stub would return the format
/// string itself.
#[test]
fn strftime_now_is_registered() {
    let out = tmpl("{{ strftime_now('%Y-%m-%d') }}")
        .render(&msgs(), "<s>", "</s>")
        .expect("strftime_now failed to render");
    assert_ne!(out, "%Y-%m-%d", "strftime_now echoed its format string");
    let parts: Vec<&str> = out.split('-').collect();
    assert_eq!(parts.len(), 3, "strftime_now rendered {out:?}, not a date");
    assert_eq!((parts[0].len(), parts[1].len(), parts[2].len()), (4, 2, 2));
    let year: u32 = parts[0].parse().expect("year is not numeric");
    assert!(
        (2026..2100).contains(&year),
        "strftime_now rendered year {year}"
    );

    // The exact call form Llama-3.1's template uses.
    let llama = tmpl("{%- set date_string = strftime_now(\"%d %b %Y\") %}{{ date_string }}")
        .render(&msgs(), "<s>", "</s>")
        .expect("the Llama-3.1 date_string preamble failed to render");
    assert_eq!(
        llama.split(' ').count(),
        3,
        "strftime_now('%d %b %Y') rendered {llama:?}"
    );
}
