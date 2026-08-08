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

// ── Python methods on primitives ────────────────────────────────────────────
//
// Jinja2's values ARE Python objects, so template authors call `str` and `dict`
// methods without a second thought. minijinja's are not, and reports
// `unknown method: string has no method named startswith`. `render` wires
// `minijinja_contrib::pycompat` through `set_unknown_method_callback` to close
// that gap.
//
// One test per method rather than one "Qwen3 renders" smoke test, on purpose:
// pycompat is an upstream crate whose method list can change, and a smoke test
// would report only that *something* broke. These name which one.
//
// Every construct below is copied from a real Hub `chat_template`, not
// invented. Each was measured to fail without the callback — `.startswith`,
// `.endswith`, `.strip`, `.lstrip`, `.rstrip`, `.split` and `.items` all report
// `unknown method`.

/// `str.startswith` / `str.endswith` — Qwen3-8B, `chat:20`, the line that
/// produced the original failure report.
#[test]
fn python_str_startswith_and_endswith() {
    let src = "{{ 'ab' if '<tool_response>x</tool_response>'.startswith('<tool_response>') else 'no' }}\
               {{ 'cd' if '<tool_response>x</tool_response>'.endswith('</tool_response>') else 'no' }}\
               {{ 'e' if 'plain'.startswith('<tool_response>') else 'f' }}";
    assert_eq!(tmpl(src).render(&msgs(), "<s>", "</s>").unwrap(), "abcdf");
}

/// `str.strip` / `str.lstrip` / `str.rstrip` **with an argument** — Qwen3-8B
/// strips `'\n'` specifically, not whitespace generally. The argument is the
/// load-bearing part: minijinja's `| trim` filter is not a substitute, because
/// `trim` cannot be told which characters to remove.
#[test]
fn python_str_strip_family_takes_a_character_argument() {
    let src = "[{{ '\\n\\nx y\\n\\n'.strip('\\n') }}]\
               [{{ '\\n\\nx y\\n\\n'.lstrip('\\n') }}]\
               [{{ '\\n\\nx y\\n\\n'.rstrip('\\n') }}]\
               [{{ '..x..'.strip('.') }}]";
    assert_eq!(
        tmpl(src).render(&msgs(), "<s>", "</s>").unwrap(),
        "[x y][x y\n\n][\n\nx y][x]"
    );
}

/// `str.split` plus the negative index Qwen3-8B pairs it with —
/// `content.split('</think>')[-1]`. A `split` that returned the wrong arity, or
/// an index that counted from the front, both survive a "renders OK" check.
#[test]
fn python_str_split_indexes_from_both_ends() {
    let src = "{{ 'a</think>b</think>c'.split('</think>') | length }}\
               |{{ 'a</think>b</think>c'.split('</think>')[0] }}\
               |{{ 'a</think>b</think>c'.split('</think>')[-1] }}";
    assert_eq!(tmpl(src).render(&msgs(), "<s>", "</s>").unwrap(), "3|a|c");
}

/// Qwen3-8B's `<think>` extraction, verbatim from its Hub template (lines 39,
/// 40 and 45) — the chained form, which is where a per-method fix that got the
/// return types subtly wrong would show up and the isolated tests above would
/// not.
#[test]
fn qwen3_think_extraction_renders() {
    let src = "{%- set content = '<think>\\nreasoning\\n</think>\\n\\nanswer' %}\
               {%- set reasoning_content = \
                   content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\
               {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\
               {{- '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}";
    assert_eq!(
        tmpl(src).render(&msgs(), "<s>", "</s>").unwrap(),
        "<think>\nreasoning\n</think>\n\nanswer"
    );
}

/// `dict.items()` — Mistral-7B-Instruct-v0.3 serialises each tool with it.
///
/// This one is latent rather than loud: the branch sits behind
/// `tools is not none`, so every no-tools request renders fine and the gap only
/// surfaces the day tools are wired. Asserted here so it cannot.
///
/// Key order is the map's iteration order, not source order — `description`
/// sorts before `name`. That is observed behaviour, pinned so a change in it is
/// a test failure rather than a silent change in every tool-calling prompt.
#[test]
fn python_dict_items() {
    let src = "{%- set tool = {\"name\": \"get_weather\", \"description\": \"Get it\", \"return\": \"skipped\"} %}\
               {%- for key, val in tool.items() if key != \"return\" %}\
               {{- key }}={{ val }};\
               {%- endfor %}";
    assert_eq!(
        tmpl(src).render(&msgs(), "<s>", "</s>").unwrap(),
        "description=Get it;name=get_weather;"
    );
}

/// Mistral-7B-Instruct-v0.3's `[AVAILABLE_TOOLS]` loop, verbatim from its Hub
/// template, driven by a tool definition of the shape the OpenAI API delivers.
///
/// `tools` is not a `render` parameter yet, so it is bound in template source —
/// which is what `transformers` does when a caller passes tools. Without this,
/// "Mistral renders" only ever means "the branch was skipped".
#[test]
fn mistral_available_tools_branch_renders() {
    let src = "{%- set tools = [{\"function\": {\"name\": \"get_weather\", \
                 \"description\": \"Get the weather\", \
                 \"parameters\": {\"type\": \"object\"}}}] %}\
        {{- \"[AVAILABLE_TOOLS] [\" }}\n\
        {%- for tool in tools %}\n\
            {%- set tool = tool.function %}\n\
            {{- '{\"type\": \"function\", \"function\": {' }}\n\
            {%- for key, val in tool.items() if key != \"return\" %}\n\
                {%- if val is string %}\n\
                    {{- '\"' + key + '\": \"' + val + '\"' }}\n\
                {%- else %}\n\
                    {{- '\"' + key + '\": ' + val|tojson }}\n\
                {%- endif %}\n\
                {%- if not loop.last %}\n\
                    {{- \", \" }}\n\
                {%- endif %}\n\
            {%- endfor %}\n\
            {{- \"}}\" }}\n\
            {%- if not loop.last %}\n\
                {{- \", \" }}\n\
            {%- else %}\n\
                {{- \"]\" }}\n\
            {%- endif %}\n\
        {%- endfor %}\n\
        {{- \"[/AVAILABLE_TOOLS]\" }}";
    assert_eq!(
        tmpl(src).render(&msgs(), "<s>", "</s>").unwrap(),
        "[AVAILABLE_TOOLS] [{\"type\": \"function\", \"function\": \
         {\"description\": \"Get the weather\", \"name\": \"get_weather\", \
         \"parameters\": {\"type\":\"object\"}}}][/AVAILABLE_TOOLS]"
    );
}

// ── Tier resolution and the sidecar ─────────────────────────────────────────
//
// Every fixture below is a temp directory, never the real checkpoint.
// TinyLlama-1.1B-Chat's snapshot ships only `config.json`, `model.safetensors`
// and `tokenizer.json` — no `tokenizer_config.json`, and its `<|user|>` markers
// are ordinary text rather than added tokens — so it resolves at tier 3 and
// cannot exercise tiers 1 or 2 at all.

use std::io::Write;

fn tmp_model_dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("lb-chat-tmpl-{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    // config.json is what fingerprint() hashes; every fixture needs one.
    let mut f = std::fs::File::create(d.join("config.json")).unwrap();
    f.write_all(br#"{"model_type":"llama"}"#).unwrap();
    d
}

/// Tier order is honoured: a model that satisfies tiers 1, 2 AND 3 resolves via
/// tier 1. Fails if the tiers are reordered or a later tier overwrites an
/// earlier result.
///
/// The fixture is built up one tier at a time, asserting at each step, because
/// "tier 1 wins" is only a claim about ORDER if the lower tiers demonstrably
/// fire for this same directory. A fixture that satisfied tier 1 alone would
/// pass this test unchanged against an implementation with no registry at all.
///
/// Concretely: the directory name carries `tinyllama` and `chat`, which is what
/// `registry::from_family` matches on, and the `tokenizer.json` lists
/// `<|im_start|>` as an *added token*, which is what `from_vocab_signature`
/// matches on. All three answers are distinguishable from each other.
#[test]
fn tokenizer_config_wins_over_registry() {
    use lightbulb::api::chat_template::registry;
    let d = tmp_model_dir("tinyllama-chat-tier-order");

    // Tier 3 alone.
    let t3 = lightbulb::api::chat_template::resolve(&d);
    assert_eq!(
        t3.resolved_by,
        Resolution::Registry,
        "tier 3 does not fire for this fixture, so the test below would prove \
         nothing about tier ORDER"
    );
    assert_eq!(t3.source, registry::ZEPHYR);

    // Tier 2 added — it must beat tier 3.
    std::fs::write(
        d.join("tokenizer.json"),
        r#"{"added_tokens":[{"id":1,"content":"<|im_start|>"}]}"#,
    )
    .unwrap();
    let t2 = lightbulb::api::chat_template::resolve(&d);
    assert_eq!(
        t2.resolved_by,
        Resolution::VocabSignature,
        "tier 2 does not fire for this fixture"
    );
    assert_eq!(t2.source, registry::CHATML);

    // Tier 1 added — it must beat both.
    std::fs::write(
        d.join("tokenizer_config.json"),
        r#"{"chat_template":"FROM_TOKENIZER_CONFIG"}"#,
    )
    .unwrap();
    let t1 = lightbulb::api::chat_template::resolve(&d);
    assert_eq!(t1.resolved_by, Resolution::TokenizerConfig);
    assert_eq!(t1.source, "FROM_TOKENIZER_CONFIG");
}

/// The sidecar outranks everything — it is the probe's persisted answer.
///
/// The tier-1 source is asserted to fire *before* the sidecar is written, for
/// the same reason as above: without that control the test would pass against
/// an implementation that never read `tokenizer_config.json`.
#[test]
fn sidecar_wins_over_tokenizer_config() {
    let d = tmp_model_dir("sidecar-wins");
    std::fs::write(
        d.join("tokenizer_config.json"),
        r#"{"chat_template":"FROM_TOKENIZER_CONFIG"}"#,
    )
    .unwrap();

    let before = lightbulb::api::chat_template::resolve(&d);
    assert_eq!(
        before.resolved_by,
        Resolution::TokenizerConfig,
        "the tier this one must outrank never fired"
    );
    assert_eq!(before.source, "FROM_TOKENIZER_CONFIG");

    let sc = lightbulb::api::chat_template::Sidecar {
        template: "FROM_SIDECAR".into(),
        resolved_by: Resolution::Probe,
        evidence: "EOS 8/8 zephyr".into(),
        resolved_at: "2026-08-08T00:00:00Z".into(),
        model_fingerprint: lightbulb::api::chat_template::fingerprint(&d),
    };
    lightbulb::api::chat_template::write_sidecar(&d, &sc).unwrap();
    let t = lightbulb::api::chat_template::resolve(&d);
    assert_eq!(t.resolved_by, Resolution::Sidecar);
    assert_eq!(t.source, "FROM_SIDECAR");
}

/// Provenance survives a round trip. Fails if the writer drops the fields for
/// brevity — which is the likely slip, since they are not needed to render.
#[test]
fn sidecar_round_trips_with_provenance() {
    let d = tmp_model_dir("round-trip");
    let sc = lightbulb::api::chat_template::Sidecar {
        template: "T".into(),
        resolved_by: Resolution::Probe,
        evidence: "EOS 8/8 with zephyr, 1/8 with llama2".into(),
        resolved_at: "2026-08-08T00:00:00Z".into(),
        model_fingerprint: lightbulb::api::chat_template::fingerprint(&d),
    };
    lightbulb::api::chat_template::write_sidecar(&d, &sc).unwrap();
    let back = lightbulb::api::chat_template::read_sidecar(&d).expect("sidecar did not read back");
    assert_eq!(back.template, "T");
    assert_eq!(back.resolved_by, Resolution::Probe);
    assert_eq!(back.evidence, "EOS 8/8 with zephyr, 1/8 with llama2");
    assert_eq!(back.resolved_at, "2026-08-08T00:00:00Z");
}

/// A sidecar written for a different checkpoint must be rejected, so a
/// directory reused for another model does not silently inherit its template.
/// This exists because writing the fingerprint but never checking it is the
/// likely slip.
///
/// The `is_some` control is not redundant with the rejection: it separates
/// "rejected because the fingerprint mismatched" from "rejected because
/// `read_sidecar` never reads anything", which would satisfy the assertion
/// below for entirely the wrong reason.
#[test]
fn stale_sidecar_is_rejected() {
    let d = tmp_model_dir("stale");
    let good = lightbulb::api::chat_template::Sidecar {
        template: "STALE".into(),
        resolved_by: Resolution::Probe,
        evidence: "e".into(),
        resolved_at: "2026-08-08T00:00:00Z".into(),
        model_fingerprint: lightbulb::api::chat_template::fingerprint(&d),
    };
    lightbulb::api::chat_template::write_sidecar(&d, &good).unwrap();
    assert!(
        lightbulb::api::chat_template::read_sidecar(&d).is_some(),
        "a MATCHING sidecar was not accepted, so the rejection below would \
         prove nothing"
    );

    let stale = lightbulb::api::chat_template::Sidecar {
        model_fingerprint: "0000000000000000".into(), // deliberately wrong
        ..good
    };
    lightbulb::api::chat_template::write_sidecar(&d, &stale).unwrap();
    assert!(
        lightbulb::api::chat_template::read_sidecar(&d).is_none(),
        "a sidecar whose fingerprint does not match the checkpoint was accepted"
    );
    let t = lightbulb::api::chat_template::resolve(&d);
    assert_ne!(t.source, "STALE", "resolution used a stale sidecar");
}

/// Every registry constant must actually render to its generation prompt.
///
/// This exists because Task 1 shipped a template whose trailing newline sat in
/// source position and was silently stripped — and the test written alongside
/// it could not detect that, because the template had been flattened until the
/// whitespace settings no longer applied to it. A constant that renders to the
/// wrong shape still renders, so nothing falls through and nothing is logged.
/// Parameterised over all three so adding a fourth without a test is not
/// possible.
#[test]
fn every_registry_constant_ends_with_its_generation_prompt() {
    use lightbulb::api::chat_template::registry;
    let expected = [
        ("zephyr", registry::ZEPHYR, "<|assistant|>\n"),
        ("chatml", registry::CHATML, "<|im_start|>assistant\n"),
        ("llama2", registry::LLAMA2, "[/INST]"),
    ];
    assert_eq!(
        expected.len(),
        registry::candidates().len(),
        "a candidate was added to the registry without a render assertion"
    );
    for (name, src, tail) in expected {
        assert!(
            registry::candidates()
                .iter()
                .any(|(n, s)| *n == name && *s == src),
            "{name} is not the constant candidates() offers under that name"
        );
        let t = ChatTemplate {
            source: src.to_string(),
            resolved_by: Resolution::Registry,
        };
        let out = t
            .render(&msgs(), "<s>", "</s>")
            .unwrap_or_else(|e| panic!("{name} failed to render: {e}"));
        assert!(
            out.ends_with(tail),
            "{name} rendered {out:?}, which does not end with {tail:?} — a \
             trailing newline in source position is stripped by Jinja"
        );
    }
}

/// No tier fires: resolution reports None rather than inventing a template.
#[test]
fn missing_everything_falls_through() {
    let d = tmp_model_dir("nothing");
    std::fs::write(d.join("config.json"), r#"{"model_type":"unknown-xyz"}"#).unwrap();
    let t = lightbulb::api::chat_template::resolve(&d);
    assert_eq!(t.resolved_by, Resolution::None);
    assert_eq!(t.source, "");
}
