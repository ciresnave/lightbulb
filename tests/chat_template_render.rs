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
