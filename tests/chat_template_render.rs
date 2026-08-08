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

/// A fresh fixture directory under `CARGO_TARGET_TMPDIR`.
///
/// **`CARGO_TARGET_TMPDIR`, not `std::env::temp_dir()`**, because tier 3 reads
/// the *path*: a system temp directory that happened to sit under a folder
/// named `qwen`, `chatml`, `llama-2` or `mistral` silently turned
/// `missing_everything_falls_through` into a test of nothing. The target tmpdir
/// is inside the checkout, so it is the repo layout that has to stay clean, and
/// that is visible to whoever breaks it.
///
/// **Nonce, not a fixed name.** The old version did `remove_dir_all` +
/// `create_dir_all` on a name derived only from `name`, so two concurrent
/// `cargo test` runs — or `cargo test` and `cargo mutants` — deleted each
/// other's fixtures mid-test.
fn tmp_model_dir(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let nonce = format!(
        "{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    );
    let d = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("lb-chat-tmpl-{name}-{nonce}"));
    std::fs::create_dir_all(&d).unwrap();
    // config.json is what fingerprint() hashes; every fixture needs one.
    let mut f = std::fs::File::create(d.join("config.json")).unwrap();
    f.write_all(br#"{"model_type":"llama"}"#).unwrap();
    d
}

/// `fingerprint`, for fixtures that are supposed to have one. It returns
/// `Option` so that a checkpoint it cannot identify fails closed rather than
/// collapsing to a constant every foreign sidecar matches.
fn fp(p: &std::path::Path) -> String {
    lightbulb::api::chat_template::fingerprint(p)
        .unwrap_or_else(|| panic!("no fingerprint for fixture {}", p.display()))
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
///
/// `resolved_by` comes back as the sidecar's OWN recorded value, not
/// `Resolution::Sidecar`. Spec §1 gives tier 0 the confidence "whatever
/// produced it — recorded", and §5 makes `resolved_by`/`evidence` the
/// load-bearing fields; flattening them would make a probed measurement and a
/// hand-written guess indistinguishable to Task 5's monitor. The `source`
/// assertion is what proves tier 0 outranks tier 1 now that the tier tag no
/// longer names the file it came from.
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
        model_fingerprint: fp(&d),
    };
    lightbulb::api::chat_template::write_sidecar(&d, &sc).unwrap();
    let t = lightbulb::api::chat_template::resolve(&d);
    assert_eq!(
        t.source, "FROM_SIDECAR",
        "the sidecar did not outrank tokenizer_config.json"
    );
    assert_eq!(
        t.resolved_by,
        Resolution::Probe,
        "resolution flattened the sidecar's recorded provenance; a probe and a \
         hand-written guess are now indistinguishable"
    );

    // A sidecar that records a guess reports a guess — the same file, the same
    // tier, a different confidence.
    let guess = lightbulb::api::chat_template::Sidecar {
        resolved_by: Resolution::Registry,
        evidence: "hand-written by an operator; not measured".into(),
        ..sc
    };
    lightbulb::api::chat_template::write_sidecar(&d, &guess).unwrap();
    let t = lightbulb::api::chat_template::resolve(&d);
    assert_eq!(t.source, "FROM_SIDECAR");
    assert_eq!(t.resolved_by, Resolution::Registry);
}

/// `SIDECAR_NAME` is part of the on-disk contract, not an internal detail: the
/// spec names it in §1 and §5, and Task 6's CLI writes it for an operator to go
/// and read. Renaming the constant is an API break for them, so the literal is
/// asserted here and the file is asserted to appear on disk under it.
#[test]
fn the_sidecar_filename_is_part_of_the_on_disk_contract() {
    assert_eq!(
        lightbulb::api::chat_template::SIDECAR_NAME,
        "lightbulb-chat-template.json"
    );

    let d = tmp_model_dir("sidecar-name");
    let sc = lightbulb::api::chat_template::Sidecar {
        template: "T".into(),
        resolved_by: Resolution::Probe,
        evidence: "e".into(),
        resolved_at: "2026-08-08T00:00:00Z".into(),
        model_fingerprint: fp(&d),
    };
    lightbulb::api::chat_template::write_sidecar(&d, &sc).unwrap();
    assert!(
        d.join("lightbulb-chat-template.json").is_file(),
        "write_sidecar did not produce {}/lightbulb-chat-template.json; contents: {:?}",
        d.display(),
        std::fs::read_dir(&d)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect::<Vec<_>>()
    );
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
        model_fingerprint: fp(&d),
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
///
/// **The CHECKPOINT is mutated, not the recorded fingerprint.** The previous
/// version of this test wrote a hard-coded wrong fingerprint into the sidecar,
/// which constrains only the `!=`: replacing `fingerprint`'s body with a
/// constant left it passing, and a constant is exactly the bug that shipped —
/// every path without a readable `config.json` hashed to `bd60acb658c79e45`.
/// Overwriting `config.json` puts the real function on the hook.
///
/// The `is_some` control is not redundant with the rejection: it separates
/// "rejected because the checkpoint changed" from "rejected because
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
        model_fingerprint: fp(&d),
    };
    lightbulb::api::chat_template::write_sidecar(&d, &good).unwrap();
    assert!(
        lightbulb::api::chat_template::read_sidecar(&d).is_some(),
        "a MATCHING sidecar was not accepted, so the rejection below would \
         prove nothing"
    );

    // Re-point the directory at another checkpoint, the way a re-download or a
    // moved symlink does. The sidecar on disk is untouched.
    std::fs::write(
        d.join("config.json"),
        br#"{"model_type":"llama","vocab_size":128256}"#,
    )
    .unwrap();
    assert_ne!(
        fp(&d),
        good.model_fingerprint,
        "fingerprint did not change when config.json did, so it is not \
         reading the checkpoint"
    );
    assert!(
        lightbulb::api::chat_template::read_sidecar(&d).is_none(),
        "a sidecar written for a different checkpoint was accepted"
    );
    let t = lightbulb::api::chat_template::resolve(&d);
    assert_ne!(t.source, "STALE", "resolution used a stale sidecar");
}

/// A checkpoint whose fingerprint cannot be computed accepts NO sidecar.
///
/// This is the shipped bug in its own right. `fingerprint` read `config.json`
/// through `unwrap_or_default()`, so every path without a readable one — two
/// unrelated empty directories among them — hashed to the same constant
/// `bd60acb658c79e45`, and a sidecar carrying it matched all of them. It was
/// then served at the highest-authority tier with `evidence` claiming it had
/// been measured on that model. Absent identity must fail closed.
#[test]
fn a_checkpoint_with_no_config_json_accepts_no_sidecar() {
    let a = tmp_model_dir("no-config-a");
    let b = tmp_model_dir("no-config-b");
    std::fs::remove_file(a.join("config.json")).unwrap();
    std::fs::remove_file(b.join("config.json")).unwrap();

    assert!(
        lightbulb::api::chat_template::fingerprint(&a).is_none(),
        "a directory with no config.json reported a fingerprint"
    );
    assert!(lightbulb::api::chat_template::fingerprint(&b).is_none());

    // The literal constant the old `unwrap_or_default()` produced, so this
    // fails if that behaviour ever comes back under another name.
    let foreign = lightbulb::api::chat_template::Sidecar {
        template: "FROM_SOME_OTHER_MODEL".into(),
        resolved_by: Resolution::Probe,
        evidence: "EOS 8/8 zephyr".into(),
        resolved_at: "2026-08-08T00:00:00Z".into(),
        model_fingerprint: "bd60acb658c79e45".into(),
    };
    lightbulb::api::chat_template::write_sidecar(&b, &foreign).unwrap();
    assert!(
        lightbulb::api::chat_template::read_sidecar(&b).is_none(),
        "a sidecar was accepted for a checkpoint that cannot be identified"
    );
    assert_ne!(
        lightbulb::api::chat_template::resolve(&b).source,
        "FROM_SOME_OTHER_MODEL"
    );
}

/// A `.gguf` file is a checkpoint too — `ModelRunner::start` accepts "either a
/// directory … or a `.gguf` file" — so it gets a real fingerprint and holds its
/// sidecar beside itself.
///
/// The swap below is the operator story from the review: probe a GGUF, later
/// drop a different model or quant at the same path. Under the constant
/// fingerprint the stale template was served at `Resolution::Sidecar` with
/// evidence claiming it had been measured on the new file.
///
/// The two file bodies differ in LENGTH deliberately: mtime resolution is
/// coarse enough that two writes this close together can share a timestamp, and
/// size alone must already separate them.
#[test]
fn a_gguf_file_is_fingerprinted_and_keeps_its_sidecar_beside_it() {
    let d = tmp_model_dir("gguf");
    let f = d.join("model-q4_k_m.gguf");
    std::fs::write(&f, b"GGUF\x03weights-v1").unwrap();

    let before = fp(&f);
    let sc = lightbulb::api::chat_template::Sidecar {
        template: "FROM_GGUF_SIDECAR".into(),
        resolved_by: Resolution::Probe,
        evidence: "EOS 8/8 zephyr".into(),
        resolved_at: "2026-08-08T00:00:00Z".into(),
        model_fingerprint: before.clone(),
    };
    lightbulb::api::chat_template::write_sidecar(&f, &sc).unwrap();
    assert_eq!(
        lightbulb::api::chat_template::resolve(&f).source,
        "FROM_GGUF_SIDECAR",
        "a .gguf checkpoint could not read back its own sidecar"
    );

    // A different model at the same path.
    std::fs::write(&f, b"GGUF\x03an-entirely-different-model-and-quantization").unwrap();
    assert_ne!(
        fp(&f),
        before,
        "swapping the .gguf file left the fingerprint unchanged"
    );
    assert!(
        lightbulb::api::chat_template::read_sidecar(&f).is_none(),
        "the previous model's sidecar was accepted for the new .gguf file"
    );
    assert_ne!(
        lightbulb::api::chat_template::resolve(&f).source,
        "FROM_GGUF_SIDECAR"
    );
}

/// A hand-edited sidecar that no longer parses is rejected rather than
/// panicking, and resolution carries on to the next tier.
#[test]
fn a_malformed_sidecar_is_rejected() {
    let d = tmp_model_dir("malformed");
    std::fs::write(
        d.join("tokenizer_config.json"),
        r#"{"chat_template":"FROM_TOKENIZER_CONFIG"}"#,
    )
    .unwrap();
    std::fs::write(
        d.join(lightbulb::api::chat_template::SIDECAR_NAME),
        "{ not json at all",
    )
    .unwrap();

    assert!(lightbulb::api::chat_template::read_sidecar(&d).is_none());
    let t = lightbulb::api::chat_template::resolve(&d);
    assert_eq!(t.resolved_by, Resolution::TokenizerConfig);
    assert_eq!(t.source, "FROM_TOKENIZER_CONFIG");
}

/// A multi-turn conversation, so the branches a single user message never
/// reaches are rendered too — LLAMA2's whole `{% else %}` arm, the one that
/// emits assistant turns and the EOS between them, is dead under `msgs()`.
fn convo() -> Vec<RawMessage> {
    vec![
        RawMessage {
            role: "user".into(),
            content: "Name the capital of France.".into(),
        },
        RawMessage {
            role: "assistant".into(),
            content: "Paris.".into(),
        },
        RawMessage {
            role: "user".into(),
            content: "And of Spain?".into(),
        },
    ]
}

/// Every registry constant's FULL rendered output, as an exact string.
///
/// The tail assertion below is kept — it is the Task 1 regression, where a
/// trailing newline sat in source position and was silently stripped — but a
/// tail is not a template. With only the tails pinned, replacing ZEPHYR's
/// entire per-message body with `{{ 'MANGLED' }}` left the suite passing, and
/// so did CHATML's, and so did turning LLAMA2's `[INST] ` into `XXX `. These
/// three constants are the only thing tiers 2 and 3 can EVER return, and a
/// wrong shape renders successfully: it does not fall through to another tier
/// and it is not logged. Exact output is the only assertion that sees it.
///
/// Parameterised over all three so adding a fourth without a render assertion
/// is not possible.
#[test]
fn every_registry_constant_renders_exactly() {
    use lightbulb::api::chat_template::registry;
    let expected = [
        (
            "zephyr",
            registry::ZEPHYR,
            "<|assistant|>\n",
            "<|user|>\nName the capital of France.</s>\n<|assistant|>\n",
            "<|user|>\nName the capital of France.</s>\n\
             <|assistant|>\nParis.</s>\n\
             <|user|>\nAnd of Spain?</s>\n\
             <|assistant|>\n",
        ),
        (
            "chatml",
            registry::CHATML,
            "<|im_start|>assistant\n",
            "<|im_start|>user\nName the capital of France.<|im_end|>\n\
             <|im_start|>assistant\n",
            "<|im_start|>user\nName the capital of France.<|im_end|>\n\
             <|im_start|>assistant\nParis.<|im_end|>\n\
             <|im_start|>user\nAnd of Spain?<|im_end|>\n\
             <|im_start|>assistant\n",
        ),
        (
            "llama2",
            registry::LLAMA2,
            "[/INST]",
            "[INST] Name the capital of France. [/INST]",
            "[INST] Name the capital of France. [/INST]Paris.</s>[INST] And of Spain? [/INST]",
        ),
    ];
    assert_eq!(
        expected.len(),
        registry::candidates().len(),
        "a candidate was added to the registry without a render assertion"
    );
    for (name, src, tail, one_turn, multi_turn) in expected {
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
        assert_eq!(out, one_turn, "{name} rendered the wrong prompt");
        assert_eq!(
            t.render(&convo(), "<s>", "</s>")
                .unwrap_or_else(|e| panic!("{name} failed to render a conversation: {e}")),
            multi_turn,
            "{name} rendered the wrong multi-turn prompt"
        );
    }
}

// ── Tier 3: which path component names the model ────────────────────────────
//
// `from_family` reads the PATH, so these need no filesystem — the literal is
// the whole input.

/// Path components are scanned leaf-first, and the first one matching any rule
/// wins.
///
/// The `qwen-experiments` case is the measured false positive: matching the
/// whole path as one lowercased string returned CHATML for a Llama-2
/// checkpoint. `api/mod.rs` builds the path as
/// `LIGHTBULB_MODELS_DIR.join(default_model)`, so a models root named `D:\qwen\`
/// mapped every model beneath it to one family — reported as
/// `Resolution::Registry`, which reads as "matched the model", not "matched
/// your folder layout".
///
/// Basename-only is not the fix, which is what the HF cache case pins: that
/// layout's leaf is a sha and names nothing.
#[test]
fn family_registry_scans_path_components_leaf_first() {
    use lightbulb::api::chat_template::registry::{self, from_family};
    use std::path::Path;

    let cases: [(&str, Option<&str>); 7] = [
        // The HF cache layout: the leaf is a snapshot sha, the name is two up.
        (
            "/home/x/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/\
             snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
            Some(registry::ZEPHYR),
        ),
        (
            "/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/abc123",
            Some(registry::CHATML),
        ),
        // The measured false positive: a parent must not outvote the model.
        (
            r"C:\models\qwen-experiments\Llama-2-7b-chat-hf",
            Some(registry::LLAMA2),
        ),
        (
            r"C:\models\qwen\Mistral-7B-Instruct-v0.3",
            Some(registry::LLAMA2),
        ),
        // A parent still answers when NOTHING nearer does — leaf-first is a
        // priority order, not a restriction to the leaf.
        (
            r"D:\qwen\some-unlabelled-checkpoint",
            Some(registry::CHATML),
        ),
        (
            r"C:\models\TinyLlama-1.1B-Chat-v1.0",
            Some(registry::ZEPHYR),
        ),
        // No component names a family anywhere.
        (r"C:\models\some-unlabelled-checkpoint", None),
    ];

    for (path, want) in cases {
        assert_eq!(
            from_family(Path::new(path)).as_deref(),
            want,
            "from_family({path:?}) picked the wrong family"
        );
    }
}

/// Rule order inside a single component. Every rule is exercised by a component
/// that satisfies it and at least one rule below it, so deleting or reordering
/// any of them fails here.
#[test]
fn family_registry_rule_precedence_within_one_component() {
    use lightbulb::api::chat_template::registry::{self, from_family};
    use std::path::Path;

    let cases: [(&str, Option<&str>); 5] = [
        // tinyllama+chat beats chatml…
        ("TinyLlama-1.1B-Chatml-merge", Some(registry::ZEPHYR)),
        // …and qwen beats llama-2.
        ("qwen-llama-2-merge", Some(registry::CHATML)),
        ("chatml-mistral-merge", Some(registry::CHATML)),
        // `tinyllama` alone is a BASE model: the `&& chat` is load-bearing, and
        // this must not reach ZEPHYR.
        ("TinyLlama-1.1B-intermediate-step-1431k", None),
        ("llama2-chat", Some(registry::LLAMA2)),
    ];

    for (name, want) in cases {
        assert_eq!(
            from_family(Path::new(name)).as_deref(),
            want,
            "from_family({name:?}) picked the wrong family"
        );
    }
}

/// No tier fires: resolution reports None rather than inventing a template.
///
/// The `from_family` precondition is asserted separately because tier 3 reads
/// the whole path: if the checkout ever lives under a directory named `qwen`,
/// `mistral` or the like, this test stops testing anything, and the failure
/// should say so rather than surface as a bare `Registry != None`.
#[test]
fn missing_everything_falls_through() {
    let d = tmp_model_dir("nothing");
    std::fs::write(d.join("config.json"), r#"{"model_type":"unknown-xyz"}"#).unwrap();
    assert!(
        lightbulb::api::chat_template::registry::from_family(&d).is_none(),
        "the fixture path {} names a model family, so no tier could be shown \
         to fall through",
        d.display()
    );
    let t = lightbulb::api::chat_template::resolve(&d);
    assert_eq!(t.resolved_by, Resolution::None);
    assert_eq!(t.source, "");
}

// ── Tier 1: the shapes `chat_template` actually takes on disk ───────────────

/// `tokenizer_config.json` may store `chat_template` as a LIST of named
/// templates rather than a string — Hermes-3 ships `default` and `tool_use` —
/// and `transformers` resolves that to the `default` entry.
///
/// Requiring `.as_str()` sent every such checkpoint to a registry guess
/// reported as a successful tier-3 match, with nothing logged: its own
/// authoritative template was sitting on disk, unread.
#[test]
fn tokenizer_config_list_form_selects_the_default_entry() {
    let d = tmp_model_dir("list-form");
    std::fs::write(
        d.join("tokenizer_config.json"),
        r#"{"chat_template":[
             {"name":"tool_use","template":"FROM_TOOL_USE"},
             {"name":"default","template":"FROM_DEFAULT"}
           ]}"#,
    )
    .unwrap();
    let t = lightbulb::api::chat_template::resolve(&d);
    assert_eq!(t.resolved_by, Resolution::TokenizerConfig);
    assert_eq!(
        t.source, "FROM_DEFAULT",
        "the list form did not select the entry named \"default\""
    );
}

/// A list with no usable `default` entry, and a `chat_template` that is neither
/// shape, both fall through to the next tier rather than resolving to something
/// invented. (Both log; the observable part is the fall-through.)
#[test]
fn an_unusable_tokenizer_config_chat_template_falls_through() {
    for body in [
        r#"{"chat_template":[{"name":"tool_use","template":"T"}]}"#,
        r#"{"chat_template":[{"name":"default","template":{"nested":"object"}}]}"#,
        r#"{"chat_template":{"default":"T"}}"#,
        r#"{"chat_template":null}"#,
    ] {
        let d = tmp_model_dir("unusable-tokenizer-config");
        std::fs::write(d.join("tokenizer_config.json"), body).unwrap();
        let t = lightbulb::api::chat_template::resolve(&d);
        assert_eq!(
            t.resolved_by,
            Resolution::None,
            "{body} resolved to something instead of falling through"
        );
    }
}

// ─── Special tokens ─────────────────────────────────────────────────────────
//
// Every fixture below uses tokens that are NOT `<s>` / `</s>`. That is the
// whole point: an implementation that hardcodes the Llama-2 pair — which is
// what the plan for this task specified — passes any test written against
// TinyLlama, because TinyLlama's tokens happen to BE that pair. Only a
// checkpoint whose tokens differ can tell the two implementations apart.

/// Tier 1: `tokenizer_config.json` wins, in BOTH shapes it is written in.
///
/// `bos_token` here is a bare string and `eos_token` a serialised `AddedToken`
/// object, because both occur on the Hub — Llama-3.1-Instruct writes strings,
/// TinyLlama's own Hub config writes objects. Handling only strings sends the
/// object-shaped half of the Hub to the id fallback with nothing logged.
#[test]
fn tokenizer_config_special_tokens_are_read_in_both_shapes() {
    let d = tmp_model_dir("llama3-tokens");
    std::fs::write(
        d.join("tokenizer_config.json"),
        r#"{"bos_token":"<|begin_of_text|>",
            "eos_token":{"content":"<|eot_id|>","lstrip":false,"normalized":false,
                         "rstrip":false,"single_word":false}}"#,
    )
    .unwrap();
    let t = lightbulb::api::chat_template::special_tokens(&d);
    assert_eq!(t.bos, "<|begin_of_text|>");
    assert_eq!(
        t.eos, "<|eot_id|>",
        "an eos_token written as an AddedToken object was not read"
    );
}

/// Tier 2: no `tokenizer_config.json` at all — ids from `config.json` resolved
/// through `tokenizer.json`'s `added_tokens`.
///
/// This is the tier that serves the checkpoint this project actually tests
/// against: the TinyLlama snapshot ships no `tokenizer_config.json`. The
/// fixture uses Qwen's ids and tokens rather than TinyLlama's so that a
/// hardcoded `<s>`/`</s>` cannot pass it.
#[test]
fn token_ids_resolve_through_added_tokens() {
    let d = tmp_model_dir("id-fallback");
    std::fs::write(
        d.join("config.json"),
        r#"{"model_type":"qwen2","bos_token_id":151643,"eos_token_id":151645}"#,
    )
    .unwrap();
    std::fs::write(
        d.join("tokenizer.json"),
        r#"{"added_tokens":[
             {"id":151643,"content":"<|endoftext|>","special":true},
             {"id":151644,"content":"<|im_start|>","special":true},
             {"id":151645,"content":"<|im_end|>","special":true}]}"#,
    )
    .unwrap();
    let t = lightbulb::api::chat_template::special_tokens(&d);
    assert_eq!(t.bos, "<|endoftext|>");
    assert_eq!(t.eos, "<|im_end|>");
}

/// A list-valued `eos_token_id` — Llama-3.1 ships `[128001, 128008, 128009]` —
/// resolves rather than being dropped on the floor.
#[test]
fn a_list_valued_token_id_resolves_to_its_first_entry() {
    let d = tmp_model_dir("list-eos-id");
    std::fs::write(
        d.join("config.json"),
        r#"{"model_type":"llama","bos_token_id":128000,"eos_token_id":[128001,128008,128009]}"#,
    )
    .unwrap();
    std::fs::write(
        d.join("tokenizer.json"),
        r#"{"added_tokens":[
             {"id":128000,"content":"<|begin_of_text|>","special":true},
             {"id":128001,"content":"<|end_of_text|>","special":true},
             {"id":128009,"content":"<|eot_id|>","special":true}]}"#,
    )
    .unwrap();
    let t = lightbulb::api::chat_template::special_tokens(&d);
    assert_eq!(t.bos, "<|begin_of_text|>");
    assert_eq!(t.eos, "<|end_of_text|>");
}

/// A checkpoint declaring neither gets EMPTY strings — never another model's
/// tokens.
///
/// The assertion is equality with the empty pair, not merely `!= "</s>"`: the
/// documented fallback is "invent nothing", and any non-empty default is some
/// specific family's token being put into every other family's prompt.
#[test]
fn an_undeclared_checkpoint_gets_no_tokens_rather_than_another_models() {
    let d = tmp_model_dir("no-tokens");
    // tmp_model_dir writes `{"model_type":"llama"}` — no ids, no tokenizer.json.
    let t = lightbulb::api::chat_template::special_tokens(&d);
    assert_eq!(t, lightbulb::api::chat_template::SpecialTokens::default());
}

/// The load-bearing one: rendering uses the CHECKPOINT'S tokens.
///
/// A single call chain — `resolve_for_model` then `render` — with a Llama-3
/// template and Llama-3 tokens. An implementation that renders with a hardcoded
/// `"<s>"`/`"</s>"` produces `…France.</s><|start_header_id|>assistant…`, which
/// still renders successfully and is never logged, so only an exact-match
/// assertion here separates the two.
#[test]
fn rendering_uses_the_checkpoints_own_tokens() {
    let d = tmp_model_dir("render-with-own-tokens");
    std::fs::write(
        d.join("tokenizer_config.json"),
        r#"{"bos_token":"<|begin_of_text|>",
            "eos_token":"<|eot_id|>",
            "chat_template":"{{ bos_token }}{% for m in messages %}{{ '<|start_header_id|>' + m.role + '<|end_header_id|>\n\n' + m.content + eos_token }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"}"#,
    )
    .unwrap();

    let r = lightbulb::api::chat_template::resolve_for_model(&d);
    assert_eq!(r.resolved_by(), Resolution::TokenizerConfig);

    let out = r.render(&msgs()).expect("render failed");
    assert_eq!(
        out,
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
         Name the capital of France.<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n"
    );
    assert!(
        !out.contains("</s>") && !out.contains("<s>"),
        "a Llama-2 special token reached a Llama-3 prompt: {out:?}"
    );
}
