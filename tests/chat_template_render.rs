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

/// A `.gguf` checkpoint reads its companion JSON from BESIDE the file.
///
/// `special_tokens` and `resolve` both did `model_path.join(...)`, producing
/// `foo.gguf/tokenizer_config.json` — a path that cannot exist, because
/// `foo.gguf` is a file. Every GGUF checkpoint therefore got `bos: ""`,
/// `eos: ""` and a registry-guess template on every request, while the warning
/// told the operator to add a `tokenizer_config.json` the code would never have
/// looked for. `fingerprint` and `sidecar_path` had already branched on
/// `is_file`; these two had not.
///
/// The assertions are exact strings, not "non-empty": an empty pair is the
/// value the defect produced, so `!= ""` would be satisfied by any accident.
#[test]
fn a_gguf_checkpoint_reads_its_metadata_from_beside_the_file() {
    let d = tmp_model_dir("gguf-metadata");
    let f = d.join("qwen3-8b-q4_k_m.gguf");
    std::fs::write(&f, b"GGUF\x03weights").unwrap();

    // Beside the file, which is where a GGUF download that ships HF metadata
    // puts it — and where `sidecar_path` already looks for the sidecar.
    std::fs::write(
        d.join("tokenizer_config.json"),
        r#"{"bos_token":"<|endoftext|>",
            "eos_token":"<|im_end|>",
            "chat_template":"FROM_GGUF_TOKENIZER_CONFIG"}"#,
    )
    .unwrap();

    let t = lightbulb::api::chat_template::special_tokens(&f);
    assert_eq!(t.bos, "<|endoftext|>");
    assert_eq!(t.eos, "<|im_end|>");

    // The same path defect sat in `resolve`'s tier 1. Left alone, a GGUF
    // checkpoint that ships its own authoritative template would still have
    // been served a family guess, reported as a successful match.
    let r = lightbulb::api::chat_template::resolve(&f);
    assert_eq!(r.source, "FROM_GGUF_TOKENIZER_CONFIG");
    assert_eq!(r.resolved_by, Resolution::TokenizerConfig);
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

/// Every `chat_template` tier 1 cannot use falls through to the next tier,
/// which then serves a REAL template — rather than tier 1 resolving to
/// something invented or to nothing at all.
///
/// The malformed shapes are the four `chat_template_field` was written against.
/// The **blank** shapes are the fifth case and were the defect: nothing about a
/// blank string is malformed, so `chat_template_field` returned it, resolution
/// stopped at `Resolution::TokenizerConfig`, and every request rendered to an
/// empty prompt — the model asked to continue nothing. `"   "` and `"\n\t "` are
/// listed beside `""` because a truncated or hand-edited file leaves whitespace
/// far more often than it leaves exactly zero bytes, so `is_empty()` would be
/// the plausible half-fix.
///
/// **The fixture resolves at tier 3, and the assertion is on the SOURCE.** An
/// assertion that `resolved_by != TokenizerConfig` would hold for a fixture
/// that fell through to nothing, which is a worse outcome than the bug for a
/// chat model — a registry guess beats the legacy join. The per-iteration
/// control (a usable template in the same file, in the same directory, asserted
/// to win first) is what stops this passing against an implementation where
/// tier 1 never fires at all.
#[test]
fn an_unusable_tokenizer_config_chat_template_falls_through() {
    use lightbulb::api::chat_template::registry;
    for body in [
        // Malformed: no usable `default` entry, or not either shape at all.
        r#"{"chat_template":[{"name":"tool_use","template":"T"}]}"#,
        r#"{"chat_template":[{"name":"default","template":{"nested":"object"}}]}"#,
        r#"{"chat_template":{"default":"T"}}"#,
        r#"{"chat_template":null}"#,
        // Blank: well-formed, and renders to an empty prompt.
        r#"{"chat_template":""}"#,
        r#"{"chat_template":"   "}"#,
        r#"{"chat_template":"\n\t "}"#,
        // Blank in the list form too — the `default` entry is found and is
        // still unusable.
        r#"{"chat_template":[{"name":"default","template":"  \n"}]}"#,
    ] {
        // `tinyllama`+`chat` in the leaf is what `registry::from_family`
        // matches, so tier 3 has a real answer for this directory.
        let d = tmp_model_dir("tinyllama-chat-unusable-tokenizer-config");

        // Control: tier 1 demonstrably fires here, for this file, in this
        // directory. Without it, "fell through" and "was never read" look the
        // same.
        std::fs::write(
            d.join("tokenizer_config.json"),
            r#"{"chat_template":"FROM_TOKENIZER_CONFIG"}"#,
        )
        .unwrap();
        let control = lightbulb::api::chat_template::resolve(&d);
        assert_eq!(
            control.source, "FROM_TOKENIZER_CONFIG",
            "tier 1 does not fire for this fixture, so the assertions below \
             would prove nothing about falling THROUGH it"
        );

        std::fs::write(d.join("tokenizer_config.json"), body).unwrap();
        let t = lightbulb::api::chat_template::resolve(&d);
        assert_eq!(
            t.source,
            registry::ZEPHYR,
            "{body} did not fall through to the tier-3 template; got {:?} at {:?}",
            t.source,
            t.resolved_by
        );
        assert_eq!(t.resolved_by, Resolution::Registry, "for {body}");
    }
}

/// A blank template is unusable at tier 0 as well — a corrupted or hand-edited
/// sidecar does not get to serve an empty prompt.
///
/// Tier 0 matters more than tier 1 here, not less: the sidecar is the
/// highest-authority tier, read before the checkpoint's own
/// `tokenizer_config.json`, so a blank one would outrank every real template on
/// disk on every subsequent server start.
///
/// The fingerprint is written correctly, so the sidecar is one this checkpoint
/// would otherwise accept — `stale_sidecar_is_rejected` covers the other
/// rejection, and a fixture rejected for two reasons at once would pass this
/// test with the blank check deleted.
#[test]
fn a_blank_sidecar_template_falls_through() {
    use lightbulb::api::chat_template::{Sidecar, registry};
    for blank in ["", "   ", "\n\t "] {
        let d = tmp_model_dir("tinyllama-chat-blank-sidecar");
        let sc = Sidecar {
            template: blank.to_string(),
            resolved_by: Resolution::Probe,
            evidence: "hand-edited down to whitespace".into(),
            resolved_at: "2026-08-13T00:00:00Z".into(),
            model_fingerprint: fp(&d),
        };
        lightbulb::api::chat_template::write_sidecar(&d, &sc).unwrap();
        assert!(
            lightbulb::api::chat_template::read_sidecar(&d).is_none(),
            "a sidecar whose template is {blank:?} was accepted"
        );
        let t = lightbulb::api::chat_template::resolve(&d);
        assert_eq!(
            t.source,
            registry::ZEPHYR,
            "a sidecar whose template is {blank:?} was served instead of falling \
             through to tier 3; got {:?} at {:?}",
            t.source,
            t.resolved_by
        );
    }
}

/// The serving chokepoint refuses a blank template even if one reaches it.
///
/// A second layer over the tier-level rule above, guarding the promise
/// `resolve_for_serving`'s doc makes to every handler: what it hands back
/// renders to something. The tiers are the real fix — they fall THROUGH, and
/// the assertions above are that they reach a real template — but a tier added
/// later that skips `non_blank` would reopen the hole silently, and this is the
/// one function every handler goes through.
///
/// Asserted against `ChatTemplate::is_usable` on hand-built values rather than
/// through `resolve_for_serving`, because no fixture on disk can produce a
/// blank template any more. A test driven through the file system would pass
/// with the backstop deleted, which is the one thing a backstop's test must not
/// do.
#[test]
fn a_blank_template_is_not_usable_whatever_tier_claims_to_have_found_it() {
    use lightbulb::api::chat_template::ChatTemplate;
    let usable = |source: &str, resolved_by| {
        ChatTemplate {
            source: source.to_string(),
            resolved_by,
        }
        .is_usable()
    };
    for blank in ["", "   ", "\n\t "] {
        // Every tier, not just one: the backstop is about a tier ADDED later,
        // so a rule that happened to be spelled `resolved_by == Registry` would
        // be exactly the wrong shape.
        for tier in [
            Resolution::Sidecar,
            Resolution::TokenizerConfig,
            Resolution::VocabSignature,
            Resolution::Registry,
            Resolution::Probe,
        ] {
            assert!(
                !usable(blank, tier),
                "a blank template ({blank:?}) claiming {tier:?} was treated as usable"
            );
            // Control: the same tier with a real source IS usable, so the
            // refusal above is about the source and not about the tier.
            assert!(
                usable("T", tier),
                "a real template claiming {tier:?} was treated as unusable"
            );
        }
    }
    assert!(
        !usable("T", Resolution::None),
        "`Resolution::None` must stay unusable however non-blank the source is"
    );
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

/// How much `render` costs per request, since it recompiles the template every
/// call.
///
/// **Not a gate — it asserts nothing about time**, because a wall-clock
/// threshold on a shared developer box is flaky in the direction that wastes
/// the most attention. It exists so the decision NOT to cache the compiled
/// template stays a measured claim rather than a judgement, and so re-checking
/// it is one command instead of a rewrite:
///
/// ```text
/// cargo test --release --test chat_template_render measure_render -- --ignored --nocapture
/// ```
///
/// Measured 2026-08-12, `--release`, 10 000 calls after a 100-call warmup:
///
/// | template | size | per call |
/// | --- | --- | --- |
/// | `registry::ZEPHYR` (what TinyLlama resolves to) | 114 B | 40, 62 µs |
/// | the Llama-3-shaped one below | 972 B | 77, 93, 100, 115, 119, 221 µs |
///
/// **Every run of the same code is listed rather than averaged**, because the
/// spread IS the result: six runs of the 972 B template span 77–221 µs, a
/// factor of 2.9. A single figure from this harness means little, and the first
/// number recorded here was 221 — the outlier — which would have overstated the
/// cost threefold had the run not been repeated. That spread is the reason no
/// threshold is asserted, and the reason the decision below is argued from
/// order of magnitude rather than from any one measurement.
///
/// **Decision: do not cache.** `render` is called once per request, against a
/// request whose generation cost is three to five orders of magnitude larger —
/// 8 tokens at this box's measured ~3.5 s/token on CPU is ~28 s, and even a
/// 10 ms/token GPU decode is ~80 ms. Taking the *worst* observed 221 µs against
/// the *fastest* of those is ~0.3 %. Caching would mean holding a compiled
/// `minijinja::Template`, which borrows from its `Environment`, so
/// `ResolvedTemplate` would have to own both and carry the lifetime — real
/// complexity against a rounding error.
///
/// **What would change the decision**, stated so it can be checked rather than
/// re-litigated: a template large enough to push this past ~1 % of decode (call
/// it >1 ms, roughly a 5 KB template — real Llama-3.1 templates with full tool
/// handling approach 4 KB, so this is not far off), or `render` moving to a
/// per-token rather than per-request call site. Neither holds today.
#[test]
#[ignore = "measurement, not a gate"]
fn measure_render_cost() {
    // Meta-Llama-3.1's template, abridged to its structural shape: a tool/system
    // preamble, `strftime_now`, and a per-message loop with a branch. A size-
    // class proxy, deliberately not a verbatim copy — the fidelity claims in
    // this file belong to the tests that assert output, not to this one.
    let src = r#"{%- set date_string = strftime_now("%d %b %Y") %}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}
{{- bos_token }}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{{- system_message }}
{{- "<|eot_id|>" }}
{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool') %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}
    {%- else %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {{- message.content | trim }}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"#;

    let t = lightbulb::api::chat_template::ResolvedTemplate {
        template: tmpl(src),
        tokens: lightbulb::api::chat_template::SpecialTokens {
            bos: "<|begin_of_text|>".to_string(),
            eos: "<|eot_id|>".to_string(),
        },
    };
    let msgs = vec![
        lightbulb::contracts::validation::RawMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
        },
        lightbulb::contracts::validation::RawMessage {
            role: "user".to_string(),
            content: "Name the capital of France.".to_string(),
        },
    ];

    // The render must actually succeed, or the timing below measures an error
    // path. This is the one thing here that IS asserted.
    let out = t
        .render(&msgs)
        .expect("the measurement template must render");
    assert!(
        out.contains("Name the capital of France."),
        "the measurement template rendered without the message: {out:?}"
    );

    for _ in 0..100 {
        let _ = t.render(&msgs).unwrap();
    }
    let n = 10_000;
    let start = std::time::Instant::now();
    for _ in 0..n {
        let _ = std::hint::black_box(t.render(&msgs).unwrap());
    }
    eprintln!(
        "MEASURED render(): {:?} per call over {n} calls, source {} bytes",
        start.elapsed() / n,
        src.len()
    );
}

// ─── The probe's report and its selection rule ──────────────────────────────
//
// Generating under a candidate template needs a model, so THAT part has no
// automated test — see `src/bin/lightbulb-probe.rs`. Formatting and selection
// do not, which is why `ProbeRow`, `format_probe_report` and
// `select_probe_winner` live in the library rather than in the binary:
// `src/bin/*.rs` is a separate crate and nothing here could reach into one.
// While the selection `match` sat in the binary it was untested, and it is the
// step that decides what gets persisted at `Resolution::Probe` — the tier
// `resolve` consults before every other one.

/// The table an operator reads must show every candidate and name no winner.
///
/// **Asserted as one whole string, not with `contains`.** This file's header
/// gives the general rule; here it is load-bearing four separate times. A
/// `contains`-based version of this test survives ALL of: swapping the yes/no
/// literals, deleting the tokens column, reversing the row order, and pairing
/// every row's outcome with the wrong candidate.
#[test]
fn probe_report_renders_every_candidate_and_picks_no_winner() {
    use lightbulb::api::chat_template::{ProbeRow, format_probe_report};
    let rows = vec![
        ProbeRow {
            candidate: "zephyr",
            source: "SRC_Z",
            stopped_on_eos: true,
            tokens_generated: 8,
            first_line: "The capital of France is Paris.".to_string(),
        },
        ProbeRow {
            candidate: "chatml",
            source: "SRC_C",
            stopped_on_eos: false,
            tokens_generated: 48,
            first_line: "The capital of France is Paris. The capital of".to_string(),
        },
        ProbeRow {
            candidate: "llama2",
            source: "SRC_L",
            stopped_on_eos: false,
            tokens_generated: 48,
            first_line: "user The capital of France is Paris, and the".to_string(),
        },
    ];

    // Every column, every row, in order.
    assert_eq!(
        format_probe_report(&rows),
        "candidate   EOS   tokens  output\n\
         zephyr      yes   8       The capital of France is Paris.\n\
         chatml      no    48      The capital of France is Paris. The capital of\n\
         llama2      no    48      user The capital of France is Paris, and the\n"
    );

    // Separate claim, so it is not folded into the layout equality: the report
    // must not announce a conclusion. Selection is a separate step
    // (`select_probe_winner`), and an operator has to be able to read a
    // two-winner or zero-winner board for what it is.
    let lower = format_probe_report(&rows).to_lowercase();
    for banned in ["winner", "best", "recommend", "selected"] {
        assert!(
            !lower.contains(banned),
            "the report named a winner ({banned}):\n{lower}"
        );
    }

    // The template SOURCE must never reach the operator's table — it is a
    // multi-line Jinja blob that would destroy the layout. This is why
    // `ProbeRow` carries it for the selection step rather than the formatter
    // printing it.
    for src in ["SRC_Z", "SRC_C", "SRC_L"] {
        assert!(
            !format_probe_report(&rows).contains(src),
            "{src} leaked into the report"
        );
    }
}

/// A generation that opens with a newline must still show the operator its text.
///
/// Measured on the first live probe run (2026-08-13): the `llama2` candidate
/// generated 17 tokens and rendered as an EMPTY cell. Its template ends
/// `[/INST]` with no trailing newline, so the model began its reply with one,
/// and `lines().next()` returned `""` — literally the first line, and useless.
/// The blank landed on the row whose behaviour differed most from the others,
/// which is the row an operator most needs to read.
///
/// Asserted against exact strings, not `!is_empty()`: an empty result is what
/// the defect produced, so any non-empty accident would satisfy a weaker check.
#[test]
fn first_line_of_skips_leading_blank_lines() {
    use lightbulb::api::chat_template::first_line_of;

    // The defect, verbatim.
    assert_eq!(
        first_line_of("\nThe capital of France is Paris."),
        "The capital of France is Paris."
    );
    // Several blank lines, and whitespace-only ones, which `is_empty()` misses.
    assert_eq!(first_line_of("\n   \n\nParis."), "Paris.");
    // Unchanged when there is no leading blank.
    assert_eq!(first_line_of("Paris.\nmore"), "Paris.");
    // Genuinely empty output stays empty — that is a real result, not a bug.
    assert_eq!(first_line_of(""), "");
    assert_eq!(first_line_of("\n  \n"), "");
    // Truncation still applies, and applies to the line that is actually shown.
    let long = "\n".to_string() + &"x".repeat(80);
    let out = first_line_of(&long);
    assert_eq!(out.chars().count(), 60);
    assert!(out.ends_with("..."), "{out:?}");
}

/// Build one probe row. `source` differs per candidate on purpose: it is what
/// `Sidecar.template` is written from, so a selection that returned the right
/// NAME carried on the wrong row would still write the wrong template.
fn probe_row(
    candidate: &'static str,
    source: &'static str,
    stopped_on_eos: bool,
    tokens_generated: usize,
) -> lightbulb::api::chat_template::ProbeRow {
    lightbulb::api::chat_template::ProbeRow {
        candidate,
        source,
        stopped_on_eos,
        tokens_generated,
        first_line: format!("{candidate} said something"),
    }
}

/// Exactly one candidate fired: that candidate wins — and it is NOT `rows[0]`.
///
/// **The row identity is asserted, not `is_ok()`.** The probe writes
/// `winner.source` into the sidecar at `Resolution::Probe`, so "an Ok" and "the
/// right Ok" are entirely different claims. `rows[0]` is deliberately a loser
/// here, which is what makes an implementation returning the first row fail
/// this test instead of passing it by luck.
#[test]
fn probe_selection_returns_the_one_candidate_that_stopped_on_eos() {
    use lightbulb::api::chat_template::select_probe_winner;

    let rows = vec![
        probe_row("zephyr", "SRC_Z", false, 48),
        probe_row("chatml", "SRC_C", true, 10),
        probe_row("llama2", "SRC_L", false, 48),
    ];
    let winner =
        select_probe_winner(&rows).expect("one candidate fired, so selection must succeed");

    // Pointer identity first: it pins the returned reference to a specific
    // element, which no field comparison can be fooled about.
    assert!(
        std::ptr::eq(winner, &rows[1]),
        "selection returned {:?}, not the row that fired",
        winner.candidate
    );
    assert_eq!(winner.candidate, "chatml");
    // The source is the value that reaches disk. Asserted separately so a
    // regression that decoupled name from source names itself.
    assert_eq!(winner.source, "SRC_C");
    assert_eq!(winner.tokens_generated, 10);
}

/// No candidate fired: refuse, and say *that* rather than something else.
///
/// The message is asserted because "an Err" is not the property — an
/// implementation that folded this board into the tie message would refuse
/// correctly and then tell the operator "0 candidates stopped on EOS ()",
/// which reads as a formatting bug rather than the real finding: none of the
/// three registry templates suits this checkpoint.
#[test]
fn probe_selection_refuses_when_no_candidate_stopped_on_eos() {
    use lightbulb::api::chat_template::select_probe_winner;

    let rows = vec![
        probe_row("zephyr", "SRC_Z", false, 48),
        probe_row("chatml", "SRC_C", false, 48),
        probe_row("llama2", "SRC_L", false, 48),
    ];
    let err = match select_probe_winner(&rows) {
        Ok(w) => panic!(
            "selection picked {} from a board where nothing fired",
            w.candidate
        ),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        err.contains("no candidate stopped on EOS"),
        "the zero-winner refusal did not say so: {err}"
    );
    // The operator must be told nothing was persisted; the probe's conclusion
    // outranks every other tier, so silence here reads as "it wrote something".
    assert!(
        err.contains("Nothing was written."),
        "the zero-winner refusal did not say nothing was written: {err}"
    );
    // And it must not have degenerated into the tie wording.
    assert!(
        !err.contains("cannot choose between them"),
        "the zero-winner board was reported as a tie: {err}"
    );
}

/// Two candidates fired: refuse. Do not tie-break, do not take the first.
///
/// This is the board that separates "requires a unique winner" from "finds a
/// winner": an implementation built on `find`/`first` returns a perfectly
/// plausible row here and writes it to disk at the highest-authority tier.
#[test]
fn probe_selection_refuses_a_two_candidate_tie() {
    use lightbulb::api::chat_template::select_probe_winner;

    let rows = vec![
        probe_row("zephyr", "SRC_Z", true, 14),
        probe_row("chatml", "SRC_C", true, 10),
        probe_row("llama2", "SRC_L", false, 48),
    ];
    let err = match select_probe_winner(&rows) {
        Ok(w) => panic!("selection guessed {} from a two-way tie", w.candidate),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        err.contains("2 candidates stopped on EOS"),
        "the tie refusal did not report the count: {err}"
    );
    // Both tied candidates named, so the operator can see WHICH tie it was.
    assert!(
        err.contains("zephyr") && err.contains("chatml"),
        "the tie refusal did not name both candidates: {err}"
    );
    // The candidate that did NOT fire must not be named as a tie participant.
    assert!(
        !err.contains("llama2"),
        "the tie refusal named a candidate that never fired: {err}"
    );
    assert!(
        err.contains("will not guess"),
        "the tie refusal did not say it will not guess: {err}"
    );
    assert!(
        err.contains("Nothing was written."),
        "the tie refusal did not say nothing was written: {err}"
    );
}

/// THREE candidates fired: still refuse, and count them correctly.
///
/// **This is the real TinyLlama board, measured 2026-08-13** — zephyr stopped
/// on EOS in 14 tokens, chatml in 10, llama2 in 3 — not a hypothetical extra
/// case. An implementation that handles "a tie" as "exactly two" mis-handles
/// the only board this probe has ever actually produced against a real
/// checkpoint, which is why the count is asserted rather than just the `Err`.
#[test]
fn probe_selection_refuses_a_three_candidate_tie() {
    use lightbulb::api::chat_template::select_probe_winner;

    let rows = vec![
        probe_row("zephyr", "SRC_Z", true, 14),
        probe_row("chatml", "SRC_C", true, 10),
        probe_row("llama2", "SRC_L", true, 3),
    ];
    let err = match select_probe_winner(&rows) {
        Ok(w) => panic!("selection guessed {} from a three-way tie", w.candidate),
        Err(e) => format!("{e:#}"),
    };
    assert!(
        err.contains("3 candidates stopped on EOS"),
        "the three-way tie was not counted as three: {err}"
    );
    assert!(
        err.contains("zephyr") && err.contains("chatml") && err.contains("llama2"),
        "the three-way tie refusal did not name all three candidates: {err}"
    );
    assert!(
        err.contains("Nothing was written."),
        "the three-way tie refusal did not say nothing was written: {err}"
    );
}

// ─── `--accept`: the operator reads the board and names the winner ──────────
//
// `select_probe_winner` above has been measured to be inert: on both
// checkpoints it has ever run against, every candidate fired EOS and it refused
// as a tie, while the OUTPUT column separated the candidates immediately to a
// human. `select_named_candidate` carries that human verdict, so it must NOT
// re-apply the discriminator that was measured not to discriminate. The four
// tests below are each killed by a different wrong implementation; the mutation
// matrix is in the commit message.

/// The named candidate is returned — located in `rows`, never re-derived from
/// the registry.
///
/// **The board is deliberately NOT in `registry::candidates()` order.** The
/// function is handed a slice and its answer must come from that slice: an
/// implementation that finds the name's position in `registry::candidates()` and
/// then indexes `rows` with it agrees with a correct one for exactly as long as
/// the probe happens to build its board in registry order, and writes a
/// different template's source to disk the moment that stops holding. That is
/// the re-look-up hazard `ProbeRow::source` exists to remove, and `zephyr` sits
/// at registry index 0 and row index 2 here so that implementation fails.
#[test]
fn probe_accept_returns_the_named_row_from_the_rows_not_the_registry() {
    use lightbulb::api::chat_template::select_named_candidate;

    let rows = vec![
        probe_row("llama2", "SRC_L", false, 48),
        probe_row("chatml", "SRC_C", false, 48),
        probe_row("zephyr", "SRC_Z", true, 8),
    ];
    let picked = select_named_candidate(&rows, "zephyr")
        .expect("zephyr is on the board, so --accept zephyr must succeed");

    // Pointer identity first: it pins the returned reference to one element,
    // which no field comparison can be fooled about.
    assert!(
        std::ptr::eq(picked, &rows[2]),
        "--accept zephyr returned the {:?} row",
        picked.candidate
    );
    assert_eq!(picked.candidate, "zephyr");
    // The source is the value that reaches disk as `Sidecar.template`. Asserted
    // separately, because returning the right NAME carried on the wrong row is
    // precisely how a wrong template gets persisted at `Resolution::Probe`.
    assert_eq!(picked.source, "SRC_Z");
    assert_eq!(picked.tokens_generated, 8);
}

/// **A candidate that did NOT stop on EOS is still accepted.** This is the
/// feature.
///
/// An implementation that filters by `stopped_on_eos` before matching the name
/// passes every other test in this file and fails only this one. It would also
/// re-impose the exact rule `--accept` exists to override, and would do it in
/// the case the operator most needs: a checkpoint that never stops — its EOS
/// arrives as ordinary text, or the answer ran past the 48-token budget — where
/// the correct candidate is obvious from what it wrote and invisible in the EOS
/// column.
///
/// The two candidates that DID fire are on the board as wrong answers, so an
/// implementation that quietly substitutes a firing row returns something
/// plausible rather than erroring, and is caught by identity rather than by
/// `is_err()`.
#[test]
fn probe_accept_takes_a_candidate_that_never_stopped_on_eos() {
    use lightbulb::api::chat_template::{ProbeRow, select_named_candidate};

    let rows = vec![
        ProbeRow {
            candidate: "zephyr",
            source: "SRC_Z",
            stopped_on_eos: true,
            tokens_generated: 2,
            first_line: "<|user|>".to_string(),
        },
        ProbeRow {
            candidate: "chatml",
            source: "SRC_C",
            stopped_on_eos: true,
            tokens_generated: 3,
            first_line: "assistant assistant".to_string(),
        },
        ProbeRow {
            candidate: "llama2",
            source: "SRC_L",
            stopped_on_eos: false,
            tokens_generated: 48,
            first_line: "Tokyo is the capital of Japan. Japan is an island".to_string(),
        },
    ];
    let picked = select_named_candidate(&rows, "llama2")
        .expect("--accept must not require the named candidate to have stopped on EOS");

    assert!(
        std::ptr::eq(picked, &rows[2]),
        "--accept llama2 returned the {:?} row — the EOS flag was consulted",
        picked.candidate
    );
    assert_eq!(picked.candidate, "llama2");
    assert_eq!(picked.source, "SRC_L");
    // Restated as a field, so the failure reads as what it is rather than as a
    // pointer mismatch: the accepted row is the one that never fired.
    assert!(
        !picked.stopped_on_eos,
        "the row --accept returned had stopped_on_eos set, so this board no \
         longer tests the override at all"
    );
}

/// An unknown name is refused, and the refusal lists the names that would work.
///
/// The operator is typing a name they read off a table one screen up, so a bare
/// "not found" leaves them guessing at the spelling of the thing just printed.
/// The message content is asserted, not `is_err()`: an implementation that
/// refused with an empty message would satisfy the weaker check while being the
/// whole defect.
///
/// The valid names are also checked against `registry::candidates()` itself, so
/// adding a fourth candidate without adding it to the message fails here rather
/// than in front of an operator.
#[test]
fn probe_accept_refuses_an_unknown_candidate_and_names_the_valid_ones() {
    use lightbulb::api::chat_template::{registry, select_named_candidate};

    let rows = vec![
        probe_row("zephyr", "SRC_Z", true, 14),
        probe_row("chatml", "SRC_C", true, 10),
        probe_row("llama2", "SRC_L", true, 3),
    ];
    let err = match select_named_candidate(&rows, "chatlm") {
        Ok(r) => panic!("--accept chatlm was honoured, returning {}", r.candidate),
        Err(e) => format!("{e:#}"),
    };

    // What was rejected, so the operator sees their own typo.
    assert!(
        err.contains("chatlm"),
        "the refusal did not repeat the name that was rejected: {err}"
    );
    // Literal, so an empty `candidates()` could not satisfy the loop below
    // vacuously.
    for valid in ["zephyr", "chatml", "llama2"] {
        assert!(
            err.contains(valid),
            "the refusal did not name the valid candidate {valid}: {err}"
        );
    }
    // And tied to the source of truth `--accept` is actually checked against.
    for (name, _) in registry::candidates() {
        assert!(
            err.contains(name),
            "`{name}` is a registry candidate that the refusal does not list: {err}"
        );
    }
    // Same promise every other probe refusal makes; this one runs after three
    // generations, so silence here reads as "it wrote something".
    assert!(
        err.contains("Nothing was written."),
        "the unknown-candidate refusal did not say nothing was written: {err}"
    );
}

/// The real SmolLM2-360M-Instruct board, measured 2026-08-12 — the measurement
/// `--accept` exists because of.
///
/// `<|im_start|>`/`<|im_end|>` are genuine added tokens there and the checkpoint
/// ships its own ChatML template, so `chatml` is the known-correct answer
/// independently of anything the probe does. All three candidates stopped on
/// EOS, so `select_probe_winner` refuses this board as a three-way tie — while
/// the output column separates it instantly: `chatml` answered, `llama2` emitted
/// `Nativescriptscript`.
///
/// `chatml` is deliberately not row 0, which is what makes an implementation
/// that returns `rows[0]` — the first row, or the first row after a redundant
/// EOS filter — fail here instead of passing by luck. Note that `zephyr`, the
/// row such an implementation returns, also fired and also answered correctly,
/// so the wrong answer is entirely plausible on this board and only identity
/// catches it.
#[test]
fn probe_accept_returns_the_named_row_when_it_is_not_first() {
    use lightbulb::api::chat_template::{ProbeRow, select_named_candidate};

    let rows = vec![
        ProbeRow {
            candidate: "zephyr",
            source: "SRC_Z",
            stopped_on_eos: true,
            tokens_generated: 8,
            first_line: "The capital of Japan is Tokyo.<|im_end|>".to_string(),
        },
        ProbeRow {
            candidate: "chatml",
            source: "SRC_C",
            stopped_on_eos: true,
            tokens_generated: 4,
            first_line: "Tokyo.<|im_end|>".to_string(),
        },
        ProbeRow {
            candidate: "llama2",
            source: "SRC_L",
            stopped_on_eos: true,
            tokens_generated: 27,
            first_line: "Nativescriptscript".to_string(),
        },
    ];
    let picked = select_named_candidate(&rows, "chatml")
        .expect("chatml is on the board, so --accept chatml must succeed");

    assert!(
        std::ptr::eq(picked, &rows[1]),
        "--accept chatml returned the {:?} row",
        picked.candidate
    );
    assert_eq!(picked.candidate, "chatml");
    assert_eq!(picked.source, "SRC_C");
    assert_eq!(picked.tokens_generated, 4);

    // The auto rule is untouched by any of this, and still refuses exactly this
    // board. Asserted here so the two paths cannot silently converge: if
    // `--accept` ever became a no-op that fell through to the EOS rule, the
    // assertion above would still hold on some board but this one would not.
    assert!(
        lightbulb::api::chat_template::select_probe_winner(&rows).is_err(),
        "the auto rule stopped refusing the three-way tie it was measured on"
    );
}

// ─── The probe must not overwrite a better answer than it can produce ───────
//
// Measured 2026-08-13 on SmolLM2-360M-Instruct, which ships its own
// `tokenizer_config.json` and so already resolved at
// `Resolution::TokenizerConfig`:
//
//     lightbulb-probe <dir> --accept chatml --yes
//     → wrote lightbulb-chat-template.json … will now resolve at Resolution::Probe
//     → exit 0
//
// The probe replaced the model author's template with `registry::CHATML`, a
// strictly worse approximation — SmolLM2's own injects a default system message
// when the caller supplies none; CHATML does not — and reported success. Since
// `resolve` reads the sidecar first, every later start would have prompted that
// model differently, and worse, than before the probe ran. Nothing warned.
//
// `probe_override_check` is the guard. Six `Resolution` variants times two
// `force` states is twelve cases, and each is asserted below, because a wrong
// implementation of a table like this characteristically gets one class right
// and another wrong: the mutation matrix is in the commit message, and every
// mutant there is killed by a test no other mutant kills.

/// The checkpoint's own `chat_template` is refused, and the refusal says which
/// tier it is protecting.
///
/// **The message is asserted, not `is_err()`.** The operator is being told they
/// pointed a guessing tool at a checkpoint that already answers the question
/// authoritatively, and the tier name is the whole content of that: a refusal
/// that named no tier, or named the wrong one, would satisfy any `matches!`
/// check while telling them nothing they can act on. The way out has to be named
/// too, or `--force` is a flag only the source code knows about.
#[test]
fn probe_override_refuses_the_checkpoints_own_template() {
    use lightbulb::api::chat_template::{ProbeOverride, probe_override_check};

    let why = match probe_override_check(Resolution::TokenizerConfig, false) {
        ProbeOverride::Refuse(why) => why,
        other => {
            panic!("the probe was allowed to overwrite the checkpoint's own template: {other:?}")
        }
    };
    assert!(
        why.contains("Resolution::TokenizerConfig"),
        "the refusal did not name the tier it is protecting: {why}"
    );
    // Not another tier's message, copied. Each refusal names its own.
    assert!(
        !why.contains("Resolution::Sidecar"),
        "the tier-1 refusal is carrying the sidecar refusal's text: {why}"
    );
    assert!(
        why.contains("--force"),
        "the refusal did not tell the operator how to override it: {why}"
    );
    // The same promise every other probe refusal ends on, and the one an
    // operator checks before going looking for a file: `--yes` was very likely
    // on the command line, so silence here reads as "it wrote something".
    assert!(
        why.ends_with("Nothing was written."),
        "the refusal did not end by saying nothing was written: {why}"
    );
}

/// An existing sidecar is refused. **A separate test from the one above on
/// purpose.**
///
/// The natural half-fix is to guard `TokenizerConfig` alone — it is the tier the
/// live defect was measured on — and leave a sidecar to be silently replaced.
/// That implementation passes every other test in this file and fails only here.
/// It matters because a sidecar is the one answer on disk that a human wrote or
/// reviewed: its `evidence` is the record of a decision, and an overwrite leaves
/// nothing to compare the new answer against.
#[test]
fn probe_override_refuses_an_existing_sidecar() {
    use lightbulb::api::chat_template::{ProbeOverride, probe_override_check};

    let why = match probe_override_check(Resolution::Sidecar, false) {
        ProbeOverride::Refuse(why) => why,
        other => panic!("an existing sidecar was silently overwritten: {other:?}"),
    };
    assert!(
        why.contains("Resolution::Sidecar"),
        "the refusal did not name the tier it is protecting: {why}"
    );
    assert!(
        !why.contains("Resolution::TokenizerConfig"),
        "the sidecar refusal is carrying the tier-1 refusal's text: {why}"
    );
    assert!(
        why.contains("--force"),
        "the refusal did not tell the operator how to override it: {why}"
    );
    assert!(
        why.ends_with("Nothing was written."),
        "the refusal did not end by saying nothing was written: {why}"
    );
}

/// A sidecar an EARLIER PROBE RUN left is refused — and it arrives as
/// `Resolution::Probe`, not `Resolution::Sidecar`.
///
/// This case is missing from any table that enumerates "tiers a checkpoint can
/// resolve at" from the resolution order alone, and it is the most likely one to
/// occur: it is what the second run of the probe against the same checkpoint
/// sees. `resolve` returns the sidecar's OWN `resolved_by` rather than
/// flattening it to `Sidecar`, and no other tier ever produces `Probe`, so
/// `Probe` here means "a sidecar this tool wrote" and carries the same
/// don't-silently-discard-it claim as any other sidecar — plus the `evidence`
/// recording the board it was chosen from, which is the only record of a
/// measurement that costs minutes to reproduce.
#[test]
fn probe_override_refuses_a_sidecar_an_earlier_probe_run_left() {
    use lightbulb::api::chat_template::{ProbeOverride, probe_override_check};

    let why = match probe_override_check(Resolution::Probe, false) {
        ProbeOverride::Refuse(why) => why,
        other => panic!("an earlier probe run's sidecar was silently overwritten: {other:?}"),
    };
    assert!(
        why.contains("Resolution::Probe"),
        "the refusal did not name the tier it is protecting: {why}"
    );
    assert!(
        why.contains("--force"),
        "the refusal did not tell the operator how to override it: {why}"
    );
    assert!(
        why.ends_with("Nothing was written."),
        "the refusal did not end by saying nothing was written: {why}"
    );
}

/// Nothing resolves: proceed, silently, in both `force` states.
///
/// This is the case the probe exists for, and it is asserted as `Proceed`
/// exactly — not "not a refusal" — because a guard that warned here would put a
/// warning on every correct use of the tool, which is how operators learn to
/// read past the warnings that mean something.
///
/// `force` is checked here as well, so that "`--force` is narrowly about
/// overriding a better source" is asserted rather than assumed: on a checkpoint
/// with nothing to override it changes nothing at all.
#[test]
fn probe_override_proceeds_silently_when_nothing_resolves_yet() {
    use lightbulb::api::chat_template::{ProbeOverride, probe_override_check};

    assert_eq!(
        probe_override_check(Resolution::None, false),
        ProbeOverride::Proceed,
        "the probe refused or warned about the very case it exists for"
    );
    assert_eq!(
        probe_override_check(Resolution::None, true),
        ProbeOverride::Proceed,
        "--force changed the verdict on a checkpoint that resolves at no tier at all"
    );
}

/// The two guessing tiers warn and proceed — they are neither refused nor
/// silent.
///
/// Both halves are the claim. Refusing here would break the probe's main use
/// (`Resolution::Registry` is what most un-probed checkpoints resolve at, so a
/// refusal would make the tool refuse nearly everything), and proceeding
/// silently would hide that one inference is being replaced by another.
///
/// The message must NOT carry the refusals' `Nothing was written.` promise:
/// something IS about to be written, and a copied refusal message would tell the
/// operator the exact opposite of what happens next.
#[test]
fn probe_override_warns_and_proceeds_when_it_replaces_a_guess() {
    use lightbulb::api::chat_template::{ProbeOverride, probe_override_check};

    for (tier, name, other) in [
        (
            Resolution::VocabSignature,
            "Resolution::VocabSignature",
            "Resolution::Registry",
        ),
        (
            Resolution::Registry,
            "Resolution::Registry",
            "Resolution::VocabSignature",
        ),
    ] {
        let why = match probe_override_check(tier, false) {
            ProbeOverride::Warn(why) => why,
            other => panic!("{name} should warn and proceed, got {other:?}"),
        };
        assert!(
            why.contains(name),
            "the warning did not name the guess being replaced: {why}"
        );
        assert!(
            !why.contains(other),
            "the {name} warning is carrying {other}'s text: {why}"
        );
        assert!(
            !why.contains("Nothing was written."),
            "the warning promises nothing was written, on a path that then writes: {why}"
        );
    }
}

/// `--force` is a **no-op** on the two tiers that only warn.
///
/// There is nothing to force past a guess, so the verdict must be identical in
/// both states — asserted as equality of the whole value, message included,
/// which is what makes it fail for an implementation that reaches for `force`
/// here and silences the warning. That implementation is plausible (`if force {
/// Proceed } else { … }` written once, at the top) and would remove the
/// operator's only notice that they replaced an existing answer, on the tiers
/// where the probe does most of its work.
#[test]
fn force_is_a_no_op_on_the_tiers_that_only_warn() {
    use lightbulb::api::chat_template::{ProbeOverride, probe_override_check};

    for tier in [Resolution::VocabSignature, Resolution::Registry] {
        assert_eq!(
            probe_override_check(tier, true),
            probe_override_check(tier, false),
            "--force changed the verdict on {tier:?}, where there is nothing to force"
        );
        // Restated positively, so the equality above cannot be satisfied by both
        // sides collapsing to `Proceed`.
        assert!(
            matches!(probe_override_check(tier, true), ProbeOverride::Warn(_)),
            "{tier:?} stopped warning under --force"
        );
    }
}

/// `--force` turns each refusal into a warning that still names the tier it
/// overrode.
///
/// Not `Proceed`: the point of the flag is that the operator meant to do this,
/// not that it did not happen. The record of what was overridden is exactly what
/// somebody reading the log a month later needs, and it is the difference
/// between this and the silent downgrade the whole guard exists to prevent.
///
/// Deliberately asserts only the `force == true` half; the `false` half has a
/// test each above, so an implementation that ignores `force` and refuses
/// regardless fails HERE and nowhere else.
#[test]
fn force_downgrades_each_refusal_to_a_warning_that_still_names_the_tier() {
    use lightbulb::api::chat_template::{ProbeOverride, probe_override_check};

    for (tier, name) in [
        (Resolution::TokenizerConfig, "Resolution::TokenizerConfig"),
        (Resolution::Sidecar, "Resolution::Sidecar"),
        (Resolution::Probe, "Resolution::Probe"),
    ] {
        let why = match probe_override_check(tier, true) {
            ProbeOverride::Warn(why) => why,
            other => panic!("--force did not get past {name}: {other:?}"),
        };
        assert!(
            why.contains(name),
            "the --force warning did not name the tier it overrode: {why}"
        );
        assert!(
            why.contains("--force"),
            "the warning did not say the override was deliberate: {why}"
        );
        assert!(
            !why.contains("Nothing was written."),
            "the --force warning promises nothing was written, on the path that writes: {why}"
        );
    }
}

// ─── The guard, as the binary actually runs it ──────────────────────────────
//
// The three tests below run the real `lightbulb-probe` and take milliseconds,
// because every one of them is stopped by a refusal that reads files and
// nothing else — before `ModelRunner::start`, so no checkpoint is needed and no
// generation happens. They assert the wiring the pure-function tests above
// cannot: that the guard is CALLED, that it is called before anything is
// written or loaded, and that `--force` reaches it and reaches nothing else.

/// A fixture that ships its own `chat_template`, so it resolves at
/// `Resolution::TokenizerConfig` — the SmolLM2 situation, without SmolLM2.
///
/// `eos_token` is present because the probe refuses an EOS-less checkpoint one
/// step earlier; without it these tests would pass for the wrong reason.
fn tmp_checkpoint_with_own_template(name: &str) -> std::path::PathBuf {
    let d = tmp_model_dir(name);
    let mut f = std::fs::File::create(d.join("tokenizer_config.json")).unwrap();
    f.write_all(
        br#"{"bos_token":"<|im_start|>","eos_token":"<|im_end|>",
             "chat_template":"{% for m in messages %}{{ m['role'] }}{% endfor %}"}"#,
    )
    .unwrap();
    d
}

/// The measured defect, as a test: the exact command line that silently
/// downgraded SmolLM2 now refuses.
///
/// Three separate claims, and the last two are what make this more than a
/// duplicate of the pure-function test above:
///
/// * it refuses, naming the tier — so the guard is actually wired into the
///   binary, which no test in this file could otherwise show, `src/bin/*.rs`
///   being a separate crate;
/// * it wrote no sidecar — the property the operator cares about;
/// * it never loaded the model — the guard runs before `ModelRunner::start`, so
///   an operator who pointed the probe at the wrong checkpoint finds out in
///   milliseconds instead of after three generations. That is also what keeps
///   this test cheap enough to run by default, unlike the `--ignored` one below.
///
/// `--yes` is on the command line deliberately: it must not bypass this, being
/// the flag for "do not ask me", not "override the checkpoint's own template".
#[test]
fn probe_binary_refuses_a_checkpoint_that_ships_its_own_template() {
    let dir = tmp_checkpoint_with_own_template("ships-its-own-template");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_lightbulb-probe"))
        .arg(&dir)
        .args(["--accept", "chatml", "--yes"])
        .stdin(std::process::Stdio::null())
        .output()
        .expect("running the probe binary");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let both = format!("--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");

    assert!(
        !out.status.success(),
        "the probe exited 0 on a checkpoint that already resolves at \
         Resolution::TokenizerConfig — this is the SmolLM2 downgrade, measured.\n{both}"
    );
    assert!(
        stderr.contains("Resolution::TokenizerConfig"),
        "the probe refused for some reason OTHER than the tier it was overriding.\n{both}"
    );
    assert!(
        stderr.contains("Nothing was written."),
        "the refusal did not tell the operator nothing was written.\n{both}"
    );
    assert!(
        !dir.join(lightbulb::api::chat_template::SIDECAR_NAME)
            .exists(),
        "the probe wrote a sidecar despite refusing; {} would then resolve at \
         Resolution::Probe, ahead of its own template.\n{both}",
        dir.display()
    );
    // Nothing was loaded and nothing was generated: the guard runs before
    // `ModelRunner::start`. Both strings are the runner's own, printed on the
    // load path before any weights are read.
    assert!(
        !stdout.contains("Loading") && !stderr.contains("model load failed"),
        "the probe loaded the model before refusing; on a real checkpoint that is \
         minutes spent to reach a verdict already available from the files.\n{both}"
    );
}

/// `--force` gets past the guard, says so, and gets past nothing else.
///
/// The fixture has a `chat_template` but no weights, so a run that reaches the
/// loader fails there — and that failure is the evidence that the guard let it
/// through. Asserted three ways round: the deliberate-override warning naming
/// the tier is present, the refusal's `Nothing was written.` is absent, and the
/// run got as far as the model loader.
///
/// Without this, a binary that parsed `--force` and never passed it to the guard
/// would pass every other test here.
#[test]
fn probe_binary_force_gets_past_the_guard_and_says_so() {
    let dir = tmp_checkpoint_with_own_template("force-past-the-guard");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_lightbulb-probe"))
        .arg(&dir)
        .args(["--accept", "chatml", "--yes", "--force"])
        .stdin(std::process::Stdio::null())
        .output()
        .expect("running the probe binary");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let both = format!("--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");

    // The warning is on STDOUT: `tracing_subscriber::fmt::layer()` writes there
    // by default, while anyhow's final `Error:` — every refusal in this file —
    // goes to stderr. That is why each assertion below names one stream.
    assert!(
        stdout.contains("--force") && stdout.contains("Resolution::TokenizerConfig"),
        "--force did not record WHAT it overrode; a log a month from now would not \
         show that the checkpoint's own template was replaced.\n{both}"
    );
    assert!(
        !both.contains("Nothing was written."),
        "--force did not get past the guard.\n{both}"
    );
    // It reached the loader, which is as far as a fixture with no weights can
    // go. `model load failed` is the runner's verdict on both backends.
    assert!(
        stderr.contains("model load failed"),
        "the run stopped before the model loader, so --force did not actually \
         resume the probe.\n{both}"
    );
    assert!(
        !dir.join(lightbulb::api::chat_template::SIDECAR_NAME)
            .exists(),
        "a sidecar was written for a checkpoint that never generated anything.\n{both}"
    );
}

/// `--force` does not bypass the refusals it has nothing to do with.
///
/// It means one thing — "yes, I mean to override a better source" — and a flag
/// that quietly grew into "write the sidecar whatever happens" is the flag the
/// probe's `--yes` doc comment already refuses to be. The two refusals checked
/// here are the two that can be reached without a checkpoint, and they are also
/// the two most likely to be hit by the same hurried operator who reached for
/// `--force` in the first place: a mistyped path, and a checkpoint that declares
/// no EOS.
///
/// Asserted on the message, not the exit status: an implementation where
/// `--force` skipped the EOS check would still exit non-zero — later, from the
/// model loader — and the difference between those two failures is the whole
/// property.
#[test]
fn force_does_not_bypass_the_probes_other_refusals() {
    // A directory with no `tokenizer_config.json` and no `tokenizer.json`, so
    // no EOS text can be resolved for it at all.
    let no_eos = tmp_model_dir("force-no-eos");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_lightbulb-probe"))
        .arg(&no_eos)
        .args(["--accept", "chatml", "--yes", "--force"])
        .stdin(std::process::Stdio::null())
        .output()
        .expect("running the probe binary");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "--force let the probe measure a checkpoint with no EOS token.\n{stderr}"
    );
    assert!(
        stderr.contains("declares no EOS token"),
        "--force bypassed the empty-EOS refusal; the run failed for some other \
         reason instead.\n{stderr}"
    );
    assert!(
        stderr.contains("Nothing was written."),
        "the empty-EOS refusal stopped promising nothing was written.\n{stderr}"
    );
    assert!(
        !no_eos
            .join(lightbulb::api::chat_template::SIDECAR_NAME)
            .exists(),
        "a sidecar was written for an unmeasurable checkpoint"
    );

    // And the one before it: a path that does not exist at all.
    let missing = no_eos.join("no-such-checkpoint");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_lightbulb-probe"))
        .arg(&missing)
        .args(["--yes", "--force"])
        .stdin(std::process::Stdio::null())
        .output()
        .expect("running the probe binary");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        !out.status.success(),
        "--force let the probe run against a path that does not exist.\n{stderr}"
    );
    assert!(
        stderr.contains("does not exist"),
        "--force bypassed the nonexistent-path refusal.\n{stderr}"
    );
}

/// Locate the checkpoint, preferring `TINYLLAMA_DIR`. Same shape as
/// `tests/chat_template_e2e.rs`'s and `tests/api_result_metadata.rs`'s.
fn tinyllama_dir() -> Option<std::path::PathBuf> {
    let p = match std::env::var_os("TINYLLAMA_DIR") {
        Some(v) => std::path::PathBuf::from(v),
        None => std::path::PathBuf::from(
            "C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        ),
    };
    p.join("model.safetensors").is_file().then_some(p)
}

/// **A characterization test: it records that the probe's discriminator does
/// NOT discriminate on TinyLlama.** Read the next paragraph before treating a
/// green run here as good news.
///
/// This test passing does **not** show the probe works. It shows the opposite,
/// pinned: run against TinyLlama-1.1B-Chat, all THREE registry candidates stop
/// on EOS (measured 2026-08-13 — zephyr 14 tokens, chatml 10, llama2 3), so the
/// probe correctly refuses to guess and exits non-zero having written nothing.
/// The refusal is the only part demonstrated to work; the measurement
/// underneath it cannot tell these three templates apart on this checkpoint,
/// even on the multi-turn conversation adopted specifically to try to.
///
/// Its value is that the limitation becomes an asserted fact rather than
/// folklore. If someone later strengthens the discriminator — a sharper prompt,
/// a stricter stop condition, scoring beyond a bare EOS flag — and it begins
/// selecting a winner on TinyLlama, this test goes RED. That is the intended
/// outcome, and whoever made it happen must then come here and rewrite this
/// test deliberately, recording the new board, instead of the change landing
/// unremarked against a suite that never looked.
///
/// **stdin is closed** (`Stdio::null()`), so `confirm`'s `read_line` returns
/// `Ok(0)` and can never answer yes. Today that path is unreachable — the tie
/// aborts first — and it stays closed for exactly the day it becomes reachable:
/// a strengthened probe must not be able to write a sidecar from inside a test
/// run. Hence the final assertion, which guards a real hazard rather than
/// tidiness: `TINYLLAMA_DIR` is the SHARED checkpoint every other checkpoint
/// test measures against, and a sidecar left there resolves at
/// `Resolution::Probe` — ahead of every other tier — silently changing what
/// those later runs measure.
///
/// Not run by default: it loads the checkpoint and generates three times, which
/// takes several minutes.
///
/// ```text
/// cargo test --test chat_template_render probe_binary -- --ignored --nocapture
/// ```
#[test]
#[ignore = "runs the probe binary against a real checkpoint; several minutes"]
fn probe_binary_refuses_tinyllama_because_all_three_candidates_fire() {
    let dir = tinyllama_dir().expect(
        "no TinyLlama checkpoint. Set TINYLLAMA_DIR to a directory containing \
         model.safetensors and tokenizer.json. This test asserts real probe behaviour \
         against a real model, so it fails rather than skipping.",
    );
    let sidecar = dir.join(lightbulb::api::chat_template::SIDECAR_NAME);
    // A sidecar already present would make the closing assertion meaningless,
    // and would already be changing what every other checkpoint test measures.
    assert!(
        !sidecar.exists(),
        "{} exists before this test ran; the shared checkpoint is already carrying a \
         Resolution::Probe answer. Remove it, then re-run.",
        sidecar.display()
    );

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_lightbulb-probe"))
        .arg(&dir)
        .stdin(std::process::Stdio::null())
        .output()
        .expect("running the probe binary");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();

    assert!(
        !out.status.success(),
        "the probe exited 0 on TinyLlama. If the discriminator now separates the \
         candidates, that is an improvement and this characterization test is stale — \
         update it with the new board.\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("candidates stopped on EOS"),
        "the probe failed for some reason OTHER than the tie this test characterizes.\
         \n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("3 candidates stopped on EOS"),
        "the tie is no longer three-way. Measured 2026-08-13: zephyr 14 tokens, chatml \
         10, llama2 3 — all three firing. Record what it is now.\
         \n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("Nothing was written."),
        "the refusal did not tell the operator nothing was written.\
         \n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    // The shared checkpoint must come out of this exactly as it went in.
    assert!(
        !sidecar.exists(),
        "the probe wrote {} despite refusing; every later run against this shared \
         checkpoint would then resolve at Resolution::Probe.",
        sidecar.display()
    );
}

// ─── Synthetic GGUF fixtures ────────────────────────────────────────────────
//
// A valid GGUF v3 header with ZERO tensors is a few hundred bytes, so every row
// of the spec's failure table gets its own file. The real 637,699,456-byte Q4_0
// has one metadata set and cannot exercise a missing key, a blank template, or
// an out-of-range token id — the cases that matter are the malformed ones.

/// One metadata value to write into a synthetic header.
enum Kv {
    Str(&'static str),
    U32(u32),
    StrArray(Vec<&'static str>),
}

/// `tensor_count` is a parameter, not a constant, because Task 2 needs a
/// one-tensor file to reach the unsupported-dtype path. Pass `0` for every
/// metadata-only fixture; the tensor-info records themselves are appended by
/// the caller (see `gguf_fixture_with_tensor`).
fn gguf_bytes(kvs: &[(&str, Kv)], tensor_count: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&tensor_count.to_le_bytes());
    b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    // Not `let mut` — this closure captures nothing, so it is `Fn` and a
    // `mut` binding would only earn an `unused_mut` warning.
    let put_str = |b: &mut Vec<u8>, s: &str| {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    };
    for (k, v) in kvs {
        put_str(&mut b, k);
        match v {
            Kv::Str(s) => {
                b.extend_from_slice(&8u32.to_le_bytes());
                put_str(&mut b, s);
            }
            Kv::U32(n) => {
                b.extend_from_slice(&4u32.to_le_bytes());
                b.extend_from_slice(&n.to_le_bytes());
            }
            Kv::StrArray(items) => {
                b.extend_from_slice(&9u32.to_le_bytes());
                b.extend_from_slice(&8u32.to_le_bytes()); // element type: string
                b.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for it in items {
                    put_str(&mut b, it);
                }
            }
        }
    }
    b
}

/// Write a synthetic `.gguf` into a fresh temp dir and return its path.
fn gguf_fixture(name: &str, kvs: &[(&str, Kv)]) -> std::path::PathBuf {
    let d = tmp_model_dir(name);
    let p = d.join("model.gguf");
    std::fs::write(&p, gguf_bytes(kvs, 0)).unwrap();
    p
}

/// Like `gguf_fixture`, plus exactly one tensor-info record.
///
/// The tensor's DATA is never written — `Content::read` parses the info table
/// and stops; it does not read the data region. That is what makes a
/// one-tensor fixture a few hundred bytes rather than a real weight.
///
/// Layout after the KV block, per `fuel-formats/src/gguf.rs:415-434`:
/// u64 name-len + name bytes, u32 n_dims, n_dims × u64 dims, u32 dtype code,
/// u64 offset.
fn gguf_fixture_with_tensor(
    tag: &str,
    kvs: &[(&str, Kv)],
    tensor: (&str, u32),
) -> std::path::PathBuf {
    let (name, dtype) = tensor;
    let mut body = gguf_bytes(kvs, 1 /* tensor_count */);
    body.extend_from_slice(&(name.len() as u64).to_le_bytes());
    body.extend_from_slice(name.as_bytes());
    body.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    body.extend_from_slice(&1u64.to_le_bytes()); // dims[0]
    body.extend_from_slice(&dtype.to_le_bytes());
    body.extend_from_slice(&0u64.to_le_bytes()); // offset
    let dir = tmp_model_dir(tag);
    let p = dir.join("model.gguf");
    std::fs::write(&p, body).expect("writing synthetic gguf");
    p
}

/// Run `f` with a tracing subscriber that writes into a buffer, and return
/// what was logged.
///
/// `with_default` is thread-local, so this is safe under the default parallel
/// test harness — each test sees only its own subscriber.
fn capture_logs(f: impl FnOnce()) -> String {
    use std::sync::{Arc, Mutex};
    #[derive(Clone)]
    struct Buf(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for Buf {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Buf {
        type Writer = Buf;
        fn make_writer(&'a self) -> Buf {
            self.clone()
        }
    }

    let buf = Buf(Arc::new(Mutex::new(Vec::new())));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(buf.clone())
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .finish();
    tracing::subscriber::with_default(subscriber, f);
    let bytes = buf.0.lock().unwrap().clone();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// A GGUF's own metadata is the model author's declaration and must be used.
///
/// Measured before this change: the real Q4_0 TinyLlama served
/// `"| ass istant | ass istant |"` because resolution never looked inside the
/// file, fell to a family guess, and rendered with an empty `eos_token`.
#[test]
fn a_gguf_declares_its_own_chat_template_and_tokens() {
    let p = gguf_fixture(
        "gguf-declares",
        &[
            ("tokenizer.chat_template", Kv::Str("FROM_GGUF_METADATA")),
            (
                "tokenizer.ggml.tokens",
                Kv::StrArray(vec!["<unk>", "<s>", "</s>"]),
            ),
            ("tokenizer.ggml.bos_token_id", Kv::U32(1)),
            ("tokenizer.ggml.eos_token_id", Kv::U32(2)),
        ],
    );

    let t = lightbulb::api::chat_template::resolve(&p);
    assert_eq!(t.source, "FROM_GGUF_METADATA");
    assert_eq!(t.resolved_by, Resolution::GgufMetadata);

    // Exact strings, not `!is_empty()`: empty is what the defect produced, so a
    // weaker check would be satisfied by any accident.
    let tk = lightbulb::api::chat_template::special_tokens(&p);
    assert_eq!(tk.bos, "<s>");
    assert_eq!(tk.eos, "</s>");
}

/// A GGUF that declares no template falls through — it is a normal checkpoint,
/// not an error.
///
/// Asserts `Resolution::None`, NOT `Registry`. `gguf_fixture` builds under
/// `<CARGO_TARGET_TMPDIR>/lb-chat-tmpl-gguf-no-template-<pid>-<n>/`, and
/// `registry::from_family` (`src/api/chat_template/registry.rs:73-101`) matches
/// lowercased path *components* against a fixed set — `tinyllama`+`chat`,
/// `qwen`/`chatml`, `llama-2`/`llama2`/`mistral`. No component of that temp path
/// satisfies any rule (`lb-chat-tmpl-…` contains `chat` but not `tinyllama`),
/// and tier 2 misses too because no `tokenizer.json` is written. So resolution
/// runs off the end and returns `None`.
///
/// The earlier version of this test was named `…falls_through_to_the_registry`
/// and asserted `assert_ne!(resolved_by, GgufMetadata)` — which passes for six
/// of the seven variants and is therefore invariant to the thing it claims to
/// check (Global Constraint 5). An implementer obeying that constraint would
/// have written `assert_eq!(…, Registry)` and been sent hunting a phantom.
#[test]
fn a_gguf_without_a_template_falls_through_to_no_template() {
    let p = gguf_fixture(
        "gguf-no-template",
        &[(
            "tokenizer.ggml.tokens",
            Kv::StrArray(vec!["<unk>", "<s>", "</s>"]),
        )],
    );
    let t = lightbulb::api::chat_template::resolve(&p);
    assert_eq!(
        t.resolved_by,
        Resolution::None,
        "a GGUF declaring no template, in a directory no family rule matches, \
         must run off the end of resolution"
    );
    // Tokens still come from the file even when the template does not.
    let tk = lightbulb::api::chat_template::special_tokens(&p);
    assert_eq!(tk.bos, "");
    assert_eq!(tk.eos, "");
}

/// Spec §7 row 2: a VALID GGUF with READABLE metadata that Fuel refuses to open
/// over a tensor quantization it does not implement.
///
/// This is the row `1740582` added and the only one with no coverage before
/// this test. It is reachable from the synthetic builder: give the file one
/// tensor whose dtype code is `20` (`IQ4_NL`), which
/// `GgmlDType::from_u32` (`fuel-ir/src/quantized.rs:35`, accepts only
/// `{0,1,2,3,6..=15,30}`) rejects at `fuel-formats/src/gguf.rs:432-433` —
/// AFTER the metadata block has already been parsed in full.
///
/// Asserts the fall-through AND that the operator is told the real cause. A
/// bare fall-through here would reproduce the §1 defect on every IQ-quantized
/// model: a family guess with no end-of-turn marker, and a log line pointing
/// at corruption that does not exist.
#[test]
fn a_gguf_with_an_unsupported_tensor_dtype_falls_through_and_says_why() {
    // `gguf_tensor_bytes` extends the builder with a tensor-info record:
    // name, u32 n_dims = 1, u64 dim = 1, u32 dtype code, u64 offset = 0.
    let p = gguf_fixture_with_tensor(
        "gguf-iq4nl",
        &[("tokenizer.chat_template", Kv::Str("{{ eos_token }}"))],
        ("blk.0.attn_q.weight", 20u32),
    );
    let logs = capture_logs(|| {
        let t = lightbulb::api::chat_template::resolve(&p);
        assert_eq!(
            t.resolved_by,
            Resolution::None,
            "Fuel cannot open this file, so its template must not be reported as \
             read; and this fixture's directory matches no family rule, so \
             resolution runs off the end"
        );
    });
    // The whole point of spec §7 row 2. A bare fall-through here is satisfied by
    // an implementation that logs nothing, or that says "not a valid GGUF" —
    // which is FALSE about this file and sends an operator hunting corruption
    // that does not exist.
    assert!(
        logs.contains("quantization") && logs.contains("ggml type table"),
        "the warn must name the real cause — a tensor quantization outside Fuel's \
         type table — not merely fall through; got: {logs}"
    );
    assert!(
        !logs.contains("not a valid GGUF"),
        "this file IS a valid GGUF with readable metadata; saying otherwise is a \
         false statement about the operator's file: {logs}"
    );
}

/// Spec §7 row 5: `tokenizer.chat_template` present but not a string.
///
/// The state MLMF's architect named as "absent is not empty". Before this test
/// it was indistinguishable from key-absent in BOTH the return type and the
/// log: `md.get(..).and_then(|v| v.to_string().ok())` yields `None` either way,
/// and only the one `debug!` in `resolve` fires. A file declaring its template
/// as an array or an integer would emit "no usable tokenizer.chat_template" and
/// drop to a family guess with no end-of-turn marker — the epic's own failure
/// mode, inside the epic's fix.
///
/// Asserts the WARN, not just the fall-through. The fall-through alone is
/// satisfied by the buggy implementation.
#[test]
fn a_non_string_gguf_template_warns_rather_than_reading_as_absent() {
    let p = gguf_fixture(
        "gguf-wrong-type",
        &[("tokenizer.chat_template", Kv::U32(7))],
    );
    let logs = capture_logs(|| {
        let t = lightbulb::api::chat_template::resolve(&p);
        assert_ne!(t.resolved_by, Resolution::GgufMetadata);
    });
    assert!(
        logs.contains("tokenizer.chat_template") && logs.contains("not a string"),
        "a present-but-undecodable template must warn naming the key and the \
         reason, not fall through as though the key were absent; got: {logs}"
    );
}

/// A blank template is UNDECLARED, not declared-empty. An empty source renders
/// to an empty prompt — a request to continue nothing.
#[test]
fn a_blank_gguf_template_is_unusable_and_falls_through() {
    // These literals are already `&'static str`, which is what `Kv::Str` takes.
    for blank in ["", "   ", "\n\t "] {
        let p = gguf_fixture("gguf-blank", &[("tokenizer.chat_template", Kv::Str(blank))]);
        let t = lightbulb::api::chat_template::resolve(&p);
        assert_ne!(
            t.resolved_by,
            Resolution::GgufMetadata,
            "blank template {blank:?} was accepted as a declaration"
        );
    }
}

/// A token id past the end of the vocab leaves that token empty rather than
/// panicking or silently picking a neighbour.
#[test]
fn an_out_of_range_token_id_leaves_the_token_empty() {
    let p = gguf_fixture(
        "gguf-oob-id",
        &[
            ("tokenizer.ggml.tokens", Kv::StrArray(vec!["<unk>", "<s>"])),
            ("tokenizer.ggml.bos_token_id", Kv::U32(1)),
            ("tokenizer.ggml.eos_token_id", Kv::U32(99)),
        ],
    );
    let tk = lightbulb::api::chat_template::special_tokens(&p);
    assert_eq!(tk.bos, "<s>", "the in-range token should still resolve");
    assert_eq!(
        tk.eos, "",
        "the out-of-range id must not resolve to anything"
    );
}

/// Ids with no token list at all are unresolvable — must not panic.
#[test]
fn token_ids_without_a_token_list_resolve_to_nothing() {
    let p = gguf_fixture(
        "gguf-no-tokens",
        &[
            ("tokenizer.ggml.bos_token_id", Kv::U32(1)),
            ("tokenizer.ggml.eos_token_id", Kv::U32(2)),
        ],
    );
    let tk = lightbulb::api::chat_template::special_tokens(&p);
    assert_eq!(tk.bos, "");
    assert_eq!(tk.eos, "");
}

/// A file with a `.gguf` name that is not a GGUF falls through quietly rather
/// than aborting resolution.
#[test]
fn a_corrupt_gguf_falls_through_instead_of_failing() {
    let d = tmp_model_dir("gguf-corrupt");
    let p = d.join("model.gguf");
    std::fs::write(&p, b"this is not a GGUF file at all").unwrap();
    let t = lightbulb::api::chat_template::resolve(&p);
    assert_ne!(t.resolved_by, Resolution::GgufMetadata);
    let tk = lightbulb::api::chat_template::special_tokens(&p);
    assert_eq!(tk.bos, "");
}

/// A DIRECTORY checkpoint is unaffected — the GGUF branch must not fire, and
/// `tokenizer_config.json` must still win. Without this, an implementation that
/// ran the GGUF reader on every path would pass everything above.
#[test]
fn a_directory_checkpoint_still_resolves_from_tokenizer_config() {
    let d = tmp_model_dir("gguf-dir-unaffected");
    std::fs::write(
        d.join("tokenizer_config.json"),
        r#"{"bos_token":"<s>","eos_token":"</s>","chat_template":"FROM_TOKENIZER_CONFIG"}"#,
    )
    .unwrap();
    let t = lightbulb::api::chat_template::resolve(&d);
    assert_eq!(t.source, "FROM_TOKENIZER_CONFIG");
    assert_eq!(t.resolved_by, Resolution::TokenizerConfig);
}

/// In-file metadata outranks a companion JSON sitting beside the `.gguf`.
/// This is the precedence decision in spec §3 and the one a reviewer could
/// reasonably take the other way — so it is pinned, not left implicit.
#[test]
fn gguf_metadata_outranks_a_companion_tokenizer_config() {
    let p = gguf_fixture(
        "gguf-outranks",
        &[("tokenizer.chat_template", Kv::Str("FROM_GGUF_METADATA"))],
    );
    std::fs::write(
        p.parent().unwrap().join("tokenizer_config.json"),
        r#"{"chat_template":"FROM_TOKENIZER_CONFIG"}"#,
    )
    .unwrap();
    let t = lightbulb::api::chat_template::resolve(&p);
    assert_eq!(t.source, "FROM_GGUF_METADATA");
    assert_eq!(t.resolved_by, Resolution::GgufMetadata);
}

/// A directory checkpoint must never reach the GGUF reader at all.
///
/// Asserts the absence of the warn, not the resolution: dropping the
/// `is_file()`/extension guard still resolves correctly (the mmap open just
/// fails), so resolution cannot discriminate. What changes is that every
/// ordinary directory checkpoint starts logging a false and alarming claim
/// about tensor quantizations.
#[test]
fn a_directory_checkpoint_is_not_run_through_the_gguf_reader() {
    let d = tmp_checkpoint_with_own_template("dir-not-gguf");
    let logs = capture_logs(|| {
        let t = lightbulb::api::chat_template::resolve(&d);
        assert_eq!(t.resolved_by, Resolution::TokenizerConfig);
    });
    assert!(
        !logs.contains("could not open this GGUF") && !logs.contains("ggml type table"),
        "a directory checkpoint was run through the GGUF reader; every start would \
         now warn about tensor quantizations for a path that is not a file: {logs}"
    );
}
