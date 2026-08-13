# Chat-Template Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prompt chat models with their own chat template instead of a `role: content` join, resolve that template per model from the cheapest available source, and record how it was chosen.

**Architecture:** A new `src/api/chat_template.rs` resolves a template once at startup through four free tiers (sidecar → `tokenizer_config.json` → vocabulary signature → family registry), renders it with `minijinja`, and stores the result on `AppState`. All three existing prompt-construction sites render through it. A rolling EOS-rate counter warns when a heuristically-chosen template looks wrong. A CLI probe generates under candidate templates and writes a provenance-carrying sidecar on confirmation.

**Tech Stack:** Rust, `minijinja` (pure Rust, no C deps), `tokenizers`, `axum`, `clap`, `serde_json`.

**Spec:** `docs/superpowers/specs/2026-08-06-chat-template-resolution-design.md` — read its **CORRECTED 2026-08-08** block in §6 before Task 3. The spec body cites one prompt site; there are three.

---

## Global Constraints

- **Build with `-j 4`.** Full-parallelism cargo on this machine races into rustc ICEs and spurious "crate not found in rlib format" errors naming different crates each run. Not your code — do not investigate.
- **Feature-gated tests are invisible by default.** `tests/api_result_metadata.rs` and `tests/fuel_engine_http.rs` open with `#![cfg(feature = "fuel-engine")]`. Without the feature they compile to **zero tests** and print `ok. 0 passed` — identical in shape to a suite that ran. **Always read the `running N tests` line, never the exit code alone.**
- **Checkpoint tests need the real invocation:** `cargo test --release --features fuel-engine --test <name> -- --ignored --nocapture --test-threads=1`. `--test-threads=1` is load-bearing: each test loads its own ~2.2 GB copy of the checkpoint.
- **Two different env vars point at the model, and setting only one fails loudly but misleadingly.** `api_result_metadata.rs`, `chat_template_e2e.rs` and `fuel_engine_http.rs` read **`TINYLLAMA_DIR`**; `contracts_live.rs` reads **`LIGHTBULB_TEST_MODEL_DIR`** and otherwise searches for `../models/llama-3b`. Setting only `TINYLLAMA_DIR` makes all 5 `contracts_live` tests panic in 0.01 s with "no model directory found" — which reads like a regression in whatever you just changed, and is not. Export both:
  ```bash
  export TINYLLAMA_DIR="C:/Users/cires/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6"
  export LIGHTBULB_TEST_MODEL_DIR="$TINYLLAMA_DIR"
  ```
- **A failing test binary stops the ones after it.** `cargo test --test a --test b` will not run `b` if `a` fails, so a suite you believe you covered may simply never have executed. Re-read the `running N tests` lines and count the suites, not just the results.
- **GPU-touching suites go through the lock:** `pwsh -NoProfile -File C:/Projects/fuel/scripts/gpu-run.ps1 -Project lightbulb -- <cmd>`.
- **When piping cargo through `grep`, print `${PIPESTATUS[0]}`.** The pipeline's status masks cargo's; this has already produced one false green on this project.
- **`rustfmt --edition 2024 <file>` every file you touch**, before committing. **But never run it on a `mod.rs`** — rustfmt follows `pub mod` declarations and reformats every module the file declares. Running it on `src/engine/mod.rs` rewrote 8 unrelated files that were not already rustfmt-clean. If you must format a `mod.rs`, check `git status` afterwards and revert everything you did not touch.
- **Never assert a property that holds whether or not the code is correct.** `assert!(x > 0)` against a count that was already nonzero proves nothing. Prefer equality against an independently-derived value.
- **The test model is TinyLlama-1.1B-Chat** at `$TINYLLAMA_DIR`, defaulting to the HF cache path in `tests/api_result_metadata.rs`. Its snapshot contains **only** `config.json`, `model.safetensors`, `tokenizer.json` — no `tokenizer_config.json`, so **tier 1 does not fire for it** and its `<|user|>` markers are ordinary text, so **tier 2 does not either**. It resolves at tier 3. Any test asserting tier 1 or 2 must build a fixture, not use the checkpoint.
- **TinyLlama is BLIND to any `bos_token` defect, so never use it to test one.** It resolves to `registry::ZEPHYR`, whose source never mentions `bos_token`, so its rendered prompt is *accidentally* correct no matter what the surrounding code does with BOS. Task 3's review found exactly this: a templated prompt was being tokenized with `add_special_tokens = true`, doubling BOS for Llama-3/Llama-2/Mistral/Gemma — and every TinyLlama-based test in the repo stayed green, because ZEPHYR emits no BOS to double. The guard tests in `src/api/openai/chat.rs` build a synthetic Llama-3-shaped checkpoint (a template opening `{{ bos_token }}` plus a `TemplateProcessing` post_processor) for this reason. **The same shape applies beyond BOS:** before testing any behaviour against the checkpoint, check that ZEPHYR actually exercises it — the tier-3 template is a small subset of what real templates do.
- **Assert on the level the defect lives at.** The BOS doubling above was invisible to *every* rendered-text assertion in the repo, including a live HTTP gate asserting content, because the render is byte-identical either way — the defect is in how a correct string is encoded. A prompt has at least three observable levels (rendered text → token ids → generated output); an assertion one level above the defect is invariant to it.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/api/chat_template.rs` **new** | `ChatTemplate`, tier resolution, `minijinja` rendering, sidecar read/write, family registry |
| `src/api/chat_template/registry.rs` **new** | Family table only — the part that rots, isolated so it can be edited without touching resolution logic |
| `src/api/mod.rs` *modify* | Resolve at startup beside `ModelRunner::start`; add field to `AppState` |
| `src/api/openai/chat.rs` *modify* | Both join sites (`:186`, `:409`) render through the template |
| `src/contracts/executor.rs` *modify* | `infer` callback takes `Vec<RawMessage>`, not `String` |
| `src/contracts/validation.rs` *modify* | **Delete** `messages_to_prompt` |
| `src/engine/eos_monitor.rs` **new** | Rolling EOS-rate counter |
| `src/bin/lightbulb-probe.rs` **new** | Probe binary. **Not** a `lightbulb-cli` subcommand: that binary has zero `lightbulb::` references and is a pure HTTP client, so giving it the engine would put CUDA link requirements on a tool that only talks to a server |
| `tests/chat_template_render.rs` **new** | Rendering and tier-order tests — no checkpoint needed |
| `tests/chat_template_e2e.rs` **new** | Behavioural, feature-gated, needs the checkpoint |

---

## Task 1: Rendering core

**Files:**
- Create: `src/api/chat_template.rs`
- Modify: `Cargo.toml`
- Modify: `src/api/mod.rs` (add `pub mod chat_template;`)
- Test: `tests/chat_template_render.rs`

**Interfaces:**
- Produces: `ChatTemplate { source: String, resolved_by: Resolution }`, `ChatTemplate::render(&self, messages: &[RawMessage], bos: &str, eos: &str) -> anyhow::Result<String>`, `enum Resolution { Sidecar, TokenizerConfig, VocabSignature, Registry, Probe, None }`
- Consumes: nothing.

- [x] **Step 1: Add the dependency**

In `Cargo.toml` under `[dependencies]`:

```toml
# Chat-template rendering. HuggingFace `chat_template` values are Jinja.
# minijinja is pure Rust with no C dependencies, which keeps the CUDA build
# unaffected.
#
# The feature list mirrors what `transformers` gives its own environment
# (`utils/chat_template_utils.py`), because these templates are authored
# against THAT dialect. Dropping one does not degrade rendering — it makes real
# checkpoint templates fail to parse, which falls through to a registry guess
# and reports success at a lower tier. That is the defect this work removes,
# wearing a provenance field that says "guess".
#
#   macros        {% macro %}
#   loop_controls {% break %} / {% continue %}  (transformers loads
#                 jinja2.ext.loopcontrols for exactly this)
#   json          the `tojson` filter
#   serde         silences a deprecation warning; `context!` over serde types
#                 is load-bearing here, not incidental
#   debug         puts template source context into errors — spec §2 requires
#                 a failing template log "which template failed and why"
#
# `loader` stays out: templates arrive as strings, never paths.
minijinja = { version = "2", features = ["macros", "loop_controls", "json", "serde", "debug"] }
```

`strftime_now` is a *global*, not a feature — Llama-3.x templates call it. Register it alongside `raise_exception` in Step 4.

- [x] **Step 2: Write the failing test**

`tests/chat_template_render.rs`:

```rust
//! Rendering is asserted against literal expected output, never "contains".
//! A template that emitted the right markers in the wrong order would pass a
//! `contains` check while prompting the model incorrectly — which is the exact
//! defect this work exists to remove.

use lightbulb::api::chat_template::{ChatTemplate, Resolution};
use lightbulb::contracts::validation::RawMessage;

fn msgs() -> Vec<RawMessage> {
    vec![
        RawMessage { role: "user".into(), content: "Name the capital of France.".into() },
    ]
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
    let t = ChatTemplate { source: src.to_string(), resolved_by: Resolution::Registry };
    let out = t.render(&msgs(), "<s>", "</s>").expect("render failed");
    assert_eq!(out, "<|user|>\nName the capital of France.</s>\n<|assistant|>\n");
}

/// Templates call `raise_exception` to reject message orders they do not
/// support. Unregistered, minijinja reports "unknown function", which is
/// indistinguishable from our own bug. This asserts we support the construct,
/// not merely that it happened not to be used.
#[test]
fn raise_exception_is_registered() {
    let src = "{{ raise_exception('nope') }}";
    let t = ChatTemplate { source: src.to_string(), resolved_by: Resolution::Registry };
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
    let t = ChatTemplate { source: src.to_string(), resolved_by: Resolution::Registry };
    assert_eq!(t.render(&msgs(), "<s>", "</s>").unwrap(), "<s>|</s>");
}
```

- [x] **Step 3: Run it to verify it fails**

```bash
cargo test -j 4 --test chat_template_render 2>&1 | tail -20
```

Expected: FAIL to compile — `unresolved import lightbulb::api::chat_template`.
**Confirm `running N tests` shows 0 and the failure is the import**, not a silent skip.

- [x] **Step 4: Implement**

`src/api/chat_template.rs`:

```rust
//! Resolve and render a model's chat template.
//!
//! `/v1/chat/completions` previously built its prompt as
//! `messages.map(|m| format!("{}: {}", m.role, m.content)).join("\n")`, which is
//! not any model's template. A chat model prompted that way behaves like a base
//! model: it continues text rather than answering, and does not reliably emit
//! EOS. Measured on TinyLlama-1.1B-Chat, EOS fired in 1 of 6 trials.

use serde::{Deserialize, Serialize};

use crate::contracts::validation::RawMessage;

pub mod registry;

/// Which tier produced the template.
///
/// Recorded rather than discarded because a name-matched guess and a probed
/// result are otherwise indistinguishable, and the first person to doubt the
/// template would have to redo the work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resolution {
    Sidecar,
    TokenizerConfig,
    VocabSignature,
    Registry,
    Probe,
    /// No template found. The caller falls back to the legacy join and logs.
    None,
}

#[derive(Debug, Clone)]
pub struct ChatTemplate {
    pub source: String,
    pub resolved_by: Resolution,
}

impl ChatTemplate {
    /// Render `messages` into a prompt string.
    ///
    /// Errors rather than panicking: a template that fails to render must fall
    /// through to the next tier, not fail the request.
    pub fn render(
        &self,
        messages: &[RawMessage],
        bos_token: &str,
        eos_token: &str,
    ) -> anyhow::Result<String> {
        let mut env = minijinja::Environment::new();

        // Templates call this to reject unsupported message orders. Without it
        // minijinja reports "unknown function", which reads like our bug.
        env.add_function("raise_exception", |msg: String| {
            Err::<(), _>(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        });

        env.add_template("chat", &self.source)?;
        let tmpl = env.get_template("chat")?;

        let msgs: Vec<_> = messages
            .iter()
            .map(|m| minijinja::context! { role => m.role, content => m.content })
            .collect();

        let out = tmpl.render(minijinja::context! {
            messages => msgs,
            bos_token => bos_token,
            eos_token => eos_token,
            add_generation_prompt => true,
        })?;
        Ok(out)
    }
}
```

Add to `src/api/mod.rs` beside the other `pub mod` lines:

```rust
pub mod chat_template;
```

Create `src/api/chat_template/registry.rs` as an empty placeholder for Task 2:

```rust
//! Family table. Isolated from resolution logic because this is the part that
//! rots — it encodes knowledge that lives in checkpoints and changes as models
//! ship. Filled in by Task 2.
```

- [x] **Step 5: Run the tests**

```bash
cargo test -j 4 --test chat_template_render 2>&1 | tail -20
```

Expected: `running 3 tests` … `3 passed; 0 failed`. **If it says `running 0 tests`, the file did not compile in — stop and fix that before proceeding.**

- [x] **Step 6: Prove the exact-match test has teeth**

Change the expected string in `zephyr_template_renders_exactly` to `"<|user|>\nName the capital of France.</s>\n"` (dropping the trailing `<|assistant|>\n`). Re-run. It **must** fail. Restore, then `touch src/api/chat_template.rs` — a byte-exact restore is not build-exact and cargo will otherwise reuse the mutant binary.

- [x] **Step 7: Format and commit**

```bash
rustfmt --edition 2024 src/api/chat_template.rs tests/chat_template_render.rs
git add Cargo.toml Cargo.lock src/api/chat_template.rs src/api/chat_template/registry.rs src/api/mod.rs tests/chat_template_render.rs
git commit -m "feat(api): Render chat templates with minijinja"
```

---

## Task 2: Tier resolution and the sidecar

**Files:**
- Modify: `src/api/chat_template.rs`
- Modify: `src/api/chat_template/registry.rs`
- Test: `tests/chat_template_render.rs`

**Interfaces:**
- Consumes: `ChatTemplate`, `Resolution` from Task 1.
- Produces: `resolve(model_dir: &Path) -> ChatTemplate`, `Sidecar { template, resolved_by, evidence, resolved_at, model_fingerprint }`, `write_sidecar(dir, &Sidecar)`, `read_sidecar(dir) -> Option<Sidecar>`, `fingerprint(dir) -> String`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/chat_template_render.rs`:

```rust
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

/// Tier order is honoured: a model with BOTH a tokenizer_config template and a
/// registry entry resolves via tier 1. Fails if the tiers are reordered or a
/// later tier overwrites an earlier result.
#[test]
fn tokenizer_config_wins_over_registry() {
    let d = tmp_model_dir("tier-order");
    std::fs::write(
        d.join("tokenizer_config.json"),
        r#"{"chat_template":"FROM_TOKENIZER_CONFIG"}"#,
    )
    .unwrap();
    let t = lightbulb::api::chat_template::resolve(&d);
    assert_eq!(t.resolved_by, Resolution::TokenizerConfig);
    assert_eq!(t.source, "FROM_TOKENIZER_CONFIG");
}

/// The sidecar outranks everything — it is the probe's persisted answer.
#[test]
fn sidecar_wins_over_tokenizer_config() {
    let d = tmp_model_dir("sidecar-wins");
    std::fs::write(
        d.join("tokenizer_config.json"),
        r#"{"chat_template":"FROM_TOKENIZER_CONFIG"}"#,
    )
    .unwrap();
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
    assert_eq!(back.resolved_by, Resolution::Probe);
    assert_eq!(back.evidence, "EOS 8/8 with zephyr, 1/8 with llama2");
}

/// A sidecar written for a different checkpoint must be rejected, so a
/// directory reused for another model does not silently inherit its template.
/// This exists because writing the fingerprint but never checking it is the
/// likely slip.
#[test]
fn stale_sidecar_is_rejected() {
    let d = tmp_model_dir("stale");
    let sc = lightbulb::api::chat_template::Sidecar {
        template: "STALE".into(),
        resolved_by: Resolution::Probe,
        evidence: "e".into(),
        resolved_at: "2026-08-08T00:00:00Z".into(),
        model_fingerprint: "0000000000000000".into(), // deliberately wrong
    };
    lightbulb::api::chat_template::write_sidecar(&d, &sc).unwrap();
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
        let t = ChatTemplate { source: src.to_string(), resolved_by: Resolution::Registry };
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
}
```

- [ ] **Step 2: Run to verify they fail**

```bash
cargo test -j 4 --test chat_template_render 2>&1 | tail -20
```

Expected: compile failure — `resolve`, `Sidecar`, `write_sidecar`, `read_sidecar`, `fingerprint` do not exist.

- [ ] **Step 3: Implement resolution**

Append to `src/api/chat_template.rs`:

```rust
use std::path::Path;

pub const SIDECAR_NAME: &str = "lightbulb-chat-template.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sidecar {
    pub template: String,
    pub resolved_by: Resolution,
    /// What justified the choice, in prose. Load-bearing: without it a guess
    /// and a measurement look the same on disk.
    pub evidence: String,
    pub resolved_at: String,
    pub model_fingerprint: String,
}

/// Hash of `config.json`. Cheap, and it changes whenever the checkpoint does.
pub fn fingerprint(model_dir: &Path) -> String {
    use std::hash::{Hash, Hasher};
    let bytes = std::fs::read(model_dir.join("config.json")).unwrap_or_default();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut h);
    format!("{:016x}", h.finish())
}

pub fn write_sidecar(model_dir: &Path, sc: &Sidecar) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(sc)?;
    std::fs::write(model_dir.join(SIDECAR_NAME), json)?;
    Ok(())
}

/// Read the sidecar, rejecting one written for a different checkpoint.
pub fn read_sidecar(model_dir: &Path) -> Option<Sidecar> {
    let raw = std::fs::read_to_string(model_dir.join(SIDECAR_NAME)).ok()?;
    let sc: Sidecar = serde_json::from_str(&raw).ok()?;
    if sc.model_fingerprint != fingerprint(model_dir) {
        tracing::warn!(
            "ignoring {SIDECAR_NAME}: fingerprint {} does not match checkpoint {}",
            sc.model_fingerprint,
            fingerprint(model_dir)
        );
        return None;
    }
    Some(sc)
}

/// Resolve a template, cheapest tier first. Stops at the first hit.
///
/// Tier 4 (the probe) is deliberately absent: it costs generations and its
/// conclusion gets persisted, so it is an operator action. See spec §4.
pub fn resolve(model_dir: &Path) -> ChatTemplate {
    // Tier 0 — sidecar.
    if let Some(sc) = read_sidecar(model_dir) {
        tracing::info!("chat template: sidecar ({:?})", sc.resolved_by);
        return ChatTemplate { source: sc.template, resolved_by: Resolution::Sidecar };
    }

    // Tier 1 — the authoritative source, when the checkpoint ships one.
    if let Ok(raw) = std::fs::read_to_string(model_dir.join("tokenizer_config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(t) = v.get("chat_template").and_then(|t| t.as_str()) {
                tracing::info!("chat template: tokenizer_config.json");
                return ChatTemplate {
                    source: t.to_string(),
                    resolved_by: Resolution::TokenizerConfig,
                };
            }
        }
    }

    // Tier 2 — vocabulary signature. Only fires when markers are *added
    // tokens*; TinyLlama's `<|user|>` is ordinary text, so this misses it.
    if let Some(t) = registry::from_vocab_signature(model_dir) {
        tracing::info!("chat template: vocabulary signature");
        return ChatTemplate { source: t, resolved_by: Resolution::VocabSignature };
    }

    // Tier 3 — family registry. Heuristic; §3's monitor exists for when it is
    // wrong.
    if let Some(t) = registry::from_family(model_dir) {
        tracing::info!("chat template: family registry (heuristic)");
        return ChatTemplate { source: t, resolved_by: Resolution::Registry };
    }

    tracing::warn!(
        "no chat template resolved for {}; falling back to the legacy role: content join. \
         Run `lightbulb-probe {}` to determine one.",
        model_dir.display(),
        model_dir.display()
    );
    ChatTemplate { source: String::new(), resolved_by: Resolution::None }
}
```

- [ ] **Step 4: Implement the registry**

Replace `src/api/chat_template/registry.rs`:

```rust
//! Family table. Isolated from resolution logic because this is the part that
//! rots — it encodes knowledge that lives in checkpoints and changes as models
//! ship.
//!
//! Tier order limits the damage: any model shipping its own `chat_template`
//! never reaches here. The registry serves only models that omit one.

use std::path::Path;

// A newline in TRAILING SOURCE POSITION is stripped by both Jinja2 and
// minijinja (`keep_trailing_newline` defaults false in each, and transformers
// does not change it). So the generation-prompt newline must sit INSIDE a
// variable tag — `{{ '<|assistant|>\n' }}` — not after the closing `%}`.
// Task 1 shipped this bug and it cost a debugging cycle; these three are
// written in the safe form deliberately.
pub const ZEPHYR: &str = "{% for m in messages %}{{ '<|' + m.role + '|>\n' + m.content + eos_token + '\n' }}{% endfor %}{{ '<|assistant|>\n' }}";
pub const CHATML: &str = "{% for m in messages %}{{ '<|im_start|>' + m.role + '\n' + m.content + '<|im_end|>\n' }}{% endfor %}{{ '<|im_start|>assistant\n' }}";
pub const LLAMA2: &str = "{% for m in messages %}{% if m.role == 'user' %}[INST] {{ m.content }} [/INST]{% else %}{{ m.content }}{{ eos_token }}{% endif %}{% endfor %}";

/// Every candidate the probe tries, named for its report.
pub fn candidates() -> Vec<(&'static str, &'static str)> {
    vec![("zephyr", ZEPHYR), ("chatml", CHATML), ("llama2", LLAMA2)]
}

/// Tier 2 — markers present as *added tokens* in the vocabulary.
///
/// Deliberately checks `added_tokens` rather than searching the whole vocab:
/// `[INST]` appears as ordinary text in many vocabularies and would produce a
/// confident wrong answer.
pub fn from_vocab_signature(model_dir: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(model_dir.join("tokenizer.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let added: Vec<String> = v
        .get("added_tokens")?
        .as_array()?
        .iter()
        .filter_map(|t| t.get("content")?.as_str().map(str::to_string))
        .collect();

    if added.iter().any(|t| t == "<|im_start|>") {
        return Some(CHATML.to_string());
    }
    if added.iter().any(|t| t == "[INST]") {
        return Some(LLAMA2.to_string());
    }
    None
}

/// Tier 3 — `config.json` fields plus the directory name.
pub fn from_family(model_dir: &Path) -> Option<String> {
    let name = model_dir.to_string_lossy().to_lowercase();

    // Directory name first: it carries the chat-variant distinction that
    // config.json's `model_type` does not. A base and a chat model share
    // `model_type: llama` but need different templates — and prompting a base
    // model with a chat template is the milder error of the two.
    if name.contains("tinyllama") && name.contains("chat") {
        return Some(ZEPHYR.to_string());
    }
    if name.contains("qwen") || name.contains("chatml") {
        return Some(CHATML.to_string());
    }
    if name.contains("llama-2") || name.contains("llama2") || name.contains("mistral") {
        return Some(LLAMA2.to_string());
    }
    None
}
```

- [ ] **Step 5: Run the tests**

```bash
cargo test -j 4 --test chat_template_render 2>&1 | tail -25
```

Expected: `running 8 tests` … `8 passed; 0 failed`.

- [ ] **Step 6: Prove the fingerprint check has teeth**

This is the assertion most likely to be vacuous. In `read_sidecar`, delete the `if sc.model_fingerprint != fingerprint(model_dir)` block. Re-run: `stale_sidecar_is_rejected` **must** fail. Restore and `touch src/api/chat_template.rs`.

- [ ] **Step 7: Format and commit**

```bash
rustfmt --edition 2024 src/api/chat_template.rs src/api/chat_template/registry.rs tests/chat_template_render.rs
git add -A src/api tests/chat_template_render.rs
git commit -m "feat(api): Resolve chat templates through four free tiers"
```

---

## Task 3: Wire into the API — both chat.rs sites

**Files:**
- Modify: `src/api/mod.rs` (`AppState`, startup)
- Modify: `src/api/openai/chat.rs` (`:186` and `:409`)
- Test: `tests/chat_template_e2e.rs` **new**

**Interfaces:**
- Consumes: `resolve`, `ChatTemplate`, `Resolution` from Tasks 1–2.
- Produces: `AppState.chat_template: Option<Arc<ResolvedTemplate>>`.

**Read the spec's CORRECTED 2026-08-08 block first.** There are two sites in this file, not one. `:409` is the streaming path and is easy to miss because the non-streaming fix looks complete on its own.

- [ ] **Step 1: Write the failing test**

`tests/chat_template_e2e.rs`:

```rust
//! Behavioural: a chat model prompted with its own template answers and stops.
#![cfg(feature = "fuel-engine")]

// Reuse the harness shape from tests/api_result_metadata.rs — same
// post_json/post_raw/sse_chunks helpers, same TINYLLAMA_DIR resolution.
// Copy them in; they are ~40 lines and sharing them across integration test
// binaries would need a `tests/common/` module, which is a larger change than
// this task warrants.

/// The rendered prompt must not contain the legacy join, on EITHER path.
///
/// Asserted through behaviour rather than by inspecting the prompt: with the
/// template applied, TinyLlama answers and emits EOS. `finish_reason` is the
/// observable, and it is the same signal that revealed the original defect.
#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint"]
async fn templated_chat_stops_on_eos() {
    let (status, v) = post_json(
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tinyllama",
            "messages": [{"role": "user", "content": "Name the capital of France."}],
            "max_tokens": 64,
            "temperature": 0.0
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    eprintln!("templated_chat_stops_on_eos: {v}");
    assert_eq!(
        v["choices"][0]["finish_reason"], "stop",
        "a templated chat model ran its whole budget instead of stopping — \
         the template is not reaching the prompt"
    );
}

/// The streaming path uses the same template. This test exists because
/// chat.rs has TWO join sites and fixing only the non-streaming one leaves
/// this green-looking and wrong.
#[tokio::test]
#[ignore = "needs the TinyLlama checkpoint"]
async fn templated_streaming_stops_on_eos() {
    let (status, body) = post_raw(
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tinyllama",
            "messages": [{"role": "user", "content": "Name the capital of France."}],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let chunks = sse_chunks(&body);
    let last = chunks.last().expect("no data frames");
    assert_eq!(
        last["choices"][0]["finish_reason"], "stop",
        "streaming ran its whole budget — chat.rs:409 still uses the legacy join"
    );
}
```

- [ ] **Step 2: Run to verify both fail**

```bash
pwsh -NoProfile -File C:/Projects/fuel/scripts/gpu-run.ps1 -Project lightbulb -- \
  cargo test --release --features fuel-engine --test chat_template_e2e -j 4 -- \
  --ignored --nocapture --test-threads=1 2>&1 | tail -25
```

Expected: `running 2 tests`, both FAIL reporting `"length"`.
This is the spec's measured baseline (EOS fired 1-in-6 with the legacy join), so a **pass here before the fix means the harness is not exercising the real path** — investigate rather than celebrate.

- [ ] **Step 3: Add the field to `AppState`**

In `src/api/mod.rs`, inside `pub struct AppState`:

```rust
    /// Chat template for the loaded model, resolved once at startup.
    ///
    /// `None` when no model runner started, or when no tier produced one — in
    /// which case the handlers fall back to the legacy join and say so.
    pub chat_template: Option<std::sync::Arc<crate::api::chat_template::ChatTemplate>>,
```

Initialise it to `None` in the `AppState { .. }` literal, then resolve it inside the `Ok(sender) =>` arm beside `state.inference_tx = Some(sender);`:

```rust
                    Ok(sender) => {
                        state.inference_tx = Some(sender);
                        let t = crate::api::chat_template::resolve(&model_path);
                        if t.resolved_by != crate::api::chat_template::Resolution::None {
                            state.chat_template = Some(std::sync::Arc::new(t));
                        }
                        println!("Started model runner for {}", model_path.display());
                    }
```

Resolution belongs here and only here: it reads files, and a per-request read would put filesystem I/O in the request path for a value that cannot change while the process runs.

- [ ] **Step 4: Replace BOTH join sites**

Add to `src/api/openai/chat.rs`:

```rust
/// Build the prompt for `messages`, preferring the model's chat template.
///
/// The legacy `role: content` join is kept as the fallback rather than deleted:
/// a server that refuses to answer because no template resolved is worse than
/// one that answers badly and logs why. Every fallback is logged, so a model
/// silently prompted as a base model leaves a trace.
fn build_prompt(state: &AppState, messages: &[ChatMessage]) -> String {
    let raw: Vec<RawMessage> = messages
        .iter()
        .map(|m| RawMessage { role: m.role.clone(), content: m.content.clone() })
        .collect();

    if let Some(t) = &state.chat_template {
        match t.render(&raw, "<s>", "</s>") {
            Ok(p) => return p,
            Err(e) => tracing::warn!(
                "chat template ({:?}) failed to render: {e}; using the legacy join",
                t.resolved_by
            ),
        }
    }
    raw.iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}
```

Then replace the join at **`:186`** (in `create_chat_completion`) and at **`:409`** (in `create_chat_stream`) with `build_prompt(&state, &request.messages)`. Both. Grep to confirm none remain:

```bash
grep -n 'format!("{}: {}", msg.role, msg.content)' src/api/openai/chat.rs
```

Expected: only the one occurrence inside `build_prompt`.

- [ ] **Step 5: Run the tests**

```bash
pwsh -NoProfile -File C:/Projects/fuel/scripts/gpu-run.ps1 -Project lightbulb -- \
  cargo test --release --features fuel-engine --test chat_template_e2e -j 4 -- \
  --ignored --nocapture --test-threads=1 2>&1 | tail -25
```

Expected: `running 2 tests` … `2 passed; 0 failed`.

- [ ] **Step 6: Confirm the existing suite still passes**

The metadata tests assert `finish_reason` on specific prompts, and **templating changes what the model generates**, so `truncated_generation_reports_length` and `eos_terminated_generation_reports_stop` may now report the opposite reason. That is a real behaviour change, not a regression — but it must be *observed and re-grounded*, not assumed either way.

```bash
pwsh -NoProfile -File C:/Projects/fuel/scripts/gpu-run.ps1 -Project lightbulb -- \
  cargo test --release --features fuel-engine --test api_result_metadata -j 4 -- \
  --ignored --nocapture --test-threads=1 2>&1 | tail -30
```

If a test flips, update **its prompt or its expectation with the observed output recorded in the doc comment**, exactly as that file already does for its prompt choices. Do not weaken an assertion to accommodate the change.

- [ ] **Step 7: Format and commit**

```bash
rustfmt --edition 2024 src/api/mod.rs src/api/openai/chat.rs tests/chat_template_e2e.rs
git add -A
git commit -m "feat(api): Prompt chat models with their own template"
```

---

## Task 4: The contract path

**Files:**
- Modify: `src/contracts/executor.rs`
- Modify: `src/contracts/validation.rs` (**delete** `messages_to_prompt`)
- Modify: `src/api/openai/chat.rs` (contract caller)
- Test: `tests/contracts_integration.rs`

**Interfaces:**
- Consumes: `build_prompt` from Task 3.
- Produces: `execute_contract` with `F: Fn(Vec<RawMessage>) -> Fut`.

> ### The interface changed in Task 3 — check signatures before copying anything below
>
> Task 3 replaced the type this task consumes, for a reason that matters here:
> the plan originally had `build_prompt` call `render(&raw, "<s>", "</s>")` with
> **hardcoded** bos/eos. That is correct only for the Llama-2 family. Llama-3
> uses `<|begin_of_text|>`/`<|eot_id|>`, Qwen uses `<|im_end|>` — and rendering
> another family's template with those literals *succeeds*, so nothing falls
> through and nothing is logged.
>
> | Was | Now |
> | --- | --- |
> | `Option<Arc<ChatTemplate>>` | `Option<Arc<ResolvedTemplate>>` |
> | `render(&msgs, bos, eos)` | `render(&msgs)` — tokens bound at resolution |
> | `.resolved_by` field | `.resolved_by()` method |
>
> **Do not reintroduce a literal `"<s>"` or `"</s>"` anywhere in this task.**
> `ResolvedTemplate` binds the checkpoint's own tokens to the template
> deliberately so that no caller *can* supply them. The contract path is the
> last remaining prompt site; hardcoding here would put the defect back with a
> provenance label claiming the template was read from the checkpoint.
>
> **And flip `InferenceJob::add_special_tokens` to `false` when you do.**
> `executor.rs`'s `InferenceJob` currently sets it `true`, which is correct
> *only while* the contract loop builds its prompt with the legacy join — raw
> text carries no special tokens, so the tokenizer must add them. The moment
> this task renders the checkpoint's template there, the prompt carries its own
> BOS and `true` doubles it: real templates open with `{{ bos_token }}` and
> Llama-family tokenizers prepend BOS again via their `TemplateProcessing`
> post_processor. The render still succeeds and nothing is logged, which is why
> this is called out rather than left to be noticed. Prefer routing the call
> through `chat.rs`'s `BuiltPrompt` so the flag travels with the text instead of
> being restated. See `chat.rs::build_prompt` for the full argument, including
> why binding `bos_token` to `""` was rejected.

The contract path re-renders per attempt against a *mutated* message list (`executor.rs:170`, `:176-179`, `:183`), so it cannot be fixed by rendering upstream. See the spec's CORRECTED block.

- [ ] **Step 1: Write the failing test**

`tests/contracts_integration.rs` already has the fixtures you need — `canned()`, `mock_infer()`, `user_messages()`, `choices()`. There is **no** `json_contract()`; build the spec inline as the existing tests do (`OutputContractSpec::EnumChoice { .. }` at `:83`).

Add a test whose stub captures what it was handed:

```rust
/// The contract path must render through the template on EVERY attempt.
///
/// The message list GROWS between attempts — inject_contract_instruction and
/// tightening_message both push — so a prompt rendered once upstream would be
/// stale from attempt 2 onward. This asserts the callback sees each mutated
/// list, which is what makes upstream rendering impossible and forces the
/// signature change.
#[tokio::test]
async fn contract_callback_sees_each_mutated_message_list() {
    let spec = OutputContractSpec::EnumChoice {
        choices: choices(),
        case_sensitive: false,
        allow_index: true,
    };

    let seen: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let seen2 = seen.clone();

    // Never returns a valid label, so all three attempts run.
    let infer = move |msgs: Vec<RawMessage>| {
        seen2.lock().unwrap().push(msgs.len());
        std::future::ready(Ok::<CompletionResult, anyhow::Error>(canned("garbage")))
    };

    let _ = execute_contract(
        &user_messages("Is the service healthy?"),
        "test-model",
        &spec,
        3,
        &[],
        infer,
    )
    .await;

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 3, "expected three attempts, saw {}", seen.len());
    assert!(
        seen[1] > seen[0],
        "attempt 2 received a message list of the same length as attempt 1 ({:?}) \
         — the tightening message is not reaching the callback",
        *seen
    );
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cargo test -j 4 --test contracts_integration 2>&1 | tail -20
```

Expected: compile failure — the callback takes `String`.

- [ ] **Step 3: Change the callback type**

In `src/contracts/executor.rs`:

```rust
where
    F: Fn(Vec<RawMessage>) -> Fut,
    Fut: Future<Output = anyhow::Result<CompletionResult>>,
```

Replace line 183:

```rust
            // The caller renders — it owns the template. The contracts module
            // deliberately no longer knows how a prompt is formatted, which is
            // what lets validation::messages_to_prompt be deleted rather than
            // left behind to drift.
            let completion = infer(msgs.clone()).await?;
```

`Vec` rather than `&[RawMessage]`: a borrowing callback returning a `Future` needs an HRTB that buys nothing here, and `msgs` is already cloned per contract at `:167`.

- [ ] **Step 4: Delete `messages_to_prompt`**

Remove it entirely from `src/contracts/validation.rs` (the `// ─── Prompt conversion helper ───` block, ~lines 138-151). Its doc comment presents the duplicated defect as intentional; re-pointing it at the template would leave a second renderer to drift.

```bash
grep -rn "messages_to_prompt" src/ tests/
```

Expected: no matches.

- [ ] **Step 5: Update `mock_infer` — this fixes every other contract test at once**

`mock_infer` (`tests/contracts_integration.rs:43`) is typed `impl Fn(String) -> ...` and is used by **every** test in the file, so they all break together and all fix together. Change only its closure parameter:

```rust
fn mock_infer(responses: Vec<&'static str>) -> (
    Arc<Mutex<VecDeque<String>>>,
    impl Fn(Vec<RawMessage>) -> std::future::Ready<anyhow::Result<CompletionResult>> + Clone,
) {
    // ... unchanged body ...
    let f = move |_msgs: Vec<RawMessage>| {
        // ... unchanged ...
    };
```

Also check `tests/contracts_live.rs`, which calls `execute_contract` too.

- [ ] **Step 6: Update the production callers**

`execute_contract_with_runner` (`executor.rs:52`) and the closure in `chat.rs` (`:340`) both take `move |prompt|`. Change to `move |msgs: Vec<RawMessage>|` and render inside. In `chat.rs` the template is on `state`, so capture it before the closure:

```rust
    let template = state.chat_template.clone();
    // ...
        move |msgs: Vec<RawMessage>| {
            let prompt = match &template {
                Some(t) => t.render(&msgs).unwrap_or_else(|e| {
                    tracing::warn!("chat template failed to render: {e}; using the legacy join");
                    legacy_join(&msgs)
                }),
                None => legacy_join(&msgs),
            };
            // ... existing body, using `prompt`
        },
```

Extract `legacy_join(&[RawMessage]) -> String` from `build_prompt` in Task 3 so both share it.

`execute_contract_with_runner` has no `AppState`; give it a `template: Option<Arc<ResolvedTemplate>>` parameter and thread it through from the caller.

- [ ] **Step 7: Run the tests**

```bash
cargo test -j 4 --test contracts_integration 2>&1 | tail -20
cargo test -j 4 --lib 2>&1 | tail -5
```

Expected: contract tests pass; **`running N tests` with N > 0**.

- [ ] **Step 8: Format and commit**

```bash
rustfmt --edition 2024 src/contracts/executor.rs src/contracts/validation.rs src/api/openai/chat.rs
git add -A
git commit -m "fix(contracts): Render each attempt through the chat template"
```

---

## Task 5: EOS-rate monitor

**Files:**
- Create: `src/engine/eos_monitor.rs`
- Modify: `src/engine/mod.rs`
- Modify: `src/api/openai/chat.rs` (record each completion)
- Test: unit tests in `src/engine/eos_monitor.rs`

**Interfaces:**
- Consumes: `FinishReason` from `src/engine/model_runner.rs`.
- Produces: `EosMonitor::new()`, `record(&self, FinishReason)`, `stop_rate(&self) -> Option<f64>`.

This **monitors and never acts.** A server that changes its prompting mid-flight on a heuristic is harder to debug than one consistently wrong that says so (spec §3).

> **Implementation notes (2026-08-12).** Four departures from the text above,
> each with its own test:
>
> 1. **Three completion sites, not one.** Step 5 wires only
>    `create_chat_completion`. The streaming path's `Done` frame and the
>    contract loop's final attempt observe a `FinishReason` too, and a monitor
>    fed by only the first never fills its window on a server whose clients
>    stream — which is indistinguishable in the log from a healthy model. All
>    three record; the contract path records once per request, not once per
>    attempt.
> 2. **The warning is edge-triggered.** Step 5's prose says "warn once per
>    crossing" and its code sample warns on every completion for as long as the
>    rate stays low. `record` returns `Some(rate)` only on the transition and
>    resets when a reading recovers.
> 3. **The message names the monitor's own window**, via `EosMonitor::window()`.
>    The sample quotes `DEFAULT_WINDOW`, which misreports any monitor built with
>    another — including every one in the tests.
> 4. **`parking_lot::Mutex`, not `std`.** `std`'s `.lock().unwrap()` turns a
>    poisoned lock into a panic on the request path, which would make an
>    observer able to fail a request.
>
> `new()` also clamps the window to `>= 1`: a zero window divides by zero, and
> `NaN < min` is `false`, so the monitor would go silently quiet rather than
> fail.

- [x] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model_runner::FinishReason;

    /// Below the window, there is not enough evidence to judge. Returning None
    /// rather than a rate computed from 3 samples is the point: a warning from
    /// a cold server pointing at the wrong subsystem is the failure mode.
    #[test]
    fn reports_nothing_until_the_window_fills() {
        let m = EosMonitor::new(20, 0.25);
        for _ in 0..19 {
            m.record(FinishReason::Length);
        }
        assert_eq!(m.stop_rate(), None);
        m.record(FinishReason::Length);
        assert_eq!(m.stop_rate(), Some(0.0));
    }

    /// The window slides — old completions must age out, or a server that
    /// recovers never stops warning.
    #[test]
    fn window_slides() {
        let m = EosMonitor::new(4, 0.25);
        for _ in 0..4 {
            m.record(FinishReason::Length);
        }
        assert_eq!(m.stop_rate(), Some(0.0));
        for _ in 0..4 {
            m.record(FinishReason::Stop);
        }
        assert_eq!(m.stop_rate(), Some(1.0), "the window did not slide");
    }

    #[test]
    fn below_threshold_is_flagged() {
        let m = EosMonitor::new(4, 0.25);
        m.record(FinishReason::Stop);
        for _ in 0..3 {
            m.record(FinishReason::Length);
        }
        // 1/4 = 0.25, which is NOT below 0.25 — boundary is inclusive-pass.
        assert!(!m.should_warn());
        let m2 = EosMonitor::new(4, 0.25);
        for _ in 0..4 {
            m2.record(FinishReason::Length);
        }
        assert!(m2.should_warn());
    }
}
```

- [x] **Step 2: Run to verify they fail**

```bash
cargo test -j 4 --lib eos_monitor 2>&1 | tail -15
```

Expected: compile failure. **If it reports `ok. 0 passed`, the filter matched nothing** — check the module is declared in `src/engine/mod.rs`.

- [x] **Step 3: Implement**

```rust
//! Rolling EOS-fire rate, as a signal that a heuristically-chosen chat
//! template is wrong.
//!
//! A correctly-templated chat model answering short questions terminates
//! nearly always. With the legacy `role: content` join, TinyLlama emitted EOS
//! in 1 of 6 trials — it behaved like a base model and continued text.
//!
//! Constants are a starting point from ONE model's measurement, not a
//! calibrated value. If it proves noisy, raise the window first: the failure
//! mode of a too-eager warning is a log line pointing at the wrong subsystem.

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::engine::model_runner::FinishReason;

pub const DEFAULT_WINDOW: usize = 20;
pub const DEFAULT_MIN_STOP_RATE: f64 = 0.25;

#[derive(Debug)]
pub struct EosMonitor {
    window: usize,
    min_stop_rate: f64,
    recent: Mutex<VecDeque<bool>>,
}

impl EosMonitor {
    pub fn new(window: usize, min_stop_rate: f64) -> Self {
        Self { window, min_stop_rate, recent: Mutex::new(VecDeque::with_capacity(window)) }
    }

    pub fn record(&self, r: FinishReason) {
        let mut q = self.recent.lock().unwrap();
        if q.len() == self.window {
            q.pop_front();
        }
        q.push_back(matches!(r, FinishReason::Stop));
    }

    /// `None` until the window is full — a rate from 3 samples is noise.
    pub fn stop_rate(&self) -> Option<f64> {
        let q = self.recent.lock().unwrap();
        if q.len() < self.window {
            return None;
        }
        Some(q.iter().filter(|s| **s).count() as f64 / q.len() as f64)
    }

    pub fn should_warn(&self) -> bool {
        self.stop_rate().is_some_and(|r| r < self.min_stop_rate)
    }
}

impl Default for EosMonitor {
    fn default() -> Self {
        Self::new(DEFAULT_WINDOW, DEFAULT_MIN_STOP_RATE)
    }
}
```

Declare in `src/engine/mod.rs`: `pub mod eos_monitor;`

- [x] **Step 4: Run the tests**

```bash
cargo test -j 4 --lib eos_monitor 2>&1 | tail -15
```

Expected: `3 passed; 0 failed`.

- [x] **Step 5: Wire it in**

Add `pub eos_monitor: std::sync::Arc<crate::engine::eos_monitor::EosMonitor>` to `AppState`, default-constructed. In `create_chat_completion`, after the runner returns, `state.eos_monitor.record(result.finish_reason);` and warn once per crossing:

```rust
    if state.eos_monitor.should_warn() {
        tracing::warn!(
            "only {:.0}% of the last {} completions stopped on EOS. The chat \
             template ({:?}) may be wrong for this model. This is a warning \
             only — no template will be changed automatically.",
            state.eos_monitor.stop_rate().unwrap_or(0.0) * 100.0,
            crate::engine::eos_monitor::DEFAULT_WINDOW,
            state.chat_template.as_ref().map(|t| t.resolved_by()),
        );
    }
```

- [x] **Step 6: Format and commit**

```bash
rustfmt --edition 2024 src/engine/eos_monitor.rs src/engine/mod.rs src/api/mod.rs src/api/openai/chat.rs
git add -A
git commit -m "feat(engine): Warn when the EOS rate suggests a wrong template"
```

---

## Task 6: The probe binary

> ### REWRITTEN 2026-08-12 after an audit. Do not implement the previous version.
>
> The earlier Task 6 put a `chat-template probe` subcommand on `lightbulb-cli`
> and generated `--trials 8` times per candidate. Both halves were wrong, and
> both were caught before implementation rather than during it:
>
> 1. **`lightbulb-cli` cannot run a model.** `grep -c "lightbulb::"
>    src/bin/lightbulb-cli.rs` returns **0** — it imports only `clap`,
>    `futures_util`, `serde`, `std::io` and talks to a server over HTTP. The
>    probe needs in-process inference. `src/main.rs` already links the library,
>    so a **sibling binary has exactly the server's dependency footprint** and
>    adds no new link risk.
> 2. **N trials cannot vary on the default build.** The only readers of
>    temperature are `src/model_fuel/engine_model.rs:231,272` — the
>    `fuel-engine` path. The default backend is greedy
>    (`parallel_model_manager.rs`, `logits_slice.argmax(0)`), so eight trials of
>    one prompt are the same generation eight times, and every row could only
>    ever read `0/8` or `8/8`. The old `format_probe_report` justified its whole
>    design with *"A 5/8-vs-4/8 result must be visible as the coin-flip it is"*
>    — an outcome the shipped default build cannot produce.
>
> **Greedy decoding is not a problem to work around here — it is the feature.**
> One generation per candidate is deterministic and reproducible, so the probe
> reports a fact rather than a sample. The confirmation gate stays, because the
> risk it guards was never sampling noise: it is that a probe over-fits its one
> prompt, and that is just as true at N=1.

**Files:**
- Create: `src/bin/lightbulb-probe.rs`
- Modify: `Cargo.toml` (add the `[[bin]]` target)
- Modify: `src/api/chat_template.rs` (add `ProbeRow` and `format_probe_report`; fix the stale operator hint in **`resolve`**'s fallthrough — see Step 6)
- Modify: this plan file (Task 2's sample quotes the same stale command)
- Test: `tests/chat_template_render.rs`

**Interfaces:**
- Consumes: `registry::candidates()`, `ChatTemplate`, `special_tokens`, `fingerprint`, `Sidecar`, `write_sidecar`, `Resolution::Probe`, `ModelRunner::start`, `InferenceJob`, `ResponseMode`, `FinishReason`.
- Produces: `format_probe_report(&[ProbeRow]) -> String`, binary `lightbulb-probe`.

**Verified against the tree at `acd9158` — do not re-derive these from memory:**

| Fact | Value |
| --- | --- |
| `fingerprint` | `pub fn fingerprint(&Path) -> Option<String>` — **`Option`**, fails closed |
| `Sidecar` fields | `template`, `resolved_by`, `evidence`, `resolved_at`, `model_fingerprint` — **five**, no `Default` |
| `write_sidecar` | `pub fn write_sidecar(&Path, &Sidecar) -> anyhow::Result<()>` |
| `ChatTemplate::render` | `render(&self, &[RawMessage], bos_token: &str, eos_token: &str)` — **needs both tokens** |
| `candidates()` order | `[("zephyr", ZEPHYR), ("chatml", CHATML), ("llama2", LLAMA2)]` |
| `ModelRunner::start` | `(impl Into<PathBuf>, max_batch_size, context_length, dtype: Option<String>) -> Result<InferenceRequestSender>` |
| `InferenceJob` | `id`, `prompt`, `max_new_tokens`, `temperature`, `add_special_tokens`, `response_mode` — all mandatory |
| `ResponseMode::Complete` | carries `oneshot::Sender<anyhow::Result<CompletionResult>>` — **the awaited value is a `Result`**; see Step 4 |
| `CompletionResult` | exposes `text: String`, `finish_reason: FinishReason`, `completion_tokens: usize` — `text` is what feeds `first_line_of` |
| Awaiting the oneshot | yields **two** `Result` layers: `Result<anyhow::Result<CompletionResult>, oneshot::RecvError>`. Handle the outer one as `chat.rs` does — `.await.map_err(\|e\| anyhow!("inference runner dropped: {e}"))?` — then the inner per Step 4 |
| `InferenceJob.id` | mandatory; any unique string (`uuid::Uuid::new_v4().to_string()` matches the rest of the repo). Immaterial at `temperature: 0.0`, but it is not optional |
| **Import paths** | `crate::engine` re-exports only `InferenceJob`, `InferenceRequestSender`, `ModelRunner`. `ResponseMode`, `FinishReason`, `CompletionResult` must be reached at `lightbulb::engine::model_runner::{…}`; `RawMessage` at `lightbulb::contracts::validation::RawMessage` |
| **`src/bin/*.rs` is a separate crate** | so it CANNOT call `chat.rs`'s `pub(crate) run_inference_once`. The oneshot/`InferenceJob` dance must be re-implemented here — the one place this plan knowingly accepts a second copy, because the crate boundary leaves no alternative |

- [ ] **Step 1: Add the binary target**

In `Cargo.toml`, beside the existing `[[bin]]`:

```toml
# The probe runs a model in-process, so it links the library the way
# `src/main.rs` does. It is deliberately NOT a subcommand of `lightbulb-cli`:
# that binary is a pure HTTP client (zero `lightbulb::` references) and giving
# it an engine dependency would put CUDA link requirements on a tool whose job
# is to talk to a server.
[[bin]]
name = "lightbulb-probe"
path = "src/bin/lightbulb-probe.rs"
```

- [ ] **Step 2: The report formatter**

Add to `src/api/chat_template.rs`, not the binary, so it is testable without a model. Note `ProbeRow` is a named struct: the old signature was `&[(&str, usize, usize)]`, and a bare tuple of two `usize`s is exactly the shape where "fired" and "total" get transposed silently.

```rust
/// One candidate's result from a probe run.
#[derive(Debug, Clone)]
pub struct ProbeRow {
    pub candidate: &'static str,
    /// The template source that produced this row.
    ///
    /// Carried, not re-looked-up by name. `Sidecar.template` is the SOURCE, and
    /// a row that holds only the name forces the selection step to go back to
    /// `candidates()` and match on a string — which is the same
    /// "two values that must correspond, related only by convention" hazard the
    /// named fields below exist to remove, just moved into the one step that
    /// writes to disk.
    pub source: &'static str,
    /// Did generation stop on EOS rather than exhaust its budget?
    pub stopped_on_eos: bool,
    pub tokens_generated: usize,
    /// First line of what the model produced, for the operator to eyeball.
    ///
    /// **The caller enforces this shape; the formatter does not.** Build it with
    /// the helper below rather than by hand — error text from `anyhow` and
    /// `minijinja` is routinely multi-line and long, and a raw newline in this
    /// field breaks the table into rows that look like extra candidates.
    pub first_line: String,
}

/// Reduce arbitrary model or error output to something that fits one table cell.
///
/// Split first, THEN truncate: truncating first can leave a newline inside the
/// 60 chars. Empty input yields an empty cell, which is a legitimate result —
/// a model that produced nothing.
pub fn first_line_of(text: &str) -> String {
    let line = text.lines().next().unwrap_or("");
    if line.chars().count() <= 60 {
        line.to_string()
    } else {
        line.chars().take(57).collect::<String>() + "..."
    }
}

/// Format probe results as a per-candidate table.
///
/// **Reports every candidate, and never picks the winner.** Selection is the
/// caller's job (Step 4) and is deliberately separate: an operator reading this
/// table must be able to see a board where two candidates both fired, or none
/// did, and draw their own conclusion. Printing only a leader would hide
/// exactly the cases where the probe should not be trusted.
///
/// Decoding is greedy, so each row is ONE deterministic generation, not a
/// sample. A row is reproducible; re-running the probe on the same checkpoint
/// and prompt must produce the same table.
pub fn format_probe_report(rows: &[ProbeRow]) -> String {
    let mut s = String::from("candidate   EOS   tokens  output\n");
    for r in rows {
        s.push_str(&format!(
            "{:<11} {:<5} {:<7} {}\n",
            r.candidate,
            if r.stopped_on_eos { "yes" } else { "no" },
            r.tokens_generated,
            r.first_line,
        ));
    }
    s
}
```

Expected output shape — **illustrative, not measured.** The token counts and generated text below are plausible values chosen to show the layout; only the *columns and ordering* are load-bearing, and candidate order is `candidates()`' order: zephyr / chatml / llama2.

```
candidate   EOS   tokens  output
zephyr      yes   8       The capital of France is Paris.
chatml      no    48      The capital of France is Paris. The capital of
llama2      no    48      user The capital of France is Paris, and the
```

- [ ] **Step 3: Test the report shape**

Generation needs a model, so the unit test covers formatting only. Add to `tests/chat_template_render.rs` (35 tests today → 36; the file is **not** feature-gated, so a plain `cargo test --test chat_template_render` really runs it):

**Assert the whole string, not `contains`.** This file's own header says so —
*"Rendering is asserted against literal expected output, never `contains`. A
template that emitted the right markers in the wrong order would pass a
`contains` check while prompting the model incorrectly"* — and the same logic
applies here. A `contains`-based version of this test survives **all four** of:
swapping the yes/no literals, deleting the tokens column, reversing row order,
and pairing every row's outcome with the wrong candidate. The expected table is
already written out above, so the equality costs nothing.

```rust
#[test]
fn probe_report_renders_every_candidate_and_picks_no_winner() {
    use lightbulb::api::chat_template::{ProbeRow, format_probe_report};
    let rows = vec![
        ProbeRow { candidate: "zephyr", source: "SRC_Z", stopped_on_eos: true, tokens_generated: 8,
                   first_line: "The capital of France is Paris.".to_string() },
        ProbeRow { candidate: "chatml", source: "SRC_C", stopped_on_eos: false, tokens_generated: 48,
                   first_line: "The capital of France is Paris. The capital of".to_string() },
        ProbeRow { candidate: "llama2", source: "SRC_L", stopped_on_eos: false, tokens_generated: 48,
                   first_line: "user The capital of France is Paris, and the".to_string() },
    ];

    // Every column, every row, in order. Kills the yes/no swap, the dropped
    // column, the reordering, and the mispairing — none of which a `contains`
    // check can see.
    assert_eq!(
        format_probe_report(&rows),
        "candidate   EOS   tokens  output\n\
         zephyr      yes   8       The capital of France is Paris.\n\
         chatml      no    48      The capital of France is Paris. The capital of\n\
         llama2      no    48      user The capital of France is Paris, and the\n"
    );

    // Separate claim, so it is not folded into the layout equality: the report
    // must not announce a conclusion. Selection belongs to the caller (Step 5),
    // and an operator has to be able to read a two-winner or zero-winner board
    // for what it is.
    let lower = format_probe_report(&rows).to_lowercase();
    for banned in ["winner", "best", "recommend", "selected"] {
        assert!(!lower.contains(banned), "the report named a winner ({banned}):\n{lower}");
    }

    // The template SOURCE must never reach the operator's table — it is a
    // multi-line Jinja blob that would destroy the layout. This is why
    // `ProbeRow` carries it for Step 5 rather than the formatter printing it.
    for src in ["SRC_Z", "SRC_C", "SRC_L"] {
        assert!(!format_probe_report(&rows).contains(src), "{src} leaked into the report");
    }
}
```

- [ ] **Step 4: The probe binary**

Create `src/bin/lightbulb-probe.rs`. Model its `main` on `src/main.rs` (env-var-free here; this one uses clap). **Every decision below changes the measured result, so none of them may be left to the implementer's judgement:**

- **Fixed prompt:** one user message, `"Name the capital of France."` — the same prompt the e2e tests use, so a probe result is comparable with them.
- **`max_new_tokens: 48`.** The EOS rate is a function of the budget: too small and a correct template looks like a failure. 48 is ~6× the 8-token answer TinyLlama gives.
- **`temperature: 0.0`.** Greedy, and the whole basis of the one-trial design.
- **`add_special_tokens: false`.** This is what the server passes for every `ResolvedTemplate::render` output, and matching it is the entire reason a probe result is comparable with what the server will then do. **Do not justify it with "the template already emits BOS"** — `grep -n "bos_token" src/api/chat_template/registry.rs` returns nothing, so none of these three candidates interpolates it, and the probe's prompts carry no BOS at all. That is a difference from the general templated case, not a reason to flip the flag: `true` would add a BOS the template did not ask for, which is exactly what `tests/api_result_metadata.rs` records as the defect Task 3's review removed. See `RequestContext::add_special_tokens`.
- **bos/eos come from `special_tokens(&model_dir)`**, never literals. `ZEPHYR` and `LLAMA2` both interpolate `eos_token`, so a literal changes the very signal being measured.
- **`ModelRunner::start(&model_dir, 1, 512, Some("f32".to_string()))`**, once, reused across candidates. Every *test* call site passes exactly this; `src/api/mod.rs` passes `Some("f32")` too but takes its context length from config (4096 in `src/main.rs`). `None` is equivalent for dtype (`parse_dtype(None) => F32`) but gratuitously different, and this run is meant to be comparable with the e2e tests.
- **Initialise `tracing_subscriber` first**, exactly as `src/main.rs:15-21` does. Without a global subscriber every `tracing::warn!` in the library is a **no-op** — including the one in `special_tokens` that says a checkpoint declares no BOS/EOS. `ModelRunner`'s `println!`/`eprintln!` would still appear, so the output would look healthy while the single warning that invalidates the measurement is silently dropped.

**The CLI surface** (the previous version declared this and the rewrite lost it):

```rust
#[derive(Parser, Debug)]
#[command(about = "Determine a checkpoint's chat template by generating under each candidate")]
struct Args {
    /// Checkpoint directory, or a single .gguf file.
    model_dir: std::path::PathBuf,
    /// Write the sidecar without the interactive confirmation. Does NOT
    /// bypass any refusal in Step 5 — a flag that suppresses a refusal is a
    /// flag that writes a wrong answer unattended.
    #[arg(long)]
    yes: bool,
}
```

Confirmation reads a line from stdin and requires an explicit `y`/`yes`. **Treat EOF and an empty line as NO** — run from a script, `read_line` returns `Ok(0)`, and an implementation that defaults empty input to "yes" writes the sidecar unattended, which is the exact thing `--yes` exists to make deliberate.

**Per candidate:** build a `ChatTemplate { source: src.to_string(), resolved_by: Resolution::Probe }`, render, send one `InferenceJob` with `ResponseMode::Complete`, await, and record a `ProbeRow`.

**Both fallible steps need an explicit arm, and this is not a formality:**

- **`render` returns `anyhow::Result<String>`.** A candidate whose render fails is not a candidate that lost — record it as a row with `stopped_on_eos: false`, `tokens_generated: 0`, and the error text in `first_line`, so the operator sees *why* rather than seeing it silently score zero.
- **The awaited value is `anyhow::Result<CompletionResult>`, and an `Err` here means the RUNNER failed, not the template.** **Abort the whole probe immediately and propagate that error** — do not map it to a losing row. `ModelRunner::start` returns `Ok(tx)` even when the model fails to load and then answers every job with `Err("model load failed: {e}")`, so this arm is exactly where a broken checkpoint announces itself. Mapping it to `stopped_on_eos: false` would burn three model runs to conclude "no template matched" about a model that never loaded, discarding a precise diagnosis the runner already handed you.

Then print `format_probe_report(&rows)`.

- [ ] **Step 5: Selection, and the four ways it must refuse**

This is the step the old plan omitted entirely, and it is the only part that writes to disk. The sidecar it produces outranks **every** other tier on the next server start, so each refusal below is load-bearing:

```rust
// Refusal -1, FIRST. Without this, the most common operator error — a typo in
// the path — gets the one diagnosis guaranteed to be wrong. `special_tokens`
// swallows all three of its file reads, so a nonexistent directory returns
// `bos: "", eos: ""` and falls into Refusal 0, telling the operator to "add a
// tokenizer_config.json" to a directory that does not exist. `ModelRunner::start`
// has no existence check either (`src/api/mod.rs` guards with
// `model_path.exists()` before calling it; nothing else does).
if !model_dir.exists() {
    anyhow::bail!("{} does not exist", model_dir.display());
}

// Refusal 0: an empty EOS makes the measurement meaningless, so do not spend
// three model runs discovering it.
let tokens = special_tokens(&model_dir);
if tokens.eos.is_empty() {
    anyhow::bail!(
        "{} declares no EOS token, so this probe cannot measure anything: ZEPHYR \
         and LLAMA2 both interpolate `eos_token` and would render with no \
         end-of-turn marker at all, while CHATML — which references neither — \
         would be unaffected. The comparison would be rigged, not close. Add a \
         tokenizer_config.json with eos_token, or a config.json with \
         eos_token_id WHOSE ID APPEARS IN tokenizer.json's added_tokens. \
         Nothing was written.",
        model_dir.display()
    );
}

// ... generation happens here, producing `rows: Vec<ProbeRow>` ...

let fired: Vec<&ProbeRow> = rows.iter().filter(|r| r.stopped_on_eos).collect();
let winner: &ProbeRow = match fired.as_slice() {
    [only] => only,
    [] => anyhow::bail!(
        "no candidate stopped on EOS, so none of these three templates suits \
         this checkpoint. (A model that failed to LOAD cannot reach this \
         message — that aborts the probe at the first candidate with the \
         runner's own error.) Nothing was written."
    ),
    many => anyhow::bail!(
        "{} candidates stopped on EOS ({}). The probe cannot choose between \
         them and will not guess. Nothing was written.",
        many.len(),
        many.iter().map(|r| r.candidate).collect::<Vec<_>>().join(", ")
    ),
};

// Refusal 3. `Option<String>`, and the Sidecar field is `String`.
let Some(fp) = fingerprint(&model_dir) else {
    anyhow::bail!(
        "cannot fingerprint {} — a directory checkpoint needs a readable \
         config.json; a .gguf file needs to be readable itself — so a sidecar \
         written here could not later be checked against the checkpoint it \
         describes. Nothing was written.",
        model_dir.display()
    );
};
```

1. **Empty EOS → refuse, before generating.** 2 of the 3 candidates interpolate `eos_token`; `CHATML` references neither. An empty token therefore does not degrade the comparison evenly — it removes the end-of-turn marker from two candidates and leaves the third intact, which biases the result rather than merely weakening it.
2. **Zero winners → refuse.** Note what this message does **not** claim. An earlier draft said a load failure and a template mismatch "look identical here" — that was false: `drain_with_error` sends `Err("model load failed: {e}")` and the runner prints to stderr, so the probe holds the discriminating value. Step 4 aborts on that `Err`, so by the time this arm is reached the model demonstrably loaded and generated. Silently picking `candidates()[0]` would still be wrong, but for the ordinary reason.
3. **Two or more winners → refuse.** Do not tie-break.
4. **`fingerprint` returning `None` → refuse.** **Do not reach for `.unwrap_or_default()`** — that is precisely the bug Task 2's review found, where every unidentifiable checkpoint hashed to the same constant and a foreign sidecar was served at `Resolution::Sidecar`, the highest-authority tier, carrying evidence claiming it was measured on this model.
5. **`--yes` does not bypass 1–4.** It skips only the interactive y/N.

Then build the sidecar with **all five fields**. `winner.source` is carried on the row — do **not** re-look-it-up from `candidates()` by name:

```rust
let sc = Sidecar {
    template: winner.source.to_string(),
    resolved_by: Resolution::Probe,
    evidence: format!(
        "probe: {} stopped on EOS in {} tokens; {} did not",
        winner.candidate,
        winner.tokens_generated,
        rows.iter().filter(|r| !r.stopped_on_eos)
            .map(|r| r.candidate).collect::<Vec<_>>().join(", ")
    ),
    resolved_at: chrono::Utc::now().to_rfc3339(),
    model_fingerprint: fp,
};
write_sidecar(&model_dir, &sc)?;   // propagate: a failed write must not report success
```

`evidence` is one prose line, matching the spec's shape and the existing fixtures in `tests/chat_template_render.rs` — **not** the whole table with its header stuffed into a JSON string. `chrono` is already a direct dependency and RFC-3339 is the convention every existing sidecar fixture uses.

- [ ] **Step 6: Fix the operator hint that already ships**

The stale hint is in **`resolve`** (`src/api/chat_template.rs:491`), in the *no tier matched at all* fallthrough — **not** in `special_tokens`, and the binding in scope is **`model_dir`**, not `model_path`. Locate it rather than trusting this line number:

**Search for `chat-template probe`, not `lightbulb-cli chat-template`** — the narrower pattern misses two references in the spec, which plan line 11 makes required reading, including a File-Structure row describing the very design this task refutes:

```bash
grep -rn "chat-template probe" src/ docs/
```

Five hits: `chat_template.rs:491`, two in this plan, and two in `docs/superpowers/specs/2026-08-06-chat-template-resolution-design.md`. Fix all of them.

The existing `format!` already consumes its single `{}` with `model_dir.display()`, so the new path needs a **second** argument:

```rust
    tracing::warn!(
        "no chat template resolved for {}; falling back to the legacy role: content join. \
         Run `lightbulb-probe {}` to determine one.",
        model_dir.display(),
        model_dir.display()
    );
```

That grep also hits **this plan's own Task 2 sample**, which quotes the old string. Update it too, and add the plan to Step 7's `git add` list — otherwise the plan keeps advertising a command that does not exist.

Do **not** add a probe hint to `special_tokens`'s tier-3 warning. That warning deliberately prints `meta.display()` rather than `model_path.display()`, with an in-code comment explaining that the two differ for a `.gguf` checkpoint; a hint added there would contradict it.

- [ ] **Step 7: Run and commit**

Note: no `| tail`. Piping cargo into `tail` reports **`tail`'s** exit status, which this plan's Global Constraints call out as having already produced a false green here.

```bash
cargo test -j 4 --test chat_template_render 2>&1 | grep -E "^running|^test result|^error"; echo "EXIT=${PIPESTATUS[0]}"
# Expect `running 36 tests` / `35 passed; 0 failed; 1 ignored` — libtest counts
# the ignored measurement harness in `running N`.
cargo build -j 4 --bin lightbulb-probe 2>&1 | grep -E "^error"; echo "EXIT=${PIPESTATUS[0]}"

# rustfmt on chat_template.rs follows its `pub mod registry;` and may reformat
# registry.rs too — the same mechanism the Global Constraints flag for mod.rs.
rustfmt --edition 2024 src/bin/lightbulb-probe.rs src/api/chat_template.rs tests/chat_template_render.rs
git status --short src/api/chat_template/   # revert registry.rs if you did not touch it

git add Cargo.toml src/bin/lightbulb-probe.rs src/api/chat_template.rs \
        tests/chat_template_render.rs docs/superpowers/plans/2026-08-08-chat-template-resolution.md
git commit -m "feat(cli): Add the chat-template probe binary"
```

`git add` names explicit paths — `git add -A` would sweep the untracked `supertool` and anything another agent has in flight.

**Deliberately not a verification step:** `cargo build --bin lightbulb-cli`. Task 6 does not touch that target's sources, and `[dependencies]` are package-scoped, so adding a second `[[bin]]` cannot change it — the command passes identically whether this task was done right, done wrong, or not at all. The only failure it could catch is a malformed manifest, which building `lightbulb-probe` already catches. A check that cannot fail is the defect this plan's Global Constraints name.

**Manual smoke test** (needs the checkpoint; the probe has no automated end-to-end test because it needs a model):

```bash
cargo run --release -j 4 --bin lightbulb-probe -- "$TINYLLAMA_DIR"
```

Expect a three-row table. TinyLlama resolves to ZEPHYR at tier 3, so `zephyr` firing EOS is the result that agrees with what the server already does — and if the table shows zero or two winners, the refusal paths above are what should happen.

**Answer `n`, or delete `$TINYLLAMA_DIR/lightbulb-chat-template.json` afterwards.** A `y` writes a sidecar into the shared HF cache snapshot, and `resolve` reads tier 0 first — so that checkpoint would then resolve at `Resolution::Probe` for every later run on this machine, falsifying this plan's own Global Constraint that it resolves at tier 3. No test asserts the tier against the real checkpoint today, so nothing breaks immediately; that is what makes it easy to leave behind.

**This smoke test does NOT exercise refusal 1.** TinyLlama's `config.json` ids 1/2 resolve to `<s>`/`</s>` through `tokenizer.json`, so its EOS is non-empty and the empty-EOS path is unreachable with this checkpoint. Cover that one with a fixture directory, not the smoke test.

### Known limits of this task, stated rather than discovered

- **`.gguf` under `--features fuel-engine` reports the wrong cause.** That build's `ModelRunner::start` refuses GGUF outright and routes to `drain_with_error`, so Step 4's abort surfaces the runner's message — which is correct behaviour, but an operator who only reads the probe's own output would blame the templates for a backend limitation. The default candlelight build handles `.gguf` normally. Not fixed here; `fingerprint`, `metadata_dir` and `sidecar_path` all branch on `is_file` correctly, so the rest of the GGUF path is sound.
- **A nonexistent `model_dir` is not pre-checked.** `ModelRunner::start` does no existence check (unlike `src/api/mod.rs`, which guards with `model_path.exists()`); the failure arrives from the loader through `drain_with_error` and Step 4 aborts with it. Acceptable, because the message names the path — but it is a runner error, not a probe error, and reads that way.
- **Four sidecar fixtures in `tests/chat_template_render.rs` carry `evidence` in the old `EOS 8/8 with zephyr, 1/8 with llama2` N-trials shape**, which this design can no longer produce. They are fixtures for *reading* sidecars, so nothing breaks, and Task 6 deliberately does not rewrite them — but do not copy that shape for the `evidence` this task writes.

---

## Final verification

- [ ] Full suite at capped parallelism:

```bash
cargo test --tests -j 4 2>&1 | grep -E "test result|error" | tail -30
```

- [ ] Feature-gated suites, **checking `running N tests` on each**:

```bash
pwsh -NoProfile -File C:/Projects/fuel/scripts/gpu-run.ps1 -Project lightbulb -- \
  cargo test --release --features fuel-engine -j 4 --test api_result_metadata --test chat_template_e2e -- \
  --ignored --test-threads=1
```

- [ ] `grep -rn 'format!("{}: {}", m.role, m.content)\|format!("{}: {}", msg.role, msg.content)' src/` returns **only** the one occurrence inside `legacy_join`.

---

## Deliberately out of scope

Carried from the spec, restated so nobody adds them opportunistically: multi-turn conversation state, tool/function-call templates, vision content, and auto-switching templates from the monitor (explicitly rejected in §3).

## Known gaps this plan does not close

- **`/v1/completions` remains template-free.** Correct per §6 — it is OpenAI's raw-text endpoint.
- **`minijinja` may silently mis-render a template Jinja2 renders differently.** Rendering *failure* falls through and logs; silent divergence is the residual risk, guarded only by Task 1's exact-match test on the shape we ship.
- **The registry will rot.** Tier order limits the blast radius to models that ship no template of their own.
