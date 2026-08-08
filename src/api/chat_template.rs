//! Resolve and render a model's chat template.
//!
//! `/v1/chat/completions` builds its prompt as
//! `messages.map(|m| format!("{}: {}", m.role, m.content)).join("\n")`
//! (`openai/chat.rs:186` non-streaming, `:409` streaming, and
//! `contracts/executor.rs:183` per retry attempt), which is not any model's
//! template. A chat model prompted that way behaves like a base model: it
//! continues text rather than answering, and does not reliably emit EOS.
//! Measured on TinyLlama-1.1B-Chat, EOS fired in 1 of 6 trials.
//!
//! This module exists to replace that join. Nothing is wired to it yet — the
//! three sites above are still on the join until Tasks 3 and 4 route them
//! through `ChatTemplate::render`.

use std::path::Path;

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
    ///
    /// A fresh `Environment` is built and the source re-parsed on **every**
    /// call. That is deliberate, and it was measured rather than assumed.
    ///
    /// Measured 2026-08-08, `--release`, N=20000 after a 200-iteration warmup,
    /// on Llama-3.1-8B-Instruct's real template (4723 bytes — the largest of
    /// the nine Hub templates checked), two messages:
    ///
    ///   this code (fresh `Environment` + parse per call)   235.2 us/call
    ///   `Environment` + parsed template reused              16.8 us/call
    ///   difference                                         218.4 us/call
    ///
    /// 218 us is under a quarter of ONE forward pass, and `render` is called
    /// once per request (or per contract retry), never per token — a request
    /// that emits a hundred tokens spends three orders of magnitude more time
    /// in the model than the whole prompt build costs. Caching would buy that
    /// 218 us in exchange for either an owned-template registry keyed by
    /// source, or a self-referential struct to escape `Environment<'source>`
    /// borrowing `&self.source`.
    ///
    /// Not worth it. Do not re-open this without a measurement that shows the
    /// call rate changed — e.g. `render` moving onto a per-token path, which
    /// would be a design error for other reasons.
    pub fn render(
        &self,
        messages: &[RawMessage],
        bos_token: &str,
        eos_token: &str,
    ) -> anyhow::Result<String> {
        let mut env = minijinja::Environment::new();

        // Match the Jinja2 dialect these templates were authored against.
        // `transformers` compiles every chat template with
        // `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`
        // (transformers/utils/chat_template_utils.py), and minijinja's defaults
        // are plain Jinja2's — both `false`. Leaving them false silently shifts
        // whitespace in any checkpoint-supplied template, which is the one
        // failure this module cannot detect: it renders, so nothing falls
        // through to the next tier and nothing is logged.
        //
        // `keep_trailing_newline` is deliberately NOT set: transformers leaves
        // it at Jinja2's default `false` too. A template author who wants a
        // trailing newline must put it inside a variable tag — `{{ '<|x|>\n' }}`
        // — as TinyLlama's real template does, because a newline in trailing
        // source position is stripped by both engines.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);

        // Jinja2's values are Python objects, so a template author gets
        // `str.startswith`, `str.strip('\n')`, `str.split(...)` and
        // `dict.items()` without asking. minijinja's are not, and it reports
        // `unknown method: string has no method named startswith`.
        //
        // This is not hypothetical. Of the nine real Hub `chat_template`
        // values checked, two fail without this callback:
        //   Qwen3-8B                   `.startswith`/`.endswith` at chat:20,
        //                              then `.split`/`.strip('\n')`/`.lstrip`/
        //                              `.rstrip` in its `<think>` handling
        //   Mistral-7B-Instruct-v0.3   `tool.items()`, in the branch a
        //                              no-tools message list never reaches
        //
        // The Mistral case is why this is registered rather than patched per
        // template: the gap is latent until someone passes `tools`, at which
        // point it becomes a render failure on a template that has "always
        // worked".
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

        // Templates call this to reject unsupported message orders. Without it
        // minijinja reports "unknown function", which reads like our bug.
        env.add_function("raise_exception", |msg: String| {
            Err::<(), _>(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        });

        // Llama-3.x templates open with
        // `{%- set date_string = strftime_now("%d %b %Y") %}` and interpolate it
        // into the system message. This is a *global*, not a minijinja feature —
        // no feature list makes it appear, so an unregistered `strftime_now`
        // turns every Llama-3.x `tokenizer_config.json` into a tier-1 parse
        // failure that falls through to a registry guess.
        //
        // `Local`, not `Utc`: transformers calls
        // `datetime.now().strftime(fmt)`, which is local time.
        env.add_function("strftime_now", |fmt: String| {
            // chrono's formatter panics on an invalid specifier at Display
            // time, so the format string is validated before it is used.
            let items: Vec<_> = chrono::format::StrftimeItems::new(&fmt).collect();
            if items
                .iter()
                .any(|i| matches!(i, chrono::format::Item::Error))
            {
                return Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("strftime_now: unsupported format string {fmt:?}"),
                ));
            }
            Ok(chrono::Local::now()
                .format_with_items(items.into_iter())
                .to_string())
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

pub const SIDECAR_NAME: &str = "lightbulb-chat-template.json";

/// A template pinned to a checkpoint on disk, with the reasoning that produced
/// it.
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
///
/// `DefaultHasher` is explicitly not a stable hash across Rust releases, so a
/// toolchain upgrade can invalidate every sidecar on disk. That is acceptable
/// here and nowhere near acceptable for a cache key: the consequence is
/// re-running a probe, and the failure direction is "ignored a valid sidecar",
/// never "accepted one from a different checkpoint".
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
///
/// The fingerprint comparison is the whole point of the function. A model
/// directory that gets re-pointed at another checkpoint — a common enough thing
/// to do with a symlink or a re-download — would otherwise inherit the previous
/// model's template at the highest-authority tier, with a `resolved_by: Probe`
/// on it claiming it was measured.
pub fn read_sidecar(model_dir: &Path) -> Option<Sidecar> {
    let raw = std::fs::read_to_string(model_dir.join(SIDECAR_NAME)).ok()?;
    let sc: Sidecar = serde_json::from_str(&raw).ok()?;
    let actual = fingerprint(model_dir);
    if sc.model_fingerprint != actual {
        tracing::warn!(
            "ignoring {SIDECAR_NAME}: fingerprint {} does not match checkpoint {}",
            sc.model_fingerprint,
            actual
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
        return ChatTemplate {
            source: sc.template,
            resolved_by: Resolution::Sidecar,
        };
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
        return ChatTemplate {
            source: t,
            resolved_by: Resolution::VocabSignature,
        };
    }

    // Tier 3 — family registry. Heuristic; §3's monitor exists for when it is
    // wrong.
    if let Some(t) = registry::from_family(model_dir) {
        tracing::info!("chat template: family registry (heuristic)");
        return ChatTemplate {
            source: t,
            resolved_by: Resolution::Registry,
        };
    }

    tracing::warn!(
        "no chat template resolved for {}; falling back to the legacy role: content join. \
         Run `lightbulb-cli chat-template probe` to determine one.",
        model_dir.display()
    );
    ChatTemplate {
        source: String::new(),
        resolved_by: Resolution::None,
    }
}
