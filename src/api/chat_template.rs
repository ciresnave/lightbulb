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
