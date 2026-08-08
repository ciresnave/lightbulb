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
//! This module exists to replace that join. Both `openai/chat.rs` sites now
//! render through `ResolvedTemplate::render` — measured on TinyLlama, the same
//! request that previously ran its whole 64-token budget inventing a Q&A
//! transcript now answers `"The capital of France is Paris.</s>"` in 8 tokens
//! and reports `finish_reason: "stop"`. `contracts/executor.rs` is still on
//! the join; it cannot be fixed by rendering upstream because it MUTATES the
//! message list between retry attempts, so its callback signature has to
//! change. That is Task 4.
//!
//! A template is always paired with the special tokens it renders with — see
//! `SpecialTokens` and `ResolvedTemplate`. Handing `render` a literal
//! `"<s>"`/`"</s>"` is correct for the Llama-2 family and silently wrong
//! everywhere else, and it fails in the one way this module cannot detect: the
//! render succeeds, so nothing falls through and nothing is logged.

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

fn digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    // Truncated to 8 bytes. This identifies a checkpoint; it is not defending
    // against an adversary who gets to pick the file contents.
    hex::encode(&Sha256::digest(bytes)[..8])
}

/// Fingerprint of the checkpoint at `model_path`, or `None` when one cannot be
/// computed.
///
/// **`None` is the point of the return type.** This previously read
/// `config.json` through `unwrap_or_default()`, so every path without a
/// readable one hashed to the same constant — two unrelated empty directories
/// and every `.gguf` file path all produced `bd60acb658c79e45`. A sidecar
/// carrying that constant then matched any such checkpoint and was served at
/// `Resolution::Sidecar`, the highest-authority tier, with `evidence` claiming
/// it had been measured on that model. `ModelRunner::start` accepts "either a
/// directory … or a `.gguf` file", so that path was reachable. A fingerprint
/// that cannot be computed must never compare equal to a stored one, which is
/// exactly what no value at all guarantees.
///
/// SHA-256 rather than `DefaultHasher`: the latter is explicitly unstable
/// across Rust releases, so a toolchain upgrade silently invalidates every
/// sidecar on disk. `sha2` is already a direct dependency, so that was a cost
/// paid for nothing.
pub fn fingerprint(model_path: &Path) -> Option<String> {
    // A single-file checkpoint (`.gguf`) has no `config.json` to hash, so the
    // file itself is the checkpoint: size + mtime + file name.
    //
    // Deliberately NOT the contents. A quantized 7B is several gigabytes and
    // this runs on the first resolve for the model. The three fields together
    // separate the cases that actually occur — a different model at the same
    // path, a different quant of the same model, a re-download — because each
    // changes the length or the mtime. They will not notice an in-place edit
    // that preserves both, which no checkpoint swap does.
    //
    // File NAME, not full path: the sidecar lives beside the file and travels
    // with it, so keying on the full path would invalidate every sidecar when
    // a checkpoint tree is moved, and buy no safety in return.
    if model_path.is_file() {
        let md = std::fs::metadata(model_path).ok()?;
        let mtime = md
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?;
        let name = model_path.file_name()?.to_string_lossy().into_owned();
        return Some(digest(
            format!("gguf:{name}:{}:{}", md.len(), mtime.as_nanos()).as_bytes(),
        ));
    }
    Some(digest(&std::fs::read(model_path.join("config.json")).ok()?))
}

/// Where the sidecar for `model_path` lives.
///
/// A directory checkpoint holds it inside. A single-file `.gguf` checkpoint
/// holds it beside the file, because a file has no inside — and both are paths
/// `ModelRunner::start` accepts, so an operator running the probe against
/// either has to find the answer where the model is.
pub fn sidecar_path(model_path: &Path) -> std::path::PathBuf {
    if model_path.is_file() {
        let mut name = model_path.as_os_str().to_os_string();
        name.push(".");
        name.push(SIDECAR_NAME);
        std::path::PathBuf::from(name)
    } else {
        model_path.join(SIDECAR_NAME)
    }
}

pub fn write_sidecar(model_path: &Path, sc: &Sidecar) -> anyhow::Result<()> {
    let json = serde_json::to_string_pretty(sc)?;
    std::fs::write(sidecar_path(model_path), json)?;
    Ok(())
}

/// Read the sidecar, rejecting one that cannot be shown to belong to this
/// checkpoint.
///
/// The fingerprint comparison is the whole point of the function. A model
/// directory that gets re-pointed at another checkpoint — a common enough thing
/// to do with a symlink or a re-download — would otherwise inherit the previous
/// model's template at the highest-authority tier, with a `resolved_by: Probe`
/// on it claiming it was measured.
///
/// Every rejection is logged, because the observable consequence of one is that
/// the server quietly falls to a registry guess. The single silent path is a
/// sidecar that is simply not there, which is the ordinary case for every
/// checkpoint that has never been probed.
pub fn read_sidecar(model_path: &Path) -> Option<Sidecar> {
    let path = sidecar_path(model_path);
    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            tracing::warn!("ignoring {}: cannot be read: {e}", path.display());
            return None;
        }
    };
    let sc: Sidecar = match serde_json::from_str(&raw) {
        Ok(sc) => sc,
        Err(e) => {
            tracing::warn!(
                "ignoring {}: not a valid {SIDECAR_NAME} document: {e}",
                path.display()
            );
            return None;
        }
    };
    let Some(actual) = fingerprint(model_path) else {
        tracing::warn!(
            "ignoring {}: no fingerprint can be computed for {} (a directory checkpoint needs a \
             readable config.json), so the sidecar cannot be shown to belong to this checkpoint",
            path.display(),
            model_path.display()
        );
        return None;
    };
    if sc.model_fingerprint != actual {
        tracing::warn!(
            "ignoring {}: fingerprint {} does not match checkpoint {}",
            path.display(),
            sc.model_fingerprint,
            actual
        );
        return None;
    }
    Some(sc)
}

/// Extract `chat_template` from a parsed `tokenizer_config.json`.
///
/// Two shapes are real. Most checkpoints store a string. Some store a **list**
/// of `{"name": …, "template": …}` entries — Hermes-3 ships `default` and
/// `tool_use` — which `transformers` supports, resolving to the entry named
/// `default` when the caller names no template.
///
/// Handling only the string form sent every such checkpoint to a registry
/// guess, reported as a successful tier-3 match, with nothing logged: the
/// checkpoint's own authoritative template was on disk and was skipped. The
/// list form costs ten lines, so it is handled rather than merely warned about.
/// Everything this still cannot use is logged, because the alternative is an
/// operator seeing a family guess for a model that ships a template.
fn chat_template_field(v: &serde_json::Value) -> Option<String> {
    let ct = v.get("chat_template")?;
    if let Some(s) = ct.as_str() {
        return Some(s.to_string());
    }
    if let Some(entries) = ct.as_array() {
        for e in entries {
            if e.get("name").and_then(serde_json::Value::as_str) != Some("default") {
                continue;
            }
            if let Some(t) = e.get("template").and_then(serde_json::Value::as_str) {
                return Some(t.to_string());
            }
        }
        let names: Vec<&str> = entries
            .iter()
            .filter_map(|e| e.get("name").and_then(serde_json::Value::as_str))
            .collect();
        tracing::warn!(
            "tokenizer_config.json lists chat_template entries {names:?}, none of which is a \
             \"default\" with a string template; falling through to the next tier"
        );
        return None;
    }
    tracing::warn!(
        "tokenizer_config.json has a chat_template that is neither a string nor a list of named \
         templates; falling through to the next tier"
    );
    None
}

/// Resolve a template, cheapest tier first. Stops at the first hit.
///
/// Tier 4 (the probe) is deliberately absent: it costs generations and its
/// conclusion gets persisted, so it is an operator action. See spec §4.
pub fn resolve(model_dir: &Path) -> ChatTemplate {
    // Tier 0 — sidecar.
    if let Some(sc) = read_sidecar(model_dir) {
        tracing::info!("chat template: sidecar, recorded as {:?}", sc.resolved_by);
        return ChatTemplate {
            source: sc.template,
            // The sidecar's OWN `resolved_by`, not `Resolution::Sidecar`.
            // Spec §1 gives tier 0 the confidence "whatever produced it —
            // recorded"; flattening every sidecar to `Sidecar` erases the one
            // distinction the file exists to carry, between a probe's
            // measurement and a hand-written guess. `Sidecar` is still the
            // right answer when that is literally what the file records.
            resolved_by: sc.resolved_by,
        };
    }

    // Tier 1 — the authoritative source, when the checkpoint ships one.
    if let Ok(raw) = std::fs::read_to_string(model_dir.join("tokenizer_config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(t) = chat_template_field(&v) {
                tracing::info!("chat template: tokenizer_config.json");
                return ChatTemplate {
                    source: t,
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

// ─── Special tokens ─────────────────────────────────────────────────────────
//
// `render` takes `bos_token` and `eos_token` because real templates reference
// them as bare variables. They must come FROM THE CHECKPOINT.
//
// Passing a hardcoded `"<s>"` / `"</s>"` is correct for exactly the Llama-2
// family and silently wrong everywhere else: Llama-3 uses `<|begin_of_text|>`
// and `<|eot_id|>`, Qwen/ChatML uses `<|endoftext|>` and `<|im_end|>`. A
// template rendered with another model's special tokens still renders — so it
// does not fall through to the next tier, nothing is logged, and the model is
// prompted with text it has never seen in training. That is the same
// silent-wrong-output failure this whole module exists to remove, and it is
// worse than the one it replaced because the prompt now LOOKS right.

/// The special tokens a chat template is rendered with, read from the
/// checkpoint.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialTokens {
    pub bos: String,
    pub eos: String,
}

/// Read one `*_token` field out of a parsed `tokenizer_config.json`.
///
/// Two shapes are real and both appear on the Hub. Most checkpoints store a
/// bare string (`"eos_token": "<|eot_id|>"`). Checkpoints saved by an older
/// `transformers` store the serialised `AddedToken` object
/// (`"eos_token": {"content": "</s>", "lstrip": false, "normalized": false, …}`)
/// — TinyLlama's own Hub `tokenizer_config.json` is of that second kind.
/// Handling only the string form would send half the Hub to the id fallback
/// below with nothing logged.
fn token_field(v: &serde_json::Value, key: &str) -> Option<String> {
    let t = v.get(key)?;
    if let Some(s) = t.as_str() {
        return Some(s.to_string());
    }
    t.get("content")?.as_str().map(str::to_string)
}

/// `config.json`'s `bos_token_id` / `eos_token_id`.
///
/// The value is usually an integer but may be a **list**: Llama-3.1-Instruct
/// ships `"eos_token_id": [128001, 128008, 128009]`, the set of ids that end
/// generation. Nothing in that list says which one a template should append
/// after a message, so a list is a guess and is logged as one. The first entry
/// is taken because it is deterministic and no ordering there is meaningful —
/// and this path is only reached by a checkpoint that ships **no**
/// `tokenizer_config.json`, which no multi-EOS model does, since such a model
/// needs one for its own template.
fn token_id(cfg: &serde_json::Value, key: &str) -> Option<u64> {
    let v = cfg.get(key)?;
    if let Some(n) = v.as_u64() {
        return Some(n);
    }
    let arr = v.as_array()?;
    if arr.len() > 1 {
        tracing::warn!(
            "config.json {key} is a list {v}; using the first entry. Ship a \
             tokenizer_config.json with an explicit {} to remove the guess.",
            key.trim_end_matches("_id")
        );
    }
    arr.first()?.as_u64()
}

/// Look an id up in `tokenizer.json`'s `added_tokens`.
///
/// `added_tokens` rather than the full `model.vocab`: BOS/EOS are always added
/// tokens, the array is short, and reading the whole vocab of a 150k-token
/// tokenizer to find two entries is work for nothing.
fn added_token_text(tokenizer: &serde_json::Value, id: u64) -> Option<String> {
    tokenizer
        .get("added_tokens")?
        .as_array()?
        .iter()
        .find(|t| t.get("id").and_then(serde_json::Value::as_u64) == Some(id))
        .and_then(|t| t.get("content").and_then(serde_json::Value::as_str))
        .map(str::to_string)
}

/// Resolve the checkpoint's BOS and EOS token text.
///
/// Sources, in order:
///
/// 1. `tokenizer_config.json`'s `bos_token` / `eos_token` — authoritative, and
///    the same field `transformers` reads.
/// 2. `config.json`'s `bos_token_id` / `eos_token_id`, looked up in
///    `tokenizer.json`'s `added_tokens`. This is what serves the checkpoint
///    this project tests against: the TinyLlama-1.1B-Chat snapshot has no
///    `tokenizer_config.json` at all (only `config.json`, `model.safetensors`
///    and `tokenizer.json`), and its ids 1 and 2 map to `<s>` and `</s>` there.
/// 3. **Empty string, with a warning naming the checkpoint.**
///
/// Tier 3 is empty rather than `"<s>"` / `"</s>"` deliberately. A default of
/// `</s>` is a *specific model family's* token, so it would put a foreign
/// special token into the prompt of every checkpoint that declares nothing —
/// unrecoverably, because the render succeeds and no tier falls through. An
/// empty string omits the marker instead: the prompt is then merely incomplete
/// rather than actively lying, single-turn prompting (where the marker sits
/// after the user's message and before the generation prompt) is barely
/// affected, and the warning names the directory so the gap is fixable. The
/// rule is "never invent a token", and empty is the only value that obeys it.
pub fn special_tokens(model_path: &Path) -> SpecialTokens {
    let mut out = SpecialTokens::default();

    // Tier 1 — the checkpoint's own declaration.
    if let Ok(raw) = std::fs::read_to_string(model_path.join("tokenizer_config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            out.bos = token_field(&v, "bos_token").unwrap_or_default();
            out.eos = token_field(&v, "eos_token").unwrap_or_default();
        }
    }

    // Tier 2 — ids from config.json, resolved through tokenizer.json. Applied
    // per token, not per file: a checkpoint that declares one and not the other
    // gets the one it declares plus a resolved fallback for the other.
    if out.bos.is_empty() || out.eos.is_empty() {
        let cfg = std::fs::read_to_string(model_path.join("config.json"))
            .ok()
            .and_then(|r| serde_json::from_str::<serde_json::Value>(&r).ok());
        let tok = std::fs::read_to_string(model_path.join("tokenizer.json"))
            .ok()
            .and_then(|r| serde_json::from_str::<serde_json::Value>(&r).ok());
        if let (Some(cfg), Some(tok)) = (cfg, tok) {
            for (slot, key) in [
                (&mut out.bos, "bos_token_id"),
                (&mut out.eos, "eos_token_id"),
            ] {
                if slot.is_empty() {
                    if let Some(text) =
                        token_id(&cfg, key).and_then(|id| added_token_text(&tok, id))
                    {
                        *slot = text;
                    }
                }
            }
        }
    }

    // Tier 3 — nothing declared. Say so; do not invent one.
    if out.bos.is_empty() || out.eos.is_empty() {
        tracing::warn!(
            "{} declares no {}; rendering its chat template with an empty string there. \
             A chat template that references the missing token will omit it. Add a \
             tokenizer_config.json with bos_token/eos_token, or a config.json with \
             bos_token_id/eos_token_id whose ids appear in tokenizer.json's added_tokens.",
            model_path.display(),
            match (out.bos.is_empty(), out.eos.is_empty()) {
                (true, true) => "BOS or EOS token",
                (true, false) => "BOS token",
                _ => "EOS token",
            }
        );
    }

    out
}

/// A resolved template together with the special tokens it must be rendered
/// with.
///
/// The pair travels together because separating them is how a template ends up
/// rendered with another model's tokens: both are read from the same checkpoint
/// at the same moment, and a caller holding only a `ChatTemplate` has to supply
/// `bos`/`eos` from somewhere, which in practice means a literal.
#[derive(Debug, Clone)]
pub struct ResolvedTemplate {
    pub template: ChatTemplate,
    pub tokens: SpecialTokens,
}

impl ResolvedTemplate {
    /// Render `messages` with this checkpoint's own special tokens.
    pub fn render(&self, messages: &[RawMessage]) -> anyhow::Result<String> {
        self.template
            .render(messages, &self.tokens.bos, &self.tokens.eos)
    }

    /// Which tier produced the template.
    pub fn resolved_by(&self) -> Resolution {
        self.template.resolved_by
    }
}

/// Resolve both the template and the special tokens for `model_path`.
///
/// Called once at startup, where `model_path` is known. Both halves read files,
/// and neither can change while the process runs.
pub fn resolve_for_model(model_path: &Path) -> ResolvedTemplate {
    ResolvedTemplate {
        template: resolve(model_path),
        tokens: special_tokens(model_path),
    }
}
