//! Resolve and render a model's chat template.
//!
//! `/v1/chat/completions` used to build its prompt as
//! `messages.map(|m| format!("{}: {}", m.role, m.content)).join("\n")` — once
//! in each of `openai/chat.rs`'s two handlers and once more in
//! `contracts/executor.rs` per retry attempt — which is not any model's
//! template. A chat model prompted that way behaves like a base model: it
//! continues text rather than answering, and does not reliably emit EOS.
//! Measured on TinyLlama-1.1B-Chat, EOS fired in 1 of 6 trials.
//!
//! This module exists to replace that join. Both `openai/chat.rs` sites now
//! render through `ResolvedTemplate::render` (see `openai/chat.rs`'s
//! `build_prompt`, called from `create_chat_completion` and
//! `create_chat_stream`) — measured on TinyLlama, the same request that
//! previously ran its whole 64-token budget inventing a Q&A transcript now
//! answers `"The capital of France is Paris.</s>"` in 8 tokens and reports
//! `finish_reason: "stop"`. `contracts/executor.rs` is still on the join; it
//! cannot be fixed by rendering upstream because it MUTATES the message list
//! between retry attempts, so its callback signature has to change. That is
//! Task 4.
//!
//! Deliberately no line numbers in this doc: the three earlier revisions of it
//! all cited call sites by line, and all three were stale within two commits.
//!
//! A template is always paired with the special tokens it renders with — see
//! `SpecialTokens` and `ResolvedTemplate`. Handing `render` a literal
//! `"<s>"`/`"</s>"` is correct for the Llama-2 family and silently wrong
//! everywhere else, and it fails in the one way this module cannot detect: the
//! render succeeds, so nothing falls through and nothing is logged.
//!
//! # A rendered prompt must be tokenized with `add_special_tokens = false`
//!
//! Templates interpolate `bos_token` into the prompt **text** (Llama-3,
//! Llama-2, Mistral and Gemma all open with `{{ bos_token }}`). Tokenizers
//! whose `post_processor` is `TemplateProcessing` prepend BOS again when asked
//! for special tokens, so a templated prompt tokenized with
//! `add_special_tokens = true` reaches the model as `128000, 128000, 128006, …`
//! — a pair it never saw in training. HuggingFace tokenizes
//! `apply_chat_template` output with `add_special_tokens=False` for exactly
//! this reason.
//!
//! The signal therefore travels with the prompt rather than being decided at
//! the tokenizer: `openai/chat.rs`'s `BuiltPrompt` carries it, `InferenceJob`
//! and `RequestContext` forward it, and each backend passes it to its single
//! `encode` call. A legacy-join prompt and `/v1/completions`'s raw text carry
//! no special tokens of their own and still need `true`.

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

        // `bos_token` goes into the prompt TEXT. Anything that tokenizes the
        // result must therefore pass `add_special_tokens = false`, or a
        // tokenizer with a `TemplateProcessing` post_processor prepends BOS a
        // second time. See the module docs; the flag is carried by
        // `openai/chat.rs`'s `BuiltPrompt`.
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

/// Where a checkpoint's companion JSON (`tokenizer_config.json`,
/// `config.json`, `tokenizer.json`) lives.
///
/// A directory checkpoint holds them inside. A single-file `.gguf` checkpoint
/// holds them **beside** the file, because a file has no inside — the same rule
/// `sidecar_path` applies, and for the same reason: `ModelRunner::start`
/// accepts either path.
///
/// Without this, `special_tokens` read `foo.gguf/tokenizer_config.json` and
/// `foo.gguf/config.json`, which cannot exist, so every GGUF checkpoint
/// resolved to `bos: "", eos: ""` on every request and the warning told the
/// operator to add a file the code would never have looked for. `fingerprint`
/// and `sidecar_path` already branched on `is_file`; this is the third.
///
/// Falls back to `model_path` itself when the file has no parent — a bare
/// relative `"foo.gguf"` yields `Some("")`, which is the current directory and
/// is what we want, so only a genuinely parentless path (a root) lands here.
fn metadata_dir(model_path: &Path) -> &Path {
    if model_path.is_file() {
        model_path.parent().unwrap_or(model_path)
    } else {
        model_path
    }
}

/// Where the sidecar for `model_path` lives.
///
/// A directory checkpoint holds it inside. A single-file `.gguf` checkpoint
/// holds it beside the file, because a file has no inside — and both are paths
/// `ModelRunner::start` accepts, so an operator running the probe against
/// either has to find the answer where the model is.
///
/// Not `metadata_dir` + a fixed name: the sidecar is named AFTER the checkpoint
/// (`foo.gguf.lightbulb-chat-template.json`), so that two `.gguf` files in one
/// directory get two sidecars instead of fighting over one.
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

    // Tiers 1 and 2 read companion JSON, which for a `.gguf` checkpoint sits
    // beside the file rather than inside it. Tier 3 deliberately keeps the
    // ORIGINAL path: `from_family` matches path components leaf-first, and for
    // a single-file checkpoint the file name is the component carrying the
    // model's name.
    let meta = metadata_dir(model_dir);

    // Tier 1 — the authoritative source, when the checkpoint ships one.
    if let Ok(raw) = std::fs::read_to_string(meta.join("tokenizer_config.json")) {
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
    if let Some(t) = registry::from_vocab_signature(meta) {
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
         Run `lightbulb-probe {}` to determine one.",
        model_dir.display(),
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

    // A `.gguf` checkpoint's companion JSON is beside the file, not inside it.
    // See `metadata_dir`.
    let meta = metadata_dir(model_path);

    // Tier 1 — the checkpoint's own declaration.
    if let Ok(raw) = std::fs::read_to_string(meta.join("tokenizer_config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            out.bos = token_field(&v, "bos_token").unwrap_or_default();
            out.eos = token_field(&v, "eos_token").unwrap_or_default();
        }
    }

    // Tier 2 — ids from config.json, resolved through tokenizer.json. Applied
    // per token, not per file: a checkpoint that declares one and not the other
    // gets the one it declares plus a resolved fallback for the other.
    if out.bos.is_empty() || out.eos.is_empty() {
        let cfg = std::fs::read_to_string(meta.join("config.json"))
            .ok()
            .and_then(|r| serde_json::from_str::<serde_json::Value>(&r).ok());
        let tok = std::fs::read_to_string(meta.join("tokenizer.json"))
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
        // Names `meta`, not `model_path`: for a `.gguf` checkpoint those differ,
        // and the operator needs the directory the files are actually read
        // from, not the path to the weights.
        tracing::warn!(
            "{} declares no {}; rendering its chat template with an empty string there. \
             A chat template that references the missing token will omit it. Add a \
             tokenizer_config.json with bos_token/eos_token, or a config.json with \
             bos_token_id/eos_token_id whose ids appear in tokenizer.json's added_tokens.",
            meta.display(),
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

/// Resolve a template for serving, in exactly the shape `AppState.chat_template`
/// holds: `None` when no tier produced one.
///
/// The `Resolution::None` check is not cosmetic. An empty template source
/// renders to an empty prompt, which reaches the model as a request to continue
/// nothing, so the handlers need to tell "no template" from "this template"
/// in order to fall back to the legacy join.
///
/// **This exists so there is exactly one copy of that check.** There were four
/// — `api/mod.rs`'s startup plus a private `chat_template_for` in each of
/// `tests/api_result_metadata.rs`, `tests/chat_template_e2e.rs` and
/// `tests/fuel_engine_http.rs` — and each copy's doc comment explained that a
/// harness drifting from startup would silently measure a prompt no client ever
/// causes. Four copies of a rule is the mechanism by which they drift, which is
/// the same defect as the three copies of the `role: content` join this module
/// was written to remove.
pub fn resolve_for_serving(model_path: &Path) -> Option<std::sync::Arc<ResolvedTemplate>> {
    let t = resolve_for_model(model_path);
    (t.resolved_by() != Resolution::None).then(|| std::sync::Arc::new(t))
}

// ─── The probe's report ─────────────────────────────────────────────────────
//
// The row type and the formatter live HERE rather than in `src/bin/
// lightbulb-probe.rs` so they are unit-testable without a model: `src/bin/*.rs`
// is a separate crate, and an integration test cannot reach into one. The
// binary keeps only the parts that genuinely need a checkpoint.

/// One candidate's result from a probe run.
#[derive(Debug, Clone)]
pub struct ProbeRow {
    pub candidate: &'static str,
    /// The template source that produced this row.
    ///
    /// Carried, not re-looked-up by name. `Sidecar.template` is the SOURCE, and
    /// a row that holds only the name forces the selection step to go back to
    /// `registry::candidates()` and match on a string — which is the same "two
    /// values that must correspond, related only by convention" hazard the
    /// named fields below exist to remove, just moved into the one step that
    /// writes to disk.
    pub source: &'static str,
    /// Did generation stop on EOS rather than exhaust its budget?
    pub stopped_on_eos: bool,
    pub tokens_generated: usize,
    /// First line of what the model produced, for the operator to eyeball.
    ///
    /// **The caller enforces this shape; the formatter does not.** Build it with
    /// [`first_line_of`] rather than by hand — error text from `anyhow` and
    /// `minijinja` is routinely multi-line and long, and a raw newline in this
    /// field breaks the table into rows that look like extra candidates.
    pub first_line: String,
}

/// Reduce arbitrary model or error output to something that fits one table cell.
///
/// Split first, THEN truncate: truncating first can leave a newline inside the
/// 60 chars. Empty input yields an empty cell, which is a legitimate result — a
/// model that produced nothing.
pub fn first_line_of(text: &str) -> String {
    // First NON-BLANK line, not first line. Measured 2026-08-13 on the first
    // live probe run: the `llama2` candidate generated 17 tokens and rendered
    // as an empty cell, because its template ends `[/INST]` with no trailing
    // newline and the model opens its reply with one. The operator saw a blank
    // row for the candidate whose behaviour differed most — the exact row they
    // most needed to read. `lines().next()` was literally correct and useless.
    let line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    if line.chars().count() <= 60 {
        line.to_string()
    } else {
        line.chars().take(57).collect::<String>() + "..."
    }
}

/// Format probe results as a per-candidate table.
///
/// **Reports every candidate, and never picks the winner.** Selection is
/// [`select_probe_winner`]'s job and is deliberately a separate step: an
/// operator reading this table must be able to see a board where two candidates
/// both fired, or none did, and draw their own conclusion. Printing only a
/// leader would hide exactly the cases where the probe should not be trusted —
/// which are precisely the boards selection refuses to act on.
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

/// Which candidate, if any, a probe run may act on.
///
/// **Refuse rather than guess** — the whole design rests on this one rule. The
/// probe's conclusion is persisted at [`Resolution::Probe`], which [`resolve`]
/// reads BEFORE every other tier on the next server start, so a wrong answer
/// here outranks the checkpoint's own `tokenizer_config.json` indefinitely and
/// silently. Exactly one candidate stopping on EOS is the only board that
/// carries a signal; zero says none of these templates suits the checkpoint,
/// and two-or-more says the discriminator failed to discriminate. Neither of
/// those is a tie to break — they are results the operator has to see.
///
/// This lives here, beside [`ProbeRow`] and [`format_probe_report`], rather
/// than in `src/bin/lightbulb-probe.rs`, for the reason given above them:
/// `src/bin/*.rs` is a separate crate and no integration test can reach into
/// one. While this `match` sat in the binary it had no test of any kind, so the
/// safety property was asserted nowhere. The binary keeps only the parts that
/// genuinely need a checkpoint.
pub fn select_probe_winner(rows: &[ProbeRow]) -> anyhow::Result<&ProbeRow> {
    let fired: Vec<&ProbeRow> = rows.iter().filter(|r| r.stopped_on_eos).collect();
    match fired.as_slice() {
        [only] => Ok(only),
        // Refusal 1. Note what this does NOT claim: a model that failed to LOAD
        // cannot reach this message, because that aborts the probe at the first
        // candidate with the runner's own error. By here the model demonstrably
        // loaded and generated.
        [] => anyhow::bail!(
            "no candidate stopped on EOS, so none of these three templates suits this checkpoint. \
             (A model that failed to LOAD cannot reach this message — that aborts the probe at \
             the first candidate with the runner's own error.) Nothing was written."
        ),
        // Refusal 2. Do not tie-break: two templates that both end the turn
        // cleanly is a result the operator has to see, not one to guess past.
        // `many.len()` is printed rather than a literal `2` because three
        // candidates firing is the measured TinyLlama board, not a hypothetical.
        many => anyhow::bail!(
            "{} candidates stopped on EOS ({}). The probe cannot choose between them and will not \
             guess. Nothing was written.",
            many.len(),
            many.iter()
                .map(|r| r.candidate)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}
