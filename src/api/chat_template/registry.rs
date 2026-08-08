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
//
// `every_registry_constant_renders_exactly` in `tests/chat_template_render.rs`
// pins each of these as an EXACT rendered string, single-turn and multi-turn,
// as well as the tail. The whole body has to be pinned, not just the tail: a
// constant that renders to the wrong shape still renders, so nothing falls
// through to another tier and nothing is logged. Edit a constant, and that
// test must be updated for the right reason.
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

/// Tier 3 — the checkpoint's name, read out of the path one component at a
/// time, **leaf-first**. The first component matching any rule wins, so the
/// name nearest the model outvotes every directory above it.
///
/// Matching the whole path as one lowercased string — what this did before —
/// let a parent directory outvote the model:
/// `from_family(r"C:\models\qwen-experiments\Llama-2-7b-chat-hf")` returned
/// CHATML. That is not hypothetical bad luck: `api/mod.rs` builds the path as
/// `LIGHTBULB_MODELS_DIR.join(default_model)`, so a models root named
/// `D:\qwen\` mapped *every* model under it to one family — reported as
/// `Resolution::Registry`, which reads as "matched the model", not "matched
/// your folder layout".
///
/// Basename-only is the other wrong answer. The HF cache layout is
/// `models--Org--Name/snapshots/<sha>`, whose leaf is a sha and carries no name
/// at all. Leaf-first keeps that working while still giving the nearest real
/// name the vote.
pub fn from_family(model_dir: &Path) -> Option<String> {
    model_dir
        .components()
        .rev()
        .find_map(|c| from_component(&c.as_os_str().to_string_lossy().to_lowercase()))
}

/// The rules, applied to one already-lowercased path component.
///
/// Order is load-bearing wherever a single component satisfies two rules —
/// `qwen-llama-2-merge` is CHATML, not LLAMA2 — so it is pinned by
/// `family_registry_rule_precedence_within_one_component` in
/// `tests/chat_template_render.rs`.
fn from_component(name: &str) -> Option<String> {
    // The name, not `config.json`'s `model_type`: it carries the chat-variant
    // distinction that `model_type` does not. A base and a chat model share
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
