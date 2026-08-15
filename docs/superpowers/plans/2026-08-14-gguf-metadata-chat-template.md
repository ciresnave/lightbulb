# GGUF Metadata Chat Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `.gguf` checkpoint resolves its chat template and special tokens from its own embedded metadata, so the served prompt is the one the model was trained on.

**Architecture:** One new private function reads the GGUF header through Fuel's existing `MmapedContent` and returns what the file declares. `resolve` and `special_tokens` consult it as an additional **source within existing tier 1** before falling through to `tokenizer_config.json`. A new `Resolution::GgufMetadata` records the provenance.

**Tech Stack:** Rust (edition 2024), `fuel-core`'s `quantized::gguf_mmap` / `quantized::gguf_file`, existing `minijinja` rendering.

**Spec:** `docs/superpowers/specs/2026-08-14-gguf-metadata-chat-template-design.md` — read §3 (precedence) and §7 (failure table) before Task 2.

## Global Constraints

- **Build with `-j 4`.** Full-parallelism cargo on this machine races into rustc ICEs and spurious `crate ... required to be available in rlib format` / `E0786 ... paging file is too small` errors naming different crates each run. Not your code — retry, do not investigate.
- **`running N tests` is the only real signal.** `tests/api_result_metadata.rs` and `tests/fuel_engine_http.rs` open with `#![cfg(feature = "fuel-engine")]`; without the feature they compile to **zero tests** and print `ok. 0 passed` — identical in shape to a suite that ran. Read the count, never the exit code.
- **When piping cargo through `grep`/`tail`, print `${PIPESTATUS[0]}`.** The pipeline's status masks cargo's; this has already produced a false green on this project.
- **`rustfmt --edition 2024 <file>`** every file you touch. **Never on a `mod.rs`** — rustfmt follows `pub mod` declarations and rewrote 8 unrelated files last time. `src/api/chat_template.rs` declares `pub mod registry;`, so after formatting it run `git status --short src/api/chat_template/` and revert `registry.rs` if you did not touch it.
- **Never assert a property that holds whether or not the code is correct.** `assert!(x > 0)` against a count that was already nonzero proves nothing. Prefer equality against an independently-derived value.
- **Stage explicit paths.** Never `git add -A` — the untracked `supertool` must stay untracked.
- **Byte-exact, no normalization.** Special tokens are rendered into prompt *text* and the tokenizer must recognise the identical bytes. Never trim, case-fold, or Unicode-normalize a token or template read from a file.
- **Baselines before you start:** `cargo test -j 4 --lib` → `running 657 tests` / `643 passed; 0 failed; 14 ignored`. `cargo test -j 4 --test chat_template_render` → `running 58 tests` / `56 passed; 0 failed; 2 ignored`.

### Verified against the tree at `d69cf65` — do not re-derive these from memory

| Fact | Value |
| --- | --- |
| Metadata reader | `fuel::quantized::gguf_mmap::MmapedContent::from_path<P: AsRef<Path>>(path) -> fuel::Result<Self>` |
| Metadata map | `MmapedContent::metadata(&self) -> &HashMap<String, Value>` |
| `Value` | `fuel::quantized::gguf_file::Value` (re-export of `fuel_formats::gguf::Value`) — variants `U8 I8 U16 I16 U32 I32 U64 I64 F32 F64 Bool String(String) Array(Vec<Value>)` |
| String accessor | `Value::to_string(&self) -> fuel::Result<&String>` — **returns `Result<&String>`, not `String`** |
| Int accessor | `Value::to_u32(&self) -> fuel::Result<u32>` (also `to_u64`, etc.) |
| Array accessor | `Value::to_vec(&self) -> fuel::Result<&Vec<Value>>` |
| Blank-template guard | `fn non_blank(source: String, origin: &str) -> Option<String>` at `chat_template.rs:440` — already exists, **reuse it** |
| `resolve` tier 1 | `chat_template.rs:530-541` |
| `special_tokens` | `pub fn special_tokens(model_path: &Path) -> SpecialTokens` at `:677`; tier 1 reads `meta.join("tokenizer_config.json")` at `:685` |
| `metadata_dir` | `fn metadata_dir(model_path: &Path) -> &Path` — returns the parent for a file, the path itself for a directory |
| `Resolution` | `Sidecar, TokenizerConfig, VocabSignature, Registry, Probe, None` |
| Exhaustive match | `probe_override_check` matches `Resolution` with **no `_` arm** — adding a variant is a compile error there **by design** |
| GGUF keys | `tokenizer.chat_template`, `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id`, `tokenizer.ggml.tokens` |

**`fuel-core` is an unconditional dependency** (`Cargo.toml:215`, no `optional`) and `fuel-engine = []` enables no dependencies. So this works in the **default** build. Do not add a feature gate.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/api/chat_template.rs` *modify* | Add `GgufDeclaration` + `read_gguf_declaration`; add `Resolution::GgufMetadata`; consult it in `resolve` and `special_tokens`; classify the new variant in `probe_override_check` |
| `tests/chat_template_render.rs` *modify* | Synthetic-GGUF builder + unit tests for every row of the spec's failure table |
| `tests/gguf_serving_e2e.rs` **new** | One `#[ignore]`d behavioural gate against the real Q4_0 file |
| `src/loaders/mlmf_wrapper.rs` **delete** | Orphaned, never compiled (Task 4) |

---

## Task 1: Read the declaration out of a GGUF header

**Files:**
- Modify: `src/api/chat_template.rs`
- Test: `tests/chat_template_render.rs`

**Interfaces:**
- Consumes: `non_blank(String, &str) -> Option<String>` (existing, `:440`).
- Produces:
  ```rust
  pub(crate) struct GgufDeclaration { pub template: Option<String>, pub bos: Option<String>, pub eos: Option<String> }
  fn read_gguf_declaration(model_path: &Path) -> Option<GgufDeclaration>
  ```
  Returns `None` when `model_path` is not a `.gguf` file or the header cannot be parsed. A `Some` with all-`None` fields means "it is a GGUF and it declares nothing" — a different fact from "not a GGUF", and the caller needs both.

- [ ] **Step 1: Write the synthetic-GGUF builder and the first failing test**

The builder is the load-bearing part of this task. A real 640 MB file has exactly one metadata set; every row of the spec's failure table needs a *different* one, so the fixtures must be constructed. GGUF v3 layout: magic `GGUF`, `u32` version, `u64` tensor count, `u64` KV count, then KV pairs of `(u64 len + bytes) key`, `u32 value-type`, value. Type tags: `8` = string, `4` = u32, `9` = array (`u32` elem type, `u64` count, elements).

Add to `tests/chat_template_render.rs`:

```rust
// ─── Synthetic GGUF fixtures ────────────────────────────────────────────────
//
// A valid GGUF v3 header with ZERO tensors is a few hundred bytes, so every row
// of the spec's failure table gets its own file. The real 640 MB Q4_0 checkpoint
// has one metadata set and cannot exercise a missing key, a blank template, or
// an out-of-range token id — the cases that matter are the malformed ones.

/// One metadata value to write into a synthetic header.
enum Kv {
    Str(&'static str),
    U32(u32),
    StrArray(Vec<&'static str>),
}

fn gguf_bytes(kvs: &[(&str, Kv)]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"GGUF");
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&0u64.to_le_bytes()); // tensor count
    b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    let mut put_str = |b: &mut Vec<u8>, s: &str| {
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
    std::fs::write(&p, gguf_bytes(kvs)).unwrap();
    p
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
            ("tokenizer.ggml.tokens", Kv::StrArray(vec!["<unk>", "<s>", "</s>"])),
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
```

- [ ] **Step 2: Run it and watch it fail**

```bash
cargo test -j 4 --test chat_template_render a_gguf_declares 2>&1 | grep -E "^error|^running|^test result"; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: a compile error — `Resolution::GgufMetadata` does not exist yet.

- [ ] **Step 3: Add the `Resolution` variant**

In `src/api/chat_template.rs`, add to `enum Resolution`, after `TokenizerConfig`:

```rust
    /// Read from the `.gguf` file's own embedded metadata
    /// (`tokenizer.chat_template`). For a single-file checkpoint this is the
    /// model author's declaration, shipped with these weights — the same
    /// authority `TokenizerConfig` carries for a directory checkpoint.
    ///
    /// A distinct variant rather than reuse of `TokenizerConfig`: this module
    /// exists to record HOW a template was chosen, and labelling a template read
    /// from file metadata as having come from a JSON file is the quiet kind of
    /// wrong the sidecar's `evidence` field was introduced to prevent.
    GgufMetadata,
```

This **will** break `probe_override_check`'s match. That is the guard working; Task 3 classifies it.

- [ ] **Step 4: Add the reader**

Add near `metadata_dir` in `src/api/chat_template.rs`:

```rust
/// What a `.gguf` file declares about its own chat template and special tokens.
///
/// `None` from [`read_gguf_declaration`] means "not a GGUF, or unreadable as
/// one". A `Some` whose fields are all `None` means "a valid GGUF that declares
/// nothing" — a different fact, and the callers need to tell them apart.
pub(crate) struct GgufDeclaration {
    pub template: Option<String>,
    pub bos: Option<String>,
    pub eos: Option<String>,
}

/// Read the declaration out of a `.gguf` header.
///
/// Uses Fuel's `MmapedContent`, not our `src/gguf/` and not MLMF. `fuel-core` is
/// an unconditional dependency and `fuel-engine` enables none, so this works in
/// the default build; `src/gguf/` is candlelight-era and slated for retirement
/// with the three `src/model/custom_*` files that take `crate::gguf::Content`.
/// The read is behind this one function so the reader can be swapped later
/// without touching resolution.
///
/// Mapping is virtual and only the header is touched, and this runs once at
/// startup — not per request.
fn read_gguf_declaration(model_path: &Path) -> Option<GgufDeclaration> {
    if !model_path.is_file()
        || !model_path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
    {
        return None;
    }

    let mc = match fuel::quantized::gguf_mmap::MmapedContent::from_path(model_path) {
        Ok(mc) => mc,
        Err(e) => {
            // Do NOT say "this is not a valid GGUF" — measured 2026-08-15, that
            // is often a false statement about the operator's file. Fuel's
            // `Content::read` parses the metadata block FIRST and in full, then
            // walks the tensor directory and calls `GgmlDType::from_u32`
            // (`fuel-formats/src/gguf.rs:432-433`), whose table at our pinned rev
            // `8771997e` (`fuel-ir/src/quantized.rs:35`) accepts only
            // {0,1,2,3,6..15,30}. Every IQ code is absent — IQ4_NL (20),
            // IQ3_S (21), IQ4_XS (23) among them — all ordinary quantizations
            // llama.cpp reads happily. On such a file the chat template is
            // decoded, sits in memory, and is discarded because of a tensor we
            // never needed. There is no metadata-only entry point to avoid it.
            //
            // The consequence is exactly the defect this epic fixes: a silent
            // fall-through to a family guess with no end-of-turn marker. So the
            // message must point at the real cause, or an operator will go
            // looking for corruption that isn't there.
            // "does not YET decode" would be false for codes 4 and 5 (Q4_2,
            // Q4_3), which Fuel's table also rejects but which are *retired*,
            // not future — ggml has eight such codes. Saying "not yet" about an
            // old file sends the operator looking forward when they should look
            // back, which is the same wrong-cause failure this whole branch
            // exists to stop. Name the symptom, not a guess at the era.
            tracing::warn!(
                "{}: Fuel could not open this GGUF ({e}); falling through to the \
                 file-based tiers. If the file is otherwise sound, the likely cause \
                 is a tensor quantization outside Fuel's ggml type table — which \
                 covers 15 codes and omits 20, both newer families (IQ*, TQ*) and \
                 retired ones (Q4_2, Q4_3). The chat template may be present and \
                 readable but is unreachable until that table covers this file.",
                model_path.display()
            );
            return None;
        }
    };
    let md = mc.metadata();

    // `to_string()` here returns `Result<&String>` — it is Fuel's accessor, not
    // `Display::to_string`. Cloning is deliberate: the mmap is dropped with `mc`.
    let template = md
        .get("tokenizer.chat_template")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .and_then(|s| non_blank(s, &format!("{} (GGUF metadata)", model_path.display())));

    // Byte-exact: the token is rendered into prompt TEXT and the tokenizer must
    // recognise the identical bytes. Never trim or normalize here.
    let tokens = md.get("tokenizer.ggml.tokens").and_then(|v| v.to_vec().ok());
    let lookup = |key: &str| -> Option<String> {
        let id = md.get(key)?.to_u32().ok()? as usize;
        let toks = tokens?;
        match toks.get(id).and_then(|t| t.to_string().ok()) {
            Some(s) => Some(s.clone()),
            None => {
                tracing::warn!(
                    "{}: GGUF {key} is {id}, which is out of range for tokenizer.ggml.tokens \
                     ({} entries); leaving that token empty.",
                    model_path.display(),
                    toks.len()
                );
                None
            }
        }
    };

    Some(GgufDeclaration {
        template,
        bos: lookup("tokenizer.ggml.bos_token_id"),
        eos: lookup("tokenizer.ggml.eos_token_id"),
    })
}
```

- [ ] **Step 5: Consult it in `resolve`**

In `src/api/chat_template.rs`, immediately **before** the existing `// Tier 1 — the authoritative source` block at `:530`:

```rust
    // Tier 1, GGUF branch — checked BEFORE `tokenizer_config.json`.
    //
    // A `.gguf` is file-scoped; a companion JSON is directory-scoped. Two
    // `.gguf` files can share a directory — which is why sidecars are named
    // after the checkpoint rather than sharing one — so a single
    // `tokenizer_config.json` beside them cannot be authoritative for both. The
    // in-file metadata is unambiguously about THESE weights. Spec §3.
    if let Some(d) = read_gguf_declaration(model_dir) {
        if let Some(t) = d.template {
            tracing::info!("chat template: GGUF metadata");
            return ChatTemplate {
                source: t,
                resolved_by: Resolution::GgufMetadata,
            };
        }
        tracing::debug!(
            "{}: valid GGUF, no usable tokenizer.chat_template; falling through.",
            model_dir.display()
        );
    }
```

- [ ] **Step 6: Consult it in `special_tokens`**

In `special_tokens`, immediately **before** the existing `// Tier 1 — the checkpoint's own declaration` block at `:685`:

```rust
    // Same precedence as `resolve`: a GGUF's own metadata outranks a companion
    // JSON, per spec §3. Fills each token independently — a file that declares
    // one and not the other gets the one it declares plus a resolved fallback
    // for the other, matching the existing per-token behaviour below.
    if let Some(d) = read_gguf_declaration(model_path) {
        if let Some(b) = d.bos {
            out.bos = b;
        }
        if let Some(e) = d.eos {
            out.eos = e;
        }
    }
```

- [ ] **Step 7: Run and format**

```bash
cargo test -j 4 --test chat_template_render 2>&1 | grep -E "^error|^running|^test result"; echo "EXIT=${PIPESTATUS[0]}"
rustfmt --edition 2024 src/api/chat_template.rs tests/chat_template_render.rs
git status --short src/api/chat_template/   # revert registry.rs if untouched by you
```

Expected: `running 59 tests` / `57 passed; 0 failed; 2 ignored`.

- [ ] **Step 8: Commit**

```bash
git add src/api/chat_template.rs tests/chat_template_render.rs
git commit -m "feat(api): Read the chat template from a GGUF's own metadata"
```

---

## Task 2: Cover the failure table

**Files:**
- Modify: `tests/chat_template_render.rs`

**Interfaces:**
- Consumes: `gguf_fixture`, `Kv`, `gguf_bytes` from Task 1; `resolve`, `special_tokens`, `Resolution::GgufMetadata`.
- Produces: nothing new.

Spec §7 lists six conditions. Task 1 covered the happy path only. **Each test below must fail against a different wrong implementation** — check that as you write them.

- [ ] **Step 1: Write the tests**

```rust
/// A GGUF that declares no template falls through — it is a normal checkpoint,
/// not an error. Asserts the OBSERVABLE consequence: the family registry answers.
#[test]
fn a_gguf_without_a_template_falls_through_to_the_registry() {
    let p = gguf_fixture(
        "gguf-no-template",
        &[("tokenizer.ggml.tokens", Kv::StrArray(vec!["<unk>", "<s>", "</s>"]))],
    );
    let t = lightbulb::api::chat_template::resolve(&p);
    assert_ne!(t.resolved_by, Resolution::GgufMetadata);
    // Tokens still come from the file even when the template does not.
    let tk = lightbulb::api::chat_template::special_tokens(&p);
    assert_eq!(tk.bos, "");
    assert_eq!(tk.eos, "");
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
    assert_eq!(tk.eos, "", "the out-of-range id must not resolve to anything");
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
```

- [ ] **Step 2: Run**

```bash
cargo test -j 4 --test chat_template_render 2>&1 | grep -E "^error|^running|^test result"; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: `running 66 tests` / `64 passed; 0 failed; 2 ignored`.

- [ ] **Step 3: Verify each test discriminates**

For each mutation below, apply it, run, note which test fails, restore, **and `touch src/api/chat_template.rs`** — restoring bytes is not enough, cargo reuses the mutant binary.

| mutation | must fail |
| --- | --- |
| `non_blank(...)` → `Some(s)` in `read_gguf_declaration` | `a_blank_gguf_template_is_unusable_and_falls_through` |
| drop the `is_file()`/extension check | `a_directory_checkpoint_still_resolves_from_tokenizer_config` |
| move the GGUF branch *after* the `tokenizer_config.json` block in `resolve` | `gguf_metadata_outranks_a_companion_tokenizer_config` |
| `toks.get(id)` → `toks.get(id.min(toks.len()-1))` | `an_out_of_range_token_id_leaves_the_token_empty` |

Report any mutation that does **not** kill a test — that is a coverage hole, not a formality.

- [ ] **Step 4: Format and commit**

```bash
rustfmt --edition 2024 tests/chat_template_render.rs
git add tests/chat_template_render.rs
git commit -m "test(api): Cover every GGUF-metadata failure mode"
```

---

## Task 3: Classify the new variant for the probe

**Files:**
- Modify: `src/api/chat_template.rs`
- Test: `tests/chat_template_render.rs`

**Interfaces:**
- Consumes: `Resolution::GgufMetadata` (Task 1); `probe_override_check(current: Resolution, force: bool) -> ProbeOverride`.
- Produces: nothing new.

Task 1's new variant breaks `probe_override_check`'s exhaustive match. That is deliberate — the match has **no `_` arm** precisely so a new variant is a compile error rather than a silent "proceed". Left unclassified, the build does not compile; classified wrongly, the probe would overwrite a checkpoint's own declaration.

- [ ] **Step 1: Write the failing test**

```rust
/// A GGUF that declares its own template must be protected from the probe
/// exactly as a `tokenizer_config.json` is — the probe would otherwise write a
/// registry candidate at `Resolution::Probe`, which `resolve` reads at tier 0,
/// AHEAD of the author's own declaration.
#[test]
fn the_probe_refuses_to_override_a_gguf_declaration() {
    use lightbulb::api::chat_template::{ProbeOverride, probe_override_check};

    match probe_override_check(Resolution::GgufMetadata, false) {
        ProbeOverride::Refuse(msg) => {
            assert!(msg.contains("GgufMetadata"), "message must name the tier: {msg}");
            assert!(msg.contains("Nothing was written."), "must match the other refusals: {msg}");
        }
        other => panic!("expected Refuse for a checkpoint's own declaration, got {other:?}"),
    }

    // `--force` still overrides, like every other refusal.
    assert!(
        !matches!(probe_override_check(Resolution::GgufMetadata, true), ProbeOverride::Refuse(_)),
        "--force must still get past the guard"
    );
}
```

- [ ] **Step 2: Run to confirm it fails to compile**

```bash
cargo test -j 4 --test chat_template_render the_probe_refuses 2>&1 | grep -E "^error" | head -5; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: `error[E0004]: non-exhaustive patterns: `Resolution::GgufMetadata` not covered` — from `probe_override_check`, which is the guard doing its job.

- [ ] **Step 3: Classify it**

In `probe_override_check`, add `Resolution::GgufMetadata` to the **same arm as `Resolution::TokenizerConfig`**, and extend that arm's message so it names whichever tier was passed rather than hardcoding "tokenizer_config.json". Both are the checkpoint's own declaration; the probe can only offer a registry candidate, which is strictly less authoritative.

- [ ] **Step 4: Run**

```bash
cargo test -j 4 --test chat_template_render 2>&1 | grep -E "^error|^running|^test result"; echo "EXIT=${PIPESTATUS[0]}"
cargo build -j 4 --bin lightbulb-probe 2>&1 | grep -E "^error"; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: `running 67 tests` / `65 passed; 0 failed; 2 ignored`, and the probe binary builds.

- [ ] **Step 5: Format and commit**

```bash
rustfmt --edition 2024 src/api/chat_template.rs tests/chat_template_render.rs
git status --short src/api/chat_template/
git add src/api/chat_template.rs tests/chat_template_render.rs
git commit -m "fix(cli): Stop the probe overriding a GGUF's own declaration"
```

---

## Task 4: The behavioural gate, and the orphan

**Files:**
- Create: `tests/gguf_serving_e2e.rs`
- Delete: `src/loaders/mlmf_wrapper.rs`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Delete the orphan**

`src/loaders/mlmf_wrapper.rs` is 6,501 bytes, dated 2026-07-28, and calls `mlmf::prelude`, `mlmf::callbacks::default_progress_callback`, `mlmf::load_safetensors`, `mlmf::load_gguf`. It was never re-declared as a module after the MLMF integration was abandoned, so it does not compile and is invisible.

```bash
grep -rn "mlmf_wrapper" src/          # expect: no output
git rm src/loaders/mlmf_wrapper.rs
cargo check -j 4 --all-targets 2>&1 | grep -cE "^error"   # expect: 0
```

If the grep prints anything, **stop** — it is referenced and this plan is wrong.

- [ ] **Step 2: Write the gate**

`tests/gguf_serving_e2e.rs` — model it on `tests/chat_template_e2e.rs` for router/runner setup. **Do not** feature-gate it: this works in the default build, and a `#![cfg(...)]` would make it compile to zero tests while printing `ok. 0 passed`.

```rust
//! Serving a real GGUF end to end.
//!
//! Known-red baseline, measured 2026-08-14 BEFORE this work, on
//! `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` @ `52e7645b`, Q4_0, over HTTP at
//! `temperature: 0.0`:
//!
//! ```text
//! content: "| ass istant | <0x0A> | ass istant | ass istant | | ass istant |"
//! Chat template … resolved via Registry (bos "", eos "")
//! ```
//!
//! The file declares `tokenizer.chat_template` and ids 1/2 → `<s>`/`</s>`;
//! resolution simply never looked inside it. Re-running this test against a
//! commit before that fix must reproduce the garbage above — that is what makes
//! this gate a gate rather than a description.
```

The harness below is `chat_template_e2e.rs`'s `post_raw` with the checkpoint
source swapped; **`AppState` needs all six fields** (`eos_monitor` included) or it
will not compile.

```rust
use axum::body::Body;
use axum::http::{Request, StatusCode};
use lightbulb::api::chat_template::{self, Resolution};
use lightbulb::api::{ApiConfig, AppState};
use lightbulb::engine::model_runner::ModelRunner;
use lightbulb::engine::{MemoryAwareScheduler, memory_aware_scheduler::MemoryAwareConfig};
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceExt;

const MISSING: &str = "no GGUF checkpoint. Set LIGHTBULB_GGUF to a .gguf file.";

fn gguf_path() -> Option<PathBuf> {
    let p = match std::env::var_os("LIGHTBULB_GGUF") {
        Some(v) => PathBuf::from(v),
        None => PathBuf::from(
            "C:/Users/cires/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        ),
    };
    p.is_file().then_some(p)
}

#[tokio::test]
#[ignore = "needs the GGUF checkpoint"]
async fn a_gguf_is_served_with_its_own_template() {
    let path = gguf_path().expect(MISSING);

    // 1. The file's own declaration is what resolution used. Exact strings, not
    //    `!is_empty()`: empty is precisely what the defect produced.
    let tk = chat_template::special_tokens(&path);
    assert_eq!(tk.bos, "<s>", "BOS did not come from GGUF metadata");
    assert_eq!(tk.eos, "</s>", "EOS did not come from GGUF metadata");
    assert_eq!(
        chat_template::resolve(&path).resolved_by,
        Resolution::GgufMetadata,
        "resolution fell through to a guess instead of reading the file"
    );

    // 2. And the model answers, rather than free-associating.
    let tx = ModelRunner::start(&path, 1, 512, Some("f32".to_string()))
        .expect("starting the model runner");
    let state = AppState {
        scheduler: Arc::new(MemoryAwareScheduler::new(MemoryAwareConfig::default())),
        config: ApiConfig::default(),
        db_pool: None,
        inference_tx: Some(tx),
        chat_template: chat_template::resolve_for_serving(&path),
        eos_monitor: Default::default(),
    };
    let app = lightbulb::api::openai::routes().with_state(state);
    let body = serde_json::json!({
        "model": "tinyllama",
        "messages": [{"role": "user", "content": "Name the capital of France."}],
        "max_tokens": 24,
        "temperature": 0.0,
    });
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .expect("router returned no response");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    eprintln!("completion: {content:?}");

    assert!(
        content.contains("Paris"),
        "the model did not answer the question: {content:?}"
    );
    // The recorded pre-fix output; named so a regression is recognisable rather
    // than just a failed `contains`.
    assert!(
        !content.contains("ass istant"),
        "this is the pre-fix free-association output: {content:?}"
    );
}
```

The two resolution assertions are what make `contains("Paris")` non-accidental — a completion could contain "Paris" for unrelated reasons, and pinning the resolved source records *why* it did.

- [ ] **Step 3: Run it**

```bash
export LIGHTBULB_GGUF="C:/Users/cires/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
cargo test --release -j 4 --test gguf_serving_e2e -- --include-ignored --nocapture --test-threads=1 2>&1 | grep -E "^running|^test result|^error"; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: `running 1 test` / `1 passed`. Budget several minutes — it loads a 640 MB model on CPU.

- [ ] **Step 4: Commit**

```bash
rustfmt --edition 2024 tests/gguf_serving_e2e.rs
git add tests/gguf_serving_e2e.rs
git commit -m "test(api): Gate GGUF serving end to end, and delete the MLMF orphan"
```

---

## Final verification

- [ ] Full suite, reading counts not exit codes:

```bash
cargo test --tests -j 4 2>&1 | grep -cE "^test result: FAILED"   # expect 0
cargo test -j 4 --lib 2>&1 | grep -E "^running|^test result"      # expect 657/643 unchanged
```

- [ ] Both feature configurations clean:

```bash
cargo check -j 4 --all-targets 2>&1 | grep -cE "^error"
cargo check -j 4 --all-targets --features fuel-engine 2>&1 | grep -cE "^error"
```

- [ ] `grep -rn "mlmf" src/` returns nothing.

## Deliberately out of scope

GGUF on the Fuel backend (needs the pin bump to `f1da2d94+`), adopting MLMF, retiring `src/gguf/` and the candlelight loaders, and sharded GGUF. See spec §8.
