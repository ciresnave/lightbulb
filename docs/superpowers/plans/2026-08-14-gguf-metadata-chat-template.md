# GGUF Metadata Chat Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `.gguf` checkpoint resolves its chat template and special tokens from its own embedded metadata, so the served prompt is the one the model was trained on.

**Architecture:** One new private function reads the GGUF header through Fuel's existing `MmapedContent` and returns what the file declares. `resolve` and `special_tokens` consult it as an additional **source within existing tier 1** before falling through to `tokenizer_config.json`. A new `Resolution::GgufMetadata` records the provenance.

**Tech Stack:** Rust (edition 2024), `fuel-core`'s `quantized::gguf_mmap` / `quantized::gguf_file`, existing `minijinja` rendering.

**Spec:** `docs/superpowers/specs/2026-08-14-gguf-metadata-chat-template-design.md` — read §3 (precedence) and §7 (failure table) before Task 2.

## Global Constraints

- **Build with `-j 4`.** Full-parallelism cargo on this machine races into rustc ICEs and spurious `crate ... required to be available in rlib format` / `E0786 ... paging file is too small` errors naming different crates each run. Not your code — retry, do not investigate.
- **`running N tests` is the only real signal.** **Three** test files open with `#![cfg(feature = "fuel-engine")]` — `tests/api_result_metadata.rs:14`, `tests/chat_template_e2e.rs:9`, and `tests/fuel_engine_http.rs:78`. Without the feature they compile to **zero tests** and print `ok. 0 passed` — identical in shape to a suite that ran. Read the count, never the exit code. **Note that `chat_template_e2e.rs` is the file Task 4 tells you to model the new test on**, and Task 4 tells you *not* to gate the new one; see Task 4 Step 2 for why the source has an attribute the copy must not.
- **When piping cargo through `grep`/`tail`, print `${PIPESTATUS[0]}`.** The pipeline's status masks cargo's; this has already produced a false green on this project.
- **A count of matching lines is not a test result.** `... | grep -c "FAILED"` returns `0` for a tree that **failed to compile**, because a compile error emits no `test result:` line at all. Never gate on the absence of a failure string; gate on the presence of the expected `running N tests` / `N passed` counts.
- **`rustfmt --edition 2024 <file>`** every file you touch. **Never on a `mod.rs`** — rustfmt follows `pub mod` declarations and rewrote 8 unrelated files last time. `src/api/chat_template.rs` declares `pub mod registry;`, so after formatting it run `git status --short src/api/chat_template/` and revert `registry.rs` if you did not touch it.
- **Never assert a property that holds whether or not the code is correct.** `assert!(x > 0)` against a count that was already nonzero proves nothing. Prefer equality against an independently-derived value.
- **Stage explicit paths.** Never `git add -A` — the untracked `supertool` must stay untracked.
- **Byte-exact, no normalization.** Special tokens are rendered into prompt *text* and the tokenizer must recognise the identical bytes. Never trim, case-fold, or Unicode-normalize a token or template read from a file.
- **Baselines before you start:** `cargo test -j 4 --lib` → `running 661 tests` / `647 passed; 0 failed; 14 ignored`. `cargo test -j 4 --test chat_template_render` → `running 58 tests` / `56 passed; 0 failed; 2 ignored`.

### Revision provenance — read this before trusting any section equally

**On this project, revisions are where defects concentrate.** Measured across
three audit rounds of this document: four of the first audit's eight non-blocking
findings came from two rework commits; the second audit's four blocking findings
included two in text written by the first round of fixes. Spend your scepticism
on what changed most recently, not on what looks least finished.

**Do not trust a hand-written list of what changed. Generate it:**

```bash
git log --oneline -- docs/superpowers/plans/2026-08-14-gguf-metadata-chat-template.md
git diff <the-commit-before-the-last-review> HEAD -- \
  docs/superpowers/plans/2026-08-14-gguf-metadata-chat-template.md \
  docs/superpowers/specs/2026-08-14-gguf-metadata-chat-template-design.md
```

> **Why this is a command and not a table.** The previous version of this section
> *was* a table listing which sections were revised. An audit found it wrong in
> two ways that both pointed scepticism away from the newest text: it marked
> Task 1 Step 1 "original" when that step contained the `gguf_bytes` signature
> change every later task consumes, and it omitted Task 2 Step 0 — ninety lines
> of brand-new helper functions — entirely. It was written from the author's
> recollection rather than from a diff, which is exactly the failure it existed
> to prevent, in the artifact built to prevent it. A snapshot of a diff rots; the
> command that produces one cannot.

Revision history, for orientation only — **the diff is authoritative**:

| commit | what it was responding to |
| --- | --- |
| `9507f06` | first draft |
| `1740582`, `c409c7e` | the MLMF relay: GGUF failure-table row split, warn text |
| `af0cda9` | audit round 1 — 4 blocking, 10 should-fix |
| this commit | audit round 2 — 4 blocking, 10 should-fix |

### Verified against the tree — do not re-derive these from memory

First written against `d69cf65`; **every row independently re-verified by audit
at `c409c7e`**, all confirmed unchanged. The tree is now at `d1eca62` (the
`parse_dtype` tests), which touched none of these.

| Fact | Value |
| --- | --- |
| Metadata reader | `fuel::quantized::gguf_mmap::MmapedContent::from_path<P: AsRef<Path>>(path) -> fuel::Result<Self>` |
| Metadata map | `MmapedContent::metadata(&self) -> &HashMap<String, Value>` |
| `Value` | `fuel::quantized::gguf_file::Value` (re-export of `fuel_formats::gguf::Value`) — variants `U8 I8 U16 I16 U32 I32 U64 I64 F32 F64 Bool String(String) Array(Vec<Value>)` |
| String accessor | `Value::to_string(&self) -> fuel::Result<&String>` — **returns `Result<&String>`, not `String`** |
| Int accessor | `Value::to_u64(&self) -> fuel::Result<u64>` (`gguf.rs:242`) — **use this, not `to_u32`**. `to_u32` (`:227`) accepts ONLY the `U32` tag and errors on `I32`/`U64`/`U16`; `to_u64` upcasts from U8/U16/U32/U64/Bool |
| Array accessor | `Value::to_vec(&self) -> fuel::Result<&Vec<Value>>` |
| Blank-template guard | `fn non_blank(source: String, origin: &str) -> Option<String>` at `chat_template.rs:440` — already exists, **reuse it** |
| `resolve` tier 1 | `chat_template.rs:530-541` |
| `special_tokens` | `pub fn special_tokens(model_path: &Path) -> SpecialTokens` at `:677`; tier 1 reads `meta.join("tokenizer_config.json")` at `:685` |
| `metadata_dir` | `fn metadata_dir(model_path: &Path) -> &Path` — returns the parent for a file, the path itself for a directory |
| `Resolution` | `Sidecar, TokenizerConfig, VocabSignature, Registry, Probe, None` |
| Exhaustive match | `probe_override_check` (`:1108`) matches `Resolution` with **no `_` arm** — adding a variant is a compile error there **by design**. It is LIBRARY code, so that error breaks every test target: repair it in the same task (Task 1 Step 3a). It is the only exhaustive match over `Resolution` in the tree; `src/api/openai/chat.rs:446` matches `Option<Resolution>`, not the variants |
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

> ### ⚠️ Correction: this row describes the single-gate design, which was falsified.
>
> "One `#[ignore]`d behavioural gate" was the plan as designed. The spec's §1
> and §9 correction boxes record why it was built as **two** tests instead —
> the root-cause claim the single gate rested on was falsified during
> implementation, so the one gate would have asserted three things that turn
> out not to pass or fail together. `tests/gguf_serving_e2e.rs` (`7b2d25d`)
> actually contains:
>
> - `a_gguf_is_served_with_its_own_template` — passes, `#[ignore]`d only for
>   needing the real checkpoint.
> - `a_gguf_completion_is_still_garbage_after_correct_templating` — `#[ignore]`d
>   as a recorded, expected-to-fail downstream defect, not a regression.
>
> Task 4 Step 2 below still describes the single-gate version; read it for the
> harness-construction mechanics (`AppState`, the `#![cfg]` line to leave
> behind) and the spec's §9 table for what was actually built.

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

The builder is the load-bearing part of this task. A real 637,699,456-byte file has exactly one metadata set; every row of the spec's failure table needs a *different* one, so the fixtures must be constructed. GGUF v3 layout: magic `GGUF`, `u32` version, `u64` tensor count, `u64` KV count, then KV pairs of `(u64 len + bytes) key`, `u32 value-type`, value. Type tags: `8` = string, `4` = u32, `9` = array (`u32` elem type, `u64` count, elements).

Add to `tests/chat_template_render.rs`:

```rust
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

This **will** break `probe_override_check`'s match with `error[E0004]`. That is
the guard working. **You must repair it in the very next step** — see Step 3a.

- [ ] **Step 3a: Classify the variant immediately, so the crate compiles again**

> **Why this is here and not in a later task.** `probe_override_check` is
> **library** code (`src/api/chat_template.rs:1108`), not a test. While its match
> is non-exhaustive the *whole crate* fails to build, which means **every test
> target fails to build** — `cargo test` prints `error[E0004]`, not test counts.
> An earlier draft of this plan deferred the fix by two tasks while instructing
> you to run tests and commit in between. It would have had you commit a
> non-compiling tree twice and read a compile error as a broken test.

In `probe_override_check`, add `GgufMetadata` to the **same arm as
`TokenizerConfig`**. Find that arm — it is its own arm, not shared with
`Sidecar` — and change its pattern to:

```rust
        Resolution::TokenizerConfig | Resolution::GgufMetadata => {
```

Leave the arm's body **exactly as it is** for now, including its message. The
message currently names `Resolution::TokenizerConfig` unconditionally, so for a
GGUF it will now say the wrong tier. **That is deliberate and Task 3 fixes it**
— Task 3's test exists to catch precisely this, and it needs a red to catch.

Classification rationale (refuse, not warn): the checkpoint declares its own
template, so a probe sidecar would outrank the author's declaration. `--force`
still overrides. Same reasoning as `TokenizerConfig`, which is why they share
an arm.

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
/// **Cost, stated accurately.** Mapping is virtual, but `Content::read`
/// materialises the *entire* metadata block eagerly — on the real TinyLlama
/// Q4_0 that is `tokenizer.ggml.tokens` as 32 000 heap `String`s, plus the
/// parallel score and token-type arrays. And this runs **twice per start**, not
/// once: `resolve_for_model` (`:770-775`) calls `resolve` and `special_tokens`,
/// and each calls this. Not per request, and not a correctness problem — but if
/// startup latency ever matters, memoise here rather than re-deriving why it is
/// slow.
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
            // {0,1,2,3,6..=15,30} — 15 codes. Every IQ code is absent: IQ4_NL
            // (20), IQ3_S (21), IQ4_XS (23) among them, all ordinary
            // quantizations llama.cpp reads happily. On such a file the chat
            // template is decoded, sits in memory, and is discarded because of
            // a tensor we never needed.
            //
            // We COULD avoid this. fuel-core re-exports the primitives —
            // `VersionedMagic::read`, `read_string`, `ValueType::from_u32`,
            // `Value::read` are all pub (`gguf_file.rs:19-21`) — so a
            // metadata-only reader that stops before the tensor directory is
            // ~25 lines at the current pin. We CHOSE not to own a hand-rolled
            // parser for a case our checkpoints do not hit. Recorded as a
            // decision so the next person does not re-derive it as an
            // impossibility. What is missing upstream is a convenience
            // function, not a capability.
            //
            // Do NOT say "not a valid GGUF" — for an IQ file that is a false
            // statement about the operator's file, and the fall-through lands
            // on a family guess with no end-of-turn marker, which is the exact
            // defect this epic fixes. Do NOT say "does not YET decode" either:
            // codes 4 and 5 (Q4_2, Q4_3) are also rejected and are *retired*,
            // not future — ggml has eight retired codes. "Not yet" sends an
            // operator with an old file looking forward when they should look
            // back. Name the symptom, not a guess at the era.
            //
            // Counting note: the 20-omitted figure is 35 live codes minus these
            // 15, per MLMF's cross-check against llama.cpp's own static_asserts
            // (27 of 27 block sizes, zero mismatches). Retired codes are NOT in
            // that 35, so they are additional to the 20 rather than part of it —
            // an earlier draft said "omits 20 ... and retired ones", which was
            // arithmetically incoherent.
            tracing::warn!(
                "{}: Fuel could not open this GGUF ({e}); falling through to the \
                 file-based tiers. If the file is otherwise sound, the likely cause \
                 is a tensor quantization outside Fuel's ggml type table, which \
                 covers 15 codes and omits 20 of the 35 currently in use (the IQ* \
                 and TQ* families) as well as retired ones such as Q4_2 and Q4_3. \
                 The chat template may be present and readable but is unreachable \
                 until that table covers this file.",
                model_path.display()
            );
            return None;
        }
    };
    let md = mc.metadata();

    // `to_string()` here returns `Result<&String>` — it is Fuel's accessor, not
    // `Display::to_string`. Cloning is deliberate: the mmap is dropped with `mc`.
    //
    // The `match` is NOT decoration. Written as
    // `.and_then(|v| v.to_string().ok())`, a template that is PRESENT but not a
    // string (an array, an integer) collapses into the same `None` as a template
    // that is ABSENT — identical in the return type AND in the log, since only
    // `resolve`'s one `debug!` would fire. A GGUF declaring its template as an
    // array would then emit "no usable chat template" and fall to a family guess
    // with no end-of-turn marker: this epic's own failure mode, inside its fix.
    // Absent is benign and common; present-but-undecodable is a malformed file
    // and must say so.
    let template = match md.get("tokenizer.chat_template") {
        None => None,
        Some(v) => match v.to_string() {
            Ok(s) => non_blank(
                s.clone(),
                &format!("{} (GGUF metadata)", model_path.display()),
            ),
            Err(e) => {
                tracing::warn!(
                    "{}: GGUF tokenizer.chat_template is present but not a string \
                     ({e}); treating the file as declaring no template. This is a \
                     malformed checkpoint, not a normal one.",
                    model_path.display()
                );
                None
            }
        },
    };

    // Byte-exact: the token is rendered into prompt TEXT and the tokenizer must
    // recognise the identical bytes. Never trim or normalize here.
    let tokens = md.get("tokenizer.ggml.tokens").and_then(|v| v.to_vec().ok());
    let lookup = |key: &str| -> Option<String> {
        // `to_u64()`, NOT `to_u32()`. Fuel's `to_u32` accepts ONLY the `U32`
        // tag (`fuel-formats/src/gguf.rs:227`) and errors on `I32`/`U64`/`U16`,
        // while `to_u64` (`:242`) upcasts from U8/U16/U32/U64/Bool. With
        // `to_u32().ok()?` a token id stored as any other integer type yields
        // `None` with NO log, `out.bos`/`out.eos` stay empty, and
        // `special_tokens`' tier-3 warning then tells the operator the
        // checkpoint "declares no BOS or EOS token" — which is false. It
        // declared one; we failed to read its type. Our fixture is safe (both
        // ids verified as tag 4 on the real file), but this costs nothing and
        // removes the class.
        let id = usize::try_from(md.get(key)?.to_u64().ok()?).ok()?;
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

- [ ] **Step 6: Consult it in `special_tokens` — AND guard the tier below it**

> **⚠ SCOPE EXPANSION, deliberate and approved — with its blast radius stated
> accurately.** This step edits `special_tokens`' existing tier 1, which
> **predates this epic**.
>
> **It changes no existing behaviour.** An earlier draft of this box claimed it
> "also affects directory checkpoints that have a `tokenizer_config.json`
> without `bos_token`/`eos_token` keys". That set is empty, and this step's own
> closing paragraph says why: nothing writes `out.bos`/`out.eos` before tier 1
> except block (a) below, which returns early for anything that is not an
> existing `.gguf` file. So for every directory checkpoint `out.bos.is_empty()`
> is unconditionally true and the guarded assignment is identical to the
> unguarded one. The claim was a false justification bolted onto a correct
> instruction — the same defect shape this epic is about, and the reason it is
> corrected here rather than quietly dropped.
>
> The real reason for the expansion stands: without it, this epic ships a
> feature that is silently defeated whenever a `.gguf` sits beside a
> `tokenizer_config.json` lacking those keys — and a feature that is silently
> defeated is worse than an absent one, because it reads as working.

**Insert-before is not sufficient on its own. Read this before writing code.**

The existing tier 1 assigns **unconditionally**, with `unwrap_or_default()`:

```rust
    // Tier 1 — the checkpoint's own declaration.        <-- src/api/chat_template.rs:684-690
    if let Ok(raw) = std::fs::read_to_string(meta.join("tokenizer_config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            out.bos = token_field(&v, "bos_token").unwrap_or_default();
            out.eos = token_field(&v, "eos_token").unwrap_or_default();
        }
    }
```

So for a `.gguf` sitting beside **any** `tokenizer_config.json`, inserting the
GGUF block above this writes `<s>`/`</s>` and then this block **overwrites
them** — and if that JSON has no `bos_token`/`eos_token` key (the common case
for a JSON shipped next to a GGUF) it overwrites them with `""`. Tier 2 below
cannot repair it, because it needs a `tokenizer.json` that usually is not there.
The result is the §1 defect: a template interpolating `eos_token` renders with
nothing there.

Note that **tier 2 immediately below is already guarded** (`if out.bos.is_empty()
|| out.eos.is_empty()`) and documents the per-token rule. Tier 1 is the only
tier missing it. You are applying the established pattern to the tier that lacks
it, not inventing one.

**(a)** Insert **before** the existing tier-1 block:

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

**(b)** Then change the existing tier-1 assignments to fill only what is still
empty, matching tier 2's rule:

```rust
    // Tier 1 — the checkpoint's own declaration.
    //
    // Per-token and only-if-empty, matching tier 2 below. This is NOT
    // cosmetic: these two lines were unconditional, so a GGUF whose own
    // metadata had already supplied `<s>`/`</s>` would have them overwritten
    // here — with `""` whenever the companion JSON omits the key, which is
    // usual for a JSON shipped beside a `.gguf`. An empty EOS renders a
    // template with no end-of-turn marker, which is the defect this module
    // exists to prevent.
    if let Ok(raw) = std::fs::read_to_string(meta.join("tokenizer_config.json")) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            if out.bos.is_empty() {
                out.bos = token_field(&v, "bos_token").unwrap_or_default();
            }
            if out.eos.is_empty() {
                out.eos = token_field(&v, "eos_token").unwrap_or_default();
            }
        }
    }
```

This is behaviour-preserving for every directory checkpoint that reaches here
with empty tokens, which is all of them today — nothing writes `out.bos` before
tier 1 except the block you just added in (a).

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
- Produces: `gguf_fixture_with_tensor`, `capture_logs` (test-local helpers, Step 0).

Spec §7 lists **eight** conditions — `1740582` added the "valid GGUF that Fuel
refuses to open" row and this plan's count was not updated with it. Task 1
covered the happy path only. **Each test below must fail against a different
wrong implementation** — check that as you write them.

- [ ] **Step 0: Two helpers the tests below need, neither of which exists yet**

Both go in `tests/chat_template_render.rs` beside `gguf_fixture`.

**(a) A fixture with one tensor**, so the unsupported-dtype row is reachable.
Task 1's `gguf_bytes` already takes a `tensor_count`; every fixture so far has
passed `0`, and a zero-tensor file never reaches `GgmlDType::from_u32` at all.

```rust
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
```

**(b) In-process log capture.** The suite has none. Both existing log
assertions (`:2252`, `:2284`) shell out to the probe binary and read its
stdout, which is far too heavy for a unit test and cannot reach `resolve`
directly. Verified there is no helper to reuse and no name collision:
`grep -n "tracing\|subscriber\|capture_logs" tests/chat_template_render.rs`
returns only those stdout lines.

```rust
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
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Buf {
        type Writer = Buf;
        fn make_writer(&'a self) -> Buf { self.clone() }
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
```

**No manifest change is needed.** `tracing-subscriber = { version = "0.3",
features = ["env-filter"] }` is declared at `Cargo.toml:133` under
`[dependencies]` (not `[dev-dependencies]`), and Cargo makes normal dependencies
available to integration tests as well as dev-dependencies. Verified rather than
recalled: `serde_json` is likewise `[dependencies]`-only (`Cargo.toml:126`) and
`tests/api_integration_tests.rs`, `tests/chat_template_e2e.rs` and others already
call `serde_json::` directly.

The `fmt` feature is on by default, so `tracing_subscriber::fmt()` and the
`MakeWriter` trait are both in scope.

- [ ] **Step 1: Write the tests**

```rust
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
        &[("tokenizer.ggml.tokens", Kv::StrArray(vec!["<unk>", "<s>", "</s>"]))],
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
    let p = gguf_fixture("gguf-wrong-type", &[("tokenizer.chat_template", Kv::U32(7))]);
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

Expected: `running 69 tests` / `67 passed; 0 failed; 2 ignored`. (58 baseline
+ 1 from Task 1 + 10 here.)

- [ ] **Step 3: Verify each test discriminates**

For each mutation below, apply it, run, note which test fails, restore, **and `touch src/api/chat_template.rs`** — restoring bytes is not enough, cargo reuses the mutant binary.

| mutation | must fail |
| --- | --- |
| `non_blank(...)` → `Some(s)` in `read_gguf_declaration` | `a_blank_gguf_template_is_unusable_and_falls_through` |
| drop the `is_file()`/extension check | `a_directory_checkpoint_is_not_run_through_the_gguf_reader` |
| move the GGUF branch *after* the `tokenizer_config.json` block in `resolve` | `gguf_metadata_outranks_a_companion_tokenizer_config` |
| `toks.get(id)` → `toks.get(id.min(toks.len()-1))` | `an_out_of_range_token_id_leaves_the_token_empty` |
| `.to_string()` → `.and_then(\|v\| v.to_string().ok())` (drop the wrong-type warn) | `a_non_string_gguf_template_warns_rather_than_reading_as_absent` |
| the `Err` warn loses its "quantization" wording | `a_gguf_with_an_unsupported_tensor_dtype_falls_through_and_says_why` |

Report any mutation that does **not** kill a test — that is a coverage hole, not a formality.

> **Every row of this table is a HYPOTHESIS, not a fact. Run it.** An audit
> proved one row wrong by compiling a scratch binary rather than reasoning about
> it: the row claiming that dropping the `is_file()` guard kills a named test was
> false, because `File::open` on a directory fails and the fall-through still
> produces the asserted result. That guard would have shipped with **zero
> coverage while this table recorded it as covered** — a table asserting kills
> that do not happen is worse than no table, because it converts an untested
> branch into a documented one.
>
> The standard, applied elsewhere in this repo (`d1eca62`): apply the mutation,
> watch the named test go **red**, note which other tests stayed green **and
> why**, restore, `touch` the file, confirm green again. A row you did not run is
> a row you have not verified, whoever wrote it.

> **Row 2 needs a test that does not exist yet — write it.** An earlier draft
> named `a_directory_checkpoint_still_resolves_from_tokenizer_config` here, and
> that mutation does **not** kill it. Measured: dropping the guard sends a
> directory into `MmapedContent::from_path`, whose first act is `File::open`,
> which on Windows fails on a directory with `Access is denied. (os error 5)`
> (on Linux the open succeeds and `Mmap::map` fails). Either way
> `read_gguf_declaration` returns `None`, `resolve` falls to tier 1, and every
> assertion in that test still holds. The guard would have shipped with zero
> coverage while this table recorded it as covered.
>
> **The observable difference is the WARN, not the resolution.** Without the
> guard, every directory-checkpoint start emits "Fuel could not open this GGUF …
> the likely cause is a tensor quantization outside Fuel's ggml type table" —
> about a directory. Add to Step 1:

```rust
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
```

> `tmp_checkpoint_with_own_template(name: &str) -> PathBuf` already exists at
> `tests/chat_template_render.rs:2187`. It builds a directory containing a
> `tokenizer_config.json` with `bos_token`, `eos_token` and a `chat_template`,
> which is exactly the tier-1 fixture this test needs. Verified, not assumed.

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

> **This task no longer repairs a compile error.** Task 1 Step 3a already
> classified the variant, because `probe_override_check` is library code and
> leaving it unclassified breaks every test target. What is left here is the
> part that classification deliberately got *wrong*: the arm's message is
> hardcoded to name `Resolution::TokenizerConfig`, so a GGUF currently gets a
> refusal that names the wrong tier. This task makes the message truthful, and
> adds the new variant to the two existing enumerations that will silently skip
> it.

- [ ] **Step 1: Write the failing test**

```rust
/// A GGUF that declares its own template must be protected from the probe
/// exactly as a `tokenizer_config.json` is — the probe would otherwise write a
/// registry candidate at `Resolution::Probe`, which `resolve` reads at tier 0,
/// AHEAD of the author's own declaration.
///
/// Asserts the message names `Resolution::GgufMetadata` WITH the `Resolution::`
/// prefix. That is the convention the rest of this file already enforces
/// (`:1943`, `:2158` assert `contains("Resolution::TokenizerConfig")`), and
/// asserting the bare word would be satisfied by a `{current:?}` interpolation
/// that breaks both of those tests.
#[test]
fn the_probe_refuses_to_override_a_gguf_declaration() {
    use lightbulb::api::chat_template::{ProbeOverride, probe_override_check};

    match probe_override_check(Resolution::GgufMetadata, false) {
        ProbeOverride::Refuse(msg) => {
            assert!(
                msg.contains("Resolution::GgufMetadata"),
                "the refusal must name the tier it is protecting, and must not \
                 claim the checkpoint resolves by tokenizer_config.json: {msg}"
            );
            assert!(
                !msg.contains("Resolution::TokenizerConfig"),
                "the refusal names the WRONG tier for a GGUF: {msg}"
            );
            assert!(
                msg.ends_with("Nothing was written."),
                "must end exactly as the other refusals do: {msg}"
            );
        }
        other => panic!("expected Refuse for a checkpoint's own declaration, got {other:?}"),
    }

    // `--force` still overrides, like every other refusal, and the warning must
    // also name the right tier — that is what the loop at `:2149` checks for
    // the other tiers.
    match probe_override_check(Resolution::GgufMetadata, true) {
        ProbeOverride::Warn(msg) => assert!(
            msg.contains("Resolution::GgufMetadata"),
            "--force must record WHAT it overrode: {msg}"
        ),
        other => panic!("--force must downgrade the refusal to a warning, got {other:?}"),
    }
}
```

- [ ] **Step 2: Run to confirm it fails on the assertion**

```bash
cargo test -j 4 --test chat_template_render the_probe_refuses 2>&1 | grep -E "^error|panicked|^test result"; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: a **test failure, not a compile error** — the refusal message says
`Resolution::TokenizerConfig` because Step 3a put `GgufMetadata` in that arm
without touching the text. If you instead see `error[E0004]`, Task 1 Step 3a was
skipped; go back and do it.

- [ ] **Step 3: Make the message name the tier it was actually given**

The arm hardcodes the **string** `Resolution::TokenizerConfig` — twice, once in
its `Warn` body and once in its `Refuse` body. There is no `"tokenizer_config.json"`
literal in this function; do not go looking for one.

**Both bodies are plain string literals ending in `.to_string()`** — they are not
format strings, so you cannot simply insert `{current:?}`. Each looks like:

```rust
                ProbeOverride::Warn(
                    "--force: overriding Resolution::TokenizerConfig — the checkpoint's own \
                     chat_template, …"
                        .to_string(),
                )
```

Convert each to a `format!`, dropping the trailing `.to_string()`:

```rust
                ProbeOverride::Warn(format!(
                    "--force: overriding Resolution::{current:?} — the checkpoint's own \
                     chat template, …"
                ))
```

**Write `Resolution::{current:?}`, not bare `{current:?}`** — the `Debug` for the
enum prints `TokenizerConfig`, and dropping the prefix would break the two
existing tests at `:1943` and `:2158` that assert on the qualified name.

**Watch for literal braces.** Once a body is a format string, any `{` or `}` in
its text must be doubled. Check both bodies before running.

The `Refuse` body also contains the phrase "it ships its own chat_template",
which is `tokenizer_config.json`-shaped wording. Reword to something true of
both sources — e.g. "it carries its own chat template" — since a GGUF ships one
inside the file rather than beside it.

That body ends with "(measured on SmolLM2-360M-Instruct, whose own template
injects a default system message that `registry::CHATML` does not)". Once the
tier is interpolated, that measurement — taken on a **directory** checkpoint —
gets cited inside a **GGUF** refusal. Scope it ("measured on a directory
checkpoint, SmolLM2-360M-Instruct: …") or move it to the `TokenizerConfig`-only
path. Do not leave a measurement attached to a case it was not measured on.

- [ ] **Step 3a: Add the variant to the two enumerations that will skip it silently**

Neither is a compile error; both just quietly cover one fewer case.

- `tests/chat_template_render.rs:1122-1128` — `a_blank_template_is_not_usable_whatever_tier_claims_to_have_found_it` iterates a **hardcoded five-tier array**, and its own comment says *"Every tier, not just one: the backstop is about a tier ADDED later."* `GgufMetadata` is that tier. Add it.
- `tests/chat_template_render.rs:2149-2153` — `force_downgrades_each_refusal_to_a_warning_that_still_names_the_tier` iterates the refusal tiers. Add `GgufMetadata`.

Then fix the now-wrong comments. There are at least **three**, not two — that
count was itself wrong in the previous revision:

- `src/api/chat_template.rs:1094` — "`force` turns the **three** refusals into warnings" → **four**. Step 3a is what makes it four, by adding `GgufMetadata` beside `TokenizerConfig`, `Sidecar` and `Probe`.
- `src/api/chat_template.rs:1063` — the bullet beginning `* [`Resolution::TokenizerConfig`] — refuse.` now describes two variants; name both.
- `src/api/chat_template.rs:1083` — "a sidecar can announce itself as any of the **six**" → seven.
- `tests/chat_template_render.rs:1918` — "**Six** `Resolution` variants times two `force` states is **twelve** cases" → seven and fourteen. **Check that the test below it actually asserts all fourteen**; if it enumerates cases explicitly, add the missing pair rather than only editing the prose.

- [ ] **Step 4: Run**

```bash
cargo test -j 4 --test chat_template_render 2>&1 | grep -E "^error|^running|^test result"; echo "EXIT=${PIPESTATUS[0]}"
cargo build -j 4 --bin lightbulb-probe 2>&1 | grep -E "^error"; echo "EXIT=${PIPESTATUS[0]}"
```

Expected: `running 70 tests` / `68 passed; 0 failed; 2 ignored`, and the probe
binary builds. (58 baseline + 1 from Task 1 + 10 from Task 2 + 1 here. If your
count differs, find out why before continuing — do not adjust the number to
match what you got.)

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

> **Its `parse_dtype` test has already been rescued — do not go looking for it.**
> This file contained the tree's **only** arm-by-arm test of a string→`DType`
> mapping (`:162-164`, `assert!(matches!(parse_dtype(Some("f16")), Ok(DType::F16)))`),
> written correctly against **its own private copy** of the function, in a file
> that has never compiled and never run. So the repo held the right test and
> zero coverage. The live `parse_dtype` in `src/loaders/mod.rs` was given real
> arm-by-arm tests in commit `d1eca62`, **before** this delete, because after it
> the knowledge that those arms need pinning would only be recoverable from a
> removed file. `f16` and `bf16` are both two bytes with different
> exponent/mantissa splits, so a swapped arm passes every width check and errors
> nowhere.

```bash
grep -rn "mlmf_wrapper" src/          # expect: no output
git rm src/loaders/mlmf_wrapper.rs
cargo check -j 4 --all-targets 2>&1 | grep -E "^error"; echo "cargo=${PIPESTATUS[0]}"   # expect: no errors, cargo=0
cargo test -j 4 --lib loaders::tests 2>&1 | grep -E "^running|^test result"; echo "cargo=${PIPESTATUS[0]}"
# expect: running 4 tests / 4 passed — the rescued coverage still there after the delete
```

If the grep prints anything, **stop** — it is referenced and this plan is wrong.

- [ ] **Step 2: Write the gate**

> ### ⚠️ Correction: what follows specifies ONE gate; TWO were built.
>
> This step and its code sample below describe a single `#[ignore]`d test
> asserting the completion, the resolved tier, and BOS/EOS together. That
> design's premise — that those three assertions pass or fail together — was
> falsified during implementation; see spec §1's and §9's correction boxes.
> What Task 4 actually produced (`7b2d25d`) is **two** tests in
> `tests/gguf_serving_e2e.rs`: `a_gguf_is_served_with_its_own_template`
> (passes; `#[ignore]`d only for needing the real checkpoint) and
> `a_gguf_completion_is_still_garbage_after_correct_templating` (`#[ignore]`d
> as a recorded, expected-to-fail downstream defect). The harness mechanics
> below — router/runner setup, `AppState`, and the `#![cfg]` line to leave
> behind — are still accurate; only the "one gate" framing is not.

`tests/gguf_serving_e2e.rs` — model it on `tests/chat_template_e2e.rs` for router/runner setup. **Do not** feature-gate it: this works in the default build, and a `#![cfg(...)]` would make it compile to zero tests while printing `ok. 0 passed`.

> **You are copying from a file that has the attribute you must not copy.**
> `tests/chat_template_e2e.rs:9` opens with `#![cfg(feature = "fuel-engine")]`.
> That is correct *for it* — it exercises the Fuel runner specifically. Yours
> exercises chat-template resolution, which is backend-independent because
> `fuel-core` is an unconditional dependency. Copy the `AppState` construction, which in
> that file lives *inside* its `post_raw` helper (`:71-85`), and **leave the
> `#![cfg]` line behind.** The sample below inlines `app.oneshot(…)` rather than
> reproducing `post_raw`, so take the `AppState` literal from `post_raw`'s body
> and ignore its wrapper.
>
> `ModelRunner::start` exists in both configurations
> (`src/engine/model_runner.rs:225` under the `#[cfg(not(feature = "fuel-engine"))]`
> at `:224`, and `:295` under the `#[cfg(feature = "fuel-engine")]` at `:294`), so the copy compiles
> either way.

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

The harness below is `chat_template_e2e.rs`'s `post_raw` **with the request call
inlined** rather than kept as a helper, and the checkpoint source swapped; **`AppState` needs all six fields** (`eos_monitor` included) or it
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

Expected: `running 1 test` / `1 passed`. Budget several minutes — it loads a 637,699,456-byte model on CPU.

- [ ] **Step 4: Commit**

```bash
rustfmt --edition 2024 tests/gguf_serving_e2e.rs
git add tests/gguf_serving_e2e.rs
git commit -m "test(api): Gate GGUF serving end to end, and delete the MLMF orphan"
```

---

## Final verification

> **The previous version of this gate was a false green.** It read
> `cargo test --tests | grep -cE "^test result: FAILED"  # expect 0`. A tree
> that **fails to compile** emits no `test result:` line at all, so that
> pipeline prints `0` and the gate passes on a broken build — the precise
> failure Global Constraint 3 names, sitting in this plan's own definition of
> done. Every command below therefore reports `${PIPESTATUS[0]}` and is checked
> against a positive expected count, never the absence of a failure string.

- [ ] Full suite, reading counts and cargo's own status:

```bash
cargo test -j 4 --lib 2>&1 | grep -E "^running|^test result"; echo "cargo=${PIPESTATUS[0]}"
# expect: running 661 tests / 647 passed; 0 failed; 14 ignored   cargo=0

cargo test -j 4 --test chat_template_render 2>&1 | grep -E "^running|^test result"; echo "cargo=${PIPESTATUS[0]}"
# expect: running 70 tests / 68 passed; 0 failed; 2 ignored      cargo=0

cargo test -j 4 --tests 2>&1 | grep -E "^running|^test result|^error"; echo "cargo=${PIPESTATUS[0]}"
# expect: cargo=0, and a `running N` line for EVERY test binary.
# A binary that fails stops the ones after it, so a short list is a failure
# signal even when nothing says FAILED.
```

- [ ] Both feature configurations compile:

```bash
cargo check -j 4 --all-targets 2>&1 | grep -E "^error" ; echo "cargo=${PIPESTATUS[0]}"
cargo check -j 4 --all-targets --features fuel-engine 2>&1 | grep -E "^error" ; echo "cargo=${PIPESTATUS[0]}"
# expect: no error lines and cargo=0 for both. `grep` finding nothing exits 1;
# that is grep's status, not cargo's — read the echoed cargo= value.
```

- [ ] The new e2e binary really contains its test (it is `#[ignore]`d, so a
      count of 0 would otherwise look like success):

```bash
cargo test -j 4 --test gguf_serving_e2e -- --list 2>&1 | tail -3; echo "cargo=${PIPESTATUS[0]}"
# expect: "1 test" — NOT "0 tests". Zero means the file got feature-gated.
```

- [ ] `grep -rn "mlmf" src/` returns nothing.

## Deliberately out of scope

GGUF on the Fuel backend (needs the pin bump to `f1da2d94+`), adopting MLMF, retiring `src/gguf/` and the candlelight loaders, and sharded GGUF. See spec §8.
