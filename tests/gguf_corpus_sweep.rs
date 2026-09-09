//! Every GGUF in a local corpus either rebuilds a tokenizer or refuses clearly.
//!
//! # Why this exists
//!
//! The tokenizer rebuild in `Content::extract_tokenizer` was verified against
//! ONE checkpoint. That is the shape of defect this repo keeps finding: a
//! sample generalised to a class. This sweeps whatever corpus is pointed at and
//! asserts the only property that must hold for ALL of them —
//!
//! **every file either produces a tokenizer or fails with an error naming the
//! reason. Never a panic, and never a silently fabricated tokenizer.**
//!
//! It deliberately does NOT assert that every file loads. Refusal is a correct
//! outcome for shapes this code does not support (`gpt2`-model GGUFs whose
//! splitting rule is unverified, and SentencePiece checkpoints carrying scores
//! but no merges and no oracle), and asserting universal success would force
//! exactly the approximation that was measured and rejected — see
//! `src/gguf/mod.rs`.
//!
//! # What it asserts about the COUNT, and why that needed adding (issue #80)
//!
//! ⚠️ **`ok + refused == files.len()` is a CONSERVATION law, not a COVERAGE
//! one.** It is invariant under every redistribution between its two terms: if
//! a change silently broke phi-3's rebuild, `ok` would fall by one, `refused`
//! would rise by one, the sum would still equal the file count, and this test
//! would print the lower number and pass. The figure quoted in ROADMAP.md and
//! in PR bodies was **measured and reported, never defended** — it could not
//! fail, it could only age.
//!
//! The fix is not a floor. `assert!(ok >= 23)` is wrong because the corpus is
//! environment-dependent via `LIGHTBULB_GGUF_CORPUS`, so a different corpus
//! legitimately yields a different count and the floor would fail for a reason
//! that is not a defect.
//!
//! Instead each file states an **obligation** derived from its own metadata
//! (see [`Obligation`]), and the test asserts BOTH directions:
//!
//! - every obligated file rebuilt — catches a rebuild that stopped happening
//! - every rebuild was obligated — catches the obligation table going stale
//!   relative to what the code actually supports
//!
//! Run:
//! ```text
//! LIGHTBULB_GGUF_CORPUS=<dir> cargo test --test gguf_corpus_sweep -- --ignored --nocapture
//! ```

use lightbulb::gguf::{Content, Value};
use std::path::PathBuf;

/// Returns the `.gguf` files found, plus any directories that could not be
/// read — reported rather than silently skipped, since a partial walk would
/// shrink the corpus this test's claim rests on.
fn corpus() -> Option<(Vec<PathBuf>, Vec<String>)> {
    let root = PathBuf::from(std::env::var_os("LIGHTBULB_GGUF_CORPUS")?);
    let mut out = Vec::new();
    let mut unreadable = Vec::new();
    let mut stack = vec![root];
    while let Some(d) = stack.pop() {
        // An unreadable directory must NOT abort the walk. It previously used
        // `read_dir(&d).ok()?`, so one permission error returned `None` from the
        // middle of the scan and the caller then reported "set
        // LIGHTBULB_GGUF_CORPUS" -- a confident diagnosis of a different
        // failure. Worse, a PARTIAL walk would silently shrink the corpus this
        // test's whole claim rests on.
        let entries = match std::fs::read_dir(&d) {
            Ok(e) => e,
            Err(e) => {
                unreadable.push(format!("{}: {e}", d.display()));
                continue;
            }
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
            {
                out.push(p);
            }
        }
    }
    out.sort();
    Some((out, unreadable))
}

/// What a corpus file OWES, derived from its own metadata rather than from a
/// remembered count.
///
/// ⚠️ **This is the lookup the sweep was missing.** The sweep ENUMERATES what
/// is present and counts outcomes; an enumeration cannot see a key that is
/// absent, so a rebuild that stopped happening simply moved a tally from one
/// column to the other. An obligation is keyed on the file's OWN CONTENT, so it
/// holds on any corpus — including one containing none of these files.
#[derive(Debug, PartialEq, Eq)]
enum Obligation {
    /// `llama` model carrying its own `tokenizer.ggml.merges`. Nothing is
    /// derived and nothing is guessed, so a refusal here is a plain defect.
    SpmDeclared,
    /// `llama` model with NO merges, whose vocabulary digest is on the
    /// derivation allowlist. The merges are reconstructed from the token list,
    /// and the warrant is the evidence that this exact vocabulary's
    /// reconstruction was checked against a file carrying real ones.
    SpmDerived,
    /// `gpt2` model whose `tokenizer.ggml.pre` names a splitting rule this
    /// build has verified id-for-id against a reference.
    ByteLevelVerifiedPre(String),
}

/// The obligation a file carries, or `None` if this build promises it nothing.
///
/// `None` covers the shapes that are CORRECTLY refused — a `bert`/`t5`/`gemma`
/// tokenizer, a `gpt2` file whose `pre` is unverified or absent, an SPM
/// vocabulary with no oracle. Those must still refuse with a reason, which is
/// this sweep's original property and is unchanged.
fn obligation(c: &Content) -> Option<Obligation> {
    let s = |k: &str| match c.metadata().get(k) {
        Some(Value::String(v)) => Some(v.clone()),
        _ => None,
    };
    match s("tokenizer.ggml.model")?.as_str() {
        "llama" => {
            // Asked through the production accessor, never re-derived here: a
            // second implementation of "is this warranted" would drift, and its
            // drift would surface as an obligation firing on a healthy file.
            if c.derived_merge_warrant().is_some() {
                Some(Obligation::SpmDerived)
            } else if c.metadata().contains_key("tokenizer.ggml.merges") {
                Some(Obligation::SpmDeclared)
            } else {
                // SPM, no merges, no warrant -- the refusal path. Not owed.
                None
            }
        }
        "gpt2" => {
            let pre = s("tokenizer.ggml.pre")?;
            Content::verified_pre_values()
                .contains(&pre.as_str())
                .then_some(Obligation::ByteLevelVerifiedPre(pre))
        }
        _ => None,
    }
}

#[test]
#[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
fn every_gguf_either_rebuilds_a_tokenizer_or_refuses_with_a_reason() {
    let (files, unreadable_dirs) =
        corpus().expect("set LIGHTBULB_GGUF_CORPUS to a directory containing .gguf files");
    assert!(
        unreadable_dirs.is_empty(),
        "the corpus walk was incomplete, so any count below understates it: {unreadable_dirs:?}"
    );
    // ⚠️ AN EMPTY CORPUS MUST NOT PASS.
    //
    // The assertions at the end are `silent.is_empty()` and
    // `ok + refused == files.len()`. Over zero files those are
    // `[] .is_empty()` and `0 + 0 == 0` — BOTH SATISFIED. Measured
    // 2026-09-02 by pointing `LIGHTBULB_GGUF_CORPUS` at an empty directory:
    // "0 files: 0 rebuilt, 0 refused" and `test result: ok`.
    //
    // So a typo in the variable, or a moved corpus, produced a green gate that
    // examined nothing — and this is the gate whose "30 files: N rebuilt"
    // figure is quoted in ROADMAP.md as the measure of GGUF support.
    assert!(
        !files.is_empty(),
        "no .gguf files under {:?}. An empty corpus satisfies every assertion below, so it must fail here instead of reporting success over nothing.",
        std::env::var("LIGHTBULB_GGUF_CORPUS").unwrap_or_default()
    );

    let mut ok = 0usize;
    let mut refused = 0usize;
    let mut silent = Vec::new();
    // A file that owed a rebuild and did not deliver one. THE regression this
    // sweep previously could not see.
    let mut broken = Vec::new();
    // A file that rebuilt without owing anything — the obligation table has
    // gone stale relative to what the code supports.
    let mut unexplained = Vec::new();
    let mut spm_declared = 0usize;
    let mut spm_derived = 0usize;
    let mut byte_level = 0usize;

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        // `Content::read` takes `AsRef<Path>`, so the `PathBuf` goes straight in.
        // `path.to_str().unwrap()` panicked on a non-UTF-8 path -- in the test
        // whose entire assertion is that this code never panics.
        let content = match Content::read(path) {
            Ok(c) => c,
            Err(e) => {
                // Not a readable GGUF at all — still an error with a reason,
                // which is the property under test. An unreadable file owes
                // nothing: its metadata was never available to owe from.
                println!("  {name:<38} UNREADABLE  {e}");
                refused += 1;
                continue;
            }
        };
        let owed = obligation(&content);
        match &owed {
            Some(Obligation::SpmDeclared) => spm_declared += 1,
            Some(Obligation::SpmDerived) => spm_derived += 1,
            Some(Obligation::ByteLevelVerifiedPre(_)) => byte_level += 1,
            None => {}
        }
        match content.extract_tokenizer() {
            Ok(tok) => {
                println!(
                    "  {name:<38} OK          vocab={} owed={owed:?}",
                    tok.get_vocab_size(true)
                );
                if owed.is_none() {
                    unexplained.push(name.clone());
                }
                ok += 1;
            }
            Err(e) => {
                let msg = e.to_string();
                // A refusal must SAY something. An empty or generic error is the
                // failure mode this test exists to catch: it is indistinguishable
                // from a shrug, and sends the next reader to the wrong place.
                if msg.trim().len() < 40 {
                    silent.push((name.clone(), msg.clone()));
                }
                if let Some(o) = owed {
                    broken.push((name.clone(), o, msg.clone()));
                }
                println!(
                    "  {name:<38} REFUSED     {}",
                    msg.lines().next().unwrap_or("")
                );
                refused += 1;
            }
        }
    }

    let obligations = spm_declared + spm_derived + byte_level;
    println!("\n  {} files: {ok} rebuilt, {refused} refused", files.len());
    println!(
        "  {obligations} obligated: {spm_declared} spm-declared, {spm_derived} spm-derived, {byte_level} byte-level-verified-pre"
    );

    assert!(
        silent.is_empty(),
        "refusals that do not explain themselves: {silent:?}"
    );

    // ⚠️ THE ASSERTION THE REPORTED COUNT NEVER HAD. A file whose own metadata
    // says this build supports it must rebuild; a refusal there is a defect no
    // matter how well it explains itself.
    assert!(
        broken.is_empty(),
        "files that OWED a rebuild and refused -- this is the regression `ok + refused == total` is blind to: {broken:#?}"
    );

    // ⚠️ AND THE OBLIGATION SET MUST NOT BE EMPTY. Every assertion above is
    // satisfied by a corpus that obligates nothing, which is how a guard dies:
    // not by breaking, but by running out of subjects while still reporting
    // green. Same failure this file already guards against for `files`.
    assert!(
        obligations > 0,
        "{} files and not one carries an obligation, so the two assertions above examined nothing. Either the corpus holds no supported checkpoint, or `obligation()` has stopped recognising the ones it does.",
        files.len()
    );

    // The other direction: a rebuild nobody required means `obligation()` no
    // longer describes what the code supports, so the count above is defended
    // only in part. Naming the files is the whole remedy -- add their family.
    assert!(
        unexplained.is_empty(),
        "these rebuilt without owing anything, so `obligation()` is stale and does not cover the reported count: {unexplained:?}"
    );

    assert_eq!(
        ok + refused,
        files.len(),
        "some file neither rebuilt nor refused"
    );
}
