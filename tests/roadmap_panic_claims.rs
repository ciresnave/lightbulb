//! ROADMAP must not assert a LIVE panic macro at a path where none is live.
//!
//! A correction can outlive the defect it corrects, and then it is a defect. This
//! file exists because that happened here, in one document, twice about the same
//! subject — one instance fixed, the neighbour left standing:
//!
//! ```text
//! ROADMAP.md ~514  "…`PerGroup` granularity WAS a live `todo!(…)` … Fixed
//!                   2026-09-02: it now returns an `Err`."        <- correct
//! ROADMAP.md 3272  "M5 KV cache compression (claimed COMPLETE, but
//!                   `src/cache/kv_compression.rs:446` IS a live `todo!()`
//!                   — see VERIFIED STATUS)"                      <- false
//! ```
//!
//! ⚠️ The false line CITES the block that refutes it. A reader who follows the
//! pointer gets the truth; a reader who stops at the dependency line gets a live
//! panic that does not exist. Measured at `8b0960be`: `src/` contains ZERO live
//! `todo!()` / `unimplemented!()` — both textual occurrences are comments, one of
//! them explaining that the macro *used to* be there.
//!
//! WHAT THIS CHECKS: every ROADMAP sentence claiming a named `.rs` path currently
//! holds a live panic macro must be true of that file. It does not check the
//! reverse — a live `todo!()` ROADMAP is silent about is a different question, and
//! `no_silent_skips.rs` is the model for that kind of sweep if anyone wants it.
//!
//! ⚠️ WHY THE NEGATIVE CASE IS CONSTRUCTED RATHER THAN SAMPLED. The obvious "it
//! fires" test is to point the checker at ROADMAP's own false line — but this
//! change REMOVES that line, so the test would expire by the fix succeeding.
//! `flags_a_false_claim` builds the text in the test body instead, so it keeps
//! working when the document is clean, permanently.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Is this source line a comment? A `todo!()` inside a comment is documentation
/// — frequently documentation *about a todo that was removed*, which is exactly
/// the shape that made the stale claim look plausible.
fn is_comment(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("//") || t.starts_with("/*") || t.starts_with('*')
}

/// Live (non-comment) occurrences of `macro_name!(` in `src`.
fn live_macro_lines(src: &str, macro_name: &str) -> Vec<usize> {
    let needle = format!("{macro_name}!(");
    src.lines()
        .enumerate()
        .filter(|(_, l)| l.contains(&needle) && !is_comment(l))
        .map(|(i, _)| i + 1)
        .collect()
}

/// A ROADMAP sentence asserting that some `.rs` path currently holds a live panic
/// macro. Returns `(line_no, path, macro_name)`.
///
/// Keyed on the present-tense assertion `is a live` / `are live`, never on the
/// macro alone — ROADMAP legitimately discusses removed macros in the past tense,
/// and flagging those would push people to delete the history that explains them.
/// Backticked `path.rs` / `path.rs:NNN` tokens in one line, as repo-relative paths.
///
/// Split out of `present_tense_claims` so the two concerns — *is this sentence a
/// present-tense claim* and *which file does it name* — are separately testable.
/// Codacy flagged the merged version at cyclomatic complexity 10; enumerating the
/// branches showed the path scan really was a second job living inside the first,
/// so this is a split rather than a number-appeasement.
fn backticked_rs_paths(line: &str) -> Vec<String> {
    line.split('`')
        .filter_map(|tok| {
            let c = tok.split(':').next().unwrap_or(tok).trim();
            (c.ends_with(".rs") && c.contains('/')).then(|| c.to_string())
        })
        .collect()
}

/// Does this lowercased line assert a live `mac!` in the present tense?
fn asserts_live(lower: &str, mac: &str) -> bool {
    (lower.contains("is a live") || lower.contains("are live"))
        && lower.contains(&format!("{mac}!"))
}

fn present_tense_claims(roadmap: &str) -> Vec<(usize, String, String)> {
    let mut out = Vec::new();
    for (i, line) in roadmap.lines().enumerate() {
        // ⚠️ Blockquotes are quoted history, not the document's own claim. A
        // DISCHARGED note has to QUOTE the wording it retires or a reader cannot
        // tell what was corrected — so a scan that counts them fires on its own
        // remedy and can only be satisfied by deleting the record. It did exactly
        // that on the first run after this file's own fix was written.
        if line.trim_start().starts_with('>') {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        for mac in ["todo", "unimplemented", "unreachable"] {
            if !asserts_live(&lower, mac) {
                continue;
            }
            for path in backticked_rs_paths(line) {
                out.push((i + 1, path, mac.to_string()));
            }
        }
    }
    out
}

#[test]
fn roadmap_does_not_claim_a_panic_that_is_not_live() {
    let root = repo_root();
    let roadmap_path = root.join("ROADMAP.md");
    let roadmap = std::fs::read_to_string(&roadmap_path)
        .unwrap_or_else(|e| panic!("cannot read {} — {e}", roadmap_path.display()));

    // The corpus must be real. A wrong path reads nothing and passes vacuously.
    assert!(
        roadmap.lines().count() > 500,
        "ROADMAP.md is only {} lines — the path is wrong, and a short file passes \
         every check below having examined nothing",
        roadmap.lines().count()
    );

    let mut violations = Vec::new();
    for (line_no, rel, mac) in present_tense_claims(&roadmap) {
        let src_path = root.join(&rel);
        if !src_path.exists() {
            violations.push(format!(
                "ROADMAP.md:{line_no} names `{rel}`, which does not exist"
            ));
            continue;
        }
        let src = std::fs::read_to_string(&src_path)
            .unwrap_or_else(|e| panic!("cannot read {} — {e}", src_path.display()));
        let live = live_macro_lines(&src, &mac);
        if live.is_empty() {
            let textual = src
                .lines()
                .filter(|l| l.contains(&format!("{mac}!(")))
                .count();
            violations.push(format!(
                "ROADMAP.md:{line_no} says `{rel}` has a live `{mac}!()`, but it has none \
                 ({textual} textual occurrence(s), all in comments)"
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "ROADMAP CLAIMS A PANIC THAT IS NOT LIVE:\n\n  {}\n\n\
         A correction that outlives its defect is a defect. If the macro was removed, \
         say so in the past tense — ROADMAP already does exactly that elsewhere for this \
         same subject, which is how the surviving claim was found.",
        violations.join("\n  ")
    );
}

#[test]
fn flags_a_false_claim() {
    // ⚠️ CONSTRUCTED, not sampled. The document this guards is being cleaned by
    // the same change that adds this test, so a negative case taken from it would
    // expire the moment the fix landed and the guard would silently verify nothing.
    let roadmap = "**Dependencies**: thing (claimed COMPLETE, but \
                   `src/made/up/path.rs:446` is a live `todo!()`)";
    let claims = present_tense_claims(roadmap);
    assert_eq!(
        claims.len(),
        1,
        "the claim parser stopped recognising the shape it exists to find: {claims:?}"
    );
    assert_eq!(claims[0].1, "src/made/up/path.rs");
    assert_eq!(claims[0].2, "todo");
}

#[test]
fn past_tense_is_not_flagged_and_comments_are_not_live() {
    // ROADMAP must stay free to record that a macro WAS there. Flagging history
    // would push an author to delete the explanation rather than keep it.
    let past = "`src/cache/kv_compression.rs`'s `PerGroup` was a live `todo!()` — fixed.";
    assert!(
        present_tense_claims(past).is_empty(),
        "a past-tense sentence was treated as a live claim"
    );

    // ⚠️ REGRESSION: a DISCHARGED note quoting the wording it retires must not be
    // read as a live claim. Without this the guard fires on its own remedy — it
    // did, on the first run after the ROADMAP fix was written.
    let discharged = "> ⚠ **DISCHARGED.** This read *\"`src/a/b.rs:446` is a live \
                      `todo!()`\"* until today.";
    assert!(
        present_tense_claims(discharged).is_empty(),
        "a quoted retraction was counted as the document's own claim"
    );

    // The two split-out halves, now separately testable — the point of the split.
    assert_eq!(
        backticked_rs_paths("see `src/a/b.rs:446` and `src/c.rs`"),
        vec!["src/a/b.rs", "src/c.rs"]
    );
    assert!(
        backticked_rs_paths("no backticked path here, and `notapath` either").is_empty(),
        "a non-path backtick was read as a file"
    );
    assert!(asserts_live("x is a live `todo!()`", "todo"));
    assert!(!asserts_live("x was a live `todo!()`", "todo"));
    assert!(
        !asserts_live("x is a live `todo!()`", "unimplemented"),
        "a claim about one macro was attributed to another"
    );

    // The distinction the whole check rests on.
    assert_eq!(live_macro_lines("    todo!(\"x\");", "todo"), vec![1]);
    assert!(live_macro_lines("    // a `todo!()` here would panic", "todo").is_empty());
    assert!(live_macro_lines("//! it was `todo!(\"grouped\")`, and", "todo").is_empty());
    assert!(live_macro_lines(" * legacy: todo!()", "todo").is_empty());
}

#[test]
fn the_document_this_check_reads_is_where_it_expects() {
    let p: &Path = &repo_root().join("ROADMAP.md");
    assert!(
        p.is_file(),
        "{} is missing — this check has no subject",
        p.display()
    );
}
