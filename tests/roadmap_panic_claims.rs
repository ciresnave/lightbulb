//! No document may assert a LIVE panic macro at a path where none is live.
//!
//! A correction can outlive the defect it corrects, and then it is a defect. This
//! file exists because that happened here — to ONE subject, in FOUR places, of
//! which one was right:
//!
//! ```text
//! ROADMAP.md ~514      "…`PerGroup` granularity WAS a live `todo!(…)` … Fixed
//!                       2026-09-02: it now returns an `Err`."          <- correct
//! ROADMAP.md 3272      "…`src/cache/kv_compression.rs:446` IS a live
//!                       `todo!()` — see VERIFIED STATUS"               <- false
//! src/model_fuel/policies.rs 24
//!                      "…gate-zero failure. `kv_compression.rs:446` IS a
//!                       live `todo!(…)`…"                              <- false
//! docs/superpowers/notes/capability-regression-audit.md 382
//!                      "Gate zero across all of `src/`: exactly ONE
//!                       `todo!()`/`unimplemented!()` — the known …
//!                       in `kv_compression.rs:446`."                   <- false
//! ```
//!
//! Measured at `a2aa2feb`: `src/` holds ZERO live `todo!` / `unimplemented!` /
//! `unreachable!`, and line 446 today is the brace closing the `PerChannel` arm.
//!
//! # ⚠️ Why this scans PARAGRAPHS and not lines — the first version missed two of three
//!
//! The predecessor of this file was line-anchored and read `ROADMAP.md` alone. It
//! caught exactly one instance, and **that was a property of the file's
//! formatting, not of the check**: ROADMAP writes a paragraph on one physical
//! line, so the path and the macro token happened to sit together. Both missed
//! instances straddle a hard wrap —
//!
//! ```text
//! policies.rs:24   "…`kv_compression.rs:446` is a live"        <- path, no macro
//! policies.rs:25   "`todo!(\"Grouped quantization …\")`, and"   <- macro, no path
//!
//! audit.md:382     "…exactly one `todo!()`/`unimplemented!()`"  <- macro, no path
//! audit.md:383     "…in `kv_compression.rs:446`."               <- path, no macro
//! ```
//!
//! Max line length: `ROADMAP.md` 501, the audit note 300, `policies.rs` 122. **A
//! line-anchored claim scanner silently measures the author's wrap width.**
//!
//! # ⚠️ And why the CITATION predicate was measured, then dropped
//!
//! The obvious strengthening is to anchor on the *structure of the evidence* — a
//! `<file>.rs:<line>` reference beside a macro token — rather than on the wording
//! of the claim. A census over 264 files reported it as a strict superset of the
//! wording rule with **zero** false positives, so it was built. Run against the
//! tree it produced **eleven**.
//!
//! **The census was narrower than the guard built from it**: it only reported
//! citations containing a `/`, so the over-attribution it would have shown was
//! invisible in its own output. And the defect it hid is not fixable by tightening
//! —
//!
//! ⚠️ **A CITATION-ANCHORED PREDICATE IS POLARITY-BLIND.** *"`src/` contains ZERO
//! live `todo!` … `kv_compression.rs:450` and `policies.rs:25`"* — this file's own
//! header, stating the exact opposite of a liveness claim — carries a citation and
//! a macro token and is indistinguishable from the claim it denies. It is stronger
//! at FINDING candidate sentences and strictly weaker at JUDGING them, and no
//! amount of scoping fixes that.
//!
//! So the wording rule stands, with paragraph joining doing the work the citation
//! rule was meant to do. **The consequence is stated rather than hidden: the
//! `audit.md:382` shape — a COUNT of live macros naming a file — is not covered.
//! It was fixed by hand in the same change.** A count check is worth building; it
//! is a different check, and bolting a fuzzy one onto this would trade a precise
//! guard for a noisy one.
//!
//! # What suppresses a match, and why each one has to
//!
//! Blockquotes, fenced blocks, and paragraphs marked `DISCHARGED`. All three are
//! quotation rather than assertion: **a retraction has to QUOTE the wording it
//! retires or a reader cannot tell what was corrected**, and this file's own
//! header has to show the false lines it exists to catch. A scan that counts them
//! fires on its own remedy and can only be satisfied by deleting the record. The
//! predecessor did exactly that on its first run; the `DISCHARGED` case was
//! predicted by the lightbulb lane before this version was run, against their own
//! in-flight fix, which writes the retired wording into a `//!` line that no
//! blockquote rule would reach.
//!
//! WHAT THIS DOES NOT CHECK: the reverse direction (a live `todo!()` the docs are
//! silent about — `no_silent_skips.rs` is the model for that sweep), and line-ref
//! DRIFT (`:1070` → `:1090`). Drift is a currency problem; flagging it would bury
//! the falsity signal in noise.
//!
//! ⚠️ WHY THE NEGATIVE CASES ARE CONSTRUCTED RATHER THAN SAMPLED. The obvious "it
//! fires" test is to point the checker at a real false line — but this change and
//! its sibling REMOVE all of them, so such a test would expire by the fix
//! succeeding. `flags_a_false_claim` builds its text in the test body, so it keeps
//! working when the tree is clean, permanently.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

const LIVE_PHRASES: [&str; 2] = ["is a live", "are live"];
const MACROS: [&str; 3] = ["todo", "unimplemented", "unreachable"];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Directories that are not authored content.
fn is_skipped_dir(name: &str) -> bool {
    matches!(name, "target" | ".git" | "node_modules" | ".venv" | "dist")
}

fn collect(dir: &Path, want: &dyn Fn(&Path) -> bool, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if !is_skipped_dir(&e.file_name().to_string_lossy()) {
                collect(&p, want, out);
            }
        } else if want(&p) {
            out.push(p);
        }
    }
}

fn has_ext(p: &Path, ext: &str) -> bool {
    p.extension().is_some_and(|e| e == ext)
}

/// Every authored document: markdown anywhere, plus the doc comments of every
/// Rust source. **A doc comment is both a comment and a document**, and the
/// version of this check that read only the comment half missed the worst of the
/// three instances — it appeared in that version's own positive control, was
/// correctly classified as "a comment", and was never asked whether it was also
/// making a claim.
fn corpus_paths(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect(root, &|p| has_ext(p, "md"), &mut out);
    for sub in ["src", "tests", "benches", "examples"] {
        collect(&root.join(sub), &|p| has_ext(p, "rs"), &mut out);
    }
    out.sort();
    out
}

/// Is this source line a comment? A `todo!()` inside a comment is documentation
/// — frequently documentation *about a todo that was removed*, which is exactly
/// the shape that made the stale claims look plausible.
fn is_comment(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("//") || t.starts_with("/*") || t.starts_with('*')
}

/// Live (non-comment) occurrences of `macro_name!(` in a Rust source.
fn live_macro_lines(src: &str, macro_name: &str) -> Vec<usize> {
    let needle = format!("{macro_name}!(");
    src.lines()
        .enumerate()
        .filter(|(_, l)| l.contains(&needle) && !is_comment(l))
        .map(|(i, _)| i + 1)
        .collect()
}

/// `(line_number, text)` for every line, with non-document lines blanked.
///
/// For markdown that is every line; for Rust it is the `//!` / `///` runs with
/// their markers stripped.
fn document_lines(path: &Path, body: &str) -> Vec<(usize, String)> {
    let rust = has_ext(path, "rs");
    body.lines()
        .enumerate()
        .map(|(i, raw)| {
            let t = raw.trim();
            let doc = t.starts_with("//!") || t.starts_with("///");
            let text = match (rust, doc) {
                (true, true) => t[3..].trim().to_string(),
                (true, false) => String::new(),
                (false, _) => raw.to_string(),
            };
            (i + 1, text)
        })
        .collect()
}

/// Blank out fenced code blocks and blockquotes — see the header for why both
/// have to be, and why a guard without this can only be satisfied by deleting the
/// history that explains it.
fn strip_quotations(lines: Vec<(usize, String)>) -> Vec<(usize, String)> {
    let mut fenced = false;
    lines
        .into_iter()
        .map(|(n, text)| {
            let t = text.trim_start();
            if t.starts_with("```") {
                fenced = !fenced;
                return (n, String::new());
            }
            if fenced || t.starts_with('>') {
                return (n, String::new());
            }
            (n, text)
        })
        .collect()
}

/// Runs of consecutive non-blank lines, joined, keyed by their first line number.
fn paragraphs(lines: Vec<(usize, String)>) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut buf: Vec<String> = Vec::new();
    let mut start = 0usize;
    for (n, text) in lines {
        if text.trim().is_empty() {
            if !buf.is_empty() {
                out.push((start, buf.join(" ")));
                buf.clear();
            }
            continue;
        }
        if buf.is_empty() {
            start = n;
        }
        buf.push(text);
    }
    if !buf.is_empty() {
        out.push((start, buf.join(" ")));
    }
    out
}

/// `*.rs` tokens named in backticks, as written. A citation may be repo-relative
/// (`src/cache/kv_compression.rs`) or a bare file name (`kv_compression.rs:446`);
/// resolution is a separate concern, deliberately.
fn cited_rs_files(text: &str) -> Vec<String> {
    text.split('`')
        .filter_map(|tok| {
            let c = tok.split(':').next().unwrap_or(tok).trim();
            (c.ends_with(".rs") && c.len() > 3).then(|| c.to_string())
        })
        .collect()
}

/// The file a "… is a live `mac!` …" assertion is ABOUT.
///
/// ⚠️ The nearest citation BEFORE the phrase, falling back to the first after it.
/// Taking every `.rs` token in the paragraph instead attributed one claim to five
/// files, because a bullet list with no blank lines between items is one
/// paragraph — measured, not supposed.
fn claimed_file(text: &str, mac: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    if !lower.contains(&format!("{mac}!")) {
        return None;
    }
    let at = LIVE_PHRASES.iter().find_map(|p| lower.find(p))?;
    cited_rs_files(&text[..at])
        .pop()
        .or_else(|| cited_rs_files(&text[at..]).into_iter().next())
}

/// Is this paragraph a record of a RETIRED claim rather than a live one?
///
/// `DISCHARGED` is the portfolio's marker for wording kept as history, and a
/// discharged note must quote what it retires to be legible at all.
///
/// Case-folds its own input rather than trusting the caller to have done it: the
/// first version took an already-lowercased string and its unit test passed the
/// original, so the test failed against a correct function.
fn is_discharged(text: &str) -> bool {
    text.to_ascii_lowercase().contains("discharged")
}

/// Claims of a live panic macro: `(first_line, cited_file, macro_name)`.
fn liveness_claims(doc: &Path, body: &str) -> Vec<(usize, String, String)> {
    let mut out = Vec::new();
    for (line_no, text) in paragraphs(strip_quotations(document_lines(doc, body))) {
        if is_discharged(&text) {
            continue;
        }
        for mac in MACROS {
            if let Some(f) = claimed_file(&text, mac) {
                out.push((line_no, f, mac.to_string()));
            }
        }
    }
    out
}

/// Index of `src` sources by bare file name, for citations written without a
/// directory. Ambiguous names are dropped rather than guessed.
fn src_index(root: &Path) -> HashMap<String, PathBuf> {
    let mut files = Vec::new();
    collect(&root.join("src"), &|p| has_ext(p, "rs"), &mut files);
    let mut seen: HashMap<String, Option<PathBuf>> = HashMap::new();
    for p in files {
        let name = p
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        seen.entry(name)
            .and_modify(|e| *e = None)
            .or_insert(Some(p));
    }
    seen.into_iter()
        .filter_map(|(k, v)| v.map(|p| (k, p)))
        .collect()
}

fn resolve(root: &Path, index: &HashMap<String, PathBuf>, cited: &str) -> Option<PathBuf> {
    if cited.contains('/') {
        let p = root.join(cited);
        return p.is_file().then_some(p);
    }
    index.get(cited).cloned()
}

#[test]
fn no_document_claims_a_panic_that_is_not_live() {
    let root = repo_root();
    let docs = corpus_paths(&root);

    // The corpus must be real. A wrong root reads nothing and passes vacuously.
    assert!(
        docs.len() > 100,
        "corpus is only {} files — the walk is wrong, and an empty corpus passes \
         every check below having examined nothing",
        docs.len()
    );

    let index = src_index(&root);
    let mut violations = Vec::new();
    let mut examined = 0usize;

    for doc in &docs {
        let Ok(body) = std::fs::read_to_string(doc) else {
            continue; // non-UTF-8 is not a document
        };
        for (line_no, cited, mac) in liveness_claims(doc, &body) {
            examined += 1;
            let rel = doc.strip_prefix(&root).unwrap_or(doc).display();
            let Some(target) = resolve(&root, &index, &cited) else {
                violations.push(format!(
                    "{rel}:{line_no} claims a live `{mac}!()` in `{cited}`, which does not \
                     resolve to a file in this repo"
                ));
                continue;
            };
            let src = std::fs::read_to_string(&target)
                .unwrap_or_else(|e| panic!("cannot read {} — {e}", target.display()));
            if live_macro_lines(&src, &mac).is_empty() {
                let textual = src
                    .lines()
                    .filter(|l| l.contains(&format!("{mac}!(")))
                    .count();
                violations.push(format!(
                    "{rel}:{line_no} says `{cited}` has a live `{mac}!()`, but it has none \
                     ({textual} textual occurrence(s), all in comments)"
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "A DOCUMENT CLAIMS A PANIC THAT IS NOT LIVE:\n\n  {}\n\n\
         ({examined} claim(s) checked across {} documents.)\n\n\
         A correction that outlives its defect is a defect. If the macro was removed, \
         say so in the past tense — ROADMAP already does exactly that for this same \
         subject, which is how the first of these was found. Wording kept as history \
         inside a blockquote, a fenced block, or a paragraph marked DISCHARGED is \
         intended and is not counted.",
        violations.join("\n  "),
        docs.len()
    );
}

#[test]
fn flags_a_false_claim() {
    // ⚠️ CONSTRUCTED, not sampled. The documents this guards are being cleaned by
    // this change and its sibling, so a negative case taken from one would expire
    // the moment the fix landed and the guard would silently verify nothing.
    let md = Path::new("x.md");

    let one_line = "**Dependencies**: thing (claimed COMPLETE, but \
                    `src/made/up/path.rs:446` is a live `todo!()`)";
    let hits = liveness_claims(md, one_line);
    assert_eq!(hits.len(), 1, "the claim shape stopped matching: {hits:?}");
    assert_eq!(hits[0].1, "src/made/up/path.rs");
    assert_eq!(hits[0].2, "todo");

    // ⚠️ THE REGRESSION THAT MATTERS: the same claim, hard-wrapped, which the
    // line-anchored predecessor could not see. The path and the macro token are on
    // different physical lines; only paragraph joining brings them together.
    let wrapped = "gate-zero failure. `kv_compression.rs:446` is a live\n\
                   `todo!(\"Grouped quantization not yet implemented\")`, and more";
    let hits = liveness_claims(md, wrapped);
    assert_eq!(
        hits.len(),
        1,
        "a claim split across a line wrap was missed — the exact defect that made \
         the first version of this check report 1 of 3: {hits:?}"
    );
    assert_eq!(hits[0].1, "kv_compression.rs");
}

#[test]
fn one_claim_is_attributed_to_one_file() {
    // ⚠️ REGRESSION. A bullet list with no blank lines is ONE paragraph. Taking
    // every `.rs` token in it attributed a single claim to five files, three of
    // which the sentence never mentions.
    let md = Path::new("x.md");
    let bullets = "* `a.rs:1` is a live `todo!()` and that is the claim\n\
                   * `b.rs:2` is fine, see `c.rs:3` and `d.rs:4`";
    let hits = liveness_claims(md, bullets);
    assert_eq!(hits.len(), 1, "one claim became {}: {hits:?}", hits.len());
    assert_eq!(
        hits[0].1, "a.rs",
        "the claim was attributed to the wrong file"
    );

    // The subject can also follow the phrase.
    let after = "there is a live `todo!()` in `late.rs:9`";
    assert_eq!(liveness_claims(md, after)[0].1, "late.rs");
}

#[test]
fn history_and_quotation_are_not_claims() {
    let md = Path::new("x.md");

    // Documents must stay free to record that a macro WAS there.
    let past = "`src/cache/kv_compression.rs:446`'s `PerGroup` was a live `todo!()` — fixed.";
    assert!(
        liveness_claims(md, past).is_empty(),
        "a past-tense sentence was treated as a live claim"
    );

    // A DISCHARGED note must quote the wording it retires, or a reader cannot tell
    // what was corrected — as a blockquote…
    let quoted = "> ⚠ **DISCHARGED.** This read *\"`src/a/b.rs:446` is a live \
                  `todo!()`\"* until today.";
    assert!(
        liveness_claims(md, quoted).is_empty(),
        "a quoted retraction was counted as the document's own claim"
    );

    // …and as a plain `//!` line, which no blockquote rule reaches. ⚠️ This case
    // was PREDICTED by the lightbulb lane against their own in-flight fix, before
    // this guard had ever been run against it.
    //
    // ⚠️ Written with `concat!` so that no line of THIS file begins with `//!`.
    // The scanner reads line shape, not Rust syntax, so a string literal whose
    // continuation lines start with a doc marker is read as documentation — and
    // the first version of this fixture made the guard fire on its own test body.
    let rs = Path::new("x.rs");
    let discharged = concat!(
        "//! DISCHARGED 2026-09-06: this entry read\n",
        "//! `kv_compression.rs:446` is a live `todo!(\"grouped\")`.\n",
        "//! It returns an `Err` now.\n"
    );
    assert!(
        liveness_claims(rs, discharged).is_empty(),
        "a DISCHARGED note was counted as the document's own claim — the guard now \
         blocks the very fix it asked for"
    );

    // A fenced example — this file's own header quotes all the false lines inside
    // one, and this file is in the corpus.
    let fenced = "Here is what it used to say:\n\n```text\n\
                  `src/a/b.rs:446` is a live `todo!()`\n```\n";
    assert!(
        liveness_claims(md, fenced).is_empty(),
        "a fenced example was counted as the document's own claim"
    );
}

#[test]
fn the_predicates_can_fail_rather_than_returning_a_false_clean_result() {
    assert_eq!(
        cited_rs_files("see `src/a/b.rs:446` and `c.rs`"),
        vec!["src/a/b.rs", "c.rs"]
    );
    assert!(
        cited_rs_files("no backticked path here, and `notapath` either").is_empty(),
        "a non-path backtick was read as a file"
    );
    assert!(
        cited_rs_files("a bare `.rs` is not a file name").is_empty(),
        "the extension alone was read as a file"
    );

    assert_eq!(
        claimed_file("`a.rs` is a live `todo!()`", "todo").as_deref(),
        Some("a.rs")
    );
    assert_eq!(claimed_file("`a.rs` was a live `todo!()`", "todo"), None);
    assert_eq!(
        claimed_file("`a.rs` is a live `todo!()`", "unimplemented"),
        None,
        "a claim about one macro was attributed to another"
    );
    assert_eq!(
        claimed_file("nothing cited, is a live `todo!()`", "todo"),
        None
    );

    assert!(is_discharged("⚠️ **DISCHARGED 2026-09-06:** this read …"));
    assert!(!is_discharged("this is an ordinary paragraph"));

    // Paragraph joining, the property the whole check now rests on.
    let joined = paragraphs(vec![
        (1, "alpha".into()),
        (2, "beta".into()),
        (3, String::new()),
        (4, "gamma".into()),
    ]);
    assert_eq!(
        joined,
        vec![(1, "alpha beta".to_string()), (4, "gamma".into())]
    );

    // Doc-comment extraction: the document half of a Rust source.
    let rs = Path::new("x.rs");
    let lines = document_lines(rs, "//! doc one\nlet x = 1; // not a doc\n/// doc two\n");
    assert_eq!(lines[0].1, "doc one");
    assert_eq!(lines[1].1, "", "a plain code line is not a document line");
    assert_eq!(lines[2].1, "doc two");

    // The distinction the whole check rests on.
    assert_eq!(live_macro_lines("    todo!(\"x\");", "todo"), vec![1]);
    assert!(live_macro_lines("    // a `todo!()` here would panic", "todo").is_empty());
    assert!(live_macro_lines("//! it was `todo!(\"grouped\")`, and", "todo").is_empty());
    assert!(live_macro_lines(" * legacy: todo!()", "todo").is_empty());
}

#[test]
fn the_documents_this_check_reads_are_where_it_expects() {
    let root = repo_root();
    assert!(
        root.join("ROADMAP.md").is_file(),
        "ROADMAP.md is missing — the check has lost its largest subject"
    );
    let docs = corpus_paths(&root);
    let md = docs.iter().filter(|p| has_ext(p, "md")).count();
    let rs = docs.iter().filter(|p| has_ext(p, "rs")).count();
    assert!(
        md > 50 && rs > 50,
        "corpus looks wrong: {md} markdown, {rs} rust — expected both well over 50"
    );
    assert!(
        docs.iter().all(|p| {
            !p.components()
                .any(|c| c.as_os_str().to_str().is_some_and(is_skipped_dir))
        }),
        "the walk descended into a build or VCS directory"
    );
}
