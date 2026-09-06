//! The corpus README states measured figures. This re-derives them.
//!
//! # Why a test in lightbulb gates a file outside lightbulb
//!
//! `$LIGHTBULB_GGUF_CORPUS/README.md` describes the corpus every GGUF test in
//! this repo runs against. It states counts. ⚠️ **It sits outside every
//! repository, so no lane's gate reaches it, and a tree-scoped audit of any
//! project returns clean while it rots.**
//!
//! It went wrong three times within twelve hours of being written — a bare date
//! with no derivation, one lane's figure projected onto another's population,
//! and two different corpus roots quoted twenty-three lines apart so the
//! arithmetic could not close. The first two were found by re-reading it; the
//! third by the Claim Auditor, who could not settle it from the file and said so.
//!
//! **The stamp names a derivation, which means a reader CAN refresh it. It does
//! not mean anyone WILL.** So this runs the derivation.
//!
//! # ⚠️ What this does NOT do
//!
//! It checks the two figures a cheap pass can re-derive: the file count, and how
//! many this reader cannot open. **It does not check the vocabulary count, the
//! per-architecture claims, or anything in the prose.** A green here means two
//! numbers agree — it is not a warrant for the document.
//!
//! ```text
//!   LIGHTBULB_GGUF_CORPUS=<dir> cargo test --test corpus_readme_figures -- --ignored --nocapture
//! ```

use std::path::PathBuf;

fn corpus_root() -> Option<PathBuf> {
    std::env::var_os("LIGHTBULB_GGUF_CORPUS").map(PathBuf::from)
}

fn gguf_files(root: &std::path::Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "gguf") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// The first integer on a line containing `needle`.
///
/// Deliberately returns `Option`: a figure the README stopped stating is a
/// different fact from a figure that disagrees, and the caller distinguishes
/// them. ⚠️ Without that, a rewritten README makes this test pass by saying
/// nothing.
fn stated_figure(text: &str, needle: &str) -> Option<u64> {
    text.lines()
        .find(|l| l.contains(needle))?
        .split(|c: char| !c.is_ascii_digit())
        .find(|t| !t.is_empty())
        .and_then(|t| t.parse().ok())
}

#[test]
#[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
fn the_corpus_readme_figures_still_hold() {
    let Some(root) = corpus_root() else {
        lightbulb::test_notice::skip_unless_required(
            "LIGHTBULB_REQUIRE_CORPUS",
            "no LIGHTBULB_GGUF_CORPUS, so the corpus README cannot be checked",
        );
        return;
    };

    let readme = root.join("README.md");
    let Ok(text) = std::fs::read_to_string(&readme) else {
        lightbulb::test_notice::skip_unless_required(
            "LIGHTBULB_REQUIRE_CORPUS",
            "no README.md at the corpus root",
        );
        return;
    };

    // ⚠️ POSITIVE CONTROL FIRST. A parser that matches nothing reports every
    // figure as absent, and "absent" would otherwise read as "nothing to check".
    let files_claim = stated_figure(&text, ".gguf files (recursive)").expect(
        "the README states no file count this parser can find. That is a BROKEN PARSER or a \
         rewritten README, not a clean result — and both need a human, not a green tick.",
    );
    let unreadable_claim = stated_figure(&text, "lightbulb   cannot open").expect(
        "the README states no lightbulb-unreadable count this parser can find. Same reasoning \
         as above: this is a parse failure, not an absence.",
    );

    let files = gguf_files(&root);
    let unreadable = files
        .iter()
        .filter(|p| lightbulb::gguf::Content::read(p).is_err())
        .count() as u64;

    eprintln!(
        "  corpus README at {}\n    files: states {files_claim}, measured {}\n    \
         unreadable by this reader: states {unreadable_claim}, measured {unreadable}",
        readme.display(),
        files.len()
    );

    assert_eq!(
        files.len() as u64,
        files_claim,
        "the corpus README states {files_claim} .gguf files and there are {}. The README is \
         outside every repository, so nothing else would have caught this.",
        files.len()
    );
    assert_eq!(
        unreadable, unreadable_claim,
        "the corpus README states this reader cannot open {unreadable_claim} files and it \
         cannot open {unreadable}. ⚠️ That number moves when OUR reader changes, not when the \
         corpus does — #51 taught it GGUF v2 and took the count from 5 to 4 the same morning \
         the figure was written."
    );
}
