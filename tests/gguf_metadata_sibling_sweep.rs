//! Does any key lightbulb reads have SIBLINGS in a real GGUF that a single-key
//! read would silently discard?
//!
//! # The shape being swept for
//!
//! `tokenizer.chat_template` is not always alone. `ggml-vocab-command-r.gguf`
//! carries `tokenizer.chat_template.rag` and `tokenizer.chat_template.tool_use`
//! beside it, plus a `tokenizer.chat_templates` ARRAY. Code that reads the bare
//! key gets a template, uses it, and reports nothing — the other two are not
//! missing, not defaulted and not warned about. They are *unseen*.
//!
//! That was a live defect until #48. This sweep exists so the next key with the
//! same shape is caught by a test rather than by somebody happening to look.
//!
//! # ⚠️ A null here means nothing without BOTH populations
//!
//! "No key has unaccounted siblings" is only worth reading if the sweep could
//! have found some. Two numbers make that checkable, and the report prints both:
//!
//! ```text
//! POPULATION SCANNED     files this sweep actually enumerated keys from
//! POPULATION SCANNABLE   files Content::read can open at all, and why not
//! ```
//!
//! The second is the one that decays. `Content::read` is ours and its coverage
//! is a property of us, not of the corpus — measured 2026-09-05, it opens 25 of
//! 30 files (26 once GGUF v2 is accepted). A sweep that silently stopped opening
//! anything would report a clean null, so the file count is asserted non-zero
//! and printed next to the verdict rather than left to be inferred from it.
//!
//! # What "accounted for" means
//!
//! A sibling `K.x` is accounted for when lightbulb reads it explicitly, OR reads
//! the prefix `"K."` — which is how the post-#48 code collects the whole family.
//! Anything else is a key present in a real checkpoint that no code path can
//! see.
//!
//! ```text
//!   LIGHTBULB_GGUF_CORPUS=<dir> cargo test --test gguf_metadata_sibling_sweep -- --ignored --nocapture
//! ```

use lightbulb::gguf::Content;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Every GGUF metadata key lightbulb reads, from string literals under `src/`.
///
/// A literal scan rather than a curated list, because a curated list is a second
/// place to forget. Its coverage is bounded and stated: keys built by `format!`
/// or concatenation are invisible to it. Measured 2026-09-05, there are none —
/// the only `format!` hits under `src/` build `"..."` and `"."`, not keys.
fn keys_lightbulb_reads(root: &Path) -> BTreeSet<String> {
    let mut files = Vec::new();
    collect_rust_files(&root.join("src"), &mut files);
    let mut keys = BTreeSet::new();
    for f in &files {
        let Ok(src) = std::fs::read_to_string(f) else {
            continue;
        };
        for literal in string_literals(&src) {
            if (literal.starts_with("tokenizer.") || literal.starts_with("general."))
                && literal
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'.')
            {
                keys.insert(literal);
            }
        }
    }
    keys
}

/// Double-quoted literals, with escapes skipped so `\"` does not end one.
fn string_literals(src: &str) -> Vec<String> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'"' {
            let start = i + 1;
            let mut j = start;
            while j < b.len() && b[j] != b'"' {
                j += if b[j] == b'\\' { 2 } else { 1 };
            }
            // `j` can land one past the end when the final byte is a backslash,
            // so the slice end is clamped rather than guarded separately.
            let end = j.min(b.len());
            if let Ok(s) = std::str::from_utf8(&b[start..end]) {
                out.push(s.to_string());
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// Keys present in a file, sitting under a key lightbulb reads, that no code
/// path can see.
///
/// Pure over the two key sets, so the detector is testable without a corpus.
/// `present` is one file's keys; `read` is what `src/` reads.
fn unaccounted_siblings(read: &BTreeSet<String>, present: &BTreeSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    for k in read {
        // A prefix entry like "tokenizer.chat_template." is a family read, not
        // a key in its own right, and has no siblings of its own.
        if k.ends_with('.') {
            continue;
        }
        let family = format!("{k}.");
        let reads_family = read.contains(&family);
        for p in present {
            if p.starts_with(&family) && !read.contains(p) && !reads_family {
                out.push(p.clone());
            }
        }
        // The plural-array form: `tokenizer.chat_templates` beside
        // `tokenizer.chat_template`. A different shape of the same hazard --
        // several values where the code expects one.
        let plural = format!("{k}s");
        if present.contains(&plural) && !read.contains(&plural) {
            out.push(plural);
        }
    }
    out.sort();
    out.dedup();
    out
}

fn corpus_files() -> Vec<PathBuf> {
    let Some(root) = std::env::var_os("LIGHTBULB_GGUF_CORPUS") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut stack = vec![PathBuf::from(root)];
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

/// The detector, held born-red against the defect that motivated this file.
///
/// Uses the REAL key names from `ggml-vocab-command-r.gguf` rather than invented
/// ones, so a rename in the GGUF spec makes this stale visibly rather than
/// leaving it passing against a fiction.
#[test]
fn the_detector_finds_the_defect_that_motivated_it() {
    let present: BTreeSet<String> = [
        "tokenizer.chat_template",
        "tokenizer.chat_template.rag",
        "tokenizer.chat_template.tool_use",
        "tokenizer.chat_templates",
        "tokenizer.ggml.tokens",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    // PRE-#48: the bare key only. All three extras are invisible.
    let before: BTreeSet<String> = ["tokenizer.chat_template", "tokenizer.ggml.tokens"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        unaccounted_siblings(&before, &present),
        vec![
            "tokenizer.chat_template.rag".to_string(),
            "tokenizer.chat_template.tool_use".to_string(),
            "tokenizer.chat_templates".to_string(),
        ],
        "reading only the bare key must be reported -- this is the pre-#48 state \
         and the whole reason this sweep exists"
    );

    // POST-#48: the family prefix and the plural array are read too.
    let after: BTreeSet<String> = [
        "tokenizer.chat_template",
        "tokenizer.chat_template.",
        "tokenizer.chat_templates",
        "tokenizer.ggml.tokens",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert!(
        unaccounted_siblings(&after, &present).is_empty(),
        "reading the family prefix must account for every sibling under it, or \
         this is not a detector, it is a permanent refusal"
    );
}

/// The literal scanner has to survive an escaped quote, or it silently fuses two
/// literals and reports a key nobody wrote.
#[test]
fn the_literal_scanner_handles_escapes() {
    let src =
        r#"let a = "tokenizer.ggml.tokens"; let b = "say \"hi\""; let c = "general.alignment";"#;
    let lits = string_literals(src);
    assert!(lits.contains(&"tokenizer.ggml.tokens".to_string()));
    assert!(
        lits.contains(&"general.alignment".to_string()),
        "an escaped quote in an earlier literal must not swallow later ones: {lits:?}"
    );
}

#[test]
#[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
fn no_key_lightbulb_reads_has_unaccounted_siblings() {
    let files = corpus_files();
    assert!(
        !files.is_empty(),
        "no .gguf files under LIGHTBULB_GGUF_CORPUS={:?}. An empty corpus makes this \
         sweep vacuously clean, and vacuously clean is indistinguishable from clean.",
        std::env::var("LIGHTBULB_GGUF_CORPUS").unwrap_or_default()
    );

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let read = keys_lightbulb_reads(&root);
    assert!(
        !read.is_empty(),
        "the literal scan found no metadata keys under src/. That is a broken \
         scanner, not a codebase that reads no metadata."
    );

    let mut scanned = 0usize;
    let mut unreadable: Vec<(String, String)> = Vec::new();
    let mut findings: Vec<(String, Vec<String>)> = Vec::new();
    let mut all_keys = BTreeSet::new();

    for path in &files {
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let content = match Content::read(path) {
            Ok(c) => c,
            // A FACT ABOUT US, not about the file.
            Err(e) => {
                unreadable.push((name, e.to_string()));
                continue;
            }
        };
        scanned += 1;
        let present: BTreeSet<String> = content.metadata().keys().cloned().collect();
        all_keys.extend(present.iter().cloned());
        let missed = unaccounted_siblings(&read, &present);
        if !missed.is_empty() {
            findings.push((name, missed));
        }
    }

    println!("\n=== GGUF metadata sibling sweep ===");
    println!("  POPULATION SCANNED   : {scanned} files");
    println!(
        "  POPULATION SCANNABLE : {scanned} of {}   <- Content::read's coverage, a fact \
         about US",
        files.len()
    );
    for (n, why) in &unreadable {
        println!("      not scannable: {n:<44} {why}");
    }
    println!(
        "  distinct keys across the scanned files: {}",
        all_keys.len()
    );
    println!("  keys lightbulb reads (src literals)   : {}", read.len());

    assert!(
        scanned > 0,
        "Content::read opened none of the {} corpus files, so this sweep measured \
         nothing. A null from a sweep that read nothing is not a clean result.",
        files.len()
    );

    let report: Vec<String> = findings
        .iter()
        .map(|(n, ks)| format!("    {n}: {}", ks.join(", ")))
        .collect();
    assert!(
        findings.is_empty(),
        "keys present in a real checkpoint, under a key lightbulb reads, that no code \
         path can see. A single-key read takes one of several and says nothing:\n{}",
        report.join("\n")
    );
    println!("  ==> no unaccounted siblings across {scanned} files\n");
}
