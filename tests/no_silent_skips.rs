//! Makes the loud-skip convention discoverable by FAILING when it is not used.
//!
//! A convention nobody can find is a convention nobody follows. `src/
//! test_notice.rs` exists and is documented, but a new test written next month
//! will reach for `eprintln!("skipping: ..."); return;` because that is what the
//! surrounding code used to look like — and it will pass, forever, silently.
//!
//! So this scans the repo for that shape and fails, naming the sites and the
//! helper to use. The convention is then learned at the moment it is violated,
//! which is the only moment anyone is looking.
//!
//! WHAT COUNTS AS A SILENT SKIP HERE: inside a `#[test]` function, a print macro
//! followed by an early `return`, where the function does not call
//! `skip_unless_required`. That is deliberately a shape, not a wording — keying
//! on the word "skip" would miss the next author's phrasing.
//!
//! THE SCANNER HAS A POSITIVE CONTROL, because "0 sites found" and "the scanner
//! is broken" are the same output. `scanner_finds_a_known_silent_skip` runs the
//! predicate over a synthetic source that contains one, and fails if it is not
//! detected. Without that, this file could rot into a no-op and read as a clean
//! repo — which is the exact failure class it exists to prevent.

use std::fs;
use std::path::{Path, PathBuf};

/// Strip `//` line comments so that prose ABOUT the pattern is not read AS the
/// pattern. Not hypothetical: an earlier version of this sweep flagged a comment
/// that documented this very hazard, in a file whose author had written it as a
/// warning.
fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|l| match l.find("//") {
            // Only when the `//` is not inside a string literal on that line.
            // Counting quotes is crude but errs toward KEEPING text, which
            // errs toward a false positive we triage rather than a false clear
            // we never see.
            Some(i) if l[..i].matches('"').count() % 2 == 0 => &l[..i],
            _ => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Bodies of every `#[test]` function, found by brace matching rather than by
/// regex — a line-oriented pattern cannot see where a function ends.
fn test_fn_bodies(src: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut search_from = 0;
    while let Some(rel) = src[search_from..].find("#[test]") {
        let at = search_from + rel;
        search_from = at + 7;
        let Some(fn_at) = src[at..].find("fn ") else {
            continue;
        };
        let name_start = at + fn_at + 3;
        let name: String = src[name_start..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let Some(open) = src[name_start..].find('{') else {
            continue;
        };
        let open = name_start + open;
        let mut depth = 0usize;
        let mut end = open;
        for (i, c) in src[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push((name, src[open..=end].to_string()));
    }
    out
}

/// A print macro followed by an early `return`, in a body that does not route
/// through the loud-skip helper.
fn is_silent_skip(body: &str) -> bool {
    if body.contains("skip_unless_required") {
        return false;
    }
    let clean = strip_line_comments(body);
    for macro_name in ["eprintln!", "println!", "eprint!", "print!"] {
        let mut from = 0;
        while let Some(rel) = clean[from..].find(macro_name) {
            let at = from + rel;
            from = at + macro_name.len();
            // Look at the text between this macro and the next few statements
            // for a `return` that is not inside a nested closure returning a
            // value. A window keeps this a "print then leave" detector rather
            // than "this function prints and also returns somewhere".
            // Clamp to a char boundary. A raw byte offset panics mid-codepoint,
            // and this repo's test output is full of `✓` -- which is how that
            // was found: the scan died inside one and the failure looked like a
            // finding until the message was read.
            let mut window_end = (at + 400).min(clean.len());
            while window_end > at && !clean.is_char_boundary(window_end) {
                window_end -= 1;
            }
            let window = &clean[at..window_end];
            if let Some(semi) = window.find(';') {
                let after = &window[semi..];
                let trimmed = after.trim_start_matches([';', ' ', '\n', '\r', '\t']);
                if trimmed.starts_with("return") {
                    return true;
                }
            }
        }
    }
    false
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// The scanner must be able to find something. Otherwise a refactor that breaks
/// the predicate turns this whole file into a green no-op, and "no silent skips"
/// becomes a statement about the scanner rather than about the repo.
#[test]
fn scanner_finds_a_known_silent_skip() {
    let positive = r#"{
        let Some(dir) = thing() else {
            eprintln!("skipping: no snapshot");
            return Ok(());
        };
    }"#;
    assert!(
        is_silent_skip(positive),
        "the scanner cannot detect a silent skip it was shown directly, so a clean \
         result from it means nothing"
    );

    // And the converted form must NOT be flagged, or every fixed site would
    // re-appear as a finding and the gate would be abandoned.
    let negative = r#"{
        let Some(dir) = thing() else {
            crate::test_notice::skip_unless_required("LIGHTBULB_REQUIRE_MODEL", "no snapshot");
            return Ok(());
        };
    }"#;
    assert!(
        !is_silent_skip(negative),
        "a site routed through skip_unless_required must not be flagged"
    );

    // Prose about the pattern is not the pattern.
    let commented = r#"{
        // Deliberately NOT the eprintln!("skipping"); return; the others use.
        assert!(true);
    }"#;
    assert!(
        !is_silent_skip(commented),
        "a comment describing the hazard must not be read as the hazard"
    );
}

#[test]
fn no_test_skips_silently() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_files(&root.join("src"), &mut files);
    rust_files(&root.join("tests"), &mut files);

    // The corpus must not be empty -- a wrong path would scan nothing and pass.
    assert!(
        files.len() > 50,
        "only {} rust files found under src/ and tests/; the scan path is wrong, and an \
         empty corpus passes every check vacuously",
        files.len()
    );

    let mut offenders = Vec::new();
    for path in &files {
        // This file contains deliberate examples of the shape, as its own
        // positive control. Scanning it would flag the control.
        if path.file_name().is_some_and(|n| n == "no_silent_skips.rs") {
            continue;
        }
        let Ok(src) = fs::read_to_string(path) else {
            continue;
        };
        for (name, body) in test_fn_bodies(&src) {
            if is_silent_skip(&body) {
                let rel = path.strip_prefix(&root).unwrap_or(path);
                offenders.push(format!("{}  ::  {name}", rel.display()));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "these tests print a notice and return, which is a PASS that asserts nothing.\n\
         cargo captures stdout/stderr for passing tests, so the notice is never shown \
         either -- see src/test_notice.rs for the measurement.\n\n{}\n\n\
         Use lightbulb::test_notice::skip_unless_required(REQUIRE_VAR, precondition), which \
         makes the notice greppable and turns the skip into a failure wherever the \
         environment claims to provision the precondition.",
        offenders.join("\n")
    );
}
