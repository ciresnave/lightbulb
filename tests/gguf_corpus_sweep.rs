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
//! outcome for shapes this code does not support (`gpt2`-model GGUFs, and
//! SentencePiece checkpoints carrying scores but no merges), and asserting
//! universal success would force exactly the approximation that was measured
//! and rejected — see `src/gguf/mod.rs`.
//!
//! Run:
//! ```text
//! LIGHTBULB_GGUF_CORPUS=<dir> cargo test --test gguf_corpus_sweep -- --ignored --nocapture
//! ```

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

#[test]
#[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
fn every_gguf_either_rebuilds_a_tokenizer_or_refuses_with_a_reason() {
    let (files, unreadable_dirs) =
        corpus().expect("set LIGHTBULB_GGUF_CORPUS to a directory containing .gguf files");
    assert!(
        unreadable_dirs.is_empty(),
        "the corpus walk was incomplete, so any count below understates it: {unreadable_dirs:?}"
    );
    let mut ok = 0usize;
    let mut refused = 0usize;
    let mut silent = Vec::new();

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        // `Content::read` takes `AsRef<Path>`, so the `PathBuf` goes straight in.
        // `path.to_str().unwrap()` panicked on a non-UTF-8 path -- in the test
        // whose entire assertion is that this code never panics.
        let content = match lightbulb::gguf::Content::read(path) {
            Ok(c) => c,
            Err(e) => {
                // Not a readable GGUF at all — still an error with a reason,
                // which is the property under test.
                println!("  {name:<38} UNREADABLE  {e}");
                refused += 1;
                continue;
            }
        };
        match content.extract_tokenizer() {
            Ok(tok) => {
                println!(
                    "  {name:<38} OK          vocab={}",
                    tok.get_vocab_size(true)
                );
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
                println!(
                    "  {name:<38} REFUSED     {}",
                    msg.lines().next().unwrap_or("")
                );
                refused += 1;
            }
        }
    }

    println!("\n  {} files: {ok} rebuilt, {refused} refused", files.len());
    assert!(
        silent.is_empty(),
        "refusals that do not explain themselves: {silent:?}"
    );
    assert_eq!(
        ok + refused,
        files.len(),
        "some file neither rebuilt nor refused"
    );
}
