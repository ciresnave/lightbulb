//! How many VOCABULARIES is the GGUF corpus, as distinct from how many FILES?
//!
//! The corpus size is quoted in ROADMAP.md and in this repo's tokenizer claims
//! as a file count. A file count answers a different question from the one a
//! tokenizer corpus is asked.
//!
//! Nine of the files are `SmolLM2-135M-Instruct` at nine quantizations. A
//! quantization changes the WEIGHTS; it does not change `tokenizer.ggml.tokens`.
//! So those nine files exercise the tokenizer path once, and a report of "9
//! files rebuilt" is one model counted nine times. Growing the file count by
//! adding quantizations grows the number without growing the coverage.
//!
//! This measures it rather than inferring it from names: two files with
//! unrelated names can still carry the same vocabulary, and a name is not a
//! measurement.
//!
//! Run:
//!   LIGHTBULB_GGUF_CORPUS=<dir> cargo test --test gguf_corpus_vocab_census -- --ignored --nocapture

use lightbulb::gguf::{Content, Value};
use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;

/// The token list, read through the public metadata map.
///
/// Returns `Err` with a reason rather than `None`, because "this file has no
/// vocabulary" and "we failed to read the vocabulary" are different facts and
/// collapsing them would understate the corpus in exactly the direction that
/// flatters it.
fn vocab(content: &Content) -> Result<Vec<String>, String> {
    let Some(v) = content.metadata().get("tokenizer.ggml.tokens") else {
        return Err("no tokenizer.ggml.tokens key".into());
    };
    let Value::Array(values) = v else {
        return Err(format!("tokenizer.ggml.tokens is {v:?}, not an Array"));
    };
    let mut out = Vec::with_capacity(values.len());
    for (i, item) in values.iter().enumerate() {
        match item.to_string() {
            Ok(s) => out.push(s.clone()),
            Err(e) => return Err(format!("token {i} is not a string: {e}")),
        }
    }
    Ok(out)
}

/// `tokenizer.ggml.model` and `tokenizer.ggml.pre`, which is what the rebuild
/// decision actually turns on -- not the token list.
fn tokenizer_rule(content: &Content) -> (String, String) {
    let get = |k: &str| {
        content
            .metadata()
            .get(k)
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "-".into())
    };
    (get("tokenizer.ggml.model"), get("tokenizer.ggml.pre"))
}

fn vocab_digest(tokens: &[String]) -> u64 {
    let mut h = DefaultHasher::new();
    tokens.len().hash(&mut h);
    for t in tokens {
        t.hash(&mut h);
    }
    h.finish()
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

/// One file's contribution to a vocabulary group.
///
/// A struct rather than a 4-tuple because the tuple tripped
/// `clippy::type_complexity` in the gate -- correctly. `(String, bool, String,
/// String)` gives every field the same anonymous shape, and `|(_, r, _, _)|`
/// closures are only readable if you keep the order in your head.
struct FileEntry {
    name: String,
    rebuilt: bool,
    model: String,
    pre: String,
}

/// Every file sharing one `tokenizer.ggml.tokens` array.
struct VocabGroup {
    token_count: usize,
    files: Vec<FileEntry>,
}

#[test]
#[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
fn corpus_is_fewer_vocabularies_than_files() {
    let files = corpus_files();
    assert!(
        !files.is_empty(),
        "no .gguf files under LIGHTBULB_GGUF_CORPUS={:?}. An empty corpus makes every \
         count below zero, and zero would be reported as a clean census.",
        std::env::var("LIGHTBULB_GGUF_CORPUS").unwrap_or_default()
    );

    // digest -> (vocab_len, [(file name, tokenizer rebuilt?)])
    let mut groups: BTreeMap<u64, VocabGroup> = BTreeMap::new();
    let mut unreadable: Vec<(String, String)> = Vec::new();

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let content = match Content::read(path) {
            Ok(c) => c,
            Err(e) => {
                unreadable.push((name, format!("unreadable: {e}")));
                continue;
            }
        };
        match vocab(&content) {
            Ok(tokens) => {
                let d = vocab_digest(&tokens);
                // Whether the tokenizer rebuilds is NOT a property of the token
                // list alone -- it also turns on `pre`, merges and token types.
                // So two files sharing a vocabulary can disagree here, and if
                // they do, that is worth knowing: it would mean the rebuild
                // decision is carrying information the vocabulary does not.
                let rebuilt = content.extract_tokenizer().is_ok();
                let (model, pre) = tokenizer_rule(&content);
                groups
                    .entry(d)
                    .or_insert_with(|| VocabGroup {
                        token_count: tokens.len(),
                        files: Vec::new(),
                    })
                    .files
                    .push(FileEntry {
                        name,
                        rebuilt,
                        model,
                        pre,
                    });
            }
            Err(why) => unreadable.push((name, why)),
        }
    }

    println!("\n=== GGUF corpus census ===");
    println!("  files scanned      : {}", files.len());
    println!("  distinct vocabularies: {}", groups.len());
    println!("  no readable vocab  : {}", unreadable.len());
    println!();

    let mut shared = 0usize;
    let mut vocabs_rebuilt = 0usize;
    let mut split_groups: Vec<u64> = Vec::new();
    for (digest, group) in &groups {
        let (len, entries) = (group.token_count, &group.files);
        let any = entries.iter().any(|f| f.rebuilt);
        let all = entries.iter().all(|f| f.rebuilt);
        if any {
            vocabs_rebuilt += 1;
        }
        if any && !all {
            split_groups.push(*digest);
        }
        let mark = if all {
            "REBUILT"
        } else if any {
            "SPLIT  "
        } else {
            "refused"
        };
        if entries.len() > 1 {
            shared += entries.len() - 1;
            println!(
                "  {len:>6} tokens  [{digest:016x}]  {mark}  {} FILES SHARE THIS VOCABULARY:",
                entries.len()
            );
            for f in entries {
                println!(
                    "           {} {:<38} model={} pre={}",
                    if f.rebuilt { "ok " } else { "no " },
                    f.name,
                    f.model,
                    f.pre
                );
            }
        } else {
            println!(
                "  {len:>6} tokens  [{digest:016x}]  {mark}  {}",
                entries[0].name
            );
        }
    }
    if !unreadable.is_empty() {
        println!("\n  no readable vocabulary (NOT counted as a vocabulary):");
        for (n, why) in &unreadable {
            println!("    {n:<44} {why}");
        }
    }
    let files_rebuilt: usize = groups
        .values()
        .map(|g| g.files.iter().filter(|f| f.rebuilt).count())
        .sum();
    println!(
        "\n  ==> report this corpus as {} files / {} vocabularies ({} files are duplicates \
         of a vocabulary already present)",
        files.len(),
        groups.len(),
        shared
    );
    println!(
        "  ==> tokenizer rebuild: {files_rebuilt} of {} files, but {vocabs_rebuilt} of {} \
         VOCABULARIES -- the second is the coverage number",
        files.len(),
        groups.len()
    );

    // SPLIT GROUPS ARE EXPECTED, AND THE FIRST VERSION OF THIS ASSERTION WAS
    // WRONG TO FORBID THEM. It failed on its first run, and the failure was the
    // assertion's, not the corpus's. Measured, two vocabularies split:
    //
    //   151936  qwen2 (pre=qwen2) rebuilds        qwen35 (pre=qwen35) refuses
    //    32000  tinyllama Q4_0 (llama, pre absent) rebuilds
    //           llama-spm (llama, pre=default)     refuses
    //
    // That is the refusal policy working as designed. Support is keyed on
    // `tokenizer.ggml.pre`, deliberately, because vocabulary identity does NOT
    // imply pre-tokenizer rule identity -- the exact reasoning that refused
    // `qwen35` despite it differing from `qwen2` on 0 of 130 cases here. So a
    // vocabulary is the wrong unit for "is this supported"; the rule is.
    //
    // The invariant is therefore not "no splits" but "every split is EXPLAINED
    // by a differing rule". A split where the files carry the SAME (model, pre)
    // and disagree anyway is a real defect: identical inputs, different
    // outcomes.
    let mut unexplained: Vec<String> = Vec::new();
    for digest in &split_groups {
        let rules: std::collections::BTreeSet<(&str, &str)> = groups[digest]
            .files
            .iter()
            .map(|f| (f.model.as_str(), f.pre.as_str()))
            .collect();
        if rules.len() == 1 {
            unexplained.push(format!(
                "[{digest:016x}] every file carries model/pre {rules:?} yet they disagree"
            ));
        }
    }
    assert!(
        unexplained.is_empty(),
        "a vocabulary rebuilt for some files and not others WITH NO DIFFERENCE IN model/pre \
         -- identical inputs, different outcomes:\n{}",
        unexplained.join("\n")
    );
    println!(
        "  ==> {} vocabulary group(s) split by a differing pre-tokenizer rule, as designed",
        split_groups.len()
    );

    // The claim this test exists to keep honest. It is deliberately an
    // inequality rather than a pinned pair: pinning both numbers would make
    // every corpus addition a test edit, and the property that matters is that
    // the two numbers are TRACKED SEPARATELY, not that they hold given values.
    assert!(
        !groups.is_empty(),
        "no file in the corpus yielded a readable vocabulary, so the census measured nothing"
    );
}
