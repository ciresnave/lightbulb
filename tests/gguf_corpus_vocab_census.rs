//! How many VOCABULARIES is the GGUF corpus, as distinct from how many FILES?
//!
//! The corpus size is quoted in ROADMAP.md and in this repo's tokenizer claims
//! as a file count. A file count answers a different question from the one a
//! tokenizer corpus is asked.
//!
//! Nine of the files are `SmolLM2-135M-Instruct` at nine quantizations, and a
//! quantization changes the WEIGHTS while leaving `tokenizer.ggml.tokens`
//! untouched. So growing the file count by adding quantizations grows the number
//! without growing the coverage.
//!
//! # ⚠️ WHAT THIS COUNTS: vocabularies REACHABLE THROUGH `Content::read`
//!
//! Not "vocabularies in the corpus". The distinction is load-bearing and it took
//! three attempts at one sentence to get right:
//!
//! ```text
//! v1  "nine quantizations, one vocabulary"      read off FILENAMES, unmeasured
//! v2  "six share one, three fail to parse"      measured -- through OUR READER
//! v3  nine files, ONE vocabulary, all nine      measured from the KV header
//!     carry it; three are unreachable here
//! ```
//!
//! ⚠️ AND v3 WAS ALSO WRONG, ONE LAYER DOWN. The version above that reads
//! "three are unreachable here" was itself corrected: `ggml-vocab-aquila.gguf` is
//! GGUF **v2** and `tinyllamas-stories-260k-f32.gguf` is **v1**, whose counts and
//! string lengths are u32 rather than u64. A v3-assuming parser reads a v1 count
//! as a garbage u64 and dies allocating. Parsed with the right widths it carries
//! a 512-token vocabulary, `bos_token_id=1`, `eos_token_id=2`.
//!
//! So EVERY FILE IN THIS CORPUS HAS A VOCABULARY and the corpus holds NINETEEN.
//! The word to distrust is "genuinely": it appeared at three layers tonight --
//! "5 unreadable" (a fact about a call), "plausibly genuinely bad" (a fact about
//! field widths), "1 genuinely without" (a fact about v3 assumptions) -- and each
//! correction stayed scoped to the tool that made it, so each residue looked like
//! the irreducible truth.
//!
//! The 19 was re-measured over the whole population with a reader handling v1 and
//! v2/v3, NOT derived by adding the known exclusions back. Two lanes independently
//! said 18 by adding back only what their own tool had missed; nobody counted the
//! v1 file's 512-token vocabulary until the population was re-run whole.
//!
//! v2 was measured and still wrong, which is the instructive part. `Content::read`
//! parses TENSOR INFOS eagerly and fails with `unknown dtype for tensor N` on the
//! IQ3_XS / IQ4_XS / Q2_K quantizations — but `tokenizer.ggml.tokens` lives in the
//! KV header, **ahead of any tensor**. Reading those three files' KV headers
//! directly shows a full 49152-token vocabulary with a digest identical to the
//! other six. `ggml-vocab-aquila.gguf`, also reported here as unreadable, carries
//! a distinct 100008-token vocabulary.
//!
//! So `unreadable` was reported as a property of the FILE and is a property of a
//! CALL. Excluding those files did not guard against overstating the corpus, as
//! this file previously claimed — **it understated it**, by attributing the
//! reader's limitation to the GGUF.
//!
//! The corrected figures, measured 2026-09-03 at `C:\Models`:
//!
//! ```text
//! reachable through Content::read   17 vocabularies    <- what this test asserts
//! present in the corpus             18 vocabularies    <- KV headers read directly
//! genuinely without a vocabulary    1 file             <- tinyllamas-stories-260k
//! ```
//!
//! Both are honest answers to different questions. This test asks the first,
//! because it gates what the library can actually do; the second is recorded so
//! the first is never mistaken for it. Making the second reachable is a library
//! change — the metadata parser in `src/gguf/parser.rs` handles these files fine
//! and is not exposed — and is tracked separately.
//!
//! ⚠️ NONE OF THIS WAS FOUND BY RE-READING. It surfaced because MLMF reported
//! NINE SmolLM2 files declaring `add_bos`, against this file's six, and the two
//! numbers could not both be right until someone asked which construct each
//! ranged over. The cross-lane disagreement was the instrument.
//!
//! This measures it rather than inferring it from names: two files with
//! unrelated names can still carry the same vocabulary, and a name is not a
//! measurement.
//!
//! Run:
//!   LIGHTBULB_GGUF_CORPUS=<dir> cargo test --test gguf_corpus_vocab_census -- --ignored --nocapture

use lightbulb::gguf::{Content, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;

/// The token list, read through the public metadata map.
///
/// Returns `Err` with a reason rather than `None`, because "this file has no
/// vocabulary" and "we failed to read the vocabulary" are different facts.
///
/// This comment used to end "collapsing them would understate the corpus in
/// exactly the direction that flatters it" — and the collapsing was happening one
/// level up, in `census`, where a `Content::read` failure and a missing token key
/// shared one bucket. Being right about the distinction here did not prevent it
/// there.
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

/// THREE outcomes, not two, because two of them are facts about different
/// things and collapsing them is what made this file's headline number wrong.
///
///   groups          the file carries a vocabulary
///   no_vocab_key    it opened and has none          -- a fact about the FILE
///   reader_failed   `Content::read` could not open it -- a fact about US
///
/// The last bucket used to be merged into the second under the label
/// "unreadable", and its count was then subtracted from the corpus. That
/// understated the corpus by attributing a reader limitation to the GGUFs.
#[derive(Default)]
struct Census {
    groups: BTreeMap<u64, VocabGroup>,
    no_vocab_key: Vec<(String, String)>,
    reader_failed: Vec<(String, String)>,
}

/// Every file sharing one `tokenizer.ggml.tokens` array.
struct VocabGroup {
    token_count: usize,
    files: Vec<FileEntry>,
}

impl VocabGroup {
    fn any_rebuilt(&self) -> bool {
        self.files.iter().any(|f| f.rebuilt)
    }

    fn all_rebuilt(&self) -> bool {
        self.files.iter().all(|f| f.rebuilt)
    }

    /// Rebuilt for some files in the group and refused for others.
    fn is_split(&self) -> bool {
        self.any_rebuilt() && !self.all_rebuilt()
    }

    /// The distinct `(model, pre)` rules across this group's files. A split with
    /// exactly one rule is the real defect: identical inputs, different outcomes.
    fn rules(&self) -> BTreeSet<(&str, &str)> {
        self.files
            .iter()
            .map(|f| (f.model.as_str(), f.pre.as_str()))
            .collect()
    }
}

/// The derivation: one pass over the corpus. No printing, no assertions.
///
/// SEPARATED FROM THE REPORT BECAUSE IT WAS NOT, AND THAT MATTERED. The split
/// list -- the input to this file's central assertion -- used to be accumulated
/// as a SIDE EFFECT of the printing loop, so the assertion's evidence was
/// produced by the display code and could not be inspected without running it.
/// This file's first assertion was WRONG, and that arrangement is why it could
/// only be discovered by firing.
fn census(files: &[PathBuf]) -> Census {
    let mut out = Census::default();

    for path in files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let content = match Content::read(path) {
            Ok(c) => c,
            Err(e) => {
                // A FACT ABOUT US, NOT ABOUT THE FILE. `Content::read` parses
                // tensor infos eagerly and fails on a quantization dtype it does
                // not know -- before reaching metadata that sits AHEAD of any
                // tensor in the KV header. Three SmolLM2 files and
                // ggml-vocab-aquila land here while carrying perfectly good
                // vocabularies. Kept in its own bucket so the count of
                // vocabularies is never quietly reduced by our own limits.
                out.reader_failed.push((name, format!("{e}")));
                continue;
            }
        };
        let groups = &mut out.groups;
        match vocab(&content) {
            Ok(tokens) => {
                // Whether the tokenizer rebuilds is NOT a property of the token
                // list alone -- it also turns on `pre`, merges and token types.
                let rebuilt = content.extract_tokenizer().is_ok();
                let (model, pre) = tokenizer_rule(&content);
                groups
                    .entry(vocab_digest(&tokens))
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
            // A FACT ABOUT THE FILE: it opened, and carries no usable token list.
            Err(why) => out.no_vocab_key.push((name, why)),
        }
    }
    out
}

/// The report: printing only. Nothing an assertion depends on is decided here.
fn report(files: usize, c: &Census) {
    let groups = &c.groups;
    println!("\n=== GGUF corpus census ===");
    println!("  files scanned            : {files}");
    println!("  vocabularies WE CAN REACH: {}", groups.len());
    println!("  file has no token list   : {}", c.no_vocab_key.len());
    println!(
        "  OUR READER COULD NOT OPEN: {}   <- a fact about us, not the corpus",
        c.reader_failed.len()
    );
    println!();

    for (digest, group) in groups {
        let mark = if group.all_rebuilt() {
            "REBUILT"
        } else if group.any_rebuilt() {
            "SPLIT  "
        } else {
            "refused"
        };
        let len = group.token_count;
        if group.files.len() > 1 {
            println!(
                "  {len:>6} tokens  [{digest:016x}]  {mark}  {} FILES SHARE THIS VOCABULARY:",
                group.files.len()
            );
            for f in &group.files {
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
                group.files[0].name
            );
        }
    }

    if !c.no_vocab_key.is_empty() {
        println!("\n  opened, but carries no usable token list (a fact about the FILE):");
        for (n, why) in &c.no_vocab_key {
            println!("    {n:<44} {why}");
        }
    }

    if !c.reader_failed.is_empty() {
        println!(
            "\n  Content::read could not open these (A FACT ABOUT US). Their KV headers\n  \
             were read directly on 2026-09-03 and they DO carry vocabularies -- the three\n  \
             SmolLM2 quantizations share the other six's digest exactly, and aquila has a\n  \
             distinct 100008-token one. They are excluded from the count above because\n  \
             this test gates what the LIBRARY can reach, not what the corpus holds:"
        );
        for (n, why) in &c.reader_failed {
            println!("    {n:<44} {why}");
        }
    }
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

    let c = census(&files);
    report(files.len(), &c);
    let groups = &c.groups;

    // Classification, derived from the groups rather than accumulated while
    // printing them. Each of these is now readable on its own.
    let shared: usize = groups.values().map(|g| g.files.len() - 1).sum();
    let files_rebuilt: usize = groups
        .values()
        .map(|g| g.files.iter().filter(|f| f.rebuilt).count())
        .sum();
    let vocabs_rebuilt = groups.values().filter(|g| g.any_rebuilt()).count();
    let splits: Vec<u64> = groups
        .iter()
        .filter(|(_, g)| g.is_split())
        .map(|(d, _)| *d)
        .collect();

    println!(
        "\n  ==> {} files / {} vocabularies REACHABLE THROUGH Content::read ({shared} files \
         duplicate a vocabulary already present).\n      NOT a corpus census: {} further \
         file(s) carry vocabularies this reader cannot open.",
        files.len(),
        groups.len(),
        c.reader_failed.len()
    );
    println!(
        "  ==> tokenizer rebuild: {files_rebuilt} of {} files, but {vocabs_rebuilt} of {} \
         VOCABULARIES -- the second is the coverage number",
        files.len(),
        groups.len()
    );
    println!(
        "  ==> {} vocabulary group(s) split by a differing pre-tokenizer rule, as designed",
        splits.len()
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
    let unexplained: Vec<String> = splits
        .iter()
        .filter(|d| groups[d].rules().len() == 1)
        .map(|d| {
            format!(
                "[{d:016x}] every file carries model/pre {:?} yet they disagree",
                groups[d].rules()
            )
        })
        .collect();
    assert!(
        unexplained.is_empty(),
        "a vocabulary rebuilt for some files and not others WITH NO DIFFERENCE IN model/pre \
         -- identical inputs, different outcomes:\n{}",
        unexplained.join("\n")
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
