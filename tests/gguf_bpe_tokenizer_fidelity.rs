//! A byte-level BPE (`gpt2`) GGUF's rebuilt tokenizer must be byte-identical to
//! the checkpoint's own.
//!
//! # Why this is a separate file from `gguf_tokenizer_fidelity`
//!
//! That file gates the SentencePiece path. This one gates a **structurally
//! different** reconstruction, not a variant of it: no normalizer, no
//! `byte_fallback`, a `ByteLevel` decoder rather than ByteFallback/Fuse/Strip,
//! and a pre-tokenizer selected per checkpoint by `tokenizer.ggml.pre`.
//! Sharing one file would mean one set of fixtures standing in for two
//! pipelines that fail in different ways.
//!
//! # `tokenizer.ggml.pre` is the whole difficulty, and 10 cases is not enough
//!
//! `pre` names a splitting rule; llama.cpp keeps a different regex per name.
//! Measured over the local corpus: **18 `gpt2` files carrying 13 distinct `pre`
//! values.**
//!
//! # Corpus size changed the answer twice, in the same direction
//!
//! Candidate pre-tokenizers were scored against llama.cpp b10757 (`llama-tokenize
//! -m <vocab>.gguf -bf <case>`, which reads the case as raw bytes so no shell
//! mangling can alter it) over the `ggml-vocab-*.gguf` files:
//!
//! ```text
//!   pre                       10 cases   30 cases   130 cases
//!   falcon                      10/10      28/30      122/130
//!   gpt-2                       10/10      30/30      128/130
//!   command-r/refact/starcoder  10/10      30/30      128/130
//!   mpt                          9/10      27/30      123/130
//! ```
//!
//! **At 10 cases `falcon` looked perfect and was not. At 30, `gpt-2` looked
//! perfect and was not.** At 130 — the 30 below plus 100 seeded-random strings
//! (seed 20260902) drawn from an alphabet mixing letters, digits, whitespace,
//! punctuation and non-Latin scripts — NOTHING scored perfectly. Those values
//! need llama.cpp's own per-`pre` regexes, which this crate's `ByteLevel` does
//! not implement, so they stay refused.
//!
//! **A tokenizer right 128 times in 130 is the exact failure this module exists
//! to prevent**, and each larger corpus caught a case the previous one called
//! clean. That is why the supported-`pre` table is a measured allowlist rather
//! than a default, and why this corpus is deliberately hostile.
//!
//! `smollm` is the one entry that survives: **0 of 130 cases disagree** with
//! `SmolLM2-360M-Instruct/tokenizer.json`, which is stronger evidence than the
//! llama.cpp differential because the reference DECLARES its pre-tokenizer
//! (`Sequence[Digits{individual_digits}, ByteLevel{add_prefix_space:false}]`)
//! rather than only exhibiting it. The 30 cases below are the shipped gate; the
//! other 100 are randomised and live in the verification run, not in the repo.
//!
//! Both halves of that `Sequence` are load-bearing and the gate proves it:
//! dropping `Digits` fails 1 of these 30 cases. Under the older 10-case corpus
//! it failed none — the same lesson one level down.
//!
//! Run:
//! ```text
//! LIGHTBULB_BPE_GGUF=<...>.gguf LIGHTBULB_BPE_REF_TOKENIZER=<...>/tokenizer.json \
//!   cargo test --test gguf_bpe_tokenizer_fidelity -- --ignored --nocapture
//! ```

/// Deliberately hostile. Each group targets a place where the candidate
/// pre-tokenizer regexes disagree; a corpus of ordinary prose does not
/// distinguish them, which is how a 28-of-30 tokenizer scored 10 of 10.
const CASES: &[&str] = &[
    // ordinary prose — the cases that do NOT discriminate
    "The capital of France is Paris.",
    "MixedCASE camelCase snake_case SCREAMING",
    // contractions: GPT-2's regex matches only the LOWERCASE forms
    "don't we've I'll can't O'Brien",
    "DON'T WE'VE THEY'LL I'M",
    "It\u{2019}s a \u{2018}quoted\u{2019} \u{201c}string\u{201d}",
    // digit grouping — where an individual-digit split shows up
    "12345 3.14 007 1,000,000",
    "123456789012345678901234567890",
    "a1b2c3d4 x9y8z7",
    "\u{663}\u{664}\u{665} \u{4e00}\u{4e8c}\u{4e09} \u{bd} \u{be} \u{2168}",
    // whitespace shapes
    "a  b\t\tc\n\nd   ",
    " leading and trailing ",
    "  \t \n  ",
    "hello\r\nworld\r\n",
    "\n\n\n\n",
    "word ",
    " word",
    // symbols, code, URLs
    "fn main() { let x = 1; }",
    "!!!???...---___",
    "https://example.com/path?q=1&r=2#frag",
    "a-b_c.d/e\u{5c}f|g~h`i",
    "def f(x): return x**2  # comment",
    "SELECT * FROM t WHERE a>=1 AND b<>2;",
    "{\"k\": [1, 2, {\"n\": null}]}",
    // unicode shapes
    "caf\u{e9} \u{65e5}\u{672c}\u{8a9e} \u{1F389} na\u{ef}ve",
    "\u{1F389}\u{1F389}\u{1F389}",
    "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}\u{4e16}\u{754c}",
    "\u{ff21}\u{ff22}\u{ff23} \u{ff11}\u{ff12}\u{ff13}",
    "e\u{301} vs \u{e9}",
    "\u{200b}zero\u{200b}width\u{200b}",
    // special-token text must not be swallowed as ordinary characters
    "<|im_start|>user\nhi<|im_end|>",
];

fn paths() -> Option<(String, String)> {
    let gguf = std::env::var("LIGHTBULB_BPE_GGUF").ok()?;
    let reference = std::env::var("LIGHTBULB_BPE_REF_TOKENIZER").ok()?;
    (std::path::Path::new(&gguf).is_file() && std::path::Path::new(&reference).is_file())
        .then_some((gguf, reference))
}

/// The ids, in order, equal the reference's — on every case.
///
/// An equality assertion rather than a spot check, for the reason the sibling
/// file records: every weaker assertion tried against the SPM defect passed
/// while the product stayed broken.
#[test]
#[ignore = "needs a gpt2-family GGUF and its reference tokenizer.json"]
fn a_byte_level_ggufs_rebuilt_tokenizer_matches_the_checkpoints_own_ids() {
    let (gguf, reference) = paths().expect(
        "set LIGHTBULB_BPE_GGUF to a gpt2-family .gguf and LIGHTBULB_BPE_REF_TOKENIZER to the matching tokenizer.json",
    );
    let ours = lightbulb::gguf::Content::read(&gguf)
        .expect("reading the GGUF")
        .extract_tokenizer()
        .expect("rebuilding the tokenizer");
    let reference = tokenizers::Tokenizer::from_file(&reference).expect("reading the reference");

    let mut diffs = Vec::new();
    for case in CASES {
        let got = ours.encode(*case, false).expect("ours").get_ids().to_vec();
        let want = reference
            .encode(*case, false)
            .expect("ref")
            .get_ids()
            .to_vec();
        if got != want {
            diffs.push(format!("  {case:?}\n    ours={got:?}\n     ref={want:?}"));
        }
    }
    assert!(
        diffs.is_empty(),
        "{} of {} cases disagree with the checkpoint's own tokenizer:\n{}",
        diffs.len(),
        CASES.len(),
        diffs.join("\n")
    );
}

/// An unverified `tokenizer.ggml.pre` is REFUSED, and the refusal names it.
///
/// The paired half of the test above. Without it, "SmolLM2 matches" would be
/// satisfied by an implementation that accepted every `pre` value and happened
/// to be right for this one — which is precisely the 28-of-30 failure mode.
/// A message that does not name the offending value sends the reader looking
/// for a defect in the vocab instead of at a one-line table.
#[test]
#[ignore = "needs a gpt2-family GGUF with an unverified `pre`"]
fn an_unverified_pre_tokenizer_is_refused_by_name() {
    let Ok(gguf) = std::env::var("LIGHTBULB_BPE_UNVERIFIED_GGUF") else {
        panic!("set LIGHTBULB_BPE_UNVERIFIED_GGUF to a gpt2 GGUF whose `pre` is not in the table")
    };
    let content = lightbulb::gguf::Content::read(&gguf).expect("reading the GGUF");
    let err = content
        .extract_tokenizer()
        .expect_err("an unverified `pre` must be refused, not approximated");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("tokenizer.ggml.pre"),
        "the refusal must name the field that caused it: {msg}"
    );
    assert!(
        msg.contains("verified") || msg.contains("Refusing"),
        "the refusal must say it is declining to approximate: {msg}"
    );
}
