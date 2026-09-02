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
//! punctuation and non-Latin scripts — nothing scored perfectly.
//!
//! # ⚠️ THOSE SCORES ARE AGAINST llama.cpp, WHICH IS NOT GROUND TRUTH
//!
//! **Read `128/130` as an OPEN QUESTION, not as a near-miss to close.** It is
//! consistent with two different situations and this measurement cannot tell
//! them apart:
//!
//! - we are wrong on 2 cases, or
//! - **llama.cpp is wrong on 2 cases and we are right.**
//!
//! The second is not hypothetical. Measured three ways over these same 130
//! cases on the SmolLM2 GGUF:
//!
//! ```text
//!   ours    vs HF reference (tokenizer.json)   0 disagree
//!   ours    vs llama.cpp                       2 disagree
//!   HF ref  vs llama.cpp                       2 disagree   <- THE SAME TWO
//! ```
//!
//! **llama.cpp disagrees with the checkpoint's own declared tokenizer, and we
//! agree with it.** Its SMOLLM regex omits the trailing `|\s+` alternative that
//! the declared `ByteLevel` rule carries, and its source comments the rewrite
//! directly (`// original regex from tokenizer.json`, above a rewritten
//! expression, for the qwen types).
//!
//! So the earlier conclusion that these values "need llama.cpp's own per-`pre`
//! regexes" is WRONG and is corrected here: adopting them would deliberately
//! reproduce a reference measured to differ from the checkpoints, and a perfect
//! score against it would be the worst outcome available — wrong in exactly the
//! places it is wrong, while looking like success.
//!
//! **Settling any of these needs that checkpoint's own `tokenizer.json`**, the
//! way `smollm` was settled. Until then they stay refused, and the reason is
//! "not verified against an authoritative reference" rather than "does not match
//! llama.cpp".
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
//! # What each `pre` is verified against
//!
//! Every allowlisted value is gated id-for-id against THAT CHECKPOINT'S OWN
//! `tokenizer.json` — not against llama.cpp, for the reason above. Fetch each
//! from `https://huggingface.co/<repo>/resolve/main/tokenizer.json`:
//!
//! ```text
//!   pre              reference repo                            corpus GGUF
//!   smollm           HuggingFaceTB/SmolLM2-360M-Instruct       SmolLM2-135M-Instruct-*
//!   gpt-2            openai-community/gpt2                     ggml-vocab-gpt-2.gguf
//!   falcon           tiiuae/falcon-7b                          ggml-vocab-falcon.gguf
//!   qwen2            Qwen/Qwen2-7B                             ggml-vocab-qwen2.gguf
//!   deepseek-coder   deepseek-ai/deepseek-coder-6.7b-instruct  ggml-vocab-deepseek-coder.gguf
//! ```
//!
//! **A reference is only admissible once its vocab matches the GGUF's.** All
//! five were checked before use: `gpt-2` and `falcon` match exactly; `qwen2`
//! (151936 vs 151643) and `deepseek-coder` (32256 vs 32000) carry extra GGUF
//! tokens, and those were confirmed to sit ENTIRELY AT THE TAIL — every id
//! below the reference's vocab size matches — so ordinary text cannot reach
//! them. `smollm`'s substitution is the 360M reference for 135M files, verified
//! byte-identical on vocab and merges; see the caveat at the allowlist entry.
//!
//! Run:
//! ```text
//! LIGHTBULB_BPE_PAIRS="<a>.gguf|<a>.json;<b>.gguf|<b>.json" \
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

/// The (GGUF, reference `tokenizer.json`) pairs to gate.
///
/// `LIGHTBULB_BPE_PAIRS` takes several, as `gguf|reference` entries separated by
/// `;`, so one run covers every allowlisted `tokenizer.ggml.pre`. **A table of
/// five entries verified by one checkpoint would be four unverified entries
/// wearing the fifth's evidence.**
///
/// Falls back to the single-pair variables when it is unset.
fn pairs() -> Vec<(String, String)> {
    if let Ok(list) = std::env::var("LIGHTBULB_BPE_PAIRS") {
        return list
            .split(';')
            .filter(|e| !e.trim().is_empty())
            .filter_map(|e| e.split_once('|'))
            .map(|(g, r)| (g.trim().to_string(), r.trim().to_string()))
            .filter(|(g, r)| std::path::Path::new(g).is_file() && std::path::Path::new(r).is_file())
            .collect();
    }
    let (Ok(gguf), Ok(reference)) = (
        std::env::var("LIGHTBULB_BPE_GGUF"),
        std::env::var("LIGHTBULB_BPE_REF_TOKENIZER"),
    ) else {
        return Vec::new();
    };
    if std::path::Path::new(&gguf).is_file() && std::path::Path::new(&reference).is_file() {
        vec![(gguf, reference)]
    } else {
        Vec::new()
    }
}

/// The ids, in order, equal the reference's — on every case.
///
/// An equality assertion rather than a spot check, for the reason the sibling
/// file records: every weaker assertion tried against the SPM defect passed
/// while the product stayed broken.
#[test]
#[ignore = "needs a gpt2-family GGUF and its reference tokenizer.json"]
fn a_byte_level_ggufs_rebuilt_tokenizer_matches_the_checkpoints_own_ids() {
    let pairs = pairs();
    assert!(
        !pairs.is_empty(),
        "set LIGHTBULB_BPE_PAIRS to `gguf|tokenizer.json` entries separated by `;`, or the single-pair LIGHTBULB_BPE_GGUF / LIGHTBULB_BPE_REF_TOKENIZER"
    );
    let mut diffs = Vec::new();
    for (gguf, reference) in &pairs {
        let ours = lightbulb::gguf::Content::read(gguf)
            .expect("reading the GGUF")
            .extract_tokenizer()
            .unwrap_or_else(|e| panic!("rebuilding the tokenizer for {gguf}: {e:#}"));
        let reference_tok =
            tokenizers::Tokenizer::from_file(reference).expect("reading the reference");
        for case in CASES {
            let got = ours.encode(*case, false).expect("ours").get_ids().to_vec();
            let want = reference_tok
                .encode(*case, false)
                .expect("ref")
                .get_ids()
                .to_vec();
            if got != want {
                diffs.push(format!(
                    "  {gguf}\n    {case:?}\n      ours={got:?}\n       ref={want:?}"
                ));
            }
        }
        eprintln!("  checked {} cases against {reference}", CASES.len());
    }
    assert!(
        diffs.is_empty(),
        "{} disagreements with the checkpoints' own tokenizers across {} pair(s):\n{}",
        diffs.len(),
        pairs.len(),
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
