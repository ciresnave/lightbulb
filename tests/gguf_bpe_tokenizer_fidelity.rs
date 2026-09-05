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
//! rather than only exhibiting it.
//!
//! # ⚠️ THE RANDOMISED 100 ARE NOW IN THE REPO, AND THAT IS A NEW CORPUS
//!
//! This used to read "the other 100 are randomised and live in the verification
//! run, not in the repo" — which meant **eight allowlist entries recorded a
//! number nobody could reproduce.** A seed and a prose description of an alphabet
//! are not a corpus; they name a stream only the original generator can produce.
//!
//! `randomised_cases()` below is committed, deterministic, and dependency-free,
//! so every future claim over 130 cases is re-runnable by anyone holding this
//! file.
//!
//! **It is NOT the historical corpus.** Those cases are unrecoverable, so the
//! numbers taken over them stay as they are, annotated with when they were taken.
//! Restating them against this generator would have produced figures that LOOKED
//! like the originals and measured something else — the exact substitution this
//! module exists to refuse. **The eight are not wrong; they are unverifiable, and
//! those need different words.**
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
//!   refact           smallcloudai/Refact-1_6B-fim              ggml-vocab-refact.gguf
//!   deepseek-llm     deepseek-ai/deepseek-llm-7b-base          ggml-vocab-deepseek-llm.gguf
//! ```
//!
//! # ⚠️ A 0-of-130 result is not automatically evidence
//!
//! **`qwen35` scored 0 of 130 and is still REFUSED.** Its obvious reference,
//! `Qwen/Qwen3-8B`, declares a pre-tokenizer and vocab BYTE-IDENTICAL to
//! `Qwen/Qwen2-7B` — so it is a reference for `qwen2`, not for that name.
//! llama.cpp does define a distinct QWEN35 rule (`[\p{L}\p{M}]+` where qwen2
//! has `\p{L}+`), and **this corpus cannot tell the two apart: measured, they
//! differ on 0 of 130 cases**, because the qwen2 normalizer is NFC and composes
//! away the combining marks the difference turns on.
//!
//! So the score was real, the corpus was the one used for every other entry,
//! and the result carried **no information about which rule is correct**. The
//! gate that would have caught a wrong rule here is blind to this particular
//! distinction, and a passing score from a blind gate is not evidence.
//! See `Content::bpe_refusal_reason`.
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

/// The seed the randomised half of the corpus is generated from.
///
/// Stated here rather than in prose because a seed described in a comment is not
/// a corpus: it names a stream nobody else can produce without the generator.
const RANDOM_SEED: u64 = 20260902;

/// How many randomised cases accompany the hand-written ones.
const RANDOM_CASES: usize = 100;

/// The alphabet the randomised cases are drawn from.
///
/// Chosen to mix the categories the candidate pre-tokenizer regexes split on:
/// ASCII letters in both cases, digits, the whitespace shapes, punctuation
/// including the apostrophes GPT-2's contraction rule keys on, and non-Latin
/// scripts (Greek, Cyrillic, Arabic-Indic digits, CJK, Hangul, Hebrew, and a
/// combining mark — the class `qwen35`'s rule would have turned on).
const ALPHABET: &[char] = &[
    'a', 'b', 'c', 'd', 'e', 'z', 'A', 'B', 'C', 'D', 'E', 'Z', '0', '1', '2', '7', '9', ' ', ' ',
    '\t', '\n', '\'', '\u{2019}', '.', ',', '-', '_', '/', '\\', '|', '(', ')', '[', ']', '{', '}',
    '<', '>', '#', '@', '$', '%', '^', '&', '*', '+', '=', '~', '`', '"', ':', ';', '?', '!',
    '\u{3b1}', '\u{3c9}', '\u{416}', '\u{44f}', '\u{663}', '\u{664}', '\u{4e00}', '\u{9f9f}',
    '\u{3042}', '\u{d55c}', '\u{301}', '\u{5d0}',
];

/// SplitMix64 — a deterministic PRNG written out here so the corpus is
/// reproducible from this file alone, with no dependency and no ambient RNG.
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The randomised half of the corpus.
///
/// # ⚠️ THIS DOES NOT REPRODUCE THE HISTORICAL 100 CASES
///
/// Eight allowlist entries record "0 of 130". That 130 was the 30 below plus 100
/// randomised strings from a verification run **whose generator was never
/// committed** — only a seed and a prose description of its alphabet survived.
/// This is a NEW generator. Its cases are not those cases, and no number it
/// produces is comparable to a historical one.
///
/// Reconstructing the original from a seed and a description would have produced
/// a number that LOOKED like the others and measured something else, which is
/// the defect this file exists to prevent. So the historical entries keep their
/// number with a note about when it was taken, and everything from here forward
/// is reproducible by anyone holding this file.
fn randomised_cases() -> Vec<String> {
    let mut state = RANDOM_SEED;
    (0..RANDOM_CASES)
        .map(|_| {
            // 1..=24 characters: long enough to contain several category
            // transitions, short enough that a failure names a small string.
            let len = 1 + (next_u64(&mut state) % 24) as usize;
            (0..len)
                .map(|_| ALPHABET[(next_u64(&mut state) % ALPHABET.len() as u64) as usize])
                .collect()
        })
        .collect()
}

/// The randomised corpus must be able to DISCRIMINATE, or "0 disagreements" over
/// it means nothing.
///
/// A generator emitting 100 empty strings, or 100 copies of one string, passes
/// the fidelity gate exactly as a good one does. That is the shape this repo
/// keeps meeting: a clean result from an instrument that could not have produced
/// a dirty one. So the corpus's own adequacy is asserted rather than assumed.
///
/// Born-red verified: fixing the length to 1 collapses it to 50 distinct cases
/// and this fails.
#[test]
fn the_randomised_corpus_can_discriminate() {
    let cases = randomised_cases();
    assert_eq!(cases.len(), RANDOM_CASES, "wrong number of cases");
    assert!(
        cases.iter().all(|c| !c.is_empty()),
        "an empty case exercises nothing"
    );

    let distinct: std::collections::BTreeSet<&String> = cases.iter().collect();
    assert!(
        distinct.len() > RANDOM_CASES * 9 / 10,
        "only {} of {} cases are distinct -- a corpus that repeats itself is smaller \
         than it claims",
        distinct.len(),
        RANDOM_CASES
    );

    // Every category the candidate pre-tokenizers split on must actually occur,
    // or the corpus is blind to exactly the distinctions it exists to test.
    let all: String = cases.concat();
    for (name, present) in [
        ("ascii letter", all.chars().any(|c| c.is_ascii_alphabetic())),
        ("digit", all.chars().any(|c| c.is_ascii_digit())),
        ("whitespace", all.chars().any(char::is_whitespace)),
        ("punctuation", all.chars().any(|c| c.is_ascii_punctuation())),
        ("apostrophe", all.contains('\'') || all.contains('\u{2019}')),
        ("non-latin", !all.is_ascii()),
        ("combining mark", all.contains('\u{301}')),
    ] {
        assert!(present, "the randomised corpus contains no {name}");
    }

    // Deterministic across runs and machines: the whole point of committing it.
    assert_eq!(
        cases,
        randomised_cases(),
        "the generator is not deterministic, so no number taken over it is reproducible"
    );
}

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
/// The `tokenizer.ggml.pre` a GGUF declares, or `"<absent>"`.
fn declared_pre(path: &str) -> String {
    use candlelight::core::quantized::gguf_file::Value;
    lightbulb::gguf::Content::read(path)
        .ok()
        .and_then(|c| match c.metadata().get("tokenizer.ggml.pre") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "<absent>".to_string())
}

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

    // COVERAGE FIRST, FIDELITY SECOND.
    //
    // An unset or short `LIGHTBULB_BPE_PAIRS` used to narrow this gate silently:
    // it would check whatever it was given, report success, and say nothing
    // about the allowlist entries it never touched. That is the shape this file
    // exists to prevent, one level up — a table of seven entries exercised by
    // one checkpoint is six entries wearing the seventh's evidence.
    let covered: std::collections::BTreeSet<String> =
        pairs.iter().map(|(gguf, _)| declared_pre(gguf)).collect();
    let missing: Vec<&str> = lightbulb::gguf::Content::verified_pre_values()
        .iter()
        .copied()
        .filter(|pre| !covered.contains(*pre))
        .collect();
    assert!(
        missing.is_empty(),
        "these verified `tokenizer.ggml.pre` values have NO fixture in this run, so their entries are unevidenced: {missing:?}. Supplied pairs cover: {covered:?}"
    );

    let mut diffs = Vec::new();
    for (gguf, reference) in &pairs {
        let ours = lightbulb::gguf::Content::read(gguf)
            .expect("reading the GGUF")
            .extract_tokenizer()
            .unwrap_or_else(|e| panic!("rebuilding the tokenizer for {gguf}: {e:#}"));
        let reference_tok =
            tokenizers::Tokenizer::from_file(reference).expect("reading the reference");
        // The hand-written hostile cases plus the committed randomised ones. The
        // size is asserted after the loop: a corpus that silently shrinks turns
        // this gate green over less than it claims.
        let random = randomised_cases();
        let all: Vec<&str> = CASES
            .iter()
            .copied()
            .chain(random.iter().map(String::as_str))
            .collect();
        for case in &all {
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
        assert_eq!(
            all.len(),
            CASES.len() + RANDOM_CASES,
            "the corpus is not the size it claims, so a clean result would be over \
             fewer cases than reported"
        );
        eprintln!(
            "  checked {} cases ({} hand-written + {} randomised, seed {RANDOM_SEED}) \
             against {reference}",
            all.len(),
            CASES.len(),
            random.len()
        );
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
    // ⚠️ THIS USED TO READ `msg.contains("verified") || msg.contains("Refusing")`
    // AND PASSED BY ACCIDENT. There are two refusal messages: one for a `pre`
    // with a recorded reason ("which this build refuses ... Verified values:")
    // and one for a merely-unknown `pre` ("has not verified ... Refusing to
    // substitute"). The second satisfies both old substrings; the FIRST
    // satisfies neither, and matched only because one entry's explanatory prose
    // happened to contain the phrase "which IS verified here".
    //
    // Measured: `qwen35` and `<absent>` have ZERO occurrences of a lowercase
    // "verified" in their reasons, so pointing this test at either of them
    // would have failed — and splitting `starcoder` out of the combined
    // starcoder|mpt reason removed the accidental word and broke it.
    //
    // Asserting on what BOTH templates guarantee instead: they say they are
    // refusing, and they name the set they would have accepted.
    let says_refusing = msg.to_lowercase().contains("refus");
    let names_the_allowlist = lightbulb::gguf::Content::verified_pre_values()
        .iter()
        .any(|v| msg.contains(v));
    assert!(
        says_refusing,
        "the refusal must say it is declining rather than approximating: {msg}"
    );
    assert!(
        names_the_allowlist,
        "the refusal must name what it WOULD have accepted, so the reader can see \
         the gap rather than only the rejection: {msg}"
    );
}
