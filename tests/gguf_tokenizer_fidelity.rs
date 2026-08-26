//! A GGUF's rebuilt tokenizer must be byte-identical to the checkpoint's own.
//!
//! # Why this exists
//!
//! `Content::extract_tokenizer` used to build a `Unigram` model whose scores
//! were INVENTED as `-(id as f64)` — the negative token index — while the
//! GGUF's real `tokenizer.ggml.scores` and `tokenizer.ggml.merges` sat unread.
//! Unigram picks the segmentation maximising total score, so fabricated scores
//! made short low-id pieces always win. Measured on
//! `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` @ `52e7645b`, Q4_0, for the prompt
//! the sibling `gguf_serving_e2e` test sends:
//!
//! ```text
//!                  old                        reference
//! id count         28                         22
//! "user"           us + er                    user    (1792)
//! "capital"        c + ap + it + al           capital (7483)
//! "France"         F + ran + ce               France  (3444)
//! newline          id 0 — THE UNK TOKEN       <0x0A>  (13)
//! BOS w/ specials  absent                     <s>     (1)
//! ```
//!
//! The model was fed UNK for every newline and shattered subwords throughout.
//!
//! # Why an equality assertion rather than a spot check
//!
//! Every earlier attempt to characterise this defect asserted something weaker
//! — that a prompt rendered correctly, that resolution picked the right tier —
//! and each passed while the product stayed broken, because **a correct
//! rendered string is not a correct token sequence**. The only assertion that
//! could have caught it is the one made here: the ids, in order, equal the
//! reference's.
//!
//! # This test has a proven red state
//!
//! Against the pre-fix `extract_tokenizer` it fails on the first arm with 28
//! ids against 22. It is not a test that has only ever been observed passing.
//!
//! Run:
//! ```text
//! LIGHTBULB_GGUF=<...>.gguf LIGHTBULB_REF_TOKENIZER=<...>/tokenizer.json \
//!   cargo test --test gguf_tokenizer_fidelity -- --ignored --nocapture
//! ```

/// The exact prompt `gguf_serving_e2e` sends, rendered through the
/// checkpoint's own chat template. Shared shape deliberately: this file asks
/// what the TOKENIZER does with that prompt, the sibling asks what the MODEL
/// does with it, and the two only compose if the string is the same one.
const PROMPT: &str = "<|user|>\nName the capital of France.</s>\n<|assistant|>\n";

/// Both paths, from environment only — a 637 MB checkpoint has no correct
/// default, and an earlier hard-coded fallback in a sibling test leaked a
/// developer's username into a public repository.
fn paths() -> Option<(String, String)> {
    let gguf = std::env::var("LIGHTBULB_GGUF").ok()?;
    let reference = std::env::var("LIGHTBULB_REF_TOKENIZER").ok()?;
    (std::path::Path::new(&gguf).is_file() && std::path::Path::new(&reference).is_file())
        .then_some((gguf, reference))
}

#[test]
#[ignore = "needs the GGUF checkpoint and the reference tokenizer.json"]
fn a_ggufs_rebuilt_tokenizer_matches_the_checkpoints_own_ids() {
    let (gguf, reference) = paths().expect(
        "set LIGHTBULB_GGUF to a .gguf and LIGHTBULB_REF_TOKENIZER to the matching tokenizer.json",
    );

    let content = lightbulb::gguf::Content::read(&gguf).expect("reading the GGUF");
    let ours = content
        .extract_tokenizer()
        .expect("rebuilding the tokenizer from GGUF metadata");
    let theirs = tokenizers::Tokenizer::from_file(&reference).expect("reading the reference");

    // Both arms. `add_special_tokens = true` is the one that exercises the
    // post-processor, whose absence made the flag inert: before the fix BOTH
    // arms returned the same 28 ids, so a single-arm test could not have seen
    // that the flag did nothing.
    for add_special in [false, true] {
        let got = ours
            .encode(PROMPT, add_special)
            .expect("encoding with ours");
        let want = theirs
            .encode(PROMPT, add_special)
            .expect("encoding with the reference");

        assert_eq!(
            got.get_ids(),
            want.get_ids(),
            "add_special_tokens={add_special}: rebuilt tokenizer disagrees with the \
             checkpoint's own.\n  ours ({} ids): {:?}\n  ours toks: {:?}\n  ref  ({} ids): {:?}\n  \
             ref  toks: {:?}",
            got.get_ids().len(),
            got.get_ids(),
            got.get_tokens(),
            want.get_ids().len(),
            want.get_ids(),
            want.get_tokens(),
        );
    }

    // Named separately from the equality above because it is the specific
    // regression that made every newline in a chat prompt an UNK token, and a
    // future change could restore the id count while losing byte fallback.
    let ids = ours.encode(PROMPT, false).unwrap();
    let unk = content
        .metadata()
        .get("tokenizer.ggml.unknown_token_id")
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0);
    assert!(
        !ids.get_ids().contains(&unk),
        "the prompt tokenized to at least one UNK (id {unk}); byte fallback is off: {:?}",
        ids.get_tokens()
    );
}
