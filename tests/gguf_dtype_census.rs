//! Which tensor dtypes each corpus checkpoint actually uses, and which of them
//! this build cannot represent.
//!
//! # Why this exists
//!
//! `tests/gguf_serving_census.rs` reports that three corpus files carry weights
//! and are refused for an unrepresentable tensor dtype. It does not say WHICH
//! dtype, so the refusal was a verdict with no table behind it — and a verdict
//! with no table is exactly what gets misread.
//!
//! # ⚠️ IT WAS MISREAD, BY ME, AND THAT IS WHAT THIS FILE PREVENTS
//!
//! candle-core 0.10.2 refuses an unknown quantization with
//!
//! ```text
//! bail!("unknown dtype for tensor {u}")   // `u` is the DTYPE, not the index
//! ```
//!
//! **The sentence's grammar attaches the number to the wrong noun.** Reading
//! `unknown dtype for tensor 21` as "tensor #21" is the natural reading and it
//! is wrong. On that misreading I published — in a merged PR and in ROADMAP —
//! that *"the tensor indices DIFFER … these files MIX quantizations, and one
//! tensor in each uses a type candle rejects."*
//!
//! Measured, all three claims fail:
//!
//! - the numbers are dtype codes, not indices
//! - it is 180–210 tensors of 272, not "one tensor in each"
//! - ⚠️ **mixing is UNIVERSAL and therefore not the discriminator.** The `Q4_0`
//!   control mixes four dtypes and loads perfectly. A property every file in
//!   the corpus has was offered as the explanation for why three of them fail.
//! - ⚠️ and the same sentence read a QUANTISATION OFF A FILENAME.
//!   `SmolLM2-135M-Instruct-Q2_K.gguf` contains NO Q2_K tensors — code `10` is
//!   absent from its directory entirely. Reading a quantisation off a filename
//!   is a guess, and this corpus falsifies it.
//!
//! **A control validates the instrument; it cannot tell you that you misread
//! the instrument's output.** I had a control and stated my uncertainty, and
//! neither helped, because the error was in the INPUT to the inference.
//!
//! # Corroboration, and exactly how independent it is
//!
//! The mlmf lane reached the same conclusion within the same ten minutes,
//! through `mlmf-gguf`'s own reader rather than by reading candle's source, and
//! got the same three numbers on the same three files.
//!
//! ⚠️ **Stated at true strength: the METHODS are independent and the ARTIFACT is
//! not.** Both lanes read the same bytes under `C:\Models`. So —
//!
//! - *"the number is a type code"* is corroborated independently: two different
//!   libraries, two different code paths, neither reading the other's source.
//! - *"these three files contain these dtypes"* is NOT independently
//!   corroborated. It is one artifact read twice, and a mislabelled corpus
//!   would produce agreement.
//!
//! ```text
//! LIGHTBULB_GGUF_CORPUS=<dir> cargo test --test gguf_dtype_census -- --ignored --nocapture
//! ```

use lightbulb::gguf::Content;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The GGML type codes `candle-core` 0.10.2 can represent, read off
/// `GgmlDType::from_u32`. Everything else is refused — this is a TABLE, not a
/// range, and the gaps are real (4, 5 are deprecated; 16..29 are the IQ
/// codebook family plus the integer and f64 types).
const CANDLE_ACCEPTS: &[u32] = &[0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 30];

fn corpus() -> Option<(Vec<PathBuf>, Vec<String>)> {
    let root = PathBuf::from(std::env::var_os("LIGHTBULB_GGUF_CORPUS")?);
    let mut out = Vec::new();
    let mut unreadable = Vec::new();
    let mut stack = vec![root];
    while let Some(d) = stack.pop() {
        // Reported, never skipped: a partial walk understates the population
        // and reports success. See `gguf_serving_census.rs`, where dropping
        // this exact guard was the defect.
        let entries = match std::fs::read_dir(&d) {
            Ok(e) => e,
            Err(e) => {
                unreadable.push(format!("{}: {e}", d.display()));
                continue;
            }
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
            {
                out.push(p);
            }
        }
    }
    out.sort();
    Some((out, unreadable))
}

/// Dtype code -> tensor count, taken from OUR parser, which reads the codes
/// without validating them. candle cannot supply this: it refuses the whole
/// directory on the first code it does not know, so the file that cannot be
/// loaded is exactly the file candle cannot describe.
fn dtype_histogram(c: &Content) -> Option<BTreeMap<u32, usize>> {
    let infos = c.lightning_tensor_infos().ok()?;
    let mut h = BTreeMap::new();
    for t in infos {
        *h.entry(t.tensor_type).or_insert(0usize) += 1;
    }
    Some(h)
}

#[test]
#[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
fn the_refused_number_is_a_dtype_code_and_not_a_tensor_index() {
    let Some((files, unreadable_dirs)) = corpus() else {
        lightbulb::test_notice::skip_unless_required(
            "LIGHTBULB_REQUIRE_CORPUS",
            "set LIGHTBULB_GGUF_CORPUS to a directory containing .gguf files",
        );
        return;
    };
    assert!(
        unreadable_dirs.is_empty(),
        "the corpus walk was incomplete: {unreadable_dirs:?}"
    );
    assert!(!files.is_empty(), "an empty corpus proves nothing here");

    // The two arms this test discriminates between. Both must be non-empty or
    // the comparison below is one-sided and vacuous.
    let mut accepted = Vec::new();
    let mut refused = Vec::new();
    // Weighted files this census cannot describe -- a fact about OUR parser
    // (it refuses GGUF v1), reported rather than dropped from a population.
    let mut no_histogram: Vec<String> = Vec::new();
    let mut vocab_only = 0usize;
    let mut unreadable_file = 0usize;

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let Ok(c) = Content::read(path) else {
            unreadable_file += 1;
            continue;
        };
        let Some(hist) = dtype_histogram(&c) else {
            // NOT A SILENT SKIP: a v1 file carries weights and has no
            // histogram here, and would otherwise vanish from a census whose
            // entire subject is a population.
            no_histogram.push(name);
            continue;
        };
        if hist.is_empty() {
            vocab_only += 1;
            continue; // vocabulary-only; no tensors to type
        }
        let unsupported: Vec<u32> = hist
            .keys()
            .copied()
            .filter(|d| !CANDLE_ACCEPTS.contains(d))
            .collect();
        let candle = c.tensor_infos().err().map(|e| e.to_string());
        let shown: Vec<String> = hist.iter().map(|(d, n)| format!("{d}:{n}")).collect();
        println!(
            "  {name:<38} {:>4} tensors  {}  unsupported={unsupported:?}",
            hist.values().sum::<usize>(),
            shown.join(" ")
        );

        match candle {
            None => {
                assert!(
                    unsupported.is_empty(),
                    "{name} was ACCEPTED by candle while carrying dtypes it does not list: {unsupported:?}"
                );
                accepted.push(name);
            }
            Some(msg) => {
                assert!(
                    !unsupported.is_empty(),
                    "{name} was REFUSED while every dtype it carries is in candle's table -- the refusal is about something else: {msg}"
                );
                // ⚠️ THE CLAIM THIS FILE EXISTS FOR. The number candle prints is
                // a DTYPE, so it must appear in the unsupported set. Under the
                // "tensor index" misreading there is no reason it would.
                let reported: Vec<u32> = msg
                    .split_whitespace()
                    .filter_map(|w| w.trim_end_matches('.').parse::<u32>().ok())
                    .collect();
                assert!(
                    reported.iter().any(|r| unsupported.contains(r)),
                    "candle's refusal for {name} names no dtype from its unsupported set {unsupported:?}; numbers seen in the message were {reported:?}. Either the message changed or the number is not a dtype after all: {msg}"
                );
                refused.push((name, unsupported));
            }
        }
    }

    println!("\n  accepted by candle: {}", accepted.len());
    println!(
        "  no histogram:       {}  {no_histogram:?}",
        no_histogram.len()
    );
    println!("  refused:            {}", refused.len());
    for (n, u) in &refused {
        println!("      {n}  unsupported dtypes {u:?}");
    }

    // ⚠️ BOTH ARMS OR NOTHING. A corpus where every file is accepted makes the
    // refusal assertion unreachable, and one where every file is refused makes
    // the acceptance assertion unreachable. Either way the test would pass
    // having compared nothing -- and a one-sided population is the shape that
    // let the original misreading stand.
    assert!(
        !accepted.is_empty(),
        "no file was accepted, so the acceptance arm never ran and this test compared nothing"
    );
    // Every file lands in exactly one bucket. A conservation law is not a
    // coverage assertion -- see issue #80 -- but it does catch a file being
    // silently dropped by some future `continue`, which is how a population
    // shrinks without anything going red.
    assert_eq!(
        accepted.len() + refused.len() + no_histogram.len() + vocab_only + unreadable_file,
        files.len(),
        "some corpus file reached none of the buckets, so the census lost it"
    );

    assert!(
        !refused.is_empty(),
        "no file was refused, so the dtype-code assertion never ran -- the claim this file exists to hold is untested on this corpus"
    );
}
