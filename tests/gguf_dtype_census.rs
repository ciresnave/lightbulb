// SPDX-License-Identifier: MIT OR Apache-2.0
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

/// The GGML type codes the live reader can represent, read off
/// `GgmlDType::from_u32` (`fuel-ir/src/quantized.rs`, Fuel 0.20.61). This is a
/// TABLE, not a range, and the remaining gaps are real (4, 5 are deprecated;
/// no code above 40 is assigned).
///
/// Was candle-core 0.10.2's table (`&[0, 1, 2, 3, 6..=15, 30]`) until two
/// changes on 2026-10-22: PR #195 removed candlelight entirely (Fuel is now
/// the only reader this crate has), and the Fuel 0.20.61 crates.io migration
/// (PR #196) confirmed GAP-211 B1 widened Fuel's own table to the set below —
/// the entire IQ family (16-23, 29), TQ (34, 35), and MXFP4/NVFP4 (39, 40),
/// on top of GAP-211 B2's decode-kernel work for IQ2/IQ3/IQ1 already landed
/// at `0.20.14`. This table is TABLE acceptance (`from_u32` returns `Ok`);
/// whether tensor data actually decodes correctly for a given type is a
/// separate claim this census does not make.
const FUEL_ACCEPTS: &[u32] = &[
    0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 34, 35,
    39, 40,
];

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

/// What one corpus file turned out to be. Each variant is a DIFFERENT fact,
/// and the tally keeps them apart so no bucket absorbs another.
enum Row {
    /// `Content::read` refused.
    Unreadable,
    /// Carries weights but our parser cannot type them (it refuses GGUF v1).
    /// NOT a silent skip -- reported, because it is a fact about OUR reader and
    /// would otherwise vanish from a census whose subject is a population.
    NoHistogram(String),
    /// Zero tensors: a vocabulary fixture, nothing to type.
    VocabOnly,
    Accepted(String),
    Refused(String, Vec<u32>),
}

/// Read one file, print its histogram, and say which row it is.
///
/// The per-file ASSERTIONS live HERE rather than in the tally, at the point
/// where both facts are in hand: what candle said, and what the file actually
/// contains. Separating them would mean carrying candle's message through the
/// `Row` type purely so the check could happen elsewhere.
///
/// ⚠️ 47 code lines against Codacy's limit of 50 — three lines of headroom.
/// The next thing added here goes in a new function, not on the end of this
/// one.
fn row_for(path: &PathBuf) -> Row {
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    let Ok(c) = Content::read(path) else {
        return Row::Unreadable;
    };
    let Some(hist) = dtype_histogram(&c) else {
        return Row::NoHistogram(name);
    };
    if hist.is_empty() {
        return Row::VocabOnly;
    }
    let unsupported: Vec<u32> = hist
        .keys()
        .copied()
        .filter(|d| !FUEL_ACCEPTS.contains(d))
        .collect();
    let shown: Vec<String> = hist.iter().map(|(d, n)| format!("{d}:{n}")).collect();
    println!(
        "  {name:<38} {:>4} tensors  {}  unsupported={unsupported:?}",
        hist.values().sum::<usize>(),
        shown.join(" ")
    );
    match c.tensor_infos().err().map(|e| e.to_string()) {
        None => {
            assert!(
                unsupported.is_empty(),
                "{name} was ACCEPTED by the live reader while carrying dtypes it does not list: {unsupported:?}"
            );
            Row::Accepted(name)
        }
        Some(msg) => {
            assert!(
                !unsupported.is_empty(),
                "{name} was REFUSED while every dtype it carries is in the live reader's table -- the refusal is about something else: {msg}"
            );
            // ⚠️ THE CLAIM THIS FILE EXISTS FOR. The number candle prints is a
            // DTYPE, so it must appear in the unsupported set. Under the
            // "tensor index" misreading there is no reason it would.
            let reported: Vec<u32> = msg
                .split_whitespace()
                .filter_map(|w| w.trim_end_matches('.').parse::<u32>().ok())
                .collect();
            assert!(
                reported.iter().any(|r| unsupported.contains(r)),
                "the live reader's refusal for {name} names no dtype from its unsupported set {unsupported:?}; numbers seen in the message were {reported:?}. Either the message changed or the number is not a dtype after all: {msg}"
            );
            Row::Refused(name, unsupported)
        }
    }
}

#[derive(Default)]
struct Census {
    accepted: Vec<String>,
    refused: Vec<(String, Vec<u32>)>,
    no_histogram: Vec<String>,
    vocab_only: usize,
    unreadable_file: usize,
}

impl Census {
    fn of(files: &[PathBuf]) -> Self {
        let mut c = Census::default();
        for path in files {
            match row_for(path) {
                Row::Unreadable => c.unreadable_file += 1,
                Row::NoHistogram(n) => c.no_histogram.push(n),
                Row::VocabOnly => c.vocab_only += 1,
                Row::Accepted(n) => c.accepted.push(n),
                Row::Refused(n, u) => c.refused.push((n, u)),
            }
        }
        c
    }

    fn total(&self) -> usize {
        self.accepted.len()
            + self.refused.len()
            + self.no_histogram.len()
            + self.vocab_only
            + self.unreadable_file
    }

    fn report(&self) {
        println!(
            "
  accepted by the live reader: {}",
            self.accepted.len()
        );
        println!(
            "  no histogram:       {}  {:?}",
            self.no_histogram.len(),
            self.no_histogram
        );
        println!("  refused:            {}", self.refused.len());
        for (n, u) in &self.refused {
            println!("      {n}  unsupported dtypes {u:?}");
        }
    }
}

/// Measured 2026-09-09 (this file's own header): 3 of this corpus' weighted
/// files refused with an unrepresentable tensor dtype. **RE-MEASURED
/// 2026-10-22, RESOLVED: 0 of 11 weighted files refuse.** Two changes landed
/// the same day: PR #195 removed candlelight entirely (the reader that used
/// to refuse the whole file outright is gone); PR #196 moved `fuel` onto
/// crates.io 0.20.61, which carries GAP-211's widened `GgmlDType` table
/// (`FUEL_ACCEPTS` above) AND its lenient `Content::read` (an unsupported
/// tensor is now skipped, never aborts the whole file -- see
/// `tests/chat_template_render.rs::a_gguf_with_an_unsupported_tensor_dtype_
/// still_resolves_its_metadata`, fixed in the same PR). Every one of this
/// corpus' weighted files, including the three that used to be in
/// `IQ3_XS`/`IQ4_XS`/`Q2_K` filenames and carry 180-210 IQ-coded tensors
/// each, now loads with `unsupported=[]`.
///
/// This test used to assert the OPPOSITE: that `the_refused_number_...` claim
/// (candle's/Fuel's dtype-refusal message names a dtype code, not a tensor
/// index) needed at least one genuinely refused file to exercise at all. It
/// no longer can -- re-running it against this same corpus now trips its own
/// "both arms or nothing" guard, correctly, because the refused arm is empty.
/// Rather than weaken that guard (it would still be right to fire if some
/// *other* corpus ever had a refused file again), this test is rewritten to
/// assert the NEW fact directly: it is a regression guard against the dtype
/// gap reopening, not a test of the old misreading-prevention claim, which
/// has no subject matter left in this corpus.
#[test]
#[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
fn the_iq_dtype_gap_stays_closed() {
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

    let c = Census::of(&files);
    c.report();

    // The one claim this test now exists to hold: every weighted file this
    // corpus has loads. A non-empty `accepted` is the positive control --
    // without it, an empty `refused` would be indistinguishable from "every
    // file in the corpus failed to even reach a bucket."
    assert!(
        !c.accepted.is_empty(),
        "no file was accepted, so this corpus proves nothing about the dtype gap"
    );
    assert!(
        c.refused.is_empty(),
        "the IQ dtype gap reopened: {:?} refused with an unsupported dtype -- \
         re-check FUEL_ACCEPTS against the Fuel version in use, this may be a \
         real regression",
        c.refused
    );

    // Every file lands in exactly one bucket. A conservation law is not a
    // coverage assertion -- see issue #80 -- but it does catch a file being
    // silently dropped by some future `continue`, which is how a population
    // shrinks without anything going red.
    assert_eq!(
        c.total(),
        files.len(),
        "some corpus file reached none of the buckets, so the census lost it"
    );
}
