//! How many corpus checkpoints Fuel's own loader can SERVE end-to-end --
//! constructing a real model with real decoded weights, not just reading
//! metadata.
//!
//! # Why this exists
//!
//! `tests/gguf_dtype_census.rs` measures METADATA/tensor-directory
//! acceptance: whether `Content::read`/`tensor_infos()` succeed. As of
//! 2026-10-22 (PR #197) that is 0 of 11 weighted corpus files refusing.
//! **Loading is NECESSARY for serving and is not SUFFICIENT** -- a file
//! whose tensor directory is representable can still fail to construct a
//! real model (a missing config field, a decode-kernel bug for a dtype the
//! table accepts but cannot yet actually decode, an unsupported
//! architecture). This file measures THAT, through the same loaders the
//! live Fuel serving path uses: `model_fuel::loader_gguf::
//! load_quantized_llama_gguf` / `model_fuel::loader_gguf_qwen3::
//! load_quantized_qwen3_gguf`.
//!
//! This replaces `tests/gguf_serving_census.rs` (deleted in PR #195 along
//! with `ParallelModelManager::load_gguf`, the candlelight-era loader it
//! measured) and `tests/enhanced_correctness_tests.rs` (same deletion,
//! different angle) -- see `docs/FUEL-GAPS-FROM-CANDLE-REMOVAL.md`'s entries
//! on both. Architecture dispatch is done HERE, by reading `general.
//! architecture` through the same public `Content::metadata()` the dtype
//! census uses, because `crate::gguf::detect_architecture`/
//! `require_llama_architecture` are `pub(crate)` -- this file lives in
//! `tests/`, a separate crate, and cannot see them. Getting the dispatch
//! wrong here means calling the wrong loader and reporting ITS refusal
//! (architecture mismatch) as this file's own refusal, which is exactly the
//! "wrong model, not a missing key" hazard those `pub(crate)` functions'
//! own doc comments name -- so this mirrors their exact check (`Value::
//! String` under `general.architecture`) rather than guessing a looser one.
//!
//! ```text
//! LIGHTBULB_GGUF_CORPUS=<dir> cargo test --test gguf_e2e_weight_load_census -- --ignored --nocapture
//! ```

use lightbulb::gguf::{Content, Value};
use lightbulb::model_fuel::loader_gguf::load_quantized_llama_gguf;
use lightbulb::model_fuel::loader_gguf_qwen3::load_quantized_qwen3_gguf;
use std::path::PathBuf;

/// Same walk as `gguf_dtype_census.rs::corpus()` -- kept as its own copy
/// rather than shared, since the two files are independent measurements and
/// sharing a helper across `tests/*.rs` files needs a `tests/support/`
/// module neither currently has.
fn corpus() -> Option<(Vec<PathBuf>, Vec<String>)> {
    let root = PathBuf::from(std::env::var_os("LIGHTBULB_GGUF_CORPUS")?);
    let mut out = Vec::new();
    let mut unreadable = Vec::new();
    let mut stack = vec![root];
    while let Some(d) = stack.pop() {
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

/// What one corpus file turned out to be, at the WEIGHT-LOADING layer.
enum Row {
    /// Metadata itself could not be read (distinct from a dtype-based
    /// refusal, which no longer happens as of PR #197 -- see
    /// `gguf_dtype_census.rs`).
    Unreadable(String),
    /// Zero tensors: a vocabulary fixture, nothing to load.
    VocabOnly,
    /// `general.architecture` names something neither loader handles.
    /// `None` inside the `Option` means the key itself was absent.
    UnsupportedArchitecture(String, Option<String>),
    /// A real model was constructed: decoded weights, tokenizer, device.
    Loaded(String, &'static str),
    /// The matching loader was called and it refused.
    Failed(String, &'static str, String),
}

fn architecture_of(c: &Content) -> Option<String> {
    match c.metadata().get("general.architecture") {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn row_for(path: &PathBuf) -> Row {
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    let Ok(content) = Content::read(path) else {
        return Row::Unreadable(name);
    };
    let tensor_count = content
        .lightning_tensor_infos()
        .map(|t| t.len())
        .unwrap_or(0);
    if tensor_count == 0 {
        return Row::VocabOnly;
    }
    match architecture_of(&content).as_deref() {
        Some("llama") => match load_quantized_llama_gguf(path) {
            Ok(_loaded) => Row::Loaded(name, "llama"),
            Err(e) => Row::Failed(name, "llama", format!("{e:#}")),
        },
        Some("qwen3") => match load_quantized_qwen3_gguf(path) {
            Ok(_loaded) => Row::Loaded(name, "qwen3"),
            Err(e) => Row::Failed(name, "qwen3", format!("{e:#}")),
        },
        other => Row::UnsupportedArchitecture(name, other.map(str::to_string)),
    }
}

#[derive(Default)]
struct Census {
    loaded: Vec<(String, &'static str)>,
    failed: Vec<(String, &'static str, String)>,
    unsupported_architecture: Vec<(String, Option<String>)>,
    vocab_only: usize,
    unreadable: Vec<String>,
}

impl Census {
    fn of(files: &[PathBuf]) -> Self {
        let mut c = Census::default();
        for path in files {
            match row_for(path) {
                Row::Unreadable(n) => c.unreadable.push(n),
                Row::VocabOnly => c.vocab_only += 1,
                Row::UnsupportedArchitecture(n, a) => c.unsupported_architecture.push((n, a)),
                Row::Loaded(n, a) => c.loaded.push((n, a)),
                Row::Failed(n, a, e) => c.failed.push((n, a, e)),
            }
        }
        c
    }

    fn total(&self) -> usize {
        self.loaded.len()
            + self.failed.len()
            + self.unsupported_architecture.len()
            + self.vocab_only
            + self.unreadable.len()
    }

    fn report(&self) {
        println!("\n  loaded end-to-end: {}", self.loaded.len());
        for (n, a) in &self.loaded {
            println!("      {n}  ({a})");
        }
        println!("  failed to load:    {}", self.failed.len());
        for (n, a, e) in &self.failed {
            println!("      {n}  ({a})  {e}");
        }
        println!(
            "  unsupported architecture: {}  {:?}",
            self.unsupported_architecture.len(),
            self.unsupported_architecture
        );
        println!("  vocab-only (no weights):  {}", self.vocab_only);
        println!(
            "  unreadable:               {}  {:?}",
            self.unreadable.len(),
            self.unreadable
        );
    }
}

/// Measured 2026-10-22, `origin/main` (post PR #197): 9 of 11 weighted
/// corpus files load end-to-end through Fuel's own loader -- INCLUDING all
/// three that used to refuse at the metadata layer for an IQ-coded tensor
/// dtype (`gguf_dtype_census.rs`'s `the_iq_dtype_gap_stays_closed`), which
/// confirms that closure holds at the weight-decode layer too, not just at
/// metadata acceptance.
///
/// The remaining 2 (`stories15M-q4_0.gguf`, `stories15M-q8_0.gguf`) failed
/// for `llama.attention.head_count_kv` not being defaulted. Fuel 0.20.71
/// (`derive_config`) now defaults it for pre-GQA exports -- **re-measured
/// 2026-10-22 against 0.20.71: that specific cause is gone**, but both
/// files still fail, now for a DIFFERENT, previously-masked key:
/// `llama.rope.freq_base`, also "required to derive a llama config, not
/// defaulted". Same tiny karpathy-era "tinystories" exports, same shape of
/// gap (an old GGUF convention that predates a key llama.cpp itself
/// defaults — llama.cpp's documented default is `10000.0`) -- a second
/// Fuel-side `derive_config` default to add, not a lightbulb fix, and not
/// yet routed as its own board row (the head_count_kv fix only just
/// landed; this is reported here so it isn't lost, not sized or routed).
///
/// **Both-arms guard**: a corpus where every file lands in `unsupported_
/// architecture`/`vocab_only`/`unreadable` would make `loaded` empty and
/// this test pass having exercised no loader at all. The exact-match on
/// `KNOWN_FAILURES` means both a further improvement (one starts loading)
/// and a new regression (a different file starts failing, or these two fail
/// for a different reason) are caught, not just silently tolerated -- which
/// is exactly what caught the head_count_kv -> freq_base cause change above:
/// this test went red against 0.20.71 until `KNOWN_FAILURES` below was
/// updated to the new cause, not just left alone because the file count
/// didn't change.
const KNOWN_FAILURES: &[(&str, &str)] = &[
    ("stories15M-q4_0.gguf", "freq_base"),
    ("stories15M-q8_0.gguf", "freq_base"),
];

/// The two assertion passes `KNOWN_FAILURES` needs: every actual failure is
/// accounted for, and every expected failure actually happened. Split out of
/// the test itself to keep that function under Codacy's 50-line limit, not
/// because this half is reused anywhere else.
fn assert_failures_match_known(c: &Census) {
    for (name, _, error) in &c.failed {
        let known = KNOWN_FAILURES
            .iter()
            .any(|(n, needle)| name == n && error.contains(needle));
        assert!(
            known,
            "{name} failed with an UNKNOWN cause (not in KNOWN_FAILURES): {error}. If this is a \
             new, understood gap, add it to KNOWN_FAILURES with its substring; if it is a \
             regression, investigate before doing either."
        );
    }
    for (name, needle) in KNOWN_FAILURES {
        assert!(
            c.failed.iter().any(|(n, _, _)| n == name),
            "{name} was expected to fail with {needle:?} (a known, routed-to-Fuel gap) but it \
             loaded successfully -- that is an improvement, not a regression: remove it from \
             KNOWN_FAILURES and update ROADMAP.md's GGUF corpus section"
        );
    }
    assert!(
        c.failed.len() == KNOWN_FAILURES.len(),
        "{} weighted file(s) failed to load through Fuel's own loader, expected exactly {}: {:?}",
        c.failed.len(),
        KNOWN_FAILURES.len(),
        c.failed
    );
}

#[test]
#[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
fn weighted_corpus_files_load_end_to_end_through_fuel() {
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

    assert!(
        !c.loaded.is_empty(),
        "no file loaded end-to-end, so this corpus proves nothing about Fuel's loader"
    );
    assert_failures_match_known(&c);

    assert_eq!(
        c.total(),
        files.len(),
        "some corpus file reached none of the buckets, so the census lost it"
    );
}
