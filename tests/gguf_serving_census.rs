//! How many corpus checkpoints could be SERVED, and why each one cannot.
//!
//! # Why this exists
//!
//! `tests/gguf_corpus_sweep.rs` reports **23 of 30 rebuild a tokenizer**, and
//! that figure has been quoted as though it said something about end-to-end
//! serving. It does not. ⚠️ **The two range over DIFFERENT POPULATIONS**, and
//! nothing in the repo said so:
//!
//! - a tokenizer needs only `tokenizer.ggml.*`, which lives in the KV header
//! - serving needs WEIGHTS, and most of this corpus has none
//!
//! `tests/gguf_serving_e2e.rs` serves exactly **one** checkpoint, named by
//! `LIGHTBULB_GGUF`. So "usable end-to-end" had a population of one chosen by an
//! environment variable — a claim with no denominator. This file supplies the
//! denominator.
//!
//! # ⚠️ WHAT THIS MEASURES, AND WHAT IT DOES NOT
//!
//! It measures **whether the live loader accepts the file**, via
//! `ParallelModelManager::load_gguf` — the reader that is actually reachable
//! (see `tests/gguf_architecture_refusal.rs` for why that distinction matters).
//!
//! **Loading is NECESSARY for serving and is not SUFFICIENT.** A model that
//! loads can still generate nonsense, which is precisely what this corpus did
//! for ten days while every gate passed. **Do not quote this file's number as a
//! serving figure** — that is the same substitution this file exists to stop,
//! one level down. Coherence is `tests/gguf_serving_e2e.rs`, and it still
//! covers one checkpoint.
//!
//! ```text
//! LIGHTBULB_GGUF_CORPUS=<dir> cargo test --test gguf_serving_census -- --ignored --nocapture
//! ```

use lightbulb::gguf::{Content, Value};
use std::path::PathBuf;

fn corpus() -> Option<Vec<PathBuf>> {
    let root = PathBuf::from(std::env::var_os("LIGHTBULB_GGUF_CORPUS")?);
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
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
    Some(out)
}

/// Why a file is or is not a serving candidate. The order of these variants is
/// the order the checks run, and each one is a DIFFERENT fact about the file.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    /// `Content::read` refused. Nothing further can be said about it.
    Unreadable,
    /// Readable, and carries ZERO tensors. ⚠️ This is the big one: a
    /// `ggml-vocab-*.gguf` is a vocabulary fixture and was never a model. It
    /// cannot be served BY CONSTRUCTION, not by defect.
    VocabularyOnly,
    /// Has weights, but declares an architecture the loader does not build.
    ForeignArchitecture(String),
    /// Has weights and a supported architecture, and the live loader accepted
    /// it. ⚠️ A CANDIDATE FOR SERVING, NOT A DEMONSTRATION OF IT.
    Loads,
    /// Has weights and a supported architecture, and the loader refused
    /// because a TENSOR'S DTYPE is not representable. ⚠️ This is a DOCUMENTED
    /// limitation, not a defect: the IQ codebook family is rejected by candle
    /// and by fuel alike, so it is not a missing table entry.
    ///
    /// ⚠️ It is kept SEPARATE from `LoadFailed` rather than folded into it,
    /// because an exemption must name the property it assumes — here, "the
    /// refusal names an unrepresentable dtype" — and not the category that
    /// usually has it ("it is a SmolLM2 file"). A future refusal for a
    /// different reason must NOT inherit this exemption.
    UnsupportedQuantization(String),
    /// Has weights and a supported architecture, and the loader refused for
    /// some OTHER reason. This is the one that is a defect.
    LoadFailed(String),
}

fn tensor_count(c: &Content) -> usize {
    // Either reader answers this; candle's is the one the loader uses, and its
    // absence is itself informative, so fall back rather than treating a
    // candle refusal as zero tensors.
    match c.tensor_infos() {
        Ok(t) => t.len(),
        Err(_) => c.lightning_tensor_infos().map(|t| t.len()).unwrap_or(0),
    }
}

fn architecture(c: &Content) -> String {
    match c.metadata().get("general.architecture") {
        Some(Value::String(s)) => s.clone(),
        _ => "<absent>".to_string(),
    }
}

fn classify(path: &PathBuf) -> Verdict {
    let Ok(c) = Content::read(path) else {
        return Verdict::Unreadable;
    };
    if tensor_count(&c) == 0 {
        return Verdict::VocabularyOnly;
    }
    let arch = architecture(&c);
    if arch != "llama" {
        return Verdict::ForeignArchitecture(arch);
    }
    drop(c);
    // One at a time, dropped immediately: the corpus holds a 637 MB checkpoint
    // and nine quantizations, and holding them all would measure the box rather
    // than the loader.
    match lightbulb::model::parallel_model_manager::ParallelModelManager::load_gguf(
        path,
        1,
        512,
        Some(candlelight::core::Device::Cpu),
        None,
    ) {
        Ok(m) => {
            drop(m);
            Verdict::Loads
        }
        Err(e) => {
            let msg = e.to_string();
            // Keyed on the PROPERTY, read out of the refusal itself.
            if msg.contains("unknown dtype for tensor") {
                Verdict::UnsupportedQuantization(msg)
            } else {
                Verdict::LoadFailed(msg)
            }
        }
    }
}

#[test]
#[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
fn the_servable_population_is_measured_rather_than_assumed() {
    let Some(files) = corpus() else {
        lightbulb::test_notice::skip_unless_required(
            "LIGHTBULB_REQUIRE_CORPUS",
            "set LIGHTBULB_GGUF_CORPUS to a directory containing .gguf files",
        );
        return;
    };
    assert!(
        !files.is_empty(),
        "an empty corpus satisfies every assertion below, so it fails here instead"
    );

    let mut unreadable = 0usize;
    let mut vocab_only = 0usize;
    let mut foreign = 0usize;
    let mut loads = 0usize;
    let mut unsupported = Vec::new();
    let mut failed = Vec::new();

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let v = classify(path);
        println!("  {name:<38} {v:?}");
        match v {
            Verdict::Unreadable => unreadable += 1,
            Verdict::VocabularyOnly => vocab_only += 1,
            Verdict::ForeignArchitecture(_) => foreign += 1,
            Verdict::Loads => loads += 1,
            Verdict::UnsupportedQuantization(e) => unsupported.push((name, e)),
            Verdict::LoadFailed(e) => failed.push((name, e)),
        }
    }

    let with_weights = files.len() - vocab_only - unreadable;
    println!("\n  {} files", files.len());
    println!("  {vocab_only:>3}  vocabulary-only -- CANNOT be served by construction");
    println!("  {unreadable:>3}  unreadable");
    println!("  {with_weights:>3}  carry weights   <- THE SERVING DENOMINATOR");
    println!("  {foreign:>3}    of those, a foreign architecture");
    println!("  {loads:>3}    of those, accepted by the live loader");
    println!(
        "  {:>3}    of those, an unrepresentable tensor dtype (documented)",
        unsupported.len()
    );
    println!(
        "  {:>3}    of those, refused for some OTHER reason",
        failed.len()
    );
    for (n, _) in &unsupported {
        println!("         unsupported quantization: {n}");
    }

    // ⚠️ THE POPULATION MUST NOT BE EMPTY. Every assertion here is satisfied by
    // a corpus of thirty vocabulary files, which is the shape this corpus very
    // nearly has -- so the guard is not hypothetical.
    assert!(
        with_weights > 0,
        "no corpus file carries weights, so this census measured nothing about serving"
    );

    // ⚠️ AND THE LOADED COUNT MUST NOT BE ZERO. `with_weights > 0` is satisfied
    // by a corpus whose every weighted file is refused, which would make the
    // serving claim vacuous while this test stayed green.
    assert!(
        loads > 0,
        "every weighted file was refused, so nothing here supports a serving claim"
    );

    // A refusal for a reason OTHER than an unrepresentable dtype is a defect
    // and must be named rather than absorbed into a ratio.
    assert!(
        failed.is_empty(),
        "llama checkpoints with weights that the live loader refused for an          unexplained reason: {failed:#?}"
    );
}
