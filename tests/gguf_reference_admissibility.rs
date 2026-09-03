//! A reference `tokenizer.json` is only admissible if its vocabulary matches the
//! GGUF's. That rule is stated in `gguf_bpe_tokenizer_fidelity.rs` and was, until
//! now, enforced by nothing.
//!
//! # The gap this closes
//!
//! `gguf_bpe_tokenizer_fidelity.rs` says, in prose:
//!
//! > **A reference is only admissible once its vocab matches the GGUF's.**
//!
//! Every entry in `VERIFIED_PRE` rests on that check having been done. It was
//! done — by hand, once, per entry, and recorded in a commit message. Nothing
//! re-runs it. So the rule that gates the whole `pre` allowlist had the shape
//! this repo keeps finding: **a stated standard with no instrument**, where
//! "nobody has re-checked it" and "it still holds" produce the same silence.
//!
//! It matters more than a one-off check normally would, because **the references
//! are remote files that can change under us.** A HuggingFace repo can be
//! updated, re-uploaded, or replaced by a mirror with a different tokenizer. The
//! evidence for an allowlist entry lives on somebody else's server.
//!
//! # What "admissible" means here, precisely
//!
//! 1. **Prefix identity.** For every id the two have in common, the GGUF's token
//!    and the reference's token at that id are byte-identical. A single
//!    disagreement means they are different vocabularies and no score computed
//!    against that reference means anything.
//! 2. **Extras only at the tail.** A GGUF may carry MORE tokens than the
//!    reference — llama.cpp appends the checkpoint's special tokens. Extras on
//!    the REFERENCE side fail: it would be describing a vocabulary this file does
//!    not carry.
//! 3. **Merge identity.** The merge list drives BPE; two vocabularies with the
//!    same tokens and different merges tokenise differently. One side declaring
//!    merges while the other does not is a disagreement, and NEITHER side
//!    declaring them fails too — this gates byte-level BPE references, and a pair
//!    with no merges cannot support that claim.
//!
//! ⚠️ **This list previously opened "three properties, checked in order, each with
//! its own failure message", and two of the three had no failure path at all.**
//! Property 2 only printed; property 3 printed `not compared` and passed whenever
//! either side's merge list was empty. Both were found by enumerating the
//! function after a complexity finding pointed at its length — not by re-reading
//! the prose, which was confidently wrong about its own code.
//!
//! Interleaved extras, the hazard property 2 names, are in fact caught by
//! property 1: an extra inside the shared range shifts ids and breaks prefix
//! identity. That is why the omission never produced a wrong verdict. **A
//! property that happens to be covered by its neighbour is not a checked
//! property**, and the doc claiming otherwise is what made it invisible.
//!
//! # This reports rather than guesses about the extras
//!
//! It does not assert that the tail extras "look special", because that would be
//! a judgement encoded as a pattern match. It prints them, with their count and a
//! sample, so a human decides whether an entry is admissible — and the printed
//! shape becomes the evidence recorded at the allowlist entry.
//!
//! Run:
//! ```text
//! LIGHTBULB_BPE_PAIRS="<a>.gguf|<a>.json;<b>.gguf|<b>.json" \
//!   cargo test --test gguf_reference_admissibility -- --ignored --nocapture
//! ```

use lightbulb::gguf::{Content, Value};
use std::collections::BTreeMap;

fn gguf_string_array(content: &Content, key: &str) -> Option<Vec<String>> {
    let Value::Array(values) = content.metadata().get(key)? else {
        return None;
    };
    let mut out = Vec::with_capacity(values.len());
    for v in values {
        out.push(v.to_string().ok()?.clone());
    }
    Some(out)
}

/// The reference's id -> token table, rebuilt from its `model.vocab` map.
fn reference_tokens(doc: &serde_json::Value) -> Vec<String> {
    let mut by_id: BTreeMap<u64, String> = BTreeMap::new();
    if let Some(map) = doc["model"]["vocab"].as_object() {
        for (tok, id) in map {
            if let Some(i) = id.as_u64() {
                by_id.insert(i, tok.clone());
            }
        }
    }
    by_id.into_values().collect()
}

/// Merges, accepting both encodings tokenizers has used: `"a b"` and `["a","b"]`.
fn reference_merges(doc: &serde_json::Value) -> Vec<String> {
    let Some(list) = doc["model"]["merges"].as_array() else {
        return Vec::new();
    };
    list.iter()
        .filter_map(|m| {
            if let Some(s) = m.as_str() {
                Some(s.to_string())
            } else {
                let pair = m.as_array()?;
                Some(format!(
                    "{} {}",
                    pair.first()?.as_str()?,
                    pair.get(1)?.as_str()?
                ))
            }
        })
        .collect()
}

struct Pair {
    gguf: String,
    reference: String,
}

fn pairs() -> Vec<Pair> {
    let Ok(spec) = std::env::var("LIGHTBULB_BPE_PAIRS") else {
        return Vec::new();
    };
    spec.split(';')
        .filter(|s| !s.trim().is_empty())
        .filter_map(|entry| {
            let (g, r) = entry.split_once('|')?;
            Some(Pair {
                gguf: g.trim().to_string(),
                reference: r.trim().to_string(),
            })
        })
        .collect()
}

#[test]
#[ignore = "needs local GGUFs and reference tokenizers; set LIGHTBULB_BPE_PAIRS"]
fn every_reference_is_admissible_for_its_gguf() {
    let pairs = pairs();
    assert!(
        !pairs.is_empty(),
        "LIGHTBULB_BPE_PAIRS is unset or empty, so this checked NOTHING. An empty pair \
         list satisfies every assertion below, which is exactly the shape this file \
         exists to close."
    );

    let mut failures: Vec<String> = Vec::new();

    for p in &pairs {
        let content = Content::read(&p.gguf).unwrap_or_else(|e| panic!("reading {}: {e}", p.gguf));
        let gguf_tokens = gguf_string_array(&content, "tokenizer.ggml.tokens")
            .unwrap_or_else(|| panic!("{} has no readable tokenizer.ggml.tokens", p.gguf));
        let gguf_merges = gguf_string_array(&content, "tokenizer.ggml.merges").unwrap_or_default();

        let text = std::fs::read_to_string(&p.reference)
            .unwrap_or_else(|e| panic!("reading {}: {e}", p.reference));
        let doc: serde_json::Value =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("parsing {}: {e}", p.reference));
        let ref_tokens = reference_tokens(&doc);
        let ref_merges = reference_merges(&doc);

        let name = std::path::Path::new(&p.gguf)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.gguf.clone());

        println!("\n=== {name} ===");
        println!(
            "  gguf: {:>7} tokens, {:>7} merges     reference: {:>7} tokens, {:>7} merges",
            gguf_tokens.len(),
            gguf_merges.len(),
            ref_tokens.len(),
            ref_merges.len()
        );

        // 1. PREFIX IDENTITY over the shared range.
        let shared = gguf_tokens.len().min(ref_tokens.len());
        let mismatches: Vec<usize> = (0..shared)
            .filter(|&i| gguf_tokens[i] != ref_tokens[i])
            .collect();
        if mismatches.is_empty() {
            println!("  prefix identity : OK over all {shared} shared ids");
        } else {
            println!(
                "  prefix identity : {} MISMATCHES in {shared} shared ids, first at id {}",
                mismatches.len(),
                mismatches[0]
            );
            for &i in mismatches.iter().take(5) {
                println!(
                    "      id {i}: gguf {:?}  reference {:?}",
                    gguf_tokens[i], ref_tokens[i]
                );
            }
            failures.push(format!(
                "{name}: {} token mismatches against {} -- these are DIFFERENT vocabularies, \
                 so any fidelity score against this reference is meaningless",
                mismatches.len(),
                p.reference
            ));
        }

        // 2. EXTRAS ONLY AT THE TAIL. Reported, not judged.
        if gguf_tokens.len() > ref_tokens.len() {
            let extras = &gguf_tokens[ref_tokens.len()..];
            println!(
                "  tail extras     : {} gguf tokens above the reference's {} ids",
                extras.len(),
                ref_tokens.len()
            );
            for t in extras.iter().take(6) {
                println!("      {t:?}");
            }
            if extras.len() > 6 {
                println!("      ... and {} more", extras.len() - 6);
            }
        } else if ref_tokens.len() > gguf_tokens.len() {
            // A REFERENCE CLAIMING TOKENS THE GGUF LACKS IS NOT THE GGUF'S
            // TOKENIZER. This branch used to print exactly that gap -- "which
            // the tail-extras rule does not cover" -- and then pass. Naming a
            // hole is not covering it.
            println!(
                "  tail extras     : INVERTED -- the reference has {} tokens the gguf does not",
                ref_tokens.len() - gguf_tokens.len()
            );
            failures.push(format!(
                "{name}: the reference declares {} tokens the GGUF does not have. Extras at \
                 the GGUF's tail are llama.cpp appending special tokens and are expected; \
                 extras on the REFERENCE side mean it describes a vocabulary this file does \
                 not carry",
                ref_tokens.len() - gguf_tokens.len()
            ));
        } else {
            println!("  tail extras     : none, sizes are equal");
        }

        // 3. MERGE IDENTITY.
        let shared_m = gguf_merges.len().min(ref_merges.len());
        let merge_mismatches: Vec<usize> = (0..shared_m)
            .filter(|&i| gguf_merges[i] != ref_merges[i])
            .collect();
        if gguf_merges.is_empty() != ref_merges.is_empty() {
            // ONE SIDE DECLARES MERGES AND THE OTHER DOES NOT. That is a
            // disagreement, not an absence of one, and it used to print
            // "not compared" and pass.
            println!(
                "  merges          : ONE SIDE ONLY -- gguf {} / reference {}",
                gguf_merges.len(),
                ref_merges.len()
            );
            failures.push(format!(
                "{name}: one side declares merges and the other does not (gguf {} / reference \
                 {}), which is a disagreement about the tokenizer's structure",
                gguf_merges.len(),
                ref_merges.len()
            ));
        } else if gguf_merges.is_empty() {
            // BOTH EMPTY. This file gates BYTE-LEVEL BPE references, and BPE is
            // driven by merges; a pair with none on either side cannot support
            // the claim this instrument exists to make. Previously "not
            // compared", which passed -- a guard conditioned on its subject
            // being non-trivial, switching itself off when the subject
            // degenerated.
            println!("  merges          : NEITHER side declares any");
            failures.push(format!(
                "{name}: neither side declares merges, so this pair cannot support a \
                 byte-level BPE claim -- admissibility here is about BPE references, and \
                 a SentencePiece pair belongs in gguf_tokenizer_fidelity instead"
            ));
        } else if merge_mismatches.is_empty() && gguf_merges.len() == ref_merges.len() {
            println!("  merges          : OK, {} identical", gguf_merges.len());
        } else if merge_mismatches.is_empty() {
            println!(
                "  merges          : first {shared_m} identical, but lengths differ \
                 ({} vs {})",
                gguf_merges.len(),
                ref_merges.len()
            );
            failures.push(format!(
                "{name}: merge lists differ in length ({} vs {}) -- BPE is driven by merges, \
                 so equal tokens with unequal merges still tokenise differently",
                gguf_merges.len(),
                ref_merges.len()
            ));
        } else {
            println!(
                "  merges          : {} MISMATCHES, first at {}",
                merge_mismatches.len(),
                merge_mismatches[0]
            );
            failures.push(format!(
                "{name}: {} merge mismatches against {}",
                merge_mismatches.len(),
                p.reference
            ));
        }
    }

    println!("\n  checked {} pair(s)", pairs.len());
    assert!(
        failures.is_empty(),
        "these references are NOT admissible for their GGUFs, so any fidelity score \
         computed against them carries no information:\n{}",
        failures.join("\n")
    );
}
