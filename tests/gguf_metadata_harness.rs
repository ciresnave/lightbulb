//! Lightbulb's half of a cross-implementation disagreement harness for GGUF
//! metadata, plus the comparator that reads two halves.
//!
//! # Why an artifact rather than a dependency
//!
//! The obvious construction is to depend on the other implementation and call
//! it. That was rejected, and the reason is the point of the whole exercise:
//!
//! **A disagreement harness is only informative if the two implementations are
//! independent.** A build dependency does not itself merge them — but it makes
//! the shortcut (calling the other crate for something this one should compute)
//! the path of least resistance for everyone who touches the harness later. A
//! self-check that looks like a comparison is the worst available outcome,
//! because it produces AGREEMENT, and agreement from a shared implementation is
//! indistinguishable from agreement between independent ones.
//!
//! That is not hypothetical here. Two lanes independently reported 18 distinct
//! vocabularies in this corpus and the answer is 19 — because each had added
//! back only what its OWN reader missed. **Cross-lane agreement is worth nothing
//! when the methods are the same**, so the methods are kept structurally apart:
//! each side writes a file, and the comparator reads two files without knowing
//! or caring who produced either.
//!
//! # ⚠️ The comparator is validated against TWO OF THIS SIDE'S OWN DUMPS
//!
//! A comparator exercised only against a real second-implementation dump can
//! never be shown to detect a disagreement it has not already seen — and the
//! expected result is zero disagreements, which is exactly what a broken
//! comparator also reports. So `the_comparator_detects_every_perturbed_field`
//! builds one dump, perturbs each field in turn, and requires the comparator to
//! name that field. It runs in CI with no corpus and no second implementation.
//!
//! # The artifact
//!
//! Deterministic: files sorted by name, named templates sorted by name, no
//! timestamps, no absolute paths. Two runs over one corpus produce byte-identical
//! output, which is what lets a disagreement be attributed to the readers rather
//! than to the run.
//!
//! Template bodies are recorded as `{len, sha256}` rather than text. A digest
//! detects disagreement precisely and keeps the artifact small; `len` is carried
//! alongside so a disagreement is immediately characterisable — same length with
//! a different digest is a content difference, different lengths is a size one —
//! without fetching either body.
//!
//! ⚠️ **An unreadable file is recorded, not skipped.** This reader cannot open
//! five files in the local corpus whose metadata another reader reads without
//! difficulty, because `Content::read` parses tensor infos eagerly and dies on an
//! unknown quantization dtype before reaching a KV block that sits ahead of any
//! tensor. Dropping those rows would hide the single largest difference between
//! the two implementations. `status: "unreadable"` with its reason is a claim
//! about THIS reader, and the comparator treats a status difference as a
//! first-class disagreement.
//!
//! Run:
//! ```text
//! LIGHTBULB_GGUF_CORPUS=<dir> LIGHTBULB_METADATA_DUMP=<out.json> \
//!   cargo test --test gguf_metadata_harness -- --ignored emit_metadata_dump --nocapture
//!
//! LIGHTBULB_DUMP_A=<a.json> LIGHTBULB_DUMP_B=<b.json> \
//!   cargo test --test gguf_metadata_harness -- --ignored compare_metadata_dumps --nocapture
//! ```

use serde_json::{Map, Value as J, json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Bumped when the shape changes. The comparator refuses to compare across
/// schemas rather than silently aligning fields that no longer mean the same
/// thing.
const SCHEMA: &str = "gguf-metadata-dump/v1";

fn digest_of(s: &str) -> J {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    json!({ "len": s.len(), "sha256": format!("{:x}", h.finalize()) })
}

fn corpus_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
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

/// One file's row, read through lightbulb's own public GGUF surface.
fn row_for(path: &Path) -> J {
    use lightbulb::gguf::{Content, Value};
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    let content = match Content::read(path) {
        Ok(c) => c,
        // A claim about THIS reader, recorded rather than dropped.
        Err(e) => {
            return json!({ "file": name, "status": "unreadable", "reason": format!("{e}") });
        }
    };
    let md = content.metadata();
    let s = |k: &str| md.get(k).and_then(|v| v.to_string().ok().cloned());
    let u = |k: &str| md.get(k).and_then(|v| v.to_u32().ok()).map(u64::from);
    let b = |k: &str| md.get(k).and_then(|v| v.to_bool().ok());

    // Both sources of template names are read, because each is lossy in a
    // different direction: the declared array omits the unnamed default, and the
    // key set says nothing about what was promised.
    let mut named = Map::new();
    for (k, v) in md.iter() {
        if let Some(n) = k.strip_prefix("tokenizer.chat_template.")
            && let Ok(body) = v.to_string()
        {
            named.insert(n.to_string(), digest_of(body));
        }
    }
    let declared: Vec<String> = match md.get("tokenizer.chat_templates") {
        Some(Value::Array(names)) => {
            let mut v: Vec<String> = names
                .iter()
                .filter_map(|n| n.to_string().ok().cloned())
                .collect();
            v.sort();
            v
        }
        _ => Vec::new(),
    };

    // The token TEXT, not only the id: the id alone cannot answer whether a
    // rendered string already begins with the BOS token, which is the question
    // this metadata exists to serve.
    let text_of = |id: Option<u64>| -> J {
        let Some(id) = id else { return J::Null };
        let text = match md.get("tokenizer.ggml.tokens") {
            Some(Value::Array(toks)) => toks
                .get(id as usize)
                .and_then(|t| t.to_string().ok().cloned()),
            _ => None,
        };
        json!({ "id": id, "text": text })
    };

    json!({
        "file": name,
        "status": "read",
        "architecture": s("general.architecture"),
        "tokenizer_model": s("tokenizer.ggml.model"),
        "tokenizer_pre": s("tokenizer.ggml.pre"),
        "template_default": s("tokenizer.chat_template").as_deref().map(digest_of),
        "template_named": J::Object(named),
        "template_names_declared": declared,
        "bos": text_of(u("tokenizer.ggml.bos_token_id")),
        "eos": text_of(u("tokenizer.ggml.eos_token_id")),
        "add_bos_declared": b("tokenizer.ggml.add_bos_token"),
        "add_eos_declared": b("tokenizer.ggml.add_eos_token"),
    })
}

fn build_dump(root: &Path) -> J {
    let files: Vec<J> = corpus_files(root).iter().map(|p| row_for(p)).collect();
    json!({ "schema": SCHEMA, "producer": "lightbulb", "files": files })
}

/// Every way two dumps can disagree about one file, as a flat list of strings.
///
/// Returns disagreements rather than a bool: a count says the readers differ,
/// and a caller acting on it needs to know WHERE.
fn compare(a: &J, b: &J) -> Vec<String> {
    let mut out = Vec::new();

    for (side, d) in [("A", a), ("B", b)] {
        if d["schema"].as_str() != Some(SCHEMA) {
            out.push(format!(
                "dump {side} declares schema {:?}, this comparator understands {SCHEMA:?}; \
                 comparing across schemas would align fields that no longer mean the same thing",
                d["schema"]
            ));
        }
    }
    if !out.is_empty() {
        return out;
    }

    let index = |d: &J| -> Map<String, J> {
        let mut m = Map::new();
        if let Some(files) = d["files"].as_array() {
            for f in files {
                if let Some(n) = f["file"].as_str() {
                    m.insert(n.to_string(), f.clone());
                }
            }
        }
        m
    };
    let (ia, ib) = (index(a), index(b));

    // A file present on one side only is a disagreement about the CORPUS, which
    // is worth reporting separately from a disagreement about a file's contents.
    let mut names: Vec<&String> = ia.keys().chain(ib.keys()).collect();
    names.sort();
    names.dedup();

    for name in names {
        match (ia.get(name), ib.get(name)) {
            (Some(_), None) => out.push(format!("{name}: present in A, absent from B")),
            (None, Some(_)) => out.push(format!("{name}: absent from A, present in B")),
            (Some(ra), Some(rb)) => {
                let mut keys: Vec<&String> = ra
                    .as_object()
                    .into_iter()
                    .chain(rb.as_object())
                    .flat_map(|o| o.keys())
                    .collect();
                keys.sort();
                keys.dedup();
                for k in keys {
                    if k == "file" {
                        continue;
                    }
                    let (va, vb) = (&ra[k.as_str()], &rb[k.as_str()]);
                    if va != vb {
                        out.push(format!("{name}: {k}: A={va} B={vb}"));
                    }
                }
            }
            (None, None) => unreachable!("name came from one of the two indexes"),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The comparator's own positive control. No corpus, no second implementation.
// ---------------------------------------------------------------------------

fn sample_dump() -> J {
    json!({
        "schema": SCHEMA,
        "producer": "test",
        "files": [
            {
                "file": "a.gguf",
                "status": "read",
                "architecture": "llama",
                "tokenizer_model": "gpt2",
                "tokenizer_pre": "smollm",
                "template_default": { "len": 12, "sha256": "aa" },
                "template_named": { "rag": { "len": 3, "sha256": "bb" } },
                "template_names_declared": ["rag"],
                "bos": { "id": 1, "text": "<s>" },
                "eos": { "id": 2, "text": "</s>" },
                "add_bos_declared": true,
                "add_eos_declared": null
            },
            { "file": "b.gguf", "status": "unreadable", "reason": "unknown dtype for tensor 20" }
        ]
    })
}

/// ⚠️ A comparator that reports "0 disagreements" is indistinguishable from one
/// that cannot report any.
///
/// The expected result over two honest dumps is zero, so the null result is
/// exactly what a broken comparator produces. Every field is therefore perturbed
/// in turn and the comparator must NAME that field — not merely return non-empty.
#[test]
fn the_comparator_detects_every_perturbed_field() {
    let base = sample_dump();
    assert!(
        compare(&base, &base).is_empty(),
        "a dump must not disagree with itself"
    );

    // Every field of the readable row, and the status field of the other.
    let fields = [
        "status",
        "architecture",
        "tokenizer_model",
        "tokenizer_pre",
        "template_default",
        "template_named",
        "template_names_declared",
        "bos",
        "eos",
        "add_bos_declared",
        "add_eos_declared",
    ];
    for f in fields {
        let mut other = base.clone();
        other["files"][0][f] = json!("PERTURBED");
        let diffs = compare(&base, &other);
        assert!(
            diffs.iter().any(|d| d.contains(f)),
            "perturbing {f} produced no disagreement naming it: {diffs:?}"
        );
    }

    // A row present on one side only, which is a disagreement about the corpus
    // rather than about a file, and is easy to drop when indexing by name.
    let mut fewer = base.clone();
    fewer["files"].as_array_mut().unwrap().pop();
    let diffs = compare(&base, &fewer);
    assert!(
        diffs
            .iter()
            .any(|d| d.contains("b.gguf") && d.contains("absent")),
        "a file present on one side only must be reported: {diffs:?}"
    );

    // A schema mismatch must stop the comparison rather than align fields whose
    // meaning has changed.
    let mut wrong_schema = base.clone();
    wrong_schema["schema"] = json!("gguf-metadata-dump/v99");
    let diffs = compare(&base, &wrong_schema);
    assert!(
        diffs.iter().any(|d| d.contains("schema")),
        "a schema mismatch must be reported and must stop the comparison: {diffs:?}"
    );
    assert_eq!(
        diffs.len(),
        1,
        "a schema mismatch must SHORT-CIRCUIT; reporting per-field diffs across \
         incompatible schemas invites acting on them: {diffs:?}"
    );
}

#[test]
#[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS and LIGHTBULB_METADATA_DUMP"]
fn emit_metadata_dump() {
    let root = std::env::var("LIGHTBULB_GGUF_CORPUS")
        .expect("set LIGHTBULB_GGUF_CORPUS to the corpus directory");
    let out = std::env::var("LIGHTBULB_METADATA_DUMP")
        .expect("set LIGHTBULB_METADATA_DUMP to the output path");
    let dump = build_dump(Path::new(&root));

    let n = dump["files"].as_array().map(Vec::len).unwrap_or(0);
    assert!(
        n > 0,
        "no .gguf files under {root:?}; an empty dump compares clean against anything"
    );
    let unreadable = dump["files"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["status"] == "unreadable")
        .count();

    std::fs::write(&out, serde_json::to_string_pretty(&dump).unwrap()).expect("writing the dump");
    eprintln!("  wrote {n} rows to {out} ({unreadable} unreadable by this reader)");
}

#[test]
#[ignore = "compares two dumps; set LIGHTBULB_DUMP_A and LIGHTBULB_DUMP_B"]
fn compare_metadata_dumps() {
    let pa = std::env::var("LIGHTBULB_DUMP_A").expect("set LIGHTBULB_DUMP_A");
    let pb = std::env::var("LIGHTBULB_DUMP_B").expect("set LIGHTBULB_DUMP_B");
    let a: J =
        serde_json::from_str(&std::fs::read_to_string(&pa).expect("reading A")).expect("parsing A");
    let b: J =
        serde_json::from_str(&std::fs::read_to_string(&pb).expect("reading B")).expect("parsing B");

    let diffs = compare(&a, &b);
    eprintln!(
        "  {} vs {}: {} disagreement(s)",
        a["producer"],
        b["producer"],
        diffs.len()
    );
    for d in &diffs {
        eprintln!("    {d}");
    }
    assert!(
        diffs.is_empty(),
        "{} disagreement(s) between {} and {}",
        diffs.len(),
        a["producer"],
        b["producer"]
    );
}

/// ⚠️ A COMPARATOR IS WORTHLESS IF EITHER SIDE IS NONDETERMINISTIC, and
/// "the two dumps looked the same" is not that check.
///
/// `build_dump` iterates `Content::metadata()`, which is a `HashMap` — an
/// unspecified and deliberately randomised iteration order. Nothing in the
/// dump's shape guarantees that order cannot reach the output; the named
/// templates land in a `serde_json::Map` (a `BTreeMap` without
/// `preserve_order`, so sorted) and the declared names are sorted explicitly.
/// Both of those are inferences from types. This measures it instead.
///
/// Raised by MLMF, who proved the same property on their side before diffing
/// against ours. A disagreement between two readers is only evidence about the
/// READERS if each is stable against itself first.
#[test]
#[ignore = "needs a local GGUF corpus; set LIGHTBULB_GGUF_CORPUS"]
fn two_runs_over_one_corpus_are_byte_identical() {
    let root = std::env::var("LIGHTBULB_GGUF_CORPUS")
        .expect("set LIGHTBULB_GGUF_CORPUS to the corpus directory");
    let root = Path::new(&root);

    let a = serde_json::to_string(&build_dump(root)).expect("serialising run A");
    let b = serde_json::to_string(&build_dump(root)).expect("serialising run B");

    let rows = build_dump(root)["files"]
        .as_array()
        .map(Vec::len)
        .unwrap_or(0);
    assert!(
        rows > 0,
        "no .gguf files under {root:?}. Two empty dumps are byte-identical, which          would pass this test while measuring nothing."
    );

    assert_eq!(
        a, b,
        "two runs over the same corpus produced different bytes, so any diff against          another reader is uninterpretable: a disagreement could be ours alone."
    );

    // POSITIVE CONTROL: the comparison must be able to see a difference. Without
    // this, `assert_eq!` on two identical constants would pass just as happily.
    let mut perturbed: J = serde_json::from_str(&b).expect("reparsing run B");
    perturbed["producer"] = json!("not-lightbulb");
    assert_ne!(
        a,
        serde_json::to_string(&perturbed).expect("serialising the perturbed dump"),
        "the byte comparison did not notice a changed field, so its agreement above          carries no information"
    );

    eprintln!(
        "  {rows} rows, two runs, byte-identical ({} bytes)",
        a.len()
    );
}
