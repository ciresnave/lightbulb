// SPDX-License-Identifier: MIT OR Apache-2.0
//! Recognising a split (sharded) GGUF checkpoint before the loaders see it.
//!
//! llama.cpp splits a large model into `<prefix>-00001-of-00004.gguf` ...
//! files. Fuel's GGUF loaders (`QuantizedLlama3Model::from_gguf`,
//! `gguf_file::Content::open`) take exactly ONE path, so loading shard 1 on
//! its own would fail much later with a "tensor not found" error that says
//! nothing about the real cause. This module names the cause up front:
//! which shard was given, how many there are, and which siblings are missing
//! (ROADMAP B5). It reads file NAMES only (via `mlmf-gguf`'s split helpers);
//! it does not open or merge shards. Loading a shard set needs a merged
//! reader that Fuel can consume, which is an open Fuel question.

use anyhow::{Result, bail};
use mlmf_gguf::split::{parse_split_file_name, shard_file_names};
use std::path::Path;

/// What the file name says about a split checkpoint.
#[derive(Debug, PartialEq, Eq)]
pub enum SplitStatus {
    /// Not a `-NNNNN-of-MMMMM.gguf` name: an ordinary single-file GGUF.
    Single,
    /// A shard. `missing` lists sibling file names absent from the directory.
    Split {
        number: u32,
        count: u32,
        missing: Vec<String>,
    },
}

/// Classify `file_name`, asking `exists` about each sibling shard name.
pub fn split_status(file_name: &str, exists: impl Fn(&str) -> bool) -> SplitStatus {
    let Some(parsed) = parse_split_file_name(file_name) else {
        return SplitStatus::Single;
    };
    // A parsed name has a plain prefix and a count in range, so this cannot
    // fail; an empty list would only understate `missing`, never hide a shard.
    let names = shard_file_names(&parsed.prefix, parsed.count).unwrap_or_default();
    let missing = names.into_iter().filter(|n| !exists(n)).collect();
    SplitStatus::Split {
        number: parsed.number,
        count: parsed.count,
        missing,
    }
}

/// Refuse a split checkpoint with an error that names the real cause.
///
/// Call at the top of every GGUF loader. A single-file GGUF passes.
pub fn refuse_split(path: &Path) -> Result<()> {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return Ok(());
    };
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    match split_status(name, |n| dir.join(n).is_file()) {
        SplitStatus::Single => Ok(()),
        SplitStatus::Split {
            number,
            count,
            missing,
        } => {
            let gaps = if missing.is_empty() {
                "all shards are present".to_string()
            } else {
                format!("missing: {}", missing.join(", "))
            };
            bail!(
                "{} is shard {number} of {count} of a split GGUF ({gaps}); sharded GGUF cannot be \
                 loaded yet (the weights loader takes one file; ROADMAP B5). Use a single-file \
                 GGUF, or merge the shards first (llama.cpp's `llama-gguf-split --merge`)",
                path.display()
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_name_is_single() {
        assert_eq!(
            split_status("model.Q4_0.gguf", |_| true),
            SplitStatus::Single
        );
        // Wrong digit width is not llama.cpp's convention.
        assert_eq!(split_status("m-1-of-2.gguf", |_| true), SplitStatus::Single);
    }

    #[test]
    fn a_complete_set_reports_no_missing() {
        let s = split_status("m-00002-of-00003.gguf", |_| true);
        assert_eq!(
            s,
            SplitStatus::Split {
                number: 2,
                count: 3,
                missing: vec![]
            }
        );
    }

    #[test]
    fn missing_siblings_are_named() {
        let s = split_status("m-00001-of-00003.gguf", |n| n == "m-00001-of-00003.gguf");
        assert_eq!(
            s,
            SplitStatus::Split {
                number: 1,
                count: 3,
                missing: vec![
                    "m-00002-of-00003.gguf".into(),
                    "m-00003-of-00003.gguf".into()
                ],
            }
        );
    }

    #[test]
    fn refuse_split_names_shard_and_gaps_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("m-00001-of-00002.gguf"), b"x").unwrap();
        let err = refuse_split(&dir.path().join("m-00001-of-00002.gguf"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("shard 1 of 2"), "{err}");
        assert!(err.contains("missing: m-00002-of-00002.gguf"), "{err}");
        // A single-file GGUF passes.
        assert!(refuse_split(&dir.path().join("plain.gguf")).is_ok());
    }
}
