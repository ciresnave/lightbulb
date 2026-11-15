// SPDX-License-Identifier: MIT OR Apache-2.0
//! Recognising a split (sharded) GGUF checkpoint before the loaders see it.
//!
//! llama.cpp splits a large model into `<prefix>-00001-of-00004.gguf` ...
//! files. Since Fuel 0.47.31 (fuel#660) its GGUF loaders open the whole set from shard 1, so Lightbulb loads a split model
//! (ROADMAP B5). This module reads file NAMES only (via `mlmf-gguf`'s split helpers): it finds the set, names what is
//! missing from an incomplete one, and picks shard 1 as the path to load.

use anyhow::{Result, bail};
use mlmf_gguf::split::{parse_split_file_name, shard_file_names};
use std::path::{Path, PathBuf};

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

/// The files of the split checkpoint `path` names, in shard order, or `None` for an ordinary single-file GGUF.
/// An incomplete set is an error that names the shard given, the count and the missing siblings.
pub fn shard_set(path: &Path) -> Result<Option<Vec<PathBuf>>> {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return Ok(None);
    };
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    match split_status(name, |n| dir.join(n).is_file()) {
        SplitStatus::Single => Ok(None),
        SplitStatus::Split {
            number,
            count,
            missing,
        } => complete_set(path, number, count, &missing).map(Some),
    }
}

/// The shard files in order, or the error for a set with `missing` siblings.
fn complete_set(path: &Path, number: u32, count: u32, missing: &[String]) -> Result<Vec<PathBuf>> {
    if !missing.is_empty() {
        bail!(
            "{} is shard {number} of {count} of a split GGUF and these are missing: {}",
            path.display(),
            missing.join(", ")
        );
    }
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let parsed = parse_split_file_name(name)
        .ok_or_else(|| anyhow::anyhow!("{} is not a split GGUF name", path.display()))?;
    let names = shard_file_names(&parsed.prefix, count)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    Ok(names.into_iter().map(|n| dir.join(n)).collect())
}

/// The path to load for `path`: itself for a single-file GGUF, shard 1 for a complete split set (any shard may be
/// named; the metadata and tokenizer live in shard 1, and Fuel opens the whole set from it). An incomplete set is an error.
pub fn resolve_load_path(path: &Path) -> Result<PathBuf> {
    match shard_set(path)? {
        None => Ok(path.to_path_buf()),
        Some(shards) => shards
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("{} names an empty shard set", path.display())),
    }
}

/// Whether `path` is the file a model list should show: a single-file GGUF, or shard 1 of a complete set. Shards 2 and
/// up are parts of a model, not models, and an incomplete set cannot be served.
pub fn is_listable(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    match parse_split_file_name(name) {
        None => true,
        Some(p) => p.number == 1 && matches!(shard_set(path), Ok(Some(_))),
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

    fn touch(dir: &Path, name: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, b"x").unwrap();
        p
    }

    #[test]
    fn an_incomplete_set_names_the_shard_and_the_gaps() {
        let dir = tempfile::tempdir().unwrap();
        let first = touch(dir.path(), "m-00001-of-00002.gguf");
        let err = shard_set(&first).unwrap_err().to_string();
        assert!(err.contains("shard 1 of 2"), "{err}");
        assert!(err.contains("m-00002-of-00002.gguf"), "{err}");
        assert!(resolve_load_path(&first).is_err());
        assert!(!is_listable(&first));
    }

    #[test]
    fn a_complete_set_resolves_to_shard_one_whichever_shard_is_named() {
        let dir = tempfile::tempdir().unwrap();
        let one = touch(dir.path(), "m-00001-of-00003.gguf");
        let two = touch(dir.path(), "m-00002-of-00003.gguf");
        let three = touch(dir.path(), "m-00003-of-00003.gguf");
        for named in [&one, &two, &three] {
            assert_eq!(resolve_load_path(named).unwrap(), one);
            assert_eq!(
                shard_set(named).unwrap().unwrap(),
                vec![one.clone(), two.clone(), three.clone()]
            );
        }
        assert!(is_listable(&one));
        assert!(!is_listable(&two) && !is_listable(&three));
    }

    #[test]
    fn a_single_file_is_itself_and_listable() {
        let dir = tempfile::tempdir().unwrap();
        let plain = touch(dir.path(), "plain.gguf");
        assert_eq!(shard_set(&plain).unwrap(), None);
        assert_eq!(resolve_load_path(&plain).unwrap(), plain);
        assert!(is_listable(&plain));
    }
}
