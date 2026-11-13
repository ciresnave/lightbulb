// SPDX-License-Identifier: MIT OR Apache-2.0
//! Model pull (ROADMAP F1): turn `repo[:quant]` into one GGUF, or a complete split set, and download it into the models
//! directory with a hash check.
//!
//! The network half is `mlmf-source-hub` (listing with sizes and hashes, download to a caller path through a `.part` file,
//! resumable, progress callback, `RemoteHash::verify`). What is Lightbulb's own is the choice: which listed file(s) match a
//! quant tag. That choice ([`gguf_units`], [`select_unit`]) is pure over [`ListedFile`], so it is tested without a network.

use anyhow::{Result, bail};
use mlmf_gguf::split::{SplitFileName, parse_split_file_name};
use mlmf_source_hub::{HubSource, Progress, RemoteFile, Revision};
use std::path::{Component, Path, PathBuf};

/// `repo[:quant]`, as typed (`Qwen/Qwen3-0.6B-GGUF:Q8_0`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullSpec {
    pub repo: String,
    pub quant: Option<String>,
}

/// Parse `owner/name` or `owner/name:QUANT`. A repo needs exactly one `/`; an empty quant is an error, not "any".
pub fn parse_spec(spec: &str) -> Result<PullSpec> {
    let (repo, quant) = match spec.split_once(':') {
        Some((_, "")) => bail!("`{spec}`: empty quant after `:`"),
        Some((r, q)) => (r, Some(q.to_string())),
        None => (spec, None),
    };
    if repo.matches('/').count() != 1 || repo.starts_with('/') || repo.ends_with('/') {
        bail!("`{repo}` is not `owner/name`");
    }
    Ok(PullSpec {
        repo: repo.to_string(),
        quant,
    })
}

/// A listed file reduced to what the choice needs (the Hub's own type is not constructible outside its crate).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListedFile {
    pub path: String,
    pub size: u64,
}

/// One thing to download: a single `.gguf`, or every shard of one split model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    /// The file name without `.gguf` (single) or the split prefix (a split set).
    pub label: String,
    /// In download order (shard 1 first).
    pub files: Vec<ListedFile>,
}

impl Unit {
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

fn base_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

type Set = (SplitFileName, Vec<ListedFile>);

fn add_to_set(sets: &mut Vec<Set>, name: SplitFileName, f: &ListedFile) {
    let same = |(k, _): &Set| k.prefix == name.prefix && k.count == name.count;
    match sets.iter_mut().find(|s| same(s)) {
        Some((_, v)) => v.push(f.clone()),
        None => sets.push((name, vec![f.clone()])),
    }
}

fn complete_set((name, mut files): Set) -> Result<Unit> {
    files.sort_by(|a, b| a.path.cmp(&b.path));
    if files.len() != name.count as usize {
        bail!(
            "split model `{}` lists {} of {} shards",
            name.prefix,
            files.len(),
            name.count
        );
    }
    Ok(Unit {
        label: name.prefix,
        files,
    })
}

/// Group `.gguf` files into single files and split sets. A split set missing a shard is an error naming the model.
pub fn gguf_units(files: &[ListedFile]) -> Result<Vec<Unit>> {
    let mut units = Vec::new();
    let mut sets: Vec<Set> = Vec::new();
    for f in files.iter().filter(|f| f.path.ends_with(".gguf")) {
        match parse_split_file_name(base_name(&f.path)) {
            None => units.push(Unit {
                label: base_name(&f.path).trim_end_matches(".gguf").to_string(),
                files: vec![f.clone()],
            }),
            Some(n) => add_to_set(&mut sets, n, f),
        }
    }
    for set in sets {
        units.push(complete_set(set)?);
    }
    Ok(units)
}

/// Does `label` carry `quant` as a whole token? Boundaries are `-`, `.` and `/` (a quant name has `_` inside:
/// `Q4_K` must not match `Q4_K_M`). Case-insensitive.
fn has_quant(label: &str, quant: &str) -> bool {
    let (l, q) = (label.to_ascii_lowercase(), quant.to_ascii_lowercase());
    l.match_indices(&q).any(|(i, _)| {
        let before = i == 0 || matches!(l.as_bytes()[i - 1], b'-' | b'.' | b'/');
        let after = l
            .as_bytes()
            .get(i + q.len())
            .is_none_or(|c| matches!(c, b'-' | b'.'));
        before && after
    })
}

fn labels(units: &[&Unit]) -> String {
    units
        .iter()
        .map(|u| u.label.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Pick the one unit `quant` names (or the only unit when `quant` is `None`), or say exactly why not.
pub fn select_unit(units: &[Unit], quant: Option<&str>) -> Result<Unit> {
    let all: Vec<&Unit> = units.iter().collect();
    let hits: Vec<&Unit> = match quant {
        Some(q) => units.iter().filter(|u| has_quant(&u.label, q)).collect(),
        None => all.clone(),
    };
    match hits.as_slice() {
        [one] => Ok((*one).clone()),
        [] if all.is_empty() => bail!("the repo lists no .gguf file"),
        [] => bail!(
            "no .gguf matches quant `{}`; the repo has: {}",
            quant.unwrap_or(""),
            labels(&all)
        ),
        many => bail!(
            "{} .gguf choices; name one as repo:QUANT: {}",
            many.len(),
            labels(many)
        ),
    }
}

/// A listed path is safe to join under the models directory: relative, no `..`, no root or drive.
pub fn safe_relative(path: &str) -> Result<PathBuf> {
    let p = Path::new(path);
    let plain = |c: Component| matches!(c, Component::Normal(_));
    if path.is_empty() || !p.components().all(plain) {
        bail!("refusing the listed path `{path}`");
    }
    Ok(p.to_path_buf())
}

/// What [`pull`] did for one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileOutcome {
    /// Already present and its hash matched: nothing downloaded.
    Present(PathBuf),
    /// Downloaded and verified.
    Downloaded(PathBuf),
}

/// Download one listed file to `dest` unless `dest` already verifies; verify after downloading. A file that fails
/// verification is removed, never left under its final name.
fn pull_file(
    hub: &HubSource,
    repo: &str,
    sha: &str,
    f: &RemoteFile,
    dest: &Path,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<FileOutcome> {
    if dest.is_file() && f.hash.verify(dest).is_ok() {
        return Ok(FileOutcome::Present(dest.to_path_buf()));
    }
    let rev = Revision::Pinned(sha.to_string());
    hub.fetch_to_with_progress(repo, &rev, &f.path, dest, on_progress)
        .map_err(|e| anyhow::anyhow!("downloading {}: {e}", f.path))?;
    if let Err(e) = f.hash.verify(dest) {
        let _ = std::fs::remove_file(dest);
        bail!("{} failed verification and was removed: {e}", f.path);
    }
    Ok(FileOutcome::Downloaded(dest.to_path_buf()))
}

/// Resolve `spec` on `hub`, pick one unit and download every file of it into `models_dir`. The revision is resolved once
/// and every file is fetched at that SHA, so a push to the repo mid-pull cannot mix two revisions.
pub fn pull(
    hub: &HubSource,
    spec: &PullSpec,
    models_dir: &Path,
    on_progress: &mut dyn FnMut(&str, Progress),
) -> Result<Vec<FileOutcome>> {
    let listing = hub
        .list_files(&spec.repo, &Revision::Ref("main".into()))
        .map_err(|e| anyhow::anyhow!("listing {}: {e}", spec.repo))?;
    let listed: Vec<ListedFile> = listing
        .files
        .iter()
        .map(|f| ListedFile {
            path: f.path.clone(),
            size: f.size,
        })
        .collect();
    let unit = select_unit(&gguf_units(&listed)?, spec.quant.as_deref())?;
    let sha = &listing.resolved_revision;
    let mut out = Vec::new();
    for lf in &unit.files {
        let remote = find_listed(&listing.files, &lf.path)?;
        let dest = models_dir.join(safe_relative(&lf.path)?);
        out.push(pull_file(hub, &spec.repo, sha, remote, &dest, &mut |p| {
            on_progress(&lf.path, p)
        })?);
    }
    Ok(out)
}

/// The listing entry for `path`, which the chosen unit was built from and so must still contain.
fn find_listed<'a>(files: &'a [RemoteFile], path: &str) -> Result<&'a RemoteFile> {
    files
        .iter()
        .find(|f| f.path == path)
        .ok_or_else(|| anyhow::anyhow!("{path} vanished from the listing"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lf(path: &str, size: u64) -> ListedFile {
        ListedFile {
            path: path.into(),
            size,
        }
    }

    fn repo() -> Vec<ListedFile> {
        vec![
            lf("README.md", 1),
            lf("m-Q4_K_M.gguf", 400),
            lf("m-Q8_0.gguf", 800),
            lf("big-Q6_K-00002-of-00002.gguf", 300),
            lf("big-Q6_K-00001-of-00002.gguf", 300),
        ]
    }

    #[test]
    fn spec_parses_repo_and_quant() {
        let s = parse_spec("Qwen/Qwen3-0.6B-GGUF:Q8_0").unwrap();
        assert_eq!(s.repo, "Qwen/Qwen3-0.6B-GGUF");
        assert_eq!(s.quant.as_deref(), Some("Q8_0"));
        assert_eq!(parse_spec("a/b").unwrap().quant, None);
        for bad in ["", "ab", "a/b/c", "a/b:", "/b", "a/"] {
            assert!(parse_spec(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn quant_matches_whole_tokens_only() {
        assert!(has_quant("model-Q4_K_M", "q4_k_m"));
        assert!(has_quant("model.Q8_0", "Q8_0"));
        assert!(
            !has_quant("model-Q4_K_M", "Q4_K"),
            "Q4_K is a prefix of Q4_K_M, not that quant"
        );
        assert!(!has_quant("model-IQ4_XS", "Q4_XS"));
    }

    #[test]
    fn units_group_singles_and_complete_split_sets() {
        let units = gguf_units(&repo()).unwrap();
        let labels: Vec<_> = units.iter().map(|u| u.label.as_str()).collect();
        assert_eq!(labels, ["m-Q4_K_M", "m-Q8_0", "big-Q6_K"]);
        let big = &units[2];
        assert_eq!(big.files[0].path, "big-Q6_K-00001-of-00002.gguf");
        assert_eq!(big.total_bytes(), 600);
    }

    #[test]
    fn a_split_set_missing_a_shard_is_an_error() {
        let files = vec![
            lf("x-00001-of-00003.gguf", 1),
            lf("x-00003-of-00003.gguf", 1),
        ];
        let err = gguf_units(&files).unwrap_err().to_string();
        assert!(err.contains("2 of 3 shards"), "{err}");
    }

    #[test]
    fn select_by_quant_picks_single_or_split() {
        let units = gguf_units(&repo()).unwrap();
        assert_eq!(select_unit(&units, Some("Q8_0")).unwrap().files.len(), 1);
        assert_eq!(select_unit(&units, Some("q6_k")).unwrap().files.len(), 2);
    }

    #[test]
    fn select_refuses_none_ambiguous_and_missing() {
        let units = gguf_units(&repo()).unwrap();
        let none = select_unit(&units, None).unwrap_err().to_string();
        assert!(
            none.contains("3 .gguf choices") && none.contains("m-Q8_0"),
            "{none}"
        );
        let miss = select_unit(&units, Some("Q2_K")).unwrap_err().to_string();
        assert!(miss.contains("no .gguf matches quant `Q2_K`"), "{miss}");
        assert!(select_unit(&[], None).is_err());
        let one = gguf_units(&[lf("only.gguf", 1)]).unwrap();
        assert_eq!(select_unit(&one, None).unwrap().label, "only");
    }

    #[test]
    fn safe_relative_refuses_escapes() {
        assert!(safe_relative("m.gguf").is_ok());
        assert!(safe_relative("sub/m.gguf").is_ok());
        for bad in ["", "../m.gguf", "/abs.gguf", "a/../../m.gguf"] {
            assert!(safe_relative(bad).is_err(), "{bad}");
        }
    }

    /// A one-file Hub on localhost: `body` is what it serves, `claimed` is the bytes whose sha256 the listing announces.
    fn serve_hub(
        body: &'static [u8],
        claimed: &'static [u8],
    ) -> (String, std::thread::JoinHandle<()>) {
        use sha2::{Digest, Sha256};
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let oid = format!("{:x}", Sha256::digest(claimed));
        let handle = std::thread::spawn(move || {
            for stream in listener.incoming().take(12) {
                let mut s = stream.unwrap();
                let mut buf = [0u8; 4096];
                let n = s.read(&mut buf).unwrap_or(0);
                let line = String::from_utf8_lossy(&buf[..n])
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string();
                let (ctype, payload): (&str, Vec<u8>) = if line.contains("/revision/") {
                    ("application/json", br#"{"sha":"abc123"}"#.to_vec())
                } else if line.contains("/tree/") {
                    let j = format!(
                        r#"[{{"type":"file","oid":"{o}","size":{n},"lfs":{{"oid":"{o}","size":{n},"pointerSize":120}},"path":"m-Q8_0.gguf"}}]"#,
                        o = oid,
                        n = body.len()
                    );
                    ("application/json", j.into_bytes())
                } else if line.contains("/resolve/abc123/m-Q8_0.gguf") {
                    ("application/octet-stream", body.to_vec())
                } else {
                    ("text/plain", Vec::new())
                };
                let head = format!(
                    "HTTP/1.1 200 OK
Content-Type: {ctype}
Content-Length: {}
Connection: close

",
                    payload.len()
                );
                let _ = s.write_all(head.as_bytes());
                let _ = s.write_all(&payload);
            }
        });
        (url, handle)
    }

    fn hub_for(url: &str, dir: &Path) -> HubSource {
        HubSource::new(dir.join(".cache")).with_endpoint(url)
    }

    #[test]
    fn pull_downloads_verifies_and_skips_what_is_present() {
        let (url, _srv) = serve_hub(b"GGUFdata", b"GGUFdata");
        let dir = tempfile::tempdir().unwrap();
        let hub = hub_for(&url, dir.path());
        let spec = parse_spec("o/n:Q8_0").unwrap();
        let first = pull(&hub, &spec, dir.path(), &mut |_, _| {}).unwrap();
        assert!(matches!(first.as_slice(), [FileOutcome::Downloaded(_)]));
        assert_eq!(
            std::fs::read(dir.path().join("m-Q8_0.gguf")).unwrap(),
            b"GGUFdata"
        );
        let again = pull(&hub, &spec, dir.path(), &mut |_, _| {}).unwrap();
        assert!(
            matches!(again.as_slice(), [FileOutcome::Present(_)]),
            "control: a verified file is not fetched again"
        );
    }

    #[test]
    fn a_download_that_fails_its_hash_is_removed() {
        let (url, _srv) = serve_hub(b"tampered!", b"GGUFdata");
        let dir = tempfile::tempdir().unwrap();
        let hub = hub_for(&url, dir.path());
        let err = pull(
            &hub,
            &parse_spec("o/n").unwrap(),
            dir.path(),
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(err.to_string().contains("failed verification"), "{err}");
        assert!(
            !dir.path().join("m-Q8_0.gguf").exists(),
            "a bad file must not stay under its final name"
        );
    }
}
