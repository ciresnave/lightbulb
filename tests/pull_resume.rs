// SPDX-License-Identifier: MIT OR Apache-2.0
//! F1 resume proof (ROADMAP F1: "resume is not shown"): an interrupted `pull` continues from the bytes already on
//! disk instead of restarting, shown against a LOCAL Hub that serves a large file with `Range` support and drops the
//! connection mid-body. No network, no credentials.
//!
//! The file is 64 MiB of deterministic bytes by default; set `LIGHTBULB_RESUME_SOURCE` to any larger local file (the
//! nomic `model.safetensors`, say) to run it on that instead. It is served under a `.gguf` name because `pull` picks
//! `.gguf` files; the bytes are never parsed.
//!
//! What each arm shows (the server logs every `resolve` request, its `Range`, and the bytes it really wrote):
//! - fault: the first request is cut after `cut` bytes, so the pull FAILS, leaves a `.part` of exactly `cut` bytes, no
//!   final file, and that partial file's sha256 differs from the whole file's (the "without resume it is wrong" control).
//! - resume: a second pull sends `Range: bytes=<cut>-`, is served exactly `total - cut` bytes, and the final file is
//!   bytes-identical (sha256) to the source.
//! - control: the SAME sequence against a server that IGNORES `Range` also ends byte-identical, but its second request
//!   is served the whole `total`. So "the hash matches" alone cannot tell resume from restart; the served byte counts do.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener};
use std::sync::{Arc, Mutex};

use lightbulb::model_fuel::pull::{FileOutcome, parse_spec, pull};
use mlmf_source_hub::HubSource;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Served {
    range: Option<String>,
    wrote: usize,
}

fn source() -> Arc<Vec<u8>> {
    if let Some(p) = std::env::var_os("LIGHTBULB_RESUME_SOURCE") {
        return Arc::new(std::fs::read(&p).unwrap_or_else(|e| panic!("reading {p:?}: {e}")));
    }
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let v: Vec<u8> = (0..64 << 20)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect();
    Arc::new(v)
}

fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

type Ctx = (
    Arc<Vec<u8>>,
    Arc<Mutex<Option<usize>>>,
    Arc<Mutex<Vec<Served>>>,
    bool,
);

/// What the stub reads off a request: its first line and its `Range` value, if any.
fn read_request(s: &mut std::net::TcpStream) -> Option<(String, Option<String>)> {
    let end = b"\r\n\r\n";
    let mut req = Vec::new();
    let mut buf = [0u8; 2048];
    while !req.windows(4).any(|w| w == end) {
        match s.read(&mut buf) {
            Ok(0) | Err(_) => return None,
            Ok(n) => req.extend_from_slice(&buf[..n]),
        }
    }
    let text = String::from_utf8_lossy(&req).to_string();
    let line = text.lines().next().unwrap_or("").to_string();
    let range = text.lines().find_map(|l| {
        l.to_ascii_lowercase()
            .strip_prefix("range:")
            .map(|v| v.trim().to_string())
    });
    Some((line, range))
}

fn send_json(s: &mut std::net::TcpStream, j: String) {
    let h = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        j.len()
    );
    let _ = s.write_all(h.as_bytes());
    let _ = s.write_all(j.as_bytes());
}

fn tree_json(oid: &str, n: usize) -> String {
    format!(
        r#"[{{"type":"file","oid":"{oid}","size":{n},"lfs":{{"oid":"{oid}","size":{n},"pointerSize":120}},"path":"m-Q8_0.gguf"}}]"#
    )
}

/// The `N` of `Range: bytes=N-`, only when the server honours ranges.
fn range_start(range: &Option<String>, honor: bool) -> Option<usize> {
    let r = range.as_deref().filter(|_| honor)?;
    r.strip_prefix("bytes=")?.strip_suffix('-')?.parse().ok()
}

/// The response head and the offset the body starts at.
fn file_head(start: Option<usize>, total: usize) -> (String, usize) {
    match start {
        Some(n) => (
            format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Type: application/octet-stream\r\nContent-Range: bytes {n}-{}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                total - 1,
                total - n
            ),
            n,
        ),
        None => (
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n"
            ),
            0,
        ),
    }
}

/// Serve the file body. The fault: only the first request that asks for the whole file is cut, mid-body.
fn serve_file(s: &mut std::net::TcpStream, ctx: &Ctx, line: &str, range: Option<String>) {
    let (body, cut, log, honor) = ctx;
    let total = body.len();
    let (head, from) = file_head(range_start(&range, *honor), total);
    let _ = s.write_all(head.as_bytes());
    if line.starts_with("HEAD") {
        return;
    }
    let limit = if range.is_none() {
        cut.lock().unwrap().take()
    } else {
        None
    };
    let end = limit.map_or(total, |c| (from + c).min(total));
    let wrote = s
        .write_all(&body[from..end])
        .map(|_| end - from)
        .unwrap_or(0);
    let _ = s.flush();
    if limit.is_some() {
        let _ = s.shutdown(Shutdown::Both); // mid-body, with Content-Length unmet
    }
    log.lock().unwrap().push(Served { range, wrote });
}

fn handle(mut s: std::net::TcpStream, ctx: Ctx, oid: String) {
    let Some((line, range)) = read_request(&mut s) else {
        return;
    };
    if line.contains("/revision/") {
        send_json(&mut s, r#"{"sha":"abc123"}"#.into());
    } else if line.contains("/tree/") {
        send_json(&mut s, tree_json(&oid, ctx.0.len()));
    } else if line.contains("/resolve/abc123/m-Q8_0.gguf") {
        serve_file(&mut s, &ctx, &line, range);
    } else {
        let _ = s
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    }
}

/// A one-file Hub. `cut`: the first body request WITHOUT a `Range` is cut after that many bytes. `honor_range`:
/// whether a `Range: bytes=N-` is answered 206 (true) or ignored with a full 200 (false).
fn hub(
    body: Arc<Vec<u8>>,
    cut: Arc<Mutex<Option<usize>>>,
    honor_range: bool,
) -> (String, Arc<Mutex<Vec<Served>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let log = Arc::new(Mutex::new(Vec::new()));
    let oid = sha(&body);
    let ctx: Ctx = (body, cut, log.clone(), honor_range);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(s) = stream else { break };
            let (ctx, oid) = (ctx.clone(), oid.clone());
            std::thread::spawn(move || handle(s, ctx, oid));
        }
    });
    (url, log)
}

struct Run {
    total: usize,
    cut: usize,
    part_len: u64,
    part_sha: String,
    full_sha: String,
    final_sha: String,
    served: Vec<Served>,
}

fn run(honor_range: bool) -> Run {
    let body = source();
    let total = body.len();
    let cut = total * 37 / 100;
    let (url, log) = hub(body.clone(), Arc::new(Mutex::new(Some(cut))), honor_range);
    let dir = tempfile::tempdir().unwrap();
    let h = HubSource::new(dir.path().join(".cache")).with_endpoint(&url);
    let spec = parse_spec("o/n:Q8_0").unwrap();

    // First pull: the connection dies mid-body.
    let err =
        pull(&h, &spec, dir.path(), &|_| None, &mut |_, _| {}).expect_err("the cut pull must fail");
    eprintln!("first pull failed as intended: {err}");
    let part = dir.path().join("m-Q8_0.gguf.part");
    assert!(part.is_file(), "a .part must be left behind");
    assert!(
        !dir.path().join("m-Q8_0.gguf").exists(),
        "no final file after a failed pull"
    );
    let part_bytes = std::fs::read(&part).unwrap();
    let part_len = part_bytes.len() as u64;
    let part_sha = sha(&part_bytes);
    drop(part_bytes);

    // Second pull: the server no longer cuts.
    let out = pull(&h, &spec, dir.path(), &|_| None, &mut |_, _| {}).expect("the second pull");
    assert!(
        matches!(out.as_slice(), [FileOutcome::Downloaded(_)]),
        "{out:?}"
    );
    let final_sha = sha(&std::fs::read(dir.path().join("m-Q8_0.gguf")).unwrap());
    let served = log.lock().unwrap().clone();
    Run {
        total,
        cut,
        part_len,
        part_sha,
        full_sha: sha(&body),
        final_sha,
        served,
    }
}

#[test]
fn an_interrupted_pull_resumes_from_the_bytes_on_disk() {
    let r = run(true);
    eprintln!(
        "RESUME total {} cut {} part {} served {:?}",
        r.total, r.cut, r.part_len, r.served
    );
    // The fault left a partial file that is NOT the file (control: without a resume the result would be wrong).
    assert!(r.part_len > 0 && (r.part_len as usize) < r.total);
    assert_ne!(
        r.part_sha, r.full_sha,
        "the partial file must differ from the whole one"
    );
    // The end state is bytes-identical.
    assert_eq!(r.final_sha, r.full_sha);
    // And it got there by resuming: the second body request asked from the .part's length and was served only the rest.
    assert_eq!(r.served.len(), 2, "{:?}", r.served);
    assert_eq!(
        r.served[1].range.as_deref(),
        Some(format!("bytes={}-", r.part_len).as_str())
    );
    assert_eq!(r.served[1].wrote, r.total - r.part_len as usize);
    assert_eq!(
        r.served[0].wrote as u64 + r.served[1].wrote as u64,
        r.total as u64,
        "across both requests the server wrote the file exactly once"
    );
}

/// The control: a server that ignores `Range` also yields a byte-identical file, but by restarting. If this test's
/// byte counts matched the resume test's, the hash check alone would be unable to tell the two apart.
#[test]
fn a_server_that_ignores_range_gets_a_restart_not_a_resume() {
    let r = run(false);
    eprintln!(
        "RESTART total {} part {} served {:?}",
        r.total, r.part_len, r.served
    );
    assert_eq!(
        r.final_sha, r.full_sha,
        "a restart still ends byte-identical"
    );
    assert_eq!(r.served.len(), 2, "{:?}", r.served);
    assert_eq!(r.served[1].wrote, r.total, "served the whole file again");
    assert!(
        r.served[0].wrote + r.served[1].wrote > r.total,
        "the bytes already on disk were sent twice"
    );
}
