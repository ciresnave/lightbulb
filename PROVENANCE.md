# Third-party provenance — audit of inherited/derived files

Audit of lightbulb's third-party-derived files, per CireSnave's provenance rule
(`C:\Projects\CIRESNAVE-EXPECTATIONS.md` §6.4a, 2026-09-17): trace each file to its **origin**,
verify our copy against it, keep the origin's licence and notices, mark the file with the origin's
licence, never invent a holder. Method: baracuda's
[`docs/provenance/METHOD.md`](C:\Projects\baracuda\docs\provenance\METHOD.md) (`ciresnave/baracuda`
`origin/main`), written from baracuda's own 2026-10-14 audit. PM-tasked. Measured at `origin/main`
`c5b7fd77`, 2026-10-14.

**Enumeration** (`git ls-files` at the measured ref, never a disk walk): no `vendor`/`third_party`/
`external` directory exists in this repo. Licence/notice files present: `LICENSE-APACHE`,
`LICENSE-MIT` (the project's own dual licence) and `src/test_notice.rs` (a test fixture about
notification logic, not a licence notice — read and ruled out). Content sweep (`git grep -i -E
'SPDX-FileCopyrightText|Copyright \(c\)|adapted from|ported from|derived from|based on|taken
from'`) returned ~50 file hits; triaged individually. **Positive control**: the same sweep style
correctly finds the one real header below, so a null result on the other ~48 hits is a reading, not
a blind spot. Only one file in the entire repo carries an `SPDX-License-Identifier` header naming a
third-party origin (`git grep -c SPDX-License-Identifier -- '*.rs'` → exactly one non-zero file).
The ~48 other hits are prose ("based on available resources", citations to papers/algorithms like
the H2O paper, or "See: <url>" comparison references for a factual claim) — a prose mention is not
a derived file, per the method's own distinction, and none of them name a file as copied.

Three items, not one, because the connected history behind the one real header also resolves two
adjacent questions already raised and settled by CireSnave on 2026-08-19/20 — recorded here so they
are not silently forgotten and so a future sweep does not re-raise them as new.

## 1. Inventory

| item | origin | licence | pinned? | class / status |
|---|---|---|---|---|
| `src/cache/parallel_cache_builder.rs` | `huggingface/candle`, `candle-nn/src/kv_cache.rs` (`ScatteredCacheBuilder`/`IndicesAndMask`/`indices_and_mask()` — confirmed present upstream, same names) | Apache-2.0 OR MIT (confirmed via candle's own `LICENSE-APACHE`/`LICENSE-MIT`; `candle-nn`'s `Cargo.toml` uses `license.workspace = true` inheriting it) | **no** — vendored 2025-10-23 (first commit `0b1dc79`), no upstream commit SHA recorded then or since | MODIFIED (file's own header: "Modified for parallel batching with variable-length sequences") |
| `src/backend/marlin_ffi.rs` + its CUDA symbol declarations | `guoqingbao/attention.rs`, `src/kernels/src/ffi.rs` (per the file's own in-tree ruling, dated 2026-08-20) | MIT (attention.rs; no LICENSE file in that repo, hence GitHub reports `license: null`) | n/a — ABI-mandated interface, ruled not independently copyrightable | **RESOLVED** (CireSnave ruling already in the file; not re-litigated here) |
| CUDA kernel sources formerly under `kernels/` | unattributed, MIT-only third party (never identified by name in the deleted code; see commit message) | MIT (as found) | n/a | **RESOLVED BY DELETION**, commit `1e16b8a4`, 2026-08-19: "Remove vendored third-party CUDA kernels — unattributed and MIT-only" |

## 2. What holds (so it is not re-audited)

- `marlin_ffi.rs`'s ABI-interface ruling (no third-party copyright owed for the `extern "C"` symbol
  declarations themselves) stands; this audit did not re-open it, only confirmed the file still
  carries its own ruling text in place, unedited.
- The `kernels/` deletion already closed that gap; no vendored CUDA source remains in the tree
  (`git ls-files | grep -i '\.cu$\|\.cuh$'` → 0 hits, confirming the deletion is complete, not
  partial).
- `src/cache/parallel_cache_builder.rs`'s licence identifier (`Apache-2.0 OR MIT`) is correct:
  candle is dual-licensed exactly that way, confirmed directly against candle's own
  `LICENSE-APACHE`/`LICENSE-MIT` files, not inferred from this project's own default.
- No NOTICE file is owed: candle ships no NOTICE file (confirmed), and the Apache-2.0 §4(d)
  requirement only applies where the origin has one.

## 3. Gap register

| id | gap | status |
|---|---|---|
| G-PCB-1 | `parallel_cache_builder.rs`'s header claims `Copyright (c) 2023 Hugging Face`, but candle's own `LICENSE-MIT` text (fetched directly) carries **no copyright line at all** — the claim is not verifiable from anything upstream states, only from "Hugging Face" being the GitHub org that hosts `candle`. Per the method's rule ("do not invent a holder... the header then names the origin... never a person"), this is corrected below to name the origin repository rather than assert an unverified legal copyright holder. | **FIXED** (this PR, comment-only — see below) |
| G-PCB-2 | No upstream candle commit SHA is pinned for the fork, and none was recorded at vendor time (2025-10-23). Per the method, this should read "last upstream commit on or before the vendor date, date-bounded and not recovered" rather than nothing — but identifying that specific commit requires cloning candle's full history (`git clone --filter=blob:none --no-checkout`), not done here (no local clone available this session). | **open** — needs a PM-authorized follow-up with an actual candle clone to resolve, or an explicit decision to accept it as permanently date-bounded |
| G-PCB-3 | No blob-level diff was performed against candle's current `kv_cache.rs` (would require the same clone as G-PCB-2) — class MODIFIED is asserted from the file's own "Modified for..." comment and the structural match (`IndicesAndMask`, `indices_and_mask()`, `ScatteredCacheBuilder` all present upstream with the same names, confirmed via direct fetch), not from a byte-level diff. | **open** — same follow-up as G-PCB-2 would resolve this too |

No gap in this register touches a vendored file, a licence file, or more than a comment-line header
— per the task's own constraint, **no code bytes were changed by this PR**.

## 4. Fix applied (comment-only; code behind it is byte-identical)

`src/cache/parallel_cache_builder.rs`'s header: replaced the unverifiable `Copyright (c) 2023
Hugging Face` line with a statement naming the origin repository and noting upstream asserts no
holder, per the method's own worked example for exactly this case. The `Copyright (c) 2025-2026 Eric
Evans` line is unchanged — that one claims copyright over this project's own modifications, which is
not in question.
