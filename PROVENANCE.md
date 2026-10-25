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
| `src/cache/parallel_cache_builder.rs` | `huggingface/candle`, `candle-nn/src/kv_cache.rs` (`ScatteredCacheBuilder`/`IndicesAndMask`/`indices_and_mask()` — confirmed present upstream, same names) | Apache-2.0 OR MIT (confirmed via candle's own `LICENSE-APACHE`/`LICENSE-MIT`; `candle-nn`'s `Cargo.toml` uses `license.workspace = true` inheriting it) | **yes, date-bounded**: `3b35cfca9462b88ab66f726277571ffb4a35378d` (2025-09-09, "Update kv_cache.rs (#3035)") — the last upstream commit to this file on or before our vendor date (2025-10-23, first commit `0b1dc79`); no exact vendor-time SHA was recorded, so this is a lower bound, not a recovered fact | MODIFIED, confirmed by blob-level comparison (§below), not just the file's own claim; **file absent at `origin/main` as of 2026-10-25** (removed with the candlelight code, PR #195), so nothing derived from it remains in `src/`; see `docs/THIRD_PARTY_NOTICES.md` §2 |
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
- **Follow-up (2026-10-14, PM-ruled): G-PCB-2 and G-PCB-3 resolved.** Cloned
  `huggingface/candle` (`git clone --filter=blob:none`) into a scratch dir outside
  `C:\Projects` (not a shared tree). `candle-nn/src/kv_cache.rs`'s commit history shows the file
  last touched, before our 2025-10-23 vendor date, at `3b35cfca9462b88ab66f726277571ffb4a35378d`
  (2025-09-09) — pinned above as the (date-bounded) origin commit.
  - **Whole-file diff is not the right test here**: our file has grown to 3118 lines against
    upstream's ~238-line `ScatteredCacheBuilder` section (13x), from unrelated additions
    (eviction, streaming, H2O policy wiring) accumulated since the fork — a naive full-file diff
    (3358 changed lines) measures that growth, not the fork's fidelity.
  - **Line-level comparison of the actually-forked pieces** instead: our `IndicesAndMask` struct
    (`parallel_cache_builder.rs:42-58`) matches upstream's (`kv_cache.rs:404-412` at the pinned
    commit) field-for-field, plus an added `active: Vec<bool>` field and doc comments. Our
    `indices_and_mask()` (`parallel_cache_builder.rs:2232-2340`) opens with the same early-return
    guard, the same per-slot loop structure, and — strongest single piece of evidence — the exact
    same performance-rationale comment, byte-for-byte: `// up being almost 10x faster with candle
    0.9.0. This has been fixed in candle 0.9.1.` (our line 2331, upstream line 572 of the pinned
    blob). `git log -S` on that exact string against `kv_cache.rs` finds only one commit,
    `cd96fa80` (2025-05-01, "Add a scattered kv cache", PR #2936) — unchanged through the pinned
    commit, confirming this isn't convergent phrasing. Our version then diverges to add
    streaming-config-aware indexing and eviction-aware masking (the real "Modified for parallel
    batching" content), substantiating the class rather than just repeating the file's own claim.
  - Licence files were NOT re-verified for byte-identity against candle's root `LICENSE-MIT`/
    `LICENSE-APACHE` (hashed and found to differ from lightbulb's own, as expected: these are
    whole-project licence files, not a vendored copy of candle's — there is no `vendor/`
    directory here for a byte-identity check to apply to, unlike baracuda's audit). The *type*
    (Apache-2.0 OR MIT) was already confirmed correct in §2 above; that is the only claim this
    file's SPDX line makes.

## 3. Gap register

| id | gap | status |
|---|---|---|
| G-PCB-1 | `parallel_cache_builder.rs`'s header claims `Copyright (c) 2023 Hugging Face`, but candle's own `LICENSE-MIT` text (fetched directly) carries **no copyright line at all** — the claim is not verifiable from anything upstream states, only from "Hugging Face" being the GitHub org that hosts `candle`. Per the method's rule ("do not invent a holder... the header then names the origin... never a person"), this is corrected below to name the origin repository rather than assert an unverified legal copyright holder. | **FIXED** (this PR, comment-only — see below) |
| G-PCB-2 | No upstream candle commit SHA was pinned for the fork. | **FIXED** (follow-up, this PR) — pinned to `3b35cfca9462b88ab66f726277571ffb4a35378d`, date-bounded (see §2) |
| G-PCB-3 | No blob-level diff was performed against candle. | **FIXED** (follow-up, this PR) — line-level comparison of the forked struct/function against the pinned commit, including a verbatim-matched comment proving direct copying (see §2); not a whole-file diff, which would have measured 13x of unrelated growth instead of fork fidelity |

No gap in this register touches a vendored file, a licence file, or more than a comment-line header
— per the task's own constraint, **no code bytes were changed by this PR**.

## 4. Fix applied (comment-only; code behind it is byte-identical)

`src/cache/parallel_cache_builder.rs`'s header: replaced the unverifiable `Copyright (c) 2023
Hugging Face` line with a statement naming the origin repository and noting upstream asserts no
holder, per the method's own worked example for exactly this case. The `Copyright (c) 2025-2026 Eric
Evans` line is unchanged — that one claims copyright over this project's own modifications, which is
not in question.
