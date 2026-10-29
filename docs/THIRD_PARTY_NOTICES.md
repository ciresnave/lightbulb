# Third-party notices

Lightbulb is licensed `MIT OR Apache-2.0` (`LICENSE-MIT`, `LICENSE-APACHE` at the repository root). This
file records third-party material that is, or was, carried in this repository, and where its credit and
licence come from. Method and evidence: [`PROVENANCE.md`](../PROVENANCE.md). Measured at `origin/main`
`baa2f63`, 2026-10-25 (`git ls-files`, never a disk walk).

## 1. Material present in the tree

| File | Origin | Licence of origin | What we carry |
|---|---|---|---|
| `src/backend/marlin_ffi.rs` (compiled only with `feature = "cuda"`) | [`guoqingbao/attention.rs`](https://github.com/guoqingbao/attention.rs), `src/kernels/src/ffi.rs` | MIT (declared in that repo's `Cargo.toml`; the repo has no `LICENSE` file, so GitHub reports none) | `extern "C"` declarations of the Marlin CUDA kernel entry points, including six parameter renames (`A`→`inputs`, `B`→`weight`, `C`→`out`, `prob_m`/`prob_k`/`prob_n`→`m`/`k`/`n`) that match that file. |

Status of `marlin_ffi.rs`: CireSnave ruled on 2026-08-20 that the block is **interface, not expression** (its
types and order are dictated by the CUDA kernel ABI), so no third-party copyright is owed and this crate's
`MIT OR Apache-2.0` is truthful for the file. The ruling is recorded in the file's own header and is not
re-opened here. The parameter-name match shows the declarations were derived from that source; the ruling is
that what was derived is not independently protectable. This notice credits the origin anyway, because
credit is owed to a source we derived from whether or not a licence requires it. The kernels these symbols
bind are **not** in this repository and nothing in the tree provides them (see `build.rs`).

## 2. Material removed from the tree (kept for the record)

| What | Origin | Removal |
|---|---|---|
| `kernels/` CUDA sources (eight files) | `guoqingbao/attention.rs` (MIT-only), byte-identical per `build.rs`'s record of the deletion; the copies carried no attribution | Deleted (build.rs dates it 2026-08-20; `PROVENANCE.md` and the commit say 2026-08-19), commit `1e16b8a4` ("Remove vendored third-party CUDA kernels — unattributed and MIT-only") |
| `src/cache/parallel_cache_builder.rs` (modified fork of `ScatteredCacheBuilder` / `IndicesAndMask`) | `huggingface/candle`, `candle-nn/src/kv_cache.rs`, pinned (date-bounded) at `3b35cfca9462b88ab66f726277571ffb4a35378d`; `Apache-2.0 OR MIT`, no NOTICE file upstream | Absent at `origin/main` (removed with the candlelight code, 2026-10-22, PR #195). No code derived from it remains in `src/`; `PROVENANCE.md` §1 describes the file as it stood. |

## 3. Dependencies

Lightbulb depends on crates from crates.io; each keeps its own licence and copyright notice, which ships in
the crate's published source. This repository does **not** vendor any dependency (no `vendor/` directory).
The per-dependency licence list is [`THIRD_PARTY_LICENSES.tsv`](THIRD_PARTY_LICENSES.tsv): one row per
`crate@version` and one column per SPDX licence found, an `X` marking each licence that crate carries. It is
**generated, not hand-written**, by `cargo deny --locked list -f tsv -l crate | grep -v '^lightbulb@'` from
`Cargo.lock` under the repository's `deny.toml` (the workspace crate's own row is excluded, since it changes with
every version bump), and the CI job `licence-list` regenerates it and fails on any difference. To refresh it
after a dependency change, run that command and redirect it to the file. It lists licences as `cargo deny`
detects them from each crate's published source; it does not reproduce the licence texts or copyright notices,
which ship in each crate's source.

## 4. Models and data

No model weights, tokenizers or datasets are tracked in this repository (`git ls-files` shows no `.gguf`,
`.safetensors` or tokenizer file). Two golden fixtures hold greedy-decode output captured from third-party
models: `tests/fixtures/model_fuel_golden/v1/tinyllama_f32_greedy.json` (TinyLlama-1.1B) and
`tests/fixtures/awq_reference/v1/qwen3_0_6b_awq_greedy.json` (Qwen3-0.6B AWQ). They contain generated text
and token ids, not weights; the licences of those models are their publishers' and are not restated here.
Synthetic test checkpoints (`src/test_models.rs`) are produced by this project. Models a user loads keep
their publishers' licences.
