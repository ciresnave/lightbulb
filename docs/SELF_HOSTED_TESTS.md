# Real-model tests on the self-hosted runner (design note, ROADMAP C1b)

Status: **design only, nothing built.** Written 2026-11-23 against `origin/main` f1f8f17 plus branch `lane-2026-11-23-c1`.
Counts are `git grep '#\[ignore'` over `tests/` and `src/` (111 lines); a count is not a list, so the groups below
name what each one needs.

## The rule being applied

CireSnave 2026-10-16 (`CIRESNAVE-EXPECTATIONS.md` 5.1g/5.1h): a test that cannot run on a GitHub-hosted runner but could run
on the self-hosted one runs there, with the 5.1h conditions (trusted triggers, same-repo branches only, no fork code, no
secrets); `#[ignore]` is a last resort with a reason, an owner and a detector. A test no runner can run is named UNVERIFIED,
never "covered". This repo has no self-hosted workflow yet (`ci.yml` only skips two TinyLlama tests by name,
"until that job lands"); `aocl/.github/workflows/selfhosted.yml` is the pattern (dispatch only, `if: github.ref ==
'refs/heads/main'`, `runs-on: [self-hosted, Windows, X64, <label>]`, a step that fails loudly when the inputs are absent),
and `vulkane`'s `VULKANE_REQUIRE_DEVICE` is the gate pattern.

## What the ignored tests need

Counts are `#[ignore` lines per file (`git grep -c`, same ref); a file's line count includes tests that need only part of
what is listed. 111 lines in all.

| Group | Lines | Needs | Env it reads today | Runs on the runner? |
|---|---|---|---|---|
| Postgres: `api_integration_tests` 18, `persistant_migrate_tests` 3 | 21 | a PostgreSQL | `LIGHTBULB_TEST_DATABASE_URL` | **Already run**: the hosted `postgres_integration` job runs both binaries with `-- --ignored` (`ci.yml`). Not a runner matter; the `#[ignore]` stays as a Postgres gate with that detector. |
| Real GGUF corpus: `src/gguf/mod.rs` 14, `gguf_metadata_harness` 3 (+ census / README-figure tests that use `LIGHTBULB_REQUIRE_CORPUS`) | ~17+ | a directory of third-party `.gguf` files (30 files measured) | `LIGHTBULB_GGUF_CORPUS`, `LIGHTBULB_REQUIRE_CORPUS`, `LIGHTBULB_METADATA_DUMP`, `LIGHTBULB_DUMP_A/B` | Yes, CPU only. |
| One real GGUF: `c1_prefill_vs_decode` 5, `gguf_serving_e2e` 3, `gguf_bpe_tokenizer_fidelity` 2 | 10 | a 637 MB TinyLlama Q4_0 GGUF; for fidelity also its reference `tokenizer.json` | `LIGHTBULB_GGUF`, `LIGHTBULB_BPE_PAIRS`, `LIGHTBULB_REF_TOKENIZER`, `LIGHTBULB_BPE_UNVERIFIED_GGUF` | Yes, CPU; minutes per test. |
| TinyLlama safetensors checkpoint: `model_fuel_golden` 16, `api_result_metadata` 5, `chat_template_e2e` 2, `chat_template_render` 2, `generate.rs` 3, `contracts_live` 6 | ~34 | the 2.2 GB checkpoint (the f32 load is ~4.4 GB of RAM) | `TINYLLAMA_DIR` or the HF cache | Yes, CPU. `model_fuel_golden` says its bit-exactness "only holds on the capture machine", so it needs that machine class, not any runner. `api_result_metadata` already has synthetic counterparts for its behaviours. |
| GPU: one CUDA + TinyLlama test, the `LB_GGUF` bench, `marlin_ffi` | ~3 | a CUDA device | `LB_GGUF` | Only on a GPU runner: UNVERIFIED until one is registered. |
| Not a runner matter | ~3 | `SemanticSearch`'s placeholder embedding (a product gap), a measurement, a test that fails on a Fuel Q6_K dequant gap | - | No. Fix the cause. |
| Long-running load tests (`load_stress_tests` 3) | 3 | a live server | - | Not assessed here. |

The groups overlap at the edges and do not sum to 111; they are what each file's own ignore reason and header say, not a
re-measurement of every test.

## Design

1. **Remove `#[ignore]` from the real-model groups and gate by environment instead.** One shared helper (the existing
   `test_notice::skip_unless_required`, already used by `gguf/mod.rs` and four test files) decides: input present -> run;
   absent and `LIGHTBULB_REQUIRE_MODEL` / `LIGHTBULB_REQUIRE_CORPUS` on -> **fail**; absent and not required -> print a
   greppable `SKIP:` line and pass. Tests that today `.expect(MISSING)` (they only ever ran by choice) move to the same
   helper. This keeps laptops and the hosted job green without `#[ignore]`.
2. **A skip passes, so it proves nothing: count it.** The runner job sets `LIGHTBULB_REQUIRE_MODEL=1` and
   `LIGHTBULB_REQUIRE_CORPUS=1` and also greps each filtered run for the exact `N passed` (as the synthetic-checkpoint step
   in `ci.yml` does), because a name filter that matches nothing exits 0.
3. **One workflow, `selfhosted.yml`:** `workflow_dispatch` first, widening to push-to-main plus a weekly schedule once the
   runner is registered; `if: github.ref == 'refs/heads/main'`; `runs-on: [self-hosted, Windows, X64, lightbulb-models]`;
   `permissions: contents: read`; no secrets; a first step that fails loudly when `LIGHTBULB_GGUF`, `TINYLLAMA_DIR` and
   `LIGHTBULB_GGUF_CORPUS` do not point at files, printing which. Same-repo PRs only behind the environment with a required
   reviewer (5.1h), which does not exist yet.
4. **Inputs are paths, never copies.** The runner owner places the models on its disk and sets the three paths as
   runner-scoped variables. Nothing is downloaded by CI and no weights enter the repo or an artifact (CireSnave's provenance rule). Each
   model keeps its own licence; the note-of-record for which licence belongs to which file is not written here and was not
   re-verified for this note, so the runner setup step should record it (path, size, sha256, licence, source URL) in the
   runner's own README, not in this repository.
5. **UNVERIFIED list.** Until a GPU runner exists, the two GPU tests are listed here as UNVERIFIED, and the job summary
   prints the list so "nobody ran it" cannot read as "it passes".

## What is blocked on whom

- Runner registration and the registration token: PM (`CIRESNAVE-EXPECTATIONS.md` 5.1h); the aocl row waits on board 467.
- The `lightbulb-models` label, the model files on the runner disk, and the three env variables: runner owner.
- Everything in step 1-2 (helper change, removing `#[ignore]`, counting) can be built now and is testable locally with the
  variables set; step 3 cannot run until the runner exists.
