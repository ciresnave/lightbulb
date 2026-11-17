<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
# Configuration reference (environment variables)

`lightbulb` has no config file and no `--config` flag; the only CLI flag is `--no-auth`. Everything is an
environment variable, read once at startup in `src/main.rs` unless a row says otherwise.
**Verified against `origin/main` at `f51b200` (2026-11-17):** every `LIGHTBULB_*` name that appears under `src/`
(35 names) is in one of the tables below. `DATABASE_URL` and `SECURITY_ALERT_EMAIL` are the two non-prefixed names.

A numeric variable that fails to parse silently falls back to its default, **except** where the row says
"refuses". `LIGHTBULB_MAX_MEMORY_BYTES` logs a warning and falls back; `LIGHTBULB_AUDIT_RETENTION_DAYS` and the
device selectors refuse to start.

## Server

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | unset | PostgreSQL URL for API keys, audit log and sessions. Logged with userinfo and query redacted. |
| `LIGHTBULB_BIND_ADDRESS` | `0.0.0.0:8080` | Listen address. |
| `LIGHTBULB_MODELS_DIR` | unset | Directory scanned for models; also the default for `lightbulb-cli pull`. |
| `LIGHTBULB_DEFAULT_MODEL` | `default` | Model id served when a request names none. |
| `LIGHTBULB_REQUEST_TIMEOUT_MS` | `300000` | Per-request timeout. |
| `LIGHTBULB_CORS_ORIGINS` | unset (any origin) | Comma-separated allowed origins. |
| `LIGHTBULB_LOG_FORMAT` | text | `json` switches the server and `lightbulb-probe` to JSON logs; anything else is text. |
| `LIGHTBULB_API_KEY` | unset | Key `lightbulb-cli` sends when `--api-key` is not given (a client setting, not read by the server). |

## Authentication and limits

| Variable | Default | Meaning |
|---|---|---|
| `LIGHTBULB_RATE_LIMIT_PER_MINUTE` | `60` | Per-API-key request rate limit. |
| `LIGHTBULB_MAX_AUTH_ATTEMPTS_PER_MINUTE_PER_IP` | `300` | Pre-auth flood limiter, every attempt counts. |
| `LIGHTBULB_MAX_AUTH_FAILURES_PER_MINUTE_PER_IP` | `10` | Strict brute-force limiter, failed attempts only. |
| `LIGHTBULB_TRUSTED_PROXIES` | empty | Comma-separated peer IPs allowed to supply `X-Forwarded-For` to the limiters. An entry that is not an IP is dropped, not an error. Empty means the header is never trusted. |
| `LIGHTBULB_JWT_SECRET` | unset | Reserved. Nothing in the crate signs or verifies a JWT with it today. There is deliberately no default. |
| `SECURITY_ALERT_EMAIL` | unset | Reserved. Setting it does not mean alert delivery is implemented. |
| `LIGHTBULB_AUDIT_RETENTION_DAYS` | unset (keep all) | Deletes `audit_logs` rows older than N days, once an hour. Whole number of at least 1; `0` or a non-number **refuses to start**. Needs PostgreSQL. |

## Model serving

| Variable | Default | Meaning |
|---|---|---|
| `LIGHTBULB_MODEL_MAX_BATCH_SIZE` | `16` | Maximum batch size for a model's decode steps. |
| `LIGHTBULB_MODEL_CONTEXT_LENGTH` | `4096` | Context length. |
| `LIGHTBULB_MODEL_MAX_QUEUE_DEPTH` | `32` | Pending-request queue depth; 0 is refused at startup. |
| `LIGHTBULB_PREFIX_SHARING` | off | `1`/`true`/`yes`/`on` enables prefix sharing. Off until a measurement shows it wins (ROADMAP research item 1). |
| `LIGHTBULB_ALLOW_OVERSIZE_MODEL` | off | Same flag syntax. Lets a model load on a GPU whose preflight says it will not fit. CPU never refuses. |
| `LIGHTBULB_DEVICE` | `auto` | `auto`, `cpu`, `cuda:<n>`, `vulkan:<n>`, or a case-insensitive substring of a device name. A failed explicit selector refuses to load. |
| `LIGHTBULB_CUDA_DEVICE_INDEX` | unset | Legacy: `n` means `LIGHTBULB_DEVICE=cuda:n`. Must be a whole number; setting both variables is an error. |

## Memory budget (scheduler and preflight)

| Variable | Default | Meaning |
|---|---|---|
| `LIGHTBULB_MAX_MEMORY_BYTES` | what the serving device reports available, else 8 GiB | Scheduler budget in bytes. A non-number logs a warning and is ignored. The chosen value and its source are logged at startup. |
| `LIGHTBULB_MEMORY_PER_SLOT_BASE` | 100 MiB | Fixed bytes charged per slot. 0 is refused. |
| `LIGHTBULB_MEMORY_PER_TOKEN` | 50 KiB | Bytes charged per token. 0 is refused. |
| `LIGHTBULB_EVICTION_PRESSURE_THRESHOLD` | `0.85` | Fraction of the budget at which eviction starts; must lie in the range the startup error states. |
| `LIGHTBULB_MEMORY_SAFETY_MARGIN` | `0.1` | Fraction of the budget held back; a margin that leaves nothing to allocate is refused. |

## Test-only variables (never read by a running server)

| Variable | Used by |
|---|---|
| `LIGHTBULB_TEST_DATABASE_URL` | API tests that write to a throwaway PostgreSQL. Must be set for them; they panic without it. |
| `LIGHTBULB_GGUF` | A path to one GGUF file for the TinyLlama tests. |
| `LIGHTBULB_GGUF_CORPUS` | Root of a local GGUF corpus for the `#[ignore]`d corpus tests (`src/gguf/mod.rs`). |
| `LIGHTBULB_AWQ_MODEL_DIR` | Test fixture directory for a Qwen3 AWQ snapshot (`src/test_models.rs`). The AWQ serving code was deleted (ROADMAP, PR #195); only the fixture lookup remains. |
| `LIGHTBULB_REQUIRE_MODEL`, `LIGHTBULB_REQUIRE_CORPUS` | Turn a missing model or corpus from a skip into a failure (`test_notice::skip_unless_required`). ON values: `1`/`true`/`yes`/`on`; OFF: `0`/`false`/`no`/`off`/empty; anything else panics. |
| `LIGHTBULB_REQUIRE_SOMETHING`, `LIGHTBULB_REQUIRE_NOTHING_AT_ALL`, `LIGHTBULB_REQUIRE_TEST_NOTICE_SELFTEST`, `LIGHTBULB_REQUIRE_TEST_NOTICE_LOOKUP` | Throwaway names used by `src/test_notice.rs`'s own self-tests; setting them does nothing. |

## Keeping this file true

Re-derive the list with `git grep -h -o -E 'LIGHTBULB_[A-Z0-9_]+' -- src | sort -u` and diff it against the names in
this file. Defaults here are copied from `src/main.rs` and `src/model_fuel/device.rs`; if a default changes there,
change it here in the same PR.
