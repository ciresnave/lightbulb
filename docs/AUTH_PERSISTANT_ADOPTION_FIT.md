# Adoption fit: auth-framework and PersistAnt in lightbulb

Written per board 196, CIRESNAVE-EXPECTATIONS §6.14 (2026-10-17): "All of our
projects should adopt auth-framework for authentication. All of our projects
should adopt PersistAnt for persistence." Docs only -- no dependency added, no
code changed. Answers: what in lightbulb's current auth/persistence maps onto
each shared crate today, what's a real gap, and in what order to move.

**Correction (same day):** the first version of this doc read both shared
crates from their local working-tree checkouts under `C:\Projects`, which this
portfolio's own `CLAUDE.md` §2 forbids ("Read `git show origin/main:<path>`,
never a working tree -- checkouts are shared between lanes") precisely because
those checkouts were stale. At `origin/main` today, PersistAnt is **0.7.4**
(not 0.6.2) and auth-framework is **0.6.0-rc71** (not 0.5.0-rc24). Both
sections below are rewritten from `git show origin/main:<path>` reads at those
versions; the verdict changes for PersistAnt and partially for auth-framework.
Section 1 (lightbulb's own code) was already correct and is unchanged.

## 1. What lightbulb has today

### 1a. Auth

All in `src/api/auth_middleware.rs` and `src/api/admin.rs`, four `axum`
middleware layered on the router (`src/api/mod.rs`):

- **`auth_middleware`** (`auth_middleware.rs:155`): pulls a `Bearer` token
  from `Authorization`, SHA-256-hashes it, and looks the hash up in Postgres
  (`api_keys.key_hash`, `WHERE ... expires_at > NOW() AND revoked_at IS
  NULL`). On a hit it inserts `ApiKeyInfo { api_key_id, role }` into request
  extensions for downstream handlers. **The plaintext key is never stored**
  -- only its hash, so a database leak does not hand out usable keys.
- **`pre_auth_attempt_limiter_middleware`** (`:53`): runs *before* the hash
  lookup, counting attempts per client IP in Postgres
  (`auth_attempt_counts`) and refusing at `TOO_MANY_REQUESTS` past a
  threshold -- a brute-force guesser never presents a valid key, so limiting
  only after a failed lookup still pays the hash+query cost per guess. Fails
  closed on a DB error.
- **`rate_limit_middleware`** (`:280`): per-API-key, fixed-window-by-minute
  counter in Postgres (`api_key_usage`), atomic `INSERT ... ON CONFLICT DO
  UPDATE ... RETURNING request_count`. Fails *open* on a DB error (a
  deliberate asymmetry from the attempt limiter above: this sits behind
  auth, so a DB hiccup here means "don't throttle," not "let an unauthenticated
  attacker through").
- **`admin_check_middleware`** (`:422`): requires `ApiKeyInfo.role ==
  "admin"`, applied only to the `/admin/*` sub-router (`admin.rs:44`).
- **`audit_middleware`** (`:349`): best-effort async log of every request
  (method, path, status, latency, api_key_id) to `audit_logs`, spawned so it
  never blocks the response.

Admin key lifecycle, `admin.rs`: `create_api_key` (role ∈
{user,admin,llm}, `role_rank` check so a caller can never mint a key
ranked above its own role -- checked twice, once by middleware, once
independently in the handler), `revoke_api_key` (idempotent, refuses to
revoke the caller's own key if it's the last live admin key), `rotate_api_key`
(one Postgres transaction: `SELECT ... FOR UPDATE` the old row, insert the
replacement, revoke the old row, commit -- so a crash mid-rotation can never
leave both keys live or neither live).

There is a `user_sessions` table (`migrations/V3__create_sessions.sql`:
session_token, ip_address, user_agent, expiry) with **zero code
references** (`grep -rn 'user_sessions\|session_token' src/` : no hits).
Dead schema -- no session-based auth is actually implemented, only
stateless bearer API keys.

~48 non-test call sites reference `ApiKeyInfo`/the four middleware
functions; 16 raw SQL call sites across the auth/admin path.

### 1b. Persistence

`deadpool_postgres::Pool` + `tokio_postgres::NoTls` (`src/api/mod.rs:433`),
migrations applied at startup via `refinery::embed_migrations!("./migrations")`
(`:438-441`). Six migration files, all additive except V6 (`ALTER TABLE
api_keys ADD COLUMN revoked_at`): `api_keys`, `audit_logs`, `user_sessions`
(dead, see above), `api_key_usage`, `auth_attempt_counts`, plus the
revocation column. Every table but the dead one is read or written from the
auth/admin path above -- there is no other Postgres usage in the binary
(`db_pool` appears in exactly the five files listed above, two of which are
test-fixture `None` defaults). Features used: `FOR UPDATE` row locks,
`ON CONFLICT ... DO UPDATE ... RETURNING`, `EXISTS` subqueries, foreign keys
with `ON DELETE {SET NULL,CASCADE}`, explicit transactions.

## 2. auth-framework (v0.6.0-rc71, read via `git show origin/main:<path>`)

Release candidate, pre-1.0. The real entry points are
`AuthFramework::{create_api_key, validate_api_key, revoke_api_key}`
(`src/auth.rs:1523-1541`), thin pass-throughs with no added logic, which
delegate to `UserManager` (`src/auth_modular/user_manager.rs`):

- **`create_api_key`** (`user_manager.rs:291-312`): generates `ak_<32-byte
  token>`, stores a JSON blob (`user_id`, `created_at`, `expires_at`) under
  `Self::api_key_storage_key(&api_key)`.
- **Hash-at-rest is now implemented** (`user_manager.rs:203-208`):
  `api_key_storage_key` names the row `"api_key_h:{digest}"` where `digest =
  crate::storage::key_names::digest("api_key", api_key)` -- "a
  domain-separated one-way digest, never the raw key" (the function's own
  doc comment). This closes the gap the stale (rc24) read of this doc
  reported. A paired `legacy_api_key_storage_key` (`:212`) and
  `get_api_key_row` (`:227-265`) migrate any key still stored under the old
  plaintext-named row on first read, with a documented concurrency argument
  against a revoke racing the migration. **Gap 1 (plaintext keys at rest):
  CLOSED.**
- **`validate_api_key`** (`:326-356`) still returns a `UserInfo` whose
  `roles` is **hardcoded to `vec!["api_user".to_string()]`**
  (`user_manager.rs:351`), regardless of what role, if any, the key's owner
  actually has. lightbulb's admin-gating and self-escalation check
  (`role_rank`) have nothing to attach to without building a parallel role
  store. **Gap 2 (no per-key role): still OPEN**, same line number
  (`:351`) as before -- unchanged between rc24 and rc71.
- **No `rotate_api_key` exists anywhere in the repository**
  (`git grep -n 'rotate_api_key\|fn rotate' origin/main -- '*.rs'` returns
  only unrelated hits: `authentication/advanced_auth.rs`'s
  `rotate_codes`, `protocols/gnap.rs`'s OAuth `rotate_token`,
  `security/secure_session.rs`'s `rotate_session`, and HMAC/RSA signing-key
  rotation in `tokens/mod.rs` -- none of these touch API keys).
  lightbulb's one-transaction atomic rotation (revoke old + insert new,
  commit or neither) has no equivalent. **Gap 3 (no atomic key rotation):
  still OPEN.**
- A Postgres storage backend exists (`storage-postgres` feature family) and
  uses `sqlx`, not `deadpool_postgres`/`tokio_postgres` -- a different
  pooling/driver stack than lightbulb's current one, so adopting it is a
  driver swap on top of the auth-logic swap, not just the latter.

### What maps close to 1:1
- Bearer-token-to-identity lookup (`validate_api_key` ↔ `auth_middleware`'s
  hash lookup) -- same shape, and now the storage side matches too
  (both hash before persisting).
- Create/revoke primitives exist on both sides with compatible semantics
  (expiry, one-shot revoke).
- Both can sit behind axum middleware (auth-framework ships
  `api-server`/`axum-integration` features for this).

### Gaps (concrete, not hypothetical; re-verified against origin/main rc71)
1. ~~No hash-at-rest for API keys~~ -- **CLOSED**, see above
   (`user_manager.rs:203-208`).
2. **No role assigned at key-creation time** -- `validate_api_key` still
   always returns `api_user` (`user_manager.rs:351`); `create_api_key`
   (`:291-312`) takes no role parameter either, so there is nowhere to put
   one even on the write side.
3. **No atomic key rotation** -- confirmed absent repo-wide, not just in
   `user_manager.rs`.
4. **Different Postgres driver** (`sqlx` vs `deadpool_postgres`/
   `tokio_postgres`) if the Postgres storage backend is adopted alongside
   the auth-logic swap.
5. Pre-1.0 release-candidate maturity (rc71) -- CireSnave's "all deps at
   latest" standing rule applies, but an RC's API can still move under a
   minor bump in ways a stable 1.x would not.

**Net change from the stale read:** one of three original regressions is
fixed upstream already (hashing); two remain (role, rotation). The adoption
blocker is now narrower than first reported, not gone.

## 3. PersistAnt (v0.7.4, read via `git show origin/main:<path>`)

Pre-1.0, but materially different from the v0.6.2 state the first version of
this doc described: PersistAnt is now a **workspace** with a separate
relational driver crate, `persistant-sqlx`
(`crates/persistant-sqlx/Cargo.toml`), implementing
`persistant_relational::RelationalBackend` for PostgreSQL (plus optional
MySQL/SQLite). **The original "no Postgres backend, key-value only" verdict
no longer holds.**

`PgBackend` (`crates/persistant-sqlx/src/lib.rs:140-227`) wraps a real
`sqlx::PgPool` and exposes `execute`/`query`/`begin(isolation)`; `begin`
returns a `PgTx` implementing `RelationalTx` with `execute`/`query`/
`commit`/`rollback` (`:210-250`). The trait doc is explicit that this is a
raw-SQL passthrough, not a query-builder DSL: "Statements are written in the
backend's own Dialect... Nothing here translates SQL."
(`crates/persistant-relational/src/backend.rs:21-22`). That means `SELECT ...
FOR UPDATE` and `INSERT ... ON CONFLICT ... RETURNING` -- the exact two
constructs lightbulb's `rotate_api_key` and rate-limit counter depend on --
are plain strings passed straight to `execute`/`query`, no translation layer
to fight.

The relational contract (`crates/persistant-relational/src/conformance.rs`)
defines 11 leaves (`Leaf` enum, `:48-86`): `Values`, `Transaction`,
`UniqueClassified`, `ForeignKey`, `CheckConstraint`,
`IsolationNoDirtyRead`, `IsolationRepeatableRead`, `IsolationSerializable`,
`Savepoint`, `RowLock` (doc comment: "a row locked `FOR UPDATE` by an open
transaction cannot be locked by another connection"), `UpsertWhere` (`INSERT
.. ON CONFLICT (..) DO UPDATE .. WHERE ..`). The checked-in conformance
matrix for this backend, `matrix/sqlx-postgres.tsv` (against a real
PostgreSQL 16.15), shows **every one of these `enabled=yes`**, plus a
separate `rel.migrate.*` family (`apply`, `checksum`, `newer`, `atomic`,
`ddl_atomic`, `concurrent`) also all `enabled=yes` -- checksummed, ordered,
resumable migrations with corruption/staleness detection
(`crates/persistant-relational/src/migrate.rs:1-60`: `ChecksumMismatch`,
`NewerThanProgram`, `Dirty`, `Failed` error variants), comparable in rigor to
lightbulb's current `refinery`-based six-file migration set.

Error classification (`persistant-sqlx/src/lib.rs:29-50`) maps Postgres
SQLSTATEs to typed `RelError` variants lightbulb's queries already produce
conditions for: `23505`→`UniqueViolation`, `23503`→`ForeignKeyViolation`,
`23514`/`23502`→`CheckViolation`, `40001`→`Serialization`,
`40P01`→`Deadlock`.

### What maps close to 1:1
- `begin(Isolation::Serializable) -> PgTx { execute, query, commit }` ↔
  lightbulb's `rotate_api_key` transaction (`SELECT ... FOR UPDATE`, insert
  replacement, revoke old, commit) -- same shape, raw SQL both sides.
- `RowLock` leaf, certified `enabled=yes` against Postgres 16.15 ↔
  lightbulb's `FOR UPDATE` usage in the same function.
- `UpsertWhere` leaf (`ON CONFLICT ... DO UPDATE ... WHERE`), certified
  `enabled=yes` ↔ lightbulb's rate-limit counter's `ON CONFLICT DO UPDATE
  ... RETURNING` (the `RETURNING` clause is a plain query result row via
  `query()`, not a separate primitive).
- `ForeignKey`/`CheckConstraint` leaves, certified `enabled=yes` ↔
  lightbulb's `api_keys`-referencing foreign keys and column constraints.
- Checksummed migrations with a dirty-state guard ↔ `refinery`'s embedded
  migration set, comparable semantics (lightbulb would need to port six SQL
  files' worth of DDL into `Migration::new(version, description, sql)`
  calls, not redesign the schema).

### Gaps (re-verified against origin/main 0.7.4; the v0.6.2 gap list no
longer applies)
1. **Separate crate, separate pool/driver.** `persistant-sqlx` wraps
   `sqlx::PgPool`, not `deadpool_postgres`/`tokio_postgres` -- adopting it
   means running two Postgres client stacks during migration, same shape as
   the auth-framework driver-swap gap above (and the same non-issue today,
   since lightbulb's only Postgres usage is already the auth/admin path
   this would replace).
2. **No evidence (in the files read) of a connection-pool-level feature
   lightbulb currently relies on beyond pooling itself** -- not confirmed
   either way for this doc; worth a short spike (`PgBackend::connect`/
   `from_pool`, `lib.rs:146-165`) before committing to a full swap, but not
   a blocker found so far.
3. Pre-1.0 maturity -- same "all deps at latest, RC API can move" caveat as
   auth-framework.

**Net change from the stale read:** the "blocked, not sized" verdict is
**wrong at current versions**. PersistAnt 0.7.4's relational driver covers
every primitive lightbulb's persistence layer uses (transactions,
serializable isolation, row locks, FK/check constraints, upsert-where,
checksummed migrations), certified against a real PostgreSQL 16.15. This is
now a **sized adoption**, not a blocked one.

## 4. Adoption order and recommendation

**Adopt PersistAnt first.** At 0.7.4 it has no remaining functional gap
against lightbulb's persistence needs: every relational primitive lightbulb
uses (transactions, serializable isolation, row locks, FK/check constraints,
upsert-where, checksummed migrations) is certified `enabled=yes` against a
real PostgreSQL 16.15. The only real cost is the mechanical one of running a
second Postgres driver stack (`sqlx` via `persistant-sqlx`) during the
migration off `deadpool_postgres`/`tokio_postgres` -- contained, since
lightbulb's only Postgres usage today is the auth/admin path this would
replace, not spread across the codebase.

**auth-framework is adoptable for the identity-lookup shape, but not yet a
like-for-like swap for the role/rotation pieces.** One of the three original
regressions is already fixed upstream (hash-at-rest). The other two
(hardcoded `api_user` role, no atomic rotation) are things lightbulb's
current code still does better. The lower-risk path: adopt
auth-framework's `create_api_key`/`validate_api_key`/`revoke_api_key` for
the bearer-token-to-identity lookup itself, but keep lightbulb's own
role/rotation logic layered on top (role stored and checked in lightbulb's
own schema, rotation still lightbulb's one-transaction revoke+issue) until
auth-framework grows a per-key role and a rotation primitive upstream.

**Why PersistAnt before auth-framework, not the other way round:** adopting
PersistAnt touches *only* the storage layer underneath logic that doesn't
change (the SQL lightbulb already runs, largely unchanged, now issued
through `PgBackend` instead of `deadpool_postgres`) -- a contained,
mechanical swap with a certified contract to verify against before and
after. Adopting auth-framework's identity logic changes *behavior*
(what's hashed, what's returned, what's enforced), which is riskier to
land correctly and depends on the still-open role/rotation gaps either
way. Do the lower-risk, higher-certainty swap first.

**Risk framing, since the PM asked for it explicitly:** a broken PersistAnt
swap tends to fail *loudly* -- a missing migration, a failed transaction, a
connection error, all of which lightbulb's existing error handling already
expects in some form (fail-closed on the attempt limiter, fail-open with a
logged error on the rate limiter) and the conformance matrix gives a concrete
thing to re-verify against (run lightbulb's own queries against `PgBackend`
and confirm the same `RelError` classification lands for the same failure
modes). A broken auth-framework identity swap is more dangerous to rush
precisely because the two gaps that still exist (no per-key role, no
rotation) are *silent* if routed around carelessly -- nothing crashes,
nothing logs an error, a key that should have been scoped to `user` behaves
as whatever role lightbulb's wrapper code assumes. Keeping role/rotation
logic on lightbulb's own side (not inside auth-framework) is what keeps that
risk from landing in the first place.

**Effort/risk:**

| Step | Effort | Risk | Why |
|---|---|---|---|
| Port lightbulb's 6 migration files to `persistant-relational::Migration` calls | Medium | Low | Mechanical translation of existing, working DDL; `rel.migrate.checksum`/`.atomic` are certified, so a ported migration either applies cleanly once or fails loudly before any app code runs |
| Swap `deadpool_postgres`/`tokio_postgres` call sites onto `PgBackend`/`PgTx` | Large | Medium | ~16 raw-SQL sites across `auth_middleware.rs`/`admin.rs`; mechanical per-site (same SQL text, different `execute`/`query` call shape) but all must move together since only one pool can be live against the schema at a time |
| Re-prove the admin self-lockout/escalation tests against `PgBackend` | Medium | Low | ~9 `#[cfg(test)]` tests in `admin.rs`; the conformance matrix already certifies the primitives they exercise (`RowLock`, `Transaction`), so this is re-running known-good tests on a new driver, not new test design |
| Wrap auth-framework's `create_api_key`/`validate_api_key`/`revoke_api_key` for the lookup path, keeping lightbulb's role/rotation logic on top | Small | Low | Hashing gap is closed upstream; the wrapper only needs to carry lightbulb's existing role/rotation code, not invent new logic |
| Add a per-key role field to auth-framework (upstream PR, benefits every project adopting it) | Medium | Medium | Touches `UserManager::create_api_key`'s stored JSON shape and `validate_api_key`'s `UserInfo` construction; affects every auth-framework consumer, not just lightbulb |
| Add atomic key rotation to auth-framework (upstream PR, benefits every project adopting it) | Medium | Medium | Needs a transaction primitive in whichever storage backend is active; the Postgres backend has one via `persistant-sqlx`/`sqlx` if auth-framework adopts PersistAnt too, other backends may not |

## Open questions for the PM / CireSnave

1. Should the PersistAnt migration-file port and the Postgres-driver swap
   land as one PR or be staged (migrations ported and dual-run first,
   driver swap second)?
2. Does `user_sessions` (dead schema, V3) get dropped as part of the
   PersistAnt migration port, since it's unused today regardless?
3. Is auth-framework's no-per-key-role and no-rotation gap something to fix
   upstream (benefiting every project) before lightbulb wraps around it, or
   is a lightbulb-side wrapper (keeping role/rotation logic local)
   acceptable as a permanent shape rather than a stopgap?
