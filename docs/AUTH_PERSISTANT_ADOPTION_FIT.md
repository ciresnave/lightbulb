# Adoption fit: auth-framework and PersistAnt in lightbulb

Written per board 196, CIRESNAVE-EXPECTATIONS §6.14 (2026-10-17): "All of our
projects should adopt auth-framework for authentication. All of our projects
should adopt PersistAnt for persistence." Docs only -- no dependency added, no
code changed. Answers: what in lightbulb's current auth/persistence maps onto
each shared crate today, what's a real gap, and in what order to move.

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

## 2. auth-framework (v0.5.0-rc24, read at `C:\Projects\auth-framework`)

Release candidate, pre-1.0, `rust-version = "1.88"`. README's own framing:
"should still be validated against your environment, storage, TLS... before
rollout."

The `AuthMethodEnum`/`ApiKeyMethod` registry path (`src/methods/mod.rs:315`)
is a **stub** -- its `authenticate()` arm for `Credential::ApiKey` always
returns a fixed failure string telling the caller to go through
`AuthFramework` directly instead. The real entry point is
`AuthFramework::{create_api_key, validate_api_key, revoke_api_key}`
(`src/auth.rs:1272-1290`), which delegate to `UserManager` (`src/auth_modular
/user_manager.rs:56-140`):

- `create_api_key(user_id, expires_in) -> String`: generates `ak_<32-byte
  token>`, stores a JSON blob (`user_id`, `created_at`, `expires_at`) in the
  storage backend keyed by **the literal plaintext key itself**
  (`format!("api_key:{}", api_key)`, `:73`) -- no hashing anywhere on this
  path (contrast `password_hash` via bcrypt elsewhere in the same file,
  `:391`). A storage-backend leak hands out live, usable keys directly. This
  is a real regression versus lightbulb's current hash-at-rest design, not
  a lateral move.
- `validate_api_key(key) -> UserInfo`: looks the plaintext up, checks
  `expires_at`, and returns a `UserInfo` whose `roles` is **hardcoded to
  `vec!["api_user"]`** (`:114`) -- the role lightbulb assigns at creation
  time (user/admin/llm, enforced by `role_rank`) has no equivalent slot here.
- **No `rotate_api_key` exists anywhere in `user_manager.rs`** (confirmed by
  grep over the file) -- only create+revoke. lightbulb's one-transaction
  atomic rotation has no direct replacement; it would have to be
  hand-built as create-then-revoke (losing the atomicity guarantee) or kept
  as lightbulb's own code calling auth-framework's primitives underneath.
- Postgres storage backend (`src/storage/postgres.rs`) exists
  (`postgres-storage` feature, default-on) and uses `sqlx`, not
  `deadpool_postgres`/`tokio_postgres` -- a different pooling/driver stack
  than lightbulb's current one, so this is a driver swap, not just an
  auth-logic swap.
- Rate limiting: `utils::rate_limit::RateLimiter` (`src/utils/rate_limit.rs`)
  is in-process/synchronous (DashMap-shaped, not DB-backed) -- correct for a
  single instance, loses lightbulb's current property of surviving a
  restart and being consistent across multiple server instances.
  `distributed::rate_limiting::DistributedRateLimiter`
  (`src/distributed/rate_limiting.rs:116`) exists for that case but needs
  **Redis** (`redis-storage` feature, `redis_url` config, `:29`) -- a new
  infra dependency lightbulb does not have today, not a Postgres path.
- RBAC exists (`enhanced-rbac` feature, `role_system` crate,
  `src/authorization_enhanced/`) and is far richer than lightbulb's flat
  three-role `role_rank` check, but it attaches to `UserInfo`/sessions, not
  to the bare bearer-key-to-role mapping lightbulb's admin gate uses today
  -- wiring it in is a design decision, not a drop-in.

### What maps close to 1:1
- Bearer-token-to-identity lookup (`validate_api_key` ↔ `auth_middleware`'s
  hash lookup) -- same shape, modulo the hashing gap above.
- Create/revoke primitives exist on both sides with compatible semantics
  (expiry, one-shot revoke).
- Both can sit behind axum middleware (`api-server`/`axum-integration`
  features exist specifically for this).

### Gaps (concrete, not hypothetical)
1. **No hash-at-rest for API keys** -- `create_api_key` stores the plaintext
   as the lookup key. Adopting as-is is a security *regression* from
   lightbulb's current SHA-256-hash design. Would need a wrapper that hashes
   before calling in, or an upstream fix.
2. **No role assigned at key-creation time** -- `validate_api_key` always
   returns `api_user`; lightbulb's admin-gating and self-escalation check
   (`role_rank`) have nothing to attach to without building a parallel
   role store.
3. **No atomic key rotation** -- `rotate_api_key` does not exist; lightbulb's
   one-transaction old-key-revoke+new-key-issue has no equivalent.
4. **Distributed rate limiting needs Redis**, a dependency lightbulb does
   not currently have; the in-process alternative drops cross-restart and
   cross-instance correctness lightbulb's current DB-backed limiter has.
5. **Different Postgres driver** (`sqlx` vs `deadpool_postgres`/
   `tokio_postgres`) -- adopting the Postgres storage backend means running
   two Postgres client stacks side by side during migration, or porting
   lightbulb's non-auth DB usage too (there is none today, so this is
   contained, but worth naming).
6. Pre-1.0 release-candidate maturity (rc24) -- CireSnave's "all deps at
   latest" standing rule applies, but an RC's API can still move under a
   minor bump in ways a stable 1.x would not.

## 3. PersistAnt (v0.6.2, read at `C:\Projects\PersistAnt`)

Pre-1.0. Its own README states the scope plainly: "a **thin** persistence
layer over Apache OpenDAL, for **blobs and records stored by key**... *
Migrations, locks and expiry are still to come.*"

`Config` (`src/store.rs:68-96`), the only way to open a `Store`, has exactly
four variants: `Memory`, `Fs { root, atomic_write_dir }`, and two test fakes
(`Faulty`, `Masked`) that wrap one of the other two. **There is no Postgres
variant, no networked backend of any kind.** The API is get/put/delete by
key plus typed records with a schema-version field (`Record::SCHEMA`) -- a
key-value/blob store, not a relational one. It has no SQL, no joins, no
transactions spanning multiple keys, no row locking, no foreign keys.

### What maps 1:1
Nothing. lightbulb's persistence needs are relational by construction: four
tables joined by foreign key (`api_keys` ← `audit_logs`, `api_key_usage`,
`auth_attempt_counts` is the exception, keyed by IP not by key), one
multi-statement transaction with a row lock (`rotate_api_key`'s `FOR UPDATE`),
atomic upsert-and-read-back counters (`ON CONFLICT ... RETURNING`), and an
`EXISTS` subquery (the last-admin-key lockout check). None of these are
expressible against a key-by-key blob store without rebuilding a relational
layer on top of it from scratch -- which is exactly the layer PersistAnt's
own README says isn't built yet ("migrations... still to come", and no
transaction primitive is mentioned at all).

### Gaps
1. **No Postgres backend exists.** This is not a missing feature to work
   around; it is the entire premise of lightbulb's persistence (foreign
   keys, transactions, aggregate queries) having no home in PersistAnt's
   current `Config` surface at all.
2. **No transactions.** `rotate_api_key`'s atomicity guarantee (old key
   revoked and new key live, or neither, never both) has no primitive to
   build on.
3. **No migrations.** Lightbulb's six-file, monotonically-numbered schema
   evolution (`refinery`) has no PersistAnt-side equivalent; the README
   lists this as future work, not a gap to engineer around today.

## 4. Adoption order and recommendation

**Do not adopt PersistAnt for lightbulb's current persistence now.** There
is no backend to adopt *into* -- Postgres is not one of PersistAnt's four
`Config` variants, and the relational operations lightbulb's auth/rate-limit
path depends on (transactions, row locks, joins, aggregate queries) are
outside what a key-value/blob store expresses at all, let alone what
PersistAnt has built so far. Revisit if/when PersistAnt grows a Postgres
(or any SQL) backend; until then this is a **blocked**, not a **sized**,
item.

**auth-framework is adoptable, but not as a like-for-like swap** -- three of
its gaps (plaintext-key storage, no per-key role, no atomic rotation) are
things lightbulb's current code does *better* today, so a naive swap is a
regression on all three. The lower-risk path is: keep lightbulb's own
hash-at-rest Postgres schema and role/rotation logic, and adopt
auth-framework only for the pieces it clearly does *more* of than lightbulb
needs to build itself (OAuth2/OIDC/SAML/WebAuthn, if those ever become
in-scope for lightbulb; its RBAC, if lightbulb's three flat roles ever need
to become real policies) -- i.e., treat it as additive capability, not a
rip-and-replace of the bearer-key path. A rip-and-replace right now would
have to reintroduce hashing, per-key roles, and atomic rotation inside
auth-framework first, which is new work in a dependency, not adoption.

**Risk framing, since the PM asked for it explicitly:** a broken persistence
swap tends to fail *loudly* here (a missing table, a failed migration, a
connection error) because every current query already has explicit error
handling (fail-closed on the attempt limiter, fail-open with a logged error
on the rate limiter) -- but PersistAnt isn't in a state to attempt this at
all, so the risk question is moot for now. A broken auth swap is the more
dangerous direction to rush: the two regressions that matter most (plaintext
keys at rest, no per-key role) are both *silent* -- nothing crashes, nothing
logs an error, the system simply becomes less secure in a way that only
shows up in an audit or a breach. That argues for not touching auth either,
until upstream closes those two gaps.

**Effort/risk if the PM wants a sized plan despite the above:**

| Step | Effort | Risk | Why |
|---|---|---|---|
| Wrap auth-framework's `create_api_key` with a hashing layer | Small | Low | One function, mirrors lightbulb's existing `Sha256::new()...finalize()` pattern (already used 3x in `admin.rs`/`auth_middleware.rs`) |
| Add a per-key role field to auth-framework (upstream PR) | Medium | Medium | Touches `UserManager::create_api_key`'s stored JSON shape and `validate_api_key`'s `UserInfo` construction; affects every auth-framework consumer, not just lightbulb |
| Add atomic key rotation to auth-framework (upstream PR) | Medium | Medium | Needs a transaction primitive in whichever storage backend is active; `sqlx`'s Postgres backend has one, the in-memory/other backends may not |
| Swap lightbulb's 48 call sites onto auth-framework's API once the above land | Large | High | Two Postgres driver stacks running at once during transition (`deadpool_postgres` + `sqlx`), 16 raw-SQL sites to retire, and the admin self-lockout/escalation tests (`admin.rs` `#[cfg(test)]`, ~9 tests) all need re-proving against the new path |
| Adopt PersistAnt for lightbulb's persistence | **Blocked** | N/A | No Postgres backend exists in PersistAnt v0.6.2 |

## Open questions for the PM / CireSnave

1. Is auth-framework's plaintext-key-at-rest and no-per-key-role gap
   something to fix upstream (in auth-framework, benefiting every project)
   before any lightbulb adoption, or is lightbulb expected to route around
   it locally?
2. Does `user_sessions` (dead schema, V3) get dropped as part of any future
   persistence work regardless of PersistAnt's trajectory, since it's
   unused today either way?
3. Should lightbulb file the "needs a Postgres/SQL backend" gap against
   PersistAnt now, so that work is visible on its own roadmap rather than
   discovered again the next time this ruling is revisited?
