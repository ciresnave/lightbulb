# PersistAnt adoption design: lightbulb's admin/auth Postgres path

Per the PM's instruction (2026-10-17, following docs/AUTH_PERSISTANT_ADOPTION_FIT.md's
verdict that PersistAnt 0.7.4 is a sized, adoptable swap): "storage-layer swap of
admin/auth path onto `persistant-sqlx::PgBackend`, contract-certified: docs PR with
options + order + a rollback; no code until I answer." This is that doc. **No source
code, Cargo.toml, or migrations/ file is touched by this PR.**

## 1. Scope

**Moves**: the entire Postgres surface, which is exactly two files plus the pool setup:

- `src/api/auth_middleware.rs` -- 4 raw-SQL call sites (`grep -c '\.query\|\.execute' `:
  4)
- `src/api/admin.rs` -- 6 raw-SQL call sites (same grep: 6), including the one
  transaction (`rotate_api_key`, `admin.rs:655-735`)
- `src/api/mod.rs:425-443` -- pool construction (`deadpool_postgres::Manager` +
  `tokio_postgres::NoTls`) and migration execution (`refinery::embed_migrations!` +
  `migrations::runner().run_async(...)`)
- `migrations/*.sql` -- 6 files: `V1__create_api_keys.sql`, `V2__create_audit_logs.sql`,
  `V3__create_sessions.sql` (dead, see
  docs/AUTH_PERSISTANT_ADOPTION_FIT.md §1a), `V4__create_rate_limits.sql`,
  `V5__create_auth_attempt_counts.sql`, `V6__add_api_key_revocation.sql`

**Does not move**: re-checked today (`grep -rln 'db_pool\|deadpool_postgres\|tokio_postgres' src/`)
-- the only other hits are `src/api/openai/chat.rs:1979` and
`src/api/openai/completions.rs:315`, both `db_pool: None` test-fixture defaults with no
query against them. This confirms docs/AUTH_PERSISTANT_ADOPTION_FIT.md's scope claim
still holds: nothing outside the auth/admin path touches Postgres.

## 2. Target API shape

Read from `origin/main` of `C:\Projects\PersistAnt`, `crates/persistant-sqlx/src/lib.rs:137-251`
(never the local working tree -- see the correction notice at the top of
docs/AUTH_PERSISTANT_ADOPTION_FIT.md for why that matters here specifically).

```rust
pub struct PgBackend { /* wraps sqlx::PgPool */ }
impl PgBackend {
    pub async fn connect(url: &str) -> Result<Self, RelError>;
    pub async fn from_pool(pool: sqlx::PgPool) -> Result<Self, RelError>;
    pub fn pool(&self) -> &sqlx::PgPool;
}
impl RelationalBackend for PgBackend {
    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64, RelError>;
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>, RelError>;
    async fn begin(&self, isolation: Isolation) -> Result<PgTx<'_>, RelError>;
}
pub struct PgTx<'a> { /* wraps sqlx::Transaction<'a, Postgres> */ }
impl RelationalTx for PgTx<'_> {
    async fn execute(&mut self, sql: &str, params: &[Value]) -> Result<u64, RelError>;
    async fn query(&mut self, sql: &str, params: &[Value]) -> Result<Vec<Row>, RelError>;
    async fn commit(self) -> Result<(), RelError>;
    async fn rollback(self) -> Result<(), RelError>;
}
```

Lightbulb's current shape, read from this repo (`src/api/admin.rs:664-735`,
`src/api/mod.rs:425-443`):

| lightbulb today | PersistAnt equivalent | Mapping note |
|---|---|---|
| `pool.get().await?` (`deadpool_postgres::Client`) | `PgBackend::from_pool(pool)` once at startup, then `&PgBackend` passed around | One-time construction change, not per-call |
| `client.query_opt(sql, &[&id])` | `backend.query(sql, &[Value::from(id)])` then take `.first()` | **No `query_opt`/`query_one` convenience exists.** Every single-row lookup (`admin.rs:567`'s lockout check, `auth_middleware.rs:209`'s hash lookup, and the `FOR UPDATE` lookup inside `rotate_api_key` at `:678`) becomes `query(...)` returning `Vec<Row>`, with the "0 or 1 row" check moved into lightbulb's own code instead of being enforced by the driver. This is the single biggest per-call-site shape change, not a 1:1 swap. **Correction (Step 2 PR):** this row originally also named `admin.rs:420` and `:709` as single-row lookups. Both are `INSERT ... RETURNING` (writes, not reads) -- `:420` is `create_api_key`'s insert (Step 3), `:709` is `rotate_api_key`'s replacement-key insert, inside the one transaction (Step 4). Verified by function boundary, not just SQL verb. |
| `client.execute(sql, &[&a, &b])` | `backend.execute(sql, &[Value::from(a), Value::from(b)])` | `Value` (`crates/persistant-relational/src/value.rs:36-45`) is `Null/Bool/Int(i64)/Float(f64)/Text(String)/Bytes(Vec<u8>)` -- deliberately narrow, no `Uuid` or timestamp variant. Every `uuid::Uuid` param lightbulb currently binds directly (`old_id` in `admin.rs:678`, `:709`'s `RETURNING id` read-back) needs to go through `Value::Text(id.to_string())` and be parsed back out; this is a real conversion lightbulb's code has to write, not a gap in PersistAnt (the doc comment says so explicitly: "a program with richer types uses the driver's own handle" -- i.e. `PgBackend::pool()`'s raw `sqlx::PgPool` is the escape hatch if this conversion is unwanted everywhere) |
| `client.transaction().await?` → `txn.query_opt/execute(...)` → `txn.commit().await?` | `backend.begin(Isolation::ReadCommitted).await?` → `tx.query/execute(...)` → `tx.commit().await?` | Same shape (begin/use/commit), and PersistAnt's `Isolation` enum lets `rotate_api_key`'s `FOR UPDATE` row-lock transaction request `Isolation::ReadCommitted` explicitly (lightbulb's current code gets Postgres's default isolation implicitly) -- a documented choice, not a behavior change, if `ReadCommitted` is picked to match current behavior exactly |
| `SELECT ... FOR UPDATE` as literal SQL text | Same literal SQL text, inside `tx.query(...)` | PersistAnt's `RowLock` conformance leaf (certified `enabled=yes`, `matrix/sqlx-postgres.tsv`) means this exact clause is expected to keep working, not translated |
| `migrations::runner().run_async(&mut **client)` (refinery) | `backend.migrate(&migrations).await?` (`Migrates` trait) | See §3 |

## 3. Migration path

Read from `origin/main` of `C:\Projects\PersistAnt`,
`crates/persistant-relational/src/migrate.rs:1-118`.

```rust
pub struct Migration { pub version: i64, pub description: String, pub sql: String }
impl Migration {
    pub fn new(version: i64, description: impl Into<String>, sql: impl Into<String>) -> Self;
}
pub trait Migrates: RelationalBackend {
    async fn migrate(&self, migrations: &[Migration]) -> Result<MigrateReport, MigrateError>;
    async fn applied_versions(&self) -> Result<Vec<i64>, MigrateError>;
}
```

Lightbulb's 6 `V{n}__description.sql` files map directly: each file's `{n}` becomes
`Migration::new(n, "description", sql_text)`'s `version`, the filename's description
segment becomes `description`, and the file's contents becomes `sql` verbatim (no
translation -- PersistAnt's own doc comment on the relational trait is explicit that
SQL is a passthrough, not a DSL; see docs/AUTH_PERSISTANT_ADOPTION_FIT.md §3). The
ledger (`_sqlx_migrations`, per `migrate.rs`'s module doc comment) replaces refinery's
own tracking table; **these are two different ledger tables, so this is a genuine
cutover, not an additive change** -- a fresh install picks one or the other, an
existing install needs the ledger itself migrated (see §6, Stage 1 rollback).

Both are forward-only: refinery has no down-migration files in this repo today (no
`U{n}__*.sql` exists, confirmed by `ls migrations/`), and PersistAnt's `MigrateError`
variants (`ChecksumMismatch`, `NewerThanProgram`, `Dirty`, `Failed`) describe refusal
conditions, not a rollback mechanism -- neither side undoes a migration automatically.
Rollback for a bad migration is addressed directly in §6, not assumed away here.

## 4. Staging options

**Option A -- Big-bang, feature-flagged.** Port all 10 call sites and all 6 migrations
in one PR behind a build-time choice (old pool vs `PgBackend`, selected by a config
value or Cargo feature), run both code paths' full test suites against each, then flip
the default and remove the old path in a follow-up once proven in whatever environment
CireSnave runs this in.

- *What breaks if wrong*: everything in the auth/admin path at once -- a subtle `Value`
  conversion bug or a missed `FOR UPDATE`/isolation mismatch surfaces across all 10
  call sites simultaneously, not incrementally.
- *How it's caught*: the existing ~9 `#[cfg(test)]` admin self-lockout/escalation tests
  (`admin.rs`) plus `postgres_integration` CI job, both re-run against the new path
  before the flag flips -- but these tests exercise the HAPPY paths and the specific
  lockout/escalation EDGE cases already known; a behavior difference in an UNTESTED
  corner (e.g. isolation-level-dependent read visibility under concurrent rotation)
  would not be caught by re-running the same tests, only by writing new ones.
- *Risk window*: short in wall-clock terms (one PR, one flip), but all-or-nothing --
  there is no intermediate state where only some call sites have been proven.

**Option B -- Incremental, by call-site family, same pool-era throughout.** Port one
function at a time (e.g. `create_api_key` first, `revoke_api_key` second,
`rotate_api_key` last since it is the one transaction and the riskiest single call
site), each as its own small PR, with both the old and new code *compiling* but only
one being *called* at any given time (no dual-write, no mirrored reads -- this is
narrower than true dual-running, see below). Each PR is independently revertable.

- *What breaks if wrong*: one function at a time; a bug in `create_api_key`'s port
  doesn't touch `revoke_api_key`, which is still on the old path until its own PR
  lands.
- *How it's caught*: the same tests as Option A, but run after each small PR instead of
  once at the end -- a regression is attributable to one specific PR, not "something in
  the big PR."
- *Risk window*: longer in wall-clock terms (N small PRs instead of one), but each
  individual exposure is small and independently revertable.

**Dual-running (mirror-write to both backends) was considered and is NOT recommended as
a third option here.** It is the right tool for a path with live, irreplaceable
production data that cannot tolerate a bad cutover (e.g. a user-facing payments
ledger), but lightbulb's entire persisted state is: API keys and their roles (easily
re-issued -- `rotate_api_key` already exists specifically to replace a key), rate-limit
counters (naturally self-healing within a fixed window), and audit logs (append-only,
not read back by any code path per docs/AUTH_PERSISTANT_ADOPTION_FIT.md §1a -- a lost
log entry during cutover is not a dangling reference anywhere). Building a mirror-write
layer to protect data this replaceable is added complexity (write to two pools, compare,
reconcile divergence) with no corresponding reduction in actual risk. This is flagged
as a judgment call in §7, not asserted unilaterally, since it depends on how this
instance is actually run (see Open Questions).

**Recommendation: Option B.** The call sites are few (10) and already naturally
separable by function; the per-PR review and revert granularity is worth more here than
the lower total PR count Option A offers, especially for `rotate_api_key`'s one
transaction, which deserves to be isolated as its own reviewed change rather than
bundled with 9 simpler call sites.

## 5. Recommended order of concrete steps

1. **Migrations only**, independent of any app code: port the 6 `.sql` files' text
   into `Migration::new(...)` calls, stand up `PgBackend` in a startup path that runs
   `backend.migrate(&migrations)` *in addition to* (not instead of) the existing
   refinery run, against a scratch/dev database, and diff the resulting schema against
   what refinery produces. This step touches zero query call sites and is fully
   reversible by deleting the added code -- no production migration ledger is touched
   yet.
2. **Read-only call sites first**: exactly 2, not 4 as an earlier version of this
   step claimed -- `admin.rs:567`'s `revoke_api_key` self-lockout check (`SELECT
   EXISTS (...)`) and `auth_middleware.rs:209`'s bearer-hash lookup (`SELECT id,
   role FROM api_keys WHERE key_hash = $1 ...`). **Correction:** `admin.rs:420` is
   `create_api_key`'s `INSERT ... RETURNING` (a write, Step 3); `admin.rs:678` and
   `:709` are both inside `rotate_api_key`'s one transaction (`SELECT ... FOR
   UPDATE` and the replacement key's `INSERT ... RETURNING`, both Step 4) -- found
   by checking each call site's actual SQL verb and enclosing function, not just
   trusting the line numbers this doc originally gave. Lowest risk of the true
   read sites: no transaction, no write. Confirms the `Value`-conversion and
   `Vec<Row>`→typed-field decoding pattern once, before repeating it on writes.
   Also surfaced a gap not in §2's table: `persistant-sqlx`'s row decoder has no
   case for Postgres's `UUID` column type at all (only `BOOL`/`INT*`/`FLOAT*`/
   `TEXT`/`VARCHAR`/`NAME`/`BPCHAR`/`BYTEA`) -- selecting a raw `uuid` column
   errors with `RelError::Unsupported`. Both ported queries cast the column to
   `::text` in the SQL itself to work around it; see `auth_middleware.rs`'s
   `lookup_api_key_by_hash`.
3. **Single-statement writes next**: exactly 5, not 3 as an earlier version of
   this step claimed -- `create_api_key`'s insert (`admin.rs`), `revoke_api_key`'s
   update (`admin.rs`), BOTH rate-limit-shaped upsert counters (the doc
   originally named only one): `pre_auth_attempt_limiter_middleware`'s
   `auth_attempt_counts` upsert and `rate_limit_middleware`'s `api_key_usage`
   upsert (both `auth_middleware.rs`, both `ON CONFLICT ... DO UPDATE ...
   RETURNING`, certified via PersistAnt's `UpsertWhere` leaf per
   docs/AUTH_PERSISTANT_ADOPTION_FIT.md §3), and `audit_middleware`'s plain
   `audit_logs` insert (`auth_middleware.rs`, no `RETURNING`, best-effort/
   fire-and-forget via `tokio::spawn` -- omitted entirely from this step's
   original list, not just miscounted). **Correction:** found by checking
   every remaining `deadpool_postgres` call site's function and SQL verb
   after step 2 landed, the same way step 2's own miscount was caught.
   `Value` has no INET variant either (the same gap §2's table documents for
   UUID) -- `client_ip`/`api_key_id` params bind through `Value::Text` with
   an explicit `::inet`/`::uuid` cast on the bind PARAMETER (not a column,
   since these are inserts, not lookups against a stored value).
4. **`rotate_api_key`'s transaction last**: the one multi-statement, row-locked,
   atomic operation. By this point the `query`/`execute` pattern is proven on both
   reads and single writes; this step is purely about the `begin`/`commit` shape and
   the isolation-level choice (see §2's table note), isolated so a problem here is
   attributable to the transaction wrapper, not to the underlying query pattern.
5. **Swap the pool itself** (`src/api/mod.rs:425-443`): remove
   `deadpool_postgres`/`tokio_postgres`/`refinery`, make `PgBackend` the only pool,
   switch the startup migration call to `backend.migrate(...)` as the sole path (no
   longer "in addition to," now instead of).
6. ~~Drop `user_sessions` (dead schema, `V3`) as part of this work~~ -- **Corrected
   (Step 5 PR):** the PM's answer to Open Question 2 (below) overrides this item as
   originally written. `user_sessions` carries forward unchanged in this step; dropping
   the dead table is deferred to its own future PR, since it's an irreversible schema
   change that deserves its own decision rather than riding along with the pool swap.

## 6. Rollback plan

- **Step 1 (migrations, dual-run in a scratch DB)**: delete the added code. Nothing in
  production was touched; there is no ledger entry to undo because this step never ran
  against a shared instance.
- **Steps 2-4 (call-site ports, one PR each)**: `git revert` the PR. No schema or data
  changed in these steps -- only which Rust code issues the same SQL text against the
  same tables, so a revert is a pure code rollback with no database-side action needed.
- **Step 5 (pool swap, the only step that touches the live ledger)**: this is the one
  stage "revert the PR" is not sufficient for, because the migration ledger itself
  changes (refinery's tracking table vs PersistAnt's `_sqlx_migrations`, per §3). Before
  this step runs against any shared instance: back up the migration-tracking table
  (whichever one is about to become inactive) and confirm the full migration history
  replays cleanly against a scratch copy of production's current schema, in both
  directions (refinery-tracked fresh install, then PersistAnt-tracked fresh install),
  before pointing this step at the real instance. If step 5 needs to be undone after
  going live: restore the backed-up ledger table and revert the code in the same
  deploy, so the running binary and the ledger it reads agree again -- reverting only
  the code while the ledger has already moved to PersistAnt's table would leave
  refinery's `embed_migrations!` looking at a ledger it doesn't recognize.
- **General principle used throughout**: every stage before 5 is designed to be
  additive-and-reversible by construction (new code path proven alongside the old one,
  nothing shared mutated), so that step 5 is the only one requiring a real rollback
  procedure rather than a revert.

## 7. Open questions for the PM / CireSnave

1. **Is this instance ever run with a shared, long-lived Postgres database that other
   processes or people depend on right now**, or is every `database_url` here
   dev/CI-scoped and disposable? The README documents production TLS deployment
   scenarios (`README.md:159-238`), but that describes capability, not confirms an
   actual running instance with live state today. This changes how cautious step 5
   needs to be -- a disposable dev database makes the big-bang Option A viable too,
   since "worst case, recreate it" is a real fallback there and isn't here.
2. **Answered (PM):** "carry user_sessions forward unchanged; dropping the dead table is
   its own later PR (irreversible = own decision)." `user_sessions` (`V3`, dead schema,
   zero code references, confirmed in §1) is carried forward unchanged by Step 5; its
   removal is its own future PR, not bundled with the pool swap.
3. **`Uuid` has no direct `Value` variant** (§2's table). Two ways to handle it:
   (a) stringify every `Uuid` through `Value::Text` at each of lightbulb's call sites
   (~6 of the 10 touch an `id` column), or (b) switch `api_keys.id` and its foreign
   keys from `UUID` to `TEXT`/`BIGINT` in the ported migrations, avoiding the
   conversion everywhere at the cost of a schema change beyond a literal port of the
   existing `.sql` files. Which is preferred? (a) keeps the schema identical to today's;
   (b) is less code per call site but means step 1's "migrations only, diffed against
   refinery's output" check in §5 would show an intentional difference, not a pure
   schema match, and needs its own sign-off.
